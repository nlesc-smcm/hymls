#ifndef ANASAZI_PHIST_SOLMGR_HPP
#define ANASAZI_PHIST_SOLMGR_HPP

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCPDecl.hpp"

#include "AnasaziConfigDefs.hpp"
#include "AnasaziTypes.hpp"
#include "AnasaziMultiVecTraits.hpp"
#include "AnasaziEigenproblem.hpp"
#include "AnasaziSolverManager.hpp"

// Include this before phist
#include "HYMLS_PhistWrapper.H"

#include "phist_macros.h"
#include "phist_kernels.h"
#include "phist_operator.h"
#include "phist_jdqr.h"
#include "phist_jadaOpts.h"
#include "phist_orthog.h"

#include "HYMLS_Solver.H"
#include "HYMLS_Tester.H"
#include "Epetra_Util.h"

using Teuchos::RCP;

namespace Anasazi {

/*!
 * \class PhistSolMgr
 * \brief Solver Manager for Jacobi Davidson in phist
 *
 * This class provides a simple interface to the Jacobi Davidson
 * eigensolver.  This manager creates
 * appropriate managers based on user
 * specified ParameterList entries (or selects suitable defaults),
 * provides access to solver functionality, and manages the restarting
 * process.
 *
 * This class is currently only implemented for real scalar types
 * (i.e. float, double).

 \ingroup anasazi_solver_framework

 */
template <class ScalarType, class MV, class OP, class PREC>
class PhistSolMgr : public SolverManager<ScalarType,MV,OP>
{
    public:

        /*!
         * \brief Basic constructor for PhistSolMgr
         *
         * This constructor accepts the Eigenproblem to be solved and a parameter list of options
         * for the solver.
         * The following options control the behavior
         * of the solver:
         * - "Which" -- a string specifying the desired eigenvalues: SM, LM, SR, LR, SI, or LI. Default: "LM."
         * - "Block Size" -- block size used by algorithm.  Default: 1.
         * - "Maximum Subspace Dimension" -- maximum number of basis vectors for subspace.  Two
         *  (for standard eigenvalue problems) or three (for generalized eigenvalue problems) sets of basis
         *  vectors of this size will be required. Default: 3*problem->getNEV()*"Block Size"
         * - "Restart Dimension" -- Number of vectors retained after a restart.  Default: NEV
         * - "Maximum Restarts" -- an int specifying the maximum number of restarts the underlying solver
         *  is allowed to perform.  Default: 20
         * - "Verbosity" -- a sum of MsgType specifying the verbosity.  Default: AnasaziErrors
         * - "Convergence Tolerance" -- a MagnitudeType specifying the level that residual norms must
         *  reach to decide convergence.  Default: machine precision
         * - "Initial Guess" -- how should initial vector be selected: "Random" or "User".
         *   If "User," the value in problem->getInitVec() will be used.  Default: "Random".
         * - "Print Number of Ritz Values" -- an int specifying how many Ritz values should be printed
         *   at each iteration.  Default: "NEV".
         */
        PhistSolMgr( const RCP< Eigenproblem<ScalarType,MV,OP> > &problem,
                                   const RCP<PREC> &prec,
                                   Teuchos::ParameterList &pl );

        /*!
         * \brief Return the eigenvalue problem.
         */
        const Eigenproblem<ScalarType,MV,OP> & getProblem() const { return *d_problem; }

        /*!
         * \brief Get the iteration count for the most recent call to solve()
         */
        int getNumIters() const { return numIters_; }

        /*!
         * \brief This method performs possibly repeated calls to the underlying eigensolver's iterate()
         *  routine until the problem has been solved (as decided by the StatusTest) or the solver manager decides to quit.
         */
        ReturnType solve();

    private:

        typedef MultiVecTraits<ScalarType,MV>        MVT;
        typedef Teuchos::ScalarTraits<ScalarType>    ST;
        typedef typename ST::magnitudeType           MagnitudeType;
        typedef Teuchos::ScalarTraits<MagnitudeType> MT;

        RCP< Eigenproblem<ScalarType,MV,OP> >           d_problem;
        RCP< PREC >                                     d_prec;
        // used to pass HYMLS object to phist
        RCP<phist_hymls_wrapper>                        d_wrapper;
        phist_jadaOpts                                  d_opts;

        bool borderedSolver;
        int numIters_; //! number of iterations performed in previous solve()

}; // class PhistSolMgr

//---------------------------------------------------------------------------//
// Prevent instantiation on complex scalar type
//---------------------------------------------------------------------------//
template <class MagnitudeType, class MV, class OP, class PREC>
class PhistSolMgr<std::complex<MagnitudeType>,MV,OP,PREC>
{
  public:

    typedef std::complex<MagnitudeType> ScalarType;
    PhistSolMgr(
            const RCP<Eigenproblem<ScalarType,MV,OP> > &problem,
            Teuchos::ParameterList &pl )
    {
        // Provide a compile error when attempting to instantiate on complex type
        MagnitudeType::this_class_is_missing_a_specialization();
    }
};

//---------------------------------------------------------------------------//
// Start member definitions
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Constructor
//---------------------------------------------------------------------------//
template <class ScalarType, class MV, class OP, class PREC>
PhistSolMgr<ScalarType,MV,OP,PREC>::PhistSolMgr(
        const RCP<Eigenproblem<ScalarType,MV,OP> > &problem,
        const RCP<PREC> &prec,
        Teuchos::ParameterList &pl )
   : d_problem(problem), d_prec(prec), numIters_(0)
{
    TEUCHOS_TEST_FOR_EXCEPTION( d_problem == Teuchos::null,                std::invalid_argument, "Problem not given to solver manager." );
    TEUCHOS_TEST_FOR_EXCEPTION( !d_problem->isProblemSet(),                std::invalid_argument, "Problem not set." );
    TEUCHOS_TEST_FOR_EXCEPTION( d_problem->getA() == Teuchos::null &&
                                d_problem->getOperator() == Teuchos::null, std::invalid_argument, "A operator not supplied on Eigenproblem." );
    TEUCHOS_TEST_FOR_EXCEPTION( d_problem->getInitVec() == Teuchos::null,  std::invalid_argument, "No vector to clone from on Eigenproblem." );
    TEUCHOS_TEST_FOR_EXCEPTION( d_problem->getNEV() <= 0,                  std::invalid_argument, "Number of requested eigenvalues must be positive.");

    // Initialize phist options
    phist_jadaOpts_setDefaults(&d_opts);
    d_opts.numEigs = d_problem->getNEV();

    if( !pl.isType<int>("Block Size") )
    {
        pl.set<int>("Block Size",1);
        d_opts.blockSize = pl.get<int>("Block Size");
    }

    if( !pl.isType<int>("Maximum Subspace Dimension") )
    {
        pl.set<int>("Maximum Subspace Dimension",3*problem->getNEV()*pl.get<int>("Block Size"));
    }

    if( !pl.isType<int>("Print Number of Ritz Values") )
    {
        int numToPrint = std::max( pl.get<int>("Block Size"), d_problem->getNEV() );
        pl.set<int>("Print Number of Ritz Values",numToPrint);
    }

    // Get convergence info
    MagnitudeType tol = pl.get<MagnitudeType>("Convergence Tolerance", MT::eps() );
    TEUCHOS_TEST_FOR_EXCEPTION( pl.get<MagnitudeType>("Convergence Tolerance") <= MT::zero(),
                                std::invalid_argument, "Convergence Tolerance must be greater than zero." );

    d_opts.convTol = tol;

    // Get maximum restarts
    d_opts.maxBas = pl.get<int>("Restart Dimension",d_problem->getNEV());
    TEUCHOS_TEST_FOR_EXCEPTION( d_opts.maxBas < d_problem->getNEV(),
            std::invalid_argument, "Restart Dimension must be at least NEV" );

    // Get maximum restarts
    if( pl.isType<int>("Maximum Restarts") )
    {
        d_opts.maxIters = d_opts.maxBas * pl.get<int>("Maximum Restarts");
        TEUCHOS_TEST_FOR_EXCEPTION( d_opts.maxIters < 0, std::invalid_argument, "Maximum Restarts must be non-negative" );
    }
    else
    {
        d_opts.maxIters = d_opts.maxBas * 20;
    }

    // Get initial guess type
    std::string initType;
    if( pl.isType<std::string>("Initial Guess") )
    {
        initType = pl.get<std::string>("Initial Guess");
        TEUCHOS_TEST_FOR_EXCEPTION( initType!="User" && initType!="Random", std::invalid_argument,
                                    "Initial Guess type must be 'User' or 'Random'." );
    }
    else
    {
        initType = "User";
    }

    if (pl.isType<bool>("Bordered Solver"))
    {
        borderedSolver = pl.get<bool>("Bordered Solver");
    }
    else
    {
        borderedSolver = true;
    }

    // Get sort type
    std::string which;
    if( pl.isType<std::string>("Which") )
    {
        which = pl.get<std::string>("Which");
        TEUCHOS_TEST_FOR_EXCEPTION( which!="LM" && which!="SM" && which!="LR" && which!="SR",
                                    std::invalid_argument,
                                    "Which must be one of LM,SM,LR,SR." );
        // Convert to eigSort_t
        d_opts.which=str2eigSort(which.c_str());
    }
    else
    {
        d_opts.which = phist_LM;
    }

  d_wrapper = Teuchos::rcp(new phist_hymls_wrapper);
  d_wrapper->solver = d_prec;
  d_wrapper->borderedSolver = borderedSolver;

  // tell phist to use a custom solver provided in d_opts->custom_solver;
  d_opts.innerSolvType = phist_USER_DEFINED;
  d_opts.customSolver = (void*)d_wrapper.get();
  d_opts.customSolver_run = HYMLS_jadaCorrectionSolver_run;
  d_opts.customSolver_run1 = HYMLS_jadaCorrectionSolver_run1;
  d_opts.custom_computeResidual = HYMLS_computeResidual;
}

template <class ScalarType>
bool eigSort(Anasazi::Value<ScalarType> const &a, Anasazi::Value<ScalarType> const &b)
{
  return (a.realpart * a.realpart + a.imagpart * a.imagpart) <
         (b.realpart * b.realpart + b.imagpart * b.imagpart);
}

//---------------------------------------------------------------------------//
// Solve
//---------------------------------------------------------------------------//
template <class ScalarType, class MV, class OP, class PREC>
ReturnType PhistSolMgr<ScalarType,MV,OP,PREC>::solve()
{
  int iflag;
  Teuchos::RCP<MV> X = MVT::Clone(*d_problem->getInitVec(), d_problem->getNEV()+1);
  Teuchos::RCP<MV> Q = MVT::Clone(*d_problem->getInitVec(), d_problem->getNEV()+1);
  Teuchos::RCP<MV> v0 = MVT::CloneCopy(*d_problem->getInitVec());

  d_opts.v0 = v0.get();
  d_opts.arno = 0;

  // create operator wrapper for computing Y=A*X using a CRS matrix
  Teuchos::RCP<phist_DlinearOp> A_op = Teuchos::rcp(new phist_DlinearOp);
  phist_DlinearOp_wrap_sparseMat(A_op.get(), Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(d_problem->getOperator()).get(), &iflag);
  TEUCHOS_TEST_FOR_EXCEPTION(iflag != 0, std::runtime_error,
    "PhistSolMgr::solve: phist_DlinearOp_wrap_sparseMat returned nonzero error code "+Teuchos::toString(iflag));

  Teuchos::RCP<phist_DlinearOp> B_op = Teuchos::null;
  if (d_problem->getM() != Teuchos::null)
  {
    Teuchos::RCP<const Epetra_CrsMatrix> B_rcp=Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(d_problem->getM());
    const Epetra_CrsMatrix* B=B_rcp.get();
    B_op = Teuchos::rcp(new phist_DlinearOp());
    phist_DlinearOp_wrap_sparseMat(B_op.get(), B, &iflag);
    TEUCHOS_TEST_FOR_EXCEPTION(iflag != 0, std::runtime_error,
      "PhistSolMgr::solve: phist_DlinearOp_wrap_sparseMat returned nonzero error code "+Teuchos::toString(iflag));
    //       Use the 'free' field aux in the B operator to pass the solver to ComputeResidual. We need to do 
    //  this right now because there is no systematic way in hymls to provide an 'additional projection space'
    B_op->aux=d_opts.customSolver;
    d_prec->SetMassMatrix(B_rcp);
  }


  // allocate memory for eigenvalues and residuals. We allocate
  // one extra entry because in the real case we may get that the
  // last EV to converge is a complex pair (requirement of JDQR)
  std::vector<ScalarType> evals(d_problem->getNEV()+1);
  std::vector<MagnitudeType> resid(d_problem->getNEV()+1);
  std::vector<int> is_cmplx(d_problem->getNEV()+1);

  std::vector<ScalarType> nrmX0(d_problem->getNEV()+1);
  phist_Dmvec_normalize(X.get(), &nrmX0[0], &iflag);
  TEUCHOS_TEST_FOR_EXCEPTION(iflag != 0, std::runtime_error,
    "PhistSolMgr::solve: phist_Dmvec_normalize returned nonzero error code "+Teuchos::toString(iflag));

  int num_eigs;

  phist_Djdqr(A_op.get(), B_op.get(), X.get(), Q.get(), NULL, &evals[0], &resid[0], &is_cmplx[0],
        d_opts, &num_eigs, &numIters_, &iflag);
  TEUCHOS_TEST_FOR_EXCEPTION(iflag != 0, std::runtime_error,
    "PhistSolMgr::solve: phist_Djdqr returned nonzero error code "+Teuchos::toString(iflag));

  Eigensolution<ScalarType,MV> sol;
  sol.numVecs = num_eigs;

  sol.index.resize(sol.numVecs);
  sol.Evals.resize(sol.numVecs);

  int i = 0;
  while (i < sol.numVecs)
  {
    sol.index[i] = i;
    sol.Evals[i].realpart = evals[i];
    if (is_cmplx[i]) {
      sol.Evals[i].imagpart = evals[i+1];
      sol.Evals[i+1].realpart = evals[i];
      sol.Evals[i+1].imagpart = -evals[i+1];
      i++;
    }
    i++;
  }

  std::sort(sol.Evals.begin(), sol.Evals.end(), eigSort<ScalarType>);

  if (sol.numVecs)
  {
    sol.Evecs = MVT::CloneCopy(*X, Teuchos::Range1D(0, sol.numVecs-1));
    sol.Espace = MVT::CloneCopy(*Q, Teuchos::Range1D(0, sol.numVecs-1));
  }
  d_problem->setSolution(sol);

  // Return convergence status
  if( sol.numVecs < d_problem->getNEV() )
    return Unconverged;

  return Converged;
}

} // namespace Anasazi

#endif // ANASAZI_PHIST_SOLMGR_HPP

