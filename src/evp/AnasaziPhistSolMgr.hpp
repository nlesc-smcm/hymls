// @HEADER
// ***********************************************************************
//
//                 Anasazi: Block Eigensolvers Package
//                 Copyright (2004) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
// @HEADER

#ifndef ANASAZI_PHIST_SOLMGR_HPP
#define ANASAZI_PHIST_SOLMGR_HPP

/*! \file AnasaziJacobiDavidsonSolMgr.hpp
 *  \brief The Anasazi::JacobiDavidsonSolMgr provides a solver manager for the JacobiDavidson eigensolver.
*/

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCPDecl.hpp"

#include "AnasaziConfigDefs.hpp"
#include "AnasaziTypes.hpp"
#include "AnasaziMultiVecTraits.hpp"
#include "AnasaziEigenproblem.hpp"
#include "AnasaziSolverManager.hpp"

// Include this before phist
#include "jadaCorrectionSolver_impl.H"

#include "phist_macros.h"
#include "phist_kernels.h"
#include "phist_operator.h"
#include "phist_jdqr.h"
#include "phist_jadaOpts.h"
#include "phist_orthog.h"

#include "HYMLS_Tester.H"
#include "Epetra_Util.h"

using Teuchos::RCP;

/** \example JacobiDavidson/JacobiDavidsonEpetraExFileIfpack.cpp
    This is an example of how to use the Anasazi::JacobiDavidsonSolMgr solver manager, using Epetra data structures and an Ifpack preconditioner.  */

namespace Anasazi {

/*!
 * \class JacobiDavidsonSolMgr
 * \brief Solver Manager for JacobiDavidson
 *
 * This class provides a simple interface to the JacobiDavidson
 * eigensolver.  This manager creates
 * appropriate orthogonalization/sort/output managers based on user
 * specified ParameterList entries (or selects suitable defaults),
 * provides access to solver functionality, and manages the restarting
 * process.
 *
 * This class is currently only implemented for real scalar types
 * (i.e. float, double).

 \ingroup anasazi_solver_framework

 \author Steven Hamilton
 */
template <class ScalarType, class MV, class OP, class PREC>
class PhistSolMgr : public SolverManager<ScalarType,MV,OP>
{
    public:

        /*!
         * \brief Basic constructor for JacobiDavidsonSolMgr
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
         * - "Orthogonalization" -- a string specifying the desired orthogonalization: DGKS, SVQB, ICGS.
         *   Default: "SVQB"
         * - "Verbosity" -- a sum of MsgType specifying the verbosity.  Default: AnasaziErrors
         * - "Convergence Tolerance" -- a MagnitudeType specifying the level that residual norms must
         *  reach to decide convergence.  Default: machine precision
         * - "Relative Convergence Tolerance" -- a bool specifying whether residual norms should be
         *  scaled by the magnitude of the corresponding Ritz value.  Care should be taken when performing
         *  scaling for problems where the eigenvalue can be very large or very small.  Default: "false".
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
        int getNumIters() const { return -1; }

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

        phist_jadaOpts_t                                d_opts;

}; // class JacobiDavidsonSolMgr

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
   : d_problem(problem), d_prec(prec)
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

    // Get sort type
    std::string which;
    if( pl.isType<std::string>("Which") )
    {
        which = pl.get<std::string>("Which");
        TEUCHOS_TEST_FOR_EXCEPTION( which!="LM" && which!="SM" && which!="LR" && which!="SR",
                                    std::invalid_argument,
                                    "Which must be one of LM,SM,LR,SR." );
        // Convert to eigSort_t
        if (which == "LM")
            d_opts.which = LM;
        if (which == "SM")
            d_opts.which = SM;
        if (which == "LR")
            d_opts.which = LR;
        if (which == "SR")
            d_opts.which = SR;
    }
    else
    {
        d_opts.which = LM;
    }
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
  Teuchos::RCP<MV> v0 = MVT::CloneCopy(*d_problem->getInitVec());

  d_opts.v0 = v0.get();
  d_opts.arno = 0;
  //~ d_opts.minBas = 5;
  //~ d_opts.initialShift=-0.05;
  //~ d_opts.initialShift=-51.0;

  // create operator wrapper for computing Y=A*X using a CRS matrix
  Teuchos::RCP<extended_Dop_t> A_op = Teuchos::rcp(new extended_Dop_t());
  phist_Dop_wrap_sparseMat((Dop_t *)A_op.get(), Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(d_problem->getOperator()).get(), &iflag);
  TEUCHOS_TEST_FOR_EXCEPTION(iflag != 0, std::runtime_error,
    "PhistSolMgr::solve: phist_Dop_wrap_sparseMat returned nonzero error code "+Teuchos::toString(iflag));
  A_op->solver = d_prec;

  Teuchos::RCP<extended_Dop_t> B_op = Teuchos::null;
  if (d_problem->getM() != Teuchos::null)
    {
    B_op = Teuchos::rcp(new extended_Dop_t());
    phist_Dop_wrap_sparseMat((Dop_t *)B_op.get(), Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(d_problem->getM()).get(), &iflag);
    TEUCHOS_TEST_FOR_EXCEPTION(iflag != 0, std::runtime_error,
      "PhistSolMgr::solve: phist_Dop_wrap_sparseMat returned nonzero error code "+Teuchos::toString(iflag));
    B_op->solver = d_prec;
  }

  d_prec->SetMassMatrix(Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(d_problem->getM()));

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

  int num_eigs, num_iters;

  phist_Djdqr((Dop_t *)A_op.get(), (Dop_t *)B_op.get(), X.get(), &evals[0], &resid[0], &is_cmplx[0],
        d_opts, &num_eigs, &num_iters, &iflag);
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

  sol.Evecs = X;

  std::sort(sol.Evals.begin(), sol.Evals.end(), eigSort<ScalarType>);

  if (sol.numVecs)
    sol.Evecs = MVT::CloneCopy(*X, Teuchos::Range1D(0, sol.numVecs-1));
  d_problem->setSolution(sol);

  // Return convergence status
  if( sol.numVecs < d_problem->getNEV() )
    return Unconverged;

  return Converged;
}

} // namespace Anasazi

#endif // ANASAZI_PHIST_SOLMGR_HPP

