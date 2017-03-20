#include "HYMLS_BorderedSolver.H"

#include "HYMLS_BorderedOperator.H"
#include "HYMLS_BorderedVector.H"
#include "HYMLS_ProjectedOperator.H"
#include "HYMLS_DenseUtils.H"

#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSVD.h"

#include "BelosBlockGmresSolMgr.hpp"
#include "BelosTypes.hpp"
#include "BelosBlockCGSolMgr.hpp"
#include "HYMLS_BelosEpetraOperator.H"

namespace HYMLS {

// constructor
BorderedSolver::BorderedSolver(Teuchos::RCP<const Epetra_RowMatrix> K,
  Teuchos::RCP<Epetra_Operator> P,
  Teuchos::RCP<Teuchos::ParameterList> params,
  int numRhs, bool validate)
  :
  BaseSolver(K, P, params, numRhs, validate),
  label_("HYMLS::BorderedSolver")
  {
  HYMLS_PROF3(label_,"Constructor");
  belosProblemPtr_=Teuchos::rcp(new ::Belos::LinearProblem<double, BorderedVector, BorderedOperator>);

  SetPrecond(precond_);

  Teuchos::ParameterList& belosList = PL().sublist("Iterative Solver");

  belosList.set("Output Style", 1);
  belosList.set("Verbosity", ::Belos::Errors+::Belos::Warnings
    +::Belos::IterationDetails
    +::Belos::StatusTestDetails
    +::Belos::FinalSummary
    +::Belos::TimingDetails);

  belosList.set("Output Stream", Tools::out().getOStream());

  // create the solver
  Teuchos::RCP<Teuchos::ParameterList> belosListPtr = Teuchos::rcp(&belosList, false);
  if (solverType_ == "CG")
    {
    belosSolverPtr_ = rcp(new
      ::Belos::BlockCGSolMgr<double, BorderedVector, BorderedOperator>
      (belosProblemPtr_, belosListPtr));
    }
  else if (solverType_ == "GMRES")
    {
    belosSolverPtr_ = Teuchos::rcp(new
      ::Belos::BlockGmresSolMgr<double, BorderedVector, BorderedOperator>
      (belosProblemPtr_, belosListPtr));
    }
  else
    {
    Tools::Error("Currently only 'GMRES' is supported as 'Belos Solver'",__FILE__,__LINE__);
    }
  }

// destructor
BorderedSolver::~BorderedSolver()
  {
  HYMLS_PROF3(label_,"Destructor");
  }

// Sets all parameters for the solver
void BorderedSolver::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params)
  {
  setParameterList(params, validateParameters_);
  }

// Sets all parameters for the solver
void BorderedSolver::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params,
  bool validateParameters)
  {
  HYMLS_PROF3(label_,"SetParameterList");

  setMyParamList(params);

  BaseSolver::setParameterList(params, validateParameters);
  }

// Sets all parameters for the solver
Teuchos::RCP<const Teuchos::ParameterList> BorderedSolver::getValidParameters() const
  {
  HYMLS_PROF3(label_, "getValidParameterList");

  BaseSolver::getValidParameters();
  return validParams_;
  }

int BorderedSolver::setBorder(Teuchos::RCP<const Epetra_MultiVector> const &V,
  Teuchos::RCP<const Epetra_MultiVector> const &W,
  Teuchos::RCP<const Epetra_SerialDenseMatrix> const &C)
  {
  if (V.is_null())
    {
    return 0;
    }

  V_ = V;
  W_ = W;
  if (W.is_null())
    {
    W_ = V;
    }

  C_ = C;

  // Set the border for the matrix and the preconditioner
  Teuchos::RCP<HYMLS::BorderedOperator> bprec
    = Teuchos::rcp_dynamic_cast<BorderedOperator>(precond_);
  if (bprec != Teuchos::null)
    {
    CHECK_ZERO(bprec->setBorder(V_, W_, C_));
    }

  return 0;
  }

void BorderedSolver::SetTolerance(double tol)
  {
  Teuchos::ParameterList& belosList = PL().sublist("Iterative Solver");
  belosList.set("Convergence Tolerance", tol);
  Teuchos::RCP<Teuchos::ParameterList> belosListPtr = rcp(&belosList, false);
  belosSolverPtr_->setParameters(belosListPtr);
  }

void BorderedSolver::SetPrecond(Teuchos::RCP<Epetra_Operator> P)
  {
  HYMLS_PROF3(label_,"SetPrecond");
  precond_ = P;
  if (precond_ == Teuchos::null) return;

  belosPrecPtr_ = Teuchos::rcp_dynamic_cast<BorderedOperator>(precond_);
  std::string lor = PL().get("Left or Right Preconditioning", lor_default_);
  if (lor == "Left")
    {
    belosProblemPtr_->setLeftPrec(belosPrecPtr_);
    }
  else if (lor == "Right")
    {
    belosProblemPtr_->setRightPrec(belosPrecPtr_);
    }
  else if (lor == "None")
    {
    // no preconditioning
    }
  }

// Applies the solver to vector X, returns the result in Y.
int BorderedSolver::ApplyInverse(const Epetra_MultiVector& X,
  Epetra_MultiVector& Y) const
  {
  Epetra_SerialDenseMatrix S, T;
  if (!V_.is_null())
    {
    CHECK_ZERO(S.Shape(V_->NumVectors(), X.NumVectors()));
    CHECK_ZERO(T.Shape(V_->NumVectors(), Y.NumVectors()));
    }

  return ApplyInverse(X, S, Y, T);
  }

//! Applies the solver to [Y T]' = [K V;W' C]\[X S]'
int BorderedSolver::ApplyInverse(const Epetra_MultiVector& X, const Epetra_SerialDenseMatrix& S,
  Epetra_MultiVector& Y, Epetra_SerialDenseMatrix& T) const
  {
  HYMLS_PROF(label_,"ApplyInverse");
  int ierr = 0;

  if (V_ == Teuchos::null)
    {
    HYMLS_DEBUG("no border set");
    CHECK_ZERO(BaseSolver::ApplyInverse(X, Y));
    return 1; // border not set
    }

  Teuchos::RCP<BorderedOperator> op = Teuchos::rcp(new BorderedOperator(operator_, V_, W_, C_));
  belosProblemPtr_->setOperator(op);

  Teuchos::RCP<BorderedVector> sol = Teuchos::rcp(new BorderedVector(Y, T));
  Teuchos::RCP<BorderedVector> rhs = Teuchos::rcp(new BorderedVector(X, S));

  if (startVec_ == "Random")
    {
    CHECK_ZERO(sol->Random());
    }
  else if (startVec_ == "Zero")
    {
    // set initial vector to 0
    CHECK_ZERO(sol->PutScalar(0.0));
    }

  CHECK_TRUE(belosProblemPtr_->setProblem(sol, rhs));

  ::Belos::ReturnType ret = ::Belos::Unconverged;
  bool status = true;
  try {
    ret = belosSolverPtr_->solve();
    } TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, status);
  if (!status) Tools::Warning("caught an exception", __FILE__, __LINE__);

  numIter_ = belosSolverPtr_->getNumIters();

  Y = *sol->Vector();
  T = *sol->Border();

  if (ret != ::Belos::Converged)
    {
    HYMLS::Tools::Warning("Belos returned "+::Belos::convertReturnTypeToString(ret)+
      "'!", __FILE__, __LINE__);

#ifdef HYMLS_TESTING
    Teuchos::RCP<const Epetra_CrsMatrix> Acrs =
      Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);
    MatrixUtils::Dump(*Acrs, "FailedMatrix.txt");
    MatrixUtils::Dump(B, "FailedRhs.txt");
#endif

    ierr = -1;
    }

  if (comm_->MyPID() == 0)
    {
    Tools::Out("++++++++++++++++++++++++++++++++++++++++++++++++");
    Tools::Out("+ Number of iterations: " + Teuchos::toString(numIter_));
    Tools::Out("++++++++++++++++++++++++++++++++++++++++++++++++");
    Tools::Out("");
    }

#ifdef HYMLS_TESTING
  Tools::Out("explicit residual test");
  Tools::out() << "we were solving (a*A*x+b*B)*x=rhs\n" <<
    "   with " << X.NumVectors() << " rhs\n" <<
    "        a = " << shiftA_<<"\n" <<
    "        b = " << shiftB_<<"\n";
  if (massMatrix_ == Teuchos::null)
    Tools::out() <<
      "        B = I\n";
  // compute explicit residual
  int dim = PL("Problem").get<int>("Dimension");
  int dof = PL("Problem").get<int>("Degrees of Freedom");

  Epetra_MultiVector resid(X.Map(),X.NumVectors());
  CHECK_ZERO(matrix_->Apply(X,resid));
  if (shiftB_ != 0.0)
    {
    Epetra_MultiVector Bx=X;

    if (massMatrix_ != Teuchos::null)
      {
      CHECK_ZERO(massMatrix_->Apply(X, Bx));
      }
    CHECK_ZERO(resid.Update(shiftB_, Bx, shiftA_));
    }
  else if (shiftA_ != 1.0)
    {
    CHECK_ZERO(resid.Scale(shiftA_));
    }
  CHECK_ZERO(resid.Update(1.0, B, -1.0));
  double *resNorm, *rhsNorm, *resNormV, *resNormP;
  resNorm  = new double[resid.NumVectors()];
  resNormV = new double[resid.NumVectors()];
  resNormP = new double[resid.NumVectors()];
  rhsNorm  = new double[resid.NumVectors()];
  B.Norm2(rhsNorm);
  resid.Norm2(resNorm);

  if (dof>=dim)
    {
    Epetra_MultiVector residV = resid;
    Epetra_MultiVector residP = resid;
    for (int i = 0; i < resid.MyLength(); i += dof)
      {
      for (int j = 0; j < resid.NumVectors(); j++)
        {
        for (int k = 0; k < dim; k++)
          {
          residP[j][i+k]=0.0;
          }
        for (int k = dim+1; k < dof; k++)
          {
          residP[j][i+k] = 0.0;
          }
        residV[j][i+dim] = 0.0;
        }
      }
    residV.Norm2(resNormV);
    residP.Norm2(resNormP);
    }

  if (comm_->MyPID() == 0)
    {
    Tools::out() << "Exp. res. norm(s): ";
    for (int ii = 0; ii < resid.NumVectors(); ii++)
      {
      Tools::out() << resNorm[ii] << " ";
      }
    Tools::out() << std::endl;
    Tools::out() << "Rhs norm(s): ";
    for (int ii = 0; ii < resid.NumVectors(); ii++)
      {
      Tools::out() << rhsNorm[ii] << " ";
      }
    Tools::out() << std::endl;
    if (dof >= dim)
      {
      Tools::out() << "Exp. res. norm(s) of V-part: ";
      for (int ii = 0; ii < resid.NumVectors(); ii++)
        {
        Tools::out() << resNormV[ii] << " ";
        }
      Tools::out() << std::endl;
      Tools::out() << "Exp. res. norm(s) of P-part: ";
      for (int ii = 0; ii < resid.NumVectors(); ii++)
        {
        Tools::out() << resNormP[ii] << " ";
        }
      Tools::out() << std::endl;
      }
    }
  delete [] resNorm;
  delete [] rhsNorm;
  delete [] resNormV;
  delete [] resNormP;
#endif

  return ierr;
  }

  }//namespace HYMLS
