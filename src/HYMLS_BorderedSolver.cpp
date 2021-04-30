#include "HYMLS_BorderedSolver.hpp"

#include "HYMLS_BorderedOperator.hpp"
#include "HYMLS_BorderedVector.hpp"
#include "HYMLS_DenseUtils.hpp"
#include "HYMLS_Macros.hpp"
#include "HYMLS_Tools.hpp"

#include "Epetra_MultiVector.h"
#include "Epetra_Operator.h"
#include "Epetra_SerialDenseMatrix.h"

#include "BelosTypes.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosSolverManager.hpp"
#include "BelosBlockGmresSolMgr.hpp"
#include "BelosBlockCGSolMgr.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

namespace HYMLS {

// constructor
BorderedSolver::BorderedSolver(Teuchos::RCP<const Epetra_Operator> K,
  Teuchos::RCP<Epetra_Operator> P,
  Teuchos::RCP<Teuchos::ParameterList> params,
  bool validate)
  :
  BaseSolver(K, P, params, validate),
  label_("HYMLS::BorderedSolver")
  {
  HYMLS_PROF3(label_,"Constructor");
  belosProblemPtr_ = Teuchos::rcp(new BelosProblemType());

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
    belosSolverPtr_ = Teuchos::rcp(new
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

int BorderedSolver::SetBorder(Teuchos::RCP<const Epetra_MultiVector> const &V,
  Teuchos::RCP<const Epetra_MultiVector> const &W,
  Teuchos::RCP<const Epetra_SerialDenseMatrix> const &C)
  {
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
    CHECK_ZERO(bprec->SetBorder(V_, W_, C_));
    }

  return 0;
  }

void BorderedSolver::SetTolerance(double tol)
  {
  Teuchos::ParameterList& belosList = PL().sublist("Iterative Solver");
  belosList.set("Convergence Tolerance", tol);
  Teuchos::RCP<Teuchos::ParameterList> belosListPtr = Teuchos::rcp(&belosList, false);
  belosSolverPtr_->setParameters(belosListPtr);
  }

void BorderedSolver::SetPrecond(Teuchos::RCP<Epetra_Operator> P)
  {
  HYMLS_PROF3(label_,"SetPrecond");
  precond_ = P;
  if (precond_ == Teuchos::null) return;

  belosPrecPtr_ = Teuchos::rcp_dynamic_cast<BorderedOperator>(precond_);
  if (belosPrecPtr_ == Teuchos::null)
    belosPrecPtr_ = Teuchos::rcp(new BorderedOperator(precond_, true));

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
  HYMLS_PROF(label_, "ApplyInverse");

  if (V_ == Teuchos::null)
    {
    Tools::Warning("No border was set in the bordered solver", __FILE__, __LINE__);
    return BaseSolver::ApplyInverse(X, Y);
    }

  Teuchos::RCP<BorderedOperator> op = Teuchos::rcp(new BorderedOperator(operator_, V_, W_, C_));
  belosProblemPtr_->setOperator(op);

  Teuchos::RCP<BorderedVector> sol = Teuchos::rcp(new BorderedVector(View, Y, T));
  Teuchos::RCP<BorderedVector> rhs = Teuchos::rcp(new BorderedVector(View, X, S));

  if (startVec_ == "Random")
    {
    CHECK_ZERO(sol->Random());
    }
  else if (startVec_ == "Zero")
    {
    // set initial vector to 0
    CHECK_ZERO(sol->PutScalar(0.0));
    }

  // Make the initial guess orthogonal to the V_ space. Not sure if we need this.
  // if (V_ != Teuchos::null)
  //   {
  //   CHECK_ZERO(DenseUtils::ApplyOrth(*V_, *sol->Vector(), Y, W_));
  //   *sol->Vector() = Y;
  //   }

  CHECK_TRUE(belosProblemPtr_->setProblem(sol, rhs));

  ::Belos::ReturnType ret = ::Belos::Unconverged;
  bool status = true;
  try {
    ret = belosSolverPtr_->solve();
    } TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, status);
  if (!status) Tools::Warning("caught an exception", __FILE__, __LINE__);

  numIter_ = belosSolverPtr_->getNumIters();

  return ConvergenceStatus(X, Y, ret);
  }

  }//namespace HYMLS
