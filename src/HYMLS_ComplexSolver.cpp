#include "HYMLS_ComplexSolver.hpp"

#include "HYMLS_config.h"

#include "HYMLS_ComplexOperator.hpp"
#include "HYMLS_ComplexVector.hpp"
#include "HYMLS_Macros.hpp"
#include "HYMLS_Tools.hpp"

#include "Epetra_Comm.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Operator.h"

#include "BelosTypes.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosSolverManager.hpp"
#include "BelosBlockGmresSolMgr.hpp"
#include "BelosBlockCGSolMgr.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

namespace HYMLS {

// constructor
ComplexSolver::ComplexSolver(Teuchos::RCP<const Epetra_Operator> K,
  Teuchos::RCP<Epetra_Operator> P,
  Teuchos::RCP<Teuchos::ParameterList> params,
  bool validate)
  :
  BaseSolver(K, P, params, validate),
  label_("HYMLS::ComplexSolver")
  {
  HYMLS_PROF3(label_, "Constructor");
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
      ::Belos::BlockCGSolMgr<std::complex<double>, BelosMultiVectorType, BelosOperatorType>
      (belosProblemPtr_, belosListPtr));
    }
  else if (solverType_ == "GMRES")
    {
    belosSolverPtr_ = Teuchos::rcp(new
      ::Belos::BlockGmresSolMgr<std::complex<double>, BelosMultiVectorType, BelosOperatorType>
      (belosProblemPtr_, belosListPtr));
    }
  else
    {
    Tools::Error("Currently only 'GMRES' is supported as 'Belos Solver'",__FILE__,__LINE__);
    }
  }

// destructor
ComplexSolver::~ComplexSolver()
  {
  HYMLS_PROF3(label_, "Destructor");
  }

// Sets all parameters for the solver
void ComplexSolver::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params)
  {
  setParameterList(params, validateParameters_);
  }

// Sets all parameters for the solver
void ComplexSolver::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params,
  bool validateParameters)
  {
  HYMLS_PROF3(label_, "SetParameterList");

  setMyParamList(params);

  BaseSolver::setParameterList(params, validateParameters);
  }

// Sets all parameters for the solver
Teuchos::RCP<const Teuchos::ParameterList> ComplexSolver::getValidParameters() const
  {
  HYMLS_PROF3(label_, "getValidParameterList");

  BaseSolver::getValidParameters();
  return validParams_;
  }

void ComplexSolver::SetTolerance(double tol)
  {
  Teuchos::ParameterList& belosList = PL().sublist("Iterative Solver");
  belosList.set("Convergence Tolerance", tol);
  Teuchos::RCP<Teuchos::ParameterList> belosListPtr = Teuchos::rcp(&belosList, false);
  belosSolverPtr_->setParameters(belosListPtr);
  }

void ComplexSolver::SetPrecond(Teuchos::RCP<Epetra_Operator> P)
  {
  HYMLS_PROF3(label_, "SetPrecond");
  precond_ = P;
  if (precond_ == Teuchos::null) return;

  belosPrecPtr_ = Teuchos::rcp(new BelosOperatorType(precond_, true));
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
int ComplexSolver::ApplyInverse(const Epetra_MultiVector& X,
  Epetra_MultiVector& Y) const
  {
  HYMLS_PROF(label_, "ApplyInverse");

  Teuchos::RCP<BelosOperatorType> op = Teuchos::rcp(new BelosOperatorType(operator_));
  belosProblemPtr_->setOperator(op);

  Teuchos::RCP<BelosMultiVectorType> sol = Teuchos::rcp(new BelosMultiVectorType(View, Y));
  Teuchos::RCP<BelosMultiVectorType> rhs = Teuchos::rcp(new BelosMultiVectorType(View, X));

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

  return ConvergenceStatus(X, Y, ret);
  }

  }//namespace HYMLS
