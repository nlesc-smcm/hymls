#include "HYMLS_Solver.hpp"

#include "HYMLS_BaseSolver.hpp"
#include "HYMLS_BorderedSolver.hpp"
#include "HYMLS_DeflatedSolver.hpp"
#include "HYMLS_BorderedDeflatedSolver.hpp"
#include "HYMLS_Macros.hpp"

#include "Teuchos_ParameterList.hpp"

namespace HYMLS {

// constructor
Solver::Solver(Teuchos::RCP<const Epetra_RowMatrix> K,
  Teuchos::RCP<Epetra_Operator> P,
  Teuchos::RCP<Teuchos::ParameterList> params,
  int numRhs)
  :
  PLA("Solver"),
  solver_(Teuchos::null),
  label_("HYMLS::Solver")
  {
  HYMLS_PROF3(label_, "Constructor");

  setParameterList(params);

  if (useDeflation_ && useBordering_)
    solver_ = Teuchos::rcp(new BorderedDeflatedSolver(K, P, params, numRhs, false));
  else if (useDeflation_)
    solver_ = Teuchos::rcp(new DeflatedSolver(K, P, params, numRhs, false));
  else if (useBordering_)
    solver_ = Teuchos::rcp(new BorderedSolver(K, P, params, numRhs, false));
  else
    solver_ = Teuchos::rcp(new BaseSolver(K, P, params, numRhs, false));

  setParameterList(params);
  }

// destructor
Solver::~Solver()
  {
  HYMLS_PROF3(label_, "Destructor");
  }

void Solver::Solver::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params)
  {
  HYMLS_PROF3(label_, "SetParameterList");

  setMyParamList(params);

  useDeflation_ = PL().get("Use Deflation", false);

  useBordering_ = PL().get("Use Bordering", false);

  if (solver_.is_null())
    return;

  solver_->setParameterList(params, false);

  if (validateParameters_)
    {
    getValidParameters();
    PL().validateParameters(VPL());
    }
  }

Teuchos::RCP<const Teuchos::ParameterList> Solver::getValidParameters() const
  {
  HYMLS_PROF3(label_, "getValidParameterList");

  if (validParams_ != Teuchos::null || solver_.is_null())
    return validParams_;

  Teuchos::RCP<const Teuchos::ParameterList> validParams = solver_->getValidParameters();
  validParams_ = Teuchos::rcp_const_cast<Teuchos::ParameterList>(validParams);

  VPL().set("Use Deflation", false,
    "Use deflation to improve the conditioning of the problem.");

  VPL().set("Use Bordering", false,
    "Use bordering instead of projections when projecting out vectors.");

  return validParams_;
  }

void Solver::SetMatrix(Teuchos::RCP<const Epetra_RowMatrix> A)
  {
  solver_->SetMatrix(A);
  }

void Solver::SetPrecond(Teuchos::RCP<Epetra_Operator> P)
  {
  solver_->SetPrecond(P);
  }

void Solver::SetMassMatrix(Teuchos::RCP<const Epetra_RowMatrix> B)
  {
  solver_->SetMassMatrix(B);
  }

int Solver::Apply(const Epetra_MultiVector& X,
  Epetra_MultiVector& Y) const
  {
  return solver_->Apply(X, Y);
  }

int Solver::ApplyMatrix(const Epetra_MultiVector& X,
  Epetra_MultiVector& Y) const
  {
  return solver_->ApplyMatrix(X, Y);
  }

int Solver::ApplyPrec(const Epetra_MultiVector& X,
  Epetra_MultiVector& Y) const
  {
  return solver_->ApplyPrec(X, Y);
  }

int Solver::ApplyMass(const Epetra_MultiVector& X,
  Epetra_MultiVector& Y) const
  {
  return solver_->ApplyMass(X, Y);
  }

int Solver::ApplyInverse(const Epetra_MultiVector& X,
  Epetra_MultiVector& Y) const
  {
  return solver_->ApplyInverse(X, Y);
  }

int Solver::SetUseTranspose(bool UseTranspose)
  {
  return solver_->SetUseTranspose(UseTranspose);
  }

bool Solver::HasNormInf() const
  {
  return solver_->HasNormInf();
  }

double Solver::NormInf() const
  {
  return solver_->NormInf();
  }

const char* Solver::Label() const
  {
  return solver_->Label();
  }

bool Solver::UseTranspose() const
  {
  return solver_->UseTranspose();
  }

const Epetra_Comm & Solver::Comm() const
  {
  return solver_->Comm();
  }

const Epetra_Map & Solver::OperatorDomainMap() const
  {
  return solver_->OperatorDomainMap();
  }

const Epetra_Map & Solver::OperatorRangeMap() const
  {
  return solver_->OperatorRangeMap();
  }

void Solver::setShift(double shiftA, double shiftB)
  {
  solver_->setShift(shiftA, shiftB);
  }

void Solver::SetTolerance(double tol)
  {
  solver_->SetTolerance(tol);
  }

int Solver::getNumIter() const
  {
  return solver_->getNumIter();
  }

int Solver::SetBorder(Teuchos::RCP<const Epetra_MultiVector> const &V,
  Teuchos::RCP<const Epetra_MultiVector> const &W,
  Teuchos::RCP<const Epetra_SerialDenseMatrix> const &C)
  {
  return solver_->SetBorder(V, W, C);
  }

int Solver::setProjectionVectors(Teuchos::RCP<const Epetra_MultiVector> V,
  Teuchos::RCP<const Epetra_MultiVector> W)
  {
  return solver_->setProjectionVectors(V, W);
  }

int Solver::SetupDeflation()
  {
  return solver_->SetupDeflation();
  }

  }//namespace HYMLS
