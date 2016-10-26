#include "HYMLS_Solver.H"

namespace HYMLS {

// constructor
Solver::Solver(Teuchos::RCP<const Epetra_RowMatrix> K,
  Teuchos::RCP<Epetra_Operator> P,
  Teuchos::RCP<Teuchos::ParameterList> params,
  int numRhs)
  :
  PLA("Solver"),
  label_("HYMLS::Solver")
  {
  HYMLS_PROF3(label_, "Constructor");

  solver_ = Teuchos::rcp(new BaseSolver(K, P, params, numRhs));

  setParameterList(params);
  }

// destructor
Solver::~Solver()
  {
  HYMLS_PROF3(label_, "Destructor");
  }

//! set solver parameters (the list is the "HYMLS"->"Solver" sublist)
void Solver::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params)
  {
  HYMLS_PROF3(label_, "SetParameterList");

  setMyParamList(params);

  useDeflation_ = PL().get("Use Deflation", false);

  if (validateParameters_)
    getValidParameters();

  solver_->setParameterList(params);
  }

//! get a list of valid parameters for this object
Teuchos::RCP<const Teuchos::ParameterList> Solver::getValidParameters() const
  {
  HYMLS_PROF3(label_, "getValidParameterList");

  if (validParams_ != Teuchos::null)
    return validParams_;

  Teuchos::RCP<const Teuchos::ParameterList> validParams = solver_->getValidParameters();
  validParams_ = Teuchos::rcp_const_cast<Teuchos::ParameterList>(validParams);

  VPL().set("Use Deflation", false,
    "Use deflation to improve the conditioning of the problem.");

  return validParams_;
  }

  }//namespace HYMLS
