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

//! set solver parameters (the list is the "HYMLS"->"Solver" sublist)
void Solver::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params)
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

//! get a list of valid parameters for this object
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

  }//namespace HYMLS
