#include "HYMLS_Solver.H"

namespace HYMLS {

// constructor
Solver::Solver(Teuchos::RCP<const Epetra_RowMatrix> K,
  Teuchos::RCP<Epetra_Operator> P,
  Teuchos::RCP<Teuchos::ParameterList> params,
  int numRhs)
  :
  PLA("Solver")
  {
  HYMLS_PROF3(label_,"Constructor");
  solver_ = Teuchos::rcp(new BaseSolver(K, P, params, numRhs));
  }

// destructor
Solver::~Solver()
  {
  HYMLS_PROF3(label_,"Destructor");
  }

  }//namespace HYMLS
