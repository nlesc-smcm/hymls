#include "HYMLS_Solver.H"
#include "HYMLS_DeflatedSolver.H"
#include "HYMLS_BorderedSolver.H"
#include "HYMLS_BorderedDeflatedSolver.H"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Epetra_MpiComm.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_CrsMatrix.h>

#include "HYMLS_UnitTests.H"

class TestableSolver: public HYMLS::Solver
  {
public:
  TestableSolver(Teuchos::RCP<const Epetra_RowMatrix> K,
    Teuchos::RCP<Epetra_Operator> P,
    Teuchos::RCP<Teuchos::ParameterList> params)
  :
  HYMLS::Solver(K, P, params)
    {}

  Teuchos::RCP<HYMLS::BaseSolver> const &Solver()
    {
    return solver_;
    }
  };

// Test that we can use the parameterlist to select various solvers
TEUCHOS_UNIT_TEST(Solver, BaseSolver)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);

  int n = 1;
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  Epetra_Map map(n, 0, Comm);

  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, map, 1));
  int A_idx = 0;
  double A_val = 1;
  A->InsertGlobalValues(0, 1, &A_val, &A_idx);
  A->FillComplete();

  TestableSolver solver(A, A, params);
  Teuchos::RCP<HYMLS::BaseSolver> base_solver =
    Teuchos::rcp_dynamic_cast<HYMLS::BaseSolver>(solver.Solver());
  TEST_INEQUALITY(base_solver, Teuchos::null);

  Teuchos::RCP<HYMLS::DeflatedSolver> deflated_solver =
    Teuchos::rcp_dynamic_cast<HYMLS::DeflatedSolver>(solver.Solver());
  TEST_EQUALITY(deflated_solver, Teuchos::null);

  Teuchos::RCP<HYMLS::BorderedSolver> bordered_solver =
    Teuchos::rcp_dynamic_cast<HYMLS::BorderedSolver>(solver.Solver());
  TEST_EQUALITY(bordered_solver, Teuchos::null);

  Teuchos::RCP<HYMLS::BorderedDeflatedSolver> borderedDeflated_solver =
    Teuchos::rcp_dynamic_cast<HYMLS::BorderedDeflatedSolver>(solver.Solver());
  TEST_EQUALITY(borderedDeflated_solver, Teuchos::null);
  }

TEUCHOS_UNIT_TEST(Solver, DeflatedSolver)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);

  int n = 1;
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::ParameterList &solverList = params->sublist("Solver");
  solverList.set("Use Deflation", true);

  Epetra_Map map(n, 0, Comm);

  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, map, 1));
  int A_idx = 0;
  double A_val = 1;
  A->InsertGlobalValues(0, 1, &A_val, &A_idx);
  A->FillComplete();

  TestableSolver solver(A, A, params);
  Teuchos::RCP<HYMLS::BaseSolver> base_solver =
    Teuchos::rcp_dynamic_cast<HYMLS::BaseSolver>(solver.Solver());
  TEST_INEQUALITY(base_solver, Teuchos::null);

  Teuchos::RCP<HYMLS::DeflatedSolver> deflated_solver =
    Teuchos::rcp_dynamic_cast<HYMLS::DeflatedSolver>(solver.Solver());
  TEST_INEQUALITY(deflated_solver, Teuchos::null);

  Teuchos::RCP<HYMLS::BorderedSolver> bordered_solver =
    Teuchos::rcp_dynamic_cast<HYMLS::BorderedSolver>(solver.Solver());
  TEST_EQUALITY(bordered_solver, Teuchos::null);

  Teuchos::RCP<HYMLS::BorderedDeflatedSolver> borderedDeflated_solver =
    Teuchos::rcp_dynamic_cast<HYMLS::BorderedDeflatedSolver>(solver.Solver());
  TEST_EQUALITY(borderedDeflated_solver, Teuchos::null);
  }

TEUCHOS_UNIT_TEST(Solver, BorderedSolver)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);

  int n = 1;
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::ParameterList &solverList = params->sublist("Solver");
  solverList.set("Use Bordering", true);

  Epetra_Map map(n, 0, Comm);

  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, map, 1));
  int A_idx = 0;
  double A_val = 1;
  A->InsertGlobalValues(0, 1, &A_val, &A_idx);
  A->FillComplete();

  TestableSolver solver(A, A, params);
  Teuchos::RCP<HYMLS::BaseSolver> base_solver =
    Teuchos::rcp_dynamic_cast<HYMLS::BaseSolver>(solver.Solver());
  TEST_INEQUALITY(base_solver, Teuchos::null);

  Teuchos::RCP<HYMLS::DeflatedSolver> deflated_solver =
    Teuchos::rcp_dynamic_cast<HYMLS::DeflatedSolver>(solver.Solver());
  TEST_EQUALITY(deflated_solver, Teuchos::null);

  Teuchos::RCP<HYMLS::BorderedSolver> bordered_solver =
    Teuchos::rcp_dynamic_cast<HYMLS::BorderedSolver>(solver.Solver());
  TEST_INEQUALITY(bordered_solver, Teuchos::null);

  Teuchos::RCP<HYMLS::BorderedDeflatedSolver> borderedDeflated_solver =
    Teuchos::rcp_dynamic_cast<HYMLS::BorderedDeflatedSolver>(solver.Solver());
  TEST_EQUALITY(borderedDeflated_solver, Teuchos::null);
  }

TEUCHOS_UNIT_TEST(Solver, BorderedDeflatedSolver)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);

  int n = 1;
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::ParameterList &solverList = params->sublist("Solver");
  solverList.set("Use Deflation", true);
  solverList.set("Use Bordering", true);

  Epetra_Map map(n, 0, Comm);

  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, map, 1));
  int A_idx = 0;
  double A_val = 1;
  A->InsertGlobalValues(0, 1, &A_val, &A_idx);
  A->FillComplete();

  TestableSolver solver(A, A, params);
  Teuchos::RCP<HYMLS::BaseSolver> base_solver =
    Teuchos::rcp_dynamic_cast<HYMLS::BaseSolver>(solver.Solver());
  TEST_INEQUALITY(base_solver, Teuchos::null);

  Teuchos::RCP<HYMLS::DeflatedSolver> deflated_solver =
    Teuchos::rcp_dynamic_cast<HYMLS::DeflatedSolver>(solver.Solver());
  TEST_INEQUALITY(deflated_solver, Teuchos::null);

  Teuchos::RCP<HYMLS::BorderedSolver> bordered_solver =
    Teuchos::rcp_dynamic_cast<HYMLS::BorderedSolver>(solver.Solver());
  TEST_INEQUALITY(bordered_solver, Teuchos::null);

  Teuchos::RCP<HYMLS::BorderedDeflatedSolver> borderedDeflated_solver =
    Teuchos::rcp_dynamic_cast<HYMLS::BorderedDeflatedSolver>(solver.Solver());
  TEST_INEQUALITY(borderedDeflated_solver, Teuchos::null);
  }
