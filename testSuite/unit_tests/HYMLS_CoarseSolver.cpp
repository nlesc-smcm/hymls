#include "HYMLS_CoarseSolver.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Epetra_MpiComm.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_SerialDenseMatrix.h>

#include "HYMLS_Macros.hpp"
#include "HYMLS_DenseUtils.hpp"

#include "HYMLS_UnitTests.hpp"

Teuchos::RCP<HYMLS::CoarseSolver> createCoarseSolver(
  Teuchos::RCP<Teuchos::ParameterList> &params,
  Teuchos::RCP<Epetra_Comm> const &comm)
  {
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(100, 0, *comm));
  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map, 2));

  Epetra_Util util;
  for (hymls_gidx i = 0; i < A->NumGlobalRows64(); i++) {
    // int A_idx = util.RandomInt() % n;
    // double A_val = -std::abs(util.RandomDouble());
    double A_val2 = std::abs(util.RandomDouble());

    // Check if we own the index
    if (A->LRID(i) == -1)
      continue;

    // CHECK_ZERO(A->InsertGlobalValues(i, 1, &A_val, &A_idx));
    CHECK_ZERO(A->InsertGlobalValues(i, 1, &A_val2, &i));
  }
  CHECK_ZERO(A->FillComplete());

  Teuchos::RCP<HYMLS::CoarseSolver> solver = Teuchos::rcp(new HYMLS::CoarseSolver(A, 0));
  solver->SetParameters(*params);

  return solver;
  }

TEUCHOS_UNIT_TEST(CoarseSolver, ApplyInverse)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::RCP<HYMLS::CoarseSolver> solver = createCoarseSolver(params, comm);
  int ierr = solver->Initialize();
  TEST_EQUALITY(ierr, 0);

  ierr = solver->Compute();
  TEST_EQUALITY(ierr, 0);

  Epetra_Map const &map = solver->OperatorRangeMap();

  Teuchos::RCP<Epetra_MultiVector> X = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  X->Random();

  Teuchos::RCP<Epetra_MultiVector> X_EX = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  X_EX->Random();

  Teuchos::RCP<Epetra_MultiVector> B = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  ierr = solver->Matrix().Multiply('N', *X_EX, *B);
  TEST_EQUALITY(ierr, 0);

  ierr = solver->ApplyInverse(*B, *X);
  TEST_EQUALITY(ierr, 0);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*X, *X_EX), <, 1e-10);
  }

TEUCHOS_UNIT_TEST(CoarseSolver, BorderedApplyInverse)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::RCP<HYMLS::CoarseSolver> solver = createCoarseSolver(params, comm);
  int ierr = solver->Initialize();

  Epetra_Map const &map = solver->OperatorRangeMap();
  Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  V->Random();
  Teuchos::RCP<Epetra_MultiVector> W = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  W->Random();
  Teuchos::RCP<Epetra_SerialDenseMatrix> C = HYMLS::UnitTests::RandomSerialDenseMatrix(2, 2, *comm);

  ierr = solver->SetBorder(V, W, C);
  TEST_EQUALITY(ierr, 0);

  ierr = solver->Compute();
  TEST_EQUALITY(ierr, 0);

  Teuchos::RCP<Epetra_MultiVector> X = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  X->Random();

  Teuchos::RCP<Epetra_MultiVector> X_EX = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  X_EX->Random();

  Teuchos::RCP<Epetra_SerialDenseMatrix> X2 = HYMLS::UnitTests::RandomSerialDenseMatrix(2, 2, *comm);

  Teuchos::RCP<Epetra_SerialDenseMatrix> X_EX2 = HYMLS::UnitTests::RandomSerialDenseMatrix(2, 2, *comm);

  Teuchos::RCP<Epetra_MultiVector> B = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  solver->Matrix().Multiply('N', *X_EX, *B);
  ierr = B->Multiply('N', 'N', 1.0, *V, *HYMLS::DenseUtils::CreateView(*X_EX2), 1.0);
  TEST_EQUALITY(ierr, 0);

  Teuchos::RCP<Epetra_SerialDenseMatrix> B2 = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));
  HYMLS::DenseUtils::CreateView(*B2)->Multiply('T', 'N', 1.0, *W, *X_EX, 0.0);
  ierr = B2->Multiply('N', 'N', 1.0, *C, *X_EX2, 1.0);
  TEST_EQUALITY(ierr, 0);

  ierr = solver->ApplyInverse(*B, *B2, *X, *X2);
  TEST_EQUALITY(ierr, 0);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*X, *X_EX), <, 1e-10);
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*X2, *X_EX2), <, 1e-10);
  }
