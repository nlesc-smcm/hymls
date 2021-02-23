#include "HYMLS_BorderedSolver.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Epetra_SerialComm.h>
#include <Epetra_MpiComm.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Util.h>
#include <Epetra_Import.h>
#include <Epetra_SerialDenseMatrix.h>

#include "HYMLS_DenseUtils.hpp"
#include "HYMLS_Preconditioner.hpp"

#include "Galeri_CrsMatrices.h"
#include "HYMLS_UnitTests.hpp"

TEUCHOS_UNIT_TEST(BorderedSolver, ApplyInverse)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = HYMLS::UnitTests::CreateTestParameterList();
  Teuchos::RCP<Epetra_CrsMatrix> A = HYMLS::UnitTests::CreateTestMatrix(params, *comm);
  Teuchos::RCP<HYMLS::Preconditioner> prec = Teuchos::rcp(new HYMLS::Preconditioner(A, params));
  int ierr = prec->Initialize();
  TEST_EQUALITY(ierr, 0);

  ierr = prec->Compute();
  TEST_EQUALITY(ierr, 0);

  Epetra_Map const &map = prec->OperatorRangeMap();

  Teuchos::RCP<Epetra_MultiVector> X = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  X->Random();

  Teuchos::RCP<Epetra_MultiVector> X_EX = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  X_EX->Random();

  Teuchos::RCP<Epetra_MultiVector> B = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  prec->Matrix().Multiply('N', *X_EX, *B);

  Teuchos::RCP<HYMLS::BorderedSolver> solver = Teuchos::rcp(new HYMLS::BorderedSolver(A, prec, params));

  ierr = solver->ApplyInverse(*B, *X);
  TEST_EQUALITY(ierr, 0);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*X, *X_EX), <, 1e-10);
  }

TEUCHOS_UNIT_TEST(BorderedSolver, BorderedApplyInverse)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = HYMLS::UnitTests::CreateTestParameterList();
  Teuchos::RCP<Epetra_CrsMatrix> A = HYMLS::UnitTests::CreateTestMatrix(params, *comm);
  Teuchos::RCP<HYMLS::Preconditioner> prec = Teuchos::rcp(new HYMLS::Preconditioner(A, params));
  int ierr = prec->Initialize();
  TEST_EQUALITY(ierr, 0);

  Epetra_Map const &map = prec->OperatorRangeMap();
  Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  V->Random();
  Teuchos::RCP<Epetra_MultiVector> W = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  W->Random();
  Teuchos::RCP<Epetra_SerialDenseMatrix> C = HYMLS::UnitTests::RandomSerialDenseMatrix(2, 2, *comm);

  Teuchos::RCP<HYMLS::BorderedSolver> solver = Teuchos::rcp(new HYMLS::BorderedSolver(A, prec, params));

  ierr = solver->SetBorder(V, W, C);
  TEST_EQUALITY(ierr, 0);

  ierr = prec->Compute();
  TEST_EQUALITY(ierr, 0);

  Teuchos::RCP<Epetra_MultiVector> X = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  X->Random();

  Teuchos::RCP<Epetra_MultiVector> X_EX = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  X_EX->Random();

  Teuchos::RCP<Epetra_SerialDenseMatrix> X2 = HYMLS::UnitTests::RandomSerialDenseMatrix(2, 2, *comm);

  Teuchos::RCP<Epetra_SerialDenseMatrix> X_EX2 = HYMLS::UnitTests::RandomSerialDenseMatrix(2, 2, *comm);

  Teuchos::RCP<Epetra_MultiVector> B = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  prec->Matrix().Multiply('N', *X_EX, *B);
  ierr = B->Multiply('N', 'N', 1.0, *V, *HYMLS::DenseUtils::CreateView(*X_EX2), 1.0);
  TEST_EQUALITY(ierr, 0);

  Teuchos::RCP<Epetra_SerialDenseMatrix> B2 = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));
  HYMLS::DenseUtils::MatMul(*W, *X_EX, *B2);
  ierr = B2->Multiply('N', 'N', 1.0, *C, *X_EX2, 1.0);
  TEST_EQUALITY(ierr, 0);

  ierr = solver->ApplyInverse(*B, *B2, *X, *X2);
  TEST_EQUALITY(ierr, 0);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*X, *X_EX), <, 1e-10);
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*X2, *X_EX2), <, 1e-10);
  }
