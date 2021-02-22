#include "HYMLS_ComplexSolver.hpp"
#include "HYMLS_ComplexBorderedSolver.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Epetra_SerialComm.h>
#include <Epetra_MpiComm.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Util.h>

#include "HYMLS_Macros.hpp"
#include "HYMLS_Preconditioner.hpp"

#include "Galeri_CrsMatrices.h"

#include "HYMLS_UnitTests.hpp"

TEUCHOS_UNIT_TEST(ComplexSolver, ApplyInverseReal)
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

  Epetra_MultiVector imag(View, *X_EX, 1, 1);
  imag.PutScalar(0.0);

  Teuchos::RCP<Epetra_MultiVector> B = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  prec->Matrix().Multiply('N', *X_EX, *B);

  Teuchos::RCP<HYMLS::ComplexSolver> solver = Teuchos::rcp(new HYMLS::ComplexSolver(A, prec, params));

  ierr = solver->ApplyInverse(*B, *X);
  TEST_EQUALITY(ierr, 0);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*X, *X_EX), <, 1e-10);
  }

TEUCHOS_UNIT_TEST(ComplexSolver, ApplyInverseImag)
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

  Epetra_MultiVector real(View, *X_EX, 0, 1);
  real.PutScalar(0.0);

  Teuchos::RCP<Epetra_MultiVector> B = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  prec->Matrix().Multiply('N', *X_EX, *B);

  Teuchos::RCP<HYMLS::ComplexSolver> solver = Teuchos::rcp(new HYMLS::ComplexSolver(A, prec, params));

  ierr = solver->ApplyInverse(*B, *X);
  TEST_EQUALITY(ierr, 0);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*X, *X_EX), <, 1e-10);
  }

TEUCHOS_UNIT_TEST(ComplexSolver, ApplyInverse)
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

  Teuchos::RCP<HYMLS::ComplexSolver> solver = Teuchos::rcp(new HYMLS::ComplexSolver(A, prec, params));

  ierr = solver->ApplyInverse(*B, *X);
  TEST_EQUALITY(ierr, 0);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*X, *X_EX), <, 1e-10);
  }

TEUCHOS_UNIT_TEST(ComplexBorderedSolver, ApplyInverseReal)
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

  Epetra_MultiVector imag(View, *X_EX, 1, 1);
  imag.PutScalar(0.0);

  Teuchos::RCP<Epetra_MultiVector> B = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  prec->Matrix().Multiply('N', *X_EX, *B);

  Teuchos::RCP<HYMLS::ComplexBorderedSolver> solver = Teuchos::rcp(new HYMLS::ComplexBorderedSolver(A, prec, params));

  ierr = solver->ApplyInverse(*B, *X);
  TEST_EQUALITY(ierr, 0);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*X, *X_EX), <, 1e-10);
  }

TEUCHOS_UNIT_TEST(ComplexBorderedSolver, ApplyInverseImag)
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

  Epetra_MultiVector real(View, *X_EX, 0, 1);
  real.PutScalar(0.0);

  Teuchos::RCP<Epetra_MultiVector> B = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  prec->Matrix().Multiply('N', *X_EX, *B);

  Teuchos::RCP<HYMLS::ComplexBorderedSolver> solver = Teuchos::rcp(new HYMLS::ComplexBorderedSolver(A, prec, params));

  ierr = solver->ApplyInverse(*B, *X);
  TEST_EQUALITY(ierr, 0);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*X, *X_EX), <, 1e-10);
  }

TEUCHOS_UNIT_TEST(ComplexBorderedSolver, ApplyInverse)
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

  Teuchos::RCP<HYMLS::ComplexBorderedSolver> solver = Teuchos::rcp(new HYMLS::ComplexBorderedSolver(A, prec, params));

  ierr = solver->ApplyInverse(*B, *X);
  TEST_EQUALITY(ierr, 0);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*X, *X_EX), <, 1e-10);
  }
