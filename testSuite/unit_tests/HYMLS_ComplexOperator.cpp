#include "HYMLS_ComplexOperator.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Epetra_SerialComm.h>
#include <Epetra_MpiComm.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Util.h>

#include "HYMLS_Macros.hpp"
#include "HYMLS_ComplexVector.hpp"

#include "HYMLS_UnitTests.hpp"

TEUCHOS_UNIT_TEST(ComplexOperator, ApplyReal)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = HYMLS::UnitTests::CreateTestParameterList();
  Teuchos::RCP<Epetra_CrsMatrix> A = HYMLS::UnitTests::CreateTestMatrix(params, *comm);

  Epetra_Map const &map = A->OperatorRangeMap();

  Teuchos::RCP<Epetra_MultiVector> XMV = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  XMV->Random();

  Teuchos::RCP<Epetra_MultiVector> BMV = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  Teuchos::RCP<Epetra_MultiVector> B_EX = Teuchos::rcp(new Epetra_MultiVector(map, 2));

  Epetra_MultiVector imag(View, *XMV, 1, 1);
  imag.PutScalar(0.0);

  int ierr = A->Multiply('N', *XMV, *B_EX);
  TEST_EQUALITY(ierr, 0);

  Teuchos::RCP<HYMLS::ComplexVector<Epetra_MultiVector> > X = Teuchos::rcp(
      new HYMLS::ComplexVector<Epetra_MultiVector>(View, *XMV));
  Teuchos::RCP<HYMLS::ComplexVector<Epetra_MultiVector> > B = Teuchos::rcp(
      new HYMLS::ComplexVector<Epetra_MultiVector>(View, *BMV));
  Teuchos::RCP<HYMLS::ComplexOperator<Epetra_Operator, Epetra_MultiVector> > op = Teuchos::rcp(
      new HYMLS::ComplexOperator<Epetra_Operator, Epetra_MultiVector>(A));

  ierr = op->Apply(*X, *B);
  TEST_EQUALITY(ierr, 0);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*BMV, *B_EX), <, 1e-10);
  }

TEUCHOS_UNIT_TEST(ComplexOperator, ApplyImag)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = HYMLS::UnitTests::CreateTestParameterList();
  Teuchos::RCP<Epetra_CrsMatrix> A = HYMLS::UnitTests::CreateTestMatrix(params, *comm);

  Epetra_Map const &map = A->OperatorRangeMap();

  Teuchos::RCP<Epetra_MultiVector> XMV = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  XMV->Random();

  Teuchos::RCP<Epetra_MultiVector> BMV = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  Teuchos::RCP<Epetra_MultiVector> B_EX = Teuchos::rcp(new Epetra_MultiVector(map, 2));

  Epetra_MultiVector imag(View, *XMV, 0, 1);
  imag.PutScalar(0.0);

  int ierr = A->Multiply('N', *XMV, *B_EX);
  TEST_EQUALITY(ierr, 0);

  Teuchos::RCP<HYMLS::ComplexVector<Epetra_MultiVector> > X = Teuchos::rcp(
      new HYMLS::ComplexVector<Epetra_MultiVector>(View, *XMV));
  Teuchos::RCP<HYMLS::ComplexVector<Epetra_MultiVector> > B = Teuchos::rcp(
      new HYMLS::ComplexVector<Epetra_MultiVector>(View, *BMV));
  Teuchos::RCP<HYMLS::ComplexOperator<Epetra_Operator, Epetra_MultiVector> > op = Teuchos::rcp(
      new HYMLS::ComplexOperator<Epetra_Operator, Epetra_MultiVector>(A));

  ierr = op->Apply(*X, *B);
  TEST_EQUALITY(ierr, 0);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*BMV, *B_EX), <, 1e-10);
  }

TEUCHOS_UNIT_TEST(ComplexOperator, Apply)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = HYMLS::UnitTests::CreateTestParameterList();
  Teuchos::RCP<Epetra_CrsMatrix> A = HYMLS::UnitTests::CreateTestMatrix(params, *comm);

  Epetra_Map const &map = A->OperatorRangeMap();

  Teuchos::RCP<Epetra_MultiVector> XMV = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  XMV->Random();

  Teuchos::RCP<Epetra_MultiVector> BMV = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  Teuchos::RCP<Epetra_MultiVector> B_EX = Teuchos::rcp(new Epetra_MultiVector(map, 2));

  int ierr = A->Multiply('N', *XMV, *B_EX);
  TEST_EQUALITY(ierr, 0);

  Teuchos::RCP<HYMLS::ComplexVector<Epetra_MultiVector> > X = Teuchos::rcp(
      new HYMLS::ComplexVector<Epetra_MultiVector>(View, *XMV));
  Teuchos::RCP<HYMLS::ComplexVector<Epetra_MultiVector> > B = Teuchos::rcp(
      new HYMLS::ComplexVector<Epetra_MultiVector>(View, *BMV));
  Teuchos::RCP<HYMLS::ComplexOperator<Epetra_Operator, Epetra_MultiVector> > op = Teuchos::rcp(
      new HYMLS::ComplexOperator<Epetra_Operator, Epetra_MultiVector>(A));

  ierr = op->Apply(*X, *B);
  TEST_EQUALITY(ierr, 0);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*BMV, *B_EX), <, 1e-10);
  }

TEUCHOS_UNIT_TEST(ComplexOperator, ApplyInverseReal)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = HYMLS::UnitTests::CreateTestParameterList();
  Teuchos::RCP<Epetra_CrsMatrix> A = HYMLS::UnitTests::CreateTestMatrix(params, *comm);

  Epetra_Map const &map = A->OperatorRangeMap();

  Teuchos::RCP<Epetra_MultiVector> XMV = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  XMV->Random();

  Teuchos::RCP<Epetra_MultiVector> BMV = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  Teuchos::RCP<Epetra_MultiVector> B_EX = Teuchos::rcp(new Epetra_MultiVector(map, 2));

  Epetra_MultiVector imag(View, *XMV, 1, 1);
  imag.PutScalar(0.0);

  int ierr = A->ApplyInverse(*XMV, *B_EX);
  TEST_EQUALITY(ierr, 0);

  Teuchos::RCP<HYMLS::ComplexVector<Epetra_MultiVector> > X = Teuchos::rcp(
      new HYMLS::ComplexVector<Epetra_MultiVector>(View, *XMV));
  Teuchos::RCP<HYMLS::ComplexVector<Epetra_MultiVector> > B = Teuchos::rcp(
      new HYMLS::ComplexVector<Epetra_MultiVector>(View, *BMV));
  Teuchos::RCP<HYMLS::ComplexOperator<Epetra_Operator, Epetra_MultiVector> > op = Teuchos::rcp(
      new HYMLS::ComplexOperator<Epetra_Operator, Epetra_MultiVector>(A));

  ierr = op->ApplyInverse(*X, *B);
  TEST_EQUALITY(ierr, 0);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*BMV, *B_EX), <, 1e-10);
  }

TEUCHOS_UNIT_TEST(ComplexOperator, ApplyInverseImag)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = HYMLS::UnitTests::CreateTestParameterList();
  Teuchos::RCP<Epetra_CrsMatrix> A = HYMLS::UnitTests::CreateTestMatrix(params, *comm);

  Epetra_Map const &map = A->OperatorRangeMap();

  Teuchos::RCP<Epetra_MultiVector> XMV = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  XMV->Random();

  Teuchos::RCP<Epetra_MultiVector> BMV = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  Teuchos::RCP<Epetra_MultiVector> B_EX = Teuchos::rcp(new Epetra_MultiVector(map, 2));

  Epetra_MultiVector imag(View, *XMV, 0, 1);
  imag.PutScalar(0.0);

  int ierr = A->ApplyInverse(*XMV, *B_EX);
  TEST_EQUALITY(ierr, 0);

  Teuchos::RCP<HYMLS::ComplexVector<Epetra_MultiVector> > X = Teuchos::rcp(
      new HYMLS::ComplexVector<Epetra_MultiVector>(View, *XMV));
  Teuchos::RCP<HYMLS::ComplexVector<Epetra_MultiVector> > B = Teuchos::rcp(
      new HYMLS::ComplexVector<Epetra_MultiVector>(View, *BMV));
  Teuchos::RCP<HYMLS::ComplexOperator<Epetra_Operator, Epetra_MultiVector> > op = Teuchos::rcp(
      new HYMLS::ComplexOperator<Epetra_Operator, Epetra_MultiVector>(A));

  ierr = op->ApplyInverse(*X, *B);
  TEST_EQUALITY(ierr, 0);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*BMV, *B_EX), <, 1e-10);
  }

TEUCHOS_UNIT_TEST(ComplexOperator, ApplyInverse)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = HYMLS::UnitTests::CreateTestParameterList();
  Teuchos::RCP<Epetra_CrsMatrix> A = HYMLS::UnitTests::CreateTestMatrix(params, *comm);

  Epetra_Map const &map = A->OperatorRangeMap();

  Teuchos::RCP<Epetra_MultiVector> XMV = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  XMV->Random();

  Teuchos::RCP<Epetra_MultiVector> BMV = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  Teuchos::RCP<Epetra_MultiVector> B_EX = Teuchos::rcp(new Epetra_MultiVector(map, 2));

  int ierr = A->ApplyInverse(*XMV, *B_EX);
  TEST_EQUALITY(ierr, 0);

  Teuchos::RCP<HYMLS::ComplexVector<Epetra_MultiVector> > X = Teuchos::rcp(
      new HYMLS::ComplexVector<Epetra_MultiVector>(View, *XMV));
  Teuchos::RCP<HYMLS::ComplexVector<Epetra_MultiVector> > B = Teuchos::rcp(
      new HYMLS::ComplexVector<Epetra_MultiVector>(View, *BMV));
  Teuchos::RCP<HYMLS::ComplexOperator<Epetra_Operator, Epetra_MultiVector> > op = Teuchos::rcp(
      new HYMLS::ComplexOperator<Epetra_Operator, Epetra_MultiVector>(A));

  ierr = op->ApplyInverse(*X, *B);
  TEST_EQUALITY(ierr, 0);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*BMV, *B_EX), <, 1e-10);
  }
