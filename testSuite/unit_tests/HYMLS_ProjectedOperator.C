#include "HYMLS_ProjectedOperator.H"

#include <Teuchos_RCP.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>

#include <BelosEpetraAdapter.hpp>
#include <BelosIMGSOrthoManager.hpp>

#include <Epetra_MpiComm.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_SerialDenseMatrix.h>
#include <Epetra_Util.h>

#include "HYMLS_DenseUtils.H"

#include "HYMLS_UnitTests.H"

TEUCHOS_UNIT_TEST(ProjectedOperator, Constructor)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  int n = 100;
  Epetra_Map map(n, 0, Comm);
  Epetra_Util util;

  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, map, 2));
  for (int i = 0; i < n; i++) {
    // int A_idx = util.RandomInt() % n;
    double A_val = util.RandomDouble();

    // Check if we own the index
    if (A->LRID(i) == -1)
      continue;

    // A->InsertGlobalValues(i, 1, &A_val, &A_idx);
    A->InsertGlobalValues(i, 1, &A_val, &i);
  }
  A->FillComplete();

  int m = 10;
  Epetra_MultiVector V(map, m);
  V.Random();

  TEST_THROW(HYMLS::ProjectedOperator op(A, Teuchos::rcp(&V, false), Teuchos::null, true),
    HYMLS::Exception);

  Teuchos::RCP<Teuchos::SerialDenseMatrix<int, double> > mat;
  typedef Belos::IMGSOrthoManager<double, Epetra_MultiVector, Epetra_Operator> orthoMan;
  Teuchos::RCP<orthoMan> imgs = Teuchos::rcp(new orthoMan("hist/orthog/imgs"));

  imgs->normalize(V, mat);
  HYMLS::ProjectedOperator op(A, Teuchos::rcp(&V, false), Teuchos::null, true);
  }

TEUCHOS_UNIT_TEST(ProjectedOperator, ConstructorWithB)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  int n = 100;
  Epetra_Map map(n, 0, Comm);
  Epetra_Util util;

  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, map, 2));
  Teuchos::RCP<Epetra_CrsMatrix> B = Teuchos::rcp(new Epetra_CrsMatrix(Copy, map, 1));
  for (int i = 0; i < n; i++) {
    // int A_idx = util.RandomInt() % n;
    double A_val = util.RandomDouble();

    // Check if we own the index
    if (A->LRID(i) == -1)
      continue;

    // A->InsertGlobalValues(i, 1, &A_val, &A_idx);
    A->InsertGlobalValues(i, 1, &A_val, &i);

    A_val = std::abs(A_val);
    if (i < 50)
      B->InsertGlobalValues(i, 1, &A_val, &i);
  }
  A->FillComplete();
  B->FillComplete();

  int m = 10;
  Epetra_MultiVector V(map, m);
  V.Random();

  Teuchos::RCP<Teuchos::SerialDenseMatrix<int, double> > mat;
  typedef Belos::IMGSOrthoManager<double, Epetra_MultiVector, Epetra_Operator> orthoMan;
  Teuchos::RCP<orthoMan> imgs = Teuchos::rcp(new orthoMan("hist/orthog/imgs", B));

  imgs->normalize(V, mat);
  TEST_THROW(HYMLS::ProjectedOperator op(A, Teuchos::rcp(&V, false), Teuchos::null, true),
    HYMLS::Exception);

  Epetra_MultiVector BV(map, m);
  B->Apply(V, BV);
  
  HYMLS::ProjectedOperator op(A, Teuchos::rcp(&V, false), Teuchos::rcp(&BV, false), true);
  }

TEUCHOS_UNIT_TEST(ProjectedOperator, Apply)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  int n = 100;
  Epetra_Map map(n, 0, Comm);
  Epetra_Util util;

  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, map, 2));
  for (int i = 0; i < n; i++) {
    // int A_idx = util.RandomInt() % n;
    double A_val = util.RandomDouble();

    // Check if we own the index
    if (A->LRID(i) == -1)
      continue;

    // A->InsertGlobalValues(i, 1, &A_val, &A_idx);
    A->InsertGlobalValues(i, 1, &A_val, &i);
  }
  A->FillComplete();

  int m = 10;
  Epetra_MultiVector V(map, m);
  V.Random();

  Teuchos::RCP<Teuchos::SerialDenseMatrix<int, double> > mat;
  typedef Belos::IMGSOrthoManager<double, Epetra_MultiVector, Epetra_Operator> orthoMan;
  Teuchos::RCP<orthoMan> imgs = Teuchos::rcp(new orthoMan("hist/orthog/imgs"));

  imgs->normalize(V, mat);
  HYMLS::ProjectedOperator op(A, Teuchos::rcp(&V, false), Teuchos::null, true);

  Epetra_MultiVector X(map, m);
  Epetra_MultiVector tmp(map, m);
  Epetra_SerialDenseMatrix testMat(m, m);
  X.Random();
  HYMLS::DenseUtils::ApplyOrth(V, X, tmp);
  X = tmp;

  HYMLS::DenseUtils::MatMul(V, X, testMat);

  double nrm = std::abs(testMat.NormInf()) + 1;
  TEST_FLOATING_EQUALITY(nrm, 1.0, 1e-12);

  Epetra_MultiVector Y(map, m);
  Y.Random();
  op.Apply(X, Y);

  HYMLS::DenseUtils::MatMul(V, Y, testMat);

  nrm = std::abs(testMat.NormInf()) + 1;
  TEST_FLOATING_EQUALITY(nrm, 1.0, 1e-12);
  }

TEUCHOS_UNIT_TEST(ProjectedOperator, ApplyWithB)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  int n = 100;
  Epetra_Map map(n, 0, Comm);
  Epetra_Util util;

  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, map, 2));
  Teuchos::RCP<Epetra_CrsMatrix> B = Teuchos::rcp(new Epetra_CrsMatrix(Copy, map, 1));
  for (int i = 0; i < n; i++) {
    // int A_idx = util.RandomInt() % n;
    double A_val = util.RandomDouble();

    // Check if we own the index
    if (A->LRID(i) == -1)
      continue;

    // A->InsertGlobalValues(i, 1, &A_val, &A_idx);
    A->InsertGlobalValues(i, 1, &A_val, &i);

    A_val = std::abs(A_val);
    if (i < 50)
      B->InsertGlobalValues(i, 1, &A_val, &i);
  }
  A->FillComplete();
  B->FillComplete();

  int m = 10;
  Epetra_MultiVector V(map, m);
  V.Random();

  Teuchos::RCP<Teuchos::SerialDenseMatrix<int, double> > mat;
  typedef Belos::IMGSOrthoManager<double, Epetra_MultiVector, Epetra_Operator> orthoMan;
  Teuchos::RCP<orthoMan> imgs = Teuchos::rcp(new orthoMan("hist/orthog/imgs", B));

  imgs->normalize(V, mat);
  Epetra_MultiVector BV(map, m);
  B->Apply(V, BV);
  HYMLS::ProjectedOperator op(A, Teuchos::rcp(&V, false), Teuchos::rcp(&BV, false), true);

  Epetra_MultiVector X(map, m);
  Epetra_MultiVector tmp(map, m);
  Epetra_SerialDenseMatrix testMat(m, m);
  X.Random();
  HYMLS::DenseUtils::ApplyOrth(V, X, tmp, Teuchos::rcp(&BV, false), true);
  X = tmp;

  B->Apply(X, tmp);
  HYMLS::DenseUtils::MatMul(V, tmp, testMat);

  double nrm = std::abs(testMat.NormInf()) + 1;
  TEST_FLOATING_EQUALITY(nrm, 1.0, 1e-12);

  Epetra_MultiVector Y(map, m);
  Y.Random();
  op.Apply(X, Y);

  // TODO: Do we want this? I'm not sure anymore. Need to check in JDQR
  // B->Apply(Y, tmp);
  // HYMLS::DenseUtils::MatMul(V, tmp, testMat);
  HYMLS::DenseUtils::MatMul(V, Y, testMat);

  nrm = std::abs(testMat.NormInf()) + 1;
  TEST_FLOATING_EQUALITY(nrm, 1.0, 1e-12);
  }

TEUCHOS_UNIT_TEST(ProjectedOperator, ApplyInverse)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  int n = 100;
  Epetra_Map map(n, 0, Comm);
  Epetra_Util util;

  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, map, 2));
  for (int i = 0; i < n; i++) {
    int A_idx = n - 1 - util.RandomInt() % (n - i);
    double A_val = util.RandomDouble();

    // Check if we own the index
    if (A->LRID(i) == -1)
      continue;

    // A->InsertGlobalValues(i, 1, &A_val, &A_idx);
    A->InsertGlobalValues(i, 1, &A_val, &i);
  }
  A->FillComplete();

  int m = 10;
  Epetra_MultiVector V(map, m);
  V.Random();

  Teuchos::RCP<Teuchos::SerialDenseMatrix<int, double> > mat;
  typedef Belos::IMGSOrthoManager<double, Epetra_MultiVector, Epetra_Operator> orthoMan;
  Teuchos::RCP<orthoMan> imgs = Teuchos::rcp(new orthoMan("hist/orthog/imgs"));

  imgs->normalize(V, mat);
  HYMLS::ProjectedOperator op(A, Teuchos::rcp(&V, false), Teuchos::null, true);

  Epetra_MultiVector X(map, m);
  Epetra_MultiVector tmp(map, m);
  Epetra_SerialDenseMatrix testMat(m, m);
  X.Random();
  HYMLS::DenseUtils::ApplyOrth(V, X, tmp);
  X = tmp;

  HYMLS::DenseUtils::MatMul(V, X, testMat);

  double nrm = std::abs(testMat.NormInf()) + 1;
  TEST_FLOATING_EQUALITY(nrm, 1.0, 1e-12);

  Epetra_MultiVector Y(map, m);
  Y.Random();
  op.ApplyInverse(X, Y);

  HYMLS::DenseUtils::MatMul(V, Y, testMat);

  nrm = std::abs(testMat.NormInf()) + 1;
  TEST_FLOATING_EQUALITY(nrm, 1.0, 1e-10);
  }

TEUCHOS_UNIT_TEST(ProjectedOperator, ApplyInverseWithB)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  int n = 100;
  Epetra_Map map(n, 0, Comm);
  Epetra_Util util;

  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, map, 2));
  Teuchos::RCP<Epetra_CrsMatrix> B = Teuchos::rcp(new Epetra_CrsMatrix(Copy, map, 1));
  for (int i = 0; i < n; i++) {
    int A_idx = n - 1 - util.RandomInt() % (n - i);
    double A_val = util.RandomDouble();

    // Check if we own the index
    if (A->LRID(i) == -1)
      continue;

    // A->InsertGlobalValues(i, 1, &A_val, &A_idx);
    A->InsertGlobalValues(i, 1, &A_val, &i);

    A_val = std::abs(A_val);
    if (i < 50)
      B->InsertGlobalValues(i, 1, &A_val, &i);
  }
  A->FillComplete();
  B->FillComplete();

  int m = 10;
  Epetra_MultiVector V(map, m);
  V.Random();

  Teuchos::RCP<Teuchos::SerialDenseMatrix<int, double> > mat;
  typedef Belos::IMGSOrthoManager<double, Epetra_MultiVector, Epetra_Operator> orthoMan;
  Teuchos::RCP<orthoMan> imgs = Teuchos::rcp(new orthoMan("hist/orthog/imgs", B));

  imgs->normalize(V, mat);
  Epetra_MultiVector BV(map, m);
  B->Apply(V, BV);
  HYMLS::ProjectedOperator op(A, Teuchos::rcp(&V, false), Teuchos::rcp(&BV, false), true);

  Epetra_MultiVector X(map, m);
  Epetra_MultiVector tmp(map, m);
  Epetra_SerialDenseMatrix testMat(m, m);
  X.Random();
  HYMLS::DenseUtils::ApplyOrth(V, X, tmp, Teuchos::rcp(&BV, false));
  X = tmp;

  // TODO: Do we want this? I'm not sure anymore. Need to check in JDQR
  // B->Apply(X, tmp);
  HYMLS::DenseUtils::MatMul(V, X, testMat);

  double nrm = std::abs(testMat.NormInf()) + 1;
  TEST_FLOATING_EQUALITY(nrm, 1.0, 1e-10);

  Epetra_MultiVector Y(map, m);
  Y.Random();
  op.ApplyInverse(X, Y);

  // TODO: Do we want this? I'm not sure anymore. Need to check in JDQR
  B->Apply(Y, tmp);
  HYMLS::DenseUtils::MatMul(V, tmp, testMat);
  // HYMLS::DenseUtils::MatMul(V, Y, testMat);

  nrm = std::abs(testMat.NormInf()) + 1;
  TEST_FLOATING_EQUALITY(nrm, 1.0, 1e-10);
  }
