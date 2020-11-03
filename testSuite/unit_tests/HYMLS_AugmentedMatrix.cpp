#include "HYMLS_AugmentedMatrix.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Epetra_MpiComm.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_Import.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_SerialDenseMatrix.h>

#include "HYMLS_Macros.hpp"
#include "HYMLS_DenseUtils.hpp"

#include "HYMLS_UnitTests.hpp"

Teuchos::RCP<Epetra_CrsMatrix> createMatrix(
  Teuchos::RCP<Epetra_Comm> const &comm)
  {
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(20, 0, *comm));
  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map, 2));

  Epetra_Util util;
  for (hymls_gidx i = 0; i < A->NumGlobalRows64(); i++) {
    double A_val2 = std::abs(util.RandomDouble());

    // Check if we own the index
    if (A->LRID(i) == -1)
      continue;

    CHECK_ZERO(A->InsertGlobalValues(i, 1, &A_val2, &i));
    }
  CHECK_ZERO(A->FillComplete());

  return A;
  }

Teuchos::RCP<Epetra_MultiVector> merge_vector(Teuchos::RCP<Epetra_MultiVector> X, Teuchos::RCP<Epetra_SerialDenseMatrix> X2,
  Epetra_Map const &map, Epetra_Map const &map2)
  {
  int *global_elements = new int[map.NumMyElements() + X2->M()];

  int pos = 0;
  for (int i = 0; i < map.NumMyElements(); i++)
    global_elements[pos++] = map.GID(i);
  for (int i = 0; i < X2->M(); i++)
    global_elements[pos++] = map.NumGlobalElements() + i;

  Teuchos::RCP<Epetra_Map> extended_map = Teuchos::rcp(new Epetra_Map(-1, pos, global_elements, 0, map.Comm()));

  Teuchos::RCP<Epetra_MultiVector> extended_X = Teuchos::rcp(
    new Epetra_MultiVector(Copy, *extended_map, X->Values(), X->Stride(), X->NumVectors()));

  Teuchos::RCP<Epetra_MultiVector> imported_X = Teuchos::rcp(
    new Epetra_MultiVector(map2, X->NumVectors()));
  Epetra_Import importer_X(map2, *extended_map);
  CHECK_ZERO((*imported_X).Import(*X, importer_X, Insert));

  for (int k = 0; k < X->NumVectors(); k++)
    for (int j = 0; j < map2.NumMyElements(); j++)
      if (map2.GID64(j) >= map.NumGlobalElements())
        (*imported_X)[k][j] = (*X2)(j - map.NumGlobalElements(), k);

  return imported_X;
  }

TEUCHOS_UNIT_TEST(AugmentedMatrix, ExtractMyRowCopy_Original)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Epetra_CrsMatrix> A = createMatrix(comm);

  Epetra_Map const &map = A->OperatorRangeMap();
  Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  V->PutScalar(0);
  Teuchos::RCP<Epetra_MultiVector> W = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  W->PutScalar(0);
  Teuchos::RCP<Epetra_SerialDenseMatrix> C = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));

  Teuchos::RCP<HYMLS::AugmentedMatrix> A2 = Teuchos::rcp(new HYMLS::AugmentedMatrix(A, V, W, C));
  Epetra_Map const &map2 = A2->OperatorRangeMap();

  Teuchos::RCP<Epetra_MultiVector> X = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  X->Random();

  Teuchos::RCP<Epetra_SerialDenseMatrix> X2 = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));

  Teuchos::RCP<Epetra_MultiVector> B = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  A->Multiply('N', *X, *B);
  B->Multiply('N', 'N', 1.0, *V, *HYMLS::DenseUtils::CreateView(*X2), 1.0);

  Teuchos::RCP<Epetra_SerialDenseMatrix> B2 = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));
  HYMLS::DenseUtils::CreateView(*B2)->Multiply('T', 'N', 1.0, *W, *X, 0.0);
  B2->Multiply('N', 'N', 1.0, *C, *X2, 1.0);

  Epetra_Map const &col_map = A2->RowMatrixColMap();
  Teuchos::RCP<Epetra_MultiVector> imported_X = merge_vector(X, X2, map, col_map);
  Teuchos::RCP<Epetra_MultiVector> imported_B = merge_vector(B, B2, map, map2);

  for (int i = 0; i < map.NumMyElements(); i++)
    {
    int num_entries;
    int length = A2->NumMyCols();
    int *indices = new int[length];
    double *values = new double[length];

    int ierr = A2->ExtractMyRowCopy(i, length, num_entries, values, indices);
    TEST_EQUALITY(ierr, 0);
    TEST_INEQUALITY(num_entries, 0);

    for (int k = 0; k < B->NumVectors(); k++)
      {
      double b_value = 0;
      for (int j = 0; j < num_entries; j++)
        b_value += values[j] * (*imported_X)[k][indices[j]];

      // Check if they are the same and nonzero
      TEST_COMPARE(std::abs((*imported_B)[k][i]), >, 1e-12);
      TEST_FLOATING_EQUALITY(b_value, (*imported_B)[k][i], 1e-12);
      }
    }
  }

TEUCHOS_UNIT_TEST(AugmentedMatrix, ExtractMyRowCopy_V)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Epetra_CrsMatrix> A = createMatrix(comm);

  Epetra_Map const &map = A->OperatorRangeMap();
  Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  V->Random();
  Teuchos::RCP<Epetra_MultiVector> W = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  W->PutScalar(0);
  Teuchos::RCP<Epetra_SerialDenseMatrix> C = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));

  Teuchos::RCP<HYMLS::AugmentedMatrix> A2 = Teuchos::rcp(new HYMLS::AugmentedMatrix(A, V, W, C));
  Epetra_Map const &map2 = A2->OperatorRangeMap();

  Teuchos::RCP<Epetra_MultiVector> X = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  X->Random();

  Teuchos::RCP<Epetra_SerialDenseMatrix> X2 = HYMLS::UnitTests::RandomSerialDenseMatrix(2, 2, *comm);

  Teuchos::RCP<Epetra_MultiVector> B = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  A->Multiply('N', *X, *B);
  B->Multiply('N', 'N', 1.0, *V, *HYMLS::DenseUtils::CreateView(*X2), 1.0);

  Teuchos::RCP<Epetra_SerialDenseMatrix> B2 = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));
  HYMLS::DenseUtils::CreateView(*B2)->Multiply('T', 'N', 1.0, *W, *X, 0.0);
  B2->Multiply('N', 'N', 1.0, *C, *X2, 1.0);

  Epetra_Map const &col_map = A2->RowMatrixColMap();
  Teuchos::RCP<Epetra_MultiVector> imported_X = merge_vector(X, X2, map, col_map);
  Teuchos::RCP<Epetra_MultiVector> imported_B = merge_vector(B, B2, map, map2);

  for (int i = 0; i < map.NumMyElements(); i++)
    {
    int num_entries;
    int length = A2->NumMyCols();
    int *indices = new int[length];
    double *values = new double[length];

    int ierr = A2->ExtractMyRowCopy(i, length, num_entries, values, indices);
    TEST_EQUALITY(ierr, 0);
    TEST_INEQUALITY(num_entries, 0);

    for (int k = 0; k < B->NumVectors(); k++)
      {
      double b_value = 0;
      for (int j = 0; j < num_entries; j++)
        b_value += values[j] * (*imported_X)[k][indices[j]];

      // Check if they are the same and nonzero
      TEST_COMPARE(std::abs((*imported_B)[k][i]), >, 1e-12);
      TEST_FLOATING_EQUALITY(b_value, (*imported_B)[k][i], 1e-12);
      }
    }
  }

TEUCHOS_UNIT_TEST(AugmentedMatrix, ExtractMyRowCopy_W)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Epetra_CrsMatrix> A = createMatrix(comm);

  Epetra_Map const &map = A->OperatorRangeMap();
  Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  V->PutScalar(0);
  Teuchos::RCP<Epetra_MultiVector> W = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  W->Random();
  Teuchos::RCP<Epetra_SerialDenseMatrix> C = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));

  Teuchos::RCP<HYMLS::AugmentedMatrix> A2 = Teuchos::rcp(new HYMLS::AugmentedMatrix(A, V, W, C));
  Epetra_Map const &map2 = A2->OperatorRangeMap();

  Teuchos::RCP<Epetra_MultiVector> X = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  X->Random();

  Teuchos::RCP<Epetra_SerialDenseMatrix> X2 = HYMLS::UnitTests::RandomSerialDenseMatrix(2, 2, *comm);

  Teuchos::RCP<Epetra_MultiVector> B = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  A->Multiply('N', *X, *B);
  B->Multiply('N', 'N', 1.0, *V, *HYMLS::DenseUtils::CreateView(*X2), 1.0);

  Teuchos::RCP<Epetra_SerialDenseMatrix> B2 = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));
  HYMLS::DenseUtils::CreateView(*B2)->Multiply('T', 'N', 1.0, *W, *X, 0.0);
  B2->Multiply('N', 'N', 1.0, *C, *X2, 1.0);

  Epetra_Map const &col_map = A2->RowMatrixColMap();
  Teuchos::RCP<Epetra_MultiVector> imported_X = merge_vector(X, X2, map, col_map);
  Teuchos::RCP<Epetra_MultiVector> imported_B = merge_vector(B, B2, map, map2);

  for (int i = 0; i < map2.NumMyElements(); i++)
    {
    int num_entries;
    int length = A2->NumMyCols();
    int *indices = new int[length];
    double *values = new double[length];

    int ierr = A2->ExtractMyRowCopy(i, length, num_entries, values, indices);
    TEST_EQUALITY(ierr, 0);
    TEST_INEQUALITY(num_entries, 0);

    for (int k = 0; k < B->NumVectors(); k++)
      {
      double b_value = 0;
      for (int j = 0; j < num_entries; j++)
        b_value += values[j] * (*imported_X)[k][indices[j]];

      // Check if they are the same and nonzero
      TEST_COMPARE(std::abs((*imported_B)[k][i]), >, 1e-12);
      TEST_FLOATING_EQUALITY(b_value, (*imported_B)[k][i], 1e-12);
      }
    }
  }

TEUCHOS_UNIT_TEST(AugmentedMatrix, ExtractMyRowCopy_C)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Epetra_CrsMatrix> A = createMatrix(comm);

  Epetra_Map const &map = A->OperatorRangeMap();
  Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  V->PutScalar(0);
  Teuchos::RCP<Epetra_MultiVector> W = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  W->PutScalar(0);
  Teuchos::RCP<Epetra_SerialDenseMatrix> C = HYMLS::UnitTests::RandomSerialDenseMatrix(2, 2, *comm);

  Teuchos::RCP<HYMLS::AugmentedMatrix> A2 = Teuchos::rcp(new HYMLS::AugmentedMatrix(A, V, W, C));
  Epetra_Map const &map2 = A2->OperatorRangeMap();

  Teuchos::RCP<Epetra_MultiVector> X = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  X->Random();

  Teuchos::RCP<Epetra_SerialDenseMatrix> X2 = HYMLS::UnitTests::RandomSerialDenseMatrix(2, 2, *comm);

  Teuchos::RCP<Epetra_MultiVector> B = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  A->Multiply('N', *X, *B);
  B->Multiply('N', 'N', 1.0, *V, *HYMLS::DenseUtils::CreateView(*X2), 1.0);

  Teuchos::RCP<Epetra_SerialDenseMatrix> B2 = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));
  HYMLS::DenseUtils::CreateView(*B2)->Multiply('T', 'N', 1.0, *W, *X, 0.0);
  B2->Multiply('N', 'N', 1.0, *C, *X2, 1.0);

  Epetra_Map const &col_map = A2->RowMatrixColMap();
  Teuchos::RCP<Epetra_MultiVector> imported_X = merge_vector(X, X2, map, col_map);
  Teuchos::RCP<Epetra_MultiVector> imported_B = merge_vector(B, B2, map, map2);

  for (int i = 0; i < map2.NumMyElements(); i++)
    {
    int num_entries;
    int length = A2->NumMyCols();
    int *indices = new int[length];
    double *values = new double[length];

    int ierr = A2->ExtractMyRowCopy(i, length, num_entries, values, indices);
    TEST_EQUALITY(ierr, 0);
    TEST_INEQUALITY(num_entries, 0);

    for (int k = 0; k < B->NumVectors(); k++)
      {
      double b_value = 0;
      for (int j = 0; j < num_entries; j++)
        b_value += values[j] * (*imported_X)[k][indices[j]];

      // Check if they are the same and nonzero
      TEST_COMPARE(std::abs((*imported_B)[k][i]), >, 1e-12);
      TEST_FLOATING_EQUALITY(b_value, (*imported_B)[k][i], 1e-12);
      }
    }
  }

TEUCHOS_UNIT_TEST(AugmentedMatrix, ExtractMyRowCopy)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Epetra_CrsMatrix> A = createMatrix(comm);

  Epetra_Map const &map = A->OperatorRangeMap();
  Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  V->Random();
  Teuchos::RCP<Epetra_MultiVector> W = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  W->Random();
  Teuchos::RCP<Epetra_SerialDenseMatrix> C = HYMLS::UnitTests::RandomSerialDenseMatrix(2, 2, *comm);

  Teuchos::RCP<HYMLS::AugmentedMatrix> A2 = Teuchos::rcp(new HYMLS::AugmentedMatrix(A, V, W, C));
  Epetra_Map const &map2 = A2->OperatorRangeMap();

  Teuchos::RCP<Epetra_MultiVector> X = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  X->Random();

  Teuchos::RCP<Epetra_SerialDenseMatrix> X2 = HYMLS::UnitTests::RandomSerialDenseMatrix(2, 2, *comm);

  Teuchos::RCP<Epetra_MultiVector> B = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  A->Multiply('N', *X, *B);
  B->Multiply('N', 'N', 1.0, *V, *HYMLS::DenseUtils::CreateView(*X2), 1.0);

  Teuchos::RCP<Epetra_SerialDenseMatrix> B2 = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));
  HYMLS::DenseUtils::CreateView(*B2)->Multiply('T', 'N', 1.0, *W, *X, 0.0);
  B2->Multiply('N', 'N', 1.0, *C, *X2, 1.0);

  Epetra_Map const &col_map = A2->RowMatrixColMap();
  Teuchos::RCP<Epetra_MultiVector> imported_X = merge_vector(X, X2, map, col_map);
  Teuchos::RCP<Epetra_MultiVector> imported_B = merge_vector(B, B2, map, map2);

  for (int i = 0; i < map2.NumMyElements(); i++)
    {
    int num_entries;
    int length = A2->NumMyCols();
    int *indices = new int[length];
    double *values = new double[length];

    int ierr = A2->ExtractMyRowCopy(i, length, num_entries, values, indices);
    TEST_EQUALITY(ierr, 0);
    TEST_INEQUALITY(num_entries, 0);

    for (int k = 0; k < B->NumVectors(); k++)
      {
      double b_value = 0;
      for (int j = 0; j < num_entries; j++)
        b_value += values[j] * (*imported_X)[k][indices[j]];

      // Check if they are the same and nonzero
      TEST_COMPARE(std::abs((*imported_B)[k][i]), >, 1e-12);
      TEST_FLOATING_EQUALITY(b_value, (*imported_B)[k][i], 1e-12);
      }
    }
  }
