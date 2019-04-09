#include "HYMLS_Tester.hpp"
#include "HYMLS_MatrixUtils.hpp"

#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_MpiComm.h"

#include <Teuchos_RCP.hpp>

#include "HYMLS_UnitTests.hpp"

// As a test problem we use a 1D Laplace matrix
Teuchos::RCP<Epetra_CrsMatrix> makeLaplaceMatrix(hymls_gidx n, Epetra_Comm &Comm)
{
  const int maxEntries = 4;
  Epetra_Map map(n, 0, Comm);
  Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::rcp<Epetra_CrsMatrix>(new Epetra_CrsMatrix(Copy, map, maxEntries));

  //Fill the matrix
  hymls_gidx *indices = new hymls_gidx[maxEntries];
  double *values = new double[maxEntries];

  for (int i = 0; i < matrix->NumMyRows(); ++i)
  {
    int k = 0;
    hymls_gidx grid = matrix->GRID64(i);
    for (hymls_gidx gcid = grid-1; gcid <= grid+1; ++gcid)
    {
      if (gcid == grid)
      {
        indices[k] = gcid;
        values[k] = -2.0;
        k++;
      }
      else if (gcid > -1 && gcid < n)
      {
        indices[k] = gcid;
        values[k] = 1.0;
        k++;
      }
    }
    TEUCHOS_ASSERT(matrix->InsertGlobalValues(grid, k, values, indices) == 0);
  }

  delete[] indices;
  delete[] values;

  return matrix;
}

bool syncStatus(bool status, Epetra_Comm const &Comm)
{
  int local_status = static_cast<int>(status);
  int global_status = 1;
  Comm.MinAll(&local_status, &global_status, 1);
  return static_cast<bool>(global_status);
}

TEUCHOS_UNIT_TEST(Tester, isSymmetric)
  {
  const int n = 10;

  int ret = 0;
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  Teuchos::RCP<Epetra_CrsMatrix> matrix;

  // FillComplete not called so the tests will fail
  matrix = makeLaplaceMatrix(n, Comm);
  TEST_EQUALITY_CONST(syncStatus(HYMLS::Tester::isSymmetric(*matrix), Comm), false);
  TEST_EQUALITY_CONST(syncStatus(HYMLS::Tester::isSymmetric(matrix->Graph()), Comm), false);

  // FillComplete called so the tests will succeed
  TEST_EQUALITY_CONST(matrix->FillComplete(), 0);
  TEST_EQUALITY_CONST(syncStatus(HYMLS::Tester::isSymmetric(*matrix), Comm), true);
  TEST_EQUALITY_CONST(syncStatus(HYMLS::Tester::isSymmetric(matrix->Graph()), Comm), true);

  // Change some things to the matrix to make it non-symmetric
  double value;
  hymls_gidx col_index;
  hymls_gidx row_index;

  // Structurally symmetric
  value = 2.0;
  row_index = 4;
  col_index = 5;
  matrix = makeLaplaceMatrix(n, Comm);
  if (matrix->LRID(row_index) != -1)
    ret = matrix->ReplaceGlobalValues(row_index, 1, &value, &col_index);

  TEST_EQUALITY_CONST(ret, 0);
  TEST_EQUALITY_CONST(matrix->FillComplete(), 0);

  TEST_EQUALITY_CONST(syncStatus(HYMLS::Tester::isSymmetric(*matrix), Comm), false);
  TEST_EQUALITY_CONST(syncStatus(HYMLS::Tester::isSymmetric(matrix->Graph()), Comm), true);

  // Unsymmetric
  value = 2.0;
  row_index = 1;
  col_index = 9;
  matrix = makeLaplaceMatrix(n, Comm);
  if (matrix->LRID(row_index) != -1)
    ret = matrix->InsertGlobalValues(row_index, 1, &value, &col_index);

  TEST_EQUALITY_CONST(ret, 0);
  TEST_EQUALITY_CONST(matrix->FillComplete(), 0);

  TEST_EQUALITY_CONST(syncStatus(HYMLS::Tester::isSymmetric(*matrix), Comm), false);
  TEST_EQUALITY_CONST(syncStatus(HYMLS::Tester::isSymmetric(matrix->Graph()), Comm), false);

  // Symmetric with numbers in the corners (harder in parallel cases)
  matrix = makeLaplaceMatrix(n, Comm);
  if (matrix->LRID(row_index) != -1)
    ret = matrix->InsertGlobalValues(row_index, 1, &value, &col_index);
  if (matrix->LRID(col_index) != -1)
    ret = matrix->InsertGlobalValues(col_index, 1, &value, &row_index);

  TEST_EQUALITY_CONST(ret, 0);
  TEST_EQUALITY_CONST(matrix->FillComplete(), 0);

  TEST_EQUALITY_CONST(syncStatus(HYMLS::Tester::isSymmetric(*matrix), Comm), true);
  TEST_EQUALITY_CONST(syncStatus(HYMLS::Tester::isSymmetric(matrix->Graph()), Comm), true);
  }
