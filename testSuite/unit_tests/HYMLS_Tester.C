#include "HYMLS_Tester.H"

#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_SerialComm.h"

#include <Teuchos_RCP.hpp>

#include "HYMLS_UnitTests.H"

// As a test problem we use a 1D Laplace matrix
Teuchos::RCP<Epetra_CrsMatrix> makeLaplaceMatrix(int n, Epetra_Comm &comm)
{
  const int maxEntries = 4;
  Epetra_Map map(n, 0, comm);
  Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::rcp<Epetra_CrsMatrix>(new Epetra_CrsMatrix(Copy, map, maxEntries));

  //Fill the matrix
  int *indices = new int[maxEntries];
  double *values = new double[maxEntries];

  for (int i = 0; i < n; ++i)
  {
    int k = 0;
    for (int j = i-1; j <= i+1; ++j)
    {
      if (j == i)
      {
        indices[k] = j;
        values[k] = -2.0;
        k++;
      }
      else if (j > -1 && j < n)
      {
        indices[k] = j;
        values[k] = 1.0;
        k++;
      }
    }
    TEUCHOS_ASSERT(matrix->InsertGlobalValues(i, k, values, indices) == 0);
  }

  delete[] indices;
  delete[] values;

  return matrix;
}

TEUCHOS_UNIT_TEST(Tester, isSymmetric)
  {
  const int n = 10;

  int ret;
  Epetra_SerialComm comm;
  Teuchos::RCP<Epetra_CrsMatrix> matrix;

  // FillComplete not called so the tests will fail
  matrix = makeLaplaceMatrix(n, comm);
  TEST_EQUALITY_CONST(HYMLS::Tester::isSymmetric(*matrix), false);
  TEST_EQUALITY_CONST(HYMLS::Tester::isSymmetric(matrix->Graph()), false);

  // FillComplete called so the tests will succeed
  TEST_EQUALITY_CONST(matrix->FillComplete(), 0);
  TEST_EQUALITY_CONST(HYMLS::Tester::isSymmetric(*matrix), true);
  TEST_EQUALITY_CONST(HYMLS::Tester::isSymmetric(matrix->Graph()), true);

  // Change some things to the matrix to make it non-symmetric
  int index;
  double value;

  value = 2.0;
  index = 5;
  matrix = makeLaplaceMatrix(n, comm);
  ret = matrix->ReplaceGlobalValues(4, 1, &value, &index);
  TEST_EQUALITY_CONST(ret, 0);
  TEST_EQUALITY_CONST(matrix->FillComplete(), 0);

  // Structurally symmetric
  TEST_EQUALITY_CONST(HYMLS::Tester::isSymmetric(*matrix), false);
  TEST_EQUALITY_CONST(HYMLS::Tester::isSymmetric(matrix->Graph()), true);

  value = 2.0;
  index = 9;
  matrix = makeLaplaceMatrix(n, comm);
  ret = matrix->InsertGlobalValues(4, 1, &value, &index);
  TEST_EQUALITY_CONST(ret, 0);
  TEST_EQUALITY_CONST(matrix->FillComplete(), 0);

  // Unsymmetric
  TEST_EQUALITY_CONST(HYMLS::Tester::isSymmetric(*matrix), false);
  TEST_EQUALITY_CONST(HYMLS::Tester::isSymmetric(matrix->Graph()), false);
  }
