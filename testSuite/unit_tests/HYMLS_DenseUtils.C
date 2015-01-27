#include "HYMLS_Tools.H"
#include "HYMLS_DenseUtils.H"
#include "HYMLS_MatrixUtils.H"
#include "HYMLS_Macros.H"

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"
#include "Epetra_MultiVector.h"
#include "Epetra_SerialDenseMatrix.h"

#include "HYMLS_UnitTests.H"

TEUCHOS_UNIT_TEST(DenseUtils, MatMul)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);

  // Construct some vectors and matrices
  int n = 10;
  Epetra_Map map(n, 0, Comm);
  Epetra_MultiVector x(map, 2);
  Epetra_MultiVector y(map, 3);

  x.PutScalar(2.0);
  y.PutScalar(3.0);

  Epetra_SerialDenseMatrix z(2, 2);

  // Test normal behaviour
  TEST_EQUALITY(HYMLS::DenseUtils::MatMul(x, y, z), 0);

  TEST_EQUALITY(z[0][0], 2.0 * 3.0 * n);
  TEST_EQUALITY(z[1][0], 2.0 * 3.0 * n);
  TEST_EQUALITY(z[2][0], 2.0 * 3.0 * n);
  TEST_EQUALITY(z[0][1], 2.0 * 3.0 * n);
  TEST_EQUALITY(z[1][1], 2.0 * 3.0 * n);
  TEST_EQUALITY(z[2][1], 2.0 * 3.0 * n);

  // Test error when different maps are used
  Epetra_Map map_long(20, 0, Comm);
  Epetra_MultiVector x_long(map_long, 2);

  TEST_EQUALITY(HYMLS::DenseUtils::MatMul(x_long, y, z), -1);

  // Test more variable vectors
  x.ReplaceGlobalValue(3, 0, 1.0);
  y.ReplaceGlobalValue(4, 1, 5.0);
  TEST_EQUALITY(HYMLS::DenseUtils::MatMul(x, y, z), 0);

  TEST_EQUALITY(z[0][0], 2.0 * 3.0 * n - 3.0);
  TEST_EQUALITY(z[1][0], 2.0 * 3.0 * n + 1.0);
  TEST_EQUALITY(z[2][0], 2.0 * 3.0 * n - 3.0);
  TEST_EQUALITY(z[0][1], 2.0 * 3.0 * n);
  TEST_EQUALITY(z[1][1], 2.0 * 3.0 * n + 4.0);
  TEST_EQUALITY(z[2][1], 2.0 * 3.0 * n);
  }

TEUCHOS_UNIT_TEST(DenseUtils, ApplyOrth)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);

  // Construct some vectors and matrices
  int n = 10;
  Epetra_Map map(n, 0, Comm);
  Epetra_MultiVector x(map, 1);
  Epetra_MultiVector y(map, 1);
  Epetra_MultiVector z(map, 1);

  HYMLS::MatrixUtils::Random(x);
  HYMLS::MatrixUtils::Random(y);

  double result;
  x.Norm2(&result);
  x.Scale(1.0 / result);

  // Test normal behaviour
  TEST_EQUALITY(HYMLS::DenseUtils::ApplyOrth(x, y, z), 0);

  Epetra_MultiVector tmp = z;
  tmp.Update(1.0, y, -1.0);
  TEST_EQUALITY(z.Dot(tmp, &result), 0);
  TEST_COMPARE  (std::abs(result), <, HYMLS_SMALL_ENTRY);
  }
