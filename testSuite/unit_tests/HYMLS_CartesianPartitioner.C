#include "HYMLS_CartesianPartitioner.H"
#include "HYMLS_Tools.H"

#include <Teuchos_RCP.hpp>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"

#include "HYMLS_UnitTests.H"

TEUCHOS_UNIT_TEST(CartesianPartitioner, Partition2D)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  DISABLE_OUTPUT;

  int nx = 8;
  int ny = 8;
  int nz = 1;
  int dof = 4;
  hymls_gidx n = nx * ny * nz * dof;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(n, 0, Comm));

  HYMLS::CartesianPartitioner part(map, nx, ny, nz, dof);

  // This will cause an exception when compiled with TESTING if it fails
  part.Partition(8, true);

  TEST_EQUALITY(part(0, 0, 0), 0);
  TEST_EQUALITY(part(0, 3, 0), 2);
  TEST_EQUALITY(part(6, 3, 0), 3);
  }

TEUCHOS_UNIT_TEST(CartesianPartitioner, Partition3D)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  DISABLE_OUTPUT;

  int nx = 8;
  int ny = 8;
  int nz = 2;
  int dof = 4;
  hymls_gidx n = nx * ny * nz * dof;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(n, 0, Comm));

  HYMLS::CartesianPartitioner part(map, nx, ny, nz, dof);

  // This will cause an exception when compiled with TESTING if it fails
  part.Partition(8, true);

  TEST_EQUALITY(part(0, 0, 0), 0);
  TEST_EQUALITY(part(0, 3, 0), 2);
  TEST_EQUALITY(part(6, 3, 0), 3);
  TEST_EQUALITY(part(0, 3, 1), 2);
  }
