#include "HYMLS_CartesianPartitioner.H"
#include "HYMLS_Tools.H"

#include <Teuchos_RCP.hpp>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"

#include "HYMLS_UnitTests.H"
#include "HYMLS_FakeComm.H"

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

TEUCHOS_UNIT_TEST(CartesianPartitioner, GID64)
  {
  FakeComm Comm;
  Comm.SetNumProc(8192);
  Comm.SetMyPID(8191);
  // DISABLE_OUTPUT;

  int nx = 1024;
  int ny = 1024;
  int nz = 1024;
  int dof = 4;
  hymls_gidx n = (hymls_gidx)nx * ny * nz * dof;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(n, 0, Comm));

  HYMLS::CartesianPartitioner part(map, nx, ny, nz, dof);
  part.Partition(4, 4, 4, true);

  long long last = 
    (long long)(1024 - 1024 / 32) * nx * ny * dof +
    (long long)(1024 - 1024 / 16) * nx * dof +
    (long long)(1024 - 1024 / 16) * dof;
  TEST_EQUALITY(part.Map().MyGlobalElements64()[0], last);
  }

