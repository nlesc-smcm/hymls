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
  hymls_gidx nz = 2;
  hymls_gidx dof = 4;
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
  DISABLE_OUTPUT;

  hymls_gidx nx = 1024;
  hymls_gidx ny = 1024;
  hymls_gidx nz = 1024;
  hymls_gidx dof = 4;
  hymls_gidx n = nx * ny * nz * dof;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(n, 0, Comm));

  HYMLS::CartesianPartitioner part(map, nx, ny, nz, dof);
  CHECK_ZERO(part.Partition(4, 4, 4, false));

  Teuchos::Array<hymls_gidx> interior_nodes;
  Teuchos::Array<Teuchos::Array<hymls_gidx> > separator_nodes;
  part.GetGroups(part.NumLocalParts()-1, interior_nodes, separator_nodes);

  long long last = n - 1;
  TEST_EQUALITY(interior_nodes.back(), last);
  }
