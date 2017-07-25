#include "HYMLS_CartesianPartitioner.H"

#include <Teuchos_RCP.hpp>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"

#include "HYMLS_config.h"
#include "HYMLS_UnitTests.H"
#include "HYMLS_FakeComm.H"
#include "HYMLS_Tools.H"

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

TEUCHOS_UNIT_TEST(CartesianPartitioner, 5DOFNodes)
  {
  FakeComm comm;
  comm.SetNumProc(1);

  DISABLE_OUTPUT;

  int nx = 8;
  int ny = 8;
  int nz = 8;
  int sx = 4;
  int sy = 4;
  int sz = 4;
  int dof = 5;
  int n = nx * ny * nz * dof;

  Epetra_Map map(n, 0, comm);
  HYMLS::CartesianPartitioner part(Teuchos::rcp(&map, false), nx, ny, nz, dof, 3);
  part.Partition(sx, sy, sz, false);

  ENABLE_OUTPUT;
  std::vector<hymls_gidx> gids(n, 0);
  for (int sd = 0; sd < part.NumLocalParts(); sd++)
    {
    Teuchos::Array<hymls_gidx> interior_nodes;
    Teuchos::Array<Teuchos::Array<hymls_gidx> > separator_nodes;
    part.GetGroups(sd, interior_nodes, separator_nodes);

    for (hymls_gidx &i: interior_nodes)
      gids[i] = i;

    for (auto &group: separator_nodes)
      for (hymls_gidx &i: group)
        gids[i] = i;
    }
  
  for (int i = 0; i < n; i++)
    TEST_EQUALITY(gids[i], i);
  }

#ifdef HYMLS_LONG_LONG
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
#endif
