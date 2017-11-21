#include "HYMLS_SkewCartesianPartitioner.H"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include "Epetra_SerialComm.h"
#include "Epetra_Map.h"

#include "HYMLS_config.h"
#include "HYMLS_UnitTests.H"
#include "HYMLS_FakeComm.H"
#include "HYMLS_Tools.H"

class TestableSkewCartesianPartitioner: public HYMLS::SkewCartesianPartitioner
  {
public:
  TestableSkewCartesianPartitioner(Teuchos::RCP<const Epetra_Map> map,
    Teuchos::RCP<Teuchos::ParameterList> const &params,
    Teuchos::RCP<const Epetra_Comm> const &comm)
    :
    HYMLS::SkewCartesianPartitioner(map, params, comm)
    {}

  int PID(int i, int j, int k)
    {
    return HYMLS::SkewCartesianPartitioner::PID(i, j, k);
    }

  int GetSubdomainPosition(int sd, int sx, int &x, int &y, int &z) const
    {
    return HYMLS::SkewCartesianPartitioner::GetSubdomainPosition(sd, sx, x, y, z);
    }

  int GetSubdomainID(int sx, int x, int y, int z) const
    {
    return HYMLS::SkewCartesianPartitioner::GetSubdomainID(sx, x, y, z);
    }
  };

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, operator)
  {
  Teuchos::RCP<FakeComm> comm = Teuchos::rcp(new FakeComm);
  comm->SetNumProc(4);
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
    new Teuchos::ParameterList);
  params->sublist("Problem").set("nx", 8);
  params->sublist("Problem").set("ny", 8);
  params->sublist("Problem").set("nz", 8);
  params->sublist("Problem").set("Degrees of Freedom", 3);
  params->sublist("Preconditioner").set("Separator Length", 4);

  TestableSkewCartesianPartitioner part(Teuchos::null, params, comm);
  part.Partition(false);

  ENABLE_OUTPUT;

  TEST_EQUALITY(part(0, 0, 0), 0);
  TEST_EQUALITY(part(0, 1, 0), 2);
  TEST_EQUALITY(part(7, 0, 0), 4);
  TEST_EQUALITY(part(3, 4, 0), 8);
  TEST_EQUALITY(part(3, 4, 3), 20);
  TEST_EQUALITY(part(3, 4, 4), 20);
  TEST_EQUALITY(part(0, 0, 4), 12);
  TEST_EQUALITY(part(7, 7, 7), 21);
  }

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, PID)
  {
  Teuchos::RCP<FakeComm> comm = Teuchos::rcp(new FakeComm);
  comm->SetNumProc(4);
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
    new Teuchos::ParameterList);
  params->sublist("Problem").set("nx", 8);
  params->sublist("Problem").set("ny", 8);
  params->sublist("Problem").set("nz", 8);
  params->sublist("Problem").set("Degrees of Freedom", 3);
  params->sublist("Preconditioner").set("Separator Length", 4);

  TestableSkewCartesianPartitioner part(Teuchos::null, params, comm);
  part.Partition(false);

  ENABLE_OUTPUT;

  TEST_EQUALITY(part.PID(0, 0, 0), 1);
  TEST_EQUALITY(part.PID(0, 1, 0), 0);
  TEST_EQUALITY(part.PID(7, 0, 0), 0);
  TEST_EQUALITY(part.PID(3, 4, 0), 1);
  TEST_EQUALITY(part.PID(3, 4, 3), 3);
  TEST_EQUALITY(part.PID(3, 4, 4), 3);
  TEST_EQUALITY(part.PID(0, 0, 4), 3);
  TEST_EQUALITY(part.PID(5, 5, 5), 2);
  TEST_EQUALITY(part.PID(7, 7, 7), 3);
  }

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, GetSubdomain)
  {
  Teuchos::RCP<FakeComm> comm = Teuchos::rcp(new FakeComm);
  comm->SetNumProc(1);
  DISABLE_OUTPUT;

  int nx = 12;
  int cl = 6;
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
    new Teuchos::ParameterList);
  params->sublist("Problem").set("nx", nx);
  params->sublist("Problem").set("ny", nx);
  params->sublist("Problem").set("nz", nx);
  params->sublist("Problem").set("x-periodic", true);
  params->sublist("Problem").set("y-periodic", true);
  params->sublist("Problem").set("z-periodic", true);
  params->sublist("Problem").set("Equations", "Stokes-C");
  params->sublist("Preconditioner").set("Separator Length", cl);

  TestableSkewCartesianPartitioner part(Teuchos::null, params, comm);
  part.Partition(false);

  ENABLE_OUTPUT;
  for (int sd = 0; sd < part.NumLocalParts(); sd++)
    {
    int gsd = part.SubdomainMap().GID(sd);
    int i, j, k;
    part.GetSubdomainPosition(gsd, cl, i, j, k);

    i = ((i + cl / 2 - 1) % nx + nx) % nx;
    j = (j % nx + nx) % nx;
    k = ((k + cl) % nx + nx) % nx;

    int gsd2 = part.GetSubdomainID(cl, i, j, k);
    TEST_EQUALITY(i, i);
    TEST_EQUALITY(j, j);
    TEST_EQUALITY(k, k);
    TEST_EQUALITY(gsd, gsd2);
    }
  }

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, 2DNodes)
  {
  Teuchos::RCP<FakeComm> comm = Teuchos::rcp(new FakeComm);
  comm->SetNumProc(1);
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
    new Teuchos::ParameterList);
  params->sublist("Problem").set("nx", 8);
  params->sublist("Problem").set("ny", 8);
  params->sublist("Problem").set("nz", 1);
  params->sublist("Problem").set("Dimension", 2);
  params->sublist("Problem").set("Equations", "Stokes-C");
  params->sublist("Preconditioner").set("Separator Length", 4);

  TestableSkewCartesianPartitioner part(Teuchos::null, params, comm);
  part.Partition(false);

  int n = part.Map().NumGlobalElements64();
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

  ENABLE_OUTPUT;
  for (int i = 0; i < n; i++)
    TEST_EQUALITY(gids[i], i);
  }

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, 1PSepPerDomain2D)
  {
  Teuchos::RCP<FakeComm> comm = Teuchos::rcp(new FakeComm);
  comm->SetNumProc(1);
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
    new Teuchos::ParameterList);
  params->sublist("Problem").set("nx", 8);
  params->sublist("Problem").set("ny", 8);
  params->sublist("Problem").set("nz", 1);
  params->sublist("Problem").set("Dimension", 2);
  params->sublist("Problem").set("Equations", "Stokes-C");
  params->sublist("Preconditioner").set("Separator Length", 4);

  TestableSkewCartesianPartitioner part(Teuchos::null, params, comm);
  part.Partition(false);

  ENABLE_OUTPUT;
  int n = part.Map().NumGlobalElements64();
  std::vector<hymls_gidx> gids(n, 0);
  for (int sd = 0; sd < part.NumLocalParts(); sd++)
    {
    int numPNodes = 0;
    Teuchos::Array<hymls_gidx> interior_nodes;
    Teuchos::Array<Teuchos::Array<hymls_gidx> > separator_nodes;
    part.GetGroups(sd, interior_nodes, separator_nodes);

    for (auto &group: separator_nodes)
      for (hymls_gidx &i: group)
        if (i % 3 == 2)
          numPNodes++;
    TEST_EQUALITY(numPNodes, 1);
    }
  }

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, 3DNodes)
  {
  Teuchos::RCP<FakeComm> comm = Teuchos::rcp(new FakeComm);
  comm->SetNumProc(1);
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
    new Teuchos::ParameterList);
  params->sublist("Problem").set("nx", 8);
  params->sublist("Problem").set("ny", 8);
  params->sublist("Problem").set("nz", 8);
  params->sublist("Problem").set("Degrees of Freedom", 4);
  params->sublist("Problem").set("Equations", "Stokes-C");
  params->sublist("Preconditioner").set("Separator Length", 4);

  TestableSkewCartesianPartitioner part(Teuchos::null, params, comm);
  part.Partition(false);

  ENABLE_OUTPUT;
  int n = part.Map().NumGlobalElements64();
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

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, 5DOFNodes)
  {
  Teuchos::RCP<FakeComm> comm = Teuchos::rcp(new FakeComm);
  comm->SetNumProc(1);
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
    new Teuchos::ParameterList);
  params->sublist("Problem").set("nx", 8);
  params->sublist("Problem").set("ny", 8);
  params->sublist("Problem").set("nz", 8);
  params->sublist("Problem").set("Equations", "Bous-C");
  params->sublist("Preconditioner").set("Separator Length", 4);

  TestableSkewCartesianPartitioner part(Teuchos::null, params, comm);
  part.Partition(false);

  ENABLE_OUTPUT;
  int n = part.Map().NumGlobalElements64();
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

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, DifferentSeparatorsSameProcs)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
    new Teuchos::ParameterList);
  params->sublist("Problem").set("nx", 16);
  params->sublist("Problem").set("ny", 16);
  params->sublist("Problem").set("nz", 16);
  params->sublist("Problem").set("Equations", "Stokes-C");
  params->sublist("Preconditioner").set("Separator Length", 2);

  TestableSkewCartesianPartitioner part(Teuchos::null, params, comm);
  part.Partition(true);

  Teuchos::RCP<Teuchos::ParameterList> params2 = Teuchos::rcp(
    new Teuchos::ParameterList);
  params2->sublist("Problem").set("nx", 16);
  params2->sublist("Problem").set("ny", 16);
  params2->sublist("Problem").set("nz", 16);
  params2->sublist("Problem").set("Equations", "Stokes-C");
  params2->sublist("Preconditioner").set("Separator Length", 8);

  TestableSkewCartesianPartitioner part2(Teuchos::null, params2, comm);
  part2.Partition(true);

  ENABLE_OUTPUT;
  Epetra_Map const &map1 = part.Map();
  Epetra_Map const &map2 = part2.Map();
  for (int i = 0; i < map1.NumMyElements(); i++)
    {
    TEST_INEQUALITY(map2.LID(map1.GID64(i)), -1);
    }
  }

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, 1PSepPerDomain3D)
  {
  Teuchos::RCP<FakeComm> comm = Teuchos::rcp(new FakeComm);
  comm->SetNumProc(1);
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
    new Teuchos::ParameterList);
  params->sublist("Problem").set("nx", 8);
  params->sublist("Problem").set("ny", 8);
  params->sublist("Problem").set("nz", 8);
  params->sublist("Problem").set("Equations", "Stokes-C");
  params->sublist("Preconditioner").set("Separator Length", 4);

  TestableSkewCartesianPartitioner part(Teuchos::null, params, comm);
  part.Partition(false);

  ENABLE_OUTPUT;
  int n = part.Map().NumGlobalElements64();
  std::vector<hymls_gidx> gids(n, 0);
  for (int sd = 0; sd < part.NumLocalParts(); sd++)
    {
    int numPNodes = 0;
    Teuchos::Array<hymls_gidx> interior_nodes;
    Teuchos::Array<Teuchos::Array<hymls_gidx> > separator_nodes;
    part.GetGroups(sd, interior_nodes, separator_nodes);

    for (auto &group: separator_nodes)
      for (hymls_gidx &i: group)
        if (i % 4 == 3)
          numPNodes++;
    TEST_EQUALITY(numPNodes, 1);
    }
  }

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, NoEmptyProcs16)
  {
  int nprocs = 16;
  Teuchos::RCP<FakeComm> comm = Teuchos::rcp(new FakeComm);
  DISABLE_OUTPUT;

  comm->SetNumProc(nprocs);
  for (int i = 0; i < nprocs; i++)
    {
    comm->SetMyPID(i);

    Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
      new Teuchos::ParameterList);
    params->sublist("Problem").set("nx", 16);
    params->sublist("Problem").set("ny", 16);
    params->sublist("Problem").set("nz", 16);
    params->sublist("Problem").set("Equations", "Stokes-C");
    params->sublist("Preconditioner").set("Separator Length", 4);

    TestableSkewCartesianPartitioner part(Teuchos::null, params, comm);
    part.Partition(false);

    TEST_INEQUALITY(part.NumLocalParts(), 0);
    }

  }

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, NoEmptyProcs128)
  {
  int nprocs = 128;
  Teuchos::RCP<FakeComm> comm = Teuchos::rcp(new FakeComm);
  DISABLE_OUTPUT;

  comm->SetNumProc(nprocs);
  for (int i = 0; i < nprocs; i++)
    {
    comm->SetMyPID(i);

    Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
      new Teuchos::ParameterList);
    params->sublist("Problem").set("nx", 32);
    params->sublist("Problem").set("ny", 32);
    params->sublist("Problem").set("nz", 32);
    params->sublist("Problem").set("Equations", "Stokes-C");
    params->sublist("Preconditioner").set("Separator Length", 4);

    TestableSkewCartesianPartitioner part(Teuchos::null, params, comm);
    part.Partition(false);

    TEST_INEQUALITY(part.NumLocalParts(), 0);
    }

  }

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, SameNumSubdomains)
  {
  int nprocs = 16;
  Teuchos::RCP<FakeComm> comm = Teuchos::rcp(new FakeComm);
  DISABLE_OUTPUT;

  comm->SetNumProc(nprocs);

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
    new Teuchos::ParameterList);
  params->sublist("Problem").set("nx", 16);
  params->sublist("Problem").set("ny", 16);
  params->sublist("Problem").set("nz", 16);
  params->sublist("Problem").set("Equations", "Stokes-C");
  params->sublist("Preconditioner").set("Separator Length", 4);

  TestableSkewCartesianPartitioner part(Teuchos::null, params, comm);
  part.Partition(false);

  int num = part.NumLocalParts();

  nprocs = 128;
  comm->SetNumProc(nprocs);

  params->sublist("Problem").set("nx", 32);
  params->sublist("Problem").set("ny", 32);
  params->sublist("Problem").set("nz", 32);
  params->sublist("Problem").set("Equations", "Stokes-C");
  params->sublist("Preconditioner").set("Separator Length", 4);

  TestableSkewCartesianPartitioner part2(Teuchos::null, params, comm);
  part2.Partition(false);

  ENABLE_OUTPUT;
  TEST_EQUALITY(part2.NumLocalParts(), num);
  }

#ifdef HYMLS_LONG_LONG
TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, GID64)
  {
  Teuchos::RCP<FakeComm> comm = Teuchos::rcp(new FakeComm);
  comm->SetNumProc(8192);
  comm->SetMyPID(8191);
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
    new Teuchos::ParameterList);
  params->sublist("Problem").set("nx", 1024);
  params->sublist("Problem").set("ny", 1024);
  params->sublist("Problem").set("nz", 1024);
  params->sublist("Problem").set("Degrees of Freedom", 4);
  params->sublist("Preconditioner").set("Separator Length", 16);

  TestableSkewCartesianPartitioner part(Teuchos::null, params, comm);
  part.Partition(false);

  TEST_EQUALITY(part(nx-2, ny-1, nz-1), (nz / sx + 1 ) * (2 * nx / sx * ny / sx + ny / sx + nx / sx) - 1);

  Teuchos::Array<hymls_gidx> interior_nodes;
  Teuchos::Array<Teuchos::Array<hymls_gidx> > separator_nodes;
  part.GetGroups(part.NumLocalParts()-1, interior_nodes, separator_nodes);
  for (hymls_gidx i: interior_nodes)
    TEST_COMPARE(i, >=, 0);
  }
#endif


TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, operator2D)
  {
  Teuchos::RCP<FakeComm> comm = Teuchos::rcp(new FakeComm);
  comm->SetNumProc(1);
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
    new Teuchos::ParameterList);
  params->sublist("Problem").set("nx", 12);
  params->sublist("Problem").set("ny", 12);
  params->sublist("Problem").set("Degrees of Freedom", 2);
  params->sublist("Preconditioner").set("Separator Length", 6);

  TestableSkewCartesianPartitioner part(Teuchos::null, params, comm);
  part.Partition(false);

  ENABLE_OUTPUT;

  TEST_EQUALITY(part(0,  0, 0), 0);
  TEST_EQUALITY(part(1,  0, 0), 0);
  TEST_EQUALITY(part(2,  0, 0), 0);
  TEST_EQUALITY(part(3,  0, 0), 0);
  TEST_EQUALITY(part(4,  0, 0), 0);
  TEST_EQUALITY(part(5,  0, 0), 3);
  TEST_EQUALITY(part(6,  0, 0), 1);
  TEST_EQUALITY(part(7,  0, 0), 1);
  TEST_EQUALITY(part(8,  0, 0), 1);
  TEST_EQUALITY(part(9,  0, 0), 1);
  TEST_EQUALITY(part(10, 0, 0), 1);
  TEST_EQUALITY(part(11, 0, 0), 4);

  TEST_EQUALITY(part(0,  1, 0), 2);
  TEST_EQUALITY(part(1,  1, 0), 0);
  TEST_EQUALITY(part(2,  1, 0), 0);
  TEST_EQUALITY(part(3,  1, 0), 0);
  TEST_EQUALITY(part(4,  1, 0), 3);
  TEST_EQUALITY(part(5,  1, 0), 3);
  TEST_EQUALITY(part(6,  1, 0), 3);
  TEST_EQUALITY(part(7,  1, 0), 1);
  TEST_EQUALITY(part(8,  1, 0), 1);
  TEST_EQUALITY(part(9,  1, 0), 1);
  TEST_EQUALITY(part(10, 1, 0), 4);
  TEST_EQUALITY(part(11, 1, 0), 4);

  TEST_EQUALITY(part(0,  2, 0), 2);
  TEST_EQUALITY(part(1,  2, 0), 2);
  TEST_EQUALITY(part(2,  2, 0), 0);
  TEST_EQUALITY(part(3,  2, 0), 3);
  TEST_EQUALITY(part(4,  2, 0), 3);
  TEST_EQUALITY(part(5,  2, 0), 3);
  TEST_EQUALITY(part(6,  2, 0), 3);
  TEST_EQUALITY(part(7,  2, 0), 3);
  TEST_EQUALITY(part(8,  2, 0), 1);
  TEST_EQUALITY(part(9,  2, 0), 4);
  TEST_EQUALITY(part(10, 2, 0), 4);
  TEST_EQUALITY(part(11, 2, 0), 4);

  TEST_EQUALITY(part(0,  3, 0), 2);
  TEST_EQUALITY(part(1,  3, 0), 2);
  TEST_EQUALITY(part(2,  3, 0), 5);
  TEST_EQUALITY(part(3,  3, 0), 3);
  TEST_EQUALITY(part(4,  3, 0), 3);
  TEST_EQUALITY(part(5,  3, 0), 3);
  TEST_EQUALITY(part(6,  3, 0), 3);
  TEST_EQUALITY(part(7,  3, 0), 3);
  TEST_EQUALITY(part(8,  3, 0), 6);
  TEST_EQUALITY(part(9,  3, 0), 4);
  TEST_EQUALITY(part(10, 3, 0), 4);
  TEST_EQUALITY(part(11, 3, 0), 4);

  TEST_EQUALITY(part(0,  4, 0), 2);
  TEST_EQUALITY(part(1,  4, 0), 5);
  TEST_EQUALITY(part(2,  4, 0), 5);
  TEST_EQUALITY(part(3,  4, 0), 5);
  TEST_EQUALITY(part(4,  4, 0), 3);
  TEST_EQUALITY(part(5,  4, 0), 3);
  TEST_EQUALITY(part(6,  4, 0), 3);
  TEST_EQUALITY(part(7,  4, 0), 6);
  TEST_EQUALITY(part(8,  4, 0), 6);
  TEST_EQUALITY(part(9,  4, 0), 6);
  TEST_EQUALITY(part(10, 4, 0), 4);
  TEST_EQUALITY(part(11, 4, 0), 4);

  TEST_EQUALITY(part(0,  5, 0), 5);
  TEST_EQUALITY(part(1,  5, 0), 5);
  TEST_EQUALITY(part(2,  5, 0), 5);
  TEST_EQUALITY(part(3,  5, 0), 5);
  TEST_EQUALITY(part(4,  5, 0), 5);
  TEST_EQUALITY(part(5,  5, 0), 3);
  TEST_EQUALITY(part(6,  5, 0), 6);
  TEST_EQUALITY(part(7,  5, 0), 6);
  TEST_EQUALITY(part(8,  5, 0), 6);
  TEST_EQUALITY(part(9,  5, 0), 6);
  TEST_EQUALITY(part(10, 5, 0), 6);
  TEST_EQUALITY(part(11, 5, 0), 4);

  TEST_EQUALITY(part(0,  6, 0), 5);
  TEST_EQUALITY(part(1,  6, 0), 5);
  TEST_EQUALITY(part(2,  6, 0), 5);
  TEST_EQUALITY(part(3,  6, 0), 5);
  TEST_EQUALITY(part(4,  6, 0), 5);
  TEST_EQUALITY(part(5,  6, 0), 8);
  TEST_EQUALITY(part(6,  6, 0), 6);
  TEST_EQUALITY(part(7,  6, 0), 6);
  TEST_EQUALITY(part(8,  6, 0), 6);
  TEST_EQUALITY(part(9,  6, 0), 6);
  TEST_EQUALITY(part(10, 6, 0), 6);
  TEST_EQUALITY(part(11, 6, 0), 9);

  TEST_EQUALITY(part(0,  7, 0), 7);
  TEST_EQUALITY(part(1,  7, 0), 5);
  TEST_EQUALITY(part(2,  7, 0), 5);
  TEST_EQUALITY(part(3,  7, 0), 5);
  TEST_EQUALITY(part(4,  7, 0), 8);
  TEST_EQUALITY(part(5,  7, 0), 8);
  TEST_EQUALITY(part(6,  7, 0), 8);
  TEST_EQUALITY(part(7,  7, 0), 6);
  TEST_EQUALITY(part(8,  7, 0), 6);
  TEST_EQUALITY(part(9,  7, 0), 6);
  TEST_EQUALITY(part(10, 7, 0), 9);
  TEST_EQUALITY(part(11, 7, 0), 9);

  TEST_EQUALITY(part(0,  8, 0), 7);
  TEST_EQUALITY(part(1,  8, 0), 7);
  TEST_EQUALITY(part(2,  8, 0), 5);
  TEST_EQUALITY(part(3,  8, 0), 8);
  TEST_EQUALITY(part(4,  8, 0), 8);
  TEST_EQUALITY(part(5,  8, 0), 8);
  TEST_EQUALITY(part(6,  8, 0), 8);
  TEST_EQUALITY(part(7,  8, 0), 8);
  TEST_EQUALITY(part(8,  8, 0), 6);
  TEST_EQUALITY(part(9,  8, 0), 9);
  TEST_EQUALITY(part(10, 8, 0), 9);
  TEST_EQUALITY(part(11, 8, 0), 9);

  TEST_EQUALITY(part(0,  9, 0), 7);
  TEST_EQUALITY(part(1,  9, 0), 7);
  TEST_EQUALITY(part(2,  9, 0), 10);
  TEST_EQUALITY(part(3,  9, 0), 8);
  TEST_EQUALITY(part(4,  9, 0), 8);
  TEST_EQUALITY(part(5,  9, 0), 8);
  TEST_EQUALITY(part(6,  9, 0), 8);
  TEST_EQUALITY(part(7,  9, 0), 8);
  TEST_EQUALITY(part(8,  9, 0), 11);
  TEST_EQUALITY(part(9,  9, 0), 9);
  TEST_EQUALITY(part(10, 9, 0), 9);
  TEST_EQUALITY(part(11, 9, 0), 9);

  TEST_EQUALITY(part(0,  10, 0), 7);
  TEST_EQUALITY(part(1,  10, 0), 10);
  TEST_EQUALITY(part(2,  10, 0), 10);
  TEST_EQUALITY(part(3,  10, 0), 10);
  TEST_EQUALITY(part(4,  10, 0), 8);
  TEST_EQUALITY(part(5,  10, 0), 8);
  TEST_EQUALITY(part(6,  10, 0), 8);
  TEST_EQUALITY(part(7,  10, 0), 11);
  TEST_EQUALITY(part(8,  10, 0), 11);
  TEST_EQUALITY(part(9,  10, 0), 11);
  TEST_EQUALITY(part(10, 10, 0), 9);
  TEST_EQUALITY(part(11, 10, 0), 9);

  TEST_EQUALITY(part(0,  11, 0), 10);
  TEST_EQUALITY(part(1,  11, 0), 10);
  TEST_EQUALITY(part(2,  11, 0), 10);
  TEST_EQUALITY(part(3,  11, 0), 10);
  TEST_EQUALITY(part(4,  11, 0), 10);
  TEST_EQUALITY(part(5,  11, 0), 8);
  TEST_EQUALITY(part(6,  11, 0), 11);
  TEST_EQUALITY(part(7,  11, 0), 11);
  TEST_EQUALITY(part(8,  11, 0), 11);
  TEST_EQUALITY(part(9,  11, 0), 11);
  TEST_EQUALITY(part(10, 11, 0), 11);
  TEST_EQUALITY(part(11, 11, 0), 9);
  }
