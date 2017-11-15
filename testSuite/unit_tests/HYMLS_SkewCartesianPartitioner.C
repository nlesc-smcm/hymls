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

  TEST_EQUALITY(part.PID(0, 0, 0), 3);
  TEST_EQUALITY(part.PID(0, 1, 0), 3);
  TEST_EQUALITY(part.PID(7, 0, 0), 3);
  TEST_EQUALITY(part.PID(3, 4, 0), 2);
  TEST_EQUALITY(part.PID(3, 4, 3), 0);
  TEST_EQUALITY(part.PID(3, 4, 4), 0);
  TEST_EQUALITY(part.PID(0, 0, 4), 1);
  TEST_EQUALITY(part.PID(7, 7, 7), 0);
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
