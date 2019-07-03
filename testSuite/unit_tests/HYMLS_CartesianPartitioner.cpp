#include "HYMLS_CartesianPartitioner.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"

#include "HYMLS_config.h"
#include "HYMLS_UnitTests.hpp"
#include "HYMLS_FakeComm.hpp"
#include "HYMLS_Tools.hpp"

TEUCHOS_UNIT_TEST(CartesianPartitioner, Partition2D)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
    new Teuchos::ParameterList);
  params->sublist("Problem").set("nx", 8);
  params->sublist("Problem").set("ny", 8);
  params->sublist("Problem").set("nz", 1);
  params->sublist("Problem").set("Degrees of Freedom", 4);

  params->sublist("Preconditioner").set("Separator Length (x)", 4);
  params->sublist("Preconditioner").set("Separator Length (y)", 2);
  params->sublist("Preconditioner").set("Separator Length (z)", 1);

  HYMLS::CartesianPartitioner part(Teuchos::null, params, *comm);

  // This will cause an exception when compiled with TESTING if it fails
  part.Partition(true);

  TEST_EQUALITY(part(0, 0, 0), 0);
  TEST_EQUALITY(part(0, 3, 0), 2);
  TEST_EQUALITY(part(6, 3, 0), 3);
  }

TEUCHOS_UNIT_TEST(CartesianPartitioner, Partition3D)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
    new Teuchos::ParameterList);
  params->sublist("Problem").set("nx", 8);
  params->sublist("Problem").set("ny", 8);
  params->sublist("Problem").set("nz", 2);
  params->sublist("Problem").set("Degrees of Freedom", 4);

  params->sublist("Preconditioner").set("Separator Length (x)", 4);
  params->sublist("Preconditioner").set("Separator Length (y)", 2);
  params->sublist("Preconditioner").set("Separator Length (z)", 2);

  HYMLS::CartesianPartitioner part(Teuchos::null, params, *comm);

  // This will cause an exception when compiled with TESTING if it fails
  part.Partition(true);

  TEST_EQUALITY(part(0, 0, 0), 0);
  TEST_EQUALITY(part(0, 3, 0), 2);
  TEST_EQUALITY(part(6, 3, 0), 3);
  TEST_EQUALITY(part(0, 3, 1), 2);
  }

TEUCHOS_UNIT_TEST(CartesianPartitioner, 5DOFNodes)
  {
  Teuchos::RCP<FakeComm> comm = Teuchos::rcp(new FakeComm);
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
    new Teuchos::ParameterList);
  params->sublist("Problem").set("nx", 8);
  params->sublist("Problem").set("ny", 8);
  params->sublist("Problem").set("nz", 8);
  params->sublist("Problem").set("Equations", "Bous-C");
  params->sublist("Preconditioner").set("Separator Length", 4);

  HYMLS::CartesianPartitioner part(Teuchos::null, params, *comm);

  // This will cause an exception when compiled with TESTING if it fails
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

TEUCHOS_UNIT_TEST(CartesianPartitioner, SamePartEveryProc)
  {
  int nprocs = 64;
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

    HYMLS::CartesianPartitioner part(Teuchos::null, params, *comm);
    TEST_MAYTHROW(part.Partition(false));

    int exp = 32 / 4 * 32 / 4 * 32 / 4 / nprocs;
    TEST_EQUALITY(part.NumLocalParts(), exp);
    }
  }

TEUCHOS_UNIT_TEST(CartesianPartitioner, MoveMap)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
    new Teuchos::ParameterList);
  params->sublist("Problem").set("nx", 16);
  params->sublist("Problem").set("ny", 16);
  params->sublist("Problem").set("Dimension", 2);
  params->sublist("Problem").set("Degrees of Freedom", 3);
  params->sublist("Preconditioner").set("Separator Length", 4);

  HYMLS::CartesianPartitioner part(Teuchos::null, params, *comm);
  part.Partition(true);

  Teuchos::RCP<const Epetra_Map> map = part.GetMap();

  params->sublist("Preconditioner").set("Separator Length", 8);

  // This throws an exception if it fails
  HYMLS::CartesianPartitioner part2(map, params, *comm);
  part2.Partition(true);
  }

#ifdef HYMLS_LONG_LONG
TEUCHOS_UNIT_TEST(CartesianPartitioner, GID64)
  {
  Teuchos::RCP<FakeComm> comm = Teuchos::rcp(new FakeComm);
  comm->SetNumProc(4096);
  comm->SetMyPID(4095);
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(
    new Teuchos::ParameterList);
  params->sublist("Problem").set("nx", 1024);
  params->sublist("Problem").set("ny", 1024);
  params->sublist("Problem").set("nz", 1024);
  params->sublist("Problem").set("Degrees of Freedom", 4);
  params->sublist("Preconditioner").set("Separator Length", 4);

  HYMLS::CartesianPartitioner part(Teuchos::null, params, *comm);
  TEST_MAYTHROW(part.Partition(false));

  Teuchos::Array<hymls_gidx> interior_nodes;
  Teuchos::Array<Teuchos::Array<hymls_gidx> > separator_nodes;
  part.GetGroups(part.NumLocalParts()-1, interior_nodes, separator_nodes);

  ENABLE_OUTPUT;

  long long last = part.Map().NumGlobalElements64() - 1;
  TEST_EQUALITY(interior_nodes.back(), last);
  }
#endif
