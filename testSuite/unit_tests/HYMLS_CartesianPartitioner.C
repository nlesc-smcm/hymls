#include "HYMLS_CartesianPartitioner.H"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"

#include "HYMLS_config.h"
#include "HYMLS_UnitTests.H"
#include "HYMLS_FakeComm.H"
#include "HYMLS_Tools.H"

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

  HYMLS::CartesianPartitioner part(Teuchos::null, params, comm);

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

  HYMLS::CartesianPartitioner part(Teuchos::null, params, comm);

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

  HYMLS::CartesianPartitioner part(Teuchos::null, params, comm);

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

#ifdef HYMLS_LONG_LONG
TEUCHOS_UNIT_TEST(CartesianPartitioner, GID64)
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
  params->sublist("Preconditioner").set("Separator Length", 4);

  HYMLS::CartesianPartitioner part(Teuchos::null, params, comm);
  part.Partition(false);

  Teuchos::Array<hymls_gidx> interior_nodes;
  Teuchos::Array<Teuchos::Array<hymls_gidx> > separator_nodes;
  part.GetGroups(part.NumLocalParts()-1, interior_nodes, separator_nodes);

  long long last = part.Map().NumGlobalElements64() - 1;
  TEST_EQUALITY(interior_nodes.back(), last);
  }
#endif
