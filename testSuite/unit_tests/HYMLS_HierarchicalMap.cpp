#include "HYMLS_HierarchicalMap.hpp"

#include "HYMLS_Exception.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"

#include "HYMLS_UnitTests.hpp"

class TestableHierarchicalMap : public HYMLS::HierarchicalMap
  {
public:
  TestableHierarchicalMap(Teuchos::RCP<const Epetra_Map> baseMap)
    :
    HierarchicalMap(baseMap)
    {}

  int AddGroup(int sd, Teuchos::Array<hymls_gidx>& gidList)
    {
    return HYMLS::HierarchicalMap::AddGroup(sd, gidList);
    }

  int Reset(int sd)
    {
    return HYMLS::HierarchicalMap::Reset(sd);
    }
  };

TEUCHOS_UNIT_TEST(HierarchicalMap, AddGroup)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  hymls_gidx n = 100;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(n, 0, *Comm));

  Teuchos::Array<hymls_gidx> gidList(n);
  for (int i = 0; i < n; i++)
    {
    gidList[i] = i;
    }

  int ret;

  TestableHierarchicalMap hmap(map);

  // Can't add group to non-existing subdomain
  ret = hmap.AddGroup(0, gidList);
  TEST_EQUALITY(ret, -1);

  // Can add group to an existing subdomain
  ret = hmap.Reset(1);
  TEST_EQUALITY(ret, 0);
  ret = hmap.AddGroup(0, gidList);
  TEST_EQUALITY(ret, 1);
  }

TEUCHOS_UNIT_TEST(HierarchicalMap, NumMySubdomains)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  hymls_gidx n = 100;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(n, 0, *Comm));
  TestableHierarchicalMap hmap(map);

  hmap.Reset(3);
  int ret = hmap.NumMySubdomains();
  TEST_EQUALITY(ret, 3);
  }

TEUCHOS_UNIT_TEST(HierarchicalMap, NumMyElements)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  hymls_gidx n = 100;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(n, 0, *Comm));

  Teuchos::Array<hymls_gidx> gidList(n);
  for (int i = 0; i < n; i++)
    {
    gidList[i] = i;
    }

  TestableHierarchicalMap hmap(map);

  hmap.Reset(2);
  hmap.AddGroup(0, gidList);
  hmap.AddGroup(1, gidList);
  hmap.AddGroup(1, gidList);
  int ret = hmap.NumMyElements();
  TEST_EQUALITY(ret, n*3);
  }

TEUCHOS_UNIT_TEST(HierarchicalMap, NumMyInteriorElements)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  hymls_gidx n = 100;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(n, 0, *Comm));

  Teuchos::Array<hymls_gidx> gidList(n);
  for (int i = 0; i < n; i++)
    {
    gidList[i] = i;
    }

  TestableHierarchicalMap hmap(map);

  hmap.Reset(2);
  hmap.AddGroup(0, gidList);
  hmap.AddGroup(1, gidList);
  hmap.AddGroup(1, gidList);
  int ret = hmap.NumMyInteriorElements();
  TEST_EQUALITY(ret, n*2);
  }

TEUCHOS_UNIT_TEST(HierarchicalMap, NumElements)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  hymls_gidx n = 100;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(n, 0, *Comm));

  Teuchos::Array<hymls_gidx> gidList(n);
  for (int i = 0; i < n; i++)
    {
    gidList[i] = i;
    }

  TestableHierarchicalMap hmap(map);

  hmap.Reset(2);
  hmap.AddGroup(0, gidList);
  hmap.AddGroup(1, gidList);
  hmap.AddGroup(1, gidList);
  int ret = hmap.NumElements(1);
  TEST_EQUALITY(ret, n*2);
  }

TEUCHOS_UNIT_TEST(HierarchicalMap, NumInteriorElements)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  hymls_gidx n = 100;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(n, 0, *Comm));

  Teuchos::Array<hymls_gidx> gidList(n);
  for (int i = 0; i < n; i++)
    {
    gidList[i] = i;
    }

  TestableHierarchicalMap hmap(map);

  hmap.Reset(2);
  hmap.AddGroup(0, gidList);
  hmap.AddGroup(1, gidList);
  hmap.AddGroup(1, gidList);
  int ret = hmap.NumInteriorElements(1);
  TEST_EQUALITY(ret, n);
  }

TEUCHOS_UNIT_TEST(HierarchicalMap, NumSeparatorElements)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  hymls_gidx n = 100;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(n, 0, *Comm));

  Teuchos::Array<hymls_gidx> gidList(n);
  for (int i = 0; i < n; i++)
    {
    gidList[i] = i;
    }

  TestableHierarchicalMap hmap(map);

  hmap.Reset(2);
  hmap.AddGroup(0, gidList);
  hmap.AddGroup(1, gidList);
  hmap.AddGroup(1, gidList);
  int ret = hmap.NumSeparatorElements(1);
  TEST_EQUALITY(ret, n);
  ret = hmap.NumSeparatorElements(0);
  TEST_EQUALITY(ret, 0);
  }

TEUCHOS_UNIT_TEST(HierarchicalMap, NumGroups)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  hymls_gidx n = 100;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(n, 0, *Comm));

  Teuchos::Array<hymls_gidx> gidList(n);
  for (int i = 0; i < n; i++)
    {
    gidList[i] = i;
    }

  TestableHierarchicalMap hmap(map);

  hmap.Reset(2);
  hmap.AddGroup(0, gidList);
  hmap.AddGroup(1, gidList);
  hmap.AddGroup(1, gidList);
  int ret = hmap.NumGroups(1);
  TEST_EQUALITY(ret, 2);
  ret = hmap.NumGroups(0);
  TEST_EQUALITY(ret, 1);
  }

TEUCHOS_UNIT_TEST(HierarchicalMap, NumSeparatorGroups)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  hymls_gidx n = 100;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(n, 0, *Comm));

  Teuchos::Array<hymls_gidx> gidList(n);
  for (int i = 0; i < n; i++)
    {
    gidList[i] = i;
    }

  TestableHierarchicalMap hmap(map);

  hmap.Reset(2);
  hmap.AddGroup(0, gidList);
  hmap.AddGroup(1, gidList);
  hmap.AddGroup(1, gidList);
  int ret = hmap.NumSeparatorGroups(1);
  TEST_EQUALITY(ret, 1);
  ret = hmap.NumSeparatorGroups(0);
  TEST_EQUALITY(ret, 0);
  }

TEUCHOS_UNIT_TEST(HierarchicalMap, NumElements2)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  hymls_gidx n = 100;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(n, 0, *Comm));

  Teuchos::Array<hymls_gidx> gidList(n);
  for (int i = 0; i < n; i++)
    {
    gidList[i] = i;
    }

  TestableHierarchicalMap hmap(map);

  hmap.Reset(2);
  hmap.AddGroup(0, gidList);
  hmap.AddGroup(1, gidList);
  hmap.AddGroup(1, gidList);
  int ret = hmap.NumElements(1, 1);
  TEST_EQUALITY(ret, n);
  ret = hmap.NumElements(1,0);
  TEST_EQUALITY(ret, n);
  }

// TEUCHOS_UNIT_TEST(HierarchicalMap, LID)
//   {
//   Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

//   int n = 100;
//   Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(n, 0, *Comm));

//   Teuchos::Array<hymls_gidx> gidList(n);
//   for (int i = 0; i < n; i++)
//     {
//     gidList[i] = i;
//     }

//   TestableHierarchicalMap hmap(map);

//   hmap.Reset(2);
//   hmap.AddGroup(0, gidList);
//   hmap.AddGroup(1, gidList);
//   hmap.AddGroup(1, gidList);
//   int ret = hmap.LID(1, 1, 50);
//   TEST_EQUALITY(ret, 150);
//   }
