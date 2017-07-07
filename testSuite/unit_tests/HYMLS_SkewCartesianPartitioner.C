#include "HYMLS_SkewCartesianPartitioner.H"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include "Epetra_SerialComm.h"
#include "Epetra_Map.h"

#include "HYMLS_UnitTests.H"

// A class for which we can set the number of processors that are available.
class FakeComm: public Epetra_SerialComm
  {
  int numProc_;
public:
  FakeComm(): Epetra_SerialComm(), numProc_() {}
  FakeComm(const FakeComm& Comm): Epetra_SerialComm(Comm), numProc_(Comm.numProc_) {}
  Epetra_Comm *Clone() const
    {return dynamic_cast<Epetra_Comm *>(new FakeComm(*this));}
  int NumProc() const
    {return numProc_;}
  int SumAll(int *PartialSums, int *GlobalSums, int Count) const
    {*GlobalSums = *PartialSums * numProc_; return 0;};
  void SetNumProc(int num)
    {numProc_ = num;}
  };

class TestableSkewCartesianPartitioner: public HYMLS::SkewCartesianPartitioner
  {
public:
  TestableSkewCartesianPartitioner(Teuchos::RCP<const Epetra_Map> map, int nx, int ny,
    int nz=1, int dof=1, int pvar=-1, GaleriExt::PERIO_Flag perio=GaleriExt::NO_PERIO)
    :
    HYMLS::SkewCartesianPartitioner(map, nx, ny, nz, dof, pvar, perio)
    {}

  int PID(int i, int j, int k)
    {
    return HYMLS::SkewCartesianPartitioner::PID(i, j, k);
    }
  };

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, operator)
  {
  FakeComm comm;
  comm.SetNumProc(4);

  int nx = 8;
  int ny = 8;
  int nz = 8;
  int sx = 4;
  int sy = 4;
  int sz = 4;
  int dof = 3;
  int n = nx * ny * nz * dof;

  Epetra_Map map(n, 0, comm);
  TestableSkewCartesianPartitioner part(Teuchos::rcp(&map, false), nx, ny, nz, dof);
  part.Partition(sx, sy, sz, false);

  TEST_EQUALITY(part(0, 0, 0), 0);
  TEST_EQUALITY(part(0, 1, 0), 2);
  TEST_EQUALITY(part(7, 0, 0), 4);
  TEST_EQUALITY(part(3, 4, 0), 8);
  TEST_EQUALITY(part(3, 4, 3), 20);
  TEST_EQUALITY(part(3, 4, 4), 20);
  TEST_EQUALITY(part(0, 0, 4), 12);
  }

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, PID)
  {
  FakeComm comm;
  comm.SetNumProc(4);

  int nx = 8;
  int ny = 8;
  int nz = 8;
  int sx = 4;
  int sy = 4;
  int sz = 4;
  int dof = 3;
  int n = nx * ny * nz * dof;

  Epetra_Map map(n, 0, comm);
  TestableSkewCartesianPartitioner part(Teuchos::rcp(&map, false), nx, ny, nz, dof);
  part.Partition(sx, sy, sz, false);

  TEST_EQUALITY(part.PID(0, 0, 0), 3);
  TEST_EQUALITY(part.PID(0, 1, 0), 3);
  TEST_EQUALITY(part.PID(7, 0, 0), 2);
  TEST_EQUALITY(part.PID(3, 4, 0), 2);
  TEST_EQUALITY(part.PID(3, 4, 3), 0);
  TEST_EQUALITY(part.PID(3, 4, 4), 0);
  TEST_EQUALITY(part.PID(0, 0, 4), 1);
  TEST_EQUALITY(part.PID(7, 7, 7), 1);
  }

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, 2DNodes)
  {
  FakeComm comm;
  comm.SetNumProc(1);

  int nx = 8;
  int ny = 8;
  int nz = 1;
  int sx = 4;
  int sy = 4;
  int sz = 1;
  int dof = 3;
  int n = nx * ny * nz * dof;

  Epetra_Map map(n, 0, comm);
  TestableSkewCartesianPartitioner part(Teuchos::rcp(&map, false), nx, ny, nz, dof, 2);
  part.Partition(sx, sy, sz, false);

  std::vector<int> gids(n, 0);
  for (int sd = 0; sd < part.NumLocalParts(); sd++)
    {
    Teuchos::Array<int> interior_nodes;
    Teuchos::Array<Teuchos::Array<int> > separator_nodes;
    part.GetGroups(sd, interior_nodes, separator_nodes);

    for (int &i: interior_nodes)
      gids[i] = i;

    for (auto &group: separator_nodes)
      for (int &i: group)
        gids[i] = i;
    }
  
  for (int i = 0; i < n; i++)
    TEST_EQUALITY(gids[i], i);
  }

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, 1PSepPerDomain2D)
  {
  FakeComm comm;
  comm.SetNumProc(1);

  int nx = 8;
  int ny = 8;
  int nz = 1;
  int sx = 4;
  int sy = 4;
  int sz = 1;
  int dof = 3;
  int n = nx * ny * nz * dof;

  Epetra_Map map(n, 0, comm);
  TestableSkewCartesianPartitioner part(Teuchos::rcp(&map, false), nx, ny, nz, dof, 2);
  part.Partition(sx, sy, sz, false);

  std::vector<int> gids(n, 0);
  for (int sd = 0; sd < part.NumLocalParts(); sd++)
    {
    int numPNodes = 0;
    Teuchos::Array<int> interior_nodes;
    Teuchos::Array<Teuchos::Array<int> > separator_nodes;
    part.GetGroups(sd, interior_nodes, separator_nodes);

    for (auto &group: separator_nodes)
      for (int &i: group)
        if (i % 3 == 2)
          numPNodes++;
    TEST_EQUALITY(numPNodes, 1);
    }
  }

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, 3DNodes)
  {
  FakeComm comm;
  comm.SetNumProc(1);

  int nx = 8;
  int ny = 8;
  int nz = 8;
  int sx = 4;
  int sy = 4;
  int sz = 4;
  int dof = 4;
  int n = nx * ny * nz * dof;

  Epetra_Map map(n, 0, comm);
  TestableSkewCartesianPartitioner part(Teuchos::rcp(&map, false), nx, ny, nz, dof, 3);
  part.Partition(sx, sy, sz, false);

  std::vector<int> gids(n, 0);
  for (int sd = 0; sd < part.NumLocalParts(); sd++)
    {
    Teuchos::Array<int> interior_nodes;
    Teuchos::Array<Teuchos::Array<int> > separator_nodes;
    part.GetGroups(sd, interior_nodes, separator_nodes);

    for (int &i: interior_nodes)
      gids[i] = i;

    for (auto &group: separator_nodes)
      for (int &i: group)
        gids[i] = i;
    }
  
  for (int i = 0; i < n; i++)
    TEST_EQUALITY(gids[i], i);
  }

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, 1PSepPerDomain3D)
  {
  FakeComm comm;
  comm.SetNumProc(1);

  int nx = 8;
  int ny = 8;
  int nz = 8;
  int sx = 4;
  int sy = 4;
  int sz = 4;
  int dof = 4;
  int n = nx * ny * nz * dof;

  Epetra_Map map(n, 0, comm);
  TestableSkewCartesianPartitioner part(Teuchos::rcp(&map, false), nx, ny, nz, dof, 3);
  part.Partition(sx, sy, sz, false);

  std::vector<int> gids(n, 0);
  for (int sd = 0; sd < part.NumLocalParts(); sd++)
    {
    int numPNodes = 0;
    Teuchos::Array<int> interior_nodes;
    Teuchos::Array<Teuchos::Array<int> > separator_nodes;
    part.GetGroups(sd, interior_nodes, separator_nodes);

    for (auto &group: separator_nodes)
      for (int &i: group)
        if (i % 4 == 3)
          numPNodes++;
    TEST_EQUALITY(numPNodes, 1);
    }
  }
