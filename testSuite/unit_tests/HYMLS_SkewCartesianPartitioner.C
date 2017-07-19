#include "HYMLS_SkewCartesianPartitioner.H"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include "Epetra_SerialComm.h"
#include "Epetra_Map.h"

#include "HYMLS_UnitTests.H"
#include "HYMLS_Tools.H"

// A class for which we can set the number of processors that are available.
class FakeComm: public Epetra_SerialComm
  {
  int numProc_;
  int pid_;
public:
  FakeComm(): Epetra_SerialComm(), numProc_(0), pid_(0) {}
  FakeComm(const FakeComm& Comm):
    Epetra_SerialComm(Comm),
    numProc_(Comm.numProc_),
    pid_(Comm.pid_)
    {}
  Epetra_Comm *Clone() const
    {return dynamic_cast<Epetra_Comm *>(new FakeComm(*this));}

  int NumProc() const
    {return numProc_;}
  void SetNumProc(int num)
    {numProc_ = num;}

  int SumAll(int *PartialSums, int *GlobalSums, int Count) const
    {*GlobalSums = *PartialSums * numProc_; return 0;};

  int MyPID() const
    {return pid_;}
  void SetMyPID(int pid)
    {pid_ = pid;}
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

  Teuchos::RCP<std::ostream> no_output
    = Teuchos::rcp(new Teuchos::oblackholestream());
  HYMLS::Tools::InitializeIO_std(Teuchos::rcp(&comm, false), no_output, no_output);

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

  HYMLS::Tools::InitializeIO(Teuchos::rcp(&comm, false));

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

  Teuchos::RCP<std::ostream> no_output
    = Teuchos::rcp(new Teuchos::oblackholestream());
  HYMLS::Tools::InitializeIO_std(Teuchos::rcp(&comm, false), no_output, no_output);

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

  HYMLS::Tools::InitializeIO(Teuchos::rcp(&comm, false));

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
  FakeComm comm;
  comm.SetNumProc(1);

  Teuchos::RCP<std::ostream> no_output
    = Teuchos::rcp(new Teuchos::oblackholestream());
  HYMLS::Tools::InitializeIO_std(Teuchos::rcp(&comm, false), no_output, no_output);

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

  HYMLS::Tools::InitializeIO(Teuchos::rcp(&comm, false));

  for (int i = 0; i < n; i++)
    TEST_EQUALITY(gids[i], i);
  }

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, 1PSepPerDomain2D)
  {
  FakeComm comm;
  comm.SetNumProc(1);

  Teuchos::RCP<std::ostream> no_output
    = Teuchos::rcp(new Teuchos::oblackholestream());
  HYMLS::Tools::InitializeIO_std(Teuchos::rcp(&comm, false), no_output, no_output);

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

  HYMLS::Tools::InitializeIO(Teuchos::rcp(&comm, false));

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

  Teuchos::RCP<std::ostream> no_output
    = Teuchos::rcp(new Teuchos::oblackholestream());
  HYMLS::Tools::InitializeIO_std(Teuchos::rcp(&comm, false), no_output, no_output);

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

  HYMLS::Tools::InitializeIO(Teuchos::rcp(&comm, false));

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

  Teuchos::RCP<std::ostream> no_output
    = Teuchos::rcp(new Teuchos::oblackholestream());
  HYMLS::Tools::InitializeIO_std(Teuchos::rcp(&comm, false), no_output, no_output);

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

  HYMLS::Tools::InitializeIO(Teuchos::rcp(&comm, false));

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

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, NoEmptyProcs16)
  {
  int nprocs = 16;
  FakeComm comm;

  Teuchos::RCP<std::ostream> no_output
    = Teuchos::rcp(new Teuchos::oblackholestream());
  HYMLS::Tools::InitializeIO_std(Teuchos::rcp(&comm, false), no_output, no_output);

  comm.SetNumProc(nprocs);
  for (int i = 0; i < nprocs; i++)
    {
    comm.SetMyPID(i);

    int nx = 16;
    int ny = 16;
    int nz = 16;
    int sx = 4;
    int sy = 4;
    int sz = 4;
    int dof = 4;
    int n = nx * ny * nz * dof;

    Epetra_Map map(n, 0, comm);
    TestableSkewCartesianPartitioner part(Teuchos::rcp(&map, false), nx, ny, nz, dof, 3);
    part.Partition(sx, sy, sz, false);

    TEST_INEQUALITY(part.NumLocalParts(), 0);
    }

  HYMLS::Tools::InitializeIO(Teuchos::rcp(&comm, false));
  }

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, NoEmptyProcs128)
  {
  int nprocs = 128;
  FakeComm comm;

  Teuchos::RCP<std::ostream> no_output
    = Teuchos::rcp(new Teuchos::oblackholestream());
  HYMLS::Tools::InitializeIO_std(Teuchos::rcp(&comm, false), no_output, no_output);

  comm.SetNumProc(nprocs);
  for (int i = 0; i < nprocs; i++)
    {
    comm.SetMyPID(i);

    int nx = 32;
    int ny = 32;
    int nz = 32;
    int sx = 4;
    int sy = 4;
    int sz = 4;
    int dof = 4;
    int n = nx * ny * nz * dof;

    Epetra_Map map(n, 0, comm);
    TestableSkewCartesianPartitioner part(Teuchos::rcp(&map, false), nx, ny, nz, dof, 3);
    part.Partition(sx, sy, sz, false);

    TEST_INEQUALITY(part.NumLocalParts(), 0);
    }

  HYMLS::Tools::InitializeIO(Teuchos::rcp(&comm, false));
  }

TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, SameNumSubdomains)
  {
  FakeComm comm;

  Teuchos::RCP<std::ostream> no_output
    = Teuchos::rcp(new Teuchos::oblackholestream());
  HYMLS::Tools::InitializeIO_std(Teuchos::rcp(&comm, false), no_output, no_output);

  int nprocs = 16;
  comm.SetNumProc(nprocs);

  int nx = 16;
  int ny = 16;
  int nz = 16;
  int sx = 4;
  int sy = 4;
  int sz = 4;
  int dof = 4;
  int n = nx * ny * nz * dof;

  Epetra_Map map(n, 0, comm);
  TestableSkewCartesianPartitioner part(Teuchos::rcp(&map, false), nx, ny, nz, dof, 3);
  part.Partition(sx, sy, sz, false);

  int num = part.NumLocalParts();

  nprocs = 128;
  comm.SetNumProc(nprocs);

  nx = 32;
  ny = 32;
  nz = 32;
  sx = 4;
  sy = 4;
  sz = 4;
  dof = 4;
  n = nx * ny * nz * dof;

  Epetra_Map map2(n, 0, comm);
  TestableSkewCartesianPartitioner part2(Teuchos::rcp(&map2, false), nx, ny, nz, dof, 3);
  part2.Partition(sx, sy, sz, false);

  HYMLS::Tools::InitializeIO(Teuchos::rcp(&comm, false));

  TEST_EQUALITY(part.NumLocalParts(), num);
  }
