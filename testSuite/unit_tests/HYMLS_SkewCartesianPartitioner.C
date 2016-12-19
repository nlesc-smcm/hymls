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
  TEST_EQUALITY(part(0, 1, 0), 1);
  TEST_EQUALITY(part(7, 0, 0), 2);
  TEST_EQUALITY(part(3, 4, 0), 3);
  TEST_EQUALITY(part(3, 4, 3), 3);
  TEST_EQUALITY(part(3, 4, 4), 7);
  TEST_EQUALITY(part(0, 0, 4), 4);
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

  TEST_EQUALITY(part.PID(0, 0, 0), 0);
  TEST_EQUALITY(part.PID(0, 1, 0), 1);
  TEST_EQUALITY(part.PID(7, 0, 0), 1);
  TEST_EQUALITY(part.PID(3, 4, 0), 0);
  TEST_EQUALITY(part.PID(3, 4, 3), 0);
  TEST_EQUALITY(part.PID(3, 4, 4), 2);
  TEST_EQUALITY(part.PID(0, 0, 4), 2);
  }


TEUCHOS_UNIT_TEST(SkewCartesianPartitioner, First)
  {
  FakeComm comm;
  comm.SetNumProc(1);

  int nx = 8;
  int ny = 8;
  int nz = 8;
  int sx = 4;
  int sy = 4;
  int sz = 4;
  int dof = 1;
  int n = nx * ny * nz * dof;

  Epetra_Map map(n, 0, comm);
  TestableSkewCartesianPartitioner part(Teuchos::rcp(&map, false), nx, ny, nz, dof);
  part.Partition(sx, sy, sz, false);

  TEST_EQUALITY(part.First(0), (-nx * sx + sx - 1) * dof);
  TEST_EQUALITY(part.First(1), -1 * dof);
  TEST_EQUALITY(part.First(2), (nx-1) * dof);
  TEST_EQUALITY(part.First(3), (nx * sx + sx - 1) * dof);
  }
