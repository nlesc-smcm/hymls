#include "HYMLS_FakeComm.H"

#include <Epetra_BasicDirectory.h>
#include <Epetra_SerialDistributor.h>
#include <Epetra_SerialComm.h>

FakeComm::FakeComm():
  numProc_(1),
  pid_(0)
  {}

FakeComm::FakeComm(const FakeComm& Comm):
  numProc_(Comm.numProc_),
  pid_(Comm.pid_)
  {}

FakeComm& FakeComm::operator=(const FakeComm & Comm) {
  numProc_ = Comm.numProc_;
  pid_ = Comm.pid_;
  return *this;
  }

Epetra_Comm *FakeComm::Clone() const {return new FakeComm(*this);}

FakeComm::~FakeComm(){};

int FakeComm::SumAll(double *PartialSums, double *GlobalSums, int Count) const
  {
  for (int i = 0; i < Count; i++)
    GlobalSums[i] = PartialSums[i] * numProc_;
  return 0;
  }
int FakeComm::SumAll(int *PartialSums, int *GlobalSums, int Count) const
  {
  for (int i = 0; i < Count; i++)
    GlobalSums[i] = PartialSums[i] * numProc_;
  return 0;
  }
int FakeComm::SumAll(long *PartialSums, long *GlobalSums, int Count) const
  {
  for (int i = 0; i < Count; i++)
    GlobalSums[i] = PartialSums[i] * numProc_;
  return 0;
  }
int FakeComm::SumAll(long long *PartialSums, long long *GlobalSums, int Count) const
  {
  for (int i = 0; i < Count; i++)
    GlobalSums[i] = PartialSums[i] * numProc_;
  return 0;
  }

int FakeComm::ScanSum(double * MyVals, double * ScanSums, int Count) const
  {
  for (int i = 0; i < Count; i++)
    ScanSums[i] = MyVals[i] * (pid_ + 1);
  return 0;
  }
int FakeComm::ScanSum(int * MyVals, int * ScanSums, int Count) const
  {
  for (int i = 0; i < Count; i++)
    ScanSums[i] = MyVals[i] * (pid_ + 1);
  return 0;
  }
int FakeComm::ScanSum(long * MyVals, long * ScanSums, int Count) const
  {
  for (int i = 0; i < Count; i++)
    ScanSums[i] = MyVals[i] * (pid_ + 1);
  return 0;
  }
int FakeComm::ScanSum(long long * MyVals, long long * ScanSums, int Count) const
  {
  for (int i = 0; i < Count; i++)
    ScanSums[i] = MyVals[i] * (pid_ + 1);
  return 0;
  }

int FakeComm::NumProc() const
  {
  return numProc_;
  }

void FakeComm::SetNumProc(int num)
  {
  numProc_ = num;
  }

int FakeComm::MyPID() const
  {
  return pid_;
  }

void FakeComm::SetMyPID(int pid)
  {
  pid_ = pid;
  }

void FakeComm::Barrier() const
  {}

int FakeComm::Broadcast(double * MyVals, int Count, int Root) const
  {
  return 0;
  }
int FakeComm::Broadcast(int * MyVals, int Count, int Root) const
  {
  return 0;
  }
int FakeComm::Broadcast(long * MyVals, int Count, int Root) const
  {
  return 0;
  }
int FakeComm::Broadcast(long long * MyVals, int Count, int Root) const
  {
  return 0;
  }
int FakeComm::Broadcast(char * MyVals, int Count, int Root) const
  {
  return 0;
  }
int FakeComm::GatherAll(double * MyVals, double * AllVals, int Count) const
  {
  for (int i = 0; i < Count; i++)
    AllVals[i] = MyVals[i];
  return 0;
  }
int FakeComm::GatherAll(int * MyVals, int * AllVals, int Count) const
  {
  for (int i = 0; i < Count; i++)
    AllVals[i] = MyVals[i];
  return 0;
  }
int FakeComm::GatherAll(long * MyVals, long * AllVals, int Count) const
  {
  for (int i = 0; i < Count; i++)
    AllVals[i] = MyVals[i];
  return 0;
  }
int FakeComm::GatherAll(long long * MyVals, long long * AllVals, int Count) const
  {
  for (int i = 0; i < Count; i++)
    AllVals[i] = MyVals[i];
  return 0;
  }
int FakeComm::MaxAll(double * PartialMaxs, double * GlobalMaxs, int Count) const
  {
  for (int i = 0; i < Count; i++)
    GlobalMaxs[i] = PartialMaxs[i];
  return 0;
  }
int FakeComm::MaxAll(int * PartialMaxs, int * GlobalMaxs, int Count) const
  {
  for (int i = 0; i < Count; i++)
    GlobalMaxs[i] = PartialMaxs[i];
  return 0;
  }
int FakeComm::MaxAll(long * PartialMaxs, long * GlobalMaxs, int Count) const
  {
  for (int i = 0; i < Count; i++)
    GlobalMaxs[i] = PartialMaxs[i];
  return 0;
  }
int FakeComm::MaxAll(long long * PartialMaxs, long long * GlobalMaxs, int Count) const
  {
  for (int i = 0; i < Count; i++)
    GlobalMaxs[i] = PartialMaxs[i];
  return 0;
  }
int FakeComm::MinAll(double * PartialMins, double * GlobalMins, int Count) const
  {
  for (int i = 0; i < Count; i++)
    GlobalMins[i] = PartialMins[i];
  return 0;
  }
int FakeComm::MinAll(int * PartialMins, int * GlobalMins, int Count) const
  {
  for (int i = 0; i < Count; i++)
    GlobalMins[i] = PartialMins[i];
  return 0;
  }
int FakeComm::MinAll(long * PartialMins, long * GlobalMins, int Count) const
  {
  for (int i = 0; i < Count; i++)
    GlobalMins[i] = PartialMins[i];
  return 0;
  }
int FakeComm::MinAll(long long * PartialMins, long long * GlobalMins, int Count) const
  {
  for (int i = 0; i < Count; i++)
    GlobalMins[i] = PartialMins[i];
  return 0;
  }
Epetra_Distributor * FakeComm::CreateDistributor() const
  {
  Epetra_SerialComm comm;
  return new Epetra_SerialDistributor(comm);
  }
Epetra_Directory * FakeComm::CreateDirectory(const Epetra_BlockMap & Map) const
  {
  return new Epetra_BasicDirectory(Map);
  }
void FakeComm::Print(std::ostream & os) const
  {return;}
void FakeComm::PrintInfo(std::ostream & os) const
  {return;}
