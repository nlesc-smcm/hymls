#include "Epetra_Comm.h"

class Epetra_Distributor;
class Epetra_Directory;

class FakeComm: public virtual Epetra_Comm
  {
  int numProc_;
  int pid_;

public:
  FakeComm();

  FakeComm(const FakeComm& Comm);

  FakeComm& operator=(const FakeComm & Comm);

  Epetra_Comm *Clone() const;

  virtual ~FakeComm();

  int SumAll(double *PartialSums, double *GlobalSums, int Count) const;
  int SumAll(int *PartialSums, int *GlobalSums, int Count) const;
  int SumAll(long *PartialSums, long *GlobalSums, int Count) const;
  int SumAll(long long *PartialSums, long long *GlobalSums, int Count) const;

  int ScanSum(double * MyVals, double * ScanSums, int Count) const;
  int ScanSum(int * MyVals, int * ScanSums, int Count) const;
  int ScanSum(long * MyVals, long * ScanSums, int Count) const;
  int ScanSum(long long * MyVals, long long * ScanSums, int Count) const;

  int NumProc() const;
  void SetNumProc(int num);

  int MyPID() const;
  void SetMyPID(int pid);

  void Barrier() const;

  int Broadcast(double * MyVals, int Count, int Root) const;
  int Broadcast(int * MyVals, int Count, int Root) const;
  int Broadcast(long * MyVals, int Count, int Root) const;
  int Broadcast(long long * MyVals, int Count, int Root) const;
  int Broadcast(char * MyVals, int Count, int Root) const;

  int GatherAll(double * MyVals, double * AllVals, int Count) const;
  int GatherAll(int * MyVals, int * AllVals, int Count) const;
  int GatherAll(long * MyVals, long * AllVals, int Count) const;
  int GatherAll(long long * MyVals, long long * AllVals, int Count) const;

  int MaxAll(double * PartialMaxs, double * GlobalMaxs, int Count) const;
  int MaxAll(int * PartialMaxs, int * GlobalMaxs, int Count) const;
  int MaxAll(long * PartialMaxs, long * GlobalMaxs, int Count) const;
  int MaxAll(long long * PartialMaxs, long long * GlobalMaxs, int Count) const;

  int MinAll(double * PartialMins, double * GlobalMins, int Count) const;
  int MinAll(int * PartialMins, int * GlobalMins, int Count) const;
  int MinAll(long * PartialMins, long * GlobalMins, int Count) const;
  int MinAll(long long * PartialMins, long long * GlobalMins, int Count) const;

  Epetra_Distributor * CreateDistributor() const;
  Epetra_Directory * CreateDirectory(const Epetra_BlockMap & Map) const;

  void Print(std::ostream & os) const;
  void PrintInfo(std::ostream & os) const;
  };
