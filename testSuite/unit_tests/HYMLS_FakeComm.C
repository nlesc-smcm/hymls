#include "HYMLS_FakeComm.H"

#include <Epetra_Distributor.h>
#include <Epetra_BasicDirectory.h>

class FakeSerialDistributor: public virtual Epetra_Distributor
  {
  int nrecvs_;
  int nsends_;
  int myproc_;

public:
  FakeSerialDistributor(const FakeComm & Comm)
    :
    nrecvs_(0),
    nsends_(0),
    myproc_(Comm.MyPID())
    {}

  FakeSerialDistributor(const FakeSerialDistributor & Plan)
    :
    nrecvs_(Plan.nrecvs_),
    nsends_(Plan.nsends_),
    myproc_(Plan.myproc_)
    {}

  Epetra_Distributor * Clone(){return new FakeSerialDistributor(*this);};
  Epetra_Distributor * ReverseClone() {return 0;}

  virtual ~FakeSerialDistributor() {}

  int CreateFromSends(const int & NumExportIDs, const int * ExportPIDs,
    bool Deterministic, int & NumRemoteIDs)
    {
    NumRemoteIDs = 0;

    //basically just do a sanity check.
    for(int i=0; i<NumExportIDs; ++i) {
      if (ExportPIDs[i] != myproc_) {
        std::cerr << "Epetra_SerialDistributor::CreateFromSends: ExportPIDs["<<i
                  <<"]=="<<ExportPIDs[i]<<", not allowed for serial case."<< std::endl;
        return -1;
        }
      ++NumRemoteIDs;
      }

    nrecvs_ = NumRemoteIDs;
    return 0;
    }

  int CreateFromRecvs(const int & NumRemoteIDs, const int * RemoteGIDs,
    const int * RemotePIDs, bool Deterministic, int & NumExportIDs,
    int *& ExportGIDs, int *& ExportPIDs)
    {
    return -1;
    }

#ifndef EPETRA_NO_64BIT_GLOBAL_INDICES
  int CreateFromRecvs(const int & NumRemoteIDs,
    const long long * RemoteGIDs, const int * RemotePIDs,
    bool Deterministic, int & NumExportIDs,
    long long *& ExportGIDs, int *& ExportPIDs)
    {
    return -1;
    }
#endif
  
  int Do(char * export_objs, int obj_size,
    int & len_import_objs, char *& import_objs)
    {
    len_import_objs = obj_size * nrecvs_;
    if (len_import_objs > 0) {
      import_objs = new char[len_import_objs];
      }

    for(int i = 0; i < len_import_objs; ++i)
     import_objs[i] = export_objs[i];

    return 0;
    }

  int DoReverse(char * export_objs, int obj_size,
    int & len_import_objs, char *& import_objs)
    {
    return -1;
    }

  int DoPosts(char * export_objs, int obj_size,
    int & len_import_objs, char *& import_objs)
    {
    return -1;
    }
  int DoWaits()
    {
    return -1;
    }

  int DoReversePosts(char * export_objs, int obj_size,
    int & len_import_objs, char *& import_objs)
    {
    return -1;
    }
  int DoReverseWaits()
    {
    return -1;
    }

  int Do(char * export_objs, int obj_size,
    int *& sizes,int & len_import_objs, char *& import_objs)
    {
    return -1;
    }
  int DoReverse(char * export_objs, int obj_size, int *& sizes,
    int & len_import_objs, char *& import_objs)
    {
    return -1;
    }

  int DoPosts(char * export_objs, int obj_size, int *& sizes,
    int & len_import_objs, char *& import_objs)
    {
    return -1;
    }

  int DoReversePosts(char * export_objs, int obj_size,
    int *& sizes, int & len_import_objs, char *& import_objs)
    {
    return -1;
    }

  virtual void Print(std::ostream & os) const
    {
    return;
    }
  };

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
  return new FakeSerialDistributor(*this);
  }
Epetra_Directory * FakeComm::CreateDirectory(const Epetra_BlockMap & Map) const
  {
  return new Epetra_BasicDirectory(Map);
  }
void FakeComm::Print(std::ostream & os) const
  {return;}
void FakeComm::PrintInfo(std::ostream & os) const
  {return;}
