#include "HYMLS_SkewCartesianPartitioner.H"
#include "HYMLS_Tools.H"
#include "Teuchos_Utils.hpp"
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_IntVector.h"
#include "Epetra_Import.h"

using Teuchos::toString;

namespace HYMLS {

// by default, everyone is assumed to own a part of the domain.
// if in Partition() it turns out that there are more processor
// partitions than subdomains, active_ is set to false for some
// ranks, which will get an empty part of the reordered map
// (cartesianMap_).

// constructor
SkewCartesianPartitioner::SkewCartesianPartitioner(
  Teuchos::RCP<const Epetra_Map> map, int nx, int ny, int nz, int dof, int pvar,
  GaleriExt::PERIO_Flag perio)
  : BasePartitioner(), label_("SkewCartesianPartitioner"),
    baseMap_(map), cartesianMap_(Teuchos::null),
    nx_(nx), ny_(ny), nz_(nz),
    npx_(-1), npy_(-1), npz_(-1),
    numLocalSubdomains_(-1),
    dof_(dof), pvar_(pvar), active_(true), perio_(perio)
  {
  HYMLS_PROF3(label_,"Constructor");
  comm_ = Teuchos::rcp(&(baseMap_->Comm()), false);

  if (baseMap_->IndexBase()!=0)
    {
    Tools::Warning("Not sure, but I _think_ your map should be 0-based",
      __FILE__, __LINE__);
    }
  }

// destructor
SkewCartesianPartitioner::~SkewCartesianPartitioner()
  {
  HYMLS_PROF3(label_,"Destructor");
  }

// get non-overlapping subdomain id
int SkewCartesianPartitioner::operator()(int i, int j, int k) const
  {
#ifdef HYMLS_TESTING
  if (!Partitioned())
    {
    Tools::Error("Partition() not yet called!", __FILE__, __LINE__);
    }
#endif
  int idx1 = (i + j + 1) / sx_;
  int idx2 = (i - j + ny_) / sx_ - ny_ / sx_;
  int sdz = k / sz_;

  return idx1 * (npx_ + 1) - idx2 * npx_ + sdz * ((npx_ + 1) * npy_);
  }

//! get non-overlapping subdomain id
int SkewCartesianPartitioner::operator()(int gid) const
  {
  int i,j,k,var;
#ifdef HYMLS_TESTING
  if (!Partitioned())
    {
    Tools::Error("Partition() not yet called!", __FILE__, __LINE__);
    }
#endif
  Tools::ind2sub(nx_, ny_, nz_, dof_, gid, i, j, k, var);
  return operator()(i, j, k);
  }

//! return number of subdomains in this proc partition
int SkewCartesianPartitioner::NumLocalParts() const
  {
  if (numLocalSubdomains_<0)
    {
    Tools::Error("not implemented correctly",__FILE__,__LINE__);
    }
  return numLocalSubdomains_;
  }

int SkewCartesianPartitioner::CreateSubdomainMap()
  {
  int NumMyElements = 0;
  int NumGlobalElements = (npx_+1) * npy_ * npz_;
  int *MyGlobalElements = new int[NumGlobalElements];

  for (int k = 0; k < nz_; k++)
    for (int j = 0; j < ny_; j++)
      for (int i = 0; i < nx_; i++)
        {
        int gsd = operator()(i, j, k);
        int pid = PID(i, j, k);
        if (pid == comm_->MyPID() and std::find(MyGlobalElements,
            MyGlobalElements + NumMyElements, gsd) == MyGlobalElements + NumMyElements)
          MyGlobalElements[NumMyElements++] = gsd;
        }

  sdMap_ = Teuchos::rcp(new Epetra_Map(-1, NumMyElements, MyGlobalElements, 0, *comm_));

  delete [] MyGlobalElements;
  return 0;
  }

int SkewCartesianPartitioner::Partition(int nparts, bool repart)
  {
  HYMLS_PROF3(label_,"Partition");
  int npx,npy,npz;
  Tools::SplitBox(nx_,ny_,nz_,nparts,npx,npy,npz);
  return this->Partition(npx,npy,npz,repart);
  }

// partition an [nx x ny x nz] grid with one DoF per node
// into nparts global subdomains.
int SkewCartesianPartitioner::Partition(int sx,int sy, int sz, bool repart)
  {
  HYMLS_PROF3(label_,"Partition (2)");

  if (sx != sy)
    Tools::Error("sx and sy should be the same", __FILE__, __LINE__);

  sx_ = sx * 2;
  sy_ = sy;
  sz_ = sz;

  npx_ = nx_ / sx_;
  npy_ = ny_ / sy_;
  npz_ = nz_ / sz_;

  std::string s1 = toString(nx_) + "x" + toString(ny_) + "x" + toString(nz_);
  std::string s2 = toString((npx_+1) * npy_) + "x" + toString(npz_);

  if (nx_ != npx_ * sx_ || ny_ != npy_ * sy_ || nz_ != npz_ * sz_)
    {
    std::string msg = "You are trying to partition an " + s1 + " domain into " + s2 + " parts.\n"
      "We currently need nx to be a multiple of npx etc.";
    Tools::Error(msg, __FILE__, __LINE__);
    }

  std::string s3 = toString(sx_) + "x" + toString(sy_) + "x" + toString(sz_);

  Tools::Out("Partition domain: ");
  Tools::Out("Grid size: " + s1);
  Tools::Out("Number of Subdomains: " + s2);
  Tools::Out("Subdomain size: " + s3);

  // case where there are more processor partitions than subdomains (experimental)
  if (comm_->MyPID() >= npx_ * npy_ * npz_)
    active_ = false;

  int color = active_? 1: 0;

  CHECK_ZERO(comm_->SumAll(&color, &nprocs_, 1));

  while (nprocs_)
    {
    if ((((npx_ * npy_ * npz_) / nprocs_) * nprocs_) == npx_ * npy_ * npz_)
      {
      break;
      }
    else
      {
      nprocs_--;
      }
    }

  if (comm_->MyPID() >= nprocs_)
    active_ = false;

  // if some processors have no subdomains, we need to
  // repartition the map even if it is a cartesian partitioned
  // map already:
  if (nprocs_ < comm_->NumProc())
    repart = true;

  HYMLS_DEBVAR(npx_);
  HYMLS_DEBVAR(npy_);
  HYMLS_DEBVAR(npz_);
  HYMLS_DEBVAR(active_);

  CHECK_ZERO(CreateSubdomainMap());

  HYMLS_DEBVAR(*sdMap_);

  numLocalSubdomains_ = sdMap_->NumMyElements();

// create redistributed map:
  cartesianMap_ = baseMap_;
  
// repartitioning may occur for two reasons, typically on coarser levels:
// a) the number of subdomains becomes smaller than the number of processes,
// b) the subdomains can't be nicely distributed among the processes.
// In both cases some processes are deactivated.
  if (repart)
    {
    Tools::Out("repartition for "+toString(nprocs_)+" procs");
    HYMLS_PROF3(label_,"repartition map");

    int numMyElements = numLocalSubdomains_ * sx_ * sy_ * sz_ * dof_;
    int *myGlobalElements = new int[numMyElements];
    int pos = 0;
    for (int i = 0; i < baseMap_->MaxAllGID()+1; i++)
      {
      if (sdMap_->LID((*this)(i)) != -1)
        {
        if (pos >= numMyElements)
          {
          Tools::Error("Index out of range", __FILE__, __LINE__);
          }
        myGlobalElements[pos++] = i;
        }
      }

    Epetra_Map tmpRepartitionedMap(-1, pos,
      myGlobalElements, baseMap_->IndexBase(), *comm_);

    Epetra_IntVector vec(*baseMap_);
    vec.PutValue(1);

    Epetra_Import import(tmpRepartitionedMap, *baseMap_);
    Epetra_IntVector repartVec(tmpRepartitionedMap);
    repartVec.Import(vec, import, Insert);

    pos = 0;
    for (int i = 0; i < repartVec.MyLength(); i++)
      {
      if (repartVec[i] == 1)
        {
        if (pos >= numMyElements)
          {
          Tools::Error("Index out of range", __FILE__, __LINE__);
          }
        myGlobalElements[pos++] = tmpRepartitionedMap.GID(i);
        }
      }

    cartesianMap_ = Teuchos::rcp(new Epetra_Map(-1, pos,
        myGlobalElements, baseMap_->IndexBase(), *comm_));

    HYMLS_DEBVAR(*repartitionedMap);
    if (myGlobalElements)
      delete [] myGlobalElements;
    }

  if (active_)
    {
    Tools::Out("Number of Partitions: " + toString(npx_ * npy_ * npz_));
    Tools::Out("Number of Local Subdomains: " + toString(NumLocalParts()));
    }
  return 0;
  }

int SkewCartesianPartitioner::GetGroups(int sd, Teuchos::Array<int> &interior_nodes,
  Teuchos::Array<Teuchos::Array<int> > &separator_nodes)
  {
  HYMLS_PROF2(label_,"GetGroups");
  Tools::Error("Not yet implemented", __FILE__, __LINE__);
  return 0;
  }

//! get the type of a variable (if more than 1 dof per node, otherwise just 0)
int SkewCartesianPartitioner::VariableType(int gid) const
  {
  return (int)(MOD(gid, dof_));
  }

//! get processor on which a grid point is located
int SkewCartesianPartitioner::PID(int i, int j, int k) const
  {
  int gsd = operator()(i, j, k);

  // Remove right boundaries
  int gsd2 = gsd - (gsd % ((npx_+1) * npy_)) / (npx_ * 2 + 1);
  // Remove boundaries of layers above this one
  gsd2 -= gsd / ((npx_+1) * npy_) * (npy_ / 2 + npx_);

  std::cout << "i="+toString(i)+
    ", j="+toString(j)+", k="+toString(k)+", gsd=" + toString(gsd)
    + ", gsd2=" + toString(gsd2) << std::endl;

  // Right boundary
  if ((gsd % ((npx_+1) * npy_)) % (npx_ * 2 + 1) == npx_ * 2)
    {
    if (i == 0)
      Tools::Error("Going into infinite recursion for i="+toString(i)+
        ", j="+toString(j)+", k="+toString(k)+", gsd=" + toString(gsd)+
        ", npx="+toString(npx_),
        __FILE__, __LINE__);
    return (gsd2 - npx_) / ((npx_ * npy_ * npz_) / nprocs_);
    }

  // Bottom boundary
  if (gsd % ((npx_+1) * npy_) >= npx_ * (npy_ + 1))
    {
    if (j == 0)
      Tools::Error("Going into infinite recursion for i="+toString(i)+
        ", j="+toString(j)+", k="+toString(k)+", gsd=" + toString(gsd),
        __FILE__, __LINE__);
    return (gsd2 - npx_ * npy_) / ((npx_ * npy_ * npz_) / nprocs_);
    }

  if (gsd2 >= npx_ * npy_ * npz_)
    Tools::Error("Subdomain index out of range", __FILE__, __LINE__);

  return gsd2 / ((npx_ * npy_ * npz_) / nprocs_);
  }

  }
