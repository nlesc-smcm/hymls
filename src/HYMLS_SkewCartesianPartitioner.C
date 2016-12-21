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
    npx_(-1), npy_(-1), npz_(-1), npl_(-1),
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

  return idx1 * (npx_ / 2 + 1) - idx2 * (npx_ / 2) + sdz * npl_;
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
  int NumGlobalElements = npl_ * npz_;
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

  std::sort(MyGlobalElements, MyGlobalElements + NumMyElements);
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

  npx_ = nx_ / sx_ * 2 + 1;
  npy_ = ny_ / sy_ / 2;
  npl_ = npx_ * npy_ + npx_ / 2;
  npz_ = nz_ / sz_;

  std::string s1 = toString(nx_) + "x" + toString(ny_) + "x" + toString(nz_);
  std::string s2 = toString(npl_) + "x" + toString(npz_);

  if (nx_ * ny_ != npx_ / 2 * 2 * npy_ * sy_ * sy * 2 || nz_ != npz_ * sz_)
    {
    std::string msg = "You are trying to partition an " + s1 + " domain into " + s2 + " parts.\n";
    Tools::Error(msg, __FILE__, __LINE__);
    }

  std::string s3 = toString(sy_*sy_*2) + "x" + toString(sz_);

  Tools::Out("Partition domain: ");
  Tools::Out("Grid size: " + s1);
  Tools::Out("Number of Subdomains: " + s2);
  Tools::Out("Subdomain size: " + s3);

  // case where there are more processor partitions than subdomains (experimental)
  if (comm_->MyPID() >= npl_ * npz_)
    active_ = false;

  int color = active_? 1: 0;

  CHECK_ZERO(comm_->SumAll(&color, &nprocs_, 1));

  while (nprocs_)
    {
    if ((((npl_ * npz_) / nprocs_) * nprocs_) == npl_ * npz_)
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
    Tools::Out("Number of Partitions: " + toString(npl_ * npz_));
    Tools::Out("Number of Local Subdomains: " + toString(NumLocalParts()));
    }
  return 0;
  }

int SkewCartesianPartitioner::First(int sd) const
  {
  int gsd = sdMap_->GID(sd);
  int xpos = (gsd % npl_) % npx_;
  int ypos = (gsd % npl_) / npx_;
  return -dof_ +
    ((xpos + 1) % (npx_ / 2 + 1)) * sx_ * dof_ + // shift to the right
    -(1 - ((xpos + 1) / (npx_ / 2 + 1))) * sy_ * dof_ + // shift first row to the left
    ypos * nx_ * sx_ * dof_ + // shift down
    -(1 - ((xpos + 1) / (npx_ / 2 + 1))) * nx_ * sy_ * dof_ + // shift first row sy up
    ((gsd / npl_) % npz_) * nx_ * ny_ * sz_ * dof_; // z-direction
  }

int SkewCartesianPartitioner::GetGroups(int sd, Teuchos::Array<int> &interior_nodes,
  Teuchos::Array<Teuchos::Array<int> > &separator_nodes)
  {
  HYMLS_PROF2(label_,"GetGroups");

  // pressure nodes that need to be retained
  Teuchos::Array<int> retained_nodes;

  Teuchos::Array<int> *nodes;

  int gsd = sdMap_->GID(sd);
  int first = First(sd);
  for (int ktype = (nz_ > 1 ? -1 : 0); ktype < (nz_ > 1 ? 2 : 1); ktype++)
    {
    if (ktype == 1)
      ktype = sz_ - 1;
    if (ktype == -1 && (first / dof_ / nx_ / ny_) % nz_ == 0)
      continue;

    int gsdLayer = gsd - (ktype == -1 ? npl_ : 0);

    for (int jtype = -1; jtype < 2; jtype++)
      {
      if (jtype == 1)
        jtype = sy_ - 1;

      for (int itype = -1; itype < 2; itype++)
        {
        if (itype == 1)
          itype = sy_ - 1;

        for (int d = 0; d < dof_; d++)
          {
          if (d == pvar_ && (itype == -1 || jtype == -1 || ktype == -1))
            continue;
          else if ((itype == 0 && jtype == 0 && ktype == 0) || d == pvar_)
            nodes = &interior_nodes;
          else
            {
            separator_nodes.append(Teuchos::Array<int>());
            nodes = &separator_nodes.back();
            }
          if (d != 0 || pvar_ < 0)
            for (int shift = !(itype == 0 && jtype == 0); shift < 2; shift++)
              {
              int jend = jtype + 1;
              if (jtype == 0 and shift == 0)
                jend = sy_;
              else if (jtype == 0)
                jend = sy_ - 1;

              int iend = itype + 1;
              if (itype == 0 and shift == 0)
                iend = sy_;
              else if (itype == 0)
                iend = sy_ - 1;

              for (int k = ktype; k < ((ktype || nz_ <= 1) ? ktype+1 : sz_-1); k++)
                {
                for (int j = jtype; j < jend; j++)
                  {
                  for (int i = itype; i < iend; i++)
                    {
                    int gid = first - i * dof_ + i * nx_ * dof_ +
                      j * dof_ + j * nx_ * dof_ +
                      k * nx_ * ny_ * dof_ + d + shift * nx_ * dof_;

                    int gsdNode = operator()(first - i * dof_ + i * nx_ * dof_ +
                      j * dof_ + j * nx_ * dof_ -
                      (ktype < 0) * sz_ * nx_ * ny_ * dof_ + d + shift * nx_ * dof_);
                    // Check if this is actually in this domain or a neighbouring domain
                    if (gsdNode == gsdLayer ||
                        (itype == -1 and gsdNode % npl_ == gsdLayer % npl_ - npx_ / 2 and
                        (gsdLayer % npl_) % npx_ != npx_ / 2 * 2) ||
                        (jtype == -1 and gsdNode % npl_ == gsdLayer % npl_ - npx_ / 2 - 1 and
                        (gsdLayer % npl_) % npx_ != npx_ / 2) ||
                      (itype == -1 and jtype == -1 and gsdNode % npl_ == gsdLayer % npl_ - npx_))
                      {
                      if (d == pvar_ && (!i || itype > 0) && !retained_nodes.size())
                        {
                        // Retained pressure nodes
                        retained_nodes.append(gid);
                        }
                      else
                        {
                        // Normal nodes in interiors and on separators
                        nodes->append(gid);
                        }
                      }
                    }
                  }
                }
              }
          else
            for (int shift = 0; shift < 2; shift++)
              {
              int jend = jtype + 1;
              if (jtype == 0 and shift == 0)
                jend = sy_;
              else if (jtype == 0)
                jend = sy_ - 1;

              int iend = itype + 1;
              if (itype == 0 and shift == 0)
                iend = sy_ - 1;
              else if (itype == 0)
                iend = sy_;

              if (!(itype == 0 || jtype == 0))
                continue;

              if (shift == 1 and itype != 0)
                continue;

              if (shift == 0 and jtype != 0)
                continue;

              for (int k = ktype; k < ((ktype || nz_ <= 1) ? ktype+1 : sz_-1); k++)
                {
                for (int j = jtype; j < jend; j++)
                  {
                  for (int i = itype; i < iend; i++)
                    {
                    int gid = first - i * dof_ + i * nx_ * dof_ +
                      j * dof_ + j * nx_ * dof_ +
                      k * nx_ * ny_ * dof_ + d + nx_ * dof_ - dof_ + shift * dof_;

                    int gsdNode = operator()(first - i * dof_ + i * nx_ * dof_ +
                      j * dof_ + j * nx_ * dof_ -
                      (ktype < 0) * sz_ * nx_ * ny_ * dof_ + d + nx_ * dof_ - dof_ + shift * dof_);
                    // Check if this is actually in this domain or a neighbouring domain
                    if (gsdNode == gsdLayer ||
                        (itype == sy_-1 and gsdNode % npl_ == gsdLayer % npl_ + npx_ / 2 and
                        (gsdLayer % npl_) % npx_ != npx_ / 2) ||
                        (jtype == -1 and gsdNode % npl_ == gsdLayer % npl_ - npx_ / 2 - 1 and
                        (gsdLayer % npl_) % npx_ != npx_ / 2))
                      {
                      // Normal nodes in interiors and on separators
                      nodes->append(gid);
                      }
                    }
                  }
                }
              }
          }
        }
      }
    }

  for (auto it = retained_nodes.begin(); it != retained_nodes.end(); ++it)
    {
    separator_nodes.append(Teuchos::Array<int>());
    separator_nodes.back().append(*it);
    }

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
  int gsd2 = gsd - (gsd % npl_) / npx_;
  // Remove boundaries of layers above this one
  gsd2 -= gsd / npl_ * (npy_ + npx_ / 2);

  int npl = npx_ / 2 * npy_ * 2;

  // Right boundary
  if ((gsd % npl_) % npx_ == npx_ / 2 * 2)
    return (gsd2 - npx_ / 2) / ((npl * npz_) / nprocs_);

  // Bottom boundary
  if (gsd % npl_ >= npl_ - npx_ / 2)
    return (gsd2 - npx_ / 2 * 2 * npy_) / ((npl * npz_) / nprocs_);

  if (gsd2 >= npl_ * npz_)
    {
    Tools::Error("Subdomain index out of range for i="+toString(i)+
      ", j="+toString(j)+", k="+toString(k)+
      ", gsd=" + toString(gsd)+", gsd2=" + toString(gsd2)+
      ", npx="+toString(npx_)+", npy="+toString(npy_)+", npz="+toString(npz_)+
      ", npl="+toString(npl_),
      __FILE__, __LINE__);
    }

  return gsd2 / ((npl * npz_) / nprocs_);
  }

  }
