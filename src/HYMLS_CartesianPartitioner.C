#include "HYMLS_CartesianPartitioner.H"
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
CartesianPartitioner::CartesianPartitioner(
  Teuchos::RCP<const Epetra_Map> map, int nx, int ny, int nz, int dof, int pvar,
  GaleriExt::PERIO_Flag perio)
  : BasePartitioner(), label_("CartesianPartitioner"),
    baseMap_(map), cartesianMap_(Teuchos::null),
    nx_(nx), ny_(ny), nz_(nz),
    npx_(-1), npy_(-1), npz_(-1),
    numLocalSubdomains_(-1),
    dof_(dof), pvar_(pvar), active_(true), perio_(perio)
  {
  HYMLS_PROF3(label_,"Constructor");
  comm_ = Teuchos::rcp(&(baseMap_->Comm()), false);

  if (baseMap_->IndexBase64() != 0)
    {
    Tools::Warning("Not sure, but I _think_ your map should be 0-based",
      __FILE__, __LINE__);
    }
  }

// destructor
CartesianPartitioner::~CartesianPartitioner()
  {
  HYMLS_PROF3(label_,"Destructor");
  }

// get non-overlapping subdomain id
int CartesianPartitioner::operator()(int i, int j, int k) const
  {
#ifdef HYMLS_TESTING
  if (!Partitioned())
    {
    Tools::Error("Partition() not yet called!", __FILE__, __LINE__);
    }
#endif
  return (k / sz_ * npy_ + j / sy_) * npx_ + i / sx_;
  }

//! get non-overlapping subdomain id
int CartesianPartitioner::operator()(hymls_gidx gid) const
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
int CartesianPartitioner::NumLocalParts() const
  {
  if (numLocalSubdomains_<0)
    {
    Tools::Error("not implemented correctly",__FILE__,__LINE__);
    }
  return numLocalSubdomains_;
  }

int CartesianPartitioner::CreateSubdomainMap()
  {
  int NumMyElements = 0;
  int NumGlobalElements = npx_ * npy_ * npz_;
  int *MyGlobalElements = new int[NumGlobalElements];

  for (int k = 0; k < npz_; k++)
    for (int j = 0; j < npy_; j++)
      for (int i = 0; i < npx_; i++)
        {
        int pid = PID(sx_ * i, sy_ * j, sz_ * k);
        if (pid == comm_->MyPID())
          MyGlobalElements[NumMyElements++] = i + j * npx_ + k * npx_ * npy_;
        }

  sdMap_ = Teuchos::rcp(new Epetra_Map(NumGlobalElements,
      NumMyElements, MyGlobalElements, 0, *comm_));

  delete [] MyGlobalElements;
  return 0;
  }

int CartesianPartitioner::Partition(int nparts, bool repart)
  {
  HYMLS_PROF3(label_,"Partition");
  int npx,npy,npz;
  Tools::SplitBox(nx_, ny_, nz_, nparts, npx, npy, npz);
  return Partition(nx_ / npx, ny_ / npy, nz_ / npz, repart);
  }

// partition an [nx x ny x nz] grid with one DoF per node
// into nparts global subdomains.
int CartesianPartitioner::Partition(int sx,int sy, int sz, bool repart)
  {
  HYMLS_PROF3(label_,"Partition (2)");
  sx_ = sx;
  sy_ = sy;
  sz_ = sz;

  npx_ = nx_ / sx_;
  npy_ = ny_ / sy_;
  npz_ = nz_ / sz_;

  std::string s1=toString(nx_)+"x"+toString(ny_)+"x"+toString(nz_);
  std::string s2=toString(npx_)+"x"+toString(npy_)+"x"+toString(npz_);

  if ((nx_!=npx_*sx_)||(ny_!=npy_*sy_)||(nz_!=npz_*sz_))
    {
    std::string msg = "You are trying to partition an "+s1+" domain into "+s2+" parts.\n"
      "We currently need nx to be a multiple of npx etc.";
    Tools::Error(msg,__FILE__,__LINE__);
    }

  std::string s3=toString(sx_)+"x"+toString(sy_)+"x"+toString(sz_);

  Tools::Out("Partition domain: ");
  Tools::Out("Grid size: "+s1);
  Tools::Out("Number of Subdomains: "+s2);
  Tools::Out("Subdomain size: "+s3);

  // case where there are more processor partitions than subdomains (experimental)
  if (comm_->MyPID()>=npx_*npy_*npz_)
    {
    active_ = false;
    }

  int color = active_? 1: 0;
  int nprocs;

  CHECK_ZERO(comm_->SumAll(&color,&nprocs,1));

  while (nprocs)
    {
    if (!Tools::SplitBox(nx_, ny_, nz_, nprocs, nprocx_, nprocy_, nprocz_, sx_, sy_, sz_))
      {
      break;
      }
    else
      {
      nprocs--;
      }
    }
  std::string s4 = Teuchos::toString(nprocx_)+"x"+Teuchos::toString(nprocy_)+"x"+Teuchos::toString(nprocz_);

  if (comm_->MyPID() >= nprocs)
    active_ = false;

  // if some processors have no subdomains, we need to
  // repartition the map even if it is a cartesian partitioned
  // map already:
  if (nprocs < comm_->NumProc())
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
    Tools::Out("repartition for "+s4+" procs");
    HYMLS_PROF3(label_,"repartition map");

    int numMyElements = numLocalSubdomains_ * sx_ * sy_ * sz_ * dof_;
    hymls_gidx *myGlobalElements = new hymls_gidx[numMyElements];
    int pos = 0;
    for (hymls_gidx i = 0; i < baseMap_->MaxAllGID64()+1; i++)
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

    Epetra_Map tmpRepartitionedMap((hymls_gidx)(-1), pos,
      myGlobalElements, (hymls_gidx)baseMap_->IndexBase64(), *comm_);

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
        myGlobalElements[pos++] = tmpRepartitionedMap.GID64(i);
        }
      }

    cartesianMap_ = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1), pos,
        myGlobalElements, (hymls_gidx)baseMap_->IndexBase64(), *comm_));

    HYMLS_DEBVAR(*repartitionedMap);
    if (myGlobalElements)
      delete [] myGlobalElements;
    }

  if (active_)
    {
    Tools::Out("Number of Partitions: " + s4);
    Tools::Out("Number of Local Subdomains: " + toString(NumLocalParts()));
    }
  return 0;
  }

int CartesianPartitioner::RemoveBoundarySeparators(Teuchos::Array<hymls_gidx> &interior_nodes,
  Teuchos::Array<Teuchos::Array<hymls_gidx> > &separator_nodes) const
  {
  // TODO: There should be a much easier way to do this, but I want to get rid
  // of it eventually for consistency. We need those things anyway for periodic
  // boundaries.

  // Remove boundary separators and add them to the interior
  Teuchos::Array<Teuchos::Array<hymls_gidx> >::iterator sep = separator_nodes.begin();
  for (; sep != separator_nodes.end(); sep++)
    {
    Teuchos::Array<hymls_gidx> &nodes = *sep;
    if (nodes.size() == 0)
      separator_nodes.erase(sep);
    else if ((nodes[0] / dof_ + 1) % nx_ == 0 && !(perio_ & GaleriExt::X_PERIO))
      {
      // Remove right side
      interior_nodes.insert(interior_nodes.end(), nodes.begin(), nodes.end());
      separator_nodes.erase(sep);
      }
    else if (ny_ > 1 && (nodes[0] / dof_ / nx_ + 1) % ny_ == 0 && !(perio_ & GaleriExt::Y_PERIO))
      {
      // Remove bottom side
      interior_nodes.insert(interior_nodes.end(), nodes.begin(), nodes.end());
      separator_nodes.erase(sep);
      }
    else if (nz_ > 1 && (nodes[0] / dof_ / nx_ / ny_ + 1) % nz_ == 0 && !(perio_ & GaleriExt::Z_PERIO))
      {
      // Remove back side
      interior_nodes.insert(interior_nodes.end(), nodes.begin(), nodes.end());
      separator_nodes.erase(sep);
      }
    else
      continue;
    sep = separator_nodes.begin();
    }

  // Add back boundary nodes to separators that are not along the boundaries
  for (sep = separator_nodes.begin(); sep != separator_nodes.end(); sep++)
    {
    hymls_gidx nodeID = -1;
    Teuchos::Array<hymls_gidx> &nodes = *sep;
    for (int i = 0; i < nodes.size(); i++)
      {
      if ((nodes[i] / dof_) % nx_ + 2 == nx_ && !(perio_ & GaleriExt::X_PERIO))
        {
        nodeID = nodes[i] + dof_;
        Teuchos::Array<hymls_gidx>::iterator it = std::find(
          interior_nodes.begin(), interior_nodes.end(), nodeID);
        if (it != interior_nodes.end())
          {
          nodes.push_back(nodeID);
          interior_nodes.erase(it);
          }
        }
      if (ny_ > 1 && (nodes[i] / dof_ / nx_) % ny_ + 2 == ny_ && !(perio_ & GaleriExt::Y_PERIO))
        {
        nodeID = nodes[i] + dof_ * nx_;
        Teuchos::Array<hymls_gidx>::iterator it = std::find(
          interior_nodes.begin(), interior_nodes.end(), nodeID);
        if (it != interior_nodes.end())
          {
          nodes.push_back(nodeID);
          interior_nodes.erase(it);
          }
        }
      if (nz_ > 1 && (nodes[i] / dof_ / nx_ / ny_) % nz_ + 2 == nz_ && !(perio_ & GaleriExt::Z_PERIO))
        {
        nodeID = nodes[i] + dof_ * nx_ * ny_;
        Teuchos::Array<hymls_gidx>::iterator it = std::find(
          interior_nodes.begin(), interior_nodes.end(), nodeID);
        if (it != interior_nodes.end())
          {
          nodes.push_back(nodeID);
          interior_nodes.erase(it);
          }
        }
      }
    std::sort(nodes.begin(), nodes.end());
    }

  // Since we added some random nodes to the end of the interior
  // we sort them here
  std::sort(interior_nodes.begin(), interior_nodes.end());

  return 0;
  }

int CartesianPartitioner::GetGroups(int sd, Teuchos::Array<hymls_gidx> &interior_nodes,
  Teuchos::Array<Teuchos::Array<hymls_gidx> > &separator_nodes)
  {
  HYMLS_PROF2(label_,"GetGroups");

  // pressure nodes that need to be retained
  Teuchos::Array<hymls_gidx> retained_nodes;

  Teuchos::Array<hymls_gidx> *nodes;

  int gsd = sdMap_->GID(sd);
  int xpos = (gsd % npx_) * sx_;
  int ypos = ((gsd / npx_) % npy_) * sy_;
  int zpos = ((gsd / npx_ / npy_) % npz_) * sz_;

  for (int ktype = (nz_ > 1 ? -1 : 0); ktype < (nz_ > 1 ? 2 : 1); ktype++)
    {
    if (ktype == 1)
      ktype = sz_ - 1;
    if (ktype == -1 && zpos == 0 && !(perio_ & GaleriExt::Z_PERIO))
      continue;
 
    for (int jtype = -1; jtype < 2; jtype++)
      {
      if (jtype == 1)
        jtype = sy_ - 1;
      if (jtype == -1 && ypos == 0 && !(perio_ & GaleriExt::Y_PERIO))
        continue;

      for (int itype = -1; itype < 2; itype++)
        {
        if (itype == 1)
          itype = sx_ - 1;
        if (itype == -1 && xpos == 0 && !(perio_ & GaleriExt::X_PERIO))
          continue;

        for (int d = 0; d < dof_; d++)
          {
          if (d == pvar_ && (itype == -1 || jtype == -1 || ktype == -1))
            continue;
          else if ((itype == 0 && jtype == 0 && ktype == 0) || (d == pvar_ && !(
                itype == sx_-1 && jtype == sy_-1 && (nz_ <= 1 || ktype == sz_-1))))
            nodes = &interior_nodes;
          else
            {
            separator_nodes.append(Teuchos::Array<hymls_gidx>());
            nodes = &separator_nodes.back();
            }

          for (int k = ktype; k < ((ktype || nz_ <= 1) ? ktype+1 : sz_-1); k++)
            {
            for (int j = jtype; j < (jtype ? jtype+1 : sy_-1); j++)
              {
              for (int i = itype; i < (itype ? itype+1 : sx_-1); i++)
                {
                hymls_gidx gid = d +
                  ((i + xpos + nx_) % nx_) * dof_ +
                  ((j + ypos + ny_) % ny_) * nx_ * dof_ +
                  ((k + zpos + nz_) % nz_) * nx_ * ny_ * dof_;
                if ((d == pvar_ && !i && !j && !k))
                  {
                  // Retained pressure nodes
                  retained_nodes.append(gid);
                  }
                else
                  // Normal nodes in interiors and on separators
                  nodes->append(gid);
                }
              }
            }
          }
        }
      }
    }

  RemoveBoundarySeparators(interior_nodes, separator_nodes);

  for (auto it = retained_nodes.begin(); it != retained_nodes.end(); ++it)
    {
    separator_nodes.append(Teuchos::Array<hymls_gidx>());
    separator_nodes.back().append(*it);
    }

  return 0;
  }

//! get the type of a variable (if more than 1 dof per node, otherwise just 0)
int CartesianPartitioner::VariableType(hymls_gidx gid) const
  {
  return gid % dof_;
  }

//! get processor on which a grid point is located
int CartesianPartitioner::PID(int i, int j, int k) const
  {
  // in which subdomain is the cell?
  int sdx = i / sx_;
  int sdy = j / sy_;
  int sdz = k / sz_;

  //how many subdomains are there per process?
  int npx = npx_ / nprocx_;
  int npy = npy_ / nprocy_;
  int npz = npz_ / nprocz_;

#ifdef HYMLS_TESTING
  if ( (npx*nprocx_!=npx_) ||
    (npy*nprocy_!=npy_) ||
    (npz*nprocz_!=npz_) )
    {
    Tools::Error("case of irregular partitioning not implemented",
      __FILE__,__LINE__);
    }
#endif
  // so, where is the cell (on which process)?
  int pidx = sdx / npx;
  int pidy = sdy / npy;
  int pidz = sdz / npz;

  return (pidz * nprocy_ + pidy) * nprocx_ + pidx;
  }

  }
