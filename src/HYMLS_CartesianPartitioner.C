#include "HYMLS_CartesianPartitioner.H"
#include "HYMLS_Tools.H"
#include "Teuchos_Utils.hpp"
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_IntVector.h"
#include "Epetra_Import.h"

using Teuchos::toString;

namespace HYMLS {

// constructor
CartesianPartitioner::CartesianPartitioner(
  Teuchos::RCP<const Epetra_Map> map,
  Teuchos::RCP<Teuchos::ParameterList> const &params,
  Epetra_Comm const &comm)
  : BasePartitioner(), label_("CartesianPartitioner"),
    baseMap_(map), cartesianMap_(Teuchos::null),
    numLocalSubdomains_(-1)
  {
  HYMLS_PROF3(label_, "Constructor");

  comm_ = Teuchos::rcp(comm.Clone());
  SetParameters(*params);
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
  return GetSubdomainID(sx_, sy_, sz_, i, j, k);
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

int CartesianPartitioner::GetSubdomainPosition(
  int sd, int sx, int sy, int sz, int &x, int &y, int &z) const
  {
  int npx = std::max(nx_ / sx, 1);
  int npy = std::max(ny_ / sy, 1);
  int npz = std::max(nz_ / sz, 1);

  x = (sd % npx) * sx;
  y = ((sd / npx) % npy) * sy;
  z = ((sd / npx / npy) % npz) * sz;

  return 0;
  }

int CartesianPartitioner::GetSubdomainID(
  int sx, int sy, int sz, int x, int y, int z) const
  {
  int npx = std::max(nx_ / sx, 1);
  int npy = std::max(ny_ / sy, 1);

  return (z / sz * npy + y / sy) * npx + x / sx;
  }

//! return the number of subdomains in this proc partition
int CartesianPartitioner::NumLocalParts() const
  {
  if (numLocalSubdomains_<0)
    {
    Tools::Error("not implemented correctly",__FILE__,__LINE__);
    }
  return numLocalSubdomains_;
  }

//! return the global number of subdomains
int CartesianPartitioner::NumGlobalParts(int sx, int sy, int sz) const
  {
  int npx = std::max(nx_ / sx, 1);
  int npy = std::max(ny_ / sy, 1);
  int npz = std::max(nz_ / sz, 1);

  return npx * npy * npz;
  }

int CartesianPartitioner::CreateSubdomainMap()
  {
  int NumMyElements = 0;
  int NumGlobalElements = NumGlobalParts(sx_, sy_, sz_);
  int *MyGlobalElements = new int[NumGlobalElements];

  for (int sd = 0; sd < NumGlobalElements; sd++)
    {
    int x, y, z;
    GetSubdomainPosition(sd, sx_, sy_, sz_, x, y, z);
    int pid = PID(x, y, z);
    if (pid == comm_->MyPID())
      MyGlobalElements[NumMyElements++] = sd;
    }

  sdMap_ = Teuchos::rcp(new Epetra_Map(-1,
      NumMyElements, MyGlobalElements, 0, *comm_));

  delete [] MyGlobalElements;

  numLocalSubdomains_ = sdMap_->NumMyElements();

  return 0;
  }

// partition an [nx x ny x nz] grid with one DoF per node
// into nparts global subdomains.
int CartesianPartitioner::Partition(bool repart)
  {
  HYMLS_PROF3(label_,"Partition (2)");

  if (baseMap_ == Teuchos::null)
    {
    hymls_gidx n = (hymls_gidx)nx_ * ny_ * nz_ * dof_;
    baseMap_ = Teuchos::rcp(new Epetra_Map(n, 0, *comm_));
    }

  if (baseMap_->IndexBase64() != 0)
    {
    Tools::Warning("Not sure, but I _think_ your map should be 0-based",
      __FILE__, __LINE__);
    }

  int npx = nx_ / sx_;
  int npy = ny_ / sy_;
  int npz = nz_ / sz_;

  std::string s1=toString(nx_)+"x"+toString(ny_)+"x"+toString(nz_);
  std::string s2=toString(npx)+"x"+toString(npy)+"x"+toString(npz);

  if ((nx_!=npx*sx_)||(ny_!=npy*sy_)||(nz_!=npz*sz_))
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

  CHECK_ZERO(CreatePIDMap());

  if (nprocs_ != comm_->NumProc())
    repart = true;

  CHECK_ZERO(CreateSubdomainMap());

  CHECK_ZERO(SetDestinationPID(baseMap_));

  HYMLS_DEBVAR(*sdMap_);

  // create redistributed map:
  cartesianMap_ = baseMap_;
  
// repartitioning may occur for two reasons, typically on coarser levels:
// a) the number of subdomains becomes smaller than the number of processes,
// b) the subdomains can't be nicely distributed among the processes.
// In both cases some processes are deactivated.
  if (repart)
    {
    Tools::Out("repartition for "+Teuchos::toString(nprocs_)+" procs");
    cartesianMap_ = RepartitionMap(baseMap_);
    }

#ifdef HYMLS_TESTING
  // Now we have a cartesian processor partitioning and no nodes have
  // to be moved between partitions. Some partitions may be empty,
  // though. Check that we do not miss anything
  for (int lid = 0; lid < cartesianMap_->NumMyElements(); lid++)
    {
    hymls_gidx gid = cartesianMap_->GID64(lid);
    if (PID(gid) != comm_->MyPID())
      {
      Tools::Error("Repartitioning seems to be necessary/have failed for gid "
        + Teuchos::toString(gid) + ".", __FILE__, __LINE__);
      }
    }
#endif

  Tools::Out("Number of Partitions: " + Teuchos::toString(nprocs_));
  Tools::Out("Number of Local Subdomains: " + toString(NumLocalParts()));

  return 0;
  }

int CartesianPartitioner::RemoveBoundarySeparators(Teuchos::Array<hymls_gidx> &interior_nodes,
  Teuchos::Array<Teuchos::Array<hymls_gidx> > &separator_nodes) const
  {
  // TODO: There should be a much easier way to do this, but I want to get rid
  // of it eventually for consistency. We need those things anyway for periodic
  // boundaries.

  // Remove boundary separators and add them to the interior
  for (auto sep = separator_nodes.begin(); sep != separator_nodes.end(); )
    {
    Teuchos::Array<hymls_gidx> &nodes = *sep;
    if (nodes.size() == 0)
      sep = separator_nodes.erase(sep);
    else if ((nodes[0] / dof_ + 1) % nx_ == 0 && !(perio_ & GaleriExt::X_PERIO))
      {
      // Remove right side
      interior_nodes.insert(interior_nodes.end(), nodes.begin(), nodes.end());
      sep = separator_nodes.erase(sep);
      }
    else if (ny_ > 1 && (nodes[0] / dof_ / nx_ + 1) % ny_ == 0 && !(perio_ & GaleriExt::Y_PERIO))
      {
      // Remove bottom side
      interior_nodes.insert(interior_nodes.end(), nodes.begin(), nodes.end());
      sep = separator_nodes.erase(sep);
      }
    else if (nz_ > 1 && (nodes[0] / dof_ / nx_ / ny_ + 1) % nz_ == 0 && !(perio_ & GaleriExt::Z_PERIO))
      {
      // Remove back side
      interior_nodes.insert(interior_nodes.end(), nodes.begin(), nodes.end());
      sep = separator_nodes.erase(sep);
      }
    else
      ++sep;
    }

  // Add back boundary nodes to separators that are not along the boundaries
  for (auto sep = separator_nodes.begin(); sep != separator_nodes.end(); ++sep)
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
  Teuchos::Array<Teuchos::Array<hymls_gidx> > &separator_nodes,
  Teuchos::Array<Teuchos::Array<int> > &group_links) const
  {
  HYMLS_PROF3(label_,"GetGroups");

  interior_nodes.clear();
  separator_nodes.clear();

  // pressure nodes that need to be retained
  Teuchos::Array<hymls_gidx> retained_nodes;

  Teuchos::Array<hymls_gidx> *nodes;

  int gsd = sdMap_->GID(sd);
  int xpos, ypos, zpos;
  GetSubdomainPosition(gsd, sx_, sy_, sz_, xpos, ypos, zpos);

  int pvar = -1;
  for (int i = 0; i < dof_; i++)
    if (variableType_[i] == 3)
      pvar = i;

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
          if (d == pvar && (itype == -1 || jtype == -1 || ktype == -1))
            continue;
          else if ((itype == 0 && jtype == 0 && ktype == 0) ||
              (d == pvar && (
                // Pressure nodes that are not in tubes
                (itype == 0 && jtype == 0) ||
                (itype == 0 && ktype == 0) ||
                (jtype == 0 && ktype == 0) ||
                // B-grid
                retain_ > 1)
              ))
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
                  (hymls_gidx)((i + xpos + nx_) % nx_) * dof_ +
                  (hymls_gidx)((j + ypos + ny_) % ny_) * nx_ * dof_ +
                  (hymls_gidx)((k + zpos + nz_) % nz_) * nx_ * ny_ * dof_;
                if (d == pvar && i >= 0 && j >= 0 && k >= 0 &&
                    retained_nodes.length() < retain_)
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

  // TODO: Actually implement this
  group_links.resize(0);
  for (int i = 0; i < separator_nodes.size(); i++)
    group_links.push_back(Teuchos::Array<int>(1, i+1));

  return 0;
  }

  }
