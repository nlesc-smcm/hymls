#include "HYMLS_CartesianPartitioner.hpp"

#include "HYMLS_config.h"

#include "HYMLS_Tools.hpp"
#include "HYMLS_Macros.hpp"
#include "HYMLS_InteriorGroup.hpp"
#include "HYMLS_SeparatorGroup.hpp"

#include "Epetra_Comm.h"
#include "Epetra_Map.h"

#include "Teuchos_Array.hpp"
#include "Teuchos_toString.hpp"
#include "Teuchos_ParameterList.hpp"

#include "GaleriExt_Periodic.h"

#include <algorithm>

using Teuchos::toString;

namespace HYMLS {

// constructor
CartesianPartitioner::CartesianPartitioner(
  Teuchos::RCP<const Epetra_Map> map,
  Teuchos::RCP<Teuchos::ParameterList> const &params,
  Epetra_Comm const &comm, int level)
  : BasePartitioner(comm, level), label_("CartesianPartitioner"),
    baseMap_(map), cartesianMap_(Teuchos::null),
    numLocalSubdomains_(-1),
    bgridTransform_(false)
  {
  HYMLS_PROF3(label_, "Constructor");

  SetParameters(*params);
  }

// destructor
CartesianPartitioner::~CartesianPartitioner()
  {
  HYMLS_PROF3(label_,"Destructor");
  }

void CartesianPartitioner::SetParameters(Teuchos::ParameterList& params)
  {
  BasePartitioner::SetParameters(params);

  Teuchos::ParameterList& precList = params.sublist("Preconditioner");
  bgridTransform_ = precList.get("B-Grid Transform", false);
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
  int npx = (nx_ - 1) / sx + 1;
  int npy = (ny_ - 1) / sy + 1;
  int npz = (nz_ - 1) / sz + 1;

  x = (sd % npx) * sx;
  y = ((sd / npx) % npy) * sy;
  z = ((sd / npx / npy) % npz) * sz;

  return 0;
  }

int CartesianPartitioner::GetSubdomainID(
  int sx, int sy, int sz, int x, int y, int z) const
  {
  int npx = (nx_ - 1) / sx + 1;
  int npy = (ny_ - 1) / sy + 1;

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
  int npx = (nx_ - 1) / sx + 1;
  int npy = (ny_ - 1) / sy + 1;
  int npz = (nz_ - 1) / sz + 1;

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

  int npx = (nx_ - 1) / sx_ + 1;
  int npy = (ny_ - 1) / sy_ + 1;
  int npz = (nz_ - 1) / sz_ + 1;

  std::string s1=toString(nx_)+"x"+toString(ny_)+"x"+toString(nz_);
  std::string s2=toString(npx)+"x"+toString(npy)+"x"+toString(npz);
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

static int GetSubdomainStartAndEnd(
  int pos, int idx, int idx_max, int dim, int max, bool perio, int &type, int &start, int &end)
  {
  int len = std::max((max + idx_max - 1) / idx_max, 1);

  if (idx == idx_max)
    type = 2;
  else if (idx >= 0)
    type = 1;
  else
    type = 0;

  start = idx;
  if (idx == idx_max)
    start = max;
  else if (idx > 0)
    start = std::min(len * idx, max);

  end = start + 1;
  if (type == 1)
    end = std::min(len * (idx + 1), max);

  if (!perio)
    {
    if (pos == 0 && idx == -1)
      return 1;
    if (pos + max + 1 == dim)
      {
      if (idx == idx_max)
        return 1;
      if (idx == idx_max - 1)
        end += 1;
      }
    }

  if (start == end)
    return 1;

  return 0;
  }

int CartesianPartitioner::GetGroups(int sd, InteriorGroup &interior_group,
  Teuchos::Array<SeparatorGroup> &separator_groups) const
  {
  HYMLS_PROF3(label_,"GetGroups");

  interior_group.nodes().clear();
  separator_groups.clear();

  // pressure nodes that need to be retained
  Teuchos::Array<hymls_gidx> retained_nodes;

  Teuchos::Array<hymls_gidx> *nodes, *nodes2;

  int gsd = sdMap_->GID(sd);
  int xpos, ypos, zpos;
  GetSubdomainPosition(gsd, sx_, sy_, sz_, xpos, ypos, zpos);

  int xmax = std::min(nx_ - xpos - 1, sx_ - 1);
  int ymax = std::min(ny_ - ypos - 1, sy_ - 1);
  int zmax = std::min(nz_ - zpos - 1, sz_ - 1);

  if (xmax == 0 || ymax == 0 || (zmax == 0 && nz_ > 1))
    Tools::Error("Can't have a subdomain of size 1", __FILE__, __LINE__);

  // FIXME: Retaining multiple nodes per separator may retain too many
  // per face when not setting directions separately.

  int iidx_max = rx_ > 1 ? rx_ : 1;
  int jidx_max = ry_ > 1 ? ry_ : 1;
  int kidx_max = rz_ > 1 ? rz_ : 1;

  for (int kidx = -1; kidx <= kidx_max; kidx++)
    {
    bool kinterior = kidx >= 0 && kidx < kidx_max;

    int ktype, kstart, kend;
    if (GetSubdomainStartAndEnd(
        zpos, kidx, kidx_max, nz_, zmax, perio_ & GaleriExt::Z_PERIO, ktype, kstart, kend))
      continue;

    for (int jidx = -1; jidx <= jidx_max; jidx++)
      {
      bool jinterior = jidx >= 0 && jidx < jidx_max;

      int jtype, jstart, jend;
      if (GetSubdomainStartAndEnd(
          ypos, jidx, jidx_max, ny_, ymax, perio_ & GaleriExt::Y_PERIO, jtype, jstart, jend))
        continue;

      for (int iidx = -1; iidx <= iidx_max; iidx++)
        {
        bool iinterior = iidx >= 0 && iidx < iidx_max;

        int itype, istart, iend;
        if (GetSubdomainStartAndEnd(
            xpos, iidx, iidx_max, nx_, xmax, perio_ & GaleriExt::X_PERIO, itype, istart, iend))
          continue;

        for (int d = 0; d < dof_; d++)
          {
          nodes2 = NULL;
          if ((variableType_[d] == VariableType::Pressure ||
              variableType_[d] == VariableType::Interior) &&
            (iidx == -1 || jidx == -1 || kidx == -1))
            continue;
          else if ((iinterior && jinterior && kinterior) ||
            variableType_[d] == VariableType::Interior ||
              (variableType_[d] == VariableType::Pressure && (
                // Pressure nodes that are not in tubes
                (iinterior && jinterior) ||
                (iinterior && kinterior) ||
                (jinterior && kinterior) ||
                // B-grid
                retainPressures_ > 1)
              ))
            nodes = &interior_group.nodes();
          else
            {
            SeparatorGroup separator;
            separator.set_type(2 * (d + dof_ * (itype + 3 * (jtype + 3 * ktype))));
            separator_groups.append(separator);
            nodes = &separator.nodes();

            if (bgridTransform_)
              {
              SeparatorGroup separator2;
              separator2.set_type(separator.type() + 1);
              separator_groups.append(separator2);
              nodes2 = &separator2.nodes();
              }
            }

          for (int k = kstart; k < kend; k++)
            {
            for (int j = jstart; j < jend; j++)
              {
              for (int i = istart; i < iend; i++)
                {
                hymls_gidx gid = d +
                  (hymls_gidx)((i + xpos + nx_) % nx_) * dof_ +
                  (hymls_gidx)((j + ypos + ny_) % ny_) * nx_ * dof_ +
                  (hymls_gidx)((k + zpos + nz_) % nz_) * nx_ * ny_ * dof_;
                if (variableType_[d] == VariableType::Pressure &&
                  i >= 0 && j >= 0 && k >= 0 &&
                  retained_nodes.length() < retainPressures_)
                  {
                  // Retained pressure nodes
                  retained_nodes.append(gid);
                  }
                else if (nodes2 && (i + xpos + j + ypos) % 2)
                  nodes2->append(gid);
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

  // Remove empty groups
  separator_groups.erase(std::remove_if(separator_groups.begin(), separator_groups.end(),
      [](SeparatorGroup &i){return i.nodes().empty();}), separator_groups.end());

  // Add retained nodes as separator groups
  for (auto it = retained_nodes.begin(); it != retained_nodes.end(); ++it)
    {
    SeparatorGroup group;
    group.append(*it);
    separator_groups.append(group);
    }

  return 0;
  }

  }
