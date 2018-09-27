#include "HYMLS_SkewCartesianPartitioner.H"
#include "HYMLS_Tools.H"
#include "Teuchos_Utils.hpp"
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_IntVector.h"
#include "Epetra_Import.h"

using Teuchos::toString;

struct Plane {
  std::vector<hymls_gidx> ptr;
  std::vector<hymls_gidx> plane;
  };

Plane buildPlane45(hymls_gidx firstNode, int length,
  hymls_gidx dirX, hymls_gidx dirY, hymls_gidx type)
  {
  hymls_gidx left = firstNode;
  hymls_gidx right = firstNode;
  int height = 2 * length;
  bool extraLayer = false;

  // Skew direction in xy-plane
  hymls_gidx dir1 = dirY + dirX;
  hymls_gidx dir2 = dirY - dirX;

  // correction for u nodes
  if (type == 0)
    {
    left -= dirX;
    height++;
    extraLayer = true;
    }
  else if (type == 3)
    {
    height++;
    extraLayer = true;
    }

  // Build the plane
  Plane plane;
  plane.ptr.push_back(0);
  for (int i = 0; i < height-1; i++)
    {
    for (hymls_gidx j = left; j <= right; j += dirX)
      plane.plane.push_back(j);
    plane.ptr.push_back(plane.plane.size());

    if (i < length-1)
      {
      left += dir2;
      right += dir1;
      }
    else if (extraLayer && i == length-1)
      {
      left += dirY;
      right += dirY;
      }
    else
      {
      left += dir1;
      right += dir2;
      }
    }
  return plane;
  }

namespace HYMLS {

// by default, everyone is assumed to own a part of the domain.
// if in Partition() it turns out that there are more processor
// partitions than subdomains, active_ is set to false for some
// ranks, which will get an empty part of the reordered map
// (cartesianMap_).

// constructor
SkewCartesianPartitioner::SkewCartesianPartitioner(
  Teuchos::RCP<const Epetra_Map> map,
  Teuchos::RCP<Teuchos::ParameterList> const &params,
  Epetra_Comm const &comm)
  : BasePartitioner(), label_("SkewCartesianPartitioner"),
    comm_(Teuchos::rcp(comm.Clone())), baseMap_(map), cartesianMap_(Teuchos::null),
    npx_(-1), npy_(-1), npz_(-1),
    numLocalSubdomains_(-1),active_(true)
  {
  HYMLS_PROF3(label_, "Constructor");

  SetParameters(*params);
  }

// destructor
SkewCartesianPartitioner::~SkewCartesianPartitioner()
  {
  HYMLS_PROF3(label_,"Destructor");
  }

// get non-overlapping subdomain id
int SkewCartesianPartitioner::operator()(int x, int y, int z) const
  {
#ifdef HYMLS_TESTING
  if (!Partitioned())
    {
    Tools::Error("Partition() not yet called!", __FILE__, __LINE__);
    }
#endif
  int sd = GetSubdomainID(sx_, x, y, z);

  return sd;
  }

//! get non-overlapping subdomain id
int SkewCartesianPartitioner::operator()(hymls_gidx gid) const
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

int SkewCartesianPartitioner::GetSubdomainPosition(
  int sd, int sx, int &x, int &y, int &z) const
  {
  int npx = nx_ / sx;
  int npy = ny_ / sx;
  int totNum2DCubes = npx * npy; // number of cubes for fixed z
  int numPerLayer = 2 * totNum2DCubes + npx + npy; // domains for fixed z
  int numPerRow = 2 * npx + 1; // domains in a row (both lattices); fixed y

  int Z = sd / numPerLayer;
  int Y = ((sd - Z * numPerLayer) / numPerRow) * 2 - 1;
  int X = ((sd - Z * numPerLayer) % numPerRow) * 2;
  if (X >= npx * 2)
    {
    X -= npx * 2 + 1;
    Y += 1;
    }

  x = (X * sx) / 2;
  y = (Y * sx) / 2;
  z = (Z - 1) * sx;

  if (x == nx_ - sx / 2 && perio_ & GaleriExt::X_PERIO)
    return 1;
  if (y == ny_ - sx / 2 && perio_ & GaleriExt::Y_PERIO)
    return 1;
  if (z == nz_ - sx && perio_ & GaleriExt::Z_PERIO)
    return 1;
  return 0;
  }

int SkewCartesianPartitioner::GetSubdomainID(
  int sx, int x, int y, int z) const
  {
  int npx = nx_ / sx;
  int npy = ny_ / sx;
  int npz = std::max(nz_ / sx, 1);

  int dir1 = npx + 1;
  int dir2 = npx;
  int dir3 = 2 * npx * npy + npx + npy;

  // which cube
  int xcube = x / sx;
  int ycube = y / sx;
  int zcube = z / sx;

  // first domain in the cube
  int sd = zcube * dir3 + ycube * (dir2 + dir1) + xcube;

  // relative coordinates
  x -= xcube * sx - 1;
  y -= ycube * sx;
  z -= zcube * sx;

  bool front = y < sx - x; // In front of the red plane
  bool right = y < x; // Right of the green plane;
  bool below = z <= y - x; // Below the blue plane left of the green plane
  if (right) below = z <= sx + y - x; // Below the blue plane right of the green plane

  if (!front)
    sd += dir1;
  if (!right)
    sd += dir2;
  if (!below)
    sd += dir3;

  if (!front && right && perio_ & GaleriExt::X_PERIO && xcube == npx - 1)
    sd -= dir2;

  if (!front && !right && perio_ & GaleriExt::Y_PERIO && ycube == npy - 1)
    sd -= dir3 - dir2;

  if (!below && perio_ & GaleriExt::Z_PERIO && zcube == npz - 1)
    sd -= npz * dir3;

  return sd;
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
  HYMLS_PROF2(label_,"CreateSubdomainMap");
  int totNum2DCubes = npx_ * npy_; // number of cubes for fixed z
  int numPerLayer = 2 * totNum2DCubes + npx_ + npy_; // domains for fixed z

  // First layer
  int NumGlobalElements = numPerLayer;
  // Other layers
  if (nz_ > 1)
    NumGlobalElements += numPerLayer * npz_;

  int *MyGlobalElements = new int[NumGlobalElements];
  int NumMyElements = 0;

  for (int sd = 0; sd < NumGlobalElements; sd++)
    {
    int i, j, k;
    if (GetSubdomainPosition(sd, sx_, i, j, k) == 1)
      continue;

    i = ((i + sx_ / 2 - 1) % nx_ + nx_) % nx_;
    j = (j % ny_ + ny_) % ny_;
    k = ((k + sx_) % nz_ + nz_) % nz_;

    int pid = PID(i, j, k);
    if (pid == comm_->MyPID())
      MyGlobalElements[NumMyElements++] = sd;
    }

  sdMap_ = Teuchos::rcp(new Epetra_Map(-1,
      NumMyElements, MyGlobalElements, 0, *comm_));

  delete[] MyGlobalElements;
  return 0;
  }

// partition an [nx x ny x nz] grid with one DoF per node
// into nparts global subdomains.
int SkewCartesianPartitioner::Partition(bool repart)
  {
  HYMLS_PROF3(label_,"Partition (2)");

  if (sx_ != sy_ || (nz_ > 1 && sx_ != sz_))
    Tools::Error("sx, sy and sz should be the same", __FILE__, __LINE__);

  if ((sx_ / 2) * 2 != sx_)
    Tools::Error("sx should be even", __FILE__, __LINE__);

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

  npx_ = nx_ / sx_;
  npy_ = ny_ / sy_;
  npz_ = nz_ / sz_;

  std::string s1 = toString(nx_) + "x" + toString(ny_) + "x" + toString(nz_);
  std::string s2 = toString(npx_) + "x" + toString(npy_) + "x" + toString(npz_);

  if ((nx_ != npx_ * sx_) || (ny_ != npy_ * sy_) || (nz_ != npz_ * sz_))
    {
    std::string msg = "You are trying to partition an " + s1 + " domain into " + s2 + " parts.\n";
    Tools::Error(msg, __FILE__, __LINE__);
    }

  std::string s3 = toString(sx_)+"x"+toString(sy_)+"x"+toString(sz_);

  Tools::Out("Partition domain: ");
  Tools::Out("Grid size: " + s1);
  Tools::Out("Number of Subdomains: " + s2);
  Tools::Out("Subdomain size: " + s3);

  // case where there are more processor partitions than subdomains (experimental)
  if (comm_->MyPID() >= npx_ * npy_ * npz_)
    active_ = false;

  int color = active_ ? 1 : 0;
  CHECK_ZERO(comm_->SumAll(&color, &nprocs_, 1));

  while (nprocs_)
    {
    if (!Tools::SplitBox(nx_, ny_, nz_, nprocs_, nprocx_, nprocy_, nprocz_, sx_, sy_, sz_))
      {
      break;
      }
    else
      {
      nprocs_--;
      }
    }

  std::string s4 = Teuchos::toString(nprocx_) + "x" +
    Teuchos::toString(nprocy_) + "x" + Teuchos::toString(nprocz_);

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

  CHECK_ZERO(getTemplate());
  CHECK_ZERO(solveGroups());

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
    cartesianMap_ = RepartitionMap(baseMap_);
    }

#ifdef HYMLS_TESTING
  // Now we have a skew cartesian processor partitioning and no nodes
  // have to be moved between partitions. Some partitions may be
  // empty, though. Check that we do not miss anything
  for (int lid = 0; lid < repartitionedMap->NumMyElements(); lid++)
    {
    hymls_gidx gid = repartitionedMap->GID64(lid);
    if (PID(gid) != comm_->MyPID())
      {
      Tools::Error("Repartitioning seems to be necessary/have failed for gid "
        + Teuchos::toString(gid) + ".", __FILE__, __LINE__);
      }
    }
#endif

  if (active_)
    {
    Tools::Out("Number of Partitions: " + s4);
    Tools::Out("Number of Local Subdomains: " + toString(NumLocalParts()));
    }
  return 0;
  }

// Builds a domain 'template' of given type at the origin. The template can
// be moved around to create the proper domain.
int SkewCartesianPartitioner::getTemplate()
  {
  HYMLS_PROF2(label_, "getTemplate");

  if (!active_)
      return 0;

  hymls_gidx nx = sx_ * 4;

  // Cartesian directions
  hymls_gidx dirX = dof_;
  hymls_gidx dirY = dof_*nx;
  hymls_gidx dirZ = dof_*nx*nx;

  // Info for each node type
  hymls_gidx firstNode[4] = {dof_*sx_/2 + dirY + dirZ * sx_,
                             dof_*sx_/2 - 0    + dirZ * sx_,
                             dof_*sx_/2 + dirY + dirZ * sx_,
                             dof_*sx_/2 + dirY + dirZ * sx_};
  int baseLength[4] = {sx_/2, sx_/2 + 1, sx_/2 + 1, sx_/2};

  std::vector<std::vector<std::vector<hymls_gidx> > > nodes;

  for (int type = 0; type < 4; type++)
    {
    nodes.emplace_back(2 * sx_ + 1);

    // Get central layer
    Plane plane = buildPlane45(firstNode[type], baseLength[type], dirX, dirY, type);
    nodes[type][sx_] = plane.plane;

    if (nz_ <= 1)
      continue;

    std::vector<hymls_gidx> bottom;
    std::vector<hymls_gidx> top = plane.plane;

    // Used in the loop to determine which nodes to assign to each layer
    std::vector<hymls_gidx> rowLength;
    for (size_t i = 0; i < plane.ptr.size()-1; i++)
      rowLength.push_back(plane.ptr[i+1] - plane.ptr[i] - 1); // -1 ???

    std::vector<hymls_gidx> activePtrs;
    for (int i = 0; i < baseLength[type]; i++)
      activePtrs.push_back(i);

    std::vector<hymls_gidx> offset;
    for (auto i: activePtrs)
      offset.push_back(rowLength[i]);

    for (int i = 0; i < sx_; i++)
      {
      // Get indices for the layers
      auto last = top.end();
      for (size_t j = 0; j < activePtrs.size(); j++)
        {
        hymls_gidx val = plane.plane[plane.ptr[activePtrs[j]] + offset[j]];
        bottom.push_back(val);
        last = std::remove(top.begin(), last, val);
        }
      top.erase(last, top.end());

      // Add layers
      if (type == 2)
        {
        // w layers are an exception with odd/even layers
        if (i % 2 == 1)
          {
          for (hymls_gidx j: top)
            nodes[type][sx_ + i].push_back(j + i * dirZ - dirY);
          for (hymls_gidx j: top)
            nodes[type][sx_ + 1 + i].push_back(j + (i + 1) * dirZ);
          }
        else
          {
          for (hymls_gidx j: bottom)
            nodes[type][i].push_back(j - (sx_ - i) * dirZ);
          if (i > 0)
            for (hymls_gidx j: bottom)
              nodes[type][i - 1].push_back(j - (sx_ - i + 1) * dirZ - dirY);
          else
            for (hymls_gidx j: plane.plane)
              nodes[type][sx_ - 1].push_back(j - dirZ - dirY);
          }
        }
      else
        {
        hymls_gidx isPvar = type == 3;
        if (i < sx_-isPvar)
          for (hymls_gidx j: bottom)
            nodes[type][i + isPvar].push_back(j - (sx_ - i - isPvar) * dirZ);
        for (hymls_gidx j: top)
          nodes[type][sx_ + 1 + i].push_back(j + (i + 1) * dirZ);
        }

      if (i < sx_ - 1)
        {
        // Update pointers and offset
        std::for_each(offset.begin(), offset.end(), [](hymls_gidx& d) { d--;});
        if (type == 3)
          {
          // pressure nodes are an exception
          if (offset[0] < 0)
            {
            activePtrs.push_back(activePtrs.back() + 1);
            activePtrs.erase(activePtrs.begin());
            offset.push_back(rowLength[activePtrs.back()]);
            offset.erase(offset.begin());
            }
          }
        else
          {
          if (offset[0] < 0)
            {
            activePtrs.erase(activePtrs.begin());
            offset.erase(offset.begin());
            }
          else if (offset[0] == 0)
            {
            activePtrs.push_back(activePtrs.back() + 1);
            offset.push_back(rowLength[activePtrs.back()]);
            }
          }
        }
      }
    }

  // As turns out, there are some unnecessary nodes, so we remove them
  // afterwards. Should be implemented with conditionals in main loop soon
  // Remove top and bottop single wall
  nodes[0].pop_back();
  nodes[0].erase(nodes[0].begin());

  nodes[1].pop_back();
  nodes[1].erase(nodes[1].begin());

  nodes[2].pop_back();

  nodes[3].pop_back();
  nodes[3].erase(nodes[3].begin());

  // // Remove more unnecessary separators, located at second and second-last
  // // layers of original template
  // for (int i = 0; i < 3; i++)
  //   {
  //   if (nodes[i].front().size())
  //     nodes[i].front().erase(std::max_element(nodes[i].front().begin(), nodes[i].front().end()));
  //   if (nodes[i].back().size())
  //     nodes[i].back().erase(std::min_element(nodes[i].back().begin(), nodes[i].back().end()));
  //   }

  // Merge the template layers
  template_.resize(0);
  template_.emplace_back();
  for (int i = 0; i < dof_; i++)
    if (variableType_[i] == 2)
      {
      std::copy(nodes[2].front().begin(), nodes[2].front().end(),
        std::back_inserter(template_.back()));
      nodes[2].erase(nodes[2].begin());

      std::for_each(template_.back().begin(), template_.back().end(),
        [i](hymls_gidx& d) { d += i;});
      break;
      }

  for (int j = 0; j < 2 * sx_ - 1; j++)
    {
    template_.emplace_back();
    for (int i = 0; i < dof_; i++)
      {
      int size = template_.back().size();
      std::copy(nodes[variableType_[i]][j].begin(),
        nodes[variableType_[i]][j].end(),
        std::back_inserter(template_.back()));

      std::for_each(template_.back().begin()+size, template_.back().end(),
        [i](hymls_gidx& d) { d += i;});
      }
    std::sort(template_.back().begin(), template_.back().end());
    }

  return 0;
  }

int SkewCartesianPartitioner::solveGroups()
  {
  HYMLS_PROF2(label_, "solveGroups");

  if (!active_)
      return 0;

  long long nx = sx_ * 4;

  // Principal directions for domain displacements
  long long dirX = dof_*sx_;
  long long dirY = dof_*nx*sx_;
  long long dirZ = dof_*nx*nx*sx_;

  // Shift the central domain by 1,1,1
  long long first = dirX + dirY + dirZ;

  long long dir1 = (dirY + dirX)/2;
  long long dir2 = (dirY - dirX)/2 + dirZ;
  long long dir3 = dirZ;

  // Positions of shifted domains
  long long positions[27] = {0, -dir3, dir3, -dir2, -dir2-dir3,
                             -dir2+dir3, dir2, dir2-dir3, dir2+dir3,
                             -dir1, -dir1-dir3, -dir1+dir3, -dir1-dir2,
                             -dir1-dir2-dir3, -dir1-dir2+dir3, -dir1+dir2,
                             -dir1+dir2-dir3, -dir1+dir2+dir3, dir1,
                             dir1-dir3, dir1+dir3, dir1-dir2,
                             dir1-dir2-dir3, dir1-dir2+dir3, dir1+dir2,
                             dir1+dir2-dir3, dir1+dir2+dir3};

  // Turn the template into a list
  std::vector<long long> tempList;
  for (auto &it: template_)
    for (auto &it2: it)
      tempList.push_back(it2 + first);

  // Find groups
  groups_.resize(0);
  std::vector<unsigned long> groupDomains;

  // First group is the interior
  groups_.emplace_back(0);
  groupDomains.push_back(1);

  for (auto &node: tempList)
    {
    // For each node, first check to which domains it belongs and store it as bits
    unsigned long listOfDomains = 0;
    for (int i = 0; i < 27; i++)
      {
      auto it = std::lower_bound(tempList.begin(), tempList.end(), node - positions[i]);
      if (it != tempList.end() && *it == node - positions[i])
        listOfDomains += 1 << i;
      }

    // Now check whether a group for this list was already created, and if
    // not, create it. Then add the nodes+domains to this group
    bool newGroup = true;
    for (size_t i = 0; i < groups_.size(); i++)
      if (groupDomains[i] == listOfDomains)
        {
        newGroup = false;
        groups_[i].push_back(node);
        break;
        }

    if (newGroup)
      {
      groups_.emplace_back(1, node);
      groupDomains.push_back(listOfDomains);
      }
    }

  // Now separate the u, v, w and p, skip the interior
  std::vector<std::vector<std::vector<hymls_gidx> > > newGroups;
  for (size_t i = 1; i < groups_.size(); i++)
    {
    auto group = groups_[i];
    newGroups.emplace_back(dof_);

    for (auto const &node: group)
      newGroups.back()[((node % dof_) + dof_) % dof_].push_back(node);
    }

  // Remove empty groups from newGroups and place them after
  // the interior in groups
  groups_.resize(1);
  for (auto &cats: newGroups)
    for (auto &group: cats)
      if (!group.empty())
        {
        std::sort(group.begin(), group.end());
        groups_.push_back(group);
        }

  return 0;
  }

std::vector<std::vector<hymls_gidx> > SkewCartesianPartitioner::createSubdomain(int sd) const
  {
  HYMLS_PROF3(label_, "createSubdomain");

  int sdx, sdy, sdz;
  GetSubdomainPosition(sd, sx_, sdx, sdy, sdz);

  int nx = 4 * sx_;

  // Move the groups to the right position and cut off parts that fall
  // outside of the domain
  std::vector<std::vector<hymls_gidx> > groups;
  for (auto &group: groups_)
    {
    groups.emplace_back();
    for (hymls_gidx const &node: group)
      {
      int var = node % dof_;
      int x = (node / dof_) % nx + sdx - 1 - sx_;
      int y = (node / dof_ / nx) % nx + sdy - 1 - sx_;
      int z = node / dof_ / nx / nx + sdz - sx_;
      if (perio_ & GaleriExt::X_PERIO) x = (x + nx_) % nx_;
      if (perio_ & GaleriExt::Y_PERIO) y = (y + ny_) % ny_;
      if (perio_ & GaleriExt::Z_PERIO) z = (z + nz_) % nz_;
      if (x >= 0 && x < nx_ && y >= 0 && y < ny_ && z >= 0 && z < nz_)
        groups.back().push_back(
          (hymls_gidx)x * dof_ +
          (hymls_gidx)nx_ * y * dof_ +
          (hymls_gidx)nx_ * ny_ * z * dof_ + var);
      }
    }

  // Remove empty groups
  groups.erase(std::remove_if(groups.begin()+1, groups.end(),
      [](std::vector<hymls_gidx> &i){return i.empty();}), groups.end());

  // Get first pressure node from interior to a new group.
  // Assumes ordering of groups by size!
  for (int pvar = 0; pvar < dof_; pvar++)
    if (variableType_[pvar] == 3)
      {
      for (hymls_gidx const &node: groups[0])
        if (((node % dof_) + dof_) % dof_ == pvar)
          {
          groups.emplace_back(1, node);
          groups[0].erase(std::remove(groups[0].begin(), groups[0].end(), node), groups[0].end());
          break;
          }
      break;
      }

  // Split separator groups that that do not belong to the same subdomain.
  // This may happen for the w-groups since the w-separators are staggered
  std::vector<std::vector<hymls_gidx> > oldGroups;
  std::copy(groups.begin() + 1, groups.end(), std::back_inserter(oldGroups));
  groups.resize(1);
  for (auto &group: oldGroups)
    {
    std::map<int, std::vector<hymls_gidx> > newGroups;
    for (hymls_gidx node: group)
      {
      int gsd = operator()(node);
      auto newGroup = newGroups.find(gsd);
      if (newGroup != newGroups.end())
        newGroup->second.push_back(node);
      else
        newGroups.emplace(gsd, std::vector<hymls_gidx>(1, node));
      }
    for (auto &newGroup: newGroups)
      groups.push_back(newGroup.second);
    }

  // Remove separator nodes that lie on the boundary of the domain.
  // We need this because those nodes don't actually border any interior
  // nodes of the other subdomain
  for (auto group = groups.begin() + 1; group != groups.end(); ++group)
    {
    std::vector<hymls_gidx> groupCopy = *group;
    for (hymls_gidx node: groupCopy)
      {
      int x = (node / dof_) % nx_;
      int y = (node / dof_ / nx_) % ny_;
      int z = node / dof_ / nx_ / ny_;
      if (dof_ > 1 && x == nx_ - 1 && variableType_[node % dof_] == 0 &&
        !(perio_ & GaleriExt::X_PERIO))
        {
        if (operator()(x, y, z) == sd)
          groups[0].push_back(node);
        group->erase(std::remove(group->begin(), group->end(), node));
        }
      else if (dof_ > 1 && y == ny_ - 1 && variableType_[node % dof_] == 1 &&
        !(perio_ & GaleriExt::Y_PERIO))
        {
        if (operator()(x, y, z) == sd)
          groups[0].push_back(node);
        group->erase(std::remove(group->begin(), group->end(), node));
        }
      else if (nz_ > 1 && dof_ > 1 && z == nz_ - 1 && variableType_[node % dof_] == 2 &&
        !(perio_ & GaleriExt::Z_PERIO))
        {
        if (operator()(x, y, z) == sd)
          groups[0].push_back(node);
        group->erase(std::remove(group->begin(), group->end(), node));
        }
      }
    }

  // Remove empty groups
  groups.erase(std::remove_if(groups.begin()+1, groups.end(),
      [](std::vector<hymls_gidx> &i){return i.empty();}), groups.end());

  // Sort the interior since there may now be new nodes at the back
  std::sort(groups[0].begin(), groups[0].end());

  return groups;
  }

int SkewCartesianPartitioner::GetGroups(int sd, Teuchos::Array<hymls_gidx> &interior_nodes,
  Teuchos::Array<Teuchos::Array<hymls_gidx> > &separator_nodes)
  {
  HYMLS_PROF3(label_,"GetGroups");

  int gsd = sdMap_->GID(sd);

  std::vector<std::vector<hymls_gidx> > nodes = createSubdomain(gsd);
  interior_nodes = nodes[0];
  std::copy(nodes.begin() + 1, nodes.end(), std::back_inserter(separator_nodes));

  return 0;
  }

//! get processor on which a grid point is located
int SkewCartesianPartitioner::PID(int i, int j, int k) const
  {
  #ifdef HYMLS_TESTING
  if (!Partitioned())
    {
    Tools::Error("Partition() not yet called!", __FILE__, __LINE__);
    }
#endif
  int sx = nx_ / nprocx_;
  int sy = ny_ / nprocy_;
  int sz = nz_ / nprocz_;

  int cl = std::min(sx, sy);
  if (nz_ > 1)
    cl = std::min(cl, sz);

  int sd = GetSubdomainID(cl, i, j, k);
  GetSubdomainPosition(sd, cl, i, j, k);

  i = ((i + cl / 2 - 1) % nx_ + nx_) % nx_;
  j = (j % ny_ + ny_) % ny_;
  k = ((k + cl) % nz_ + nz_) % nz_;

  // In which cube is the cell?
  int sdx = i / sx;
  int sdy = j / sy;
  int sdz = k / sz;

  int pid = (sdz * nprocy_ + sdy) * nprocx_ + sdx;

#ifdef HYMLS_TESTING
  if (pid < 0 || pid >= nprocx_ * nprocy_ * nprocz_)
    Tools::Error("Invalid PID "+Teuchos::toString(pid),
      __FILE__, __LINE__);
#endif

  return pid;
  }

  }
