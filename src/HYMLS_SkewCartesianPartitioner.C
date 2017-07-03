#include "HYMLS_SkewCartesianPartitioner.H"
#include "HYMLS_Tools.H"
#include "Teuchos_Utils.hpp"
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_IntVector.h"
#include "Epetra_Import.h"

using Teuchos::toString;

struct Plane {
  std::vector<int> ptr;
  std::vector<int> plane;
  };

Plane buildPlane45(int firstNode, int length, int dirX, int dirY, int dof, int pvar)
  {
  int left = firstNode;
  int right = firstNode;
  int height = 2 * length;
  bool extraLayer = false;

  // Skew direction in xy-plane
  int dir1 = dirY + dirX;
  int dir2 = dirY - dirX;

  // correction for u nodes
  if (((firstNode % dof) + dof) % dof  == 0)
    {
    left -= dirX;
    height++;
    extraLayer = true;
    }
  else if (((firstNode % dof) + dof) % dof  == pvar)
    {
    height++;
    extraLayer = true;
    }

  // Build the plane
  Plane plane;
  plane.ptr.push_back(0);
  for (int i = 0; i < height-1; i++)
    {
    for (int j = left; j <= right; j += dirX)
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

void removeFromList(std::vector<int> &in,
  std::vector<int> const &toRemove)
  {
  auto current = in.begin();
  auto last = in.begin();
  auto remove = toRemove.begin();

  while (current != in.end())
    {
    bool endReached = remove == toRemove.end() || current == in.end();
    if (!endReached && *remove < *current)
      ++remove;
    else if (!endReached && *remove == *current)
      ++current;
    else
      {
      *last = *current;
      ++current;
      ++last;
      }
    }
  in.erase(last, in.end());
  }

void removeFromList(
  std::vector<std::vector<int> > &in,
  std::vector<std::vector<int> > const &toRemove)
  {
  for (auto &l: in)
    for (auto &removeList: toRemove)
      removeFromList(l, removeList);
  }

void removeFromList(
  std::vector<std::vector<int> > &in,
  std::vector<std::vector<int> const *> const &toRemove)
  {
  for (auto &l: in)
    for (auto &removeList: toRemove)
      removeFromList(l, *removeList);
  }

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
int SkewCartesianPartitioner::operator()(int x, int y, int z) const
  {
#ifdef HYMLS_TESTING
  if (!Partitioned())
    {
    Tools::Error("Partition() not yet called!", __FILE__, __LINE__);
    }
#endif
  int dir1 = npx_ + 1;
  int dir2 = npx_;
  int dir3 = 2*npx_*npy_ + npx_ + npy_;

  // which cube
  int xcube = x / sx_;
  int ycube = y / sx_;
  int zcube = z / sx_;

  // first domain in the cube
  int sd = zcube * dir3 + ycube*(dir2 + dir1) + xcube;

  // relative coordinates
  x -= xcube * sx_;
  y -= ycube * sx_;
  z -= zcube * sx_;
  
  if (y < sx_-x) // red
    {
    if (y < x) // green
      {
      if (!(z <= sx_ + y-x)) // blue
        sd += dir3;
      }
    else
      {
      if (z <= y-x) // blue
        sd += dir2;
      else
        sd += dir2+dir3;
      }
    }
  else
    {
    if (y < x) // green
      {
      if (z <= sx_ + y-x) // blue
        sd += dir1;
      else 
        sd += dir1+dir3;
      }
    else
      {
      if (z <= y-x) // blue
        sd += dir1+dir2;
      else 
        sd += dir1+dir2+dir3;
      }
    }
  return sd;
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
  int totNum2DCubes = npx_ * npy_; // number of cubes for fixed z
  int numPerLayer = 2 * totNum2DCubes + npx_ + npy_; // domains for fixed z
  int numPerRow = 2 * npx_ + 1; // domains in a row (both lattices); fixed y

  int NumMyElements = 0;
  int NumGlobalElements = (npz_ + 1) * numPerLayer;
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
  sdMap_ = Teuchos::rcp(new Epetra_Map(NumGlobalElements,
      NumMyElements, MyGlobalElements, 0, *comm_));

  delete [] MyGlobalElements;
  return 0;
  }

int SkewCartesianPartitioner::Partition(int nparts, bool repart)
  {
  HYMLS_PROF3(label_, "Partition");
  int npx, npy, npz;
  Tools::SplitBox(nx_, ny_, nz_, nparts, npx, npy, npz);
  return Partition(nx_ / npx, ny_ / npy, nz_ / npz, repart);
  }

// partition an [nx x ny x nz] grid with one DoF per node
// into nparts global subdomains.
int SkewCartesianPartitioner::Partition(int sx,int sy, int sz, bool repart)
  {
  HYMLS_PROF3(label_,"Partition (2)");

  if (sx != sy || sz != sz)
    Tools::Error("sx, sy and sz should be the same", __FILE__, __LINE__);

  sx_ = sx;
  sy_ = sy;
  sz_ = sz;

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

  int color = active_? 1: 0;

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

  template_ = getTemplate();
  groups_ = solveGroups(template_);
  splitTemplate();

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

// Builds a domain 'template' of given type at the origin. The template can
// be moved around to create the proper domain.
std::vector<std::vector<int> > SkewCartesianPartitioner::getTemplate() const
  {
  HYMLS_PROF2(label_, "getTemplate");
  // Principal directions

  // Cartesian directions
  int dirX = dof_;
  int dirY = dof_*nx_;
  int dirZ = dof_*nx_*ny_;

  // Info for each node type
  int firstNode[4] = {dof_*sx_/2 + 0 + dirY - dirY*(sx_/2+1),
                      dof_*sx_/2 + 1 - 0    - dirY*(sx_/2+1),
                      dof_*sx_/2 + 2 - dirZ - dirY*(sx_/2+1),
                      dof_*sx_/2 + 3 + dirY - dirY*(sx_/2+1)};
  int baseLength[4] = {sx_/2, sx_/2 + 1, sx_/2 + 1, sx_/2};

  std::vector<std::vector<std::vector<int> > > nodes;

  for (int type = 0; type < dof_; type++)
    {
    nodes.emplace_back(2 * sx_ + 1);

    // Get central layer
    Plane plane = buildPlane45(firstNode[type], baseLength[type], dirX, dirY, dof_, pvar_);
    if (type != 3)
      nodes[type][sx_] = plane.plane;

    std::vector<int> bottom;
    std::vector<int> top = plane.plane;

    // Used in the loop to determine which nodes to assign to each layer
    std::vector<int> rowLength;
    for (int i = 0; i < plane.ptr.size()-1; i++)
      rowLength.push_back(plane.ptr[i+1] - plane.ptr[i] - 1); // -1 ???

    std::vector<int> activePtrs;
    for (int i = 0; i < baseLength[type]; i++)
      activePtrs.push_back(i);

    std::vector<int> offset;
    for (auto i: activePtrs)
      offset.push_back(rowLength[i]);

    for (int i = 0; i < sx_; i++)
      {
      // Get indices for the layers
      auto last = top.end();
      for (int j = 0; j < activePtrs.size(); j++)
        {
        int val = plane.plane[plane.ptr[activePtrs[j]] + offset[j]];
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
          for (int j: nodes[type][i - 1])
            nodes[type][i].push_back(j + dirY + dirZ);
          for (int j: top)
            nodes[type][sx_ + 1 + i].push_back(j + (i + 1) * dirZ);
          }
        else
          {
          for (int j: bottom)
            nodes[type][i].push_back(j - (sx_ - i) * dirZ);
          for (int j: nodes[type][sx_ + i])
            nodes[type][sx_ + 1 + i].push_back(j + dirY + dirZ);
          }
        }
      else
        {
        int isPvar = type == pvar_;
        for (int j: bottom)
          nodes[type][i + isPvar].push_back(j - (sx_ - i - isPvar) * dirZ);
        for (int j: top)
          nodes[type][sx_ + 1 + i].push_back(j + (i + 1) * dirZ);
        }

      if (i < sx_ - 1)
        {
        // Update pointers and offset
        std::for_each(offset.begin(), offset.end(), [](int& d) { d--;});
        if (type == pvar_)
          {
          // pressure nodes are an exception
          if (offset[0] < 0)
            {
            activePtrs.erase(activePtrs.begin());
            activePtrs.push_back(activePtrs.back() + 1);
            offset.erase(offset.begin());
            offset.push_back(rowLength[activePtrs.back()]);
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

  nodes[2].erase(nodes[2].begin());

  nodes[3].pop_back();
  nodes[3].erase(nodes[3].begin());

  // Remove more unnecessary separators, located at second and second-last
  // layers of original template
  for (int i = 0; i < 3; i++)
    {
    nodes[i].front().erase(std::max_element(nodes[i].front().begin(), nodes[i].front().end()));
    nodes[i].back().erase(std::min_element(nodes[i].back().begin(), nodes[i].back().end()));
    }

  // Merge the template layers
  std::vector<std::vector<int> > newNodes;
  newNodes.push_back(nodes[2].front());
  nodes[2].erase(nodes[2].begin());
  for (int j = 0; j < nodes[0].size(); j++)
    {
    newNodes.emplace_back();
    for (int i = 0; i < nodes.size(); i++)
      std::copy(nodes[i][j].begin(), nodes[i][j].end(), std::back_inserter(newNodes.back()));
    std::sort(newNodes.back().begin(), newNodes.back().end());
    }

  return newNodes;
  }

std::vector<std::vector<int> > SkewCartesianPartitioner::solveGroups(
  std::vector<std::vector<int> > const &temp) const
  {
  HYMLS_PROF2(label_, "solveGroups");
  // Principal directios for domain displacements
  int dirX = dof_*sx_;
  int dirY = dof_*nx_*sx_;
  int dirZ = dof_*nx_*ny_*sx_;

  int dir1 = (dirY + dirX)/2; 
  int dir2 = (dirY - dirX)/2 + dirZ; 
  int dir3 = dirZ;

  // Create model problem
  int positions[27] = {0, -dir3, dir3, -dir2, -dir2-dir3,
                       -dir2+dir3, dir2, dir2-dir3, dir2+dir3,
                       -dir1, -dir1-dir3, -dir1+dir3, -dir1-dir2,
                       -dir1-dir2-dir3, -dir1-dir2+dir3, -dir1+dir2,
                       -dir1+dir2-dir3, -dir1+dir2+dir3, dir1,
                       dir1-dir3, dir1+dir3, dir1-dir2,
                       dir1-dir2-dir3, dir1-dir2+dir3, dir1+dir2,
                       dir1+dir2-dir3, dir1+dir2+dir3};

  // Turn the template into a list
  std::vector<int> tempList;
  for (auto &it: temp)
    for (auto &it2: it)
      tempList.push_back(it2);

  // Setup all domains in the test problem
  std::vector<std::vector<int> > domain;
  for (auto &it: positions)
    {
    domain.emplace_back(tempList);
    std::for_each(domain.back().begin(), domain.back().end(), [it](int& d) { d += it;});
    }

  // Find groups. Note that domain[0] is the domain that we want to solve for
  std::vector<std::vector<int> > groups;
  std::vector<unsigned long long> groupDomains;

  // First group is the interior
  groups.emplace_back(0);
  groupDomains.push_back(1);

  for (auto &node: domain[0])
    {
    // For each node, first check to which domains it belongs and store it as bits
    unsigned long long listOfDomains = 0;
    for (int i = 0; i < domain.size(); i++)
      {
      auto it = std::lower_bound(domain[i].begin(), domain[i].end(), node);
      if (it != domain[i].end() && *it == node)
        listOfDomains += 1 << i;
      }

    // Now check whether a group for this list was already created, and if
    // not, create it. Then add the nodes+domains to this group
    bool newGroup = true;
    for (int i = 0; i < groups.size(); i++)
      if (groupDomains[i] == listOfDomains)
        {
        newGroup = false;
        groups[i].push_back(node);
        break;
        }

    if (newGroup)
      {
      groups.emplace_back(1, node);
      groupDomains.push_back(listOfDomains);
      }
    }

  // Now separate the u, v, w and p, skip the interior
  std::vector<std::vector<std::vector<int> > > newGroups;
  for (int i = 1; i < groups.size(); i++)
    {
    auto group = groups[i];
    newGroups.emplace_back(dof_);

    for (auto &node: group)
      newGroups.back()[((node % dof_) + dof_) % dof_].push_back(node);
    }

  // Remove empty groups from newGroups and place them after
  // the interior in groups
  groups.resize(1);
  std::sort(groups[0].begin(), groups[0].end());
  for (auto &cats: newGroups)
    for (auto &group: cats)
      if (!group.empty())
        {
        std::sort(group.begin(), group.end());
        groups.push_back(group);
        }

  return groups;
  }

void SkewCartesianPartitioner::splitTemplate()
  {
  HYMLS_PROF2(label_, "splitTemplate");
  // Get top and bottom half of template
  topHalf_.resize(0);
  bottomHalf_.resize(0);
  std::copy(template_.begin(), template_.begin() + sx_, std::back_inserter(bottomHalf_));
  std::copy(template_.begin() + sx_, template_.end(), std::back_inserter(topHalf_));

  removeCols_.resize(0);
  removeCols_.emplace_back();
  int first = dof_ * sx_ / 2 - sx_ * dof_ * nx_ * ny_ - dof_ * (sx_/2+1) * nx_;
  for (int i = 0; i < sx_+2; i++)
    removeCols_.back().push_back(first + i * dof_ * nx_ * ny_);

  removeCols_.emplace_back(removeCols_[0]);
  std::for_each(removeCols_.back().begin(), removeCols_.back().end(), [](int& d) {d += 1;});
  removeCols_.emplace_back(removeCols_[0]);
  std::for_each(removeCols_.back().begin(), removeCols_.back().end(), [](int& d) {d += 2;});
  removeCols_.emplace_back(removeCols_[0]);
  std::for_each(removeCols_.back().begin(), removeCols_.back().end(),
    [this](int& d) {d += 2 + this->dof_*(this->sx_-1)*this->nx_*this->ny_
        + this->dof_*(this->sx_+1)*this->nx_;});

  removeCols_.emplace_back(removeCols_[0]);
  std::for_each(removeCols_.back().begin(), removeCols_.back().end(),
    [this](int& d) {d += 1 + this->dof_*(this->sx_ / 2) +
        this->dof_ * (this->sx_ / 2) * this->nx_;});
  removeCols_.emplace_back(removeCols_[4]);
  std::for_each(removeCols_.back().begin(), removeCols_.back().end(), [](int& d) {d += 1;});
  removeCols_.emplace_back(removeCols_[5]);
  std::for_each(removeCols_.back().begin(), removeCols_.back().end(),
    [this](int& d) {d += this->dof_ * this->nx_;});

  std::vector<std::vector<int> > NSintersect(1);
  std::vector<std::vector<int> > EWintersect(1);
  for (int type = 0; type < dof_; type++)
    for (int jj = 0; jj < sx_; jj++)
      {
      // South
      for (int i = 0; i < sx_/2+1; i++)
        for (int j = 0; j < sx_ + 1; j++)
          NSintersect[0].push_back(dof_ * j + i * dof_ * nx_ - sx_ * dof_ * nx_ * ny_
            - dof_ * (sx_/2+1) * nx_ + jj * dof_ * nx_ * ny_ + type);
      // West
      for (int i = 0; i < sx_+2; i++)
        for (int j = 0; j < sx_/2; j++)
          EWintersect[0].push_back(dof_ * j + i * dof_ * nx_ - sx_ * dof_ * nx_ * ny_
            - dof_ * (sx_/2+1) * nx_ + jj * dof_ * nx_ * ny_ + type);
      }

  std::sort(NSintersect[0].begin(), NSintersect[0].end());
  std::sort(EWintersect[0].begin(), EWintersect[0].end());

  north_.resize(0);
  west_.resize(0);
  south_.resize(0);
  east_.resize(0);

  north_.emplace_back(bottomHalf_);
  removeFromList(north_[0], NSintersect);
  south_.emplace_back(bottomHalf_);
  removeFromList(south_[0], north_[0]);
  east_.emplace_back(bottomHalf_);
  removeFromList(east_[0], EWintersect);
  west_.emplace_back(bottomHalf_);
  removeFromList(west_[0], east_[0]);

  std::for_each(NSintersect[0].begin(), NSintersect[0].end(),
    [this](int& d) {d += this->sx_ * this->dof_ * this->nx_ * this->ny_;});
  std::for_each(EWintersect[0].begin(), EWintersect[0].end(),
    [this](int& d) {d += this->sx_ * this->dof_ * this->nx_ * this->ny_;});

  north_.emplace_back(topHalf_);
  removeFromList(north_[1], NSintersect);
  south_.emplace_back(topHalf_);
  removeFromList(south_[1], north_[1]);
  east_.emplace_back(topHalf_);
  removeFromList(east_[1], EWintersect);
  west_.emplace_back(topHalf_);
  removeFromList(west_[1], east_[1]);
  }

std::vector<std::vector<int> > SkewCartesianPartitioner::createSubdomain(int sd,
  std::vector<std::vector<int> > temp,
  std::vector<std::vector<int> > groups) const
  {
  HYMLS_PROF2(label_, "createSubdomain");

  int totNum2DCubes = npx_ * npy_; // number of cubes for fixed z
  int numPerLayer = 2 * totNum2DCubes + npx_ + npy_; // domains for fixed z
  int numPerRow = 2*npx_ + 1; // domains in a row (both lattices); fixed y

  // Get domain coordinates and its first node
  // Considers 'superposed lattices
  int Z = sd / numPerLayer;
  double Y = ((sd - Z * numPerLayer) / numPerRow) - 0.5;
  double X = (sd - Z * numPerLayer) % numPerRow;
  int lattice = 1;
  if (X >= npx_)
    {
    X -= npx_ + 0.5;
    Y += 0.5;
    lattice = 2;
    }

  int firstNode = dof_ * sx_ * (X + Y * nx_) + dof_ * nx_ * (sx_/2) + dof_ * nx_ * ny_ * sx_ * Z;

  double eps = 1e-8;
  std::vector<std::vector<int> const *> toRemove;
  if (lattice == 1)
    {
    if (std::abs(Y + 0.5) < eps)
      {
      if (Z == 0)
        {
        std::for_each(bottomHalf_.begin(), bottomHalf_.end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        std::for_each(south_[1].begin(), south_[1].end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        }
      else if (Z == npz_)
        {
        std::for_each(topHalf_.begin(), topHalf_.end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        std::for_each(south_[0].begin(), south_[0].end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        }
      else
        {
        std::for_each(south_[0].begin(), south_[0].end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        std::for_each(south_[1].begin(), south_[1].end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        }
      }
    else if (std::abs(Y - npy_ + 0.5) < eps)
      {
      if (Z == 0)
        {
        std::for_each(bottomHalf_.begin(), bottomHalf_.end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        std::for_each(north_[1].begin(), north_[1].end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        }
      else if (Z == npz_)
        {
        std::for_each(topHalf_.begin(), topHalf_.end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        std::for_each(north_[0].begin(), north_[0].end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        }
      else
        {
        std::for_each(north_[0].begin(), north_[0].end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        std::for_each(north_[1].begin(), north_[1].end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        }
      }
    else
      {
      if (Z == 0)
        {
        std::for_each(bottomHalf_.begin(), bottomHalf_.end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        }
      else if (Z == npz_)
        {
        std::for_each(topHalf_.begin(), topHalf_.end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        }
      }

    if (std::abs(X - npx_ + 1) < eps)
      {
      toRemove.push_back(&removeCols_[4]);
      toRemove.push_back(&removeCols_[5]);
      toRemove.push_back(&removeCols_[6]);
      }
    }
  else if (lattice == 2)
    {
    if (std::abs(X + 0.5) < eps)
      {
      if (Z == 0)
        {
        std::for_each(bottomHalf_.begin(), bottomHalf_.end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        std::for_each(west_[1].begin(), west_[1].end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        }
      else if (Z == npz_)
        {
        std::for_each(topHalf_.begin(), topHalf_.end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        std::for_each(west_[0].begin(), west_[0].end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        }
      else
        {
        std::for_each(west_[0].begin(), west_[0].end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        std::for_each(west_[1].begin(), west_[1].end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        }
      }
    else if (std::abs(X - npx_ + 0.5) < eps)
      {
      if (Z == 0)
        {
        std::for_each(bottomHalf_.begin(), bottomHalf_.end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        std::for_each(east_[1].begin(), east_[1].end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        }
      else if (Z == npz_)
        {
        std::for_each(topHalf_.begin(), topHalf_.end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        std::for_each(east_[0].begin(), east_[0].end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        }
      else
        {
        std::for_each(east_[0].begin(), east_[0].end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        std::for_each(east_[1].begin(), east_[1].end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        }
      }
    else
      {
      if (Z == 0)
        {
        std::for_each(bottomHalf_.begin(), bottomHalf_.end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        }
      else if (Z == npz_)
        {
        std::for_each(topHalf_.begin(), topHalf_.end(),
          [&toRemove](std::vector<int> const &d) {toRemove.push_back(&d);});
        }
      }

    if (std::abs(Y) < eps)
      {
      toRemove.push_back(&removeCols_[1]);
      toRemove.push_back(&removeCols_[2]);
      }
    else if (std::abs(Y - npy_ + 1) < eps)
      {
      toRemove.push_back(&removeCols_[3]);
      }
    }

  // Get all groups, and remove the entries that are on removeList
  removeFromList(groups, toRemove);

  // Remove empty groups
  groups.erase(std::remove_if(groups.begin(), groups.end(),
      [](std::vector<int> &i){return i.empty();}), groups.end());

  // Move the groups to the right position
  for (int i = 0; i < groups.size(); i++)
    std::for_each(groups[i].begin(), groups[i].end(),
      [firstNode](int& d) { d += firstNode;});

  // Get first pressure node from interior to a new group.
  // Assumes ordering of groups by size!
  for (int &node: groups[0])
    if (((node % dof_) + dof_) % dof_ == pvar_)
      {
      groups.emplace_back(1, node);
      groups[0].erase(std::remove(groups[0].begin(), groups[0].end(), node), groups[0].end());
      break;
      }

  // Split separator groups that that do not belong to the same subdomain.
  // This may happen for the w-groups since the w-separators are staggered
  std::vector<std::vector<int> > oldGroups;
  std::copy(groups.begin() + 1, groups.end(), std::back_inserter(oldGroups));
  groups.resize(1);
  for (auto &group: oldGroups)
    {
    std::map<int, std::vector<int> > newGroups;
    for (int node: group)
      {
      int gsd = operator()(node);
      auto newGroup = newGroups.find(gsd);
      if (newGroup != newGroups.end())
        newGroup->second.push_back(node);
      else
        newGroups.emplace(gsd, std::vector<int>(1, node));
      }
    for (auto &newGroup: newGroups)
      groups.push_back(newGroup.second);
    }

  // Remove separator nodes that lie on the boundary of the domain.
  // We need this because those nodes don't actually border any interior
  // nodes of the other subdomain
  for (auto group = groups.begin() + 1; group != groups.end(); ++group)
    {
    std::vector<int> groupCopy = *group;
    for (int node: groupCopy)
      {
      int x = (node / dof_) % nx_;
      int y = (node / dof_ / nx_) % ny_;
      int z = node / dof_ / nx_ / ny_;
      if (x == nx_ - 1 && node % dof_ == 0)
        {
        if (operator()(x, y, z) == sd)
          groups[0].push_back(node);
        group->erase(std::remove(group->begin(), group->end(), node));
        }
      else if (y == ny_ - 1 && node % dof_ == 1)
        {
        if (operator()(x, y, z) == sd)
          groups[0].push_back(node);
        group->erase(std::remove(group->begin(), group->end(), node));
        }
      else if (z == nz_ - 1 && node % dof_ == 2)
        {
        if (operator()(x, y, z) == sd)
          groups[0].push_back(node);
        group->erase(std::remove(group->begin(), group->end(), node));
        }
      }
    }

  // Remove empty groups
  groups.erase(std::remove_if(groups.begin(), groups.end(),
      [](std::vector<int> &i){return i.empty();}), groups.end());

  return groups;
  }

int SkewCartesianPartitioner::GetGroups(int sd, Teuchos::Array<int> &interior_nodes,
  Teuchos::Array<Teuchos::Array<int> > &separator_nodes)
  {
  HYMLS_PROF2(label_,"GetGroups");

  int gsd = sdMap_->GID(sd);

  std::vector<std::vector<int> > nodes = createSubdomain(gsd, template_, groups_);
  interior_nodes = nodes[0];
  std::copy(nodes.begin() + 1, nodes.end(), std::back_inserter(separator_nodes));

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
  #ifdef HYMLS_TESTING
  if (!Partitioned())
    {
    Tools::Error("Partition() not yet called!", __FILE__, __LINE__);
    }
#endif
  int dir1 = nprocx_ + 1;
  int dir2 = nprocx_;
  int dir3 = 2*nprocx_*nprocy_ + nprocx_ + nprocy_;

  int sx = nx_ / nprocx_;
  int sy = ny_ / nprocy_;
  int sz = nz_ / nprocz_;

  int cl = std::max(sx, sy);
  cl = std::max(cl, sz);

  // which cube
  int xcube = i / cl;
  int ycube = j / cl;
  int zcube = k / cl;

  // first domain in the cube
  int sd = zcube * dir3 + ycube * (dir2 + dir1) + xcube;

  // relative coordinates
  i -= xcube * cl;
  j -= ycube * cl;
  k -= zcube * cl;

  if (j < cl - i) // red
    {
    if (j < i) // green
      {
      if (!(k <= cl + j - i)) // blue
        sd += dir3;
      }
    else
      {
      if (k <= j - i) // blue
        sd += dir2;
      else
        sd += dir2+dir3;
      }
    }
  else
    {
    if (j < i) // green
      {
      if (k <= cl + j - i) // blue
        sd += dir1;
      else 
        sd += dir1+dir3;
      }
    else
      {
      if (k <= j - i) // blue
        sd += dir1+dir2;
      else 
        sd += dir1+dir2+dir3;
      }
    }

  int totNum2DCubes = nprocx_ * nprocy_; // number of cubes for fixed z
  int numPerLayer = 2 * totNum2DCubes + nprocx_ + nprocy_; // domains for fixed z
  int numPerRow = 2 * nprocx_ + 1; // domains in a row (both lattices); fixed y

  int Z = sd / numPerLayer;
  double Y = ((sd - Z * numPerLayer) / numPerRow) - 0.5;
  double X = (sd - Z * numPerLayer) % numPerRow;
  if (X >= nprocx_)
    {
    X -= nprocx_ + 0.5;
    Y += 0.5;
    }

  int firstNode = cl * (X + Y * nx_) + nx_ * (cl/2) + nx_ * ny_ * cl * Z;
  i = ((firstNode % nx_) + nx_) % nx_;
  j = (((firstNode / nx_) % ny_) + ny_) % ny_;
  k = (((firstNode / nx_ / ny_) % nz_) + nz_) % nz_;

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
