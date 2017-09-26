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
  if (dof > 1 && ((firstNode % dof) + dof) % dof  == 0)
    {
    left -= dirX;
    height++;
    extraLayer = true;
    }
  else if (dof > 1 && pvar != -1 && ((firstNode % dof) + dof) % dof  == pvar)
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

  if (baseMap_->IndexBase64() != 0)
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
  x -= xcube * sx_ - 1;
  y -= ycube * sx_;
  z -= zcube * sx_;

  bool front = y < sx_-x; // In front of the red plane
  bool right = y < x; // Right of the green plane;
  bool below = z <= y-x; // Below the blue plane left of the green plane
  if (right) below = z <= sx_ + y-x; // Below the blue plane right of the green plane

  if (!front)
    sd += dir1;
  if (!right)
    sd += dir2;
  if (!below)
    sd += dir3;

  if (!front && right && perio_ & GaleriExt::X_PERIO && xcube == npx_-1)
    sd -= dir2;

  if (!front && !right && perio_ & GaleriExt::Y_PERIO && ycube == npy_-1)
    sd -= dir3 - dir2;

  if (!below && perio_ & GaleriExt::Z_PERIO && zcube == npz_-1)
    sd -= npz_ * dir3;

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

    i = (((i + sx_ / 2) % nx_) + nx_) % nx_;
    j = (((j + sx_ / 2) % ny_) + ny_) % ny_;
    k = (((k + sx_) % nz_) + nz_) % nz_;
    int pid = PID(i, j, k);
    if (pid == comm_->MyPID())
      MyGlobalElements[NumMyElements++] = sd;
    }

  sdMap_ = Teuchos::rcp(new Epetra_Map(-1,
      NumMyElements, MyGlobalElements, 0, *comm_));

  delete[] MyGlobalElements;
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

  if (sx != sy || (nz_ > 1 && sx != sz))
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

    // Determine which gids belong to subdomains on this processor
    // by looping over the cubes in which they exist
    int numMyElements = numLocalSubdomains_ * sx_ * sy_ * sz_ * dof_;
    hymls_gidx *myGlobalElements = new hymls_gidx[numMyElements];
    int pos = 0;
    for (int sd = 0; sd < numLocalSubdomains_; sd++)
      {
      int gsd = sdMap_->GID(sd);
      int x, y, z;
      GetSubdomainPosition(gsd, sx_, x, y, z);

      // Determine unique indices so we don't try to add the same node
      // multiple times
      std::vector<int> xindices;
      for (int i = x; i < x + sx_; i++)
        xindices.push_back(((i % nx_) + nx_) % nx_);
      std::sort(xindices.begin(), xindices.end());
      xindices.erase(std::unique(xindices.begin(), xindices.end()), xindices.end());

      std::vector<int> yindices;
      for (int j = y; j < y + sy_; j++)
        yindices.push_back(((j % ny_) + ny_) % ny_);
      std::sort(yindices.begin(), yindices.end());
      yindices.erase(std::unique(yindices.begin(), yindices.end()), yindices.end());

      std::vector<int> zindices;
      for (int k = z; k < z + 2 * sz_; k++)
        zindices.push_back(((k % nz_) + nz_) % nz_);
      std::sort(zindices.begin(), zindices.end());
      zindices.erase(std::unique(zindices.begin(), zindices.end()), zindices.end());

      for (int k: zindices)
        for (int j: yindices)
          for(int i: xindices)
            for (int var = 0; var < dof_; var++)
              {
              hymls_gidx gid = Tools::sub2ind(nx_, ny_, nz_, dof_, i, j, k, var);
              if (sdMap_->LID((*this)(gid)) == sd)
                {
                if (pos >= numMyElements)
                  {
                  Tools::Error("Index out of range", __FILE__, __LINE__);
                  }
                myGlobalElements[pos++] = gid;
                }
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
std::vector<std::vector<hymls_gidx> > SkewCartesianPartitioner::getTemplate() const
  {
  HYMLS_PROF2(label_, "getTemplate");
  // Principal directions

  hymls_gidx nx = sx_ * 4;

  // Cartesian directions
  hymls_gidx dirX = dof_;
  hymls_gidx dirY = dof_*nx;
  hymls_gidx dirZ = dof_*nx*nx;

  // Info for each node type
  hymls_gidx firstNode[4] = {dof_*sx_/2 + 0 + dirY + dirZ * sx_,
                             dof_*sx_/2 + 1 - 0    + dirZ * sx_,
                             dof_*sx_/2 + 2 - dirZ + dirZ * sx_,
                             dof_*sx_/2 + pvar_ + dirY + dirZ * sx_};
  hymls_gidx baseLength[4] = {sx_/2, sx_/2 + 1, sx_/2 + 1, sx_/2};

  std::vector<std::vector<std::vector<hymls_gidx> > > nodes;

  for (int type = 0; type < 4; type++)
    {
    nodes.emplace_back(2 * sx_ + 1);

    // Get central layer
    Plane plane = buildPlane45(firstNode[type], baseLength[type], dirX, dirY, dof_, pvar_);
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
          for (hymls_gidx j: nodes[type][i - 1])
            nodes[type][i].push_back(j + dirY + dirZ);
          for (hymls_gidx j: top)
            nodes[type][sx_ + 1 + i].push_back(j + (i + 1) * dirZ);
          }
        else
          {
          for (hymls_gidx j: bottom)
            nodes[type][i].push_back(j - (sx_ - i) * dirZ);
          for (hymls_gidx j: nodes[type][sx_ + i])
            nodes[type][sx_ + 1 + i].push_back(j + dirY + dirZ);
          }
        }
      else
        {
        hymls_gidx isPvar = type == pvar_;
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
  std::vector<std::vector<hymls_gidx> > newNodes;
  if (nz_ > 1 && dof_ > 1)
    {
    newNodes.push_back(nodes[2].front());
    nodes[2].erase(nodes[2].begin());
    }
  else
    newNodes.emplace_back();

  // TODO: Make users able to choose which ones to use from the parameter list
  for (int j = 0; j < 2 * sx_ - 1; j++)
    {
    newNodes.emplace_back();
    if (dof_ > 1)
      std::copy(nodes[0][j].begin(), nodes[0][j].end(), std::back_inserter(newNodes.back()));
    if (nz_ > 1 && dof_ > 1)
      std::copy(nodes[2][j].begin(), nodes[2][j].end(), std::back_inserter(newNodes.back()));
    if (pvar_ != -1)
      std::copy(nodes[3][j].begin(), nodes[3][j].end(), std::back_inserter(newNodes.back()));

    // All other nodes are v-type nodes
    for (int i = 0; i < dof_; i++)
      {
      if (i == 0 && dof_ > 1)
        continue;
      if (i == 2 && dof_ > 1 && nz_ > 1)
        continue;
      if (i == pvar_)
        continue;
      int size = newNodes.back().size();
      std::copy(nodes[1][j].begin(), nodes[1][j].end(), std::back_inserter(newNodes.back()));
      std::for_each(newNodes.back().begin()+size, newNodes.back().end(),
        [i](hymls_gidx& d) { d += i-1;});
      }
    std::sort(newNodes.back().begin(), newNodes.back().end());
    }

  return newNodes;
  }

std::vector<std::vector<hymls_gidx> > SkewCartesianPartitioner::solveGroups(
  std::vector<std::vector<hymls_gidx> > const &temp) const
  {
  HYMLS_PROF2(label_, "solveGroups");

  hymls_gidx nx = sx_ * 4;

  // Principal directions for domain displacements
  hymls_gidx dirX = dof_*sx_;
  hymls_gidx dirY = dof_*nx*sx_;
  hymls_gidx dirZ = dof_*nx*nx*sx_;

  // Shift the central domain by 1,1,1
  hymls_gidx first = dirX + dirY + dirZ;

  hymls_gidx dir1 = (dirY + dirX)/2; 
  hymls_gidx dir2 = (dirY - dirX)/2 + dirZ; 
  hymls_gidx dir3 = dirZ;

  // Create model problem
  hymls_gidx positions[27] = {0, -dir3, dir3, -dir2, -dir2-dir3,
                              -dir2+dir3, dir2, dir2-dir3, dir2+dir3,
                              -dir1, -dir1-dir3, -dir1+dir3, -dir1-dir2,
                              -dir1-dir2-dir3, -dir1-dir2+dir3, -dir1+dir2,
                              -dir1+dir2-dir3, -dir1+dir2+dir3, dir1,
                              dir1-dir3, dir1+dir3, dir1-dir2,
                              dir1-dir2-dir3, dir1-dir2+dir3, dir1+dir2,
                              dir1+dir2-dir3, dir1+dir2+dir3};

  // Turn the template into a list
  std::vector<hymls_gidx> tempList;
  for (auto &it: temp)
    for (auto &it2: it)
      tempList.push_back(it2);

  // Setup all domains in the test problem
  std::vector<std::vector<hymls_gidx> > domain;
  for (auto &it: positions)
    {
    domain.emplace_back(tempList);
    std::for_each(domain.back().begin(), domain.back().end(),
      [it, first](hymls_gidx& d) { d += it + first;});
    }

  // Find groups. Note that domain[0] is the domain that we want to solve for
  std::vector<std::vector<hymls_gidx> > groups;
  std::vector<unsigned long long> groupDomains;

  // First group is the interior
  groups.emplace_back(0);
  groupDomains.push_back(1);

  for (auto &node: domain[0])
    {
    // For each node, first check to which domains it belongs and store it as bits
    unsigned long long listOfDomains = 0;
    for (size_t i = 0; i < domain.size(); i++)
      {
      auto it = std::lower_bound(domain[i].begin(), domain[i].end(), node);
      if (it != domain[i].end() && *it == node)
        listOfDomains += 1 << i;
      }

    // Now check whether a group for this list was already created, and if
    // not, create it. Then add the nodes+domains to this group
    bool newGroup = true;
    for (size_t i = 0; i < groups.size(); i++)
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
  std::vector<std::vector<std::vector<hymls_gidx> > > newGroups;
  for (size_t i = 1; i < groups.size(); i++)
    {
    auto group = groups[i];
    newGroups.emplace_back(dof_);

    for (auto &node: group)
      newGroups.back()[((node % dof_) + dof_) % dof_].push_back(node);
    }

  // Remove empty groups from newGroups and place them after
  // the interior in groups
  groups.resize(1);
  for (auto &cats: newGroups)
    for (auto &group: cats)
      if (!group.empty())
        {
        std::sort(group.begin(), group.end());
        groups.push_back(group);
        }

  return groups;
  }

std::vector<std::vector<hymls_gidx> > SkewCartesianPartitioner::createSubdomain(int sd,
  std::vector<std::vector<hymls_gidx> > groups) const
  {
  HYMLS_PROF2(label_, "createSubdomain");

  int sdx, sdy, sdz;
  GetSubdomainPosition(sd, sx_, sdx, sdy, sdz);

  int nx = 4 * sx_;

  // Move the groups to the right position and cut off parts that fall
  // outside of the domain
  std::vector<std::vector<hymls_gidx> > newGroups;
  for (auto &group: groups)
    {
    newGroups.emplace_back();
    for (hymls_gidx &node: group)
      {
      int var = node % dof_;
      int x = (node / dof_) % nx + sdx - 1 - sx_;
      int y = (node / dof_ / nx) % nx + sdy - 1 - sx_;
      int z = node / dof_ / nx / nx + sdz - sx_;
      if (perio_ & GaleriExt::X_PERIO) x = (x + nx_) % nx_;
      if (perio_ & GaleriExt::Y_PERIO) y = (y + ny_) % ny_;
      if (perio_ & GaleriExt::Z_PERIO) z = (z + nz_) % nz_;
      if (x >= 0 && x < nx_ && y >= 0 && y < ny_ && z >= 0 && z < nz_)
        newGroups.back().push_back(x * dof_ + nx_ * y * dof_ + nx_ * ny_ * z * dof_ + var);
      }
    }
  groups = newGroups;

  // Remove empty groups
  groups.erase(std::remove_if(groups.begin()+1, groups.end(),
      [](std::vector<hymls_gidx> &i){return i.empty();}), groups.end());

  // Get first pressure node from interior to a new group.
  // Assumes ordering of groups by size!
  for (hymls_gidx &node: groups[0])
    if (((node % dof_) + dof_) % dof_ == pvar_)
      {
      groups.emplace_back(1, node);
      groups[0].erase(std::remove(groups[0].begin(), groups[0].end(), node), groups[0].end());
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
      if (dof_ > 1 && x == nx_ - 1 && node % dof_ == 0 &&
        !(perio_ & GaleriExt::X_PERIO))
        {
        if (operator()(x, y, z) == sd)
          groups[0].push_back(node);
        group->erase(std::remove(group->begin(), group->end(), node));
        }
      else if (y == ny_ - 1 && node % dof_ == 1 &&
        !(perio_ & GaleriExt::Y_PERIO))
        {
        if (operator()(x, y, z) == sd)
          groups[0].push_back(node);
        group->erase(std::remove(group->begin(), group->end(), node));
        }
      else if (nz_ > 1 && z == nz_ - 1 && node % dof_ == 2 &&
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
  HYMLS_PROF2(label_,"GetGroups");

  int gsd = sdMap_->GID(sd);

  std::vector<std::vector<hymls_gidx> > nodes = createSubdomain(gsd, groups_);
  interior_nodes = nodes[0];
  std::copy(nodes.begin() + 1, nodes.end(), std::back_inserter(separator_nodes));

  return 0;
  }

//! get the type of a variable (if more than 1 dof per node, otherwise just 0)
int SkewCartesianPartitioner::VariableType(hymls_gidx gid) const
  {
  return gid % dof_;
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

  int npx = nx_ / cl;
  int npy = ny_ / cl;

  int dir1 = npx + 1;
  int dir2 = npx;
  int dir3 = 2*npx*npy + npx + npy;

  // which cube
  int xcube = i / cl;
  int ycube = j / cl;
  int zcube = k / cl;

  // first domain in the cube
  int sd = zcube * dir3 + ycube * (dir2 + dir1) + xcube;

  // relative coordinates
  i -= xcube * cl - 1;
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

  GetSubdomainPosition(sd, cl, i, j, k);
  i = ((i - 1) % nx_ + nx_) % nx_;
  j = ((j - 1) % ny_ + ny_) % ny_;
  k = (k % nz_ + nz_) % nz_;

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
