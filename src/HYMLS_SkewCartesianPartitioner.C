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
  int x = i / sx_;
  int y = j / sx_;
  int z = k / sx_;

  // Get position within the cube
  i = i - sx_ * x;
  j = j - sx_ * y;
  k = k - sx_ * z;

  int totDomsPerLayer = (npx_ * npy_ + npx_ * (npy_+1) + (npx_+1) * npy_);

  int sd = z * totDomsPerLayer + y * npx_ + x;

  // Find type of domain; based on cube
  if (i <= sx_/2 - 2 &&
    j >= i + 1 && j <= sx_-(i+2) &&
    k >= i + 1 && k <= sx_-(i+2))
    sd += npx_*npy_ + npx_ + y*(npx_+1)*(y>0);
  else if (i >= sx_/2 &&
    j <= i && j >= sx_-(i+1) &&
    k <= i && k >= sx_-(i+1))
    sd += npx_*npy_ + npx_ + y*(npx_+1) + 1;
  else if (j <= sx_/2 - 2 &&
    i >= j && i <= sx_-(j+2) &&
    k >= j + 1 && k <= sx_-(j+2)) 
    sd += npx_*npy_ + y*(npx_+1)*(y>0);
  else if (j >= sx_/2 &&
    i <= j - 1 && i >= sx_-(j+1) &&
    k <= j && k >= sx_-(j+1))
    sd += npx_*npy_ + y*(npx_+1) + 2*npx_ + 1;
  else if (k >= sx_/2)
    sd += npy_*(3*npx_+1) + npx_;
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
  int NumMyElements = 0;
  int NumGlobalElements = 3 * npx_ * npy_ * npz_ + npx_ * npy_ + npy_ * npz_ + npz_ * npx_;
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

  templateX_ = domainTemplate('x');
  templateY_ = domainTemplate('y');
  templateZ_ = domainTemplate('z');

  int directions[3] = {1, nx_, nx_*ny_};
  groupsX_ = solveGroups('x', templateX_, templateY_, templateZ_, directions);
  groupsY_ = solveGroups('y', templateX_, templateY_, templateZ_, directions);
  groupsZ_ = solveGroups('z', templateX_, templateY_, templateZ_, directions);

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


// Builds m by n matrix with specified coefficients (1,1), (1,n) and (m,1)
// elements and 'gaps' colskip and rowskip;
std::vector<std::vector<int> > SkewCartesianPartitioner::buildMatrix(int c_11,
  int c_1n, int c_m1, int colskip, int rowskip, int offset, int var) const
  {
  std::vector<std::vector<int> > matrix;
  int numrows = (c_m1 - c_11) / rowskip + 1;
  int numcols = (c_1n - c_11) / colskip + 1;
  for (int i = 0; i < numrows; i++)
    {
    matrix.emplace_back(numcols, offset * dof_ + var);
    for (int j = 0; j < numcols; j++)
      matrix[i][j] += (c_11 + i * rowskip + j * colskip) * dof_;
    }
  return matrix;
  }


int SkewCartesianPartitioner::removeOverlappingVNodes(
  std::vector<std::vector<int> > &plane, int type) const
  {
  if (type == 0)
    {
    plane.front().erase(plane.front().begin(), plane.front().begin() + plane.front().size() / 2);
    plane.back().erase(plane.back().begin(), plane.back().begin() + plane.back().size() / 2);
    }
  else if (type == 2)
    {
    for (int i = plane.size() / 2; i < plane.size(); i++)
      {
      plane[i].pop_back();
      plane[i].erase(plane[i].begin());
      }
    }
  return 0;
  }

// Builds a domain 'template' of given type at the origin. The template can
// be moved around to create the proper domain.
std::vector<std::vector<std::vector<std::vector<int> > > >
SkewCartesianPartitioner::domainTemplate(char type) const
  {
  int center = sx_ / 2 + 1;
  std::vector<std::vector<std::vector<std::vector<int> > > > nodes(dof_);

  switch (type)
    {
    case 'x':
    {
      //Directions
      int dir1 = nx_;
      int dir2 = nx_ * ny_;
      int dir3 = 1;

      // Baseplane offset
      int offset = -(dir1 + dir2 + dir3);

      // u nodes
      for (int i = 1; i < center; i++)
        nodes[0].insert(nodes[0].begin(), buildMatrix(i*dir1+i*dir2,
            (sx_-i+1)*dir1+i*dir2, (sx_-i+1)*dir2+i*dir1,
            dir1, dir2, offset-i*dir3, 0));

      for (int i = 1; i < center; i++)
        nodes[0].push_back(buildMatrix(i*dir1+i*dir2,
            (sx_-i+1)*dir1+i*dir2, (sx_-i+1)*dir2+i*dir1,
            dir1, dir2, offset+(i-1)*dir3, 0));

      // v nodes
      for (int i = 1; i < center; i++)
        {
        nodes[1].insert(nodes[1].begin(), buildMatrix(i*dir1+i*dir2,
            (sx_-i)*dir1+i*dir2, (sx_-i+1)*dir2+i*dir1,
            dir1, dir2, offset-i*dir3, 1));
        removeOverlappingVNodes(nodes[1].front(), 0);
        }

      nodes[1].push_back(buildMatrix(dir2,
          sx_*dir1+dir2, sx_*dir2,
          dir1, dir2, offset, 1));

      for (int i = 1; i < center; i++)
        {
        nodes[1].push_back(buildMatrix(i*dir1+i*dir2,
            (sx_-i)*dir1+i*dir2, (sx_-i+1)*dir2+i*dir1,
            dir1, dir2, offset+i*dir3, 1));
        removeOverlappingVNodes(nodes[1].back(), 0);
        }

      // w nodes
      for (int i = 1; i < center; i++)
        nodes[2].insert(nodes[2].begin(), buildMatrix(i*dir1+i*dir2,
            (sx_-i)*dir1+i*dir2, (sx_-i)*dir2+i*dir1,
            dir1, dir2, offset-i*dir3, 2));

      nodes[2].push_back(buildMatrix(0,
          sx_*dir1, sx_*dir2,
          dir1, dir2, offset, 2));

      for (int i = 1; i < center; i++)
        nodes[2].push_back(buildMatrix(i*dir1+i*dir2,
            (sx_-i)*dir1+i*dir2, (sx_-i)*dir2+i*dir1,
            dir1, dir2, offset+i*dir3, 2));

      // p nodes
      for (int i = 1; i < center; i++)
        nodes[3].insert(nodes[3].begin(), buildMatrix(i*dir1+i*dir2,
            (sx_-i+1)*dir1+i*dir2, (sx_-i+1)*dir2+i*dir1,
            dir1, dir2, offset-(i-1)*dir3, 3));

      for (int i = 2; i < center; i++)
        nodes[3].push_back(buildMatrix(i*dir1+i*dir2,
            (sx_-i+1)*dir1+i*dir2, (sx_-i+1)*dir2+i*dir1,
            dir1, dir2, offset+(i-1)*dir3, 3));
    }
    break;
    case 'y':
    {
      // Directions
      int dir1 = 1;
      int dir2 = nx_ * ny_;
      int dir3 = nx_;
    
      // Baseplane offset
      int offset = -(dir1 + dir2 + dir3);

      // u nodes
      for (int i = 1; i < center; i++)
        nodes[0].insert(nodes[0].begin(), buildMatrix((i-1)*dir1+i*dir2,
            (sx_-i)*dir1+i*dir2, (sx_-i+1)*dir2+(i-1)*dir1,
            dir1, dir2, offset-(i-1)*dir3, 0));

      for (int i = 1; i < center; i++)
        nodes[0].push_back(buildMatrix((i-1)*dir1+i*dir2,
            (sx_-i)*dir1+i*dir2, (sx_-i+1)*dir2+(i-1)*dir1,
            dir1, dir2, offset+i*dir3, 0));

      // v nodes
      for (int i = 1; i < center; i++)
        nodes[1].insert(nodes[1].begin(), buildMatrix(i*dir1+i*dir2,
            (sx_-i)*dir1+i*dir2,(sx_-i+1)*dir2+i*dir1,
            dir1, dir2, offset-i*dir3, 1));

      for (int i = 1; i < center; i++)
        nodes[1].push_back(buildMatrix((i-1)*dir1+i*dir2,
            (sx_-i+1)*dir1+i*dir2, (sx_-i+1)*dir2+(i-1)*dir1,
            dir1, dir2, offset+(i-1)*dir3, 1));

      // w nodes
      for (int i = 1; i < center; i++)
        nodes[2].insert(nodes[2].begin(), buildMatrix(i*dir1+i*dir2,
            (sx_-i)*dir1+i*dir2, (sx_-i)*dir2+i*dir1,
            dir1, dir2, offset-i*dir3, 2));

      nodes[2].push_back(buildMatrix(0,
          sx_*dir1, sx_*dir2,
          dir1, dir2, offset, 2));

      for (int i = 1; i < center; i++)
        nodes[2].push_back(buildMatrix(i*dir1+i*dir2,
            (sx_-i)*dir1+i*dir2, (sx_-i)*dir2+i*dir1,
            dir1, dir2, offset+i*dir3, 2));

      // p nodes
      for (int i = 1; i < center; i++)
        nodes[3].insert(nodes[3].begin(), buildMatrix(i*dir1+i*dir2,
            (sx_-i)*dir1+i*dir2, (sx_-i+1)*dir2+i*dir1,
            dir1, dir2, offset-(i-1)*dir3, 3));

      for (int i = 2; i < center; i++)
        nodes[3].push_back(buildMatrix((i-1)*dir1+i*dir2,
            (sx_-i+1)*dir1+i*dir2, (sx_-i+1)*dir2+(i-1)*dir1,
            dir1, dir2, offset+(i-1)*dir3, 3));
    }
    break;
    case 'z':
    {
      // Directions
      int dir1 = 1;
      int dir2 = nx_;
      int dir3 = nx_ * ny_;
      
      // Baseplane
      int offset = -(dir1 + dir2 + dir3);

      // u nodes
      for (int i = 1; i < center; i++)
        nodes[0].insert(nodes[0].begin(), buildMatrix((i-1)*dir1+i*dir2,
            (sx_-i)*dir1+i*dir2, (sx_-i+1)*dir2+(i-1)*dir1,
            dir1, dir2, offset-(i-1)*dir3, 0));

      for (int i = 1; i < center; i++)
        nodes[0].push_back(buildMatrix((i-1)*dir1+i*dir2,
            (sx_-i)*dir1+i*dir2, (sx_-i+1)*dir2+(i-1)*dir1,
            dir1, dir2, offset+i*dir3, 0));

      // v nodes
      for (int i = 1; i < center; i++)
        {
        nodes[1].insert(nodes[1].begin(), buildMatrix((i-1)*dir1+(i-1)*dir2,
            (sx_-i+1)*dir1+(i-1)*dir2, (sx_-i)*dir2+(i-1)*dir1,
            dir1, dir2, offset-(i-1)*dir3, 1));
        removeOverlappingVNodes(nodes[1].front(), 2);
        }

      for (int i = 1; i < center; i++)
        {
        nodes[1].push_back(buildMatrix((i-1)*dir1+(i-1)*dir2,
            (sx_-i+1)*dir1+(i-1)*dir2, (sx_-i)*dir2+(i-1)*dir1,
            dir1, dir2, offset+i*dir3, 1));
        removeOverlappingVNodes(nodes[1].back(), 2);
        }

      // w nodes
      for (int i = 1; i < center; i++)
        nodes[2].insert(nodes[2].begin(), buildMatrix(i*dir1+i*dir2,
            (sx_-i)*dir1+i*dir2, (sx_-i)*dir2+i*dir1,
            dir1, dir2, offset-i*dir3, 2));

      nodes[2].push_back(buildMatrix(0,
          sx_*dir1, sx_*dir2,
          dir1, dir2, offset, 2));

      for (int i = 1; i < center; i++)
        nodes[2].push_back(buildMatrix(i*dir1+i*dir2,
            (sx_-i)*dir1+i*dir2, (sx_-i)*dir2+i*dir1,
            dir1, dir2, offset+i*dir3, 2));

      // p nodes
      for (int i = 1; i < center; i++)
        nodes[3].insert(nodes[3].begin(), buildMatrix(i*dir1+i*dir2,
            (sx_-i)*dir1+i*dir2, (sx_-i)*dir2+i*dir1,
            dir1, dir2, offset-(i-1)*dir3, 3));

      for (int i = 1; i < center; i++)
        nodes[3].push_back(buildMatrix(i*dir1+i*dir2,
            (sx_-i)*dir1+i*dir2, (sx_-i)*dir2+i*dir1,
            dir1, dir2, offset+i*dir3, 3));
    }
    break;
    }
  return nodes;
  }

int SkewCartesianPartitioner::getDomainInfo(int sd, int &x, int &y, int &z, char &type) const
  {
  // Find z of layer that contains the subdomain
  int totDomsPerLayer = npx_*npy_ + npx_ * (npy_+1) + (npx_+1) * npy_;
  z = sd / totDomsPerLayer;

  // first z-domain in the layer
  int firstDomain = z * totDomsPerLayer;

  // number of z-domains in the layer
  int zDomsPerLayer = npx_ * npy_;

  if (sd >= firstDomain && sd <= firstDomain + zDomsPerLayer - 1)
    {
    // The first zDomsPerLayer are z-domains
    type = 'z';
  
    // Find x and y location
    y = (sd - firstDomain) / npx_;
    x = (sd - firstDomain) % npx_;
    }
  else
    {
    // First non-z-domain (this is always a y-domain)
    int yStart = firstDomain + zDomsPerLayer;
  
    // Find y location
    y = (sd - yStart) / (2*npx_+1);
  
    // Find x location (will be updated for x type)
    x = (sd - yStart) % (2*npx_+1);
  
    // Note: for fixed y, we have npx_ y-domains, then (npx_+1) x-domains
    // so x numbering is 0, 1, 2,, npx_-1, 0, 1, ..., npx_
    // Therefore we have to subtract npx_ from x that was found above
    // if we have an x-domain!
    if (x < npx_)
      type = 'y';
    else
      {
      type = 'x';
    
      // Update x location
      x = x - npx_;
      }
    }
  return 0;
  }

std::vector<int> SkewCartesianPartitioner::template2list(
  std::vector<std::vector<std::vector<std::vector<int> > > > const &temp) const
  {
  std::vector<int> out;
  for (auto &it: temp)
    for (auto &it2: it)
      for (auto &it3: it2)
        for (auto &it4: it3)
          out.push_back(it4);

  std::sort(out.begin(), out.end());
  return out;
  }

std::vector<std::vector<int> > SkewCartesianPartitioner::solveGroups(char type,
  std::vector<std::vector<std::vector<std::vector<int> > > > const &tempX,
  std::vector<std::vector<std::vector<std::vector<int> > > > const &tempY,
  std::vector<std::vector<std::vector<std::vector<int> > > > const &tempZ,
  int directions[3]) const
  {
  int sx_ = tempX[2].size() - 1; // Just so we do not have to provide this as input

  std::vector<int> template1;
  std::vector<int> template2;
  std::vector<int> template3;

  int dir1, dir2, dir3;

  switch (type)
    {
    case 'x':
      template1 = template2list(tempX);
      dir1 = directions[1];
      template2 = template2list(tempY);
      dir2 = directions[2];
      template3 = template2list(tempZ);
      dir3 = directions[0];
      break;
    case 'y':
      template1 = template2list(tempY);
      dir1 = directions[2];
      template2 = template2list(tempZ);
      dir2 = directions[0];
      template3 = template2list(tempX);
      dir3 = directions[1];
      break;
    case 'z':
      template1 = template2list(tempZ);
      dir1 = directions[0];
      template2 = template2list(tempX);
      dir2 = directions[1];
      template3 = template2list(tempY);
      dir3 = directions[2];
      break;
    }

  // Create model problem
  // same type domains
  int pos1[11] = {0, dir1, dir2, dir3, -dir1, -dir2, -dir3, dir1+dir2,
                  -dir1+dir2, dir1-dir2, -dir1-dir2};

  // different types
  int pos2[12] = {0, dir2, -dir2, -dir3, dir2-dir3, -dir2-dir3, dir1,
                  dir1+dir2, dir1-dir2, dir1-dir3, dir1+dir2-dir3, dir1-dir2-dir3};
  int pos3[12] = {0, -dir1, dir1, -dir3, -dir1-dir3, dir1-dir3, dir2,
                  -dir1+dir2, dir1+dir2, dir2-dir3, -dir1+dir2-dir3, dir1+dir2-dir3};

  // Setup all domains in the test problem
  std::vector<std::vector<int> > domain;
  for (auto &it: pos1)
    {
    domain.emplace_back(template1);
    std::for_each(domain.back().begin(), domain.back().end(),
      [this, it](int& d) {d += this->dof_ * this->sx_ * it;});
    }

  for (auto &it: pos2)
    {
    domain.emplace_back(template2);
    std::for_each(domain.back().begin(), domain.back().end(),
      [this, it](int& d) {d += this->dof_ * this->sx_ * it;});
    }

  for (auto &it: pos3)
    {
    domain.emplace_back(template3);
    std::for_each(domain.back().begin(), domain.back().end(),
      [this, it](int& d) {d += this->dof_ * this->sx_ * it;});
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
      if (std::find(domain[i].begin(), domain[i].end(), node) != domain[i].end())
        listOfDomains += 1 << i;

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
  for (auto &cats: newGroups)
    for (auto &group: cats)
      if (!group.empty())
        groups.push_back(group);

  return groups;
  }

std::vector<std::vector<int> > SkewCartesianPartitioner::createSubdomain(
  int x, int y, int z, char type,
  std::vector<std::vector<std::vector<std::vector<int> > > > temp,
  std::vector<std::vector<int> > groups) const
  {
  std::vector<int> numFirstHalf;
  std::vector<int> coordinates;
  std::vector<std::vector<int> > removeInfo;

  int firstCell = sx_ * (x + y * nx_ + z * nx_ * ny_);
  int coordMax = -1;

  switch (type)
    {
    case 'x':
      // Number of layers
      numFirstHalf.push_back(sx_ / 2 + 1);
      numFirstHalf.push_back(sx_ / 2 + 1);
      numFirstHalf.push_back(sx_ / 2 + 1);
      numFirstHalf.push_back(sx_ / 2);

      // Order of coordinates
      coordinates.push_back(x);
      coordinates.push_back(y);
      coordinates.push_back(z);

      coordMax = npx_;

      // From which node types to remove a row/col. {i,j} contains 
      // nodes (1u, 2=v, 3=w, 4=p) to remove from half i and coordinate j
      removeInfo.emplace_back();
      removeInfo.back().push_back(1);
      removeInfo.back().push_back(2);
      removeInfo.emplace_back(1, 2);
      removeInfo.emplace_back();
      removeInfo.emplace_back();
      break;
    case 'y':
      // Number of layers
      numFirstHalf.push_back(sx_ / 2);
      numFirstHalf.push_back(sx_ / 2 + 1);
      numFirstHalf.push_back(sx_ / 2 + 1);
      numFirstHalf.push_back(sx_ / 2);

      // Order of coordinates
      coordinates.push_back(y);
      coordinates.push_back(x);
      coordinates.push_back(z);

      coordMax = npy_;

      // From which node types to remove a row/col. {i,j} contains 
      // nodes (1u, 2=v, 3=w, 4=p) to remove from half i and coordinate j
      removeInfo.emplace_back();
      removeInfo.back().push_back(0);
      removeInfo.back().push_back(1);
      removeInfo.back().push_back(2);
      removeInfo.emplace_back(1, 2);
      removeInfo.emplace_back(1, 0);
      removeInfo.emplace_back();
      break;
    case 'z':
      // Number of layers
      numFirstHalf.push_back(sx_ / 2);
      numFirstHalf.push_back(sx_ / 2);
      numFirstHalf.push_back(sx_ / 2 + 1);
      numFirstHalf.push_back(sx_ / 2);

      // Order of coordinates
      coordinates.push_back(z);
      coordinates.push_back(x);
      coordinates.push_back(y);

      coordMax = npz_;

      // From which node types to remove a row/col. {i,j} contains 
      // nodes (1=u, 2=v, 3=w, 4=p) to remove from half i and coordinate j
      removeInfo.emplace_back();
      removeInfo.back().push_back(0);
      removeInfo.back().push_back(1);
      removeInfo.back().push_back(2);
      removeInfo.emplace_back();
      removeInfo.back().push_back(1);
      removeInfo.back().push_back(2);
      removeInfo.emplace_back();
      removeInfo.back().push_back(0);
      removeInfo.back().push_back(1);
      removeInfo.emplace_back(1, 1);
      break;
    }

  
  // Get first and second half of template
  std::vector<std::vector<std::vector<std::vector<int> > > > firstHalf;
  std::vector<std::vector<std::vector<std::vector<int> > > > secondHalf;
  std::vector<int> removeList;

  // Move the template and the groups
  for (int i = 0; i < temp.size(); i++)
    for (int j = 0; j < temp[i].size(); j++)
      for (int k = 0; k < temp[i][j].size(); k++)
        std::for_each(temp[i][j][k].begin(), temp[i][j][k].end(),
          [this, firstCell](int& d) { d += this->dof_ * firstCell;});

  for (int i = 0; i < groups.size(); i++)
    std::for_each(groups[i].begin(), groups[i].end(),
      [this, firstCell](int& d) { d += this->dof_ * firstCell;});

  for (int i = 0; i < temp.size(); i++)
    {
    firstHalf.emplace_back();
    std::copy(temp[i].begin(), temp[i].begin() + numFirstHalf[i],
      std::back_inserter(firstHalf[i]));
    secondHalf.emplace_back();
    std::copy(temp[i].begin() + numFirstHalf[i], temp[i].end(),
      std::back_inserter(secondHalf[i]));
    }

  std::vector<std::vector<std::vector<std::vector<int> > > > &tempRef = temp;
  std::vector<std::vector<int> > removePos;
  // Check whether to throw away a half
  if (coordinates[0] == 0)
    {
    tempRef = secondHalf;
    removeList = template2list(firstHalf);
    removePos.emplace_back();
    removePos.emplace_back(dof_, 0);
    }
  else if (coordinates[0] == coordMax)
    {
    tempRef = firstHalf;
    removeList = template2list(secondHalf);
    removePos.emplace_back(numFirstHalf);
    std::for_each(removePos.back().begin(), removePos.back().end(), [](int& d) {d--;});
    removePos.emplace_back();
    }
  else // Both halves
    {
    removePos.emplace_back(numFirstHalf);
    std::for_each(removePos.back().begin(), removePos.back().end(), [](int& d) {d--;});
    removePos.emplace_back(numFirstHalf);
    }

  // Add excess nodes to RemoveList
  for (int half = 0; half < removePos.size(); half++)
    {
    if (!removePos[half].empty())
      {
      for (int coord = 1; coord < 3; coord++)
        {
        if (coordinates[coord] == 0)
          {
          for (int &n: removeInfo[half * 2 + coord - 1])
            {
            if (coord == 1)
              {
              // Compute the maximum length to account for v-domains that miss
              // half of a column
              size_t maxLen = 0;
              for (int i = 0; i < tempRef[n][removePos[half][n]].size(); i++)
                maxLen = std::max(maxLen, tempRef[n][removePos[half][n]][i].size());
              for (int i = 0; i < tempRef[n][removePos[half][n]].size(); i++)
                if (tempRef[n][removePos[half][n]][i].size() == maxLen)
                  removeList.push_back(tempRef[n][removePos[half][n]][i][0]);
              }
            else if (coord == 2)
              for (int i = 0; i < tempRef[n][removePos[half][n]][0].size(); i++)
                removeList.push_back(tempRef[n][removePos[half][n]][0][i]);
            }
          }
        }
      }
    }

  // Get all groups, and remove the entries that are on removeList
  for (auto &group: groups)
    {
    auto last = group.end();
    for (int &toRemove: removeList)
      last = std::remove(group.begin(), last, toRemove);
    group.erase(last, group.end());
    }

  // Get first pressure node from interior to a new group.
  // Assumes ordering of groups by size!
  for (int &node: groups[0])
    if (node % dof_ == pvar_)
      {
      groups.emplace_back(1, node);
      groups[0].erase(std::remove(groups[0].begin(), groups[0].end(), node), groups[0].end());
      break;
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

  char type;
  int x, y, z;
  getDomainInfo(sd, x, y, z, type);
  std::vector<std::vector<int> > nodes;
  
  switch (type)
    {
    case 'x':
      nodes = createSubdomain(x, y, z, type, templateX_, groupsX_);
      break;
    case 'y':
      nodes = createSubdomain(x, y, z, type, templateY_, groupsY_);
      break;
    case 'z':
      nodes = createSubdomain(x, y, z, type, templateZ_, groupsZ_);
      break;
    }

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
  return 0;
  }

  }
