#include "HYMLS_CartesianPartitioner.H"
#include "HYMLS_Tools.H"
#include "Teuchos_Utils.hpp"
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_IntVector.h"
#include "Epetra_Import.h"
#include "HYMLS_MatrixUtils.H"

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#include "Epetra_MpiDistributor.h"
#endif

using Teuchos::toString;

namespace HYMLS {

// constructor
CartesianPartitioner::CartesianPartitioner(
  Teuchos::RCP<const Epetra_Map> map, int nx, int ny, int nz, int dof, int pvar,
  GaleriExt::PERIO_Flag perio)
  : BasePartitioner(), label_("CartesianPartitioner"),
    baseMap_(map), nx_(nx), ny_(ny), nz_(nz),
    numLocalSubdomains_(-1), numGlobalSubdomains_(-1),
    dof_(dof), pvar_(pvar), perio_(perio)
  {
  HYMLS_PROF3(label_,"Constructor");
  active_=true; // by default, everyone is assumed to own a part of the domain.
  // if in Partition() it turns out that there are more processor
  // partitions than subdomains, active_ is set to false for some
  // ranks, which will get an empty part of the reordered map
  // (cartesianMap_).

  comm_=Teuchos::rcp(&(baseMap_->Comm()),false);
  cartesianMap_=Teuchos::null;

  if (baseMap_->IndexBase()!=0)
    {
    Tools::Warning("Not sure, but I _think_ your map should be 0-based",
      __FILE__, __LINE__);
    }

  HYMLS_DEBVAR(nx_);
  HYMLS_DEBVAR(ny_);
  HYMLS_DEBVAR(nz_);
  HYMLS_DEBVAR(dof_);

  npx_=-1;// indicates that Partition() hasn't been called
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
      Tools::Error("Partition() not yet called!",__FILE__,__LINE__);
      }
#endif      
    int ii = i / sx_;
    int jj = j / sy_;
    int kk = k / sz_;

    int ind = Tools::sub2ind(npx_,npy_,npz_,ii,jj,kk);
    //HYMLS_DEBUG("Partition ID["<<i<<","<<j<<","<<k<<"] = "<<ind);
    return ind;
    }

Teuchos::RCP<Epetra_Map> CartesianPartitioner::CreateSubdomainMap(
  int num_active) const
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

  Teuchos::RCP<Epetra_Map> result = Teuchos::rcp(new Epetra_Map(NumGlobalElements,
      NumMyElements, MyGlobalElements, 0, *comm_));

  delete [] MyGlobalElements;
  return result;
  }

int CartesianPartitioner::Partition(int nparts, bool repart)
  {
  HYMLS_PROF3(label_,"Partition");
  int npx,npy,npz;
  Tools::SplitBox(nx_,ny_,nz_,nparts,npx,npy,npz);
  return this->Partition(npx,npy,npz,repart);
  }

// partition an [nx x ny x nz] grid with one DoF per node
// into nparts global subdomains.
int CartesianPartitioner::Partition(int npx_in,int npy_in, int npz_in, bool repart)
  {
  HYMLS_PROF3(label_,"Partition (2)");
  npx_=npx_in;
  npy_=npy_in;
  npz_=npz_in;

  HYMLS_DEBVAR(npx_);
  HYMLS_DEBVAR(npy_);
  HYMLS_DEBVAR(npz_);

  sx_=nx_/npx_;
  sy_=ny_/npy_;
  sz_=nz_/npz_;

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
  std::string s4=Teuchos::toString(nprocx_)+"x"+Teuchos::toString(nprocy_)+"x"+Teuchos::toString(nprocz_);

  int my_npx = npx_ / nprocx_;
  int my_npy = npy_ / nprocy_;
  int my_npz = npz_ / nprocz_;

  if (comm_->MyPID()>=nprocs)
    {
    active_ = false;
    }

  // if some processors have no subdomains, we need to
  // repartition the map even if it is a cartesian partitioned
  // map already:
  if (nprocs<comm_->NumProc()) repart=true;

  int rank=comm_->MyPID();
  int rankI=-1,rankJ=-1,rankK=-1;

  if (active_)
    {
    Tools::ind2sub(nprocx_,nprocy_,nprocz_,rank,rankI,rankJ,rankK);
    numLocalSubdomains_=my_npx*my_npy*my_npz;
    }
  else
    {
    numLocalSubdomains_=0;
    }

  HYMLS_DEBVAR(npx_);
  HYMLS_DEBVAR(npy_);
  HYMLS_DEBVAR(npz_);
  HYMLS_DEBVAR(active_);
  HYMLS_DEBVAR(rank);
  HYMLS_DEBVAR(rankI);
  HYMLS_DEBVAR(rankJ);
  HYMLS_DEBVAR(rankK);

  sdMap_=CreateSubdomainMap(nprocs);
  HYMLS_DEBVAR(*sdMap_);

  numLocalSubdomains_ = sdMap_->NumMyElements();
  numGlobalSubdomains_ = sdMap_->NumGlobalElements();

// create redistributed map:
  Teuchos::RCP<Epetra_Map> repartitionedMap =
    Teuchos::rcp_const_cast<Epetra_Map>(baseMap_);

// repartitioning may occur for two reasons, typically on coarser levels:
// a) the number of subdomains becomes smaller than the number of processes,
// b) the subdomains can't be nicely distributed among the processes.
// In both cases some processes are deactivated.
  if (repart)
    {
    Tools::Out("repartition for "+s4+" procs");
    HYMLS_PROF3(label_,"repartition map");

    int numMyElements = numLocalSubdomains_ * sx_ * sy_ * sz_ * dof_;
    int *myGlobalElements = new int[numMyElements];
    int pos = 0;
    for (int i = 0; i < baseMap_->MaxAllGID()+1; i++)
      {
      if (LSID(i) != -1)
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

    repartitionedMap = Teuchos::rcp(new Epetra_Map(-1, pos,
        myGlobalElements, baseMap_->IndexBase(), *comm_));

    HYMLS_DEBVAR(*repartitionedMap);
    if (myGlobalElements)
      delete [] myGlobalElements;
    }

  // note: NumLocalParts() is simply numLocalSubdomains_.
  int *NumElementsInSubdomain = new int[NumLocalParts()];
  for (int sd=0;sd<NumLocalParts();sd++) NumElementsInSubdomain[sd]=0;

  subdomainPointer_.resize(NumLocalParts()+1);

  // now we have a cartesian processor partitioning and no nodes have to
  // be moved between partitions. Some partitions may be empty, though.
  for (int lid=0;lid<repartitionedMap->NumMyElements();lid++)
    {
    int gid = repartitionedMap->GID(lid);
    int lsd=LSID(gid);

    if (lsd<0)
      {
      Tools::Error("repartitioning seems to be necessary/have failed for gid "
        + Teuchos::toString(gid) + ".", __FILE__, __LINE__);
      }
    NumElementsInSubdomain[lsd]++;
    }

  subdomainPointer_[0]=0;
  for (int i=0;i<NumLocalParts();i++)
    {
    subdomainPointer_[i+1]=subdomainPointer_[i]+NumElementsInSubdomain[i];
    }

  int NumMyElements=repartitionedMap->NumMyElements();
  if (subdomainPointer_[NumLocalParts()]!=NumMyElements)
    {
    Tools::Error("repartitioning - sanity check failed",__FILE__,__LINE__);
    }
  int *MyGlobalElements = new int[NumMyElements];
  for (int i=0;i<NumLocalParts();i++) NumElementsInSubdomain[i]=0;

  for (int lid=0;lid<repartitionedMap->NumMyElements();lid++)
    {
    int gid = repartitionedMap->GID(lid);
    int lsd=LSID(gid);
    MyGlobalElements[subdomainPointer_[lsd] + NumElementsInSubdomain[lsd]] = gid;
    NumElementsInSubdomain[lsd]++;
    }
  // sort nodes per subdomain
  for (int i=0;i<NumLocalParts();i++)
    {
    std::sort(MyGlobalElements + subdomainPointer_[i],
      MyGlobalElements + subdomainPointer_[i+1]);
    }

  HYMLS_DEBVAR(NumMyElements);

  cartesianMap_=Teuchos::rcp(new Epetra_Map(-1,NumMyElements,MyGlobalElements,0,*comm_));
  HYMLS_DEBVAR(*cartesianMap_);

  if (active_)
    {
    Tools::Out("Number of Partitions: "+s4);
    Tools::Out("Partition: ["+toString(rankI)+" "+toString(rankJ)+" "+toString(rankK)+"]");
    Tools::Out("Number of Local Subdomains: "+toString(NumLocalParts()));

    delete [] MyGlobalElements;
    delete [] NumElementsInSubdomain;
    }
  return 0;
  }

int CartesianPartitioner::RemoveBoundarySeparators(Teuchos::Array<int> &interior_nodes,
  Teuchos::Array<Teuchos::Array<int> > &separator_nodes) const
  {
  // TODO: There should be a much easier way to do this, but I want to get rid
  // of it eventually for consistency. We need those things anyway for periodic
  // boundaries.

  // Remove boundary separators and add them to the interior
  Teuchos::Array<Teuchos::Array<int> >::iterator sep = separator_nodes.begin();
  for (; sep != separator_nodes.end(); sep++)
    {
    Teuchos::Array<int> &nodes = *sep;
    if (nodes.size() == 0)
      separator_nodes.erase(sep);
    else if ((nodes[0] / dof_ + 1) % nx_ == 0)
      {
      // Remove right side
      interior_nodes.insert(interior_nodes.end(), nodes.begin(), nodes.end());
      separator_nodes.erase(sep);
      }
    else if (ny_ > 1 && (nodes[0] / dof_ / nx_ + 1) % ny_ == 0)
      {
      // Remove bottom side
      interior_nodes.insert(interior_nodes.end(), nodes.begin(), nodes.end());
      separator_nodes.erase(sep);
      }
    else if (nz_ > 1 && (nodes[0] / dof_ / nx_ / ny_ + 1) % nz_ == 0)
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
    int nodeID = -1;
    Teuchos::Array<int> &nodes = *sep;
    for (int i = 0; i < nodes.size(); i++)
      {
      if ((nodes[i] / dof_) % nx_ + 2 == nx_)
        {
        nodeID = nodes[i] + dof_;
        Teuchos::Array<int>::iterator it = std::find(
          interior_nodes.begin(), interior_nodes.end(), nodeID);
        if (it != interior_nodes.end())
          {
          nodes.push_back(nodeID);
          interior_nodes.erase(it);
          }
        }
      if (ny_ > 1 && (nodes[i] / dof_ / nx_) % ny_ + 2 == ny_)
        {
        nodeID = nodes[i] + dof_ * nx_;
        Teuchos::Array<int>::iterator it = std::find(
          interior_nodes.begin(), interior_nodes.end(), nodeID);
        if (it != interior_nodes.end())
          {
          nodes.push_back(nodeID);
          interior_nodes.erase(it);
          }
        }
      if (nz_ > 1 && (nodes[i] / dof_ / nx_ / ny_) % nz_ + 2 == nz_)
        {
        nodeID = nodes[i] + dof_ * nx_ * ny_;
        Teuchos::Array<int>::iterator it = std::find(
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

int CartesianPartitioner::GetGroups(int sd, Teuchos::Array<int> &interior_nodes,
  Teuchos::Array<Teuchos::Array<int> > &separator_nodes)
  {
  HYMLS_PROF2(label_,"GetGroups");

  // presure nodes that need to be retained
  Teuchos::Array<int> retained_nodes;

  Teuchos::Array<int> *nodes;
  int first = cartesianMap_->GID(First(sd));
  first = ((first / dof_) % nx_) / sx_ * sx_ * dof_ + first / (sy_ * nx_ * dof_) * (sy_ * nx_ * dof_);
  for (int ktype = (nz_ > 1 ? -1 : 0); ktype < (nz_ > 1 ? 2 : 1); ktype++)
    {
    if (ktype == 1)
      ktype = sz_ - 1;
    if (ktype == -1 && (first / dof_ / nx_ / ny_) % nz_ == 0)
      continue;
 
    for (int jtype = -1; jtype < 2; jtype++)
      {
      if (jtype == 1)
        jtype = sy_ - 1;
      if (jtype == -1 && (first / dof_ / nx_) % ny_ == 0)
        continue;

      for (int itype = -1; itype < 2; itype++)
        {
        if (itype == 1)
          itype = sx_ - 1;
        if (itype == -1 && (first / dof_) % nx_ == 0)
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
            separator_nodes.append(Teuchos::Array<int>());
            nodes = &separator_nodes.back();
            }

          for (int k = ktype; k < ((ktype || nz_ <= 1) ? ktype+1 : sz_-1); k++)
            {
            for (int j = jtype; j < (jtype ? jtype+1 : sy_-1); j++)
              {
              for (int i = itype; i < (itype ? itype+1 : sx_-1); i++)
                {
                int gid = first + i * dof_ + j * nx_ * dof_ + k * nx_ * ny_ * dof_ + d;
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
    separator_nodes.append(Teuchos::Array<int>());
    separator_nodes.back().append(*it);
    }

  return 0;
  }

  }
