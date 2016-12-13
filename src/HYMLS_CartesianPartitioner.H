#ifndef HYMLS_CARTESIAN_PARTITONER_H
#define HYMLS_CARTESIAN_PARTITONER_H

#include "HYMLS_config.h"
#include <iostream>
#include "HYMLS_Tools.H"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Epetra_Map.h"
#include "HYMLS_BasePartitioner.H"

#include "GaleriExt_Periodic.h"

class Epetra_Comm;
class Epetra_Import;

namespace HYMLS {

class CartesianPartitioner : public BasePartitioner
  {
public:

  //! constructor
  CartesianPartitioner(Teuchos::RCP<const Epetra_Map> map, int nx, int ny,
    int nz=1, int dof=1, int pvar=-1, GaleriExt::PERIO_Flag perio=GaleriExt::NO_PERIO);

  //! destructor
  virtual ~CartesianPartitioner();

  //! return global partition ID of a cell (i,j,k)
  int operator()(int i, int j, int k) const;

  //! creates the map from global to local partition IDs. The implementation
  //! may assume that npx_, sx_ etc. are already set so that operator() works.
  //! num_active indicates how many procs should get nodes (if there are at
  //! least num_active nodes)
  Teuchos::RCP<Epetra_Map> CreateSubdomainMap(int num_active) const;

  //! partition an [nx x ny x nz] grid with (possibly) several DoF per node
  //! into nparts global subdomains.
  int Partition(int nparts, bool repart=true);

  //! partition an [nx x ny x nz] grid with one DoF per node
  //! into npx*npy*npz global subdomains. If repart=true,
  //! the map may need repartitioning to match the cartesian
  //! layout of the new partitioned map. Otherwise it is
  //! assumed that the map already has a global cartesian
  //! processor partitioning (as generated i.e. by the
  //! MatrixUtils::CreateMap functions).
  int Partition(int npx, int npy, int npz, bool repart=true);

private:
  //! Method to remove separators at the boundary when they are not
  //! strictly needed. This is when there are no periodic boundary
  //! conditions
  int RemoveBoundarySeparators(Teuchos::Array<int> &interior_nodes,
    Teuchos::Array<Teuchos::Array<int> > &separator_nodes) const;

public:
  //! Get interior and separator groups of the subdomain sd
  int GetGroups(int sd, Teuchos::Array<int> &interior_nodes,
    Teuchos::Array<Teuchos::Array<int> > &separator_nodes);

  //! is this class fully set up?
  inline bool Partitioned() const {return (npx_>0);}

  //! get the type of a variable (if more than 1 dof per node, otherwise just 0)
  inline int DofPerNode() const
    {
    return dof_;
    }

  //! get the type of a variable (if more than 1 dof per node, otherwise just 0)
  inline int VariableType(int gid) const
    {
    return (int)(MOD(gid, dof_));
    }

  //! get non-overlapping subdomain id
  int operator()(int gid) const
    {
    int i,j,k,var;
#ifdef HYMLS_TESTING
    if (!Partitioned())
      {
      Tools::Error("Partition() not yet called!",__FILE__,__LINE__);
      }
#endif
    Tools::ind2sub(nx_,ny_,nz_,dof_,gid,i,j,k,var);
    return operator()(i,j,k);
    }

  //! return the repartitioned/reordered map
  inline const Epetra_Map& Map() const {return *cartesianMap_;}

  //! return number of subdomains in this proc partition
  int NumLocalParts() const
    {
    if (numLocalSubdomains_<0)
      {
      Tools::Error("not implemented correctly",__FILE__,__LINE__);
      }
    return numLocalSubdomains_;
    }

  //! return number of subdomains in this proc partition
  int NumGlobalParts() const
    {
    if (numGlobalSubdomains_<0)
      {
      Tools::Error("not implemented correctly",__FILE__,__LINE__);
      }
    return numGlobalSubdomains_;
    }

  //! get number of elements in local subdomain <part>
  inline int NumElements(int part) const {return First(part+1)-First(part);}

  //! get the local index of the first element in subdomain i.
  //! For i=NumLocalParts()+1, Map().NumLocalElements() is returned.
  inline int First(int i) const {return subdomainPointer_[i];}

  //! get global index of element j in local subdomain i
  inline int GID(int i, int j) const {return Map().GID(First(i)+j);}

  //!
  std::string label_;

protected:

  //! communicator
  Teuchos::RCP<const Epetra_Comm> comm_;

  //! original non-overlapping map
  Teuchos::RCP<const Epetra_Map> baseMap_;

  //! non-overlapping cartesian map (all subdomains are owned by
  //! exactly one process/belong to only one partition)
  Teuchos::RCP<Epetra_Map> cartesianMap_;

  //! global grid size
  int nx_, ny_, nz_;

  //! number of subdomains
  int npx_,npy_,npz_;

  //! processor distribution
  int nprocx_,nprocy_,nprocz_;

  //! number of subdomains on this proc
  int numLocalSubdomains_;

  //! number of subdomains on this proc
  int numGlobalSubdomains_;

  //! subdomain size
  int sx_,sy_,sz_;

  //! number of variables per node
  int dof_;

  //! pressure node
  int pvar_;

  //! maps global to local subdomain ID
  Teuchos::RCP<Epetra_Map> sdMap_;

  //! indicates wether the processor is active (owns any nodes/subdomains)
  bool active_;

  //! type of periodicity in the problem
  GaleriExt::PERIO_Flag perio_;

  //! stencil width, currently set to 1.
  static const int stencil_width_=1;

  //! pointers into map
  Teuchos::Array<int> subdomainPointer_;

protected:

  //! get local subdomain id
  inline int LSID(int i, int j, int k) const
    {
#ifdef HYMLS_TESTING
    if (sdMap_==Teuchos::null)
      {
      Tools::Error("sdMap not created yet!",__FILE__,__LINE__);
      }
#endif
    return sdMap_->LID((*this)(i,j,k));
    }

  //! get local subdomain id
  inline int LSID(int gid) const
    {
#ifdef HYMLS_TESTING
    if (sdMap_==Teuchos::null)
      {
      Tools::Error("sdMap not created yet!",__FILE__,__LINE__);
      }
#endif
    return sdMap_->LID((*this)(gid));
    }

  //! get processor on which a grid point is located
  inline int PID(int i, int j, int k) const
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

    int pid = Tools::sub2ind(nprocx_,nprocy_,nprocz_,1,pidx,pidy,pidz,0);
/*
  HYMLS_DEBUG("ijk: "<<i<<" "<<j<<" "<<k);
  HYMLS_DEBUG("np_loc: "<<npx<<" "<<npy<<" "<<npz);
  HYMLS_DEBUG("sd: "<<sdx<<" "<<sdy<<" "<<sdz);
  HYMLS_DEBUG("PID: "<<pidx<<" "<<pidy<<" "<<pidz<<" ("<<pid<<")");
*/
    return pid;
    }

  //! get processor on which a GID is located
  inline int PID(int gid) const
    {
    int i,j,k,var;
    Tools::ind2sub(nx_,ny_,nz_,dof_,gid,i,j,k,var);
    return PID(i,j,k);
    }

  };

  }
#endif
