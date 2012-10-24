#ifndef HYMLS_BASE_CARTESIAN_PARTITONER_H
#define HYMLS_BASE_CARTESIAN_PARTITONER_H

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

/* implements common functionality of cartesian partitioners
   like the tricky flow() function.
    
    classes derived from this base only need to implement
    operator()(i,j,k) and a function CreateSubdomainMap(),
    which returns a map of the owned global partition numbers.
    
*/
class BaseCartesianPartitioner : public BasePartitioner
  {
  public:
  
  //! constructor
  BaseCartesianPartitioner(Teuchos::RCP<const Epetra_Map> map, int nx, int ny, 
        int nz=1, int dof=1, GaleriExt::PERIO_Flag perio=GaleriExt::NO_PERIO);
  
  //! destructor
  ~BaseCartesianPartitioner();

  //! pure virtual functions
  //@{
  
  //! return global partition ID of a cell (i,j,k)
  virtual int operator()(int i,int j,int k) const = 0;
  
  //! creates the map from global to local partition IDs. The implementation
  //! may assume that npx_, sx_ etc. are already set so that operator() works
  virtual Teuchos::RCP<Epetra_Map> CreateSubdomainMap() const = 0;
  
  //@}
  
  
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
  virtual int Partition(int npx, int npy, int npz, bool repart=true);
  
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
  inline int operator()(int gid) const
    {
    int i,j,k,var;
#ifdef TESTING    
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
              
  //! impose ordering on subdomains
  int flow(int gid1, int gid2);
  
  //! get the node distance used in this object.
  //! for instance, on the second level of our method,
  //! 'ngiehboring nodes' are s gids apart, where
  //! s is the separator length. This value defaults
  //! to 1 and can be set using the SetNodeDistance()
  //! function.
  inline double GetNodeDistance() {return node_distance_;}

  //! set the node distance to be assumed by the object (influences
  //! the behavior of the flow() function)
  void SetNodeDistance(double dist) {node_distance_=dist;}

  //!
  std::string label_;

  private:
  
  //! compute distance between two points/subdomains
  int calc_distance(int n, int i1, int i2, bool perio);

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
  
  //! maps global to local subdomain ID
  Teuchos::RCP<Epetra_Map> sdMap_;
 
  //! indicates wether the processor is active (owns any nodes/subdomains)
  bool active_;
  
  //! type of periodicity in the problem
  GaleriExt::PERIO_Flag perio_;
  
  //! stencil width, currently set to 1.
  static const int stencil_width_=1;
  
  //! physical distance (in grid cells) between neighboring
  //! nodes. This should be set to the separator length on 
  //! coarse levels.
  int node_distance_;

  //! pointers into map
  Teuchos::Array<int> subdomainPointer_;
  
protected:

  //! get local subdomain id 
  inline int LSID(int i, int j, int k) const
    {
#ifdef TESTING
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
#ifdef TESTING
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
    int sdx = (int)floor((double)(i)/(double)sx_);
    int sdy = (int)floor((double)(j)/(double)sy_);
    int sdz = (int)floor((double)(k)/(double)sz_);
    
    //how many subdomains are there per process?
    int npx = (int)(npx_/nprocx_);
    int npy = (int)(npy_/nprocy_);
    int npz = (int)(npz_/nprocz_);

#ifdef TESTING    
    if ( (npx*nprocx_!=npx_) ||
         (npy*nprocy_!=npy_) ||
         (npz*nprocz_!=npz_) )
         {
         Tools::Error("case of irregular partitioning not implemented",
                __FILE__,__LINE__);
         }
#endif
    // so, where is the cell (on which process)?
    int pidx, pidy, pidz;
    pidx = (int)(sdx/npx);
    pidy = (int)(sdy/npy);
    pidz = (int)(sdz/npz);

    int pid=Tools::sub2ind(nprocx_,nprocy_,nprocz_,1,pidx,pidy,pidz,0);
/*
    DEBUG("ijk: "<<i<<" "<<j<<" "<<k);
    DEBUG("np_loc: "<<npx<<" "<<npy<<" "<<npz);
    DEBUG("sd: "<<sdx<<" "<<sdy<<" "<<sdz);
    DEBUG("PID: "<<pidx<<" "<<pidy<<" "<<pidz<<" ("<<pid<<")");
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
