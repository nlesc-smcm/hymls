#ifndef HYMLS_CARTESIAN_PARTITONER_H
#define HYMLS_CARTESIAN_PARTITONER_H

#include <iostream>
#include "HYMLS_Tools.H"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Epetra_Map.h"
#include "HYMLS_BasePartitioner.H"

class Epetra_Comm;
class Epetra_Import;

namespace HYMLS {

/*! simple cartesian mesh partitioner. The partitioning is so amazingly
    primitive that it doesn't take any graph into account,
    it just assumes something like a 9- or 27-point stencil 
    2- or 3D on a structured cartesian grid. It works for non-scalar
    problems: The number of degrees of freedom is guessed from the 
    size of the input map, and all dof belonging to one graph node are
    assumed to sit next to each other.
*/
class CartesianPartitioner : public BasePartitioner
  {
  public:
  
  //! constructor
  CartesianPartitioner(Teuchos::RCP<const Epetra_Map> map, int nx, int ny, 
        int nz=1, int dof=1);
  
  //! destructor
  ~CartesianPartitioner();
  
  //! partition an [nx x ny x nz] grid with (possibly) several DoF per node
  //! into nparts global subdomains.
  void Partition(int nparts);
  

  //! partition an [nx x ny x nz] grid with one DoF per node
  //! into npx*npy*npz global subdomains.
  void Partition(int npx, int npy, int npz);
  
  //! get number of local partitions
  int NumLocalParts(){return numLocalSubdomains_;}

  //! get global number of partitions
  int NumGlobalParts(){return npx_*npy_*npz_;}
  
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
  inline int operator()(int i, int j, int k) const
    {
    if (!Partitioned())
      {
      Tools::Error("Partition() not yet called!",__FILE__,__LINE__);
      }
      
    int ii=(int)floor((double)(i/sx_));
    int jj=(int)floor((double)(j/sy_));
    int kk=(int)floor((double)(k/sz_));
    
    int ind=Tools::sub2ind(npx_,npy_,npz_,ii,jj,kk);
    //DEBUG("Partition ID["<<i<<","<<j<<","<<k<<"] = "<<ind);
    return ind;
    }

  //! get non-overlapping subdomain id 
  inline int operator()(int gid) const
    {
    int i,j,k,var;
    Tools::ind2sub(nx_,ny_,nz_,dof_,gid,i,j,k,var);
    return operator()(i,j,k);
    }

  //! return the repartitioned/reordered map
  inline const Epetra_Map& Map() const {return *cartesianMap_;}

  //! get number of elements in local subdomain <part>
  inline int NumElements(int part) const {return First(part+1)-First(part);}
    
  //! get the local index of the first element in subdomain i.
  //! For i=NumLocalParts()+1, Map().NumLocalElements() is returned.
  inline int First(int i) const {return subdomainPointer_[i];}
          
  //! get global index of element j in local subdomain i
  inline int GID(int i, int j) const {return Map().GID(First(i)+j);}
              
                

  protected:

  //! communicator
  Teuchos::RCP<const Epetra_Comm> comm_;
  
  //! original non-overlapping map
  Teuchos::RCP<const Epetra_Map> baseMap_;
  
  //! non-overlapping cartesian map (all subdomains are owned by
  //! exactly one process/belong to only one partition)
  Teuchos::RCP<Epetra_Map> cartesianMap_;
  
  //!

  //! global grid size
  int nx_, ny_, nz_;
  
  //! number of subdomains
  int npx_,npy_,npz_;

  //! number of subdomains on this proc
  int numLocalSubdomains_;
  
  //! subdomain size
  int sx_,sy_,sz_;
  
  //! number of variables per node
  int dof_;

  //! pointers into map
  Teuchos::Array<int> subdomainPointer_;
    
  };

}
#endif
