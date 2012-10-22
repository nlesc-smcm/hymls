#ifndef HYMLS_CARTESIAN_PARTITONER_H
#define HYMLS_CARTESIAN_PARTITONER_H

#include <iostream>
#include "HYMLS_Tools.H"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Epetra_Map.h"
#include "HYMLS_BaseCartesianPartitioner.H"

#include "GaleriExt_Periodic.h"

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
class CartesianPartitioner : public BaseCartesianPartitioner
  {
  public:
  
  //! constructor
  CartesianPartitioner(Teuchos::RCP<const Epetra_Map> map, int nx, int ny, 
        int nz=1, int dof=1, GaleriExt::PERIO_Flag perio=GaleriExt::NO_PERIO);
  
  //! destructor
  ~CartesianPartitioner();
  
  //! partition an [nx x ny x nz] grid with one DoF per node
  //! into npx*npy*npz global subdomains. If repart=true,   
  //! the map may need repartitioning to match the cartesian
  //! layout of the new partitioned map. Otherwise it is    
  //! assumed that the map already has a global cartesian   
  //! processor partitioning (as generated i.e. by the      
  //! MatrixUtils::CreateMap functions).
  int Partition(int npx, int npy, int npz, bool repart=true);
  
  //! get non-overlapping subdomain id 
  inline int operator()(int i, int j, int k) const
    {
#ifdef TESTING    
    if (!Partitioned())
      {
      Tools::Error("Partition() not yet called!",__FILE__,__LINE__);
      }
#endif      
    int ii=(int)floor((double)(i/sx_));
    int jj=(int)floor((double)(j/sy_));
    int kk=(int)floor((double)(k/sz_));
    
    int ind=Tools::sub2ind(npx_,npy_,npz_,ii,jj,kk);
    //DEBUG("Partition ID["<<i<<","<<j<<","<<k<<"] = "<<ind);
    return ind;
    }

private:

  //! get local subdomain id 
  inline int LSID(int i, int j, int k) const
    {
#ifdef TESTING
    if (sdMap_==Teuchos::null)
      {
      Tools::Error("not partitioned yet!",__FILE__,__LINE__);
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
      Tools::Error("not partitioned yet!",__FILE__,__LINE__);
      }
#endif      
    return sdMap_->LID(BaseCartesianPartitioner::operator()(gid));
    }

  };

}
#endif
