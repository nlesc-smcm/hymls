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
  
  //! get non-overlapping subdomain id 
  inline int operator()(int i, int j, int k) const
    {
#ifdef TESTING    
    if (!Partitioned())
      {
      Tools::Error("Partition() not yet called!",__FILE__,__LINE__);
      }
#endif      
    int ii = i / sx_;
    int jj = j / sy_;
    int kk = k / sz_;

    int ind = Tools::sub2ind(npx_,npy_,npz_,ii,jj,kk);
    //DEBUG("Partition ID["<<i<<","<<j<<","<<k<<"] = "<<ind);
    return ind;
    }

  Teuchos::RCP<Epetra_Map> CreateSubdomainMap(int nActive) const;

  };

}
#endif
