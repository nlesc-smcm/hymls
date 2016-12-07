#ifndef HYMLS_OVERLAPPING_PARTITIONER_H
#define HYMLS_OVERLAPPING_PARTITIONER_H

#include "HYMLS_config.h"

#include "HYMLS_HierarchicalMap.H"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "GaleriExt_Periodic.h"
#include "HYMLS_PLA.H"

namespace Teuchos 
  {
  class ParameterList;
  }

namespace HYMLS {
class BasePartitioner;
class BaseNodeClassifier;
}

class Epetra_IntVector;
class Epetra_Import;

namespace HYMLS {

/*! This class offers an 'entrance point' to a hierarchy of
    HierarchicalMaps. It constructs a interior/separator   
    ordering based on the graph of the system matrix and a 
    non-overlapping partitioning (BasePartitioner). As it  
    inherits HierarchicalMap it can be used to create a    
    hierarchical ordering by repeatedly calling the Spawn()
    function. It also adds the SpawnNextLevel() function,  
    which is specific for our solver and selects the V-sum 
    nodes to form the next level object.
*/
class OverlappingPartitioner: public HierarchicalMap,
                              public PLA
  {
  
public:

  //! constructor
  OverlappingPartitioner(Teuchos::RCP<const Epetra_RowMatrix> K, 
      Teuchos::RCP<Teuchos::ParameterList> params, int level=1);

  //! destructor
  ~OverlappingPartitioner();

  //! get the partitioner used internally
  const BasePartitioner& Partitioner() const {return *partitioner_;}

  //! this class allows spawning a next level object for the variables  
  //! retained in a reduced problem                                     
  //! TODO: unclutter the BaseO.P., RecursiveO.P., and this class       
  //! (I think the RecursiveO.P. should become the Base, and this class 
  //! the sole implementation)                                          
  //!                                                                   
  //! Whereas the Spawn functions in HierarchicalMap                    
  //! retain a copy of the spawned object, this class doesn't. So you   
  //! should call it only once and keep an RCP yourself.                
  //!                                                                   
  Teuchos::RCP<const OverlappingPartitioner> SpawnNextLevel
        (Teuchos::RCP<const Epetra_RowMatrix> reducedMatrix,
        Teuchos::RCP<Teuchos::ParameterList> newParamList) const;
  
  //! from the PLA base class
  void setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params);

  //! get a list of valid parameters for this object
  Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;

protected:

  //! matrix
  Teuchos::RCP<const Epetra_RowMatrix> matrix_;

  //! physical dimension (i.e. 2D, 3D)
  int dim_;
  
  //! grid size
  int nx_,ny_,nz_;
  
  //! subdomain size
  int sx_,sy_,sz_;
  
  //! partitioning strategy
  std::string partitioningMethod_;

  //! classification strategy
  std::string classificationMethod_;
  
  //! partitioning object used for the subdomains
  //Teuchos::RCP<Ifpack_OverlappingPartitioner> partitioner;
  Teuchos::RCP<BasePartitioner> partitioner_;
  
  //! object that constructs the nodeType vector
  Teuchos::RCP<BaseNodeClassifier> classifier_;

  //! number of variables per grid-cell
  int dof_;
  
  //! indicates which is the pressure variable (-1 if none)
  int pvar_;
 
  //! indicates if in a certain direction there are periodic BC
  GaleriExt::PERIO_Flag perio_;

  //! indicates the type of variables in the problem
  //! supported types are "Laplace", "Uncoupled", "Retain X",
  //! where X is the number of these unknowns that should be
  //! retained in the Schur-complement (e.g "Retain 1" 
  //! pressure for (Navier-)Stokes on a C-grid)
  Teuchos::Array<std::string> variableType_;

  //! indicates if isolated variables of this type should be
  //! retained in the Schur-complement. This is applicable  
  //! i.e. for the pressure in Stokes-type problems, where  
  //! an invalid continuity equation may occur if all       
  //! surrounding velocities belong to separators (full     
  //! conservation cell)                                    
  Teuchos::Array<bool> retainIsolated_;
  
private:
  
  //! Step 1: non-overlapping partitioning
  int Partition();

  //! Step 2: construct overlapping maps after partitioning
  //! the result is a HierarchicalMap with three groups per
  //! subdomain: interior, separator and retained.
  int DetectSeparators();

  int RemoveBoundarySeparators(Teuchos::Array<int> &interior_nodes,
    Teuchos::Array<Teuchos::Array<int> > &separator_nodes) const;
    
  //!@}


  };

}
#endif
