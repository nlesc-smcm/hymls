#ifndef HYMLS_OVERLAPPING_PARTITIONER_H
#define HYMLS_OVERLAPPING_PARTITIONER_H

#include "HYMLS_config.h"

#include "HYMLS_HierarchicalMap.H"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "HYMLS_PLA.H"

namespace Teuchos 
  {
  class ParameterList;
  }

namespace HYMLS {
class BasePartitioner;
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
  virtual ~OverlappingPartitioner();

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

protected:

  //! matrix
  Teuchos::RCP<const Epetra_RowMatrix> matrix_;
  
  //! partitioning strategy
  std::string partitioningMethod_;
  
  //! partitioning object used for the subdomains
  //Teuchos::RCP<Ifpack_OverlappingPartitioner> partitioner;
  Teuchos::RCP<BasePartitioner> partitioner_;
  
private:
  
  //! Step 1: non-overlapping partitioning
  int Partition();

  //! Step 2: construct overlapping maps after partitioning
  //! the result is a HierarchicalMap with three groups per
  //! subdomain: interior, separator and retained.
  int DetectSeparators();

  int RemoveBoundarySeparators(Teuchos::Array<hymls_gidx> &interior_nodes,
    Teuchos::Array<Teuchos::Array<hymls_gidx> > &separator_nodes) const;
    
  //!@}


  };

}
#endif
