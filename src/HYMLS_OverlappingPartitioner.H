#ifndef HYMLS_OVERLAPPING_PARTITIONER_H
#define HYMLS_OVERLAPPING_PARTITIONER_H

#include "HYMLS_config.h"

#include "HYMLS_HierarchicalMap.H"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "HYMLS_PLA.H"

#include <string>

namespace Teuchos
  {
class ParameterList;
  }

class Epetra_Map;

namespace HYMLS {
class BasePartitioner;
  }

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
  OverlappingPartitioner(
    Teuchos::RCP<const Epetra_Map> map,
    Teuchos::RCP<Teuchos::ParameterList> params, int level=1,
    Teuchos::RCP<const Epetra_Map> overlappingMap=Teuchos::null);

  //! destructor
  virtual ~OverlappingPartitioner();

  //! Step 1: non-overlapping partitioning
  Teuchos::RCP<const BasePartitioner> Partition();

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
  Teuchos::RCP<const OverlappingPartitioner> SpawnNextLevel(
    Teuchos::RCP<const Epetra_Map> map,
    Teuchos::RCP<const Epetra_Map> overlappingMap) const;

  //! from the PLA base class
  void setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params);

protected:

  //! partitioning strategy
  std::string partitioningMethod_;

private:

  //! Step 2: construct overlapping maps after partitioning
  //! the result is a HierarchicalMap with three groups per
  //! subdomain: interior, separator and retained.
  int DetectSeparators(Teuchos::RCP<const BasePartitioner> partitioner);

  int RemoveBoundarySeparators(Teuchos::Array<hymls_gidx> &interior_nodes,
    Teuchos::Array<Teuchos::Array<hymls_gidx> > &separator_nodes) const;

  //! Parameterlist for the next level
  Teuchos::RCP<Teuchos::ParameterList> nextLevelParams_;

  //!@}
  };

  }
#endif
