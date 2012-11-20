#ifndef HYMLS_BASE_NODE_CLASSIFIER_H
#define HYMLS_BASE_NODE_CLASSIFIER_H

#include "Teuchos_RCP.hpp"

namespace HYMLS {
class BasePartitioner;
}

class Epetra_IntVector;

namespace HYMLS {

//! implementations of this class assign a node type ID to each node, 
//! for instance:
//! <0: interior but treated as subcell of the subdomain,
//! 0: interior 
//! 1: face separator (edge in 2D)
//! 2: edge separator (vertex in 2D)
//! 3: vertex in 3D
//! >=4: retain in Schur-complement
//! 
//! This class is a virtual base class, there may be smarter ways to do things
//! for certain classes of problems. A default implementation can be found in 
//! StandardNodeClassifier.
//!
//! The actual numbers put in the node type vector are problem dependent, in 
//! general you will need to understand quite well what OverlappingPartitioner
//! does with these numbers in order to implement your own.
class BaseNodeClassifier
  {
  
public:

  //! called by constructor
  virtual int BuildNodeTypeVector() = 0;

  //! get node type vector
  virtual const Epetra_IntVector& Vector() const = 0;

  //! get node type vector with overlap
  virtual const Epetra_IntVector& OverlappingVector() const = 0;

  //! get node type vector
  virtual Teuchos::RCP<const Epetra_IntVector> GetVector() const = 0;

  //! get node type vector with overlap
  virtual Teuchos::RCP<const Epetra_IntVector> GetOverlappingVector() const = 0;

  };

}
#endif
