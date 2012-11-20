#ifndef HYMLS_CARTESIAN_NODE_CLASSIFIER_H
#define HYMLS_CARTESIAN_NODE_CLASSIFIER_H

#include "HYMLS_StandardNodeClassifier.H"
#include "GaleriExt_Periodic.h"
#include "Teuchos_RCP.hpp"

namespace Teuchos
  {
  class ParameterList;
  }

namespace HYMLS {
class BasePartitioner;
}

class Epetra_IntVector;

namespace HYMLS {

//! this class assigns a node type ID to each node, for instance
//! 0: interior
class CartesianStokesClassifier: public StandardNodeClassifier
  {
  
public:

  //! constructor
  CartesianStokesClassifier(Teuchos::RCP<const Epetra_CrsGraph> parGraph, 
      Teuchos::RCP<const BasePartitioner> part, 
      const Teuchos::Array<std::string>& varType,
      const Teuchos::Array<bool>& retIsol,
      GaleriExt::PERIO_Flag perio,
      int dim, int level=1, int nx=0, int ny=0, int nz=0);

  //! destructor
  ~CartesianStokesClassifier();

  //! called by constructor
  int BuildNodeTypeVector();

protected: 
  
  //!
  GaleriExt::PERIO_Flag perio_;
  
  //!
  int dim_;
  
private:

  //! update the nodeType vector to look like this:
  //!   0 0 1 0 0
  //!   0 0 1 0 0
  //!   1 1 2 1 1
  //!   0 0 1 0 0
  //!   0 0 1 0 0
  //! The function is called twice in DetectSeparators to get the following
  //! node types:
  //!   0: interior
  //!   1: face separator in 3D or edge in 2D
  //!   2: edge separator in 3D or vertex in 2D
  //!   3: vertex in 3D
  //!   5: retained P-node
  int UpdateNodeTypeVector_CartStokes(const Epetra_CrsGraph& parallelGraph, 
                      const Epetra_IntVector& p_nodeTypes,
                            Epetra_IntVector& nodeTypes) const;

  //! a slightly modified version of StandardNodeClassifier's implementation
  int DetectFCC_CartStokes(const Epetra_CrsGraph& parallelGraph, 
                      const Epetra_IntVector& p_nodeTypes,
                            Epetra_IntVector& p_updateNT) const;

  //! Detect subdomains in which some separator variables can   
  //! be made 'interior' again. This is 3D Stokes-specific, in  
  //! domains, which case 'full conservation tubes' (FCTs) can  
  //! be formed on the edges between 4 subdomains. The P-nodes  
  //! and one of the velocity components in the interior of such
  //! a tube can be eliminated (one P-node has to be retained,  
  //! as usual).
  int DetectFCT(const Epetra_CrsGraph& parallelGraph, 
                      const Epetra_IntVector& p_nodeTypes,
                            Epetra_IntVector& nodeTypes) const;

  //!@}

  };

}
#endif
