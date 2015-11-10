#ifndef HYMLS_STANDARD_NODE_CLASSIFIER_H
#define HYMLS_STANDARD_NODE_CLASSIFIER_H

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "HYMLS_BaseNodeClassifier.H"

namespace Teuchos 
  {
  class ParameterList;
  }

namespace HYMLS {
class BasePartitioner;
}

class Epetra_IntVector;
class Epetra_CrsGraph;

namespace HYMLS {

//! this class assigns a node type ID to each node, for instance
//! 0: interior
class StandardNodeClassifier : public BaseNodeClassifier
  {
  
public:

  //! constructor
  //! the arguments level, nx,ny,nz are just for pretty-printing the 
  //! nodeType vector in debugging mode.
  StandardNodeClassifier(Teuchos::RCP<const Epetra_CrsGraph> parGraph, 
      Teuchos::RCP<const BasePartitioner> part, 
      const Teuchos::Array<std::string>& variableType,
      const Teuchos::Array<bool>& retainIsolated,
      int level=1, int nx=0, int ny=0, int nz=0,
      std::string label="StandardNodeClassifier");

  //! destructor
  virtual ~StandardNodeClassifier();

  //! called by constructor
  int BuildNodeTypeVector();

  //! get node type vector
  const Epetra_IntVector& Vector() const 
    {
    return *nodeType_;
    }

  //! get node type vector with overlap
  const Epetra_IntVector& OverlappingVector() const 
    {
    return *p_nodeType_;
    }

  //! get node type vector
  Teuchos::RCP<const Epetra_IntVector> GetVector() const
    {
    return nodeType_;
    }

  //! get node type vector with overlap
  Teuchos::RCP<const Epetra_IntVector> GetOverlappingVector() const
    {
    return p_nodeType_;
    }

  //!
  std::string Label() const {return label_;}

protected: 

  //! graph of the problem and a variant with overlap
  //! between partitions (but not subdomains)
  Teuchos::RCP<const Epetra_CrsGraph> parallelGraph_;
  
  //! vector of node types (0: interior, 1: standard separator, 
  //! 2: secondary separator (vertex in 2D, edge in 3D),        
  //! 3: vertex (3D only), 4: retained in Schur complement.     
  Teuchos::RCP<Epetra_IntVector> nodeType_, p_nodeType_;
  
  //! partitioning object used for the subdomains
  //Teuchos::RCP<Ifpack_OverlappingPartitioner> partitioner;
  Teuchos::RCP<const BasePartitioner> partitioner_;

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
  
  //!
  std::string label_;
  
  //! just for PrintNodeTypeVector
  int myLevel_,nx_,ny_,nz_,dof_;

protected:

  
  //!\name functions called from BuildNodeTypeVector()
  //!@{
  
  //! build initial vector with 0 (interior), 1 (separator) or 5 (retained P-node)
  int BuildInitialNodeTypeVector(const Epetra_CrsGraph& parallelGraph, 
                                        Epetra_IntVector& nodeTypes) const;

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
  int UpdateNodeTypeVector(const Epetra_CrsGraph& parallelGraph, 
                      const Epetra_IntVector& p_nodeTypes,
                            Epetra_IntVector& nodeTypes) const;

  //! this is a non-elegant bug-fix for multi-level Stokes, it will
  //! only work for cartesian Grid Navier-Stokes problems.
  int UpdateNodeTypeVectorStokes(const Epetra_CrsGraph& parallelGraph, 
                      const Epetra_IntVector& p_nodeTypes,
                            Epetra_IntVector& nodeTypes) const;
  

  //! detect isolated P-nodes and increase their nodeType and
  //! that of the surrounding V-nodes by 1. This gives a 1 for
  //! the P-node and a 2,3 or 4 for the V-nodes (in 3D). As the
  //! increments may come from different processors, we need to
  //! put them into a separate vector and perform an Import-Add
  //! operation to get the correct values in parallel.
  int DetectFCC(const Epetra_CrsGraph& parallelGraph, 
                      const Epetra_IntVector& p_nodeTypes,
                            Epetra_IntVector& p_update) const;

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

//! pretty-print the nodeType vector for debugging
std::ostream& PrintNodeTypeVector(const Epetra_IntVector& nt,std::ostream& os,std::string label);

  //!@}

  };

}
#endif
