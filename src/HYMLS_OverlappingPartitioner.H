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
class Epetra_CrsGraph;
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

  //! returns true if a separator group belongs to the same subdomain
  //! as the interior nodes of that subdomain, according to the original
  //! non-overlapping partitioning. This is useful when looping over
  //! separators but not wanting to access any node twice.
  bool GroupOwnedBySubdomain(int sd, int grp) const
    {
    if (this->NumElements(sd,grp)==0)
      {
      return true;
      }
    int gid1 = GID(sd,grp,0);
    int gid0 = GID(sd,0,0);
    return ((*partitioner_)(gid0)==(*partitioner_)(gid1));
    }
  
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

  //! write the graph used by this class to a text file
  
  //! file originalGraph<level>.txt: original matrix graph on this level
  int DumpGraph() const;
  
  //! from the PLA base class
  void setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params);

  //! get a list of valid parameters for this object
  Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;
      
        
protected: 
  
  //! matrix
  Teuchos::RCP<const Epetra_RowMatrix> matrix_;

  //! graph of the problem and a variant with overlap
  //! between partitions (but not subdomains)
  Teuchos::RCP<const Epetra_CrsGraph> graph_;
  Teuchos::RCP<Epetra_CrsGraph> p_graph_;
  
  //! for going between graph and p_Graph, nodeType and p_nodeType
  Teuchos::RCP<Epetra_Import> importOverlap_;
  
  //! vector of node types (0: interior, 1: standard separator, 
  //! 2: secondary separator (vertex in 2D, edge in 3D),        
  //! 3: vertex (3D only), 4: retained in Schur complement.     
  Teuchos::RCP<const Epetra_IntVector> nodeType_, p_nodeType_;
  
  //! physical dimension (i.e. 2D, 3D)
  int dim_;
  
  //! grid size
  int nx_,ny_,nz_;
  
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
  
  //! this parameter can be set by the entry "Substitute Graph" 
  //! in the "Problem"->"Partitioner" sublist, and it           
  //! defaults to 'true'. If true, the DetectSeparators() and   
  //! GroupSeparators() functions (i.e. the ordering) are based 
  //! on an idealized graph consisting of 9-/27-point Laplacians
  //! and identity matrices (for uncoupled variables).          
  //! If false, the graph of the original matrix is used, which 
  //! probably leads to lower quality partitioning and not to   
  //! grid-independent convergence.                             
  //! TODO - this parameter is deprecated and should be removed.
  bool substituteGraph_;
  
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

  //! 1. build the graph on which we base the separator detection,
  //     either just the pattern of A or, for saddlepoint problems,
  //     (graph_)
  //     the pattern of A+BB' B; B' 0].
  //! 2. build parallel (overlapping) graph so we can access rows
  //!    on separators easily. Creates p_graph_ from graph_
  //!    (p_graph_)
  //! also constructs the importer, importOverlap_, between the two.
  int CreateGraph();

  //! Step 2: construct overlapping maps after partitioning
  //! the result is a HierarchicalMap with three groups per
  //! subdomain: interior, separator and retained.
  int DetectSeparators();
  
  //! Step 3: reorder and group separators. 
  //! The result is a HierarchicalMap with a
  //! group per subdomain for interior and a
  //! group for each separator, e.g. for the
  //! northern U-face separator, etc. and a 
  //! separate group for each retained P-   
  //! node.
  int GroupSeparators();
  
  //!\name functions called from DetectSeparators()
  //!@{
  
  //! form list with interior, separator and retained nodes for subdomain
  // sd. Links separators to subdomains.
  int BuildNodeLists(int sd, const Epetra_CrsGraph& parallelGraph, 
                              const Epetra_IntVector& nodeTypes,
                              const Epetra_IntVector& p_nodeTypes,
                              Teuchos::Array<int>& interiorNodes,
                              Teuchos::Array<int>& separatorNodes,
                              Teuchos::Array<int>& retainedNodes) const;
    
  //!@}


  };

}
#endif
