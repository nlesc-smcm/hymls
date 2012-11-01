#ifndef HYMLS_OVERLAPPING_PARTITIONER_H
#define HYMLS_OVERLAPPING_PARTITIONER_H

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
}

class Epetra_IntVector;

namespace HYMLS {

/*! This class offers an 'entrance point' to a hierarchy of
    HierarchicalMaps. It constructs a interior/
    separator ordering based on a problem-dependent graph. as it
    inherits HierarchicalMap it can be used to 
    create a hierarchical ordering by repeatedly calling the 
    Spawn() function.
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
  Teuchos::RCP<const Epetra_CrsGraph> graph_,parallelGraph_;
  
  //! vector of node types (0: interior, 1: standard separator, 
  //! 2: secondary separator (vertex in 2D, edge in 3D),        
  //! 3: vertex (3D only), 4: retained in Schur complement.     
  Teuchos::RCP<Epetra_IntVector> nodeType_, p_nodeType_;
  
  //! physical dimension (i.e. 2D, 3D)
  int dim_;
  
  //! grid size
  int nx_,ny_,nz_;
  
  //! partitioning strategy
  std::string partitioningMethod_;
  
  //! partitioning object used for the subdomains
  //Teuchos::RCP<Ifpack_OverlappingPartitioner> partitioner;
  Teuchos::RCP<BasePartitioner> partitioner_;

  //! number of variables per grid-cell
  int dof_;
 
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
  
  //! partition - Cartesian
  int Partition();

  //! create graph to be used for identifying separators
  int CreateGraph();

  //! construct overlapping maps after partitioning
  int DetectSeparators();
  
  //! reorder and group separators
  int GroupSeparators();
  
  //! for sub-cells of subdomains, like e.g. the edges
  //! in 3D Navier-Stokes, form the separators by     
  //! copying them from neighboring subdomains
  int FixSubCells();
  
  //! build parallel graph after detecting separators
  Teuchos::RCP<Epetra_CrsGraph> CreateParallelGraph();

  //!\name functions called from DetectSeparators()
  //!@{
  
  //! build initial vector with 0 (interior), 1 (separator) or 2 (retained
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
  
  //! detect isolated P-nodes and mark the surrounding V-nodes
  //! as '4' in the nodeType vector. This is Stokes-specific  
  //! functionality.
  int DetectFCC(const Epetra_CrsGraph& parallelGraph, 
                      const Epetra_IntVector& p_nodeTypes,
                            Epetra_IntVector& nodeTypes) const;

  //! Detect subdomains in which some separator variables can   
  //! be made 'interior' again. This is 3D Stokes-specific, in  
  //! which case the CartFlow partitioner provides the edge sub-
  //! domains, which form 'full conservation tubes' (FCTs). The 
  //! P-nodes and one of the velocity components in the interior
  //! of such a tube can be eliminated.
  int DetectFCT(const Epetra_CrsGraph& parallelGraph, 
                      const Epetra_IntVector& p_nodeTypes,
                            Epetra_IntVector& nodeTypes) const;
/*
  //! fix nodeType of separators around FCTs
  int CorrectFCT(const Epetra_CrsGraph& parallelGraph, 
                      const Epetra_IntVector& p_nodeTypes,
                            Epetra_IntVector& nodeTypes) const;
*/
  //! form list with interior, separator and retained nodes for subdomain
  // sd. Links separators to subdomains.
  int BuildNodeLists(int sd, const Epetra_CrsGraph& parallelGraph, 
                              const Epetra_IntVector& nodeTypes,
                              const Epetra_IntVector& p_nodeTypes,
                              Teuchos::Array<int>& interiorNodes,
                              Teuchos::Array<int>& separatorNodes,
                              Teuchos::Array<int>& retainedNodes) const;
/*
  //! BuildNodeLists cannot be used for full conservation tubes (FCTs),
  //! which occur on the edges of 3D Stokes subdomains. These should   
  //! have the entire face separators as groups so that they are con-  
  //! sistent with the subdomain grouping.
  int BuildNodeLists2(int sd, const Epetra_CrsGraph& parallelGraph, 
                              const Epetra_IntVector& nodeTypes,
                              const Epetra_IntVector& p_nodeTypes,
                              Teuchos::Array<int>& interiorNodes,
                              Teuchos::Array<int>& separatorNodes,
                              Teuchos::Array<int>& retainedNodes) const;
*/  
  //! this is a helper function to associate separators with subdomains 
  //! correctly, in particular to find separator nodes in corners which 
  //! are not directly connected to any interior nodes of the subdomain.
  int FindMissingSepNodes(int my_sd, 
                           const Epetra_CrsGraph& parallelGraph,
                           const Epetra_IntVector& p_nodeTypes,
                           const std::set<int>& input_array,
                                 std::set<int>& output_array) const;
    
  //!@}


#ifdef DEBUGGING
//! pretty-print the nodeType vector for debugging
std::ostream& PrintNodeTypeVector(const Epetra_IntVector& nt,std::ostream& os,std::string label);
#endif

  };

}
#endif
