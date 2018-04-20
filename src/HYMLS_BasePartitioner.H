#ifndef HYMLS_BASE_PARTITONER_H
#define HYMLS_BASE_PARTITONER_H

#include "HYMLS_config.h"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "GaleriExt_Periodic.h"

class Epetra_Map;

namespace Teuchos {
class ParameterList;
  }

namespace HYMLS {

/*! Base class for partitioning in HYMLS - on this 
  partitioning we build our HID.
*/

class BasePartitioner
  {
  public:

  //! constructor
  BasePartitioner(){}
  
  //! destructor
  virtual ~BasePartitioner(){}

  //! set parameters for the partitioner like separator length
  virtual void SetParameters(Teuchos::ParameterList& params);

  //! get a pararmeterlist with increased separator lengths for
  //! the next level
  virtual void SetNextLevelParameters(Teuchos::ParameterList& params);
  
  //! partition an [nx x ny x nz] grid with one DoF per node
  //! into npx*npy*npz global subdomains. If repart==true,  
  //! the partitioner is allowed to change the global       
  //! distribution of the nodes, otherwise it may only      
  //! partition the owned nodes on each processor.          
  virtual int Partition(bool repart) = 0;

  //! Get interior and separator groups of the subdomain sd
  virtual int GetGroups(int sd, Teuchos::Array<hymls_gidx> &interior_nodes,
    Teuchos::Array<Teuchos::Array<hymls_gidx> > &separator_nodes) = 0;
  
  //! get number of local partitions
  virtual int NumLocalParts() const = 0;
  
  //! is this class fully set up?
  virtual bool Partitioned() const = 0;
  
  //! return the number of variables per grid point
  //! (default implementation returns 1)
  virtual int DofPerNode() const {return dof_;}
  
  //! for problems with multiple dof per node: get an
  //! integer indicating which variable type a gid has
  //! (this is used in HID to group separators by type)
  //! default implementation returns 0
  virtual int VariableType(hymls_gidx gid) const {return variableType_[gid % dof_];}
  
  //! get non-overlapping global subdomain id 
  virtual int operator()(hymls_gidx gid) const = 0;
  
  //! get the (re-)partitioned map (here elements belonging
  //! to a subdomain have contiguous local indexing)
  virtual const Epetra_Map& Map() const = 0;
  
  //! get the map with global IDs of the subdomains
  virtual const Epetra_Map& SubdomainMap() const = 0;
  
  //! get the (re-)partitioned map (here elements belonging
  //! to a subdomain have contiguous local indexing)
  virtual Teuchos::RCP<const Epetra_Map> GetMap() const 
    {return Teuchos::rcp(&Map(),false);}

  //! global grid size
  hymls_gidx nx_, ny_, nz_;

  //! subdomain size
  int sx_,sy_,sz_;

  //! coarsening factor
  int cx_,cy_,cz_;

  //! dimension of the problem
  int dim_;

  //! number of variables per node
  int dof_;

  //! type of periodicity in the problem
  GaleriExt::PERIO_Flag perio_;

  //! type of the variables per node
  Teuchos::Array<int> variableType_;

  };

}
#endif
