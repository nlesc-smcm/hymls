#ifndef HYMLS_BASE_PARTITONER_H
#define HYMLS_BASE_PARTITONER_H

#include "HYMLS_config.h"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "GaleriExt_Periodic.h"

class Epetra_Map;
class Epetra_Comm;

namespace Teuchos {
class ParameterList;
  }

namespace HYMLS {

class SeparatorGroup;

/*! Base class for partitioning in HYMLS - on this
  partitioning we build our HID.
*/

class BasePartitioner
  {
protected:
  enum class VariableType
    {
    Velocity_U,
    Velocity_V,
    Velocity_W,
    Pressure,
    Interior
    };

public:

  //! constructor
  BasePartitioner(Epetra_Comm const &comm, int level);

  //! destructor
  virtual ~BasePartitioner(){}

  //! set parameters for the partitioner like separator length
  virtual void SetParameters(Teuchos::ParameterList& params);

  //! get a pararmeterlist with increased separator lengths for
  //! the next level
  virtual void SetNextLevelParameters(Teuchos::ParameterList& params) const;

  //! partition an [nx x ny x nz] grid with one DoF per node
  //! into npx*npy*npz global subdomains. If repart==true,
  //! the partitioner is allowed to change the global
  //! distribution of the nodes, otherwise it may only
  //! partition the owned nodes on each processor.
  virtual int Partition(bool repart) = 0;

  //! Get interior and separator groups of the subdomain sd
  virtual int GetGroups(int sd, Teuchos::Array<hymls_gidx> &interior_nodes,
    Teuchos::Array<SeparatorGroup> &separator_groups) const = 0;

  //! get number of local partitions
  virtual int NumLocalParts() const = 0;

  //! get number of global partitions
  virtual int NumGlobalParts(int sx, int sy, int sz) const = 0;

  //! is this class fully set up?
  virtual bool Partitioned() const = 0;

  //! get non-overlapping global subdomain id
  virtual int operator()(hymls_gidx gid) const = 0;

  //! get the (re-)partitioned map (here elements belonging
  //! to a subdomain have contiguous local indexing)
  virtual const Epetra_Map& Map() const {return *GetMap();}

  //! get the map with global IDs of the subdomains
  virtual const Epetra_Map& SubdomainMap() const = 0;

  //! get the (re-)partitioned map (here elements belonging
  //! to a subdomain have contiguous local indexing)
  virtual Teuchos::RCP<const Epetra_Map> GetMap() const = 0;

  //! move every local part of the map to a different processor
  virtual Teuchos::RCP<const Epetra_Map> MoveMap(
    Teuchos::RCP<const Epetra_Map> baseMap) const;

protected:

  //! Get the position of the first node of the subdomain
  virtual int GetSubdomainPosition(int sd, int sx, int sy, int sz, int &x, int &y, int &z) const = 0;

  //! Get the subdomain id from the position of a node in the subdomain
  virtual int GetSubdomainID(int sx, int sy, int sz, int x, int y, int z) const = 0;

  //! Create a map of what processor a subdomain belongs to
  virtual int CreatePIDMap();

  //! Repartitioning may occur for two reasons, typically on coarser levels:
  //! a) the number of subdomains becomes smaller than the number of processes,
  //! b) the subdomains can't be nicely distributed among the processes.
  //! In both cases some processes are deactivated.
  virtual Teuchos::RCP<const Epetra_Map> RepartitionMap(
    Teuchos::RCP<const Epetra_Map> baseMap) const;

  //! Set the destination pid on the coarser grid
  virtual int SetDestinationPID(
    Teuchos::RCP<const Epetra_Map> baseMap);

  //! get processor on which a grid point is located
  virtual int PID(hymls_gidx gid) const;

  //! get processor on which a grid point is located
  virtual int PID(int i, int j, int k) const;

  //! communicator
  Teuchos::RCP<const Epetra_Comm> comm_;

  //! level
  int myLevel_;

  //! global grid size
  int nx_, ny_, nz_;

  //! subdomain size
  int sx_,sy_,sz_;

  //! coarsening factor
  int cx_,cy_,cz_;

  //! amount of nodes retained per separator
  int rx_,ry_,rz_;

  //! dimension of the problem
  int dim_;

  //! number of variables per node
  int dof_;

  //! number of processors in use
  int nprocs_;

  //! number of pressure nodes to retain
  int retainPressures_;

  //! type of periodicity in the problem
  GaleriExt::PERIO_Flag perio_;

  //! type of the variables per node
  Teuchos::Array<VariableType> variableType_;

  //! map of what processor a subdomain belongs to
  Teuchos::RCP<Teuchos::Array<int> > pidMap_;

  //! pid which all nodes on this processor have to be moved to
  mutable int destinationPID_;

  };

  }
#endif
