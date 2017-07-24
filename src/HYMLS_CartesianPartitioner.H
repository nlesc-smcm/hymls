#ifndef HYMLS_CARTESIAN_PARTITONER_H
#define HYMLS_CARTESIAN_PARTITONER_H

#include "HYMLS_config.h"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "HYMLS_BasePartitioner.H"

#include "GaleriExt_Periodic.h"

class Epetra_Comm;
class Epetra_Import;
class Epetra_Map;

namespace HYMLS {

class CartesianPartitioner : public BasePartitioner
  {
public:

  //! constructor
  CartesianPartitioner(Teuchos::RCP<const Epetra_Map> map, int nx, int ny,
    int nz=1, int dof=1, int pvar=-1, GaleriExt::PERIO_Flag perio=GaleriExt::NO_PERIO);

  //! destructor
  virtual ~CartesianPartitioner();

  //! return global partition ID of a cell (i,j,k)
  int operator()(int i, int j, int k) const;

  //! get non-overlapping subdomain id
  int operator()(hymls_gidx gid) const;

  //! partition an [nx x ny x nz] grid with (possibly) several DoF per node
  //! into nparts global subdomains.
  int Partition(int nparts, bool repart=true);

  //! partition an [nx x ny x nz] grid with one DoF per node
  //! into subdomains with separator length sx. If repart=true,
  //! the map may need repartitioning to match the cartesian
  //! layout of the new partitioned map. Otherwise it is
  //! assumed that the map already has a global cartesian
  //! processor partitioning
  int Partition(int sx, int sy, int sz, bool repart=true);

private:
  //! creates the map from global to local partition IDs. The implementation
  //! may assume that npx_, sx_ etc. are already set so that operator() works.
  int CreateSubdomainMap();

  //! Method to remove separators at the boundary when they are not
  //! strictly needed. This is when there are no periodic boundary
  //! conditions
  int RemoveBoundarySeparators(Teuchos::Array<hymls_gidx> &interior_nodes,
    Teuchos::Array<Teuchos::Array<hymls_gidx> > &separator_nodes) const;

public:
  //! Get interior and separator groups of the subdomain sd
  int GetGroups(int sd, Teuchos::Array<hymls_gidx> &interior_nodes,
    Teuchos::Array<Teuchos::Array<hymls_gidx> > &separator_nodes);

  //! is this class fully set up?
  inline bool Partitioned() const
    {
    return npx_ > 0;
    }

  //! get the type of a variable (if more than 1 dof per node, otherwise just 0)
  inline int DofPerNode() const
    {
    return dof_;
    }

  //! get the type of a variable (if more than 1 dof per node, otherwise just 0)
  int VariableType(hymls_gidx gid) const;

  //! return the repartitioned/reordered map
  inline const Epetra_Map& Map() const
    {
    return *cartesianMap_;
    }

  //! return the repartitioned/reordered map
  inline const Epetra_Map& SubdomainMap() const
    {
    return *sdMap_;
    }

  //! return number of subdomains in this proc partition
  int NumLocalParts() const;

protected:

  //!
  std::string label_;

  //! communicator
  Teuchos::RCP<const Epetra_Comm> comm_;

  //! original non-overlapping map
  Teuchos::RCP<const Epetra_Map> baseMap_;

  //! non-overlapping cartesian map (all subdomains are owned by
  //! exactly one process/belong to only one partition)
  Teuchos::RCP<const Epetra_Map> cartesianMap_;

  //! global grid size
  hymls_gidx nx_, ny_, nz_;

  //! number of subdomains
  int npx_,npy_,npz_;

  //! processor distribution
  int nprocx_,nprocy_,nprocz_;

  //! number of subdomains on this proc
  int numLocalSubdomains_;

  //! subdomain size
  int sx_,sy_,sz_;

  //! number of variables per node
  int dof_;

  //! pressure node
  int pvar_;

  //! maps global to local subdomain ID
  Teuchos::RCP<Epetra_Map> sdMap_;

  //! indicates wether the processor is active (owns any nodes/subdomains)
  bool active_;

  //! type of periodicity in the problem
  GaleriExt::PERIO_Flag perio_;

protected:

  //! get processor on which a grid point is located
  int PID(int i, int j, int k) const;
  };

  }
#endif
