#ifndef HYMLS_CARTESIAN_PARTITONER_H
#define HYMLS_CARTESIAN_PARTITONER_H

#include "HYMLS_BasePartitioner.hpp"

#include "HYMLS_config.h"
#include "Teuchos_RCP.hpp"

class Epetra_Comm;
class Epetra_Map;

namespace Teuchos {
class ParameterList;
template <typename T> class Array;
  }

namespace HYMLS {

class CartesianPartitioner : public BasePartitioner
  {
public:

  //! constructor
  CartesianPartitioner(Teuchos::RCP<const Epetra_Map> map,
    Teuchos::RCP<Teuchos::ParameterList> const &params,
    Epetra_Comm const &comm);

  //! destructor
  virtual ~CartesianPartitioner();

  //! return global partition ID of a cell (i,j,k)
  int operator()(int i, int j, int k) const;

  //! get non-overlapping subdomain id
  int operator()(hymls_gidx gid) const;

  //! partition an [nx x ny x nz] grid with one DoF per node
  //! into subdomains with separator length sx. If repart=true,
  //! the map may need repartitioning to match the cartesian
  //! layout of the new partitioned map. Otherwise it is
  //! assumed that the map already has a global cartesian
  //! processor partitioning
  int Partition(bool repart=true);

protected:

  //! Get the position of the first node of the subdomain
  int GetSubdomainPosition(int sd, int sx, int sy, int sz, int &x, int &y, int &z) const;

  //! Get the subdomain id from the position of a node in the subdomain
  int GetSubdomainID(int sx, int sy, int sz, int x, int y, int z) const;

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
    Teuchos::Array<Teuchos::Array<hymls_gidx> > &separator_nodes) const;

  //! is this class fully set up?
  inline bool Partitioned() const
    {
    return cartesianMap_ != Teuchos::null;
    }

  //! return the repartitioned/reordered map
  inline Teuchos::RCP<const Epetra_Map> GetMap() const
    {
    return cartesianMap_;
    }

  //! return the repartitioned/reordered map
  inline const Epetra_Map& SubdomainMap() const
    {
    return *sdMap_;
    }

  //! return the number of subdomains in this proc partition
  int NumLocalParts() const;

  //! return the global number of subdomains
  int NumGlobalParts(int sx, int sy, int sz) const;

protected:

  //! label
  std::string label_;

  //! original non-overlapping map
  Teuchos::RCP<const Epetra_Map> baseMap_;

  //! non-overlapping cartesian map (all subdomains are owned by
  //! exactly one process/belong to only one partition)
  Teuchos::RCP<const Epetra_Map> cartesianMap_;

  //! number of subdomains on this proc
  int numLocalSubdomains_;

  //! maps global to local subdomain ID
  Teuchos::RCP<Epetra_Map> sdMap_;
  };

  }
#endif
