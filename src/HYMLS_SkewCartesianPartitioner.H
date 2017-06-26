#ifndef HYMLS_SKEW_CARTESIAN_PARTITONER_H
#define HYMLS_SKEW_CARTESIAN_PARTITONER_H

#include "HYMLS_config.h"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "HYMLS_BasePartitioner.H"

#include "GaleriExt_Periodic.h"

class Epetra_Comm;
class Epetra_Import;
class Epetra_Map;

namespace HYMLS {

class SkewCartesianPartitioner : public BasePartitioner
  {
public:

  //! constructor
  SkewCartesianPartitioner(Teuchos::RCP<const Epetra_Map> map, int nx, int ny,
    int nz=1, int dof=1, int pvar=-1, GaleriExt::PERIO_Flag perio=GaleriExt::NO_PERIO);

  //! destructor
  virtual ~SkewCartesianPartitioner();

  //! return global partition ID of a cell (i,j,k)
  int operator()(int i, int j, int k) const;

  //! get non-overlapping subdomain id
  int operator()(int gid) const;

  //! partition an [nx x ny x nz] grid with (possibly) several DoF per node
  //! into nparts global subdomains.
  int Partition(int nparts, bool repart=true);

  //! partition an [nx x ny x nz] grid with one DoF per node
  //! into npx*npy*npz global subdomains. If repart=true,
  //! the map may need repartitioning to match the cartesian
  //! layout of the new partitioned map. Otherwise it is
  //! assumed that the map already has a global cartesian
  //! processor partitioning (as generated i.e. by the
  //! MatrixUtils::CreateMap functions).
  int Partition(int npx, int npy, int npz, bool repart=true);

protected:
  //! creates the map from global to local partition IDs. The implementation
  //! may assume that npx_, sx_ etc. are already set so that operator() works.
  int CreateSubdomainMap();

  std::vector<std::vector<int> > buildMatrix(int c_11, int c_1n, int c_m1,
    int colskip, int rowskip, int offset, int var) const;

  int removeOverlappingVNodes(std::vector<std::vector<int> > &plane, int type) const;

  std::vector<std::vector<std::vector<std::vector<int> > > > domainTemplate(char type) const;

  int getDomainInfo(int sd, int &x, int &y, int &z, char &type) const;

  std::vector<int> template2list(
    std::vector<std::vector<std::vector<std::vector<int> > > > const &temp) const;

  std::vector<std::vector<int> > solveGroups(char type,
  std::vector<std::vector<std::vector<std::vector<int> > > > const &tempX,
  std::vector<std::vector<std::vector<std::vector<int> > > > const &tempY,
  std::vector<std::vector<std::vector<std::vector<int> > > > const &tempZ,
    int directions[3]) const;

  std::vector<std::vector<int> > createSubdomain(
    int x, int y, int z, char type,
    std::vector<std::vector<std::vector<std::vector<int> > > > temp,
    std::vector<std::vector<int> > groups) const;

public:

  //! Get interior and separator groups of the subdomain sd
  int GetGroups(int sd, Teuchos::Array<int> &interior_nodes,
    Teuchos::Array<Teuchos::Array<int> > &separator_nodes);

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
  int VariableType(int gid) const;

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

  //! Subdomain templates
  std::vector<std::vector<std::vector<std::vector<int> > > > templateX_,
    templateY_, templateZ_;

  //! Subdomain template groups
  std::vector<std::vector<int> > groupsX_, groupsY_, groupsZ_;

  //! global grid size
  int nx_, ny_, nz_;

  //! number of subdomains
  int npx_, npy_, npz_;

  //! processor distribution
  int nprocs_, nprocx_, nprocy_, nprocz_;

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

  //! pointers into map
  Teuchos::Array<int> subdomainPointer_;

protected:

  //! get processor on which a grid point is located
  int PID(int i, int j, int k) const;

  };

  }
#endif
