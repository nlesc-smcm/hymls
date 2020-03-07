#ifndef HYMLS_HIERARCHICAL_MAP_H
#define HYMLS_HIERARCHICAL_MAP_H

#include "HYMLS_config.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Epetra_Map.h"

#include <iosfwd>
#include <string>

// forward declarations
class Epetra_Comm;
class Epetra_IntSerialDenseVector;
class Epetra_LongLongSerialDenseVector;

namespace HYMLS {

class InteriorGroup;
class SeparatorGroup;

//! class for hierarchically partitioned maps

/*! This class allows constructing Epetra_Maps which are sub-maps
    of a given base-map. The terminology we use comes from the   
    HYMLS application of a map partitioned into 
        - partitions (1 per proc)
        - subdomains (many per proc)
        - groups (several per subdomain
    
    The groups are again divided into 'interior' and 'separator' groups,
    the first (main) group of each subdomain is called interior
*/
class HierarchicalMap
  {
  
public:

  enum SpawnStrategy 
    {
    Interior=0,   /* retain only interior (level 0) nodes (no overlap)  */
    
    Separators=1, /* retain all separator elements as new 'interior'    */
                /* nodes, and keep separators between physical        */
                /* partitions (processors) as separators.             */
    LocalSeparators=2, /* retain all local separators, possibly regrouped  */
    All=3       /* retain all elements. This is most useful for creating */
                /* subdomain maps using SpawnMap(), for Spawn() it just  */
                /* returns this object.                                  */
    };

  // constructor - empty object
  HierarchicalMap(Teuchos::RCP<const Epetra_Map> baseMap,
                  Teuchos::RCP<const Epetra_Map> baseOverlappingMap=Teuchos::null,
                  int numMySubdomains=0,
                  std::string label="HierarchicalMap",
                  int level=1);

  //! destructor
  virtual ~HierarchicalMap();

  //! print domain decomposition to file
  std::ostream& Print(std::ostream& os) const;
  
  //! \name Functions to access the reordering defined by this class
  
  //@{
  
  //! get the local number of subdomains
  inline int NumMySubdomains() const {return groupPointer_->size();}

  //! total number of nodes in subdomain sd (interior and separators)  
  inline int NumElements(int sd) const
    {
    return static_cast<int>(
      *((*groupPointer_)[sd].end()-1) - *((*groupPointer_)[sd].begin()));
    }

  //! total number of interior nodes in subdomain sd
  inline int NumInteriorElements(int sd) const
    {
    return NumElements(sd,0);
    }

  //! total number of separator nodes in subdomain sd
  inline int NumSeparatorElements(int sd) const
    {
    return static_cast<int>(*((*groupPointer_)[sd].end()-1) - (*groupPointer_)[sd][1]);
    }

  //! number of groups in subdomain sd (interior + separator)
  inline int NumGroups(int sd) const
    {
    return (*groupPointer_)[sd].size()-1;
    }

  //! number of separator groups in subdomain sd
  inline int NumSeparatorGroups(int sd) const
    {
    return NumGroups(sd)-1;
    }

  //! total number of nodes in subdomain sd, group grp.
  //! grp 0 are the interior elements, group 1:NumSeparatorGroups are 
  //! the separator groups.
  inline int NumElements(int sd, int grp) const
    {
    return static_cast<int>((*groupPointer_)[sd][grp+1] - (*groupPointer_)[sd][grp]);
    }

  //! given a subdomain, returns a list of GIDs that belong to the subdomain
  int getSeparatorGIDs(int sd, hymls_gidx *gids) const;

#ifdef HYMLS_LONG_LONG
  //! given a subdomain, returns a list of GIDs that belong to the subdomain
  int getSeparatorGIDs(int sd, Epetra_LongLongSerialDenseVector &inds) const;
#else
  //! given a subdomain, returns a list of GIDs that belong to the subdomain
  int getSeparatorGIDs(int sd, Epetra_IntSerialDenseVector &inds) const;
#endif

  //! returns the separator groups for a certain subdomain
  InteriorGroup const &GetInteriorGroup(int sd) const;

  //! returns the separator groups for a certain subdomain
  Teuchos::Array<SeparatorGroup> const &GetSeparatorGroups(int sd) const;
  //@}

  //! creates a 'next generation' object that retains certain nodes.
  
  /*!                                                                   
                                                                        
  Currently this function allows doing the following:                   
                                                                        
  strat==Interior:    returns an object that has the same number of     
                      subdomains but only one group per subdomain,      
                      the interior nodes. The new object's Map() is     
                      a map without overlap that contains only the      
                      interior nodes of this object.                    
                                                                        
   strat==Separators: the new object contains all the local separator   
                      groups as new interior groups (each group forms   
                      the interior of exactly one subdomain). The last   
                      subdomain has a number of separator groups        
                      containing the off-processor separators connecting
                      to subdomains in the original 'this' object.      
                                                                        
    TODO: in the second case there may be a smarter implementation that 
          retains more information about the non-local separators.      
                                                                        
   strat=LocalSeparators: This object is used for constructing the      
                          orthogonal transform, it has only local sepa- 
                          rators.
                                                                        
    The 'Interior' object's Map() gives the variables to be eliminated  
    in the first place.                                                 
                                                                        
    The 'Separator' object's Map() gives the map for the Schur-         
    complement. Its indexing functions can be used to loop over the     
    rows of a sparse matrix (new interior nodes) or its columns (new    
    interior+separator nodes).                                          
                                                                        
  */
  virtual Teuchos::RCP<const HierarchicalMap> Spawn(SpawnStrategy strat) const;

  //! spawn a map containing all Interior or all Separator nodes belonging to one subdomain,
  //! or All nodes belonging to one subdomain.
  Teuchos::RCP<const Epetra_Map> SpawnMap(int sd, SpawnStrategy strat) const;
  

  //!\name data member access
  //@{  
  
  //!
  const Epetra_Comm& Comm() const {return baseMap_->Comm();}

  //!
  std::string Label() const {return label_;}
  
  //! get a reference to the non-overlapping map used inside this class
  const Epetra_Map& Map() const 
    {
    return *baseMap_;
    }

  //! get a reference to the overlapping map used inside this class
  const Epetra_Map& OverlappingMap() const 
    {
    return *overlappingMap_;
    }

  //!
  int Level() const {return myLevel_;}
  
    
  //! get a pointer to the non-overlapping map used inside this class
  Teuchos::RCP<const Epetra_Map> GetMap() const 
    {
    return baseMap_;
    }

  //! get a reference to the overlapping map used inside this class
  Teuchos::RCP<const Epetra_Map> GetOverlappingMap() const 
    {
    return overlappingMap_;
    }

protected:

  //@}
  
  //! add an interior group of GIDs to an existing subdomain.
  //! FillComplete() should not have been called.
  int AddInteriorGroup(int sd, InteriorGroup const &group);

  //! add a separator group of GIDs to an existing subdomain. Returns the group id
  //! of the new group. FillComplete() should not have been called.
  int AddSeparatorGroup(int sd, SeparatorGroup const &group);
  
  //! get back a group that was added as described above. Used for debugging purposes
  //! only
  Teuchos::Array<hymls_gidx> GetGroup(int sd, int grp) const;
  
  //! delete the map and all subdomains and groups that have been added so far
  int Reset(int num_sd);

  //! finalize the setup procedure by building the map
  int FillComplete();
  
  //! indicates if any more changes can be made (FillComplete() has been called)
  bool Filled() const {return overlappingMap_!=Teuchos::null;}

  //! label
  std::string label_;
  
  //! level ID
  int myLevel_;
  
  //! initial map (p0)
  Teuchos::RCP<const Epetra_Map> baseMap_;

  //! initial overlapping map from the previous level
  Teuchos::RCP<const Epetra_Map> baseOverlappingMap_;

private:

  //! overlapping map p1 (with minimal overlap between subdomains)
  Teuchos::RCP<const Epetra_Map> overlappingMap_;
  
  //! pointer to index subdomains and groups
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<hymls_gidx> > > groupPointer_;
  
  //! list of ordered GIDs which will be transformed into a map in FillComplete
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<hymls_gidx> > > gidList_;

  //! list of interior groups per subdomain
  Teuchos::RCP<Teuchos::Array<InteriorGroup> > interior_groups_;

  //! list of separator groups per subdomain
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<SeparatorGroup> > > separator_groups_;

  //! array of spawned objects (so we avoid building the same thing over and over again)
  mutable Teuchos::Array<Teuchos::RCP<const HierarchicalMap> > spawnedObjects_;
  
  //! array of spawned maps
  mutable Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Epetra_Map> > > spawnedMaps_;
  
  //! protected constructor - does not allow any more changes
  //! (FillComplete() has been called), this is used for spawning
  //! objects like a map with all separators etc.
  HierarchicalMap(
    Teuchos::RCP<const Epetra_Map> baseMap,
    Teuchos::RCP<const Epetra_Map> overlappingMap,
    Teuchos::RCP<Teuchos::Array< Teuchos::Array<hymls_gidx> > > groupPointer,
    Teuchos::RCP<Teuchos::Array< Teuchos::Array<hymls_gidx> > > gidList,
    Teuchos::RCP<Teuchos::Array<InteriorGroup> > interior_groups,
    Teuchos::RCP<Teuchos::Array<Teuchos::Array<SeparatorGroup> > > separator_groups,
    std::string label, int level);
  
  //! \name private member functions
  //! @{

  //!
  Teuchos::RCP<const HierarchicalMap> SpawnInterior() const;
  //!
  Teuchos::RCP<const HierarchicalMap> SpawnSeparators() const;
  //!
  Teuchos::RCP<const HierarchicalMap> SpawnLocalSeparators() const;

  //@}    
  };

std::ostream & operator<<(std::ostream& os, const HierarchicalMap& h);

}
#endif
