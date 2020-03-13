#include "HYMLS_HierarchicalMap.hpp"

#include "HYMLS_config.h"

#include "HYMLS_Tools.hpp"
#include "HYMLS_Macros.hpp"
#include "HYMLS_InteriorGroup.hpp"
#include "HYMLS_SeparatorGroup.hpp"

#include "Epetra_Comm.h"
#include "Epetra_SerialComm.h"
#include "Epetra_Map.h"
#include "Epetra_IntVector.h"
#include "Epetra_Import.h"
#include "Epetra_IntSerialDenseVector.h"
#include "Epetra_LongLongSerialDenseVector.h"

#include <iostream>
#include <algorithm>

namespace HYMLS {

//empty constructor
HierarchicalMap::HierarchicalMap(
  Teuchos::RCP<const Epetra_Map> baseMap,
  Teuchos::RCP<const Epetra_Map> baseOverlappingMap,
  int numMySubdomains,
  std::string label, int level)
  :
  label_(label),
  myLevel_(level),
  baseMap_(baseMap),
  baseOverlappingMap_(baseOverlappingMap),
  overlappingMap_(Teuchos::null)
  {
  HYMLS_LPROF2(label_,"Constructor");
  Reset(numMySubdomains);
  }

//private constructor
HierarchicalMap::HierarchicalMap(
  Teuchos::RCP<const Epetra_Map> baseMap,
  Teuchos::RCP<const Epetra_Map> overlappingMap,
  Teuchos::RCP<Teuchos::Array<InteriorGroup> > interior_groups,
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<SeparatorGroup> > > separator_groups,
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<Teuchos::Array<SeparatorGroup> > > > linked_separator_groups,
  std::string label, int level)
  :
  label_(label),
  myLevel_(level),
  baseMap_(baseMap),
  baseOverlappingMap_(overlappingMap),
  overlappingMap_(overlappingMap),
  interior_groups_(interior_groups),
  separator_groups_(separator_groups),
  linked_separator_groups_(linked_separator_groups)
  {
  HYMLS_LPROF2(label_,"HierarchicalMap Constructor");
  spawnedObjects_.resize(3); // can currently spawn Interior, Separator and LocalSeparator objects
  spawnedMaps_.resize(3);
  for (int i = 0; i < spawnedObjects_.size(); i++)
    spawnedObjects_[i] = Teuchos::null;
  for (int i = 0; i < spawnedMaps_.size(); i++)
    {
    spawnedMaps_[i].resize(NumMySubdomains());
    for (int sd = 0; sd < NumMySubdomains(); sd++)
      spawnedMaps_[i][sd] = Teuchos::null;
    }
  }

HierarchicalMap::~HierarchicalMap()
  {
  HYMLS_LPROF3(label_,"Destructor");
  }

int HierarchicalMap::NumMySubdomains() const
  {
  if (interior_groups_ != Teuchos::null)
    return interior_groups_->length();
  return separator_groups_->length();
  }

int HierarchicalMap::NumInteriorElements(int sd) const
  {
  return GetInteriorGroup(sd).length();
  }

int HierarchicalMap::NumSeparatorElements(int sd) const
  {
  int num = 0;
  for (SeparatorGroup const &group: GetSeparatorGroups(sd))
    num += group.length();
  return num;
  }

int HierarchicalMap::NumSeparatorGroups(int sd) const
  {
  return GetSeparatorGroups(sd).length();
  }

int HierarchicalMap::NumLinkedSeparatorGroups(int sd) const
  {
  return GetLinkedSeparatorGroups(sd).length();
  }

int HierarchicalMap::Reset(int numMySubdomains)
  {
  HYMLS_LPROF2(label_, "Reset");
  interior_groups_ = Teuchos::rcp(new Teuchos::Array<InteriorGroup>(numMySubdomains));
  separator_groups_ = Teuchos::rcp(new Teuchos::Array<Teuchos::Array<SeparatorGroup> >(numMySubdomains));

  spawnedObjects_.resize(3); // can currently spawn Interior, Separator and LocalSeparator objects
  spawnedMaps_.resize(3);
  for (int i = 0; i < spawnedObjects_.size(); i++)
    spawnedObjects_[i] = Teuchos::null;
  overlappingMap_ = Teuchos::null;
  return 0;
  }

int HierarchicalMap::LinkSeparators(
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<SeparatorGroup> > > separator_groups,
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<Teuchos::Array<SeparatorGroup> > > > linked_separator_groups) const
  {
  for (int sd = 0; sd < NumMySubdomains(); sd++)
    for (SeparatorGroup const &group: (*separator_groups)[sd])
      {
      bool found = false;
      for (auto &linked_groups: (*linked_separator_groups)[sd])
        if (group.type() == linked_groups[0].type())
          {
          linked_groups.append(group);
          found = true;
          break;
          }
      if (!found)
        (*linked_separator_groups)[sd].append(Teuchos::Array<SeparatorGroup>(1, group));
      }
  return 0;
  }

int HierarchicalMap::FillComplete()
  {
  HYMLS_LPROF2(label_,"FillComplete");
  for (int i = 0; i < spawnedObjects_.size(); i++)
    spawnedObjects_[i] = Teuchos::null;

  for (int i = 0; i < spawnedMaps_.size(); i++)
    {
    spawnedMaps_[i].resize(NumMySubdomains());
    for (int sd = 0; sd < NumMySubdomains(); sd++)
      {
      spawnedMaps_[i][sd] = Teuchos::null;
      }
    }

  Teuchos::RCP<const Epetra_Map> map = baseMap_;
  if (baseOverlappingMap_ != Teuchos::null)
    map = baseOverlappingMap_;

  // Merge all separator GIDs on this processor into one list.
  // Put all interior GIDs that are present in the baseMap_ in the
  // newGidList.
  Teuchos::Array<hymls_gidx> separatorGIDs;
  for (int sd = 0; sd < NumMySubdomains(); sd++)
    {
    // Interior nodes don't need communication. Just add those
    // that are present in the baseMap_
    InteriorGroup new_group;
    for (hymls_gidx gid: GetInteriorGroup(sd).nodes())
      if (map->MyGID(gid))
        new_group.append(gid);
    (*interior_groups_)[sd].nodes() = new_group.nodes();

    // Now add the separator groups. We can avoid communication
    // if the baseOverlappingMap_ is present.
    for (SeparatorGroup &group: (*separator_groups_)[sd])
      {
      if (baseOverlappingMap_ != Teuchos::null)
        {
        SeparatorGroup new_group;
        for (hymls_gidx gid: group.nodes())
          if (map->MyGID(gid))
            new_group.append(gid);
        group.nodes() = new_group.nodes();
        }
      else
        std::copy(group.nodes().begin(), group.nodes().end(),
          std::back_inserter(separatorGIDs));
      }
    }

  // Communication is only required if there is no overlapping map
  // present already.
  if (baseOverlappingMap_ == Teuchos::null)
    {
    // Make sure there is only one entry of each of them.
    std::sort(separatorGIDs.begin(), separatorGIDs.end());
    auto end = std::unique(separatorGIDs.begin(), separatorGIDs.end());

    // Communicate between processors which elements are actually
    // present in the baseMap_ of the processor that owns the nodes.
    // This is because the Partitioner gives us all possible elements
    // belonging to the subdomain, not only the elements that are in
    // the map.
    int numElements = std::distance(separatorGIDs.begin(), end);
    Teuchos::RCP<Epetra_Map> tmpOverlappingMap =
      Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1), numElements, separatorGIDs.getRawPtr(),
          (hymls_gidx)baseMap_->IndexBase64(), Comm()));

    HYMLS_DEBVAR(*tmpOverlappingMap);

    Epetra_IntVector vec(*baseMap_);
    for (auto i = separatorGIDs.begin(); i != end; ++i)
      {
      int lid = baseMap_->LID(*i);
      if (lid != -1)
        vec[lid] = 1;
      }

    separatorGIDs.clear();

    Epetra_Import imp(*tmpOverlappingMap, *baseMap_);
    Epetra_IntVector overlappingVec(*tmpOverlappingMap);
    overlappingVec.Import(vec, imp, Insert);

    for (int sd = 0; sd < NumMySubdomains(); sd++)
      {
      for (SeparatorGroup &group: (*separator_groups_)[sd])
        {
        SeparatorGroup new_group;
        for (hymls_gidx gid: group.nodes())
          {
          // If it is present in the overlappingVec the element actually belongs
          // to the baseMap_ on some processor
          if (overlappingVec[tmpOverlappingMap->LID(gid)])
            new_group.append(gid);
          }
        group.nodes() = new_group.nodes();
        }
      }
    }

  // Make a new overlapping map with elements that are present on some processor
  Teuchos::Array<hymls_gidx> allGIDs;
  for (int sd = 0; sd < NumMySubdomains(); sd++)
    {
    // Remove empty separator groups
    (*separator_groups_)[sd].erase(std::remove_if(
        (*separator_groups_)[sd].begin(), (*separator_groups_)[sd].end(),
      [](SeparatorGroup const &i){return i.nodes().empty();}),
      (*separator_groups_)[sd].end());

    InteriorGroup const &group = GetInteriorGroup(sd);
    std::copy(group.nodes().begin(), group.nodes().end(), std::back_inserter(allGIDs));

    for (SeparatorGroup const &group: GetSeparatorGroups(sd))
      std::copy(group.nodes().begin(), group.nodes().end(), std::back_inserter(allGIDs));
    }

  std::sort(allGIDs.begin(), allGIDs.end());
  auto last = std::unique(allGIDs.begin(), allGIDs.end());

  overlappingMap_ = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1), std::distance(allGIDs.begin(), last),
      allGIDs.getRawPtr(), (hymls_gidx)baseMap_->IndexBase64(), Comm()));

  // Link together separator groups that have the same type, e.g. when they
  // are on the same separator.
  linked_separator_groups_ = Teuchos::rcp(
    new Teuchos::Array<Teuchos::Array<Teuchos::Array<SeparatorGroup> > >(NumMySubdomains()));

  LinkSeparators(separator_groups_, linked_separator_groups_);

  return 0;
  }

int HierarchicalMap::AddInteriorGroup(int sd, InteriorGroup const &group)
  {
  HYMLS_LPROF3(label_,"AddInteriorGroup");

  if (sd >= interior_groups_->size())
    {
    Tools::Warning("invalid subdomain index", __FILE__, __LINE__);
    return -1; // You should Reset with the right amount of sd
    }

  HYMLS_DEBVAR(sd);
  HYMLS_DEBVAR(group.nodes());

  (*interior_groups_)[sd] = group;

  return 0;
  }

int HierarchicalMap::AddSeparatorGroup(int sd, SeparatorGroup const &group)
  {
  HYMLS_LPROF3(label_,"AddSeparatorGroup");

  if (sd >= separator_groups_->size())
    {
    Tools::Warning("invalid subdomain index", __FILE__, __LINE__);
    return -1; // You should Reset with the right amount of sd
    }

  HYMLS_DEBVAR(sd);
  HYMLS_DEBVAR(group.nodes());

  (*separator_groups_)[sd].append(group);

  return (*separator_groups_)[sd].length() - 1;
  }

InteriorGroup const &HierarchicalMap::GetInteriorGroup(int sd) const
  {
  return (*interior_groups_)[sd];
  }

Teuchos::Array<SeparatorGroup> const &HierarchicalMap::GetSeparatorGroups(int sd) const
  {
  return (*separator_groups_)[sd];
  }

Teuchos::Array<Teuchos::Array<SeparatorGroup> > const &HierarchicalMap::GetLinkedSeparatorGroups(int sd) const
  {
  return (*linked_separator_groups_)[sd];
  }

//! print domain decomposition to file
std::ostream& HierarchicalMap::Print(std::ostream& os) const
  {
  if (!Filled())
    {
    os << "(object not filled)" << std::endl;
    return os;
    }
  HYMLS_LPROF3(label_,"Print");
  int rank=Comm().MyPID();

  if (rank==0)
    {
    os << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    os << "% Domain decomposition and separators, level " << myLevel_ << "       %" << std::endl;
    os << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
    os << std::endl;
    }

  Comm().Barrier();

  for (int proc = 0; proc < Comm().NumProc(); proc++)
    {
    if (proc == rank)
      {
      os << "%Partition " << rank << std::endl;
      os << "%=============" << std::endl;
      for (int sd = 0; sd < NumMySubdomains(); sd++)
        {
        os << "p{" << myLevel_ << "}{" << rank + 1 << "}.groups{" << sd + 1 << "} = {";

        os << "[";
        InteriorGroup const &group = GetInteriorGroup(sd);
        for (hymls_gidx gid: group.nodes())
          os << gid << ",";
        os << "]";

        for (SeparatorGroup const &group: GetSeparatorGroups(sd))
          {
          os << ",..." << std::endl;
          os << "[";
          for (hymls_gidx gid: group.nodes())
            os << gid << ",";
          os << "]";
          }
        os << "};\n" << std::endl;
        }
      }//rank
    Comm().Barrier();
    }//proc

  if (rank==0)
    {
    os << "%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$%" << std::endl;
    }
  return os;
  }

Teuchos::RCP<const HierarchicalMap>
HierarchicalMap::Spawn(SpawnStrategy strat) const
  {
  if (!Filled())
    Tools::Error("object not filled",__FILE__,__LINE__);

  int idx = (int)strat;

  if (spawnedObjects_.size() < idx + 1)
    {
    Tools::Error("Bad strategy!", __FILE__, __LINE__);
    }

  Teuchos::RCP<const HierarchicalMap> object = spawnedObjects_[idx];

  if (object == Teuchos::null)
    {
    HYMLS_LPROF3(label_,"Spawn");
    if (strat == Interior)
      {
      object = SpawnInterior();
      }
    else if (strat == Separators)
      {
      object = SpawnSeparators();
      }
    else if (strat == LocalSeparators)
      {
      object = SpawnLocalSeparators();
      }
    else
      {
      Tools::Error("Bad strategy!",__FILE__,__LINE__);
      }
    spawnedObjects_[idx] = object;
    }
  return object;
  }


Teuchos::RCP<const HierarchicalMap>
HierarchicalMap::SpawnInterior() const
  {
  HYMLS_LPROF3(label_, "SpawnInterior");

  Teuchos::RCP<const HierarchicalMap> newObject = Teuchos::null;
  Teuchos::RCP<Epetra_Map> newMap = Teuchos::null;

  hymls_gidx base = baseMap_->IndexBase64();

  int num_interior_elements = 0;
  for (int sd = 0; sd < NumMySubdomains(); sd++)
    num_interior_elements += GetInteriorGroup(sd).length();

  hymls_gidx *myElements = new hymls_gidx[num_interior_elements];
  int pos = 0;
  for (int sd = 0; sd < NumMySubdomains(); sd++)
    {
    InteriorGroup const &group = GetInteriorGroup(sd);
    std::copy(group.nodes().begin(), group.nodes().end(), myElements + pos);
    pos += group.length();
    }

  newMap = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1), pos, myElements, base, Comm()));
  delete [] myElements;

  newObject = Teuchos::rcp(new HierarchicalMap(newMap, newMap,
      interior_groups_, Teuchos::null, Teuchos::null, "Interior Nodes", myLevel_));

  return newObject;
  }

/////////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<const HierarchicalMap>
HierarchicalMap::SpawnSeparators() const
  {
  HYMLS_LPROF3(label_, "SpawnSeparators");

  if (!Filled())
    Tools::Error("object not filled", __FILE__, __LINE__);

  Teuchos::RCP<const HierarchicalMap> newObject = Teuchos::null;
  Teuchos::RCP<Epetra_Map> newMap = Teuchos::null;
  Teuchos::RCP<Epetra_Map> newOverlappingMap = Teuchos::null;
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<SeparatorGroup> > > new_separator_groups =
    Teuchos::rcp(new Teuchos::Array<Teuchos::Array<SeparatorGroup> >(NumMySubdomains()));
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<Teuchos::Array<SeparatorGroup> > > > new_linked_separator_groups =
    Teuchos::rcp(new Teuchos::Array<Teuchos::Array<Teuchos::Array<SeparatorGroup> > >(NumMySubdomains()));

  Teuchos::Array<hymls_gidx> done;
  Teuchos::Array<hymls_gidx> localGIDs;
  Teuchos::Array<hymls_gidx> overlappingGIDs;

  for (int sd = 0; sd < NumMySubdomains(); sd++)
    {
    for (SeparatorGroup const &group: GetSeparatorGroups(sd))
      {
      hymls_gidx first_node = group[0];
      if (std::find(done.begin(), done.end(), first_node) == done.end())
        {
        for (hymls_gidx gid: group.nodes())
          {
          overlappingGIDs.append(gid);
          if (baseMap_->MyGID(gid))
            localGIDs.append(gid);
          }
        (*new_separator_groups)[sd].append(group);
        done.append(first_node);
        }
      }
    }

  newOverlappingMap = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1), overlappingGIDs.size(),
      overlappingGIDs.getRawPtr(), (hymls_gidx)baseMap_->IndexBase64(), Comm()));

  newMap = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1), localGIDs.size(),
      localGIDs.getRawPtr(), (hymls_gidx)baseMap_->IndexBase64(), Comm()));

  LinkSeparators(new_separator_groups, new_linked_separator_groups);

  newObject = Teuchos::rcp(new HierarchicalMap(newMap, newOverlappingMap,
      Teuchos::null, new_separator_groups, new_linked_separator_groups, "Separator Nodes", myLevel_));

  return newObject;
  }

////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<const HierarchicalMap>
HierarchicalMap::SpawnLocalSeparators() const
  {
  HYMLS_LPROF3(label_, "SpawnLocalSeparators");

  Teuchos::RCP<const HierarchicalMap> newObject = Teuchos::null;
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<SeparatorGroup> > > new_separator_groups =
    Teuchos::rcp(new Teuchos::Array<Teuchos::Array<SeparatorGroup> >(NumMySubdomains()));
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<Teuchos::Array<SeparatorGroup> > > > new_linked_separator_groups =
    Teuchos::rcp(new Teuchos::Array<Teuchos::Array<Teuchos::Array<SeparatorGroup> > >(NumMySubdomains()));

  // Start out from the standard Separator object. All local separators are located
  // in its baseMap_
  Teuchos::RCP<const HierarchicalMap> sepObject = Spawn(Separators);

  for (int sd = 0; sd < sepObject->NumMySubdomains(); sd++)
    {
    for (SeparatorGroup const &group: sepObject->GetSeparatorGroups(sd))
      {
      hymls_gidx first_node = group[0];
      if (sepObject->GetMap()->MyGID(first_node))
        (*new_separator_groups)[sd].append(group);
      }
    }

  LinkSeparators(new_separator_groups, new_linked_separator_groups);

  newObject = Teuchos::rcp(new HierarchicalMap(sepObject->GetMap(), sepObject->GetMap(),
      Teuchos::null, new_separator_groups, new_linked_separator_groups, "Local Separator Nodes", myLevel_));

  return newObject;
  }

Teuchos::RCP<const Epetra_Map> HierarchicalMap::SpawnMap(int sd, SpawnStrategy strat) const
  {
  HYMLS_LPROF3(label_,"SpawnMap");

  if (!Filled())
    Tools::Error("object not filled",__FILE__,__LINE__);

  if (sd < 0 || sd > NumMySubdomains())
    {
    Tools::Error("subdomain index out of range",__FILE__,__LINE__);
    }

  int idx = (int)strat;
  if (idx >= spawnedMaps_.size())
    {
    Tools::Error("strategy index out of range",__FILE__,__LINE__);
    }

  Teuchos::RCP<const Epetra_Map> map = spawnedMaps_[idx][sd];

  if (map == Teuchos::null)
    {
    HYMLS_DEBUG("Spawn map for subdomain " << sd);

    Epetra_SerialComm comm;
    if (strat == Interior)
      {
      HYMLS_DEBUG("interior map");

      InteriorGroup const &group = GetInteriorGroup(sd);
      map = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1), group.length(), &group[0],
          (hymls_gidx)baseMap_->IndexBase64(), comm));
      }
    else if (strat == Separators)
      {
      HYMLS_DEBUG("separator map");

      int length = NumSeparatorElements(sd);
      hymls_gidx *gids = new hymls_gidx[length];

      int pos = 0;
      for (SeparatorGroup const &group: GetSeparatorGroups(sd))
        {
        std::copy(group.nodes().begin(), group.nodes().end(), gids + pos);
        pos += group.length();
        }

      map = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1), length, gids,
          (hymls_gidx)baseMap_->IndexBase64(), comm));

      delete[] gids;
      }
    else
      {
      Tools::Error("Bad strategy!", __FILE__, __LINE__);
      }
    spawnedMaps_[idx][sd] = map;
    }

  return map;
  }

// this doesn't formally belong to this class but has to be implemented somewhere
std::ostream & operator << (std::ostream& os, const HierarchicalMap& h)
  {
  return h.Print(os);
  }

#ifdef HYMLS_LONG_LONG
//! given a subdomain, returns a list of GIDs that belong to the subdomain
int HierarchicalMap::GetSeparatorGIDs(int sd, Epetra_LongLongSerialDenseVector &gids) const
  {
  HYMLS_LPROF3(label_, "getSubdomainGIDs");
  if (sd < 0 || sd > NumMySubdomains())
    {
    Tools::Warning("Subdomain index out of range!", __FILE__, __LINE__);
    return -1;
    }

  Teuchos::RCP<const Epetra_Map> map = SpawnMap(sd, Separators);

// resize input arrays if necessary
  if (gids.Length() != map->NumMyElements())
    {
    CHECK_ZERO(gids.Size(map->NumMyElements()));
    }

  return map->MyGlobalElements(gids.Values());
  }
#else
//! given a subdomain, returns a list of GIDs that belong to the subdomain
int HierarchicalMap::GetSeparatorGIDs(int sd, Epetra_IntSerialDenseVector &gids) const
  {
  HYMLS_LPROF3(label_, "getSubdomainGIDs");
  if (sd < 0 || sd > NumMySubdomains())
    {
    Tools::Warning("Subdomain index out of range!", __FILE__, __LINE__);
    return -1;
    }

  Teuchos::RCP<const Epetra_Map> map = SpawnMap(sd, Separators);

  // resize input arrays if necessary
  if (gids.Length() != map->NumMyElements())
    {
    CHECK_ZERO(gids.Size(map->NumMyElements()));
    }

  return map->MyGlobalElements(gids.Values());
  }
#endif
  }
