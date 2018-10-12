#include "HYMLS_OverlappingPartitioner.H"
#include "HYMLS_Tools.H"

#include "HYMLS_CartesianPartitioner.H"
#include "HYMLS_SkewCartesianPartitioner.H"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StrUtils.hpp"
#include "Teuchos_Array.hpp"

#include "GaleriExt_Periodic.h"

#include "Teuchos_StandardParameterEntryValidators.hpp"

namespace HYMLS {

//constructor

// we call the base class constructor with a lot of null-pointers and create the
// data structures ourselves in the constructor. This means that the base class
// is not fully initialized during the constructor, but afterwards it is.
// This is OK because the base class constructor is mostly intended for spawning
// a new level from an existing one.
OverlappingPartitioner::OverlappingPartitioner(
  Teuchos::RCP<const Epetra_Map> map,
  Teuchos::RCP<Teuchos::ParameterList> params, int level,
  Teuchos::RCP<const Epetra_Map> overlappingMap)
  :
  HierarchicalMap(map, overlappingMap, 0, "OverlappingPartitioner", level),
  PLA("Problem")
  {
  HYMLS_PROF2(Label(),"Constructor");

  setParameterList(params);

  Teuchos::RCP<const BasePartitioner> partitioner = Partition();

  // Set the parameters for the next level
  nextLevelParams_ = Teuchos::rcp(new Teuchos::ParameterList(*getMyParamList()));
  partitioner->SetNextLevelParameters(*nextLevelParams_);

  HYMLS_DEBVAR(*nextLevelParams_);

  // we replace the map passed in by the user by the one generated
  // by the partitioner. This has two purposes:
  // - the partitioner may decide to repartition the domain, for
  //   instance if there are more processor partitions than sub-
  //   domains
  // - the data layout becomes more favorable because the nodes
  //   of a subdomain are contiguous in the partitioner's map.
  baseMap_ = partitioner->GetMap();

  if (baseOverlappingMap_ != Teuchos::null)
      baseOverlappingMap_ = partitioner->MoveMap(baseOverlappingMap_);

#ifdef HYMLS_DEBUGGING__disabled_
  HYMLS_DEBUG("Partition numbers:");
  for (int i=0;i<Map().NumMyElements();i++)
  {
      hymls_gidx gid = Map().GID64(i);
      HYMLS_DEBUG(gid << " " << (*partitioner)(gid));
  }
#endif

  // add the subdomains to the base class so we can start inserting groups of nodes
  Reset(partitioner->NumLocalParts());

  CHECK_ZERO(DetectSeparators(partitioner));
  HYMLS_DEBVAR(*this);
  return;
  }

OverlappingPartitioner::~OverlappingPartitioner()
  {
  HYMLS_PROF3(Label(),"Destructor");
  }

void OverlappingPartitioner::setParameterList(
  const Teuchos::RCP<Teuchos::ParameterList>& params)
  {
  HYMLS_PROF3(Label(),"setParameterList");
  
  setMyParamList(params);

  partitioningMethod_ = PL("Preconditioner").get(
      "Partitioner", "Cartesian");
  retainNodes_ = PL("Preconditioner").get(
      "Retain Nodes", retainNodes_);
  retainNodes_ = PL("Preconditioner").get(
      "Retain Nodes at Level " + Teuchos::toString(myLevel_),
                          retainNodes_);
  }

Teuchos::RCP<const BasePartitioner> OverlappingPartitioner::Partition()
  {
  HYMLS_PROF2(Label(), "Partition");
  Teuchos::RCP<BasePartitioner> partitioner = Teuchos::null;
  if (partitioningMethod_ == "Cartesian")
    {
    partitioner = Teuchos::rcp(new
      CartesianPartitioner(GetMap(), getMyNonconstParamList(), Comm()));
    }
  else if (partitioningMethod_ == "Skew Cartesian")
    {
    partitioner = Teuchos::rcp(new
      SkewCartesianPartitioner(GetMap(), getMyNonconstParamList(), Comm()));
    }
  else
    {
    Tools::Error("Up to now we only support Cartesian partitioning",
      __FILE__, __LINE__);
    }

  CHECK_ZERO(partitioner->Partition(false));

  return partitioner;
  }

int OverlappingPartitioner::DetectSeparators(Teuchos::RCP<const BasePartitioner> partitioner)
  {
  HYMLS_PROF2(Label(),"DetectSeparators");

  // nodes to be eliminated exactly in the next step
  Teuchos::Array<hymls_gidx> interior_nodes;
  // separator nodes
  Teuchos::Array<Teuchos::Array<hymls_gidx> > separator_nodes;
  // links between separator nodes
  Teuchos::Array<Teuchos::Array<int> > group_links;

  for (int sd = 0; sd < partitioner->NumLocalParts(); sd++)
    {
    interior_nodes.resize(0);
    separator_nodes.resize(0);

    CHECK_ZERO(partitioner->GetGroups(sd, interior_nodes, separator_nodes, group_links));

    std::sort(interior_nodes.begin(), interior_nodes.end());
    AddGroup(sd, interior_nodes);

    Teuchos::Array<Teuchos::Array<int> > newGroupLinks;
    int separatorIdx = 1;
    for (auto const &groupLink: group_links)
      {
      newGroupLinks.push_back(Teuchos::Array<int>());
      for (int gid: groupLink)
        {
        if (separator_nodes[gid-1].size() > 0)
          {
          std::sort(separator_nodes[gid-1].begin(), separator_nodes[gid-1].end());
          AddGroup(sd, separator_nodes[gid-1]);

          newGroupLinks.back().push_back(separatorIdx);
          separatorIdx++;
          }
        }
      }

    // Remove empty groups
    newGroupLinks.erase(std::remove_if(newGroupLinks.begin(), newGroupLinks.end(),
        [](Teuchos::Array<int> const &i){return i.empty();}), newGroupLinks.end());

    if (newGroupLinks.size() > 0)
      AddGroupLinks(sd, newGroupLinks);
    }

  // and rebuild map and global groupPointer
  CHECK_ZERO(FillComplete());
  return 0;
  }

Teuchos::RCP<const OverlappingPartitioner> OverlappingPartitioner::SpawnNextLevel(
  Teuchos::RCP<const Epetra_Map> map,
  Teuchos::RCP<const Epetra_Map> overlappingMap) const
  {
  HYMLS_PROF2(Label(), "SpawnNextLevel");

  Teuchos::RCP<const OverlappingPartitioner> newLevel;
  newLevel = Teuchos::rcp(new OverlappingPartitioner(
      map, nextLevelParams_, Level()+1, overlappingMap));
  return newLevel;
  }

}//namespace

