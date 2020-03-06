#include "HYMLS_OverlappingPartitioner.hpp"

#include "HYMLS_config.h"

#include "HYMLS_Tools.hpp"
#include "HYMLS_BasePartitioner.hpp"
#include "HYMLS_Macros.hpp"

#include "HYMLS_SeparatorGroup.hpp"
#include "HYMLS_CartesianPartitioner.hpp"
#include "HYMLS_SkewCartesianPartitioner.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_toString.hpp"

#include <algorithm>

class Epetra_Map;

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
  }

Teuchos::RCP<const BasePartitioner> OverlappingPartitioner::Partition()
  {
  HYMLS_PROF2(Label(), "Partition");
  Teuchos::RCP<BasePartitioner> partitioner = Teuchos::null;
  if (partitioningMethod_ == "Cartesian")
    {
    partitioner = Teuchos::rcp(new
      CartesianPartitioner(GetMap(), getMyNonconstParamList(), Comm(), myLevel_));
    }
  else if (partitioningMethod_ == "Skew Cartesian")
    {
    partitioner = Teuchos::rcp(new
      SkewCartesianPartitioner(GetMap(), getMyNonconstParamList(), Comm(), myLevel_));
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
  Teuchos::Array<SeparatorGroup> separator_groups;

  for (int sd = 0; sd < partitioner->NumLocalParts(); sd++)
    {
    interior_nodes.resize(0);
    separator_groups.resize(0);

    CHECK_ZERO(partitioner->GetGroups(sd, interior_nodes, separator_groups));

    std::sort(interior_nodes.begin(), interior_nodes.end());
    AddGroup(sd, interior_nodes);

    for (auto &group: separator_groups)
      {
      std::sort(group.nodes().begin(), group.nodes().end());
      AddGroup(sd, group.nodes());
      }
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

