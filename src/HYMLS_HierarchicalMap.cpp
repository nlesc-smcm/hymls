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
  Teuchos::RCP<Teuchos::Array< Teuchos::Array<hymls_gidx> > > groupPointer,
  Teuchos::RCP<Teuchos::Array< Teuchos::Array<hymls_gidx> > > gidList,
  Teuchos::RCP<Teuchos::Array< Teuchos::Array<SeparatorGroup> > > separator_groups,
  std::string label, int level)
  :
  label_(label),
  myLevel_(level),
  baseMap_(baseMap),
  baseOverlappingMap_(overlappingMap),
  overlappingMap_(overlappingMap),
  groupPointer_(groupPointer),
  gidList_(gidList),
  separator_groups_(separator_groups)
  {
  HYMLS_LPROF2(label_,"HierarchicalMap Constructor");
  spawnedObjects_.resize(4); // can currently spawn Interior, Separator and 
  // LocalSeparator objects and 
  // return a self-reference (All)    
  spawnedMaps_.resize(4);
  for (int i=0;i<spawnedObjects_.size();i++) spawnedObjects_[i]=Teuchos::null;
  for (int i=0;i<spawnedMaps_.size();i++)  
    {
    spawnedMaps_[i].resize(NumMySubdomains());
    for (int sd=0;sd<NumMySubdomains();sd++)
      {
      spawnedMaps_[i][sd]=Teuchos::null;
      }
    }
  }

HierarchicalMap::~HierarchicalMap()
  {
  HYMLS_LPROF3(label_,"Destructor");
  }

int HierarchicalMap::Reset(int numMySubdomains)
  {
  HYMLS_LPROF2(label_,"Reset");
  groupPointer_=Teuchos::rcp(new Teuchos::Array<Teuchos::Array<hymls_gidx> >(numMySubdomains));
  gidList_=Teuchos::rcp(new Teuchos::Array<Teuchos::Array<hymls_gidx> >(numMySubdomains));
  interior_groups_ = Teuchos::rcp(new Teuchos::Array<InteriorGroup>(numMySubdomains));
  separator_groups_ = Teuchos::rcp(new Teuchos::Array<Teuchos::Array<SeparatorGroup> >(numMySubdomains));

  for (int i=0;i<numMySubdomains;i++)
    {
    (*gidList_)[i].resize(0);
    (*groupPointer_)[i].resize(1);
    (*groupPointer_)[i][0]=0;
    }

  spawnedObjects_.resize(4); // can currently spawn Interior, Separator and 
  // LocalSeparator objects and 
  // return a self-reference (All)    
  spawnedMaps_.resize(4);
  for (int i=0;i<spawnedObjects_.size();i++) spawnedObjects_[i]=Teuchos::null;
  overlappingMap_=Teuchos::null;
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

  Teuchos::RCP<Teuchos::Array<Teuchos::Array<hymls_gidx> > > newGroupPointer =
    Teuchos::rcp(new Teuchos::Array<Teuchos::Array<hymls_gidx> >());
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<hymls_gidx> > > newGidList =
    Teuchos::rcp(new Teuchos::Array<Teuchos::Array<hymls_gidx> >());

  Teuchos::RCP<const Epetra_Map> map = baseMap_;
  if (baseOverlappingMap_ != Teuchos::null)
    map = baseOverlappingMap_;

  // Merge all separator GIDs on this processor into one list.
  // Put all interior GIDs that are present in the baseMap_ in the
  // newGidList.
  Teuchos::Array<hymls_gidx> separatorGIDs;
  for (int sd = 0; sd < NumMySubdomains(); sd++)
    {
    newGidList->append(Teuchos::Array<hymls_gidx>());
    newGroupPointer->append(Teuchos::Array<hymls_gidx>(1));

    // Interior nodes don't need communication. Just add those
    // that are present in the baseMap_
    for (int j = 0; j < NumElements(sd, 0); j++)
      {
      hymls_gidx gid = GID(sd, 0, j);
      if (map->MyGID(gid))
        (*newGidList)[sd].append(gid);
      }
    hymls_gidx offset = *((*newGroupPointer)[sd].end() - 1);
    int len = (*newGidList)[sd].size() - offset;
    (*newGroupPointer)[sd].append(len + offset);

    // Now add the separator groups. We can avoid communication
    // if the baseOverlappingMap_ is present.
    for (SeparatorGroup &group: (*separator_groups_)[sd])
      {
      if (baseOverlappingMap_ != Teuchos::null)
        {
        SeparatorGroup new_group;
        for (hymls_gidx gid: group.nodes())
          if (map->MyGID(gid))
            {
            (*newGidList)[sd].append(gid);
            new_group.append(gid);
            }

        hymls_gidx offset = *((*newGroupPointer)[sd].end()-1);
        int len = (*newGidList)[sd].size() - offset;
        if (len > 0)
          (*newGroupPointer)[sd].append(len + offset);
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
        Teuchos::Array<hymls_gidx> gidList;
        for (hymls_gidx gid: group.nodes())
          {
          // If it is present in the overlappingVec the element actually belongs
          // to the baseMap_ on some processor
          if (overlappingVec[tmpOverlappingMap->LID(gid)])
            {
            gidList.append(gid);
            new_group.append(gid);
            }
          }
        hymls_gidx offset = *((*newGroupPointer)[sd].end()-1);
        int len = gidList.size();
        if (len > 0)
          {
          (*newGroupPointer)[sd].append(offset + len);
          std::copy(gidList.begin(), gidList.end(), std::back_inserter((*newGidList)[sd]));
          }
        group.nodes() = new_group.nodes();
        }
      }
    }

  // Make a new overlapping map with elements that are present on some processor
  Teuchos::Array<hymls_gidx> allGIDs;
  for (int sd = 0; sd < NumMySubdomains(); sd++)
    {
    std::copy((*newGidList)[sd].begin(), (*newGidList)[sd].end(),
      std::back_inserter(allGIDs));

    // Remove empty separator groups
    (*separator_groups_)[sd].erase(std::remove_if(
        (*separator_groups_)[sd].begin(), (*separator_groups_)[sd].end(),
        [](SeparatorGroup const &i){return i.nodes().empty();}),
      (*separator_groups_)[sd].end());
    }
  std::sort(allGIDs.begin(), allGIDs.end());
  auto last = std::unique(allGIDs.begin(), allGIDs.end());
  overlappingMap_ = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1), std::distance(allGIDs.begin(), last),
      allGIDs.getRawPtr(), (hymls_gidx)baseMap_->IndexBase64(), Comm()));

  gidList_ = newGidList;
  groupPointer_ = newGroupPointer;

  return 0;
  }

int HierarchicalMap::AddInteriorGroup(int sd, InteriorGroup const &group)
  {
  HYMLS_LPROF3(label_,"AddInteriorGroup");

  if (sd >= groupPointer_->size())
    {
    Tools::Warning("invalid subdomain index",__FILE__,__LINE__);
    return -1; // You should Reset with the right amount of sd
    } 

  HYMLS_DEBVAR(sd);
  HYMLS_DEBVAR(group.nodes());
  hymls_gidx offset=*((*groupPointer_)[sd].end()-1);
  int len = group.length();
  (*groupPointer_)[sd].append(offset+len);
  if (len>0)
    {
    std::copy(group.nodes().begin(),group.nodes().end(),std::back_inserter((*gidList_)[sd]));
    }
  (*interior_groups_)[sd] = group;
  return (*groupPointer_)[sd].length()-1;
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

Teuchos::Array<SeparatorGroup> const &HierarchicalMap::SeparatorGroups(int sd) const
  {
  return (*separator_groups_)[sd];
  }

Teuchos::Array<hymls_gidx> HierarchicalMap::GetGroup(int sd, int grp) const
  {
  HYMLS_LPROF3(label_,"GetGroup");

  if (sd >= groupPointer_->size())
    Tools::Error("Invalid subdomain index", __FILE__, __LINE__);

  if (grp >= (*groupPointer_)[sd].size())
    Tools::Error("Invalid group index", __FILE__, __LINE__);

  hymls_gidx offset = *((*groupPointer_)[sd].begin() + grp);
  int len = *((*groupPointer_)[sd].begin() + grp + 1) - offset;

  if (offset + len > (*gidList_)[sd].size())
    Tools::Error("Invalid group index", __FILE__, __LINE__);

  Teuchos::Array<hymls_gidx> gidList;
  std::copy((*gidList_)[sd].begin() + offset, (*gidList_)[sd].begin() + offset + len,std::back_inserter(gidList));

  return gidList;
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

    for (int proc=0;proc<Comm().NumProc();proc++)
      {
      if (proc==rank)
        {
        os << "%Partition " << rank << std::endl;
        os << "%=============" << std::endl;
        //os << "% Number of rows in groupPointer: " << (*groupPointer_).size() << std::endl;
        for (int sd=0;sd<NumMySubdomains();sd++)
          {
          os << "p{" << myLevel_ << "}{" << rank+1 << "}.grpPtr{" << sd+1 << "}=[";
          for (int i=0;i<(*groupPointer_)[sd].size()-1;i++)
            {
            os << (*groupPointer_)[sd][i] << ",";
            }
          os << *((*groupPointer_)[sd].end()-1) << "];" << std::endl;
          os << "   p{" << myLevel_ << "}{" << rank+1 << "}.sd{" << sd+1 << "} = [";
          for (int grp=0;grp<(*groupPointer_)[sd].size()-1;grp++)
            {
            for (int i=(*groupPointer_)[sd][grp]; i<(*groupPointer_)[sd][grp+1];i++)
              {
              os << " " << (*gidList_)[sd][i];
              }
            os << " ..." << std::endl;
            }
          os << "];" << std::endl << std::endl;
          }
        }//rank
      Comm().Barrier();
      }//proc

    if (rank==0)
      {
      os << "%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$%\n" << std::endl;
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
          for (int grp = 0; grp < NumGroups(sd); grp++)
            {
            Teuchos::Array<hymls_gidx> gidList = GetGroup(sd, grp);
            if (grp > 0)
              {
              os << ",..." << std::endl;
              }
            os << "[";
            for (int i = 0; i < NumElements(sd, grp); i++)
              {
              os << gidList[i] << ",";
              }
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
    {
    Tools::Error("object not filled",__FILE__,__LINE__);
    }
  int idx=(int)strat;
  
  if (spawnedObjects_.size()<idx+1)
    {
    Tools::Error("Bad strategy!",__FILE__,__LINE__);
    }

  Teuchos::RCP<const HierarchicalMap>
        object = spawnedObjects_[idx];
    
  if (object==Teuchos::null)
    {
    HYMLS_LPROF3(label_,"Spawn");
    if (strat==Interior)
      {
      object=SpawnInterior();
      }
    else if (strat==Separators)
      {
      object=SpawnSeparators();
      }
    else if (strat==LocalSeparators)
      {
      object=SpawnLocalSeparators();
      }
    else if (strat==All)
      {
      object=Teuchos::rcp(this,false);
      }
    else
      {
      Tools::Error("Bad strategy!",__FILE__,__LINE__);
      }
    spawnedObjects_[idx]=object;
    }
  return object;
  }


Teuchos::RCP<const HierarchicalMap>
HierarchicalMap::SpawnInterior() const
  {
  HYMLS_LPROF3(label_,"SpawnInterior");

  Teuchos::RCP<const HierarchicalMap> newObject=Teuchos::null;
  Teuchos::RCP<Epetra_Map> newMap=Teuchos::null;
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<hymls_gidx> > > newGroupPointer =
    Teuchos::rcp(new Teuchos::Array<Teuchos::Array<hymls_gidx> >());
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<hymls_gidx> > > newGidList =
    Teuchos::rcp(new Teuchos::Array<Teuchos::Array<hymls_gidx> >());
  
  hymls_gidx base = baseMap_->IndexBase64();  
  
  int num = NumMyInteriorElements();
  hymls_gidx *myElements = new hymls_gidx[num];
  int pos = 0;
  for (int sd = 0; sd < NumMySubdomains(); sd++)
    {
    int len = (*groupPointer_)[sd][1];
    std::copy((*gidList_)[sd].begin(), (*gidList_)[sd].begin()+len, myElements+pos);
    pos += len;
    }

  newMap = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1), pos, myElements, base, Comm()));
  delete [] myElements;

  newGroupPointer->resize(NumMySubdomains());
  if (NumMySubdomains()>0)
    {
    (*newGroupPointer)[0].resize(2);
    (*newGroupPointer)[0][0]=0;
    (*newGroupPointer)[0][1]=NumInteriorElements(0);
    }
  for (int sd=1;sd<NumMySubdomains();sd++)
    {
    (*newGroupPointer)[sd].resize(2);
    (*newGroupPointer)[sd][0]=(*newGroupPointer)[sd-1][1];
    (*newGroupPointer)[sd][1]=(*newGroupPointer)[sd][0]+NumInteriorElements(sd);
    }

  newObject = Teuchos::rcp(new HierarchicalMap(newMap, newMap,
      newGroupPointer, gidList_, Teuchos::null, "Interior Nodes",myLevel_) );

  return newObject;
  }

/////////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<const HierarchicalMap>
HierarchicalMap::SpawnSeparators() const
  {
  HYMLS_LPROF3(label_,"SpawnSeparators");

  if (!Filled()) Tools::Error("object not filled", __FILE__, __LINE__);

  Teuchos::RCP<const HierarchicalMap> newObject = Teuchos::null;
  Teuchos::RCP<Epetra_Map> newMap = Teuchos::null;
  Teuchos::RCP<Epetra_Map> newOverlappingMap = Teuchos::null;
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<hymls_gidx> > > newGroupPointer =
    Teuchos::rcp(new Teuchos::Array<Teuchos::Array<hymls_gidx> >());
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<hymls_gidx> > > newGidList =
    Teuchos::rcp(new Teuchos::Array<Teuchos::Array<hymls_gidx> >());
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<SeparatorGroup> > > new_separator_groups =
    Teuchos::rcp(new Teuchos::Array<Teuchos::Array<SeparatorGroup> >(NumMySubdomains()));

  Teuchos::Array<hymls_gidx> done;
  Teuchos::Array<hymls_gidx> localGIDs;
  Teuchos::Array<hymls_gidx> overlappingGIDs;

  for (int sd = 0; sd < NumMySubdomains(); sd++)
    {
    newGidList->append(Teuchos::Array<hymls_gidx>());
    newGroupPointer->append(Teuchos::Array<hymls_gidx>(2));
    for (SeparatorGroup const &group: (*separator_groups_)[sd])
      {
      hymls_gidx first_node = group.nodes()[0];
      if (std::find(done.begin(), done.end(), first_node) == done.end())
        {
        for (hymls_gidx gid: group.nodes())
          {
          overlappingGIDs.append(gid);
          if (baseMap_->MyGID(gid))
            {
            localGIDs.append(gid);
            }
          }
        hymls_gidx offset = *((*newGroupPointer)[sd].end()-1);
        int len = group.length();
        if (len > 0)
          {
          (*newGroupPointer)[sd].append(offset + len);
          std::copy(group.nodes().begin(), group.nodes().end(),
            std::back_inserter((*newGidList)[sd]));
          }
        else
          Tools::Error("This should not happen", __FILE__, __LINE__);
        (*new_separator_groups)[sd].append(group);
        done.append(first_node);
        }
      }
    }

  newOverlappingMap = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1), overlappingGIDs.size(),
      overlappingGIDs.getRawPtr(), (hymls_gidx)baseMap_->IndexBase64(), Comm()));

  newMap = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1), localGIDs.size(),
      localGIDs.getRawPtr(), (hymls_gidx)baseMap_->IndexBase64(), Comm()));

  newObject = Teuchos::rcp(new HierarchicalMap(newMap, newOverlappingMap,
      newGroupPointer, newGidList, new_separator_groups, "Separator Nodes", myLevel_));

  return newObject;

  }

////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<const HierarchicalMap>
HierarchicalMap::SpawnLocalSeparators() const
  { 
  HYMLS_LPROF3(label_,"SpawnLocalSeparators");

  Teuchos::RCP<const HierarchicalMap> newObject=Teuchos::null;
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<hymls_gidx> > > newGidList =
    Teuchos::rcp(new Teuchos::Array<Teuchos::Array<hymls_gidx> >());
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<hymls_gidx> > > newGroupPointer =
    Teuchos::rcp(new Teuchos::Array<Teuchos::Array<hymls_gidx> >());
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<SeparatorGroup> > > new_separator_groups =
    Teuchos::rcp(new Teuchos::Array<Teuchos::Array<SeparatorGroup> >(NumMySubdomains()));

  // Start out from the standard Separator object. All local separators are located
  // in its baseMap_
  Teuchos::RCP<const HierarchicalMap> sepObject = Spawn(Separators);

  for (int sd = 0; sd < sepObject->NumMySubdomains(); sd++)
    {
    newGidList->append(Teuchos::Array<hymls_gidx>());
    newGroupPointer->append(Teuchos::Array<hymls_gidx>(2));
    for (SeparatorGroup const &group: sepObject->SeparatorGroups(sd))
      {
      hymls_gidx first_node = group.nodes()[0];
      if (sepObject->GetMap()->MyGID(first_node))
        {
        hymls_gidx offset = *((*newGroupPointer)[sd].end()-1);
        int len = group.length();
        if (len > 0)
          {
          (*newGroupPointer)[sd].append(offset + len);
          std::copy(group.nodes().begin(), group.nodes().end(),
            std::back_inserter((*newGidList)[sd]));
          }
        else
          Tools::Error("This should not happen", __FILE__, __LINE__);
        (*new_separator_groups)[sd].append(group);
        }
      }
    }

  newObject = Teuchos::rcp(new HierarchicalMap(sepObject->GetMap(), sepObject->GetMap(),
      newGroupPointer, newGidList, new_separator_groups, "Local Separator Nodes", myLevel_));

  return newObject;
  }

  Teuchos::RCP<const Epetra_Map> HierarchicalMap::SpawnMap
        (int sd, SpawnStrategy strat) const
    {
    HYMLS_LPROF3(label_,"SpawnMap");
    if ((sd<0)||(sd>NumMySubdomains()))
      {
      Tools::Error("subdomain index out of range",__FILE__,__LINE__);
      }
    int idx=(int)strat;
    if (idx>=spawnedMaps_.size())
      {
      Tools::Error("strategy index out of range",__FILE__,__LINE__);
      }
      
    if (spawnedMaps_[idx].size()<NumMySubdomains())
      {
      spawnedMaps_[idx].resize(NumMySubdomains());
      for (int i=0;i<NumMySubdomains();i++) spawnedMaps_[idx][sd]=Teuchos::null;
      }
      
    Teuchos::RCP<const Epetra_Map> map = spawnedMaps_[idx][sd];
    
    if (map==Teuchos::null)
      {
      HYMLS_DEBUG("Spawn map for subdomain " << sd);
      // int* MyElements = overlappingMap_->MyGlobalElements();      
      hymls_gidx offset = -1;
      int length = -1;
      if (strat==Interior)
        {
        HYMLS_DEBUG("interior map");
        offset = (*groupPointer_)[sd][0];
        length = NumInteriorElements(sd);
        }
      else if (strat==Separators)
        {
        HYMLS_DEBUG("separator map");
        offset = (*groupPointer_)[sd][1];
        length = NumSeparatorElements(sd);
        }
      else if (strat==All)
        {
        HYMLS_DEBUG("complete map");
        offset = (*groupPointer_)[sd][0];
        length = NumElements(sd);
        }
      HYMLS_DEBVAR(offset);
      HYMLS_DEBVAR(length);
      Epetra_SerialComm comm;
      map = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1), length, &((*gidList_)[sd][0+offset]),
          (hymls_gidx)baseMap_->IndexBase64(), comm));
                
      spawnedMaps_[idx][sd]=map;
      }
    
    return map;
    }

// this doesn't formally belong to this class but has to be implemented somewhere
std::ostream & operator << (std::ostream& os, const HierarchicalMap& h)
  {
  return h.Print(os);
  }

//! given a subdomain, returns a list of GIDs that belong to the subdomain
int HierarchicalMap::getSeparatorGIDs(int sd, hymls_gidx *gids) const
  {
  HYMLS_LPROF3(label_, "getSubdomainGIDs");
  if (sd < 0 || sd > NumMySubdomains())
    {
    Tools::Warning("Subdomain index out of range!", __FILE__, __LINE__);
    return -1;
    }

  int pos = 0;
  for (SeparatorGroup const &group: SeparatorGroups(sd))
    for (hymls_gidx gid: group.nodes())
      gids[pos++] = gid;

  return 0;
  }

#ifdef HYMLS_LONG_LONG
//! given a subdomain, returns a list of GIDs that belong to the subdomain
int HierarchicalMap::getSeparatorGIDs(int sd, Epetra_LongLongSerialDenseVector &gids) const
  {
  HYMLS_LPROF3(label_, "getSubdomainGIDs");
  if (sd < 0 || sd > NumMySubdomains())
    {
    Tools::Warning("Subdomain index out of range!", __FILE__, __LINE__);
    return -1;
    }

  int nrows = NumSeparatorElements(sd);

  // resize input arrays if necessary
  if (gids.Length() != nrows)
    {
    CHECK_ZERO(gids.Size(nrows));
    }

  return getSeparatorGIDs(sd, gids.Values());
  }
#else
//! given a subdomain, returns a list of GIDs that belong to the subdomain
int HierarchicalMap::getSeparatorGIDs(int sd, Epetra_IntSerialDenseVector &gids) const
  {
  HYMLS_LPROF3(label_, "getSubdomainGIDs");
  if (sd < 0 || sd > NumMySubdomains())
    {
    Tools::Warning("Subdomain index out of range!", __FILE__, __LINE__);
    return -1;
    }

  int nrows = NumSeparatorElements(sd);

  // resize input arrays if necessary
  if (gids.Length() != nrows)
    {
    CHECK_ZERO(gids.Size(nrows));
    }

  return getSeparatorGIDs(sd, gids.Values());
  }
#endif
}
