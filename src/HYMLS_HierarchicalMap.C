// if you activate this option, the resulting Schur-complement
// and preconditioner look like those in MATLAB (with singletons
// and pressures shifted to the end of the ordering). This only 
// works for 2D Navier-Stokes, otherwise an error is thrown.
#ifdef MATLAB_COMPATIBILITY_MODE
#define SHIFT_PRESSURE_TO_END
#define SHIFT_SINGLETONS_TO_END
#endif

#include <iostream>

//#include "HYMLS_no_debug.H"

#include "HYMLS_HierarchicalMap.H"
#include "HYMLS_Tools.H"
#include "HYMLS_BasePartitioner.H"
#include "HYMLS_CartesianPartitioner.H"
#include "HYMLS_SepNode.H"

#include "Epetra_Comm.h"
#include "Epetra_SerialComm.h"
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Vector.h"
#include "Epetra_IntVector.h"
#include "Epetra_Import.h"
#include "Epetra_IntSerialDenseVector.h"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StrUtils.hpp"
#include "Epetra_Util.h"

#include <algorithm>


#include "HYMLS_MatrixUtils.H"

namespace HYMLS {

  //empty constructor
  HierarchicalMap::HierarchicalMap(Teuchos::RCP<const Epetra_Comm> comm, 
        Teuchos::RCP<const Epetra_Map> baseMap, int numMySubdomains,
        std::string label, int level)
        : comm_(comm),
          label_(label),
          baseMap_(baseMap),
          overlappingMap_(Teuchos::null),
          myLevel_(level)
    {
    HYMLS_PROF3(label_,"Constructor");
    this->Reset(numMySubdomains);
    }

  //private constructor
  HierarchicalMap::HierarchicalMap(
        Teuchos::RCP<const Epetra_Comm> comm, 
        Teuchos::RCP<const Epetra_Map> baseMap,
        Teuchos::RCP<const Epetra_Map> overlappingMap,
        Teuchos::RCP<Teuchos::Array< Teuchos::Array<int> > > groupPointer,
        Teuchos::RCP<Teuchos::Array< Teuchos::Array<int> > > gidList,
        std::string label, int level)
  : comm_(comm),
    label_(label),
    baseMap_(baseMap),
    overlappingMap_(overlappingMap),
    groupPointer_(groupPointer),
    gidList_(gidList),
    myLevel_(level)
    {
    HYMLS_PROF3(label_,"HierarchicalMap Constructor");
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
    HYMLS_PROF3(label_,"Destructor");
    }

  int HierarchicalMap::Reset(int numMySubdomains)
    {
    HYMLS_PROF2(label_,"Reset");
    groupPointer_=Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >(numMySubdomains));
    gidList_=Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >(numMySubdomains));
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
  HYMLS_PROF2(label_,"FillComplete");
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

  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGroupPointer =
    Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGidList =
    Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());

  // Merge all GIDs on this processor into one list
  Teuchos::Array<int> allGids;
  for (int sd = 0; sd < NumMySubdomains(); sd++)
    {
    std::copy((*gidList_)[sd].begin(), (*gidList_)[sd].end(),
      std::back_inserter(allGids));
    }

  // Make sure there is only one entry of each of them
  std::sort(allGids.begin(), allGids.end());
  Teuchos::Array<int>::iterator end = std::unique(allGids.begin(), allGids.end());

  // Communicate between processors which elements are actually present in the
  // baseMap_ of the processor that owns the nodes. This is because the
  // Partitioner gives us all possible elements belonging to the subdomain,
  // not only the elements that are in the map
  int numElements = std::distance(allGids.begin(), end);
  Teuchos::RCP<Epetra_Map> tmpOverlappingMap =
    Teuchos::rcp(new Epetra_Map(-1, numElements, &allGids[0],
        baseMap_->IndexBase(), *comm_));

  HYMLS_DEBVAR(*tmpOverlappingMap);

  Epetra_IntVector vec(*baseMap_);
  for (Teuchos::Array<int>::iterator i = allGids.begin(); i != end; i++)
    {
    int lid = baseMap_->LID(*i);
    if (lid != -1)
      vec[lid] = 1;
    }
  Epetra_Import imp(*tmpOverlappingMap, *baseMap_);
  Epetra_IntVector overlappingVec(*tmpOverlappingMap);
  overlappingVec.Import(vec, imp, Insert);

  for (int sd = 0; sd < NumMySubdomains(); sd++)
    {
    newGidList->append(Teuchos::Array<int>());
    newGroupPointer->append(Teuchos::Array<int>(1));
    for (int grp = 0; grp < NumGroups(sd); grp++)
      {
      Teuchos::Array<int> gidList;
      for (int j = 0; j < NumElements(sd,grp); j++)
        {
        int gid = GID(sd, grp, j);
        // If it is present in the overlappingVec the element actually belongs
        // to the baseMap_ on some processor
        if (overlappingVec[tmpOverlappingMap->LID(gid)])
          {
          gidList.append(gid);
          }
        }
      int offset = *((*newGroupPointer)[sd].end()-1);
      int len = gidList.size();
      if (len > 0)
        {
        (*newGroupPointer)[sd].append(offset + len);
        std::copy(gidList.begin(), gidList.end(), std::back_inserter((*newGidList)[sd]));
        }
      }
    }

  // Make a new overlapping map with elements that are present on some processor
  allGids.resize(0);
  for (int sd = 0; sd < NumMySubdomains(); sd++)
    {
    std::copy((*newGidList)[sd].begin(), (*newGidList)[sd].end(),
      std::back_inserter(allGids));
    }
  overlappingMap_ = Teuchos::rcp(new Epetra_Map(-1, allGids.size(),
      &allGids[0], baseMap_->IndexBase(), *comm_));

  gidList_ = newGidList;
  groupPointer_ = newGroupPointer;
  return 0;
  }

int HierarchicalMap::AddGroup(int sd, Teuchos::Array<int>& gidList)
  {
  HYMLS_PROF3(label_,"AddGroup");

  if (sd>=groupPointer_->size())
    {
    Tools::Warning("invalid subdomain index",__FILE__,__LINE__);
    return -1; // You should Reset with the right amount of sd
    }

  HYMLS_DEBVAR(sd);
  HYMLS_DEBVAR(gidList);
  int offset=*((*groupPointer_)[sd].end()-1);
  int len = gidList.size();
  (*groupPointer_)[sd].append(offset+len);
  if (len>0)
    {
    std::copy(gidList.begin(),gidList.end(),std::back_inserter((*gidList_)[sd]));
    }
  return (*groupPointer_)[sd].length()-1;
  }

Teuchos::Array<int> HierarchicalMap::GetGroup(int sd, int grp) const
  {
  HYMLS_PROF3(label_,"GetGroup");

  if (sd >= groupPointer_->size())
    Tools::Error("Invalid subdomain index", __FILE__, __LINE__);

  if (grp >= (*groupPointer_)[sd].size())
    Tools::Error("Invalid group index", __FILE__, __LINE__);

  int offset = *((*groupPointer_)[sd].begin() + grp);
  int len = *((*groupPointer_)[sd].begin() + grp + 1) - offset;

  if (offset + len > (*gidList_)[sd].size())
    Tools::Error("Invalid group index", __FILE__, __LINE__);

  Teuchos::Array<int> gidList;
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
    HYMLS_PROF3(label_,"Print");
    int rank=comm_->MyPID();
    
    if (rank==0)
      {
      os << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl;
      os << "% Domain decomposition and separators, level "<<myLevel_<< "       %"<<std::endl;
      os << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"<<std::endl;
      os << std::endl;
      }

    for (int proc=0;proc<comm_->NumProc();proc++)
      {
      if (proc==rank)
        {
        os << "%Partition "<<rank<<std::endl;
        os << "%============="<<std::endl;
        //os << "% Number of rows in groupPointer: " << (*groupPointer_).size() << std::endl;
        for (int sd=0;sd<NumMySubdomains();sd++)
          {
          os << "p{"<<myLevel_<<"}{"<<rank+1<<"}.grpPtr{"<<sd+1<<"}=["; 
          int offset = (*groupPointer_)[sd][0];
          for (int i=0;i<(*groupPointer_)[sd].size()-1;i++)
            {
            os<<(*groupPointer_)[sd][i]-offset<<",";
            }
          os<<*((*groupPointer_)[sd].end()-1)-offset<<"];"<<std::endl;
          os << "   p{"<<myLevel_<<"}{"<<rank+1<<"}.sd{"<<sd+1<<"} = [";
          for (int grp=0;grp<(*groupPointer_)[sd].size()-1;grp++)
            {
            for (int i=(*groupPointer_)[sd][grp]; i<(*groupPointer_)[sd][grp+1];i++)
              {
              os << " " << overlappingMap_->GID(i);
              }
            os << " ..." << std::endl;
            }
          os << "];"<<std::endl<<std::endl;
          }
        }//rank
      comm_->Barrier();
      }//proc
   
    comm_->Barrier();

    if (rank==0)
      {
      os << "%$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$%"<<std::endl;
      }
  return os;
  }


Teuchos::RCP<const Epetra_Map> HierarchicalMap::GetUniqueOverlappingMap() const
  {
  HYMLS_PROF3(label_,"GetUniqueOverlappingMap");

  // Get elements from the original overlapping map which contains duplicate elements
  Teuchos::Array<int> myElements(overlappingMap_->NumMyElements());
  overlappingMap_->MyGlobalElements(&myElements[0]);

  // Sort the list and only keep the unique elements
  std::sort(myElements.begin(), myElements.end());
  auto last = std::unique(myElements.begin(), myElements.end());
  myElements.erase(last, myElements.end());

  // Return a new overlapping map with only unique elements
  return Teuchos::rcp(new Epetra_Map(-1, myElements.size(), &myElements[0],
      overlappingMap_->IndexBase(), *comm_));
  }

Teuchos::RCP<const HierarchicalMap> 
HierarchicalMap::Spawn(SpawnStrategy strat,
        Teuchos::RCP<Teuchos::Array<HYMLS::SepNode> > regroup) const
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
    HYMLS_PROF3(label_,"Spawn");
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
      object=SpawnLocalSeparators(regroup);
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
  HYMLS_PROF3(label_,"SpawnInterior");

  Teuchos::RCP<const HierarchicalMap> newObject=Teuchos::null;
  Teuchos::RCP<Epetra_Map> newMap=Teuchos::null;
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGroupPointer =
        Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGidList =
        Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());
  
  int base = baseMap_->IndexBase();  
  
  int num = NumMyInteriorElements();
  int *myElements = new int[num];
  int pos = 0;
  for (int sd = 0; sd < NumMySubdomains(); sd++)
    {
    int len = (*groupPointer_)[sd][1];
    std::copy((*gidList_)[sd].begin(), (*gidList_)[sd].begin()+len, myElements+pos);
    pos += len;
    }

  newMap = Teuchos::rcp(new Epetra_Map(-1, pos, myElements, base, *comm_));
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

  newObject = Teuchos::rcp(new HierarchicalMap(comm_, newMap, newMap,
      newGroupPointer, gidList_, "Interior Nodes",myLevel_) );

  return newObject;
  }

/////////////////////////////////////////////////////////////////////////////////////

  Teuchos::RCP<const HierarchicalMap> 
  HierarchicalMap::SpawnSeparators() const
  {
  HYMLS_PROF3(label_,"SpawnSeparators");

  if (!Filled()) Tools::Error("object not filled", __FILE__, __LINE__);

  Teuchos::RCP<const HierarchicalMap> newObject = Teuchos::null;
  Teuchos::RCP<Epetra_Map> newMap = Teuchos::null;
  Teuchos::RCP<Epetra_Map> newOverlappingMap = Teuchos::null;
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGroupPointer =
        Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGidList =
        Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());

  Teuchos::Array<int> done;
  Teuchos::Array<int> localGIDs;
  Teuchos::Array<int> overlappingGIDs;

  for (int sd=0;sd<NumMySubdomains();sd++)
    {
    newGidList->append(Teuchos::Array<int>());
    newGroupPointer->append(Teuchos::Array<int>(2));
    for (int grp=1;grp<=NumSeparatorGroups(sd);grp++)
      {
      if (NumElements(sd,grp)>0 &&
        std::find(done.begin(), done.end(), GID(sd, grp, 0)) == done.end())
        {
        Teuchos::Array<int> gidList;
        for (int j=0;j<NumElements(sd,grp);j++)
          {
          int gid = GID(sd, grp, j);
          gidList.append(gid);
          overlappingGIDs.append(gid);
          if (baseMap_->MyGID(gid))
            {
            localGIDs.append(gid);
            }
          }
        int offset = *((*newGroupPointer)[sd].end()-1);
        int len = gidList.size();
        if (len>0)
          {
          (*newGroupPointer)[sd].append(offset + len);
          std::copy(gidList.begin(), gidList.end(), std::back_inserter((*newGidList)[sd]));
          }
        done.append(GID(sd, grp, 0));
        }
      }
    }

  newOverlappingMap = Teuchos::rcp(new Epetra_Map(-1, overlappingGIDs.size(),
      &overlappingGIDs[0], baseMap_->IndexBase(), *comm_));

  newMap = Teuchos::rcp(new Epetra_Map(-1, localGIDs.size(),
      &localGIDs[0], baseMap_->IndexBase(), *comm_));

  newObject = Teuchos::rcp(new HierarchicalMap(comm_, newMap, newOverlappingMap,
      newGroupPointer, newGidList, "Separator Nodes", myLevel_));

  return newObject;

  }

////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<const HierarchicalMap> 
HierarchicalMap::SpawnLocalSeparators
        (Teuchos::RCP<Teuchos::Array<HYMLS::SepNode> > regroup) const
  { 
  HYMLS_PROF3(label_,"SpawnLocalSeparators");

  Teuchos::RCP<const HierarchicalMap> newObject=Teuchos::null;
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGidList =
        Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGroupPointer =
        Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());
  
  // Start out from the standard Separator object. All local separators are located
  // in its baseMap_
  Teuchos::RCP<const HierarchicalMap> sepObject = this->Spawn(Separators);

  for (int sd = 0; sd < sepObject->NumMySubdomains(); sd++)
    {
    newGidList->append(Teuchos::Array<int>());
    newGroupPointer->append(Teuchos::Array<int>(2));
    for (int grp = 1; grp < sepObject->NumGroups(sd); grp++)
      {
      if (sepObject->NumElements(sd, grp) > 0 && sepObject->GetMap()->MyGID(sepObject->GID(sd, grp, 0)))
        {
        Teuchos::Array<int> gidList = sepObject->GetGroup(sd, grp);
        int offset = *((*newGroupPointer)[sd].end()-1);
        int len = gidList.size();
        if (len>0)
          {
          (*newGroupPointer)[sd].append(offset + len);
          std::copy(gidList.begin(), gidList.end(), std::back_inserter((*newGidList)[sd]));
          }
        }
      }
    }

  newObject = Teuchos::rcp(new HierarchicalMap(comm_, sepObject->GetMap(), sepObject->GetMap(),
      newGroupPointer, newGidList, "Local Separator Nodes", myLevel_));

  return newObject;
  }

  Teuchos::RCP<const Epetra_Map> HierarchicalMap::SpawnMap
        (int sd, SpawnStrategy strat) const
    {
    HYMLS_PROF3(label_,"SpawnMap");
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
      HYMLS_DEBUG("Spawn map for subdomain "<<sd);
      // int* MyElements = overlappingMap_->MyGlobalElements();      
      int offset = -1;
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
      map = Teuchos::rcp(new Epetra_Map(-1, length, &((*gidList_)[sd][0+offset]),
                baseMap_->IndexBase(), comm));
                
      spawnedMaps_[idx][sd]=map;
      }
    
    return map;
    }

// this doesn't formally belong to this class but has to be implemented somewhere
std::ostream & operator<<(std::ostream& os, const HierarchicalMap& h)
  {
  return h.Print(os);
  }

//! given a list of GIDs, returns a list of subdomains to which they belong
int HierarchicalMap::getSubdomainList(int num_gids, int* gids, int* sd) const
  {
  HYMLS_PROF3(label_,"getSubdomainList");
  int offset=0; // we do a cyclic search, which
                // should be efficient if the indices
                // belong to the same or adjacent subdomains
  for (int i=0;i<num_gids;i++)
    {
    sd[i]=-1;
    int lid = overlappingMap_->LID(gids[i]);
    if (lid>0)
      {
      for (int j=0;j<NumMySubdomains();j++)
        {
        int jj=MOD(offset+j,NumMySubdomains());
        if (((*groupPointer_)[jj][0]<=lid)
         && (*((*groupPointer_)[jj].end()-1)>lid))
           {
           sd[i]=jj;
           offset=jj;
           break;
           }
        }
      }
    }
  return 0;
  }

//! given a subdomain, returns a list of GIDs that belong to the subdomain
int HierarchicalMap::getSeparatorGIDs(int sd, int *gids) const
  {
  HYMLS_PROF3(label_, "getSubdomainGIDs");
  if (sd < 0 || sd > NumMySubdomains())
    {
    Tools::Warning("Subdomain index out of range!", __FILE__, __LINE__);
    return -1;
    }

  // create vector with global indices
  int pos = 0;

  // loop over all groups except the first (first is interior elements),
  // that is separator groups and retained elements
  int numGroups = NumGroups(sd);
  for (int grp = 1; grp < numGroups; grp++)
    {
    // loop over all elements of each separator group
    int numElements = NumElements(sd, grp);
    for (int j = 0; j < numElements; j++)
      {
      gids[pos++] = GID(sd, grp, j);
      }
    }
  return 0;
  }

//! given a subdomain, returns a list of GIDs that belong to the subdomain
int HierarchicalMap::getSeparatorGIDs(int sd, Epetra_IntSerialDenseVector &gids) const
  {
  HYMLS_PROF3(label_, "getSubdomainGIDs");
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

}
