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
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StrUtils.hpp"
#include "Epetra_Util.h"

#include <algorithm>


#include "HYMLS_MatrixUtils.H"


typedef Teuchos::Array<int>::iterator int_i;

namespace HYMLS {

  //empty constructor
  HierarchicalMap::HierarchicalMap(Teuchos::RCP<const Epetra_Comm> comm, 
        Teuchos::RCP<const Epetra_Map> baseMap, int numMySubdomains,
        std::string label, int level)
        : comm_(comm),
          baseMap_(baseMap),
          overlappingMap_(Teuchos::null),
          myLevel_(level), label_(label+" (level "+Teuchos::toString(level)+")")
    {
    START_TIMER3(label_,"Constructor");
    this->Reset(numMySubdomains);
    }

  //private constructor
  HierarchicalMap::HierarchicalMap(
        Teuchos::RCP<const Epetra_Comm> comm, 
        Teuchos::RCP<const Epetra_Map> baseMap,
        Teuchos::RCP<const Epetra_Map> overlappingMap,
        Teuchos::RCP<Teuchos::Array< Teuchos::Array<int> > > groupPointer,
        std::string label, int level)
  : comm_(comm),
    baseMap_(baseMap),
    overlappingMap_(overlappingMap),
    groupPointer_(groupPointer),
    myLevel_(level),
    label_(label+" (level "+Teuchos::toString(level)+")")
    {
    START_TIMER3(label_,"HierarchicalMap Constructor");
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
    START_TIMER3("HierarchicalMap","Destructor");
    }

  int HierarchicalMap::Reset(int numMySubdomains)
    {
    START_TIMER2(label_,"Reset");
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
    START_TIMER2(label_,"FillComplete");
    for (int i=0;i<spawnedObjects_.size();i++) spawnedObjects_[i]=Teuchos::null;
    for (int i=0;i<spawnedMaps_.size();i++)  
      {
      spawnedMaps_[i].resize(NumMySubdomains());
      for (int sd=0;sd<NumMySubdomains();sd++)
        {
        spawnedMaps_[i][sd]=Teuchos::null;
        }
      }
    // adjust the group pointer and form contiguous array of GIDs
    Teuchos::Array<int> all_gids;
    for (int sd=0;sd<NumMySubdomains();sd++)
      {
      std::copy((*gidList_)[sd].begin(),(*gidList_)[sd].end(),std::back_inserter(all_gids));
      if (sd>0)
        {
        for (int_i j=(*groupPointer_)[sd].begin();j!=(*groupPointer_)[sd].end();j++)
          {
          *j+=*((*groupPointer_)[sd-1].end()-1);
          }
        }
      }
    int numel = all_gids.size();
    int *my_gids = numel>0? &(all_gids[0]) : NULL;
    
    // sort all elements groupwise in lexicographic ordering.
    // Note that sorting sepnodes just puts them in the right
    // group-wise ordering and in each group they sometimes become
    // sorted in a strange way.
    for (int sd=0;sd<NumMySubdomains();sd++)
      {
      for (int grp=0;grp<NumGroups(sd);grp++)
        {
        int len=NumElements(sd,grp);
        Epetra_Util::Sort(true,len, my_gids + (*groupPointer_)[sd][grp],
          0, NULL, 0, NULL);
        }
      }
    
    overlappingMap_ = Teuchos::rcp(new Epetra_Map
        (-1,numel,my_gids,baseMap_->IndexBase(),*comm_));
    // we keep the gidLists so we can add more subdomains and groups
    // and call FillComplete() again.
    all_gids.resize(0);
    return 0;
    }

  int HierarchicalMap::FillStart()
    {
    START_TIMER2(label_,"FillStart");
    // adjust the group pointer back to local indexing per subdomain
    for (int sd=0;sd<NumMySubdomains();sd++)
      {
      DEBVAR(sd);
      if (sd>0)
        {
        for (int_i j=(*groupPointer_)[sd].begin();j!=(*groupPointer_)[sd].end();j++)
          {
          DEBVAR(*j);
          *j-=*((*groupPointer_)[sd-1].end()-1);
          }
        }
      }
    overlappingMap_ = Teuchos::null;
    return 0;
    }

  int HierarchicalMap::AddSubdomain(int min_id)
    {
    START_TIMER3(label_,"AddSubdomain");
    if (Filled()) this->FillStart();
    int id = groupPointer_->size();
    if (id<min_id) id=min_id;
    groupPointer_->resize(id+1);
    gidList_->resize(id+1);
    (*groupPointer_)[id].resize(1);
    (*groupPointer_)[id][0]=0;
    return id;
    }
    
  int HierarchicalMap::AddGroup(int sd, Teuchos::Array<int>& gidList)
    {
    START_TIMER3(label_,"AddGroup");
    if (Filled()) this->FillStart();

    if (sd>=groupPointer_->size())
      {
      Tools::Warning("invalid subdomain index",__FILE__,__LINE__);
      return -1; //AddSubdomain has to be called to generate a valid sd
      }
      
    DEBVAR(sd);
    DEBVAR(gidList);
    int offset=*((*groupPointer_)[sd].end()-1);
    int len = gidList.size();
    (*groupPointer_)[sd].append(offset+len);
    if (len>0)
      {
      std::copy(gidList.begin(),gidList.end(),std::back_inserter((*gidList_)[sd]));
      }
    return (*groupPointer_)[sd].length()-1;
    }

  //! print domain decomposition to file
  std::ostream& HierarchicalMap::Print(std::ostream& os) const
    {
    if (!Filled())
      {
      os << "(object not filled)" << std::endl;
      return os;
      }
    START_TIMER3(label_,"Print");
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

  //! given a subdomain sd and a global node ID gid, return the local
  //! group id in which the GID appears first, or -1 if it is not found
  //! in this subdomain
  int HierarchicalMap::GetGroupID(int sd, int gid) const
    {
    if (!Filled()) return -2;
    if (sd>=NumMySubdomains()) return -3;
    int group=-1; // if not found we return -1
    for (int grp=0;grp<NumGroups(sd);grp++)
      {
      for (int i=0;i<NumElements(sd,grp);i++)
        {
        if (GID(sd,grp,i)==gid)
          {
          group=grp;
          break;
          }
        }//i
      if (group>=0) break;
      }
    return group;
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
    START_TIMER3(label_,"Spawn");
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
  START_TIMER3(label_,"SpawnInterior");    

  Teuchos::RCP<const HierarchicalMap> newObject=Teuchos::null;
  Teuchos::RCP<Epetra_Map> newMap=Teuchos::null;
  Teuchos::RCP<Epetra_Map> newOverlappingMap=Teuchos::null;
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGroupPointer =
        Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());
  
  int base = baseMap_->IndexBase();  
  
  int num = NumMyInteriorElements();
  int *myElements = new int[num];
  int pos=0;
  for (int sd=0;sd<NumMySubdomains();sd++)
    for (int j=0;j<NumInteriorElements(sd);j++)
      {
      myElements[pos++]=GID(sd,0,j);
      }
  newMap=Teuchos::rcp(new Epetra_Map(-1,num,myElements,base,*comm_));
  newOverlappingMap=newMap;

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

  newObject = Teuchos::rcp(new HierarchicalMap
        (comm_,newMap,newOverlappingMap,newGroupPointer,"Interior Nodes",myLevel_) );

  return newObject;
  }

/////////////////////////////////////////////////////////////////////////////////////

  Teuchos::RCP<const HierarchicalMap> 
  HierarchicalMap::SpawnSeparators() const
  {
  START_TIMER3(label_,"SpawnSeparators");

  if (!Filled()) Tools::Error("object not filled",__FILE__,__LINE__);

  Teuchos::RCP<const HierarchicalMap> newObject=Teuchos::null;
  Teuchos::RCP<Epetra_Map> newMap=Teuchos::null;
  Teuchos::RCP<Epetra_Map> newOverlappingMap=Teuchos::null;
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGroupPointer =
        Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());
  
  int base = baseMap_->IndexBase();  

  // all separator groups should appear as a new subdomain,
  // and they should appear exactly once.
  // all separator groups shared by at least two processor partitions
  // should appear as separator group of the last subdomain, exactly
  // once each.

    
  // (2) make a unique list of separator nodes on this partition and
  // one of those connected to this 
  // processor partition (not sorted correctly)
  Teuchos::Array<int> InteriorIDs;
  Teuchos::Array<int> SeparatorIDs;

  for (int sd=0;sd<NumMySubdomains();sd++)
    {
    for (int grp=1;grp<=NumSeparatorGroups(sd);grp++)
      {
      if (NumElements(sd,grp)>0)
        {
        if (baseMap_->MyGID(GID(sd,grp,0)))
          {
          for (int j=0;j<NumElements(sd,grp);j++)
            {
            InteriorIDs.append(GID(sd,grp,j));
            }
          }
        else
          {
          for (int j=0;j<NumElements(sd,grp);j++)
            {
            SeparatorIDs.append(GID(sd,grp,j));
            }
          }
        }
      }
    }
    
  // each separator node belongs to exactly one separator group,
  // so if we call 'unique' we keep each separator (including those
  // on a different partition) exactly once
  std::sort(InteriorIDs.begin(), InteriorIDs.end());
  int_i end_interior=std::unique(InteriorIDs.begin(),InteriorIDs.end());
  std::sort(SeparatorIDs.begin(), SeparatorIDs.end());
  int_i end_separators=std::unique(SeparatorIDs.begin(),SeparatorIDs.end());
    
  int numInteriorElements=std::distance(InteriorIDs.begin(),end_interior);
  int numSeparatorElements=std::distance(SeparatorIDs.begin(),end_separators);  
  int numOverlappingElements = numInteriorElements+numSeparatorElements;
  
  DEBVAR(numInteriorElements);
  DEBVAR(numSeparatorElements);
 
  // make a temporary overlapping map that has
  // overlap between physical partiitions and is not ordered
  // correctly.

  int *myOverlappingElements = new int[numOverlappingElements];
  int pos=0;
  for (int_i i=InteriorIDs.begin();i!=end_interior;i++)
    {
    myOverlappingElements[pos++]=*i;
    }
  for (int_i i=SeparatorIDs.begin();i!=end_separators;i++)
    {
    myOverlappingElements[pos++]=*i;
    }

  Teuchos::RCP<Epetra_Map> tmpOverlappingMap =
    Teuchos::rcp(new Epetra_Map(-1,numOverlappingElements,myOverlappingElements,base,*comm_));

  DEBVAR(*tmpOverlappingMap);
  
  Teuchos::Array<int> groupSize;
  Epetra_IntVector groupID(*tmpOverlappingMap);
  
  groupID.PutValue(-1);
  
  // assign group-IDs to owned separators (new subdomains)
  int num_subdomains=0;


//CAVEAT: this is just for debugging/testing, if it is defined
// the code may not work for general problems!
#ifdef SHIFT_PRESSURE_TO_END
  const int dof=3; // assuming 2D Stokes here
  const int pressure=2;
#else
  const int dof=1;
  const int pressure=-1;
#endif  

#ifdef SHIFT_SINGLETONS_TO_END
  const int singleton=1;
#else
  const int singleton=0;
#endif

  // first non-singletons  
  for (int sd=0;sd<NumMySubdomains();sd++)
    {
    DEBVAR(sd);
    for (int grp=1;grp<NumGroups(sd);grp++)
      {
      DEBVAR(grp);
      DEBVAR(NumElements(sd,grp));
      if (NumElements(sd,grp)>singleton)
        {
        int gid = GID(sd,grp,0);
        if (baseMap_->MyGID(gid))
          {
          int lid = tmpOverlappingMap->LID(gid);
#ifdef TESTING
          if (lid<0) Tools::Error("inconsistency in ordering!!!",__FILE__,__LINE__);
#endif
          if (groupID[lid]==-1) 
            {
            DEBUG("new group: "<<num_subdomains);
            groupSize.append(NumElements(sd,grp));
            for (int j=0;j<NumElements(sd,grp);j++)
              {
              int gid_j = this->GID(sd,grp,j);
              int lid_j = tmpOverlappingMap->LID(gid_j);
#ifdef DEBUGGING
              Tools::deb() << gid_j<<"/"<<lid_j<<" ";
#endif
              groupID[lid_j]=num_subdomains;
              }// for j
            DEBUG("");
            num_subdomains++;
            }// if not assigned
#ifdef DEBUGGING
          else
            {
            DEBVAR(groupID[lid]);
            }
#endif            
          }// if belongs to me
        }// non-singleton
      }// for grp
    }//for sd

  // non-pressure singletons
  if (singleton==1)
    {
    for (int sd=0;sd<NumMySubdomains();sd++)
      {
      for (int grp=1;grp<NumGroups(sd);grp++)
        {
        if (NumElements(sd,grp)==singleton)
          {
          int gid = GID(sd,grp,0);
          if (MOD(gid,dof)!=pressure)
            {
            if (baseMap_->MyGID(gid))
              {
              int lid = tmpOverlappingMap->LID(gid);
              if (groupID[lid]<0)
                {
                groupSize.append(1);
                groupID[lid]=num_subdomains;
                num_subdomains++;
                }// if not assigned
              }// if belongs to me
            }// non-pressure
          }//singleton
        }// for grp
      }//for sd
    }// singletons shifted to end?
  // pressure singletons
  if (pressure>=0)
    {
    for (int sd=0;sd<NumMySubdomains();sd++)
      {
      for (int grp=1;grp<NumGroups(sd);grp++)
        {
        if (NumElements(sd,grp)==1)
          {
          int gid = GID(sd,grp,0);
          if (MOD(gid,dof)==pressure)
            {
            if (baseMap_->MyGID(gid))
              {
              int lid = tmpOverlappingMap->LID(gid);
              if (groupID[lid]<0)
                {
                groupSize.append(1);
                groupID[lid]=num_subdomains;
                num_subdomains++;
                }// if not assigned
              }// if belongs to me
            }// pressure
          }//singleton
        }// for grp
      }//for sd
    }// shift pressures to end
    
  // if there are no local separator groups, 
  // transform them into a single group without
  // any elements. We do this because we want to
  // add non-local separators as groups of the 
  // last new subdomain.
  if (num_subdomains==0 && (NumMySubdomains()>0))
    {
    groupSize.append(0);
    num_subdomains++;
    }
      
  DEBVAR(num_subdomains);
  
  // assign group-IDs to separators on other partitions
  // (new separators)
  int num_separator_groups=0;
  DEBUG("off-processor nodes:");
  for (int sd=0;sd<NumMySubdomains();sd++)
    {
    for (int grp=1;grp<NumGroups(sd);grp++)
      {
      int gid = GID(sd,grp,0);
      if (!baseMap_->MyGID(gid))
        {
        int lid = tmpOverlappingMap->LID(gid);
#ifdef TESTING
          if (lid<0) Tools::Error("inconsistency in ordering!!!",__FILE__,__LINE__);
#endif        
        if (groupID[lid]<0)
          {
          DEBUG("new group: "<<num_subdomains+num_separator_groups);
          groupSize.append(NumElements(sd,grp));
          for (int j=0;j<NumElements(sd,grp);j++)
            {
            int gid_j = this->GID(sd,grp,j);
            int lid_j = tmpOverlappingMap->LID(gid_j);
            groupID[lid_j]=num_subdomains+num_separator_groups;
#ifdef DEBUGGING
              Tools::deb() << gid_j <<"/"<<lid_j<<" ";
#endif
            }// for j
          DEBUG("");
          num_separator_groups++;
          }// if not assigned
#ifdef DEBUGGING
        else
          {
          DEBVAR(groupID[lid]);
          }
#endif
        }// if belongs to someone else
      }// for grp
    }//for sd

  DEBVAR(num_separator_groups);
  
  DEBVAR(groupSize);
  DEBVAR(groupID);

#ifdef TESTING
for (int i=0;i<groupID.MyLength();i++)
  {
  if (groupID[i]<0)
    {
    std::cerr<<"P"+Teuchos::toString(comm_->MyPID())+": "+Label()+
    " - node "+Teuchos::toString(tmpOverlappingMap->GID(i))+
    " not assigned to any group!\n";
    }
  }
#endif

  // groupPointer may have empty subdomain group
  newGroupPointer->resize(num_subdomains);

  // currently we add all new separator elements
  // (i.e. those on other partitions) to a single
  // group (group 0). TODO: can we do something smarter?
  pos=0;
  for (int i=0;i<num_subdomains;i++)
    {
    (*newGroupPointer)[i].resize(2);
    (*newGroupPointer)[i][0]=pos;
    pos+=groupSize[i];
    (*newGroupPointer)[i][1]=pos;
    }
    
    
  // add new separator nodes to the end of the ordering (last subdomain)
  for (int i=0;i<num_separator_groups; i++)
    {
    pos+=groupSize[num_subdomains+i];
    (*newGroupPointer)[num_subdomains-1].append(pos);
    }

  // now we have the groupPointer - use it to reorder the elements of the map
  for (int i=0;i<num_subdomains+num_separator_groups;i++)
    {
    groupSize[i]=0;
    }
    
  for (int i=0;i<groupID.MyLength();i++)
    {
    // all new separator groups (groupID>=num_subdomains) belong to the last subdomain:
    int idx=groupID[i];
    int sd = std::min(idx,num_subdomains-1);
    // all new interior elements are put into groups 0 of their subdomain:
    int grp = std::max(idx-num_subdomains+1, 0);
    int pos = groupSize[idx];
    
    if (sd==-1)
      {
      /*
      std::cerr << "PROC: "<< comm_->MyPID()<<std::endl;
      std::cerr << "i="<<i<<std::endl;
      std::cerr << "groupID[i]="<<idx<<std::endl;
      std::cerr << "entire groupID array: "<<groupID<<std::endl;
      */
      HYMLS::Tools::Warning("unassigned GID!",__FILE__,__LINE__);
      }
    
    myOverlappingElements[(*newGroupPointer)[sd][grp]+pos]=tmpOverlappingMap->GID(i);
    groupSize[idx]++;
    }
// this can be used to get nicer pictures, but is 
// otherwise probably completely irrelevant:
//#define CENTRAL_VSUMS 1
#ifdef CENTRAL_VSUMS
  // we now move the most central node on each separator to the first
  // position. Thus we make a node in the center the Vsum node.
  // TODO: does this really matter?
  for (int sd=0;sd<num_subdomains;sd++)
    {
    for (int grp=0;grp<(*newGroupPointer)[sd].length()-1;grp++)
      {
      int sep_len = (*newGroupPointer)[sd][grp+1]-(*newGroupPointer)[sd][grp];
      if (sep_len>1)
        {
        int center  = sep_len/2;
        std::swap(myOverlappingElements[(*newGroupPointer)[sd][grp]],
              myOverlappingElements[(*newGroupPointer)[sd][grp]+center]);
        }
      }
    }
#endif    
  // first elements form the new non-overlapping map
  newMap=Teuchos::rcp(new 
    Epetra_Map(-1,numInteriorElements,myOverlappingElements,base,*comm_));

  newOverlappingMap=Teuchos::rcp(new 
    Epetra_Map(-1,numOverlappingElements,myOverlappingElements,base,*comm_));
  
  delete [] myOverlappingElements;
  DEBVAR(*newOverlappingMap);   
  
  newObject = Teuchos::rcp(new HierarchicalMap
        (comm_,newMap,newOverlappingMap,newGroupPointer,"Separator Nodes",myLevel_) );
  
  return newObject;
  }

////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<const HierarchicalMap> 
HierarchicalMap::SpawnLocalSeparators
        (Teuchos::RCP<Teuchos::Array<HYMLS::SepNode> > regroup) const
  { 
  START_TIMER3(label_,"SpawnLocalSeparators");

  Teuchos::RCP<const HierarchicalMap> newObject=Teuchos::null;
  Teuchos::RCP<Epetra_Map> newMap=Teuchos::null;
  Teuchos::RCP<Epetra_Map> newOverlappingMap=Teuchos::null;
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGroupPointer =
        Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());
  
  int base = baseMap_->IndexBase();  
  
  // start out from the standard Separator object. It's interior groups are the new 
  // subdomains, and we split them according to the criteria given by the user in the
  // SepNode array.
  Teuchos::RCP<const HierarchicalMap> sepObject 
        = this->Spawn(Separators);
        
  // the number of elements in the new object is the number of interior
  // elements in the old one
  int NumMyElements = sepObject->NumMyInteriorElements();
  int* MyElements=new int[NumMyElements];
  newGroupPointer->resize(sepObject->NumMySubdomains());

  int pos=0;
  for (int sep=0;sep<sepObject->NumMySubdomains();sep++)
    {
    DEBVAR(sep)
    (*newGroupPointer)[sep].append(pos);
    if (regroup!=Teuchos::null)
      {
      // sort the SepNodes
      int begin = sepObject->LID(sep,0,0);
      int end = begin + sepObject->NumElements(sep,0);
      std::sort(regroup->begin()+begin,regroup->begin()+end);
      MyElements[pos++] = (*regroup)[begin].GID();
      DEBUG("|"<<pos-1<<"| "<<(*regroup)[begin]);
      for (int j=1;j<sepObject->NumElements(sep,0);j++)
        {
        MyElements[pos] = (*regroup)[begin+j].GID();
        DEBUG("|"<<pos<<"| "<<(*regroup)[begin+j]);
        // note: comparing SepNode objects means comparing there
        // connectivity and variable type, not their GIDs. So 
        // this call does the grouping (together with the sort)
        if ((*regroup)[begin+j]!=(*regroup)[begin+j-1])
          {
          DEBUG("this is a new group");
          (*newGroupPointer)[sep].append(pos);
          }
        pos++;
        }
      }
    else
      {
      for (int j=0;j<sepObject->NumElements(sep,0);j++)
        {
        MyElements[pos++] = sepObject->GID(sep,0,j);
        }
      }
    (*newGroupPointer)[sep].append(pos);
    }

  //for this object the map and overlapping map are the same
  newMap=Teuchos::rcp(new Epetra_Map(-1,NumMyElements,MyElements,base,*comm_));

  newOverlappingMap=newMap;
  
  delete [] MyElements;

  newObject = Teuchos::rcp(new HierarchicalMap
        (comm_,newMap,newOverlappingMap,newGroupPointer,"Local Separator Nodes",myLevel_) );
  
//  std::cout << *sepObject << std::endl;
//  std::cout << *newObject << std::endl;
  return newObject;
  }

  Teuchos::RCP<const Epetra_Map> HierarchicalMap::SpawnMap
        (int sd, SpawnStrategy strat) const
    {
    START_TIMER3(label_,"SpawnMap");
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
      DEBUG("Spawn map for subdomain "<<sd);
      int* MyElements = overlappingMap_->MyGlobalElements();      
      int offset,length;
      if (strat==Interior)
        {
        DEBUG("interior map");
        offset = (*groupPointer_)[sd][0];
        length = NumInteriorElements(sd);
        }
      else if (strat==Separators)
        {
        DEBUG("separator map");
        offset = (*groupPointer_)[sd][1];
        length = NumSeparatorElements(sd);
        }
      else if (strat==All)
        {
        DEBUG("complete map");
        offset = (*groupPointer_)[sd][0];
        length = NumElements(sd);
        }
      DEBVAR(offset);
      DEBVAR(length);
      Epetra_SerialComm comm;
      map = Teuchos::rcp(new Epetra_Map(-1, length, MyElements+offset,
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
  START_TIMER3(label_,"getSubdomainList");
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

      
  
}
