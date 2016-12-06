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
    for (int i=0;i<spawnedObjects_.size();i++) spawnedObjects_[i]=Teuchos::null;
    for (int i=0;i<spawnedMaps_.size();i++)  
      {
      spawnedMaps_[i].resize(NumMySubdomains());
      for (int sd=0;sd<NumMySubdomains();sd++)
        {
        spawnedMaps_[i][sd]=Teuchos::null;
        }
      }

    Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGroupPointer =
      Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());
    Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGidList =
      Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());
  
  int base = baseMap_->IndexBase();
  Teuchos::Array<int> IDs;

  std::cout << "BASEMAP " << *baseMap_ << std::endl;

  for (int sd=0;sd<NumMySubdomains();sd++)
    {
    for (int grp=0;grp<NumGroups(sd);grp++)
      {
      for (int j=0;j<NumElements(sd,grp);j++)
        {
        IDs.append(GID(sd,grp,j));
        }
      }
    }

  std::sort(IDs.begin(), IDs.end());
  Teuchos::Array<int>::iterator end=std::unique(IDs.begin(),IDs.end());

  int numElements=std::distance(IDs.begin(),end);
  int numOverlappingElements = numElements;

  int *myOverlappingElements = new int[numOverlappingElements];
  int pos=0;
  for (Teuchos::Array<int>::iterator i=IDs.begin();i!=end;i++)
    {
    myOverlappingElements[pos++]=*i;
    }

  Teuchos::RCP<Epetra_Map> tmpOverlappingMap =
    Teuchos::rcp(new Epetra_Map(-1,numOverlappingElements,myOverlappingElements,base,*comm_));

  HYMLS_DEBVAR(*tmpOverlappingMap);

  Epetra_IntVector vec(*baseMap_);
  for (Teuchos::Array<int>::iterator i=IDs.begin();i!=end;i++)
    {
    int lid = baseMap_->LID(*i);
    if (lid != -1)
      vec[lid] = 1;
    }
  Epetra_Import imp(*tmpOverlappingMap, *baseMap_);
  Epetra_IntVector overlappingVec(*tmpOverlappingMap);
  overlappingVec.Import(vec, imp, Insert);

  pos = 0;
  for (Teuchos::Array<int>::iterator i=IDs.begin();i!=end;i++)
    {
    if (overlappingVec[tmpOverlappingMap->LID(*i)])
      myOverlappingElements[pos++] = *i;
    }

  overlappingMap_=Teuchos::rcp(new 
    Epetra_Map(-1,pos,myOverlappingElements,base,*comm_));
  
  delete [] myOverlappingElements;

  for (int sd=0;sd<NumMySubdomains();sd++)
    {
    newGidList->append(Teuchos::Array<int>());
    newGroupPointer->append(Teuchos::Array<int>(1));
    for (int grp=0;grp<NumGroups(sd);grp++)
      {
      Teuchos::Array<int> gidList;
      for (int j=0;j<NumElements(sd,grp);j++)
        {
        int gid = GID(sd, grp, j);
        if (overlappingVec[tmpOverlappingMap->LID(gid)])
          {
          gidList.append(gid);
          }
        }
      int offset = *((*newGroupPointer)[sd].end()-1);
      int len = gidList.size();
      if (len>0)
        {
        (*newGroupPointer)[sd].append(offset + len);
        std::copy(gidList.begin(), gidList.end(), std::back_inserter((*newGidList)[sd]));
        }
      }
    }

    Teuchos::Array<int> all_gids;
    for (int sd=0;sd<NumMySubdomains();sd++)
      std::copy((*newGidList)[sd].begin(),(*newGidList)[sd].end(),std::back_inserter(all_gids));
    int numel = all_gids.size();
    int *my_gids = numel>0? &(all_gids[0]) : NULL;
    overlappingMap_ = Teuchos::rcp(new Epetra_Map
        (-1,numel,my_gids,baseMap_->IndexBase(),*comm_));
  
    // adjust the group pointer and form contiguous array of GIDs
    // Teuchos::Array<int> all_gids;
    // for (int sd=0;sd<NumMySubdomains();sd++)
    //   {
    //   std::copy((*newGidList)[sd].begin(),(*newGidList)[sd].end(),std::back_inserter(all_gids));
      // if (sd>0)
      //   {
      //   for (Teuchos::Array<int>::iterator j = (*groupPointer_)[sd].begin();
      //        j != (*groupPointer_)[sd].end(); j++)
      //     {
      //     *j += *((*groupPointer_)[sd-1].end()-1);
      //     }
      //   }
      // }
  //   gidList_ = newGidList;
  //   groupPointer_ = newGroupPointer;
    // int numel = all_gids.size();
    // int *my_gids = numel>0? &(all_gids[0]) : NULL;
    
  //   // // sort all elements groupwise in lexicographic ordering.
  //   // // Note that sorting sepnodes just puts them in the right
  //   // // group-wise ordering and in each group they sometimes become
  //   // // sorted in a strange way.
  //   // for (int sd=0;sd<NumMySubdomains();sd++)
  //   //   {
  //   //   for (int grp=0;grp<NumGroups(sd);grp++)
  //   //     {
  //   //     int len=NumElements(sd,grp);
  //   //     Epetra_Util::Sort(true,len, my_gids + (*groupPointer_)[sd][grp],
  //   //       0, NULL, 0, NULL);
  //   //     }
  //   //   }
    // overlappingMap_ = Teuchos::rcp(new Epetra_Map
    //     (-1,numel,my_gids,baseMap_->IndexBase(),*comm_));

  std::cout << "REALOVERLAPPINGMAP " << *overlappingMap_ << std::endl;
    gidList_ = newGidList;
    std::cout << "GRPPTR " << *groupPointer_ << std::endl;
    groupPointer_ = newGroupPointer;
    std::cout << "GRPPTR " << *groupPointer_ << std::endl;
  //   // we keep the gidLists so we can add more subdomains and groups
  //   // and call FillComplete() again.
  //   // for (int sd = NumMySubdomains()-1; sd > 0; sd--)
  //   //   for (Teuchos::Array<int>::iterator j = (*groupPointer_)[sd].begin();
  //   //        j != (*groupPointer_)[sd].end(); j++)
  //   //     *j -= *((*groupPointer_)[sd-1].end()-1);

  // // // Get elements from the original overlapping map which contains duplicate elements
  // // Teuchos::Array<int> myElements(overlappingMap_->NumMyElements());
  // // overlappingMap_->MyGlobalElements(&myElements[0]);

  // // // Sort the list and only keep the unique elements
  // // std::sort(myElements.begin(), myElements.end());
  // // auto last = std::unique(myElements.begin(), myElements.end());
  // // myElements.erase(last, myElements.end());

  // // // Return a new overlapping map with only unique elements
  // // overlappingMap_ =  Teuchos::rcp(new Epetra_Map(-1, myElements.size(), &myElements[0],
  // //     overlappingMap_->IndexBase(), *comm_));
  //     all_gids.resize(0);
    return 0;
    }

  int HierarchicalMap::FillStart()
    {
    HYMLS_PROF2(label_,"FillStart");
    // adjust the group pointer back to local indexing per subdomain
    // for (int sd = NumMySubdomains()-1; sd > 0; sd--)
    //   {
    //   for (Teuchos::Array<int>::iterator j = (*groupPointer_)[sd].begin();
    //        j != (*groupPointer_)[sd].end(); j++)
    //     {
    //     HYMLS_DEBVAR(*j);
    //     *j -= *((*groupPointer_)[sd-1].end()-1);
    //     }
    //   }
    overlappingMap_ = Teuchos::null;
    return 0;
    }
    
  int HierarchicalMap::AddGroup(int sd, Teuchos::Array<int>& gidList)
    {
    HYMLS_PROF3(label_,"AddGroup");
    // if (Filled()) this->FillStart();

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
    // if (Filled())
    //   for (int sd = NumMySubdomains()-1; sd > 0; sd--)
    //     for (Teuchos::Array<int>::iterator j = (*groupPointer_)[sd].begin();
    //          j != (*groupPointer_)[sd].end(); j++)
    //       *j -= *((*groupPointer_)[sd-1].end()-1);

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

    // if (Filled())
    //   for (int sd = 1; sd < NumMySubdomains(); sd++)
    //     for (Teuchos::Array<int>::iterator j = (*groupPointer_)[sd].begin();
    //          j != (*groupPointer_)[sd].end(); j++)
    //       *j += *((*groupPointer_)[sd-1].end()-1);

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
  Teuchos::RCP<Epetra_Map> newOverlappingMap=Teuchos::null;
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGroupPointer =
        Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGidList =
        Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());
  
  int base = baseMap_->IndexBase();  
  
  int num = NumMyInteriorElements();
  int *myElements = new int[num];
  int pos=0;
  for (int sd=0;sd<NumMySubdomains();sd++)
    {
    newGidList->append(Teuchos::Array<int>());
    newGroupPointer->append(Teuchos::Array<int>(1));
    Teuchos::Array<int> gidList;
    for (int j=0;j<NumInteriorElements(sd);j++)
      {
      int gid = GID(sd,0,j);
      if (baseMap_->MyGID(gid))
        {
        myElements[pos++]=gid;
        gidList.append(gid);
        }
      }
    int len = gidList.size();
    if (len>0)
      {
      (*newGroupPointer)[sd].append(len);
      std::copy(gidList.begin(), gidList.end(), std::back_inserter((*newGidList)[sd]));
      }
    }

  newMap=Teuchos::rcp(new Epetra_Map(-1,pos,myElements,base,*comm_));
  newOverlappingMap=newMap;
  std::cout << "INTMAP " << *newMap;
  delete [] myElements;

  // newGroupPointer->resize(NumMySubdomains());
  // if (NumMySubdomains()>0)
  //   {
  //   (*newGroupPointer)[0].resize(2);
  //   (*newGroupPointer)[0][0]=0;
  //   (*newGroupPointer)[0][1]=NumInteriorElements(0);
  //   }
  // for (int sd=1;sd<NumMySubdomains();sd++)
  //   {
  //   (*newGroupPointer)[sd].resize(2);
  //   (*newGroupPointer)[sd][0]=(*newGroupPointer)[sd-1][1];
  //   (*newGroupPointer)[sd][1]=(*newGroupPointer)[sd][0]+NumInteriorElements(sd);
  //   }

  // newObject = Teuchos::rcp(new HierarchicalMap
  //   (comm_,newMap,newOverlappingMap,newGroupPointer,gidList_,"Interior Nodes",myLevel_) );
  newObject = Teuchos::rcp(new HierarchicalMap
    (comm_,newMap,newOverlappingMap,newGroupPointer,newGidList,"Interior Nodes",myLevel_) );

  return newObject;
  }

/////////////////////////////////////////////////////////////////////////////////////

  Teuchos::RCP<const HierarchicalMap> 
  HierarchicalMap::SpawnSeparators() const
  {
  HYMLS_PROF3(label_,"SpawnSeparators");

  if (!Filled()) Tools::Error("object not filled",__FILE__,__LINE__);

  Teuchos::RCP<const HierarchicalMap> newObject=Teuchos::null;
  Teuchos::RCP<Epetra_Map> newMap=Teuchos::null;
  Teuchos::RCP<Epetra_Map> newOverlappingMap=Teuchos::null;
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGroupPointer =
        Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGidList =
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

  // for (int sd=0;sd<NumMySubdomains();sd++)
  //   {
  //   for (int grp=1;grp<=NumSeparatorGroups(sd);grp++)
  //     {
  //     if (NumElements(sd,grp)>0)
  //       {
  //       if (baseMap_->MyGID(GID(sd,grp,0)))
  //         {
  //         for (int j=0;j<NumElements(sd,grp);j++)
  //           {
  //           InteriorIDs.append(GID(sd,grp,j));
  //           }
  //         }
  //       else
  //         {
  //         for (int j=0;j<NumElements(sd,grp);j++)
  //           {
  //           SeparatorIDs.append(GID(sd,grp,j));
  //           }
  //         }
  //       }
  //     }
  //   }

  for (int sd=0;sd<NumMySubdomains();sd++)
    {
    for (int grp=1;grp<=NumSeparatorGroups(sd);grp++)
      {
      if (NumElements(sd,grp)>0)
        {
        for (int j=0;j<NumElements(sd,grp);j++)
          {
          int gid = GID(sd,grp,j);
          if (baseMap_->MyGID(gid))
            {
            InteriorIDs.append(gid);
            }
          else
            {
            SeparatorIDs.append(gid);
            }
          }
        }
      }
    }

  assert(SeparatorIDs.length() == 0);

  std::cout << InteriorIDs << std::endl;
  std::cout << SeparatorIDs << std::endl;
    
  // each separator node belongs to exactly one separator group,
  // so if we call 'unique' we keep each separator (including those
  // on a different partition) exactly once
  std::sort(InteriorIDs.begin(), InteriorIDs.end());
  Teuchos::Array<int>::iterator end_interior=std::unique(InteriorIDs.begin(),InteriorIDs.end());
  std::sort(SeparatorIDs.begin(), SeparatorIDs.end());
  Teuchos::Array<int>::iterator end_separators=std::unique(SeparatorIDs.begin(),SeparatorIDs.end());

  std::cout << InteriorIDs << std::endl;
  std::cout << SeparatorIDs << std::endl;
    
  int numInteriorElements=std::distance(InteriorIDs.begin(),end_interior);
  int numSeparatorElements=std::distance(SeparatorIDs.begin(),end_separators);
  int numOverlappingElements = numInteriorElements+numSeparatorElements;

  Teuchos::Array<int> all = InteriorIDs;
  all.resize(numInteriorElements);
  all.insert(all.end(), SeparatorIDs.begin(), SeparatorIDs.end());
  std::sort(all.begin(), all.end());
  std::unique(all.begin(),all.end());
  std::cout << all << std::endl;
  int sop = 0;
  for (int j=0; j<64; j++)
    {
    for (int i=0; i<64; i++)
      {
      if (sop < all.size() && i+j*64 == all[sop])
        {
        std::cout << "* ";
        sop++;
        }
      else
        std::cout << ". ";
      }
    std::cout << std::endl;
    }
  
  HYMLS_DEBVAR(numInteriorElements);
  HYMLS_DEBVAR(numSeparatorElements);
 
  // make a temporary overlapping map that has
  // overlap between physical partiitions and is not ordered
  // correctly.

  int *myOverlappingElements = new int[numOverlappingElements];
  int pos=0;
  for (Teuchos::Array<int>::iterator i=InteriorIDs.begin();i!=end_interior;i++)
    {
    myOverlappingElements[pos++]=*i;
    }
  for (Teuchos::Array<int>::iterator i=SeparatorIDs.begin();i!=end_separators;i++)
    {
    myOverlappingElements[pos++]=*i;
    }

  Teuchos::RCP<Epetra_Map> tmpOverlappingMap =
    Teuchos::rcp(new Epetra_Map(-1,numOverlappingElements,myOverlappingElements,base,*comm_));

  HYMLS_DEBVAR(*tmpOverlappingMap);

  Epetra_IntVector vec(*baseMap_);
  for (Teuchos::Array<int>::iterator i=InteriorIDs.begin();i!=end_interior;i++)
    {
    int lid = baseMap_->LID(*i);
    if (lid != -1)
      vec[lid] = 1;
    }
  Epetra_Import imp(*tmpOverlappingMap, *baseMap_);
  Epetra_IntVector overlappingVec(*tmpOverlappingMap);
  overlappingVec.Import(vec, imp, Insert);

  std::cout << InteriorIDs << std::endl;
  std::cout << *baseMap_;
  std::cout << *tmpOverlappingMap;

  pos = 0;
  for (Teuchos::Array<int>::iterator i=InteriorIDs.begin();i!=end_interior;i++)
    {
    if (overlappingVec[tmpOverlappingMap->LID(*i)])
      myOverlappingElements[pos++] = *i;
    }
  // first elements form the new non-overlapping map
  newMap=Teuchos::rcp(new 
    Epetra_Map(-1,pos,myOverlappingElements,base,*comm_));

  for (Teuchos::Array<int>::iterator i=SeparatorIDs.begin();i!=end_separators;i++)
    {
    if (overlappingVec[tmpOverlappingMap->LID(*i)])
      myOverlappingElements[pos++] = *i;
    }

  newOverlappingMap=Teuchos::rcp(new 
    Epetra_Map(-1,pos,myOverlappingElements,base,*comm_));

  std::cout << *newMap;
  std::cout << *newOverlappingMap;
  
  delete [] myOverlappingElements;
  HYMLS_DEBVAR(*newOverlappingMap);

  Teuchos::Array<int> todo;

  for (int sd=0;sd<NumMySubdomains();sd++)
    {
    newGidList->append(Teuchos::Array<int>());
    newGroupPointer->append(Teuchos::Array<int>(2));
    for (int grp=1;grp<=NumSeparatorGroups(sd);grp++)
      {
      if (NumElements(sd,grp)>0 &&
        std::find(todo.begin(), todo.end(), GID(sd, grp, 0)) == todo.end())
      // if (NumElements(sd,grp)>0)
        {
        Teuchos::Array<int> gidList;
        for (int j=0;j<NumElements(sd,grp);j++)
          {
          int gid = GID(sd, grp, j);
          if (overlappingVec[tmpOverlappingMap->LID(gid)])
            {
            gidList.append(gid);
            }
          }
        int offset = *((*newGroupPointer)[sd].end()-1);
        int len = gidList.size();
        if (len>0)
          {
          (*newGroupPointer)[sd].append(offset + len);
          std::copy(gidList.begin(), gidList.end(), std::back_inserter((*newGidList)[sd]));
          }
        todo.append(GID(sd, grp, 0));
        }
      }
    }

  Teuchos::Array<int> all_gids;
  for (int sd=0;sd<NumMySubdomains();sd++)
    std::copy((*newGidList)[sd].begin(),(*newGidList)[sd].end(),std::back_inserter(all_gids));
  int numel = all_gids.size();
  int *my_gids = numel>0? &(all_gids[0]) : NULL;
  newOverlappingMap = Teuchos::rcp(new Epetra_Map
    (-1,numel,my_gids,baseMap_->IndexBase(),*comm_));
  newMap = newOverlappingMap;

  all_gids.resize(0);
  todo.resize(0);
  for (int sd=0;sd<NumMySubdomains();sd++)
    {
    for (int grp=1;grp<=NumSeparatorGroups(sd);grp++)
      {
      int gid = GID(sd, grp, 0);
      if (NumElements(sd,grp)>0 &&
        std::find(todo.begin(), todo.end(), gid) == todo.end() && baseMap_->MyGID(gid))
      // if (NumElements(sd,grp)>0)
        {
        for (int j=0;j<NumElements(sd,grp);j++)
          {
          int gid = GID(sd, grp, j);
          if (overlappingVec[tmpOverlappingMap->LID(gid)])
            {
            all_gids.append(gid);
            }
          }
        todo.append(GID(sd, grp, 0));
        }
      }
    }
  numel = all_gids.size();
  my_gids = numel>0? &(all_gids[0]) : NULL;
  newMap = Teuchos::rcp(new Epetra_Map
    (-1,numel,my_gids,baseMap_->IndexBase(),*comm_));
  // newObject = Teuchos::rcp(new HierarchicalMap
  //   (comm_,newMap,newOverlappingMap,groupPointer_,gidList_,"Separator Nodes",myLevel_) );
  newObject = Teuchos::rcp(new HierarchicalMap
    (comm_,newMap,newOverlappingMap,newGroupPointer,newGidList,"Separator Nodes",myLevel_) );
  
  return newObject;
  
  Teuchos::Array<int> groupSize;
  Epetra_IntVector groupID(*tmpOverlappingMap);
  
  groupID.PutValue(-1);
  
  // assign group-IDs to owned separators (new subdomains)
  int num_subdomains=0;


//CAVEAT: this is just for debugging/testing, if it is defined
// the code may not work for general problems!
#ifdef SHIFT_PRESSURE_TO_END
  const int dof=4; // assuming 3D Stokes here
  const int pressure=3;
#warning "using SHIFT_PRESSURE_TO_END, a debugging feature for 3D Stokes only."
#else
  const int dof=1;
  const int pressure=-1;
#endif  

#ifdef SHIFT_SINGLETONS_TO_END
  const int singleton=1;
#warning "using SHIFT_SINGLETONS_TO_END, a debugging feature."
#else
  const int singleton=0;
#endif

  // first non-singletons  
  for (int sd=0;sd<NumMySubdomains();sd++)
    {
    HYMLS_DEBVAR(sd);
    for (int grp=1;grp<NumGroups(sd);grp++)
      {
      HYMLS_DEBVAR(grp);
      HYMLS_DEBVAR(NumElements(sd,grp));
      if (NumElements(sd,grp)>singleton)
        {
        int gid = GID(sd,grp,0);
        if (baseMap_->MyGID(gid))
          {
          int lid = tmpOverlappingMap->LID(gid);
#ifdef HYMLS_TESTING
          if (lid<0) Tools::Error("inconsistency in ordering!!!",__FILE__,__LINE__);
#endif
          if (groupID[lid]==-1) 
            {
            HYMLS_DEBUG("new group: "<<num_subdomains);
            groupSize.append(NumElements(sd,grp));
            for (int j=0;j<NumElements(sd,grp);j++)
              {
              int gid_j = this->GID(sd,grp,j);
              int lid_j = tmpOverlappingMap->LID(gid_j);
#ifdef HYMLS_DEBUGGING
              Tools::deb() << gid_j<<"/"<<lid_j<<" ";
#endif
              groupID[lid_j]=num_subdomains;
              }// for j
            HYMLS_DEBUG("");
            num_subdomains++;
            }// if not assigned
#ifdef HYMLS_DEBUGGING
          else
            {
            HYMLS_DEBVAR(groupID[lid]);
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
      
  HYMLS_DEBVAR(num_subdomains);
  
  // assign group-IDs to separators on other partitions
  // (new separators)
  int num_separator_groups=0;
  HYMLS_DEBUG("off-processor nodes:");
  for (int sd=0;sd<NumMySubdomains();sd++)
    {
    for (int grp=1;grp<NumGroups(sd);grp++)
      {
      int gid = GID(sd,grp,0);
      if (!baseMap_->MyGID(gid))
        {
        int lid = tmpOverlappingMap->LID(gid);
#ifdef HYMLS_TESTING
          if (lid<0) Tools::Error("inconsistency in ordering!!!",__FILE__,__LINE__);
#endif        
        if (groupID[lid]<0)
          {
          HYMLS_DEBUG("new group: "<<num_subdomains+num_separator_groups);
          groupSize.append(NumElements(sd,grp));
          for (int j=0;j<NumElements(sd,grp);j++)
            {
            int gid_j = this->GID(sd,grp,j);
            int lid_j = tmpOverlappingMap->LID(gid_j);
            groupID[lid_j]=num_subdomains+num_separator_groups;
#ifdef HYMLS_DEBUGGING
              Tools::deb() << gid_j <<"/"<<lid_j<<" ";
#endif
            }// for j
          HYMLS_DEBUG("");
          num_separator_groups++;
          }// if not assigned
#ifdef HYMLS_DEBUGGING
        else
          {
          HYMLS_DEBVAR(groupID[lid]);
          }
#endif
        }// if belongs to someone else
      }// for grp
    }//for sd

  HYMLS_DEBVAR(num_separator_groups);
  
  HYMLS_DEBVAR(groupSize);
  HYMLS_DEBVAR(groupID);

#ifdef HYMLS_TESTING
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
    std::cout << numOverlappingElements << " " << (*newGroupPointer)[sd][grp]+pos << std::endl;
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
  HYMLS_DEBVAR(*newOverlappingMap);   
  
  newObject = Teuchos::rcp(new HierarchicalMap
    (comm_,newMap,newOverlappingMap,newGroupPointer,gidList_,"Separator Nodes",myLevel_) );
  
  return newObject;
  }

////////////////////////////////////////////////////////////////////////////////

Teuchos::RCP<const HierarchicalMap> 
HierarchicalMap::SpawnLocalSeparators
        (Teuchos::RCP<Teuchos::Array<HYMLS::SepNode> > regroup) const
  { 
  HYMLS_PROF3(label_,"SpawnLocalSeparators");

  Teuchos::RCP<const HierarchicalMap> newObject=Teuchos::null;
  Teuchos::RCP<Epetra_Map> newMap=Teuchos::null;
  Teuchos::RCP<Epetra_Map> newOverlappingMap=Teuchos::null;
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGidList =
        Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > newGroupPointer =
        Teuchos::rcp(new Teuchos::Array<Teuchos::Array<int> >());
  
  int base = baseMap_->IndexBase();  
  
  // start out from the standard Separator object. It's interior groups are the new 
  // subdomains, and we split them according to the criteria given by the user in the
  // SepNode array.
  Teuchos::RCP<const HierarchicalMap> sepObject 
        = this->Spawn(Separators);

  // TODO: Remove this array
  Teuchos::Array<int> todo;

  for (int sd = 0; sd < sepObject->NumMySubdomains(); sd++)
    {
    newGidList->append(Teuchos::Array<int>());
    newGroupPointer->append(Teuchos::Array<int>(2));
    for (int grp = 1; grp < sepObject->NumGroups(sd); grp++)
      {
      if (sepObject->NumElements(sd, grp) > 0 && baseMap_->MyGID(sepObject->GID(sd, grp, 0)) &&
        std::find(todo.begin(), todo.end(), sepObject->GID(sd, grp, 0)) == todo.end())
        {
        Teuchos::Array<int> gidList = sepObject->GetGroup(sd, grp);
        int offset = *((*newGroupPointer)[sd].end()-1);
        int len = gidList.size();
        if (len>0)
          {
          (*newGroupPointer)[sd].append(offset + len);
          std::copy(gidList.begin(), gidList.end(), std::back_inserter((*newGidList)[sd]));
          }
        todo.append(sepObject->GID(sd, grp, 0));
        }
      }
    }
  Teuchos::Array<int> all_gids;
  for (int sd=0;sd<NumMySubdomains();sd++)
    {
    std::copy((*newGidList)[sd].begin(), (*newGidList)[sd].end(),std::back_inserter(all_gids));
    }

  // std::sort(all_gids.begin(), all_gids.end());
  // Teuchos::Array<int>::iterator end_interior = std::unique(all_gids.begin(),all_gids.end());
  
  // int numel = std::distance(all_gids.begin(), end_interior);
  int numel = all_gids.size();
  int *my_gids = numel>0? &(all_gids[0]) : NULL;
  
  newMap = Teuchos::rcp(new Epetra_Map
    (-1,numel,my_gids,baseMap_->IndexBase(),*comm_));

  std::cout << "LOCALSEPS " << *newMap;
  
  newObject = Teuchos::rcp(new HierarchicalMap
      (comm_,newMap,newMap,newGroupPointer,newGidList,"Local Separator Nodes",myLevel_) );
  return newObject;

  // Teuchos::RCP<const HierarchicalMap> sepObject 
  //       = this->Spawn(Separators);

  
  // the number of elements in the new object is the number of interior
  // elements in the old one
  // int NumMyElements = sepObject->NumMyInteriorElements();
  // int* MyElements=new int[NumMyElements];
  // newGroupPointer->resize(sepObject->NumMySubdomains());

  // int pos=0;
  // for (int sep=0;sep<sepObject->NumMySubdomains();sep++)
  //   {
  //   HYMLS_DEBVAR(sep)
  //   (*newGroupPointer)[sep].append(pos);
  //   if (regroup!=Teuchos::null)
  //     {
  //     // sort the SepNodes
  //     int begin = sepObject->LID(sep,0,0);
  //     int end = begin + sepObject->NumElements(sep,0);
  //     std::sort(regroup->begin()+begin,regroup->begin()+end);
  //     MyElements[pos++] = (*regroup)[begin].GID();
  //     HYMLS_DEBUG("|"<<pos-1<<"| "<<(*regroup)[begin]);
  //     for (int j=1;j<sepObject->NumElements(sep,0);j++)
  //       {
  //       MyElements[pos] = (*regroup)[begin+j].GID();
  //       HYMLS_DEBUG("|"<<pos<<"| "<<(*regroup)[begin+j]);
  //       // note: comparing SepNode objects means comparing there
  //       // connectivity and variable type, not their GIDs. So 
  //       // this call does the grouping (together with the sort)
  //       if ((*regroup)[begin+j]!=(*regroup)[begin+j-1])
  //         {
  //         HYMLS_DEBUG("this is a new group");
  //         (*newGroupPointer)[sep].append(pos);
  //         }
  //       pos++;
  //       }
  //     }
  //   else
  //     {
  //     for (int j=0;j<sepObject->NumElements(sep,0);j++)
  //       {
  //       MyElements[pos++] = sepObject->GID(sep,0,j);
  //       }
  //     }
  //   (*newGroupPointer)[sep].append(pos);
  //   }

  // //for this object the map and overlapping map are the same
  // newMap=Teuchos::rcp(new Epetra_Map(-1,NumMyElements,MyElements,base,*comm_));

  // newOverlappingMap=newMap;
  
  // delete [] MyElements;

  // newObject = Teuchos::rcp(new HierarchicalMap
  //       (comm_,newMap,newOverlappingMap,newGroupPointer,"Local Separator Nodes",myLevel_) );
  
//  std::cout << *sepObject << std::endl;
//  std::cout << *newObject << std::endl;
  newMap = Teuchos::rcp(new Epetra_Map(sepObject->Map()));

  newObject = Teuchos::rcp(new HierarchicalMap
    (comm_,newMap,newMap,groupPointer_,gidList_,"Local Separator Nodes",myLevel_) );
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
