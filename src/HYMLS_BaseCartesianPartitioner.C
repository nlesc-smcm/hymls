#include "HYMLS_BaseCartesianPartitioner.H"
#include "HYMLS_Tools.H"
#include "Teuchos_Utils.hpp"
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "HYMLS_MatrixUtils.H"

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#include "Epetra_MpiDistributor.h"
#endif


using Teuchos::toString;

#ifdef DEBUGGING
#define FLOW_DEBUGGING
#endif

#ifdef FLOW_DEBUGGING
#define FLOW_DEBUG(s) DEBUG(s)
#else
#define FLOW_DEBUG(s)
#endif

namespace HYMLS {

  
  // constructor 
  BaseCartesianPartitioner::BaseCartesianPartitioner
        (Teuchos::RCP<const Epetra_Map> map, int nx, int ny, int nz, int dof, 
        GaleriExt::PERIO_Flag perio)
        : BasePartitioner(), label_("BaseCartesianPartitioner"),
                baseMap_(map), nx_(nx), ny_(ny),nz_(nz),dof_(dof),perio_(perio),
                numLocalSubdomains_(-1),numGlobalSubdomains_(-1)
        {
        START_TIMER3(label_,"Constructor");        
        node_distance_=1; //default, can be adjusted by calling SetNodeDistance()    
        active_=true; // by default, everyone is assumed to own a part of the domain.
                      // if in Partition() it turns out that there are more processor
                      // partitions than subdomains, active_ is set to false for some
                      // ranks, which will get an empty part of the reordered map    
                      // (cartesianMap_).
        
        comm_=Teuchos::rcp(&(baseMap_->Comm()),false);
        cartesianMap_=Teuchos::null;
        
        if (baseMap_->IndexBase()!=0)
          {
          Tools::Warning("Not sure, but I _think_ your map should be 0-based",
                __FILE__, __LINE__);
          }

        DEBVAR(nx_);
        DEBVAR(ny_);
        DEBVAR(nz_);
        DEBVAR(dof_);

        graph_=Teuchos::null;
        pvar_=-1;
        
        npx_=-1;// indicates that Partition() hasn't been called
        }
        
        
  // destructor
  BaseCartesianPartitioner::~BaseCartesianPartitioner()
    {
    START_TIMER3(label_,"Destructor");
    }

  int BaseCartesianPartitioner::flow(int gid1, int gid2) const
    {
    int sd1 = (*this)(gid1);
    int sd2 = (*this)(gid2);
    
    FLOW_DEBUG("### FLOW("<<gid1<<", "<<gid2<<") ###");
    
    if (sd1==sd2)
      {
      FLOW_DEBUG("# same subdomain, return 0");
      return 0;
      }

    int i1,j1,k1,i2,j2,k2;
    int var1,var2;
      
    Tools::ind2sub(nx_,ny_,nz_,dof_,gid1,i1,j1,k1,var1);
    Tools::ind2sub(nx_,ny_,nz_,dof_,gid2,i2,j2,k2,var2);
/*
    if (var2==pvar_)
      {
      FLOW_DEBUG("# coupling to pressure, return 0");
      return 0;
      }
  */
/*
    if (var1!=var2)
      {
      FLOW_DEBUG("# different variables, return 0");
      return 0;
      }
*/
    bool adjacent=false;
      int di,dj,dk;

      di=calc_distance(nx_,i1,i2,(perio_&GaleriExt::X_PERIO));
      dj=calc_distance(ny_,j1,j2,(perio_&GaleriExt::Y_PERIO));
      dk=calc_distance(nz_,k1,k2,(perio_&GaleriExt::Z_PERIO));

    if (graph_==Teuchos::null) // old style - look at 'node distance' and guess...
      {
#ifdef FLOW_DEBUGGING
DEBVAR(di);
DEBVAR(dj);
DEBVAR(dk);
#endif
    // if the cells are not connected:
      if ((std::abs(di)>stencil_width_) || 
        (std::abs(dj)>stencil_width_) || 
        (std::abs(dk)> stencil_width_))
        {
        FLOW_DEBUG("# not adjacent grid cells, return 0 [based on guesswork!]");
        adjacent=false;
        }
      }
    else
      {
      int len;
      int *cols;
      int lrid1 = graph_->LRID(gid1);
      CHECK_ZERO(graph_->ExtractMyRowView(lrid1,len,cols));
      for (int j=0;j<len;j++)
        {
        if (graph_->GCID(cols[j])==gid2)
          {
          adjacent=true;
          break;
          }
        }
      FLOW_DEBUG("# not adjacent grid cells, return 0 [based on graph]");
      }
    if (!adjacent) return 0;
    
    // the cells are connected and in different subdomains, so we have to
    // return a nonzero value.

    // for non-periodic problems it is fairly simple:
    if (perio_==GaleriExt::NO_PERIO)
      {
      FLOW_DEBUG("# return "<<sd1-sd2);
      return sd1-sd2;
      }
      
    // problem is periodic in at least one direction

    int value;
    int I1,J1,K1,I2,J2,K2;
    int dI, dJ, dK;
    int dum;

    Tools::ind2sub(npx_,npy_,npz_,1,sd1,I1,J1,K1,dum);
    Tools::ind2sub(npx_,npy_,npz_,1,sd2,I2,J2,K2,dum);

    dI=calc_distance(npx_,I1,I2,(perio_&GaleriExt::X_PERIO));
    dJ=calc_distance(npy_,J1,J2,(perio_&GaleriExt::Y_PERIO));
    dK=calc_distance(npz_,K1,K2,(perio_&GaleriExt::Z_PERIO));

    if (std::abs(dK)> 0)
      {
#ifdef FLOW_DEBUGGING
      if (dk<0)
        {
        DEBUG("# top edge, return "<<dk);
        }
      else
        {
        DEBUG("# bottom edge, return "<<dk);
        }
#endif      
      return dk;
      }
    if (std::abs(dJ)> 0)
      {
#ifdef FLOW_DEBUGGING
      if (dj<0)
        {
        DEBUG("# north edge, return "<<dj);
        }
      else
        {
        DEBUG("# south edge, return "<<dj);
        }
#endif      
      return dj;
      }
    if (std::abs(dI)> 0)
      {
#ifdef FLOW_DEBUGGING
      if (di<0)
        {
        DEBUG("# east edge, return "<<di);
        }
      else
        {
        DEBUG("# west edge, return "<<di);
        }
#endif      
      return di;
      }
    FLOW_DEBUG("weird case, return 0");
    return 0;
    }

  // private
  int BaseCartesianPartitioner::calc_distance(int n, int i1,int i2,bool perio) const
    {
    int di=i1-i2;
    if (perio)
      {
      if (i1<i2)
        {
        i1+=n;
        }
      else
        {
        i2+=n;
        }
      if (std::abs(i1-i2)<std::abs(di))
        {
        di=i1-i2;
        }
      }
    return ceil((double)di/node_distance_);
    }

  int BaseCartesianPartitioner::Partition(int nparts, bool repart)
    {
    START_TIMER3(label_,"Partition");
    int npx,npy,npz;
    Tools::SplitBox(nx_,ny_,nz_,nparts,npx,npy,npz);
    return this->Partition(npx,npy,npz,repart);
    }
    

  // partition an [nx x ny x nz] grid with one DoF per node
  // into nparts global subdomains.
  int BaseCartesianPartitioner::Partition
        (int npx_in,int npy_in, int npz_in, bool repart)
    {
    START_TIMER3(label_,"Partition (2)");
    npx_=npx_in;
    npy_=npy_in;
    npz_=npz_in;
        
    DEBVAR(npx_);
    DEBVAR(npy_);
    DEBVAR(npz_);
    
    sx_=nx_/npx_;
    sy_=ny_/npy_;
    sz_=nz_/npz_;

    std::string s1=toString(nx_)+"x"+toString(ny_)+"x"+toString(nz_);
    std::string s2=toString(npx_)+"x"+toString(npy_)+"x"+toString(npz_);
    
    if ((nx_!=npx_*sx_)||(ny_!=npy_*sy_)||(nz_!=npz_*sz_))
      {
      std::string msg = "You are trying to partition an "+s1+" domain into "+s2+" parts.\n"
                        "We currently need nx to be a multiple of npx etc.";
      Tools::Error(msg,__FILE__,__LINE__);
      }

    std::string s3=toString(sx_)+"x"+toString(sy_)+"x"+toString(sz_);

    Tools::Out("Partition domain: ");
    Tools::Out("Grid size: "+s1);
    Tools::Out("Number of Subdomains: "+s2);
    Tools::Out("Subdomain size: "+s3);
    
    // case where there are more processor partitions than subdomains (experimental)
    if (comm_->MyPID()>=npx_*npy_*npz_)
      {
      active_ = false;
      }
     

    int color = active_? 1: 0; 
    int nprocs;

    CHECK_ZERO(comm_->SumAll(&color,&nprocs,1));

    // if some processors have no subdomains, we need to 
    // repartition the map even if it is a cartesian partitioned
    // map already:
    if (nprocs<comm_->NumProc()) repart=true; 
 
    Tools::SplitBox(npx_,npy_,npz_,nprocs,nprocx_,nprocy_,nprocz_);

    std::string s4 =
    Teuchos::toString(nprocx_)+"x"+Teuchos::toString(nprocy_)+"x"+Teuchos::toString(nprocz_);

    if (
        ((int)(nx_/nprocx_)*nprocx_!=nx_)||
        ((int)(ny_/nprocy_)*nprocy_!=ny_)||
        ((int)(nz_/nprocz_)*nprocz_!=nz_) )
      {
      std::string msg="You are trying to partition an "+s1+" domain on "+s4+" procs.\n"
        "We currently need nx to be a multiple of nprocx etc.";
      HYMLS::Tools::Error(msg,__FILE__,__LINE__);
      }
        
    int rank=comm_->MyPID();
    int rankI=-1,rankJ=-1,rankK=-1;
    int ioff=-1,joff=-1,koff=-1;

    if (active_)
      {
      Tools::ind2sub(nprocx_,nprocy_,nprocz_,rank,rankI,rankJ,rankK);

      ioff=rankI*npx_/nprocx_*sx_;
      joff=rankJ*npy_/nprocy_*sy_;
      koff=rankK*npz_/nprocz_*sz_;
   
      numLocalSubdomains_=(npx_/nprocx_)*(npy_/nprocy_)*(npz_/nprocz_);
      }
    else
      {
      numLocalSubdomains_=0;
      }
        
    DEBVAR(npx_);
    DEBVAR(npy_);
    DEBVAR(npz_);
    DEBVAR(active_);
    DEBVAR(rank);
    DEBVAR(rankI);
    DEBVAR(rankJ);
    DEBVAR(rankK);        

    sdMap_=CreateSubdomainMap();
    DEBVAR(*sdMap_);
    
    numLocalSubdomains_ = sdMap_->NumMyElements();
    numGlobalSubdomains_ = sdMap_->NumGlobalElements();
    
// create redistributed map:
Teuchos::RCP<Epetra_Map> repartitionedMap = 
        Teuchos::rcp_const_cast<Epetra_Map>(baseMap_);

if (repart==true)
  {
  START_TIMER3(label_,"repartition map");
#ifdef HAVE_MPI
  Teuchos::RCP<const Epetra_MpiComm> mpiComm =
        Teuchos::rcp_dynamic_cast<const Epetra_MpiComm>(comm_);
  if (mpiComm==Teuchos::null) Tools::Error("need an MpiComm here",__FILE__,__LINE__);
  Teuchos::RCP<Epetra_MpiDistributor> Distor =
        Teuchos::rcp(new Epetra_MpiDistributor(*mpiComm));
  // check how many of the owned GIDs in the map need to be
  // moved to someone else:
  int numSends = 0;
  for (int i=0;i<baseMap_->NumMyElements();i++)
    {
    int gid = baseMap_->GID(i);
    if (LSID(gid)<0) numSends++;
    }

  int numLocal = baseMap_->NumMyElements() - numSends;
  
  DEBVAR(numSends);
  DEBVAR(numLocal);

  //determine which GIDs we have to move, and where they will go
  int *sendGIDs = new int[numSends];
  int *sendPIDs = new int[numSends];

  int pos=0;

  for (int i=0;i<baseMap_->NumMyElements();i++)
    {
#ifdef TESTING
if (pos>numSends) Tools::Error("sanity check failed",__FILE__,__LINE__);
#endif    
    int gid = baseMap_->GID(i);
    int pid = this->PID(gid); // global partition ID
    if (pid!=comm_->MyPID())
      {
      sendGIDs[pos] = gid;
      sendPIDs[pos++] = pid;
      }
    }
    
  int numRecvs;
  CHECK_ZERO(Distor->CreateFromSends(numSends, sendPIDs, true, numRecvs));

  DEBVAR(numRecvs);
  
  char* sbuf = reinterpret_cast<char*>(sendGIDs);
  int numRecvChars=numRecvs*sizeof(int);
  char* rbuf = new char[numRecvChars];
  
  CHECK_ZERO(Distor->Do( sbuf,
          sizeof(int),
          numRecvChars,
          rbuf));

  int *recvGIDs = reinterpret_cast<int*>(rbuf);

#ifdef TESTING
  if (numRecvs*sizeof(int)!=numRecvChars)
    {
    Tools::Error("sanity check failed",__FILE__,__LINE__);
    }   
#endif  

  int NumMyElements = numLocal + numRecvs;
  DEBVAR(NumMyElements);
  int *MyGlobalElements = new int[NumMyElements];
  pos=0;
  for (int i=0;i<baseMap_->NumMyElements();i++)
    {
    int gid = baseMap_->GID(i);
    if (this->LSID(gid)>=0) MyGlobalElements[pos++]=gid;
    }
  for (int i=0;i<numRecvs;i++) MyGlobalElements[pos+i]=recvGIDs[i];
        
  std::sort(MyGlobalElements, MyGlobalElements+NumMyElements);

  repartitionedMap = Teuchos::rcp(new Epetra_Map
        (-1,NumMyElements,MyGlobalElements, baseMap_->IndexBase(), *comm_));

  DEBVAR(*repartitionedMap);

DEBUG(std::flush);

  delete [] sendPIDs;
  delete [] sendGIDs;
  delete [] rbuf;
  delete [] MyGlobalElements;  

#endif
  }

  // note: NumLocalParts() is simply numLocalSubdomains_.
  int *NumElementsInSubdomain = new int[NumLocalParts()];
  for (int sd=0;sd<NumLocalParts();sd++) NumElementsInSubdomain[sd]=0;

  subdomainPointer_.resize(NumLocalParts()+1);

  // now we have a cartesian processor partitioning and no nodes have to 
  // be moved between partitions. Some partitions may be empty, though.
  for (int lid=0;lid<repartitionedMap->NumMyElements();lid++)
    {
    int gid = repartitionedMap->GID(lid);
    int lsd=LSID(gid);
    
#ifdef TESTING
    if (lsd<0)
      {
      Tools::Error("repartitioning seems to be necessary/have failed.",
        __FILE__,__LINE__);
      }
#endif    
    NumElementsInSubdomain[lsd]++;
    }

  subdomainPointer_[0]=0;
  for (int i=0;i<NumLocalParts();i++)
    {
    subdomainPointer_[i+1]=subdomainPointer_[i]+NumElementsInSubdomain[i];
    }

  int NumMyElements=repartitionedMap->NumMyElements();
#ifdef TESTING
  if (subdomainPointer_[NumLocalParts()]!=NumMyElements)
    {
    Tools::Error("repartitioning - sanity check failed",__FILE__,__LINE__);
    }
#endif  
  int *MyGlobalElements = new int[NumMyElements];
  for (int i=0;i<NumLocalParts();i++) NumElementsInSubdomain[i]=0;

  for (int lid=0;lid<repartitionedMap->NumMyElements();lid++)
    {
    int gid = repartitionedMap->GID(lid);
    int lsd=LSID(gid);
    MyGlobalElements[subdomainPointer_[lsd] + NumElementsInSubdomain[lsd]] = gid;
    NumElementsInSubdomain[lsd]++;
    }
  // sort nodes per subdomain
  for (int i=0;i<NumLocalParts();i++)
    {
    std::sort(MyGlobalElements + subdomainPointer_[i],
              MyGlobalElements + subdomainPointer_[i+1]);
    }

DEBVAR(NumMyElements);

cartesianMap_=Teuchos::rcp(new Epetra_Map(-1,NumMyElements,MyGlobalElements,0,*comm_));
DEBVAR(*cartesianMap_);
    
    if (active_)
      {
      Tools::Out("Number of Partitions: "+s4);
      Tools::Out("Partition: ["+toString(rankI)+" "+toString(rankJ)+" "+toString(rankK)+"]");
      Tools::Out("Number of Local Subdomains: "+toString(NumLocalParts()));

      delete [] MyGlobalElements;
      delete [] NumElementsInSubdomain;    
      }
    return 0;    
    }



}
