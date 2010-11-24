#include "HYMLS_CartesianPartitioner.H"
#include "HYMLS_Tools.H"
#include "Teuchos_Utils.hpp"
#include "Epetra_Comm.h"
#include "Epetra_Map.h"

using Teuchos::toString;

namespace HYMLS {

  
  // constructor 
  CartesianPartitioner::CartesianPartitioner
        (Teuchos::RCP<const Epetra_Map> map, int nx, int ny, int nz, int dof)
        : BasePartitioner(), baseMap_(map), nx_(nx), ny_(ny),nz_(nz),dof_(dof)
        {
        
        comm_=Teuchos::rcp(&(baseMap_->Comm()),false);
        cartesianMap_=Teuchos::null;
        
        if (baseMap_->IndexBase()!=0)
          {
          Tools::Error("Not sure, but I _think_ your map should be 0-based",__FILE__,__LINE__);
          }
/*! this test will fail if the partitioner is used for a coarser level where i.e.
    most of the pressures have been eliminated, so that a lot more velocities are
    left than pressures.
    
        if (baseMap_->NumGlobalElements()!=nx*ny*nz*dof_)
          {
          Tools::Error("Base map incompatible with grid-size!",__FILE__,__LINE__);
          }
*/          
        DEBVAR(nx_);
        DEBVAR(ny_);
        DEBVAR(nz_);
        DEBVAR(dof_);
        
        npx_=-1;// indicates that Partition() hasn't been called
        
        // initially we put one subdomain in every partition
        int nparts=comm_->NumProc();
        }
        
        
  // destructor
  CartesianPartitioner::~CartesianPartitioner()
    {
    DEBUG("CartesianPartitioner::~CartesianPartitioner()");
    }


  void CartesianPartitioner::Partition(int nparts)
    {
    DEBUG("Cartesian Partitioner");
    int npx,npy,npz;
    Tools::SplitBox(nx_,ny_,nz_,nparts,npx,npy,npz);
    this->Partition(npx,npy,npz);
    }
  
  // partition an [nx x ny x nz] grid with one DoF per node
  // into nparts global subdomains.
  void CartesianPartitioner::Partition(int npx_in,int npy_in, int npz_in)
    {
    npx_=npx_in;
    npy_=npy_in;
    npz_=npz_in;
    
    DEBUG("Cartesian Partitioner: ");
    DEBVAR(npx_);
    DEBVAR(npy_);
    DEBVAR(npz_);
    
    sx_=nx_/npx_;
    sy_=ny_/npy_;
    sz_=nz_/npz_;
    
    if ((nx_!=npx_*sx_)||(ny_!=npy_*sy_)||(nz_!=npz_*sz_))
      {
      Tools::Error("We currently need nx to be a multiple of npx etc.",
                __FILE__,__LINE__);
      }

    Tools::Out("Partition domain: ");
    Tools::Out("Grid size: "+toString(nx_)+"x"+toString(ny_)+"x"+toString(nz_));
    Tools::Out("Number of Subdomains: "+toString(npx_)+"x"+toString(npy_)+"x"+toString(npz_));
    Tools::Out("Subdomain size: "+toString(sx_)+"x"+toString(sy_)+"x"+toString(sz_));
    
    // redistribute map so that variables in a subdomain belong to one processor:
    int nprocs=comm_->NumProc();
 
     int nprocx,nprocy,nprocz; // these are for the phyisical partitioning
    Tools::SplitBox(npx_,npy_,npz_,nprocs,nprocx,nprocy,nprocz);

    if (
        ((int)(nx_/nprocx)*nprocx!=nx_)||
        ((int)(ny_/nprocy)*nprocy!=ny_)||
        ((int)(nz_/nprocz)*nprocz!=nz_) )
      {
      Tools::Error("We currently need nx to be a multiple of nprocx etc.",
                __FILE__,__LINE__);
      }
    
    int rank=comm_->MyPID();
    int rankI,rankJ,rankK;
    Tools::ind2sub(nprocx,nprocy,nprocz,rank,rankI,rankJ,rankK);
    
    int ioff=rankI*npx_/nprocx*sx_;
    int joff=rankJ*npy_/nprocy*sy_;
    int koff=rankK*npz_/nprocz*sz_;
   
    numLocalSubdomains_=(npx_/nprocx)*(npy_/nprocy)*(npz_/nprocz);
   
    // create redistributed map:
    
    // count number of local elements
    int NumMyElements=0;

    for (int kk=0;kk<npz_/nprocz;kk++)
      for (int jj=0;jj<npy_/nprocy;jj++)
        for (int ii=0;ii<npx_/nprocx;ii++)
          {
          for (int k=0;k<sz_;k++)
            for (int j=0;j<sy_;j++)
              for (int i=0;i<sx_;i++)
                {
                for (int v=0;v<dof_;v++)
                  {
                  int gid=Tools::sub2ind(nx_,ny_,nz_,dof_,
                       ioff+ii*sx_+i,
                       joff+jj*sy_+j,
                       koff+kk*sz_+k,
                       v);
                  // the gid is in principle on this partition,
                  // but we also have to check if it was in the
                  // original map:
                  int pid,lid;
                  baseMap_->RemoteIDList(1,&gid,&pid,&lid);
                  if (pid!=-1)
                    {
                    NumMyElements++;
                    }
                  }
                }
          }
    
    int *MyGlobalElements = new int[NumMyElements];
    int *NumElementsInSubdomain = new int[NumLocalParts()];
    
    int pos=0;
    int sd=0;
    
    for (int kk=0;kk<npz_/nprocz;kk++)
      for (int jj=0;jj<npy_/nprocy;jj++)
        for (int ii=0;ii<npx_/nprocx;ii++)
          {
          NumElementsInSubdomain[sd]=0;
          for (int k=0;k<sz_;k++)
            for (int j=0;j<sy_;j++)
              for (int i=0;i<sx_;i++)
                {
                for (int v=0;v<dof_;v++)
                  {
                  int gid = Tools::sub2ind(nx_,ny_,nz_,dof_,
                       ioff+ii*sx_+i,
                       joff+jj*sy_+j,
                       koff+kk*sz_+k,
                       v);
                  int pid,lid;
                  baseMap_->RemoteIDList(1,&gid,&pid,&lid);
                  if (pid!=-1)
                    {
                    MyGlobalElements[pos++] =gid;
                    NumElementsInSubdomain[sd]++;
                    }
                }
              }
          sd++;
          }
      
    cartesianMap_=Teuchos::rcp(new Epetra_Map(-1,NumMyElements,MyGlobalElements,0,*comm_));
    delete [] MyGlobalElements;
    
    subdomainPointer_.resize(NumLocalParts()+1);
    subdomainPointer_[0]=0;
    for (int i=0;i<NumLocalParts();i++)
      {
      subdomainPointer_[i+1]=subdomainPointer_[i]+NumElementsInSubdomain[i];
      }
    
    delete [] NumElementsInSubdomain;
    
    Tools::Out("Number of Partitions: "+toString(nprocx)+"x"+toString(nprocy)+"x"+toString(nprocz));
    Tools::Out("Partition: ["+toString(rankI)+" "+toString(rankJ)+" "+toString(rankK)+"]");
    Tools::Out("Number of Local Subdomains: "+toString(NumLocalParts()));
        
    }


  
}
