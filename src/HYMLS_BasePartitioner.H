#ifndef HYMLS_BASE_PARTITONER_H
#define HYMLS_BASE_PARTITONER_H

#include "Teuchos_RCP.hpp"
#include "HYMLS_Tools.H"

class Epetra_Comm;
class Epetra_Map;

namespace HYMLS {

/*! Base class for partitioning in HYMLS - on this 
    partitioning we build our HID.

  The idea is that to iterate over all GID's in subdomain ordering,     
      you could use something like this:                                
                                                        
        for (int p=0;p<NumLocalParts();p++)             
          for (int i=First(p);i<First(p+1);i++)         
            {                                           
            int row=Map().GID(i);                       
            ...                                         
            }                                           
                                                        
  or equivalently:                                      
                                                        
        for (int p=0;p<NumLocalParts();p++)             
          for (int i=0;i<NumElements(p);i++)            
            {                                           
            int row=GID(p,i);                           
            ...                                         
            }                                           
                                                        
  */

class BasePartitioner
  {
  public:

  //! constructor
  BasePartitioner(){;}
  
  //! destructor
  virtual ~BasePartitioner(){}
  
  //! partition an [nx x ny x nz] grid with one DoF per node
  //! into npx*npy*npz global subdomains. If repart==true,  
  //! the partitioner is allowed to change the global       
  //! distribution of the nodes, otherwise it may only      
  //! partition the owned nodes on each processor.          
  virtual int Partition(int nparts, bool repart) = 0;
  
  //! get number of local partitions
  virtual int NumLocalParts() = 0;

  //! get global number of partitions
  virtual int NumGlobalParts() = 0;
  
  //! is this class fully set up?
  virtual bool Partitioned() const = 0;
  
  //! return the number of variables per grid point
  //! (default implementation returns 1)
  virtual int DofPerNode() const {return 1;}
  
  //! for problems with multiple dof per node: get an
  //! integer indicating which variable type a gid has
  //! (this is used in HID to group separators by type)
  //! default implementation returns 0
  virtual int VariableType(int gid) const {return 0;}
  
  //! get non-overlapping global subdomain id 
  virtual int operator()(int gid) const = 0;
  
  //! given a local subdomain index, find the global one.
  virtual int GPID(int sd) const
    {
    int gpid = -1;
    if (NumElements(sd)>0) gpid = (*this)(GID(sd,0));
#ifdef TESTING
    if (gpid==-1) Tools::Error
        ("partitioner function not implemented for the case of empty subdomains!",
                __FILE__,__LINE__);
#endif    
    return gpid;        
    }
  
  //! get the (re-)partitioned map (here elements belonging
  //! to a subdomain have contiguous local indexing)
  virtual const Epetra_Map& Map() const = 0;
  
  //! get the (re-)partitioned map (here elements belonging
  //! to a subdomain have contiguous local indexing)
  virtual Teuchos::RCP<const Epetra_Map> GetMap() const 
    {return Teuchos::rcp(&Map(),false);}
  
  //! get number of elements in local subdomain <part>
  virtual int NumElements(int part) const = 0;
  
  //! get the local index of the first element in subdomain i.
  //! For i=NumLocalParts()+1, Map().NumLocalElements() is returned.
  virtual int First(int i) const = 0;

  //! get global index of element j in local subdomain i
  virtual int GID(int i, int j) const = 0;
  
  //! this function imposes an ordering on the subdomains  
  //! to make sure that periodic boundary conditions are   
  //! handled correctly by the solver. For non-periodic    
  //! problems it is OK to use the default implementation, 
  //! which is flow(gid1,gid2)=sd(gid1)-sd(gid2).          
  //!                                                      
  //! If gid1 and gid2 belong to different variable types  
  //! (e.g. u- and v-velocities, or a velocity and a pres- 
  //! sure, flow(gid1,gid2) should always return 0 to      
  // make sure our algorithms do the same for Navier-Stokes
  // as for Stokes.                                         
  //!                                                      
  //! For periodic problems it should be something like    
  //! this for the 2-domain case:                          
  //!                                                      
  //!   sd1  |     sd2  |       sd1 ...                    
  //!                                                      
  //! then flow(gid1,gid2)==0 if gid1 and gid2 are in      
  //!           same subdomain or not connected.           
  //!      flow(gid1,gid2)>0  if the edge between gid1     
  //!           and gid2 crosses a subdomain boundary in   
  //!           one direction (e.g. from left to right)    
  //!      flow(gid1,gid2)<0  if the edge between gid1     
  //!           and gid2 crosses a subdomain boundary in   
  //!           the other direction (from right to left)   
    
  //!                                                      
  //! this makes sure that the following separator nodes   
  //! (*) are selected:                                    
  //!   sd1 *|     sd2 *|       sd1 ...                    
  //!                                                      
  //! This is straight-forward to implement for cartesian  
  //! partitioning. For other partitioners, one might look 
  //! at the orientation of a gradient operator.           
  //!                                                      
  virtual int flow(int gid1, int gid2)
    {
    if (VariableType(gid1)==VariableType(gid2))
      {
      return operator()(gid1)-operator()(gid2);
      }
    return 0;
    };
      
  };

}
#endif
