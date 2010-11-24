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
  virtual ~BasePartitioner(){DEBUG("BasePartitioner::~BasePartitioner()");}
  
  //! partition an [nx x ny x nz] grid with one DoF per node
  //! into npx*npy*npz global subdomains.
  virtual void Partition(int nparts) = 0;
  
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
  
  //! get non-overlapping subdomain id 
  virtual int operator()(int gid) const = 0;
  
  //! get the (re-)partitioned map (here elements belonging
  //! to a subdomain have contiguous local indexing)
  virtual const Epetra_Map& Map() const = 0;
  
  //! get number of elements in local subdomain <part>
  virtual int NumElements(int part) const = 0;
  
  //! get the local index of the first element in subdomain i.
  //! For i=NumLocalParts()+1, Map().NumLocalElements() is returned.
  virtual int First(int i) const = 0;

  //! get global index of element j in local subdomain i
  virtual int GID(int i, int j) const = 0;
      
  };

}
#endif
