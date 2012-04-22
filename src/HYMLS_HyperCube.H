#ifndef HYMLS_HYPERCUBE_H
#define HYMLS_HYPERCUBE_H

#include <iostream>
#include "Teuchos_RCP.hpp"

class Epetra_MpiComm;

namespace HYMLS {

//! this is a very simple class that    
//! allows renumbering the processors   
//! in a run in a node-aware way.       
//! If the machine you are working on   
//! looks like this:                    
//!     oo  oo                          
//!     oo--oo                          
//!      |  |   , with o indicating     
//!     oo--oo    the cores of a node,  
//!     oo  oo                          
//!                                     
//! then this class creates an MpiComm  
//! that has the following numbering of 
//! the ranks:                          
//!                                     
//!  0, 4   1, 5                        
//!  8,12-- 9,13                        
//!     |  |                            
//!  2, 6-- 3, 7                        
//! 10,14  11,15                        
//!                                     
//! this makes sure that, when reducing 
//! the number of active procs, the     
//! number of nodes (and thus available 
//! memory) stays as large as possible. 
//!                                     
class HyperCube
{

public: 

//!
HyperCube();

//!
~HyperCube();

//!
const Epetra_MpiComm& Comm() const {return *reorderedComm_;}

//!
Epetra_MpiComm& Comm() {return *reorderedComm_;}

//!
Teuchos::RCP<const Epetra_MpiComm> GetComm() const {return reorderedComm_;}

//!
Teuchos::RCP<Epetra_MpiComm> GetComm() {return reorderedComm_;}

//!
std::ostream& Print(std::ostream& os) const;

protected:

//!
int numNodes_;
//!
int nodeNumber_;
//!
int numProcOnNode_;
//!
int rankOnNode_;
//!
int maxProcPerNode_;
//!
Teuchos::RCP<Epetra_MpiComm> commWorld_;
//!
Teuchos::RCP<Epetra_MpiComm> reorderedComm_;
};

}//namespace

std::ostream& operator<<(std::ostream& os,const HYMLS::HyperCube& C);

#endif
