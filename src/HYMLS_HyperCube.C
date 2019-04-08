#include "HYMLS_HyperCube.H"

#include <mpi.h>
#include "Epetra_MpiComm.h"

#include "Teuchos_Array.hpp"
#include "Teuchos_StrUtils.hpp"

#include "HYMLS_Macros.H"

namespace HYMLS {

HyperCube::HyperCube()
  {
  HYMLS_PROF3("HyperCube","HyperCube");
  commWorld_ = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  // figure out on which node we are:
  char* procname=new char[MPI_MAX_PROCESSOR_NAME];
  int procname_len=0; 
  MPI_Get_processor_name(procname,&procname_len);
  std::string proc(procname);
  delete [] procname;
  HYMLS_DEBVAR(proc);
  for (size_t i=0; i<proc.length();i++)
    {
    if (proc[i]<'0' || proc[i]>'9') proc[i]='0';
    }
  int node_id = Teuchos::StrUtils::atoi(proc);
  HYMLS_DEBVAR(node_id);
  Teuchos::Array<int> all_nodes(commWorld_->NumProc());
  CHECK_ZERO(commWorld_->GatherAll(&node_id,&all_nodes[0],1));
  std::sort(all_nodes.begin(),all_nodes.end());
  Teuchos::Array<int>::iterator new_end = std::unique(all_nodes.begin(),all_nodes.end());
  numNodes_ = std::distance(all_nodes.begin(),new_end);
  HYMLS_DEBVAR(numNodes_);
  nodeNumber_=0;
  while (all_nodes[nodeNumber_]!=node_id) nodeNumber_++;
  HYMLS_DEBVAR(nodeNumber_);

  Teuchos::Array<int> my_node(numNodes_);
  all_nodes.resize(numNodes_);
  for (int i=0;i<numNodes_;i++) 
    {
    my_node[i]=0;
    all_nodes[i]=0;
    }
  my_node[nodeNumber_]++;
  
  CHECK_ZERO(commWorld_->ScanSum(&my_node[0],&all_nodes[0],numNodes_));
  rankOnNode_=all_nodes[nodeNumber_]-1;
  HYMLS_DEBVAR(rankOnNode_);

  for (int i=0;i<numNodes_;i++) 
    {
    all_nodes[i]=0;
    }
  CHECK_ZERO(commWorld_->SumAll(&my_node[0],&all_nodes[0],numNodes_));  
  
  maxProcPerNode_=0;
  for (int i=0;i<numNodes_;i++) 
    {
    maxProcPerNode_=std::max(maxProcPerNode_,all_nodes[i]);
    }
  HYMLS_DEBVAR(maxProcPerNode_);
    
  // how many local rank 0, rank 1 etc are there on all these nodes?
  Teuchos::Array<int> my_proc_counts(maxProcPerNode_);
  Teuchos::Array<int> proc_counts(maxProcPerNode_);
  Teuchos::Array<int> scan_proc(maxProcPerNode_);

  my_proc_counts[rankOnNode_]=1;
  CHECK_ZERO(commWorld_->SumAll(&my_proc_counts[0],&proc_counts[0],maxProcPerNode_));  
  CHECK_ZERO(commWorld_->ScanSum(&my_proc_counts[0],&scan_proc[0],maxProcPerNode_));  
    
  int newRank=0;
  for (int i=0;i<rankOnNode_;i++) newRank+=proc_counts[i];
  newRank+=scan_proc[rankOnNode_];
  HYMLS_DEBVAR(newRank);

  // create a new reordered comm with all ranks still in it.
  int color=1;  

  MPI_Comm NewComm;
  MPI_Comm_split(commWorld_->Comm(),color,newRank,&NewComm);
  
  reorderedComm_=Teuchos::rcp(new Epetra_MpiComm(NewComm));

//#ifdef HYMLS_TESTING  
  for (int i=0;i<reorderedComm_->NumProc();i++)
    {
    if (reorderedComm_->MyPID()==i)
      {
      std::stringstream ss;
      ss << "PID "<<commWorld_->MyPID()<<" (on node "<<nodeNumber_<<") => PID "<<reorderedComm_->MyPID();
      std::cout << ss.str() << std::endl << std::flush;
      }
    reorderedComm_->Barrier();
    }
//#endif
  return;    
  }
  
HyperCube::~HyperCube()
  {
  }

std::ostream& HyperCube::Print(std::ostream& os) const
  {
  os << "processor topology"<<std::endl;
  os << "=================="<<std::endl;
  os << "# procs = "<<commWorld_->NumProc()<<std::endl;
  os << "# nodes = "<<numNodes_<<std::endl;
  os << "max ppn = "<<maxProcPerNode_<<std::endl;
  os << "adjacent ranks placed on \n";
  os << "different nodes if possible"<<std::endl;
  return os;
  }

}//namespace

std::ostream& operator<<(std::ostream& os, const HYMLS::HyperCube& C)
  {
  C.Print(os);
  return os;
  }
