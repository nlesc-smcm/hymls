#include <mpi.h>
#include <iostream>

#include "HYMLS_StandardNodeClassifier.H"
#include "HYMLS_Tools.H"
#include "HYMLS_BasePartitioner.H"

#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_CrsGraph.h"
#include "Epetra_Vector.h"
#include "Epetra_IntVector.h"
#include "Epetra_Import.h"
#include "Teuchos_StrUtils.hpp"

#include "HYMLS_MatrixUtils.H"

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#endif

#include "Teuchos_StandardCatchMacros.hpp"

/*
#ifndef TESTING
#define TESTING 1
#endif
*/
namespace HYMLS {

  //constructor
  
  // gets an overlapping graph of the matrix such that  
  // each processor can access the separators between the partitions, 
  // and a partitioning object
  StandardNodeClassifier::StandardNodeClassifier(
        Teuchos::RCP<const Epetra_CrsGraph> parG,
        Teuchos::RCP<const BasePartitioner> P, 
        const Teuchos::Array<std::string>& varType, 
        const Teuchos::Array<bool>& retIsol,
        int level, int nx, int ny, int nz,
        std::string label) :
                myLevel_(level),
                partitioner_(P),
                parallelGraph_(parG),
                nx_(nx),ny_(ny),nz_(nz),
                label_(label+" (level "+Teuchos::toString(level)+")")
    {
    START_TIMER3(Label(),"Constructor");
    dof_=partitioner_->DofPerNode();
    if (varType.length()!=dof_ || retIsol.length()!=dof_)
      {
      Tools::Error("invalid input",__FILE__,__LINE__);
      }
    variableType_.resize(dof_);
    retainIsolated_.resize(dof_);
    for (int i=0;i<dof_;i++)
      {
      variableType_[i]=varType[i];
      retainIsolated_[i]=retIsol[i];
      }
    nodeType_=Teuchos::rcp(new Epetra_IntVector(partitioner_->Map()));
    nodeType_->PutValue(-1);

    p_nodeType_=Teuchos::rcp(new Epetra_IntVector(parallelGraph_->RowMap()));
    return;
    }
    
  StandardNodeClassifier::~StandardNodeClassifier()
    {
    START_TIMER3(Label(),"Destructor");
    }



  
int StandardNodeClassifier::BuildNodeTypeVector()
  {
  START_TIMER2(Label(),"BuildNodeTypeVector");

  if (Teuchos::is_null(parallelGraph_))
    {
    Tools::Error("parallelGraph not yet constructed!",__FILE__,__LINE__);
    }
  
  CHECK_ZERO(this->BuildInitialNodeTypeVector(*parallelGraph_,*nodeType_));

  // now every subdomain has marked those nodes it owns and 
  // that should become interior, separator or retained nodes.

  // import partition overlap
  Epetra_Import import(p_nodeType_->Map(),nodeType_->Map());
  CHECK_ZERO(p_nodeType_->Import(*nodeType_,import,Insert));

#if defined(STORE_MATRICES)||defined(TESTING)
std::ofstream nodeTypeStream;
nodeTypeStream.open(("nodeTypes_L"+Teuchos::toString(myLevel_)+
        "_"+Teuchos::toString(nodeType_->Comm().MyPID())+".txt").c_str(),std::ios::trunc);
this->PrintNodeTypeVector(*p_nodeType_,nodeTypeStream,"initial");
#endif

  // increase the node type of nodes that only connect to separators.         
  // A type 1 sep node that only connects to sep nodes becomes a type         
  // 2 sep node. 
  CHECK_ZERO(this->UpdateNodeTypeVector(*parallelGraph_,*p_nodeType_, *nodeType_));
    
  // import to "spread the word"
  CHECK_ZERO(p_nodeType_->Import(*nodeType_,import,Insert));

#if defined(STORE_MATRICES)||defined(TESTING)
    this->PrintNodeTypeVector(*p_nodeType_,nodeTypeStream,"step 1");
#endif

  // An interior (type 0) node that only connects to sep          
  // nodes becomes a 'retained' (type 4) node if this was specified in        
  // the "Partitioner" -> "Variable X" sublist by "Retain Isolated".   
  // Also upgrade the separator nodes it connects to to 4. This resolves the  
  // following situation in the C-grid Stokes problem (full conservation      
  // cell):                                                                   
  //                                                                          
  // * in a cell where four straight separators meet, the pressure is         
  //   retained in the Schur-complement to make sure that the div-equation    
  //   doesn't lead to an empty row in the subdomain matrix:                  
  //                                                                          
  //           |   |                                                          
  //           |   |                                                          
  //           |   |                                                          
  // ----------+-v-+--------------                                            
  //           u p u                                                          
  // ----------+-v-+--------------                                            
  //           |   |                                                          
  //           |   |                                                          
  //           |   |                                                          
  //                                                                          
  // We increase the nodeType of all variables around such a cell by one. For 
  // the parallel 3D case we need to do this consistently, for instance the   
  // u-node at the right cell boundary has to be increaqsed by +1 for the FCC 
  // in the corner and by +1 by the adjacent cell, which is on a different    
  // processor, possibly. So we introduce an extra vector here.               
  CHECK_ZERO(this->DetectFCC(*parallelGraph_,*p_nodeType_, *nodeType_));

  CHECK_ZERO(p_nodeType_->Import(*nodeType_,import,Insert));

#if defined(STORE_MATRICES) || defined(TESTING)
  this->PrintNodeTypeVector(*p_nodeType_,nodeTypeStream,"with FCCs");
#endif

    CHECK_ZERO(this->UpdateNodeTypeVector(*parallelGraph_,*p_nodeType_, *nodeType_));

  // import again
  CHECK_ZERO(p_nodeType_->Import(*nodeType_,import,Insert));

#if defined(STORE_MATRICES) || defined(TESTING)
this->PrintNodeTypeVector(*p_nodeType_,nodeTypeStream,"final");
#endif

  
  return 0;
  }

  //! build initial vector with 0 (interior), 1 (separator) or 2 (retained)
  int StandardNodeClassifier::BuildInitialNodeTypeVector(
        const Epetra_CrsGraph& G, Epetra_IntVector& nodeType) const
  {
  START_TIMER3(Label(),"BuildInitialNodeTypeVector");


  int *cols;
  int len;
  Teuchos::Array<int> retain(dof_);
  Teuchos::Array<int> retained(dof_);

  for (int var=0;var<dof_;var++)
    {
    Teuchos::Array<string> vartype=Teuchos::StrUtils::stringTokenizer(variableType_[var]);
    retain[var]=0;
    if ((vartype.size()==2)&&(vartype[0]=="Retain"))
      {
      retain[var]=Teuchos::StrUtils::atoi(vartype[1]);
      }
    }


  // loop over all local subdomains and put a value in nodeType: 
  // 0: interior 
  // 1: separator 
  // 4: retained
  // this vector does not yet relate the subdomains to the separators, 
  // but it finds all separator and retained nodes. 
  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    for (int var=0;var<dof_;var++)
      {
      retained[var]=0;
      }
    int sub=(*partitioner_)(partitioner_->GID(sd,0));// global subdomain ID
    for (int i=partitioner_->First(sd);i<partitioner_->First(sd+1);i++)
      {
      int row=partitioner_->Map().GID(i);
      int type=partitioner_->VariableType(row);      
      
      int len;
      nodeType[i] = 0; // assume interior
      
      // for variables of the "Retain X" type
      // the first X are retained and the rest
      // is automatically interior.
      if (retain[type]>0)
        {
        if (retained[type]<retain[type])
          {
          DEBUG("retain "<<row<<" in Schur complement");
          nodeType[i]=5;
          retained[type]++;
          }
        }
      else
        {
        CHECK_ZERO(G.ExtractMyRowView(G.LRID(row),len,cols));

        // if a node connects to a subdomain with higher ID,            
        // it is marked as a separator node. To make this work          
        // for periodic boundary conditions, we built in a function     
        // to impose an ordering on the subdomains in the class         
        // BasePartitioner. Just using the higher subdomain ID would    
        // result in a situation like this:                             
        //                                                              
        // -------------+ +-------------                
        // SD2          | |         SD3                 
        //              | |                             
        //--------------+ +-------------                
        //                                              
        //--------------+ +-------------                
        //              | |                             
        // SD0          | |         SD1                 
        //              | |                             
        //--------------+ +-------------                
        // *****************************                
        //                                              
        // where the pressures marked by * are identified as    
        // isolated because they do not couple to interior V-   
        // nodes in subdomain 0. However, they do couple to     
        // interior V-nodes of subdomain 2. So the partitioner  
        // has to make sure that the separator nodes are taken  
        // from SD2 instead: flow(gid1,gid2) has to be either   
        // positive or negative, depending on wether gid1 and   
        // gid2 couple across the inner or the outer boundary.  
        for (int j=0;j<len;j++)
          {
          int cj=G.GCID(cols[j]);
          //int colsub=(*partitioner_)(cols[j]);
          if (partitioner_->flow(row,cj)<0)
            {
            nodeType[i]=1;
            break;
            }
          }
        }// not retained
      }//i
    }//sd
  return 0;
  }

  //! detect isolated interior nodes and mark them 'retain' (3)
  int StandardNodeClassifier::UpdateNodeTypeVector(
                      const Epetra_CrsGraph& G, 
                      const Epetra_IntVector& p_nodeType,
                            Epetra_IntVector& nodeType) const
  {
  START_TIMER3(Label(),"UpdateNodeTypeVector");
  
  int MaxNumEntriesPerRow = G.MaxNumIndices();
  
  int *cols;
  int len;
 
  const Epetra_BlockMap& map = nodeType.Map();
  const Epetra_BlockMap& p_map = p_nodeType.Map();
#ifdef TESTING
  if (!map.SameAs(partitioner_->Map()))
    {
    Tools::Error("nodeType vector based on wrong map!",__FILE__,__LINE__);
    }
  if (!p_map.SameAs(G.RowMap()))
    {
    Tools::Error("p_nodeType vector based on wrong map!",__FILE__,__LINE__);
    }
#endif  
  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    for (int i=partitioner_->First(sd); i<partitioner_->First(sd+1);i++)
      {
      int row=map.GID(i);
      int lrow = G.LRID(row);
      CHECK_ZERO(G.ExtractMyRowView(lrow,len,cols));
      int my_type = p_nodeType[lrow];
      int var_i = partitioner_->VariableType(row);
      int min_neighbor = 99;
      // check, for instance, if this edge separator node only connects to
      // other edge separator nodes, in which case it becomes a vertex. To
      for (int j=0;j<len;j++)
        {
        int gcid = G.GCID(cols[j]);
#ifdef TESTING
        if (p_map.LID(gcid)==-1)
          {
          std::string msg="parallel map used does not contain all necessary nodes, node "
          +Teuchos::toString(gcid)+" not found on processor "
          +Teuchos::toString(nodeType_->Comm().MyPID());
          Tools::Error(msg,__FILE__,__LINE__);
          }
#endif
        int var_j = partitioner_->VariableType(gcid);
        if (gcid!=row)
          {
          if (var_i==var_j)
            {
            min_neighbor=std::min(min_neighbor,p_nodeType[p_map.LID(gcid)]);
            }
          }
        }//j
      // not interior and not a P-node:
      if (my_type!=0 && my_type!=5)
        {
        if (min_neighbor>=my_type)
          {
          //DEBUG("increase node level of "<<row);
          nodeType[i]=min_neighbor+1;
          }
        }
      }//i
    }//sd
  
  return 0;
  }

  //! detect isolated P-nodes and form full conservation cells
  int StandardNodeClassifier::DetectFCC(
                      const Epetra_CrsGraph& G,
                      const Epetra_IntVector& p_nodeType,
                            Epetra_IntVector& nodeType) const
  {
  START_TIMER3(Label(),"DetectFCC");
  
  int MaxNumEntriesPerRow = G.MaxNumIndices();
  
  int *cols;
  int len;

  // map of partitioner_ and nodeType_
  const Epetra_BlockMap& map = partitioner_->Map();
  // (row-)map of parallelGraph_ (=G) and p_nodeType/p_update
  const Epetra_BlockMap& p_map = p_nodeType.Map();

  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    for (int i=partitioner_->First(sd); i<partitioner_->First(sd+1);i++)
      {
      int row=map.GID(i);
      int lrow = p_map.LID(row);
      int var_i=partitioner_->VariableType(row);
      if (retainIsolated_[var_i]) 
        {
        CHECK_ZERO(G.ExtractMyRowView(lrow,len,cols));
        
        int min_neighbor = 99;
        // check, for instance, if this edge separator node only connects to
        // other edge separator nodes, in which case it becomes a vertex. 
        for (int j=0;j<len;j++)
          {
          int gcid=G.GCID(cols[j]);
          int var_j = partitioner_->VariableType(gcid);
          if (gcid!=row)
            {
            min_neighbor=std::min(min_neighbor,p_nodeType[p_map.LID(gcid)]);
            }
          }//j
        if ((min_neighbor>0))
          {
          DEBUG(" full conservation cell around p: "<<row);
#ifdef DEBUGGING          
          Tools::deb() << "Div-row: ";
          for (int j=0;j<len;j++)
            {
            Tools::deb() << G.GCID(cols[j]) << " ";
            }
          Tools::deb() << std::endl;
#endif
          nodeType[i]=5;
          // all surrounding (velocity) nodes
          // are to be retained. As this is a
          // row from the 'Div' part of the  
          // matrix, there are only connec-  
          // tions to velocities, but the P- 
          // node itself may be in there as a
          // 0 entry.
          for (int j=0;j<len;j++)
            {
            int gcid=G.GCID(cols[j]);
            int var_j=partitioner_->VariableType(gcid);
            if (retainIsolated_[var_j]==false)
              {
              if (map.MyGID(gcid))
                {
                nodeType[map.LID(gcid)]=4;
                }
              else
                {
                // we would have to modify p_nodeType and then import it    
                // back to nodeType. For our applications this occurs e.g.  
                // in parallel 3D NS simulations where the P-nodes on 'line-
                // separators' (separating 4 subdomains) are retained and   
                // therefore velocities on a different partition have to be 
                // retained in the next Schur-Complement. However, in all   
                // situations I can think of these will be retained by the  
                // owning partiiton anyway because of a retained P-node in  
                // the adjacent cell.
                }
              }
            }
          
          }
        }
      }//i
    }//sd
    
  return 0;
  }

//
std::ostream& StandardNodeClassifier::PrintNodeTypeVector
  (const Epetra_IntVector& nT,std::ostream& os,std::string label)
  {
  START_TIMER2(Label(),"PrintNodeTypeVector");
  os << "nodeType ("<<label<<")"<<std::endl;
    
  if (nT.Map().NumMyElements()==0) 
    {
    os << "[empty partition]"<<std::endl;
    return os; 
    }
  int imin,jmin,kmin,vmin,imax,jmax,kmax,vmax;
  imin=nx_;imax=-1;jmin=ny_;jmax=-1;kmin=nz_;kmax=-1;
  for (int lid=0;lid<nT.MyLength();lid++)
    {
    int gid=nT.Map().GID(lid);
    int i,j,k,v;
    Tools::ind2sub(nx_,ny_,nz_,dof_,gid,i,j,k,v);
    imin=std::min(imin,i); imax=std::max(imax,i);
    jmin=std::min(jmin,j); jmax=std::max(jmax,j);
    kmin=std::min(kmin,k); kmax=std::max(kmax,k);
    }
    
  os << "partition: ["<<imin<<".."<<imax<<"]x["<<jmin<<".."<<jmax<<"]x["<<kmin<<".."<<kmax<<"]\n";

    os << "GIDs "<<std::endl;
    for (int k=kmin;k<=kmax;k++)
      {
      os << "k="<<k<<std::endl;
      for (int j=jmax;j>=0;j--)
        {
        for (int i=imin;i<=imax;i++)
          {
          int gid=-1,lid=-1;
          for (int v=0;v<dof_;v++)
            {
            gid=Tools::sub2ind(nx_,ny_,nz_,dof_,i,j,k,v);
            lid = nT.Map().LID(gid);
            if (lid>=0) break;
            }
          if (lid>=0)
            {
            int gid=nT.Map().GID(lid);
            os << std::setw(6)<<gid<<" ";
            }
          else
            {
            os << "      " << " ";
            }
          }
        os << std::endl;
        }
      }

  for (int v=0;v<dof_;v++)
    {
    os << "variable "<<v<<std::endl;
    for (int k=kmin;k<=kmax;k++)
      {
      os << "k="<<k<<std::endl;
      for (int j=jmax;j>=0;j--)
        {
        for (int i=imin;i<=imax;i++)
          {
          int gid = Tools::sub2ind(nx_,ny_,nz_,dof_,i,j,k,v);
          int lid = nT.Map().LID(gid);
          if (lid>=0)
            {
            os << (nT[lid]<0? " ":"  ") << nT[lid];
            }
          else
            {
            os << "   ";
            }
          }
        os << std::endl;
        }
      }
    }
  return os;
  }

}//namespace
