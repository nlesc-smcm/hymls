#include <mpi.h>
#include <iostream>

#include "HYMLS_CartesianStokesClassifier.H"
#include "HYMLS_Tools.H"
#include "HYMLS_CartesianPartitioner.H"

#include "Epetra_Comm.h"
#include "Epetra_SerialComm.h"
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Vector.h"
#include "Epetra_IntVector.h"
#include "Epetra_Import.h"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StrUtils.hpp"
#include "Teuchos_ArrayView.hpp"
#include "Epetra_Util.h"

#include <algorithm>

#ifdef DEBUGGING
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_RowMatrixOut.h"
#endif

#include "EpetraExt_Reindex_CrsMatrix.h"

#include "HYMLS_MatrixUtils.H"

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#endif

#include "Teuchos_StandardParameterEntryValidators.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

//#undef DEBUG
//#define DEBUG(s) std::cerr << s << std::endl;
//#undef DEBVAR
//#define DEBVAR(s) std::cerr << #s << " = "<< s << std::endl;

typedef Teuchos::Array<int>::iterator int_i;

namespace HYMLS {

  //constructor
  
  // gets an overlapping graph of the matrix such that  
  // each processor can access the separators between the partitions, 
  // and a partitioning object.
  // Creates the node-type vector which can be obtained by GetVector()
  // and GetOverlappingVector().
  CartesianStokesClassifier::CartesianStokesClassifier(
        Teuchos::RCP<const Epetra_CrsGraph> parG,
        Teuchos::RCP<const BasePartitioner> P,
        const Teuchos::Array<std::string>& varType,
        const Teuchos::Array<bool>& retIsol,
        GaleriExt::PERIO_Flag perio, int dim,
        int level, int nx, int ny, int nz) : 
                StandardNodeClassifier(parG,P,varType,retIsol,
                        level,nx,ny,nz,
                        "CartesianStokesClassifier"),
                        perio_(perio),dim_(dim)
    {
    START_TIMER3(Label(),"Constructor");    
    return;
    }
    
  CartesianStokesClassifier::~CartesianStokesClassifier()
    {
    START_TIMER3(Label(),"Destructor");
    }



  
int CartesianStokesClassifier::BuildNodeTypeVector()
  {
  START_TIMER2(Label(),"BuildNodeTypeVector");
  //! first we import our original matrix into the ordering defined by the partitioner.

  if (Teuchos::is_null(parallelGraph_))
    {
    Tools::Error("parallelGraph not yet constructed!",__FILE__,__LINE__);
    }
  
  nodeType_=Teuchos::rcp(new Epetra_IntVector(partitioner_->Map()));
  nodeType_->PutValue(-1);

  CHECK_ZERO(this->BuildInitialNodeTypeVector(*parallelGraph_,*nodeType_));

  // now every subdomain has marked those nodes it owns and 
  // that should become interior, separator or retained nodes.

  // import partition overlap
  p_nodeType_=Teuchos::rcp(new Epetra_IntVector(parallelGraph_->RowMap()));
  Epetra_Import import(p_nodeType_->Map(),nodeType_->Map());
  CHECK_ZERO(p_nodeType_->Import(*nodeType_,import,Insert));

#if defined(STORE_MATRICES)||defined(TESTING)
std::ofstream nodeTypeStream;
nodeTypeStream.open(("nodeTypes_L"+Teuchos::toString(myLevel_)+
        "_"+Teuchos::toString(nodeType_->Comm().MyPID())+".txt").c_str(),ios::trunc);
this->PrintNodeTypeVector(*p_nodeType_,nodeTypeStream,"initial");
#endif

  // increase the node type of nodes that only connect to separators.         
  // A type 1 sep node that only connects to sep nodes becomes a type         
  // 2 sep node. 

  // the UpdateNodeTypeVector function doesn't work well for 
  // Navier-Stokes on coarser levels because the corner V-   
  // nodes get new couplings by eliminating P-nodes. So we   
  // use a specialized implementation which is based on FCCs 
  // instead.
  CHECK_ZERO(this->UpdateNodeTypeVector_CartStokes(*parallelGraph_,*p_nodeType_, *nodeType_));

  // import again
  CHECK_ZERO(p_nodeType_->Import(*nodeType_,import,Insert));

#if defined(STORE_MATRICES) || defined(TESTING)
this->PrintNodeTypeVector(*p_nodeType_,nodeTypeStream,"step 2");
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
  Epetra_IntVector p_updateNT(p_nodeType_->Map());
  Epetra_IntVector updateNT(nodeType_->Map());
  CHECK_ZERO(p_updateNT.PutValue(0));
  CHECK_ZERO(this->DetectFCC_CartStokes(*parallelGraph_,*p_nodeType_, p_updateNT));

  CHECK_ZERO(updateNT.Export(p_updateNT,import,Add));
  
  // add update
  for (int i=0;i<nodeType_->MyLength();i++)
    {
    (*nodeType_)[i]+=updateNT[i];
    }

  CHECK_ZERO(p_nodeType_->Import(*nodeType_,import,Insert));

#if defined(STORE_MATRICES) || defined(TESTING)
  this->PrintNodeTypeVector(*p_nodeType_,nodeTypeStream,"with FCCs");
#endif
  
  // we now have the following node types
  // 0: interior
  // 1: edge (2D) or face (3D) separator node
  // 2: corner (2D) or edge (3D) separator node
  // 3: 3D corner 
  // 4: retained V-nodes. In 2D these are full conservation cells.
  //    In 3D they can be clustered to form full conservation tubes, 
  // 5: retained P-nodes
  CHECK_ZERO(this->DetectFCT(*parallelGraph_,*p_nodeType_, *nodeType_));

  // import again
  CHECK_ZERO(p_nodeType_->Import(*nodeType_,import,Insert));

#if defined(STORE_MATRICES)||defined(TESTING)
  this->PrintNodeTypeVector(*p_nodeType_,nodeTypeStream,"with FCTs");
#endif
    
  return 0;
  }


  int CartesianStokesClassifier::UpdateNodeTypeVector_CartStokes(
                      const Epetra_CrsGraph& G, 
                      const Epetra_IntVector& p_nodeType,
                            Epetra_IntVector& nodeType) const
  {
  START_TIMER3(Label(),"UpdateNodeTypeVector_CartStokes");
  
  // we use geometrical info for the moment to get the nodeType vector correct,
  // we want 2 on edges and 3 in vertices (in 3D).
  for (int lid=0;lid<nodeType.MyLength();lid++)
    {
    int gid=nodeType.Map().GID(lid);
    int i,j,k,var;
    Tools::ind2sub(nx_,ny_,nz_,dof_,gid,i,j,k,var);
    if (var<dim_ && nodeType[lid]>0)
      {
      int my_part=(*partitioner_)(gid);
      int ip1,jp1,kp1;
      ip1 = (perio_&GaleriExt::X_PERIO)? MOD(i+1,nx_) : std::min(i+1,nx_-1);
      jp1 = (perio_&GaleriExt::Y_PERIO)? MOD(j+1,ny_) : std::min(j+1,ny_-1);
      kp1 = (perio_&GaleriExt::Z_PERIO)? MOD(k+1,nz_) : std::min(k+1,nz_-1);
      
      int nb_i = Tools::sub2ind(nx_,ny_,nz_,dof_,ip1,j,k,var);
      int nb_j = Tools::sub2ind(nx_,ny_,nz_,dof_,i,jp1,k,var);
      int nb_k = Tools::sub2ind(nx_,ny_,nz_,dof_,i,j,kp1,var);
      
      bool edge_i = (my_part!=(*partitioner_)(nb_i));
      bool edge_j = (my_part!=(*partitioner_)(nb_j));
      bool edge_k = (my_part!=(*partitioner_)(nb_k));

      if (edge_i&&edge_j) nodeType[lid]++;
      if ((edge_i||edge_j)&&edge_k) nodeType[lid]++;
      }
    else if (var==dim_)
      {
      int my_part=(*partitioner_)(gid);

      // fix a few P-nodes at the boundary
      bool i0=(i==0 && !(perio_&GaleriExt::X_PERIO));
      bool j0=(j==0 && !(perio_&GaleriExt::Y_PERIO));
      bool k0=(k==0 && !(perio_&GaleriExt::Z_PERIO));
      int ip1 = (perio_&GaleriExt::X_PERIO)? MOD(i+1,nx_) : std::min(i+1,nx_-1);
      int jp1 = (perio_&GaleriExt::Y_PERIO)? MOD(j+1,ny_) : std::min(j+1,ny_-1);
      int kp1 = (perio_&GaleriExt::Z_PERIO)? MOD(k+1,nz_) : std::min(k+1,nz_-1);
      
      int nb_i = Tools::sub2ind(nx_,ny_,nz_,dof_,ip1,j,k,var);
      int nb_j = Tools::sub2ind(nx_,ny_,nz_,dof_,i,jp1,k,var);
      int nb_k = Tools::sub2ind(nx_,ny_,nz_,dof_,i,j,kp1,var);
      
      bool edge_i = (my_part!=(*partitioner_)(nb_i));
      bool edge_j = (my_part!=(*partitioner_)(nb_j));
      bool edge_k = (my_part!=(*partitioner_)(nb_k));
      
      // we put in a 4 here, it will be increased to 5 by
      // DetectFCC and the P-node will be 'retained'
      nodeType[lid]= (edge_i&&edge_j&&k0)?4:nodeType[lid];
      nodeType[lid]= (edge_i&&edge_k&&j0)?4:nodeType[lid];
      nodeType[lid]= (edge_j&&edge_k&&i0)?4:nodeType[lid];
      }
    }
  
  return 0;
  }
  
  //! detect isolated P-nodes and form full conservation cells
  int CartesianStokesClassifier::DetectFCC_CartStokes(
                      const Epetra_CrsGraph& G,
                      const Epetra_IntVector& p_nodeType,
                            Epetra_IntVector& p_update) const
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
          p_update[p_map.LID(row)]++;
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
            if (partitioner_->VariableType(gcid)<dim_)
              {
              if (p_map.MyGID(gcid))
                {
                p_update[p_map.LID(gcid)]++;
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
                Tools::Error("incorrect parallel graph",__FILE__,__LINE__);
                }
              }
            }
          
          }
        }
      }//i
    }//sd
    
  return 0;
  }

  // Additional step for 3D Stokes - form full conservation tubes
  int CartesianStokesClassifier::DetectFCT(
                      const Epetra_CrsGraph& G, 
                      const Epetra_IntVector& p_nodeType,
                            Epetra_IntVector& nodeType) const
  {
  if (dim_<3 || dof_<4) return 0;
  START_TIMER3(Label(),"DetectFCT");

  // we currently have the following node types for
  // V-nodes, assuming that UpdateNodeTypeVector 
  // has been called twice and DetectFCC once.
  //                            
  // 0: interior                
  // 1: face separator node     
  // 2 and 3: edges            
  // 4: these can be eliminated 
  //    unless they connect to   
  //    type 2 nodes.
  //                            
  // 1 1 2 3 1 1                
  // 4 4 4 5 4 4                
  // 1 1 2 3 1 1                
  // 1 1 2 3 1 1                
  // 1 1 2 3 1 1                
  //                            
  // For pressures we have type 5 for the P-node which is  
  // retained per subdomain and type 1 in the edge separa- 
  // tors and corners. A type 1 P-node can be eliminated   
  // if it does not connect to a type 5 V-node.            
  //                                                       

  int MaxNumEntriesPerRow = G.MaxNumIndices();
  
  int *cols;
  int len;
 
  const Epetra_BlockMap& map = nodeType.Map();
  const Epetra_BlockMap& p_map = p_nodeType.Map();
  
  // for all V-nodes which are currently marked 4,      
  // make them interior if they do not connect to       
  // type 2 V-nodes.                                    
  // A P-node marked as '1' can be eliminated if        
  // it does not connect toa type-5 V-node.             
  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    DEBVAR(sd);
    for (int i=partitioner_->First(sd); i<partitioner_->First(sd+1);i++)
      {
      int row=map.GID(i);
      bool eliminate=false;
      bool retain=false;
      int type_i = p_nodeType[p_map.LID(row)];
      int var_i = partitioner_->VariableType(row);
      if (var_i==dim_ && type_i==1) // candidate P-node for elimination
        {
        eliminate=true;
        DEBUG("P-Node "<<partitioner_->GID(sd,i));
        int lrow = G.LRID(row);
        CHECK_ZERO(G.ExtractMyRowView(lrow,len,cols));
        for (int j=0;j<len;j++)
          {
          int gcid=G.GCID(cols[j]);
          if (p_nodeType[p_map.LID(gcid)]==5)
            {
            eliminate=false;
            break;
            }
          }//j
        retain=!eliminate;
        }
      else if (type_i==4) //candidate V-node for elimination
        {
        eliminate=true;
        DEBUG("V-Node "<<partitioner_->GID(sd,i));
        int lrow = G.LRID(row);
        int sd_i = (*partitioner_)(row);
        CHECK_ZERO(G.ExtractMyRowView(lrow,len,cols));
        for (int j=0;j<len;j++)
          {
          int gcid=G.GCID(cols[j]);
          DEBUG("\t"<<gcid<<" ["<<p_nodeType[p_map.LID(gcid)]<<"]")
          int type_j = p_nodeType[p_map.LID(gcid)];
          int var_j = partitioner_->VariableType(gcid);
          if (var_j==var_i)
            {
            if (type_j==2)
              {
              eliminate=false;
              break;
              }
            }
          }//j
        }//if 
      if (eliminate)
        {
        DEBUG("eliminate node "<<partitioner_->GID(sd,i)<<" as interior");
        nodeType[i]=var_i-dim_-1;
        }
      if (retain)
        {
        DEBUG("retain node "<<partitioner_->GID(sd,i)<<" as single P-node");
        nodeType[i]=5;
        }
      }//i
    }//sd

  // for new interior V-nodes we now have -2/-3/-4 (u/v/w), and for P-nodes
  // to be eliminated -1. Put the same value as the connected V-node in the
  // P-node to get this:      
  //                          
  //                -2        
  //              -2          
  //            -2            
  //  -1 -1 -1 *              
  //          -3              
  //          -3              
  //          -3              
  // The negative nodeType entries serve to form subcells of subdomain sd
  // in GroupSeparators() later on.                                      
  //                                                                     

  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    for (int i=partitioner_->First(sd); i<partitioner_->First(sd+1);i++)
      {
      int row=map.GID(i);
      int type_i = nodeType[i];
      int var_i = partitioner_->VariableType(row);
      if (var_i==dim_ && type_i<0) // P-node to be eliminated
        {
        int lrow = G.LRID(row);
        CHECK_ZERO(G.ExtractMyRowView(lrow,len,cols));
        for (int j=0;j<len;j++)
          {
          int gcid=G.GCID(cols[j]);
          int type_j=nodeType[map.LID(gcid)];
          nodeType[i]=std::min(type_i,type_j);
          type_i=nodeType[i];
          }//j
        }//if
      }//i
    }//sd

  return 0;  
  }

}//namespace
