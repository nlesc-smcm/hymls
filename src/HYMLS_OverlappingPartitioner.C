#include <mpi.h>
#include <iostream>

#include "HYMLS_OverlappingPartitioner.H"
#include "HYMLS_Tools.H"
#include "HYMLS_BasePartitioner.H"
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
#include "Epetra_Util.h"

#include "Galeri_Utils.h"

#include "HYMLS_SepNode.H"

#include <algorithm>


#ifdef DEBUGGING
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_RowMatrixOut.h"
#endif

#include "Galeri_Maps.h"
#include "Galeri_Periodic.h"
#include "Galeri_Star2D.h"
#include "Galeri_Star3D.h"

#include "EpetraExt_Reindex_CrsMatrix.h"

#include "HYMLS_MatrixUtils.H"

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#endif

typedef Teuchos::Array<int>::iterator int_i;

namespace HYMLS {

  //constructor
  
  // we call the base class constructor with a lot of null-pointers and create the
  // data structures ourselves in the constructor. This means that the base class
  // is not fully initialized during the constructor, but afterwards it is.
  // This is OK because the base class constructor is mostly intended for spawning
  // a new level from an existing one.
  OverlappingPartitioner::OverlappingPartitioner(Teuchos::RCP<const Epetra_RowMatrix> K, 
      Teuchos::RCP<Teuchos::ParameterList> params)
      : RecursiveOverlappingPartitioner(Teuchos::rcp(&(K->Comm()),false),
                                        Teuchos::rcp(&(K->RowMatrixRowMap()),false),
                                        Teuchos::null,
                                        Teuchos::rcp(new Teuchos::Array< Teuchos::Array<int> >()),
                                        "OverlappingPartitioner"),
      matrix_(K),
      params_(params)
    {
    
    START_TIMER(label_,"Constructor");


    UpdateParameters();
    
    // try to construct or guess the connectivity of a related scalar problem
    // ('Geometry Matrix')
    if (substituteGraph_)
      {
      CreateGraph();
      }
    else
      {
      Teuchos::RCP<const Epetra_CrsMatrix> myCrsMatrix = 
          Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);

      if (Teuchos::is_null(myCrsMatrix))
        {
        Tools::Error("we need a CrsMatrix here!",__FILE__,__LINE__);
        }
      // copy the graph
      graph_=Teuchos::rcp(new Epetra_CrsGraph(myCrsMatrix->Graph()));
      }

    Partition();
        
    DetectSeparators();
    
    GroupSeparators();
        
    
    DEBVAR(*this);
    
    STOP_TIMER(label_,"Constructor");
    }
    
  OverlappingPartitioner::~OverlappingPartitioner()
    {
    DEBUG("OverlappingPartitioner::~OverlappingPartitioner()");
    }



void OverlappingPartitioner::UpdateParameters()
  {
  START_TIMER2(label_,"UpdateParameters");
  Teuchos::ParameterList& problParams=params_->sublist("Problem");
  
  dim_=problParams.get("Dimension",2);  
  
  if (!problParams.isSublist("Problem Definition"))
    {
    // this sublist is created by class Solver if you set "Equations" to something
    // it recognizes (like "Laplace" or "Stokes-C"). You can also set it manually.
    Tools::Error("the internal sublist 'Problem Definition' is missing.",
        __FILE__,__LINE__);
    }
  
  Teuchos::ParameterList defiParams=problParams.sublist("Problem Definition");
  
  dof_=defiParams.get("Degrees of Freedom",1);
  
  perio_=defiParams.get("Periodicity",Galeri::NO_PERIO);
  
  if (dim_==2 && perio_!=Galeri::NO_PERIO)
    {
    HYMLS::Tools::Error("Cannot handle periodic BC in 2D right now!",
        __FILE__,__LINE__);
    }
  
  // on the first level we typically substitute the graph by an idealized
  // one to get a 'good' HID reordering. This typically fails on coarser
  // levels and one may want to disable it also for later use on unstructured
  // grids.
  substituteGraph_ = defiParams.get("Substitute Graph",true);
  
  variableType_.resize(dof_);
  retainIsolated_.resize(dof_);  
  
  for (int i=0;i<dof_;i++)
    {
    Teuchos::ParameterList& varList=defiParams.sublist("Variable "+Teuchos::toString(i));
    variableType_[i]=varList.get("Variable Type","Laplace");
    retainIsolated_[i]=varList.get("Retain Isolated",false);
    }
  
  nx_=problParams.get("nx",-1);
  ny_=problParams.get("ny",nx_);
  if (dim_>2)
    {
    nz_=problParams.get("nz",nx_);
    }
  else
    {
    nz_=1;
    }
  if (nx_==-1)
    {
    Tools::Error("You must presently specify nx, ny (and possibly nz) in the input file",__FILE__,__LINE__);
    }  
    
  partitioningMethod_=params_->sublist("Solver").get("Partitioner","Cartesian");  
  
  DEBVAR(*params_);
  
  STOP_TIMER2(label_,"UpdateParameters");
  }

void OverlappingPartitioner::CreateGraph()
  {
  START_TIMER2(label_,"CreateGraph");
  
  // in the most general case we could use the pattern of given matrix. This could be done
  // whenever we get a CrsMatrix, but will typically not work because our
  // OverlappingPartitioner algorithm is not so general. (The resulting ordering will not have the quality
  // needed by the solver in the sense that the groups are not proper separators etc.)

  Teuchos::RCP<const Epetra_CrsMatrix> myCrsMatrix = 
        Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);

  if (Teuchos::is_null(myCrsMatrix))
    {
    Tools::Error("we need a CrsMatrix here!",__FILE__,__LINE__);
    }

  // we now look at the variable types:
  // "Laplace": treated as 9- (27-)point stencil in 2D (3D)
  // "Uncoupled": treated as identity
  // "z-Laplace": uncoupled in x- and y-direction, Laplace in z- (for tHCM, mostly)
  // "Retain X": treated as identity (get special attention in GroupSeparators())

  // we will replace the stencil for "Laplace" using a matrix created by Galeri:
  Teuchos::ParameterList galeriList;

  galeriList.set("nx",nx_);
  galeriList.set("ny",ny_);
  galeriList.set("nz",nz_);

  std::string mapType="Cartesian"+Teuchos::toString(dim_)+"D";

  Teuchos::RCP<const Epetra_Map> scalarMap;

  Epetra_SerialComm serialComm;
  
  DEBVAR(*baseMap_);

  // create a dof-1 map
  if (dof_==1) 
    {
    scalarMap=baseMap_;
    }
  else
    {
    try 
     {
     DEBUG("create a simple scalar map");
     scalarMap=Teuchos::rcp(Galeri::CreateMap(mapType, serialComm, galeriList));
     } catch (Galeri::Exception G) {G.Print();}
   }
    
    
  Teuchos::RCP<Epetra_CrsMatrix> scalarLaplace;
  
  // replace the u, v, T and S operators by Laplace 27-point stencils
  // we assume that the variables are ordered per grid cell as u,v,w,p,T,S
  DEBUG("create new stencil for Laplace-type variables");
  try
   {
   if (dim_==2)
     {
     scalarLaplace=Teuchos::rcp(Galeri::Matrices::Star2D(scalarMap.get(),nx_,ny_,
                                       1.0,1.0,1.0,
                                       1.0,1.0,1.0,
                                       1.0,1.0,1.0));
     }
   else if (dim_==3)
     {
     scalarLaplace=Teuchos::rcp(Galeri::Matrices::Star3D(scalarMap.get(),nx_,ny_,nz_,
                                       1.0,1.0,1.0,1.0,perio_));
     }
   } catch (Galeri::Exception G) {G.Print();}  


  
  // I built this in for THCM, it is not really
  // nice to have it here because it is so application-
  // specific (I use it for the w-variables in THCM).
  Teuchos::RCP<Epetra_CrsMatrix> zLaplace=Teuchos::null;
  
  for (int var=0;var<dof_;var++)
    {
    if (variableType_[var]=="z-Laplace")
      {
      if (zLaplace == Teuchos::null)
        {
        int le,ri,lo,up,be,ab;
        DEBUG("create new stencil for z-Laplace variables");
        zLaplace = Teuchos::rcp(new Epetra_CrsMatrix(Copy,*scalarMap,3));
        std::vector<int> indices(3);
        std::vector<double> values(3);
        for (int i=0;i<3;i++) values[i]=1.0;
        
        int len;        
        for (int i=0;i<zLaplace->NumMyRows();i++)
          {
          int grid = zLaplace->GRID(i);
          Galeri::GetNeighboursCartesian3d(grid, nx_, ny_, nz_,
                             le, ri, lo, up, be, ab);
          len=0;
          indices[len++]=grid;
          if (be!=-1) indices[len++]=be;
          if (ab!=-1) indices[len++]=ab;
          CHECK_ZERO(zLaplace->InsertGlobalValues(grid, len, &values[0], &indices[0]));
          }
        CHECK_ZERO(zLaplace->FillComplete());
        }
      }
    }

  int len;
  int maxlen=scalarLaplace->MaxNumEntries();
  int* indices=new int[maxlen];
  double* values=new double[maxlen];

  Teuchos::RCP<Epetra_CrsMatrix> crsMatrix = Teuchos::rcp(new Epetra_CrsMatrix
        (Copy,*baseMap_,maxlen));
  
  DEBUG("Put in the new stencils");
  for (int var=0;var<dof_;var++)
    {
    if (variableType_[var]=="Laplace")
      {
      for (int i=0;i<crsMatrix->NumMyRows()/dof_;i++)
        {
        int point_gid=crsMatrix->GRID(i*dof_)/dof_;
        CHECK_ZERO(scalarLaplace->ExtractGlobalRowCopy(point_gid,maxlen,len,values,indices));
        // adjust column indices
        for (int j=0;j<len;j++) indices[j]=dof_*indices[j]+var;

        int gid=crsMatrix->GRID(i*dof_+var);
        CHECK_ZERO(crsMatrix->InsertGlobalValues(gid,len,values,indices));
        }
      }
    else if (variableType_[var]=="z-Laplace")
      {
      for (int i=0;i<crsMatrix->NumMyRows()/dof_;i++)
        {
        int point_gid=crsMatrix->GRID(i*dof_)/dof_;
        CHECK_ZERO(zLaplace->ExtractGlobalRowCopy(point_gid,maxlen,len,values,indices));
        // adjust column indices
        for (int j=0;j<len;j++) indices[j]=dof_*indices[j]+var;

        int gid=crsMatrix->GRID(i*dof_+var);
        CHECK_ZERO(crsMatrix->InsertGlobalValues(gid,len,values,indices));
        }
      }
    else if (variableType_[var]=="Uncoupled")
      {
      // put in an identity matrix
      for (int i=0;i<crsMatrix->NumMyRows()/dof_;i++)
        {
        int gid=crsMatrix->GRID(i*dof_+var);
        double val=1.0;
        CHECK_ZERO(crsMatrix->InsertGlobalValues(gid,1,&val,&gid));
        }
      }
    else 
      {
      // keep stencil for "Retain X" variables: We ignore them
      // when looking for separators but need the original connectivity
      // to check for isolated variables
      for (int i=0;i<crsMatrix->NumMyRows()/dof_;i++)
        {
        int lid=i*dof_+var;
        int gid=crsMatrix->GRID(lid);
        CHECK_ZERO(myCrsMatrix->ExtractGlobalRowCopy(gid,maxlen,len,values,indices));
        CHECK_ZERO(crsMatrix->InsertGlobalValues(gid,len,values,indices));
        }
      }
    }
  
  CHECK_ZERO(crsMatrix->FillComplete());
  
  delete [] indices;
  delete [] values;

  // copy the graph
  graph_=Teuchos::rcp(new Epetra_CrsGraph(crsMatrix->Graph()));
    
#ifdef DEBUGGING
MatrixUtils::Dump(*crsMatrix, "graph.txt");
#endif
  STOP_TIMER2(label_,"CreateGraph");
  }




void OverlappingPartitioner::Partition()
  {
  START_TIMER2(label_,"Partition");
  if (partitioningMethod_=="Cartesian")
    {
    partitioner_=Teuchos::rcp(new CartesianPartitioner(baseMap_,nx_,ny_,nz_,dof_,perio_));
    }
  else
    {
    Tools::Error("Up to now we only support Cartesian partitioning",__FILE__,__LINE__);
    }
    
  Teuchos::ParameterList& solverParams=params_->sublist("Solver");
  int npx,npy,npz;
  if (solverParams.isParameter("Separator Length (x)"))
    {
    int sx=solverParams.get("Separator Length (x)",4);
    int sy=solverParams.get("Separator Length (y)",sx);
    int sz=solverParams.get("Separator Length (z)",nz_>1?sx:1);
    npx=(nx_>1)? (int)(nx_/sx): 1;
    npy=(ny_>1)? (int)(ny_/sy): 1;
    npz=(nz_>1)? (int)(nz_/sz): 1;
    }
  else 
    {
    int numGlobalSubdomains=solverParams.get("Number of Subdomains", 
                4*comm_->NumProc());
    Tools::SplitBox(nx_,ny_,nz_,numGlobalSubdomains,npx,npy,npz);
    }
  
  Teuchos::RCP<CartesianPartitioner> cartPart
        = Teuchos::rcp_dynamic_cast<CartesianPartitioner>(partitioner_);
        
  if (cartPart!=Teuchos::null) 
    {
    cartPart->Partition(npx,npy,npz);
    }
  else
    {
    partitioner_->Partition(npx*npy*npz);
    }

  // sanity check
  if (dof_!=partitioner_->DofPerNode())
    {
    Tools::Error("Incompatible map passed to partitioner",__FILE__,__LINE__);
    }


#ifdef DEBUGGING
DEBUG("Partition numbers:");
for (int i=0;i<baseMap_->NumMyElements();i++)
  {
  int gid=baseMap_->GID(i);
  DEBUG(gid << " " << (*partitioner_)(gid));
  }
#endif  
  STOP_TIMER2(label_,"Partition");
  return;
  }
  
void OverlappingPartitioner::DetectSeparators()
  {
  START_TIMER2(label_,"DetectSeparators");
  //! first we import our original matrix into the ordering defined by the partitioner.

  DEBUG("Import Graph...");
  
  if (Teuchos::is_null(graph_))
    {
    Tools::Error("Graph not yet constructed!",__FILE__,__LINE__);
    }


  Epetra_Import importRepart(partitioner_->Map(),*baseMap_);

  int MaxNumEntriesPerRow=graph_->MaxNumIndices();
  
  Epetra_CrsGraph G(Copy,partitioner_->Map(),MaxNumEntriesPerRow,false);

  CHECK_ZERO(G.Import(*graph_,importRepart,Insert));
  CHECK_ZERO(G.FillComplete());
    
  int *cols = new int[MaxNumEntriesPerRow];
  
  Teuchos::Array<Teuchos::Array<int> > interior_nodes(partitioner_->NumLocalParts());
  Teuchos::Array<Teuchos::Array<int> > separator_nodes(partitioner_->NumLocalParts());
  Teuchos::Array<Teuchos::Array<int> > retained_nodes(partitioner_->NumLocalParts());

  DEBUG("Detect separators");

  bool interior;
  Teuchos::Array<int> retain(dof_);
  Teuchos::Array<int> retained(dof_);

  int np_schur=0; // count number of pressures in Schur-complement
                  // (we don't need this count, but it is convenient when 
                  // comparing the ordering with MATLAB)

  for (int var=0;var<dof_;var++)
    {
    Teuchos::Array<string> vartype=Teuchos::StrUtils::stringTokenizer(variableType_[var]);
    retain[var]=0;
    if ((vartype.size()==2)&&(vartype[0]=="Retain"))
      {
      retain[var]=Teuchos::StrUtils::atoi(vartype[1]);
      }
    }

  // iterate over all local subdomains
  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    for (int var=0;var<dof_;var++)
      {
      retained[var]=0;
      }
    DEBVAR(sd);
    int sub=(*partitioner_)(partitioner_->GID(sd,0));// global subdomain ID
    DEBVAR(sub);
    for (int i=partitioner_->First(sd);i<partitioner_->First(sd+1);i++)
      {
      int row=partitioner_->Map().GID(i);
      int type=partitioner_->VariableType(row);
      /*
      DEBVAR(row);
      DEBVAR(type);
      DEBVAR(retain[type]);
      */
      int len;
      interior=true;
      
      // for variables of the "Retain X" type
      // the first X are retained and the rest
      // is automatically interior.
      if (retain[type]>0)
        {
        if (retained[type]<retain[type])
          {
          DEBUG("retain "<<row<<" in Schur complement");
          interior=false;
          retained[type]++;
          np_schur++;
          }
        }
      else
        {
        CHECK_ZERO(G.ExtractGlobalRowCopy(row,MaxNumEntriesPerRow,len,cols));

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
          //int colsub=(*partitioner_)(cols[j]);
          if (partitioner_->flow(row,cols[j])<0)
            {
            DEBUG("node "<<row<<" not interior");
            interior=false;
            }
          }
        // if an interior node connects to a node in a subdomain with lower ID,   
        // that node will be included as a separator node and thus                
        // overlap is created. We do this only for variables of the same type,    
        // so a u-node coupling to a p-node doesn't make that p-node a separator  
        // node.                                                                  
        // This strategy should work fine for orderings like this:                
        //                                                                        
        //                +-vij--+                                                
        //                |      |                                                
        //                | pij uij pi+1j ...                                     
        //                |      |                                                
        //                +-vij-1+         and similar (also B-grids and scalar   
        //                                 problems of course)                    
        //                                                                        
        if (interior)
          {
          for (int j=0;j<len;j++)
            {
            //int colsub=(*partitioner_)(cols[j]);
            int coltype=partitioner_->VariableType(cols[j]);
            DEBVAR(type);
            DEBVAR(coltype);
            if ((partitioner_->flow(row,cols[j])>0) && (coltype==type))
              {
              separator_nodes[sd].append(cols[j]);
              DEBUG("  include "<<cols[j]);
              }
            }
          }
        }

      if (interior)
        {
        interior_nodes[sd].append(row);
        }
      else if (retain[type]>0) // and not interior...
        {
        retained_nodes[sd].append(row);
        }
      else // not retained and not interior
        {
        separator_nodes[sd].append(row);
        }
      }
    }

    // move any nodes that have become isolated from 'interior'                 
    // to 'retained' if this was specified in the "Problem Definition" list     
    // by "Retain Isolated (var)". Also move the separator nodes it connects    
    // to to the 'retained list. This resolves the following situation in       
    // the C-grid Stokes problem:                                               
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

  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    int_i i=interior_nodes[sd].begin();
    while (i!=interior_nodes[sd].end())
      {
      int row = *i;
      int var = partitioner_->VariableType(row);
      bool retained=false;
      
      if (retainIsolated_[var])
        {
        int len;
        bool any_interior=false;
        CHECK_ZERO(G.ExtractGlobalRowCopy(row,MaxNumEntriesPerRow,len,cols));
        for (int j=0;j<len;j++)
          {
          int_i found = std::find(interior_nodes[sd].begin(),interior_nodes[sd].end(),cols[j]);
          if (found!=interior_nodes[sd].end())
            {
            any_interior=true;
            break;
            }
          }
        if (!any_interior)
          {
          DEBUG("isolated node found: "<<row);
          np_schur++;
          // move this element from the interior to the retained list
          retained_nodes[sd].append(row);
          retained=true;
          // now move any other nodes that this one may connect to
          // to the 'retained' array. In the situation sketched above,
          // the u and v variables thus each form their own group, which
          // is important for our preconditioner because they couple to
          // the p-node whereas their groupmates don't. Here we just append
          // them to the list of retained nodes. Later we remove them from 
          // all global lists they belong to.
          for (int j=0;j<len;j++)
            {
            retained_nodes[sd].append(cols[j]);
            }
          }
        }
      if (retained)
        {
        i=interior_nodes[sd].erase(i);
        }
      else
        {
        i++;
        }
      }// while
    }//sd

  int global_np_schur;
  comm_->SumAll(&np_schur,&global_np_schur,1);
  
  if (comm_->MyPID()==0 && global_np_schur>0)
    {
    Tools::Out("Number of retained pressures: "+Teuchos::toString(global_np_schur));
    }

  // now collect the GIDs of retained nodes on all procs,
  // and remove them from all the other lists
  int numMyRetained=0;
  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    numMyRetained+=retained_nodes[sd].size();
    }

  int *myRetainedNodes=NULL;
    
  if (numMyRetained>0)
    {    
    myRetainedNodes=new int[numMyRetained];

    int pos=0;
    for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
      {
      for (int_i j=retained_nodes[sd].begin();j!=retained_nodes[sd].end();j++)
        {
        myRetainedNodes[pos++]=*j;
        }
      }
  
    Teuchos::ArrayView<int> view(myRetainedNodes, numMyRetained);

#ifdef DEBUGGING
  DEBUG("all my retained nodes (with overlap)");
  for (Teuchos::ArrayView<int>::iterator i=view.begin();i!=view.end();i++)
    {
    Tools::deb() << *i << " ";
    }
  Tools::deb() << std::endl;
#endif
  
    std::sort(view.begin(),view.end());
    Teuchos::ArrayView<int>::iterator end
        = std::unique(view.begin(),view.end());

#ifdef DEBUGGING
  DEBUG("all my retained nodes (without overlap)");
  for (Teuchos::ArrayView<int>::iterator i=view.begin();i!=end;i++)
    {
    Tools::deb() << *i << " ";
    }
  Tools::deb() << std::endl;
#endif
        
    numMyRetained=std::distance(view.begin(),end);
    }
  
  DEBVAR(numMyRetained);
    
  int numAllRetained;
  
  comm_->SumAll(&numMyRetained,&numAllRetained,1);

  DEBVAR(numAllRetained);

  int* allRetainedNodes=NULL;
  
  if (comm_->NumProc()==1)
    {
    allRetainedNodes=myRetainedNodes;
    }
  else if (numAllRetained>0)
    {
    allRetainedNodes = new int[numAllRetained];
    
    // here we need am MPI_Allgatherv operation because
    // the local arrays have different sizes. Unfortunately
    // this operation is not provided by Epetra_Comm, so we
    // extract the original MPI_Comm and do the operation
    // manually.
    
    Teuchos::RCP<const Epetra_MpiComm> comm
        = Teuchos::rcp_dynamic_cast<const Epetra_MpiComm>(comm_);

    if (comm==Teuchos::null && (comm_->NumProc()>1))
      {
      Tools::Error("Not an MPI-Comm!",__FILE__,__LINE__);
      }
      
    MPI_Comm mpiComm = comm->Comm();
    
    int *recv_counts = new int[comm_->NumProc()];
    int *recv_disps = new int[comm_->NumProc()];
    
    CHECK_ZERO(comm_->GatherAll(&numMyRetained, recv_counts, 1));
    
    recv_disps[0]=0;
    for (int i=1;i<comm_->NumProc();i++)
      {
      recv_disps[i]=recv_disps[i-1]+recv_counts[i-1];
      }
    
    CHECK_ZERO(MPI_Allgatherv(myRetainedNodes,  numMyRetained,MPI_INT,
                   allRetainedNodes, recv_counts, recv_disps,MPI_INT,
                   mpiComm));
                   
    delete [] recv_counts;
    delete [] recv_disps;
    }

  if (numAllRetained>0)
    {
    Teuchos::ArrayView<int> view_all(allRetainedNodes, numAllRetained);
  
    std::sort(view_all.begin(),view_all.end());
    Teuchos::ArrayView<int>::iterator all_end
          = std::unique(view_all.begin(),view_all.end());

#ifdef DEBUGGING
    DEBUG("all global retained nodes");
    for (Teuchos::ArrayView<int>::iterator i=view_all.begin();i!=all_end;i++)
      {
      Tools::deb() << *i << " ";
      }
    Tools::deb() << std::endl;
#endif

    // now once more loop over all local subdomains and move retained
    // nodes (retained by anyone!) from interior/separator_nodes to  
    // retained_nodes.
    for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
      {
      for (Teuchos::ArrayView<int>::iterator j=view_all.begin();j!=all_end;j++)
        {
        // there may be several instances of this node in separator_nodes
        // at this point, delete all of them (later on we use std::unique
        // to get rid of duplicate values when inserting them in the new 
        // map)
        bool first=true;
        while (true)
          {            
          int_i found=std::find(separator_nodes[sd].begin(),separator_nodes[sd].end(),*j);
          if (found==separator_nodes[sd].end())
            {
            break;
            }
          if (first)
            {
            retained_nodes[sd].append(*found);
            first=false;
            }
          separator_nodes[sd].erase(found);
          }
        }
      }
    }
  
  if (allRetainedNodes!=myRetainedNodes) delete [] allRetainedNodes;
  if ( myRetainedNodes!=NULL) delete []  myRetainedNodes;

  delete [] cols;
  

  DEBUG("Create a map with all overlap");
  // (this map doesn't have the correct grouping of separators, all separators
  // around a subdomain are considered to be one group)
  groupPointer_->resize(partitioner_->NumLocalParts());

  int last;

  int NumMyElements=0;
  Teuchos::Array<Teuchos::Array<int> > my_nodes(partitioner_->NumLocalParts());


  (*groupPointer_)[0].append(0);

  // iterate over all local subdomains
  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    DEBVAR(sd);
    DEBVAR(interior_nodes[sd]);
    DEBVAR(separator_nodes[sd]);
    DEBVAR(retained_nodes[sd]);
    if (sd>0)
      {
      last=(*groupPointer_)[sd-1].size()-1;
      (*groupPointer_)[sd].append((*groupPointer_)[sd-1][last]);
      }
    std::copy(interior_nodes[sd].begin(),interior_nodes[sd].end(),std::back_inserter(my_nodes[sd]));
    last=(*groupPointer_)[sd].size()-1;
    (*groupPointer_)[sd].append((*groupPointer_)[sd][last]+interior_nodes[sd].size());

    std::sort(separator_nodes[sd].begin(),separator_nodes[sd].end());
    int_i begin = separator_nodes[sd].begin();
    int_i end = separator_nodes[sd].end();
    end = std::unique(begin, end);
    std::copy(begin,end,std::back_inserter(my_nodes[sd]));
    last=(*groupPointer_)[sd].size()-1;
    (*groupPointer_)[sd].append((*groupPointer_)[sd][last]+std::distance(begin,end));

    std::sort(retained_nodes[sd].begin(),retained_nodes[sd].end());
    begin = retained_nodes[sd].begin();
    end = retained_nodes[sd].end();
    end = std::unique(begin, end);
    std::copy(begin,end,std::back_inserter(my_nodes[sd]));
    last=(*groupPointer_)[sd].size()-1;
    (*groupPointer_)[sd].append((*groupPointer_)[sd][last]+std::distance(begin,end));
    
    NumMyElements+=my_nodes[sd].size();
    }

  int* MyElements=new int[NumMyElements];

  // and create a list of all overlapping partition elements:

  int pos=0;

  // iterate over all local subdomains
  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    for (int i=0;i<my_nodes[sd].size();i++)
      {
      MyElements[pos++]=my_nodes[sd][i];
      }
    }
  
  
  overlappingMap_=Teuchos::rcp(new  Epetra_Map
        (-1,NumMyElements, MyElements,partitioner_->Map().IndexBase(),*comm_));

  delete [] MyElements;    
  STOP_TIMER2(label_,"DetectSeparators");
  }


void OverlappingPartitioner::GroupSeparators()
  {
  START_TIMER2(label_,"GroupSeparators");
  if (Teuchos::is_null(graph_))
    {
    Tools::Error("Graph not yet constructed!",__FILE__,__LINE__);
    }
  if (Teuchos::is_null(overlappingMap_))
    {
    Tools::Error("Separators not yet detected!",__FILE__,__LINE__);
    }

  // we first create a graph G that has overlap between processors and only
  // columns associated with interior nodes.

  // a map that has overlap only between processors, not between subdomains (row map of G)
  Teuchos::RCP<Epetra_Map> overlappingRowMap;

  // ... and a map that contains only interior nodes, but all of them on every proc (col map of G)
  Teuchos::RCP<Epetra_Map> interiorColMap;
  
  DEBUG("Before grouping separators:");
  DEBVAR(*overlappingMap_);
  
  // first the row map
  
  // throw out duplicate GIDs on each proc using STL
  std::vector<int> inds(overlappingMap_->NumMyElements());
  
  for (int i=0;i<overlappingMap_->NumMyElements();i++)
    {
    inds[i]=overlappingMap_->GID(i);
    }
    
  // make sure GIDs are unique on each proc:
  std::sort(inds.begin(),inds.end());
  int_i end=std::unique(inds.begin(),inds.end());
  
  int NumMyElements = std::distance(inds.begin(),end);
  
  int *MyElements = new int[NumMyElements];
  
  int pos=0;
  for (int_i i=inds.begin();i!=end;i++)
    {
    MyElements[pos++]=*i;
    }

   overlappingRowMap = Teuchos::rcp(new Epetra_Map(
        -1,NumMyElements,MyElements,partitioner_->Map().IndexBase(),*comm_));

  delete [] MyElements;
  
  // now the col map
  NumMyElements=0;

  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    NumMyElements+=(*groupPointer_)[sd][1]-(*groupPointer_)[sd][0];
    // omit separator elements [1]-[2]-1, but add 'retained' ones:
//    NumMyElements+=(*groupPointer_)[sd][3]-(*groupPointer_)[sd][2];
    }
  
  MyElements = new int[NumMyElements];
  
  pos=0;
  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    for (int j=(*groupPointer_)[sd][0];j<(*groupPointer_)[sd][1];j++)
      {
      MyElements[pos++]=overlappingMap_->GID(j);
      }
/*
    for (int j=(*groupPointer_)[sd][2];j<(*groupPointer_)[sd][3];j++)
      {
      MyElements[pos++]=overlappingMap_->GID(j);
      }
*/      
    }

  DEBUG("Create interiorMap");
  // this is a temporary map containing all interior nodes of all subdomains
  // on all processors. We use it to create an overlapping graph.
   interiorColMap = Teuchos::rcp(new Epetra_Map(
        -1,NumMyElements,MyElements,partitioner_->Map().IndexBase(),*comm_));

  delete [] MyElements;
DEBUG("INTERIOR ROW MAP");
DEBVAR(*interiorColMap);

  DEBUG("Create interiorColMap");
  interiorColMap = MatrixUtils::AllGather(*interiorColMap,false);

DEBVAR(*overlappingRowMap);
DEBVAR(*interiorColMap);

  DEBUG("create importer...");

  Teuchos::RCP<Epetra_Import> importOverlap =
    Teuchos::rcp(new Epetra_Import(*overlappingRowMap, *baseMap_));
    
  int MaxNumEntriesPerRow=graph_->MaxNumIndices();
  DEBVAR(MaxNumEntriesPerRow);
  
  // in this graph, all connections to separator nodes have been dropped. We use
  // it to figure out what subdomains a separator actually separates:

  Teuchos::RCP<Epetra_CrsGraph> G = Teuchos::rcp(new Epetra_CrsGraph
      (Copy,*overlappingRowMap,*interiorColMap,MaxNumEntriesPerRow,false));

DEBUG("import...");
  CHECK_ZERO(G->Import(*graph_,*importOverlap,Insert));
#ifdef DEBUGGING
for (int i=0;i<comm_->NumProc();i++)
  {
  if (comm_->MyPID()==i)
    {
    std::ofstream ofs1("rowMap.txt",ios::app);
    std::ofstream ofs2("colMap.txt",ios::app);
    ofs1 << *overlappingRowMap << std::endl;
    ofs2 << *interiorColMap << std::endl;
    ofs1.close();
    ofs2.close();
    }
  comm_->Barrier();
  comm_->Barrier();
  comm_->Barrier();
  comm_->Barrier();
  }
#endif


  CHECK_ZERO(G->FillComplete());
  DEBUG("build separator lists...");

  int* cols = new int[G->MaxNumIndices()];
  int len;

    Teuchos::Array<SepNode> sepNodes;
    
    NumMyElements = overlappingMap_->NumMyElements();
    MyElements=new int[NumMyElements];

  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    int numSepNodes=(*groupPointer_)[sd][2]-(*groupPointer_)[sd][1]; 
                // this does not include retained variables, 
                // which start at groupPointer[2]
                                                               
    DEBVAR(sd);
    DEBVAR(numSepNodes);

    sepNodes.resize(numSepNodes);

    // will contain for every separator node the subdomains it connects to
    // and as last entry in each row the GID of the separator node. We use
    // this array as a criterion when grouping nodes
    Teuchos::Array<int> connectedSubs;

    for (int i=0;i<numSepNodes;i++)
      {
      int row=overlappingMap_->GID((*groupPointer_)[sd][1]+i);

      //DEBUG("Process node "<<i<<", GID "<<row);
      CHECK_ZERO(G->ExtractGlobalRowCopy(row,MaxNumEntriesPerRow,len,cols));
      connectedSubs.resize(len);
      DEBVAR(row);
      for (int j=0;j<len;j++)
        {
        //DEBUG("col: "<<cols[j]<<" part: "<<(*partitioner_)(cols[j]));
        connectedSubs[j]=(*partitioner_)(cols[j]);
        }
      int variableType=partitioner_->VariableType(row);
      SepNode S(row,connectedSubs,variableType);
      sepNodes[i]=S;
      }
      
  // now we sort the nodes by subdomains they connect to.
  // That way we get the correct ordering and only have to
  // set the group pointers:
  DEBUG("Sort sep nodes");
  std::sort(sepNodes.begin(),sepNodes.end());
  DEBVAR(sepNodes);
  

  // list of retained nodes. We sort these by subdomain and variable type so that
  // pressures appear at the end of the ordering.
  int numRetained=(*groupPointer_)[sd][3]-(*groupPointer_)[sd][2];
  Teuchos::Array<SepNode> retNodes(numRetained);
  Teuchos::Array<int> conSub(1);
  conSub[0]=sd;
  
  for (int i=(*groupPointer_)[sd][2];i<(*groupPointer_)[sd][3];i++)
    {
    int gid=overlappingMap_->GID(i);
    int varType = partitioner_->VariableType(gid);
    SepNode S(gid,conSub,varType);
    retNodes[i-(*groupPointer_)[sd][2]]=S;
    }

  // move singletons to the end of the ordering. 
  // singletons are groups with only one element.
  // TODO: this is probably unnecessary here - when
  //       calling RecursiveOverlappingPartitioner::SpawnSeparators
  //       they are moved to the end of the ordering (per process)
  Teuchos::Array<SepNode>::iterator i=sepNodes.begin();
    int num_members=1;
    bool last_member=false;
    while (i!=sepNodes.end())
      {
      Teuchos::Array<SepNode>::iterator next = i+1;
      if (next==sepNodes.end())
        {
        last_member=true;
        }
      else if (*i!=*next)
        {
        last_member=true;
        }
      else
        {
        last_member=false;
        num_members++;
        }

      // move singleton group to 'retained' elements
      if (last_member && (num_members==1))
        {
        SepNode S(i->GID(),conSub,i->type());
        retNodes.append(S);
        i=sepNodes.erase(i);
        ((*groupPointer_)[sd][2])--;  // retained nodes are between groupPointer[sd][2] and [3]
        }
      else
        {
        i++;
        }
      }

  std::sort(retNodes.begin(),retNodes.end());

  // place interior nodes into new map (unchanged)
  for (int i=(*groupPointer_)[sd][0];i<(*groupPointer_)[sd][1];i++)
    {
    MyElements[i]=overlappingMap_->GID(i);
    }
  
  // place reordered separator nodes in new map:
  for (int i=(*groupPointer_)[sd][1];i<(*groupPointer_)[sd][2];i++)
    {
    int j=i-(*groupPointer_)[sd][1];
    MyElements[i]=sepNodes[j].GID();
    }
    
    
  for (int i=(*groupPointer_)[sd][2];i<(*groupPointer_)[sd][3];i++)
    {
    int j=i-(*groupPointer_)[sd][2];
    MyElements[i]=retNodes[j].GID();
    }
    
    
  // adjust group pointer:
  
  int start_sub=(*groupPointer_)[sd][0];
  int start_sep=(*groupPointer_)[sd][1];
  int start_retained=(*groupPointer_)[sd][2];
  
  int num_retained=(*groupPointer_)[sd][3]-(*groupPointer_)[sd][2];
  
  Teuchos::Array<SepNode> tmp=sepNodes;
  Teuchos::Array<SepNode>::iterator end = std::unique(tmp.begin(),tmp.end());
  
  int numGroups=std::distance(tmp.begin(),end);  
  numGroups +=num_retained;     // each retained node is forced 
                                // to start a group of its own
  
  DEBVAR(numGroups);
  
  (*groupPointer_)[sd].resize(numGroups+2); //+1 for interior, +1 for last element
  (*groupPointer_)[sd][0]=start_sub;
  (*groupPointer_)[sd][1]=start_sep;
  int pos=2;
  for (int i=1;i<sepNodes.size();i++)
    {
    if (sepNodes[i]!=sepNodes[i-1])
      {
      // new group starts
      (*groupPointer_)[sd][pos++]=start_sep+i;
      }
    }
  if (start_sep!=start_retained) // there may not be any separator nodes
    {
    (*groupPointer_)[sd][pos++]=start_retained;
    }

  // retained separator nodes: each in a separate group
  for (int i=0;i<num_retained;i++)
    {
    (*groupPointer_)[sd][pos]=(*groupPointer_)[sd][pos-1]+1;
    pos++;
    }
  }
   
  // sort all elements groupwise in lexicographic ordering.
  // Note that sorting sepnodes just puts them in the right
  // group-wise ordering and in each group they sometimes become
  // sorted in a strange way.
  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    for (int grp=0;grp<NumGroups(sd);grp++)
      {
      int len=NumElements(sd,grp);
      Epetra_Util::Sort(true,len, MyElements + (*groupPointer_)[sd][grp],
        0, NULL, 0, NULL);
      }
    }
  

  // rebuild the map with the new ordering:  
  overlappingMap_=Teuchos::rcp(new  Epetra_Map
        (-1,NumMyElements, MyElements,partitioner_->Map().IndexBase(),*comm_));

  DEBVAR(*overlappingMap_);

  delete [] cols;
  delete [] MyElements;

  STOP_TIMER2(label_,"GroupSeparators");
  }



Teuchos::RCP<const OverlappingPartitioner> OverlappingPartitioner::SpawnNextLevel
        (Teuchos::RCP<const Epetra_RowMatrix> Ared, Teuchos::RCP<Teuchos::ParameterList> newList) const
  {
  START_TIMER2(label_,"SpawnNextLevel");
  
  *newList = *params_;

  if (newList->sublist("Solver").get("Partitioner","Cartesian")!="Cartesian")
    {
    Tools::Error("Can currently only handle cartesian partitioners",__FILE__,__LINE__);
    }
  
  int old_sx = newList->sublist("Solver").get("Separator Length",-1);
    
  if (old_sx == -1)
    {
    Tools::Error("'Separator Length' parameter required for spawning!",
                __FILE__, __LINE__);
    }
  
  int new_sx = old_sx*old_sx;
  
  newList->sublist("Solver").set
        ("Separator Length",new_sx);
    
  // the next level typically doesn't really resemble a   
  // structured grid anymore, so we base the partitioning 
  // on the matrix graph rather than an idealized graph.  
  newList->sublist("Problem").sublist
        ("Problem Definition").set("Substitute Graph",false);
  
  Teuchos::RCP<const OverlappingPartitioner> newLevel
        = Teuchos::rcp(new OverlappingPartitioner(Ared, newList));
        
  STOP_TIMER2(label_,"SpawnNextLevel");
  return newLevel;
  }




}
