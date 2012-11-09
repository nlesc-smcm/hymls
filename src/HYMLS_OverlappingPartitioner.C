#include <mpi.h>
#include <iostream>

#include "HYMLS_OverlappingPartitioner.H"
#include "HYMLS_Tools.H"
#include "HYMLS_BaseCartesianPartitioner.H"
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

#include "Ifpack_OverlappingRowMatrix.h"

#include "Galeri_Utils.h"

#include "HYMLS_SepNode.H"

#include <algorithm>

#ifdef DEBUGGING
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_RowMatrixOut.h"
#endif

#include "Galeri_Maps.h"
#include "GaleriExt_Periodic.h"
#include "Galeri_Star2D.h"
#include "GaleriExt_Star3D.h"

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
  
  // we call the base class constructor with a lot of null-pointers and create the
  // data structures ourselves in the constructor. This means that the base class
  // is not fully initialized during the constructor, but afterwards it is.
  // This is OK because the base class constructor is mostly intended for spawning
  // a new level from an existing one.
  OverlappingPartitioner::OverlappingPartitioner(Teuchos::RCP<const Epetra_RowMatrix> K, 
      Teuchos::RCP<Teuchos::ParameterList> params, int level)
      : HierarchicalMap(Teuchos::rcp(&(K->Comm()),false),
                        Teuchos::rcp(&(K->RowMatrixRowMap()),false),
                        0,"OverlappingPartitioner",level),
                        PLA("Problem"), matrix_(K)
    {
    START_TIMER3(Label(),"Constructor");

    setParameterList(params);

    CHECK_ZERO(this->Partition());
    
    // try to construct or guess the connectivity of a related scalar problem
    // ('Geometry Matrix')
    if (substituteGraph_)
      {
      DEBUG("substitute matrix graph");
      CreateGraph();
      }
    else
      {
      DEBUG("use original matrix graph");
      Teuchos::RCP<const Epetra_CrsMatrix> myCrsMatrix = 
          Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);

      if (Teuchos::is_null(myCrsMatrix))
        {
        Tools::Error("we need a CrsMatrix here!",__FILE__,__LINE__);
        }
      // copy the graph
      graph_=Teuchos::rcp(&(myCrsMatrix->Graph()),false);
      }

    // construct a graph with overlap between partitions (for finding/grouping    
    // separators in parallel).
    parallelGraph_=CreateParallelGraph();
    DEBVAR(*parallelGraph_);
    int nzgraph = parallelGraph_->NumMyNonzeros();
    REPORT_SUM_MEM(Label(),"graph with overlap",0,nzgraph,GetComm());
        
#ifdef DEBUGGING
  this->DumpGraph();
#endif
    CHECK_ZERO(this->DetectSeparators());
    DEBVAR(*this);
    CHECK_ZERO(this->GroupSeparators());   
    DEBVAR(*this);
    return;
    }
    
  OverlappingPartitioner::~OverlappingPartitioner()
    {
    START_TIMER3(Label(),"Destructor");
    }



void OverlappingPartitioner::setParameterList
        (const Teuchos::RCP<Teuchos::ParameterList>& params)
  {
  START_TIMER3(Label(),"setParameterList");
  
  setMyParamList(params);
    
  dim_=PL().get("Dimension",2);
  
 
  dof_=PL().get("Degrees of Freedom",1);
  
  perio_=PL().get("Periodicity",GaleriExt::NO_PERIO);
  
  substituteGraph_ = PL().get("Substitute Graph",true);
  
  if (substituteGraph_ && dim_==2 && perio_!=GaleriExt::NO_PERIO)
    {
    // not implemented
    HYMLS::Tools::Error("Cannot handle periodic BC in 2D with 'Substitute Graph'",
        __FILE__,__LINE__);
    }
  
  
  variableType_.resize(dof_);
  retainIsolated_.resize(dof_);  

  for (int i=0;i<dof_;i++)
    { 
    Teuchos::ParameterList& varList=PL().sublist("Variable "+Teuchos::toString(i)); 
    variableType_[i]=varList.get("Variable Type","Laplace");
    retainIsolated_[i]=varList.get("Retain Isolated",false);
    }
  
  nx_=PL().get("nx",-1);
  ny_=PL().get("ny",nx_);
  if (dim_>2)
    {
    nz_=PL().get("nz",nx_);
    }
  else
    {
    nz_=1;
    }
  if (nx_==-1)
    {
    Tools::Error("You must presently specify nx, ny (and possibly nz) in the 'Problem' sublist",__FILE__,__LINE__);
    }  
    
  partitioningMethod_=PL("Preconditioner").get("Partitioner","Cartesian");  
  
    if (validateParameters_)
      {
      this->getValidParameters();
      PL().validateParameters(VPL());
      }
  DEBVAR(PL());
  }

Teuchos::RCP<const Teuchos::ParameterList> OverlappingPartitioner::getValidParameters() const
  {
  if (validParams_!=Teuchos::null) return validParams_;
  START_TIMER3(Label(),"getValidParameters");

  VPL().set("Dimension",2,"physical dimension of the problem");   
  VPL().set("Degrees of Freedom",1,"number of unknowns per node");
  
  VPL().set("Periodicity",GaleriExt::NO_PERIO,"does the problem have periodic BC?"
        " (flag constructed by Preconditioner object)");
  
  VPL().set("Substitute Graph",false,"use idealized graph for partitioning."
        "This flag should not be used anymore except for development/testing.");
        
  Teuchos::RCP<Teuchos::StringToIntegralParameterEntryValidator<int> >
       varValidator = Teuchos::rcp(new Teuchos::StringToIntegralParameterEntryValidator<int>(
                                                    Teuchos::tuple<std::string>
                                                    ( "Laplace","Uncoupled",
                                                    "Retain 1","Retain 2"),"Variable Type"));
  int max_dofs = std::max(dof_,6);                                                      
  for (int i=0;i<max_dofs;i++)
    { 
    Teuchos::ParameterList& varList = VPL().sublist("Variable "+Teuchos::toString(i),
        false, "For each of the dofs in the problem, a list like this instructs the "
               "OverlappingPartitioner object how to treat the variable."
               "For some pre-defined problems which you can select by setting the "
               "'Problem'->'Equations' parameter, the lists are generated by the "
               "Preconditioner object and you don't have to worry about them.");
    varList.set("Variable Type","Laplace",
        "describes how the variable should be treated by the partitioner",
        varValidator);
    varList.set("Retain Isolated",false, 
                "For flow problems we must ensure that isolated pressure "
                "points are retained in the SC, that's what this flag is used for"); 
    }
  
  VPL().set("nx",16,"number of nodes in x-direction");
  VPL().set("ny",16,"number of nodes in y-direction");
  VPL().set("nz",1,"number of nodes in z-direction");
/*  
  VPL().set("Cluster Retained Nodes",false,
        "(only relevant for 3D Navier-Stokes), form full conservation tubes at subdomain\n" 
        " edges to reduce the size of the Schur Complement and the number of retained P-nodes");
*/
  
  Teuchos::RCP<Teuchos::StringToIntegralParameterEntryValidator<int> >
        partValidator = Teuchos::rcp(
                new Teuchos::StringToIntegralParameterEntryValidator<int>(
                    Teuchos::tuple<std::string>("Cartesian","CartFlow"),"Partitioner"));
    
    VPL("Preconditioner").set("Partitioner", "Cartesian",
        "Type of partitioner to be used to define the subdomains",
        partValidator);

  return validParams_;
  }
  
int OverlappingPartitioner::CreateGraph()
  {
  START_TIMER2(Label(),"CreateGraph");
  
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
  
  // create a dof-1 map
  if (dof_==1) 
    {
    scalarMap=GetMap();
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
     {//TODO: this whole section is only used for debugging now - we want to get to the
      //      point where using the original matrix graph works for all problems in 2D and
      //      3D on each level.
      
     // putting in zeros at the right location makes the stencil a 'cross' rather than a 
     // 'star', but the algorithm should work for that case now.
     scalarLaplace=Teuchos::rcp(Galeri::Matrices::Star2D(scalarMap.get(),nx_,ny_,
                                       1.0,1.0,1.0,
                                       1.0,1.0,0.0,
                                       0.0,0.0,0.0));
     }     
   else if (dim_==3)
     {
     scalarLaplace=Teuchos::rcp(GaleriExt::Matrices::Star3D(scalarMap.get(),nx_,ny_,nz_,
                                       1.0,1.0,0.0,0.0,perio_));
//     scalarLaplace=Teuchos::rcp(GaleriExt::Matrices::Star3D(scalarMap.get(),nx_,ny_,nz_,
//                                       1.0,1.0,1.0,1.0,perio_));
     }
   } catch (Galeri::Exception G) {G.Print();}  

   //turn star into cross by dropping the zeros:
   scalarLaplace=MatrixUtils::DropByValue(scalarLaplace);
  
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
        (Copy,Map(),maxlen));
  
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

    int nzgraph = graph_->NumMyNonzeros();
    REPORT_SUM_MEM(Label(),"aux graph",0,nzgraph,GetComm());

  return 0;    
  }


int OverlappingPartitioner::Partition()
  {
  START_TIMER2(Label(),"Partition");
  if (partitioningMethod_=="Cartesian")
    {
    partitioner_=Teuchos::rcp(new CartesianPartitioner
        (GetMap(),nx_,ny_,nz_,dof_,perio_));
    }
  else
    {
    Tools::Error("Up to now we only support Cartesian partitioning",__FILE__,__LINE__);
    }
    
  Teuchos::ParameterList& solverParams=PL("Preconditioner");
  DEBVAR(PL("Preconditioner"));
  int npx,npy,npz;
  int sx=-1;
  int sy,sz,base_sx,base_sy,base_sz;
  if (solverParams.isParameter("Separator Length (x)"))
    {
    sx=solverParams.get("Separator Length (x)",4);
    sy=solverParams.get("Separator Length (y)",sx);
    sz=solverParams.get("Separator Length (z)",nz_>1?sx:1);
    }
  else if (solverParams.isParameter("Separator Length"))
    {
    sx=solverParams.get("Separator Length",4);
    sy=sx; sz=nz_>1?sx:1;
    }
  else
    {
    Tools::Error("Separator Length not set",__FILE__,__LINE__);
    }
  if (solverParams.isParameter("Base Separator Length (x)"))
    {
    base_sx=solverParams.get("Base Separator Length (x)",sx);
    base_sy=solverParams.get("Base Separator Length (y)",base_sx);
    base_sz=solverParams.get("Base Separator Length (z)",nz_>1?base_sx:1);
    }
  else if (solverParams.isParameter("Base Separator Length"))
    {
    base_sx=solverParams.get("Separator Length",sx);
    base_sy=base_sx; sz=nz_>1?base_sx:1;
    }
  else
    {
    base_sx = sx;
    base_sy = sy;
    base_sz = sz;
    }
    
  if (sx>0)
    {
    npx=(nx_>1)? (int)(nx_/sx): 1;
    npy=(ny_>1)? (int)(ny_/sy): 1;
    npz=(nz_>1)? (int)(nz_/sz): 1;
    }
  else 
    {
    int numGlobalSubdomains=solverParams.get("Number of Subdomains", 
                4*Comm().NumProc());
    Tools::SplitBox(nx_,ny_,nz_,numGlobalSubdomains,npx,npy,npz);
    }

  
  Teuchos::RCP<BaseCartesianPartitioner> cartPart
        = Teuchos::rcp_dynamic_cast<BaseCartesianPartitioner>(partitioner_);

  if (cartPart!=Teuchos::null) 
    {
    CHECK_ZERO(cartPart->Partition(npx,npy,npz, false));
    cartPart->SetNodeDistance((double)(sx/base_sx));
    }
  else
    {
    CHECK_ZERO(partitioner_->Partition(npx*npy*npz, false));
    }
  // sanity check
  if (dof_!=partitioner_->DofPerNode())
    {
    Tools::Error("Incompatible map passed to partitioner",__FILE__,__LINE__);
    }

  // we replace the map passed in by the user by the one generated
  // by the partitioner. This has two purposes:                   
  // - the partitioner may decide to repartition the domain, for  
  //   instance if there are more processor partitions than sub-  
  //   domains                                                    
  // - the data layout becomes more favorable because the nodes   
  //   of a subdomain are contiguous in the partitioner's map.    
  SetMap(partitioner_->GetMap());

#ifdef DEBUGGING
DEBUG("Partition numbers:");
for (int i=0;i<Map().NumMyElements();i++)
  {
  int gid=Map().GID(i);
  DEBUG(gid << " " << (*partitioner_)(gid));
  }
#endif  

  // add the subdomains to the base class so we can start inserting groups of nodes
  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    this->AddSubdomain();
    }

  return 0;
  }
  
int OverlappingPartitioner::DetectSeparators()
  {
  START_TIMER2(Label(),"DetectSeparators");
  //! first we import our original matrix into the ordering defined by the partitioner.

  if (Teuchos::is_null(graph_))
    {
    Tools::Error("Graph not yet constructed!",__FILE__,__LINE__);
    }

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

#ifdef DEBUGGING
this->PrintNodeTypeVector(*p_nodeType_,Tools::deb(),"initial");
#endif

  // increase the node type of nodes that only connect to separators.         
  // A type 1 sep node that only connects to sep nodes becomes a type         
  // 2 sep node. 
  CHECK_ZERO(this->UpdateNodeTypeVector(*parallelGraph_,*p_nodeType_, *nodeType_));

  // import to "spread the word"
  CHECK_ZERO(p_nodeType_->Import(*nodeType_,import,Insert));

#ifdef DEBUGGING
  this->PrintNodeTypeVector(*p_nodeType_,Tools::deb(),"step 1");
#endif

  //... then do it again: this is required to get a 3 in the corners in 3D
  CHECK_ZERO(this->UpdateNodeTypeVector(*parallelGraph_,*p_nodeType_, *nodeType_));

  // import again
  CHECK_ZERO(p_nodeType_->Import(*nodeType_,import,Insert));

#ifdef DEBUGGING
if (dim_>2)
  {
  this->PrintNodeTypeVector(*p_nodeType_,Tools::deb(),"step 2");
  }
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
  CHECK_ZERO(this->DetectFCC(*parallelGraph_,*p_nodeType_, *nodeType_));

  // import again
  CHECK_ZERO(p_nodeType_->Import(*nodeType_,import,Insert));

#ifdef DEBUGGING
if (dof_>1)
  {
  this->PrintNodeTypeVector(*p_nodeType_,Tools::deb(),"with FCCs");
  }
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

#ifdef DEBUGGING
if (dim_>2 && dof_>1)
  {
  this->PrintNodeTypeVector(*p_nodeType_,Tools::deb(),"with FCTs");
  }
#endif
  
  // put them into lists
 
  // nodes to be eliminated exactly in the next step  
  Teuchos::Array<int> interior_nodes;
  // separator nodes
  Teuchos::Array<int> separator_nodes;
  // nodes to be retained in the Schur complement (typically pressures)
  Teuchos::Array<int> retained_nodes;

  // first we do all 'regular' subdomains.
  // The special ones (FCCs and FCTs for CartStokes partitioner)
  // are treated in FixSubCells() because they need the regular 
  // subdomains to be finished already (FixSubCells is called   
  // after GroupSeparators()).
  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    interior_nodes.resize(0);
    separator_nodes.resize(0);
    retained_nodes.resize(0);
    CHECK_ZERO(this->BuildNodeLists(sd,*parallelGraph_, *nodeType_, *p_nodeType_,
              interior_nodes, separator_nodes, retained_nodes));

    DEBVAR(sd);
    DEBVAR(interior_nodes);
    DEBVAR(separator_nodes);
    DEBVAR(retained_nodes);

    // in this function we just form three groups per subdomain:
    // interior, separator or retained. In GroupSeparators() this
    // grouping is refined.
    this->AddGroup(sd, interior_nodes);
    this->AddGroup(sd, separator_nodes);
    this->AddGroup(sd, retained_nodes);    
    }
  // build the temporary map with three groups per regular subdomain
  // (interior separator retained)
  CHECK_ZERO(this->FillComplete());
  
  REPORT_SUM_MEM(Label(),"map with overlap",0,OverlappingMap().NumMyElements(),GetComm());
  REPORT_SUM_MEM(Label(),"int vectors",0,nodeType_->MyLength()
        +p_nodeType_->MyLength(),GetComm());
  
  return 0;
  }

  //! build initial vector with 0 (interior), 1 (separator) or 2 (retained)
  int OverlappingPartitioner::BuildInitialNodeTypeVector(
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
            }
          }
        }// not retained
      }//i
    }//sd
  return 0;
  }

  //! detect isolated interior nodes and mark them 'retain' (3)
  int OverlappingPartitioner::UpdateNodeTypeVector(
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
#ifdef TESTING
        if (p_map.LID(G.GCID(cols[j]))==-1)
          {
          std::string msg="parallel map used does not contain all necessary nodes, node "
          +Teuchos::toString(G.GCID(cols[j]))+" not found on processor "
          +Teuchos::toString(Comm().MyPID());
          Tools::Error(msg,__FILE__,__LINE__);
          }
#endif
        int var_j = partitioner_->VariableType(G.GCID(cols[j]));
        if (G.GCID(cols[j])!=row)
          {
          if (var_i==var_j)
            {
            min_neighbor=std::min(min_neighbor,p_nodeType[cols[j]]);
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
  int OverlappingPartitioner::DetectFCC(
                      const Epetra_CrsGraph& G, 
                      const Epetra_IntVector& p_nodeType,
                            Epetra_IntVector& nodeType) const
  {
  START_TIMER3(Label(),"DetectFCC");
  
  int MaxNumEntriesPerRow = G.MaxNumIndices();
  
  int *cols;
  int len;
 
  const Epetra_BlockMap& map = nodeType.Map();
  const Epetra_BlockMap& p_map = p_nodeType.Map();
  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    for (int i=partitioner_->First(sd); i<partitioner_->First(sd+1);i++)
      {
      int row=map.GID(i);
      int lrow = G.LRID(row);
      int var_i=partitioner_->VariableType(row);
      if (retainIsolated_[var_i]) 
        {
        CHECK_ZERO(G.ExtractMyRowView(lrow,len,cols));
        
        int min_neighbor = 99;
        // check, for instance, if this edge separator node only connects to
        // other edge separator nodes, in which case it becomes a vertex. 
        for (int j=0;j<len;j++)
          {
          int var_j = partitioner_->VariableType(G.GCID(cols[j]));
          if (G.GCID(cols[j])!=row)
            {
            min_neighbor=std::min(min_neighbor,p_nodeType[cols[j]]);
            }
          }//j
        if ((min_neighbor>0))
          {
          DEBUG(" full conservation cell around p: "<<row);
#ifdef DEBUGGING          
          Tools::deb() << "Div-row: ";
          for (int j=0;j<len;j++)
            {
            Tools::deb() << p_map.GID(cols[j]) << " ";
            }
          Tools::deb() << std::endl;
#endif
          nodeType[i]++;
          // all surrounding (velocity) nodes
          // are to be retained. As this is a
          // row from the 'Div' part of the  
          // matrix, there are only connec-  
          // tions to velocities.            
          for (int j=0;j<len;j++)
            {
            if (map.MyGID(p_map.GID(cols[j])))
              {
#ifdef TESTING
              if (partitioner_->VariableType(G.GCID(cols[j]))>=dim_)
                {
                Tools::Warning("unexpected Div-row, we're assuming Stokes-type matrix here",
                                __FILE__,__LINE__);
                }
#endif              
              nodeType[map.LID(p_map.GID(cols[j]))]++;
              }
            /*
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
            */
            }
          }
        }
      }//i
    }//sd
    
  return 0;
  }

  // Additional step for 3D Stokes - form full conservation tubes
  int OverlappingPartitioner::DetectFCT(
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
  // 2: and 3: edges            
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
  // if it connects to two type 4 velocities.

  int MaxNumEntriesPerRow = G.MaxNumIndices();
  
  int *cols;
  int len;
 
  const Epetra_BlockMap& map = nodeType.Map();
  const Epetra_BlockMap& p_map = p_nodeType.Map();
  
  // for all V-nodes which are currently marked 'retained',
  // make them interior if they only connect to local P-nodes  
  // and separator V-nodes
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
        int numType4=0;
        for (int j=0;j<len;j++)
          {
          if (p_nodeType[cols[j]]==4) numType4++;
          }//j
        DEBVAR(numType4);
        eliminate= (numType4==2);
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
          DEBUG("\t"<<G.GCID(cols[j])<<" ["<<nodeType[map.LID(G.GCID(cols[j]))]<<"]")
          int type_j = p_nodeType[cols[j]];
          int var_j = partitioner_->VariableType(G.GCID(cols[j]));
          if (var_j==var_i)
            {
//            if (type_j!=1 && type_j!=4)
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
        nodeType[i]=0;
        }
      if (retain)
        {
        DEBUG("retain node "<<partitioner_->GID(sd,i)<<" as single P-node");
        nodeType[i]=5;
        }
      }//i
    }//sd
    
  return 0;  
  }

  //! form list with interior, separator and retained nodes for subdomain
  // sd. Links separators to subdomains.
  int OverlappingPartitioner::BuildNodeLists(int sd, 
                              const Epetra_CrsGraph& G, 
                              const Epetra_IntVector& nodeType,
                              const Epetra_IntVector& p_nodeType,
                              Teuchos::Array<int>& interior,
                              Teuchos::Array<int>& separator,
                              Teuchos::Array<int>& retained) const
  {
  START_TIMER3(Label(),"BuildNodeLists");

  int MaxNumEntriesPerRow = G.MaxNumIndices();
  
  int *cols = new int[MaxNumEntriesPerRow];
  int len;
  
  if (sd>partitioner_->NumLocalParts())
    {
    Tools::Error("not implemeneted",__FILE__,__LINE__);
    }

  int sd_i = partitioner_->GPID(sd);
  DEBVAR(sd);
  DEBVAR(sd_i);
  
  const Epetra_BlockMap& p_map = p_nodeType.Map();
  
#ifdef TESTING
  if (!partitioner_->Map().SameAs(nodeType.Map()))
    {
    Tools::Error("nodeType must be based on partitioner's map",
        __FILE__,__LINE__);
    }
#endif
  for (int i=partitioner_->First(sd); i<partitioner_->First(sd+1);i++)
    {
    int row=partitioner_->Map().GID(i);
    int var_i = partitioner_->VariableType(row);
    // check for non-local separators of the subdomain
    if (nodeType[i]==0)
      {
      // add any separator nodes on adjacent subdomains
      CHECK_ZERO(G.ExtractGlobalRowCopy(row,MaxNumEntriesPerRow,len,cols));
      for (int j=0;j<len;j++)
        {
        int sd_j = (*partitioner_)(cols[j]);
        int var_j = partitioner_->VariableType(cols[j]);
        int nt_j = (*p_nodeType_)[p_map.LID(cols[j])];
        if ((sd_j!=sd_i)&&(var_i==var_j)&&nt_j>0)
          {
          DEBUG("include "<<cols[j]<<" from "<<row);
          if (nt_j>=4)
            {
            retained.append(cols[j]);
            }
          else
            {
            separator.append(cols[j]);
            }
          }
        }
      }
    // put node i in the correct list
    if (nodeType[i]==0)
      {
      interior.append(row);
      }
    else if (nodeType[i]>=4)
      {
      retained.append(row);
      }
    else 
      {
      separator.append(row);
      }
    }

  // now we still miss some nodes that do not have an edge to any interior nodes
  // of subdomain sd. Those are secondary separators, and we have to spot them. 
  // For instance node a for subdomain D in this picture:
  //            
  // C  |  D    
  //   c-d      
  //----+|----  
  //   a|b      
  //A   |    B  
  //            
  // to do this, we check for each pair of separator nodes if
  // they have a common edge to another non-local separator node.
  
  // In general a single iteration may not be enough, as the newly
  // introduced nodes may again connect to missing separators   
  // (e.g. vertices in the corners of 3D subdomains, the first iteration
  // would only find the 'edges').
  std::set<int> separatorL1;
  std::set<int> separatorL2;
  std::set<int> separatorL3;
    
  separatorL1.insert(separator.begin(),separator.end());
  separatorL1.insert(retained.begin(),retained.end());
  // consider the 3D case. We get the following situations (slice in x-y plane)
  // for the lower right subdomain:                             
  //                                                            
  //     'interior in z-direction'      'vertical separator'    
  //                                                            
  //            |                                 |             
  //     SD3    |  SD4                            |             
  //    --------+---------              ----------+--------     
  //           C|AAAAAAAAA                       D|AAAAAAAA     
  //           B|                                C|AAAAAAAA     
  //     SD1   B| SD2                            C|AAAAAAAA     
  //           B|                                C|AAAAAAAA     
  //                    
  // so far we have dealed with the A and B nodes (B's are included  
  // from neighboring subdomains in a consistent way). The first call
  // to 'FindMissingSepNodes' finds all the C nodes by looking for   
  // shared edges between 'A or B' nodes which are not in the same   
  // subdomain. This is sufficient to get the ordering right in 2D.  
  CHECK_ZERO(this->FindMissingSepNodes(sd_i,G,p_nodeType,separatorL1,separatorL2));
  
  // The D-node in the corner of the 3D subdomain has an edge to an A-
  // and two C-nodes on subdomain 1 (from the perspective of SD2).
  // For all other subdomains it has an edge to two C-nodes on 
  // different subdomains.
  CHECK_ZERO(this->FindMissingSepNodes(sd_i,G,p_nodeType,separatorL2,separatorL3));
  separator.resize(0);
  std::copy(separatorL1.begin(),separatorL1.end(),std::back_inserter(separator));
  std::copy(separatorL2.begin(),separatorL2.end(),std::back_inserter(separator));
  std::copy(separatorL3.begin(),separatorL3.end(),std::back_inserter(separator));
  
  separatorL1.insert(separatorL2.begin(),separatorL2.end());
  separatorL1.insert(separatorL3.begin(),separatorL3.end());

  separator.clear();
  retained.clear();

  for (std::set<int>::iterator i=separatorL1.begin();i!=separatorL1.end();i++)
    {
    if (p_nodeType[p_map.LID(*i)]<4)
      {
      separator.append(*i);
      }
    else
      {
      retained.append(*i);
      }
    }
  
 
  delete [] cols;  
  return 0;
  }

//
int OverlappingPartitioner::FindMissingSepNodes
        (int my_sd, const Epetra_CrsGraph& G, const Epetra_IntVector& p_nodeType,
         const std::set<int>& in, std::set<int>& out) const

  {
  START_TIMER3(Label(),"FindMissingSepNodes");
  const Epetra_BlockMap& p_map=p_nodeType.Map();
  int *colsI;
  int *colsJ;
  int lenI,lenJ;

  for (std::set<int>::const_iterator i=in.begin(); i!=in.end();i++)
    {
    int lid_i=p_map.LID(*i);
    DEBVAR(*i);
    int sd_i = (*partitioner_)(*i);
    int type_i = p_nodeType[lid_i];
    int var_i = partitioner_->VariableType(*i);
    for (std::set<int>::const_iterator j=in.begin(); j!=in.end();j++)
      {
      if (*i!=*j)
        {
        int lid_j=p_map.LID(*j);
        int sd_j = (*partitioner_)(*j);
        int type_j = p_nodeType[lid_j];
        int var_j = partitioner_->VariableType(*j);
        // a level 1 node can include level 2 nodes, but not level 0
        // or 1 (and the same holds on each level). This results in 
        // the corner being included in the subdomain for Laplace.  
        // The isRetainedV statement fixes this for Stokes problems.
        // The reason we need the corner is that node C connects to 
        // separator nodes on various subdomains, and from each sub-
        // domain the SC gets a contribution in the A22 block, so   
        // each subdomain must have the C-node.                     
        //                   |                  
        //                   |                  
        //         ----------C---------         
        //                   |                  
        //                   |                  
        if (type_i==type_j && var_i==var_j) 
          {
          DEBUG("\t check "<<*i<<" ["<<type_i<<"] and "<<*j<<" ["<<type_j<<"]");
          CHECK_ZERO(G.ExtractMyRowView(lid_i,lenI,colsI));
          CHECK_ZERO(G.ExtractMyRowView(lid_j,lenJ,colsJ));
          for (int ii=0;ii<lenI;ii++)
            {
            int sd_ii = (*partitioner_)(G.GCID(colsI[ii]));
            int type_ii = p_nodeType[p_map.LID(G.GCID(colsI[ii]))];
            int var_ii = partitioner_->VariableType(G.GCID(colsI[ii]));
            if (sd_ii!=my_sd)
              {
              if ((type_i<type_ii))
                {
                for (int jj=0;jj<lenJ;jj++)
                  {
                  if (G.GCID(colsI[ii])==G.GCID(colsJ[jj]))
                    {
                    DEBUG("\t\t missed node "<<G.GCID(colsJ[jj])<<" inserted");
                    out.insert(G.GCID(colsJ[jj]));
                    }// match
                  }//jj
                }// conditions on node type
              }//if sd_ii != sd_jj and not a P-node
            }//ii
          }// same type i j
        }//if i!=j
      }//j
    }//i
  return 0;
  }


// reorders the separators found in DtectSeparators()
// into groups suitable for our transformations.
int OverlappingPartitioner::GroupSeparators()
  {
  START_TIMER2(Label(),"GroupSeparators");
  if (Teuchos::is_null(graph_))
    {
    Tools::Error("Graph not yet constructed!",__FILE__,__LINE__);
    }
  if (Teuchos::is_null(parallelGraph_))
    {
    Tools::Error("overlapping Graph not yet constructed!",__FILE__,__LINE__);
    }
  if (!Filled())
    {
    Tools::Error("Separators not yet detected!",__FILE__,__LINE__);
    }

  const Epetra_BlockMap& p_map = p_nodeType_->Map();
  
  // copy old data structures and reset base class (HierarchicalMap)
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > groupPointer
        = this->GetGroupPointer();

  Teuchos::RCP<const Epetra_Map> overlappingMap =
        this->GetOverlappingMap();
  
  this->Reset(partitioner_->NumLocalParts());
      
  DEBUG("build separator lists...");

  int MaxNumElements=parallelGraph_->MaxNumIndices();

  int* cols = new int[MaxNumElements];
  int len;

  Teuchos::Array<SepNode> sepNodes;
  Teuchos::Array<int> gidList;
    
  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
#ifdef TESTING
    if (sd>=groupPointer->size())
      {
      Tools::Error("(temporary) groupPointer array incorrect",__FILE__,__LINE__);
      }
    // [ interior separators retained] 
    if ((*groupPointer)[sd].size()!=4)
      {
      Tools::Error("(temporary) groupPointer array incorrect",__FILE__,__LINE__);
      }
#endif

    int numSepNodes=(*groupPointer)[sd][2]-(*groupPointer)[sd][1]; 
                // this does not include retained variables, 
                // which start at groupPointer[2]
                                                               
    DEBVAR(sd);
    DEBVAR(numSepNodes);

    sepNodes.resize(numSepNodes);

    // will contain for every separator node the subdomains it connects to
    // and as last entry in each row the GID of the separator node. We use
    // this array as a criterion when grouping nodes
    Teuchos::Array<int> connectedSubs;
    int new_id = partitioner_->NumLocalParts()+1;
    for (int i=0;i<numSepNodes;i++)
      {
      int row=overlappingMap->GID((*groupPointer)[sd][1]+i);
      int type_i = (*p_nodeType_)[p_map.LID(row)];

      //DEBUG("Process node "<<i<<", GID "<<row);
      connectedSubs.resize(1);
      connectedSubs[0]=(*partitioner_)(row);
      //CHECK_ZERO(parallelGraph_->ExtractGlobalRowCopy(row,MaxNumElements,len,cols));
      int ierr=parallelGraph_->ExtractGlobalRowCopy(row,MaxNumElements,len,cols);
      if (ierr!=0)
        {
        Tools::Error("extracting global row "+Teuchos::toString(row)+
                " failed on rank "+Teuchos::toString(Comm().MyPID()),__FILE__,__LINE__);
        }
      for (int j=0;j<len;j++)
        {
        // We only consider edges to lower-level nodes here,
        // e.g. from face separators to interior, from 
        // edges to faces and from vertices to edges. We also
        // skip edges to subcells (full conservation tubes in Stokes)
        int type_j=(*p_nodeType_)[p_map.LID(cols[j])];
        if (type_j<type_i)
          {
          int flow = partitioner_->flow(row,cols[j]);

          // if the row and col are not in the same subdomain, multiply
          // the partition ID by +1 or -1, depending on the "direction of
          // flow" across the separator. If we don't do this, for periodic
          // BC we can get two different separators identified as one, e.g
          //                                                              
          // | SD1 | SD2 |        (here separators s1 and s2 both connect 
          // s1    s2    s1       to subdomains SD1 and SD2)              
          //                                                              
          if (flow)
            {
            int sign = flow/abs(flow);
            connectedSubs.append(sign*(*partitioner_)(cols[j]));
            }
          }// if nodeType
        }//j
      if (connectedSubs.length()==0)
        {
        // this is a singleton as they appear in the corners
        // of subdomains (only connected to separators). Give
        // it a unique ID so that it isn't put in the same group
        // as other singletons around the same subdomain.
        connectedSubs.append(new_id++);
        }//if
      int variableType= type_i*10 +
                        partitioner_->VariableType(row);
      SepNode S(row,connectedSubs,variableType);
      sepNodes[i]=S;
      }//i
      
    // now we sort the nodes by subdomains they connect to.
    // That way we get the correct ordering and only have to
    // set the group pointers:
    DEBUG("Sort sep nodes");
    std::sort(sepNodes.begin(),sepNodes.end());
  
    // list of retained nodes. We sort these by subdomain and variable type so that
    // pressures appear at the end of the ordering.
    int numRetained=(*groupPointer)[sd][3]-(*groupPointer)[sd][2];
    Teuchos::Array<SepNode> retNodes(numRetained);
    Teuchos::Array<int> conSub(1);
    conSub[0]=sd;
  
    for (int i=(*groupPointer)[sd][2];i<(*groupPointer)[sd][3];i++)
      {
      int gid=overlappingMap->GID(i);
      int varType = partitioner_->VariableType(gid);
      SepNode S(gid,conSub,varType);
      retNodes[i-(*groupPointer)[sd][2]]=S;
      }
      
    // move singletons to the end of the ordering. 
    // singletons are groups with only one element.
    // TODO: this is probably unnecessary here - when
    //       calling HierarchicalMap::SpawnSeparators
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
        ((*groupPointer)[sd][2])--;  // retained nodes are between groupPointer[sd][2] and [3]
        }
      else
        {
        i++;
        }
      }//i

    std::sort(retNodes.begin(),retNodes.end());

    DEBVAR(sepNodes);
    DEBVAR(retNodes);
    
    DEBUG("add the new groups...");

    // place interior nodes into new map (unchanged)
    int num = (*groupPointer)[sd][1]-(*groupPointer)[sd][0];
    gidList.resize(num);
    for (int i=0;i<num;i++)
      {
      gidList[i]=overlappingMap->GID((*groupPointer)[sd][0]+i);
      }
    this->AddGroup(sd,gidList);
  
    // place reordered separator nodes in new map:
    gidList.resize(0);
    for (int i=0;i<sepNodes.size()-1;i++)
      {
      gidList.append(sepNodes[i].GID());
      if (sepNodes[i+1]!=sepNodes[i])
        {
        // new group starts, insert previous one
        this->AddGroup(sd,gidList);
        gidList.resize(0);
        }
      }
    if (sepNodes.length()>0)
      {
      gidList.append((sepNodes.end()-1)->GID());
      this->AddGroup(sd,gidList);
      }

    num = (*groupPointer)[sd][3]-(*groupPointer)[sd][2];
    gidList.resize(1);        
    for (int i=0;i<num;i++)
      {
      gidList[0]=retNodes[i].GID();
      this->AddGroup(sd,gidList);
      }//i
    }//sd
  
  sepNodes.resize(0);
  gidList.resize(0);
  
  // and rebuild map and global groupPointer
  CHECK_ZERO(this->FillComplete());

  delete [] cols;
  return 0;
  }

Teuchos::RCP<const OverlappingPartitioner> OverlappingPartitioner::SpawnNextLevel
        (Teuchos::RCP<const Epetra_RowMatrix> Ared, Teuchos::RCP<Teuchos::ParameterList> newList) const
  {
  START_TIMER3(Label(),"SpawnNextLevel");

  *newList = *getMyParamList();

  std::string partType=newList->sublist("Preconditioner").get("Partitioner","Cartesian");
  if (partType!="Cartesian" && partType!="CartFlow")
    {
    Tools::Error("Can currently only handle cartesian partitioners",__FILE__,__LINE__);
    }
  
  int dim = newList->sublist("Problem").get("Dimension",-1);
  if (dim==-1) Tools::Error("'Dimension' not set in 'Problem' subist",  
        __FILE__,__LINE__);
  int base_sx=-1,base_sy,base_sz;
  int old_sx=-1,old_sy,old_sz;

  if (newList->sublist("Preconditioner").isParameter("Separator Length (x)"))
    {
    old_sx = newList->sublist("Preconditioner").get("Separator Length (x)",old_sx);
    old_sy = newList->sublist("Preconditioner").get("Separator Length (y)",old_sx);
    old_sz = newList->sublist("Preconditioner").get("Separator Length (z)",dim_>2?old_sx:1);
    }
 else if (newList->sublist("Preconditioner").isParameter("Separator Length"))
    {
    old_sx = newList->sublist("Preconditioner").get("Separator Length",old_sx);
    old_sy = old_sx;
    old_sz = dim>2?old_sx:1;
    }
  else
    {
    Tools::Error("Separator Length not set in list",__FILE__,__LINE__);
    }

  if (newList->sublist("Preconditioner").isParameter("Base Separator Length (x)"))
    {
    base_sx = newList->sublist("Preconditioner").get("Base Separator Length (x)",-1);
    base_sy = newList->sublist("Preconditioner").get("Base Separator Length (y)",base_sx);
    base_sz = newList->sublist("Preconditioner").get("Base Separator Length (z)",dim>2?base_sx:1);
    }
  else if (newList->sublist("Preconditioner").isParameter("Base Separator Length"))
    {
    base_sx = newList->sublist("Preconditioner").get("Base Separator Length",-1);
    base_sy = base_sx;
    base_sz=dim>2?base_sx:1;
    }
    
  if (base_sx == -1) // assume that this is the first level
    {
    base_sx = old_sx;
    base_sy = old_sy;
    base_sz = old_sz;
    newList->sublist("Preconditioner").set("Base Separator Length (x)",base_sx);
    newList->sublist("Preconditioner").set("Base Separator Length (y)",base_sy);
    newList->sublist("Preconditioner").set("Base Separator Length (z)",base_sz);
    }
    
  int new_sx = old_sx*base_sx;
  int new_sy = old_sy*base_sy;
  int new_sz = old_sz*base_sz;
  
  if (newList->sublist("Preconditioner").isParameter("Separator Length (x)"))
    {
    newList->sublist("Preconditioner").set
        ("Separator Length (x)",new_sx);

    newList->sublist("Preconditioner").set
        ("Separator Length (y)",new_sy);

    newList->sublist("Preconditioner").set
        ("Separator Length (z)",new_sz);
    }

  if (newList->sublist("Preconditioner").isParameter("Separator Length"))
    {
    newList->sublist("Preconditioner").set
        ("Separator Length",new_sx);
    }    
  // the next level typically doesn't really resemble a   
  // structured grid anymore, so we base the partitioning 
  // on the matrix graph rather than an idealized graph.  
  // in class OverlappingPartitioner, the matrix graph is 
  // preprocessed to make sure that our separator detec-  
  // tion works correctly.                                
  newList->sublist("Problem").set("Substitute Graph",false);
  
  DEBVAR(*newList);
  
  Teuchos::RCP<const OverlappingPartitioner> newLevel;
  newLevel = Teuchos::rcp(new OverlappingPartitioner(Ared, newList,Level()+1));
  return newLevel;
  }


  // construct a graph with overlap between partitions (for finding/grouping    
  // separators in parallel). We need two levels of overlap because we may      
  // have to include nodes as separators of a subdomain that are not physically 
  // connected to a node in the subdomain (secondary separator nodes like node  
  // (a) in subdomain D in this picture):                                       
  //          :               
  //   C      :       D       
  //         c-d              
  // ........|.|............  
  //         a-b              
  //   A      :       B       
  //          :               
  //                          
  // ... actually we need 3 levels of overlap in 3D.
  Teuchos::RCP<Epetra_CrsGraph> OverlappingPartitioner::CreateParallelGraph()
    {
    START_TIMER2(Label(),"CreateParallelGraph");
    
    if (graph_==Teuchos::null) 
      {
      Tools::Error("graph not yet constructed",__FILE__,__LINE__);
      }
    if (partitioner_->Partitioned()==false)
      {
      Tools::Error("domain not yet partitioned",__FILE__,__LINE__);
      }

    Teuchos::RCP<Epetra_CrsGraph> G=Teuchos::null;

    Epetra_Import importRepart(partitioner_->Map(),graph_->RowMap());

    int MaxNumEntriesPerRow=graph_->MaxNumIndices();
    Teuchos::RCP<Epetra_CrsGraph> G_repart = Teuchos::rcp
        (new Epetra_CrsGraph(Copy,partitioner_->Map(),MaxNumEntriesPerRow,false));
    CHECK_ZERO(G_repart->Import(*graph_,importRepart,Insert));
    CHECK_ZERO(G_repart->FillComplete());
    
    if (Comm().NumProc()==1)
      {
      return G_repart;
      }
    // build a test matrix - we create an overlapping matrix and then extract its graph
    Teuchos::RCP<Epetra_CrsMatrix> Atest = 
        Teuchos::rcp(new Epetra_CrsMatrix(Copy,*G_repart));
//TODO - check how to do this with substituted graph. This is a new implementation because
//       we had seg faults in Trilinos 10.x with using the OverlapGraph directly.
//       Actually, we should kick out the whole 'substituteGraph' idea because it doesn't 
//       work in multi-level mode
    if (substituteGraph_) 
        Tools::Error("graph substitution needs fix", __FILE__,__LINE__);
    CHECK_ZERO(Atest->Import(*matrix_,importRepart,Insert));
    CHECK_ZERO(Atest->FillComplete());

    Ifpack_OverlappingRowMatrix Aov(Atest, dim_);
    Teuchos::RCP<const Epetra_Map> overlappingMap = 
        Teuchos::rcp(&(Aov.RowMatrixRowMap()),false);
    Teuchos::RCP<Epetra_Import> importOverlap =
      Teuchos::rcp(new Epetra_Import(*overlappingMap, graph_->RowMap()));

    G = Teuchos::rcp(new Epetra_CrsGraph
      (Copy,*overlappingMap,graph_->MaxNumIndices(),false));

  CHECK_ZERO(G->Import(*graph_,*importOverlap,Insert));
  CHECK_ZERO(G->FillComplete());
    
    return G;
  }
  

int OverlappingPartitioner::DumpGraph() const
  {
  START_TIMER2(Label(),"DumpGraph");
  std::string filename="matrixGraph"+Teuchos::toString(Level())+".txt";
  
  Teuchos::RCP<Epetra_CrsMatrix> graph=
       Teuchos::rcp(new Epetra_CrsMatrix(Copy,*graph_));
  graph->PutScalar(1.0);
  MatrixUtils::Dump(*graph,filename);
  return 0;
  }

#ifdef DEBUGGING
std::ostream& OverlappingPartitioner::PrintNodeTypeVector
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
  int min_gid = nT.Map().MinMyGID();
  int max_gid = nT.Map().MaxMyGID();
  Tools::ind2sub(nx_,ny_,nz_,dof_,min_gid,imin,jmin,kmin,vmin);
  Tools::ind2sub(nx_,ny_,nz_,dof_,max_gid,imax,jmax,kmax,vmax);
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
            os << setw(4)<<gid<<" ";
            }
          else
            {
            os << "    " << " ";
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
            os << nT[lid]<<" ";
            }
          else
            {
            os << " " << " ";
            }
          }
        os << std::endl;
        }
      }
    }
  return os;
  }
#endif
}//namespace
