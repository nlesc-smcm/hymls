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
      : RecursiveOverlappingPartitioner(Teuchos::rcp(&(K->Comm()),false),
                                        Teuchos::rcp(&(K->RowMatrixRowMap()),false),
                                        Teuchos::null,
                                        Teuchos::rcp(new Teuchos::Array< Teuchos::Array<int> >()),
                                        "OverlappingPartitioner", level),
        PLA("Problem"), matrix_(K)
    {    
    START_TIMER(label_,"Constructor");

    setParameterList(params);
    
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

    Partition();

    // construct a graph with overlap between partitions (for finding/grouping    
    // separators in parallel).
    parallelGraph_=CreateParallelGraph();
        
    //TODO: the algorithm works a lot on the graph of the matrix,
    //      and we do a lot of ExtractGlobalRowCopy() operations,
    //      which copy the data all the time. We might consider  
    //      creating a global copy (or view) once and then just  
    //      looking up the entries (to make it faster).


#ifdef DEBUGGING
  this->DumpGraph();
#endif
    
    DetectSeparators();
    
    GroupSeparators();
            
    }
    
  OverlappingPartitioner::~OverlappingPartitioner()
    {
    START_TIMER3(label_,"Destructor");
    }



void OverlappingPartitioner::setParameterList
        (const Teuchos::RCP<Teuchos::ParameterList>& params)
  {
  START_TIMER3(label_,"setParameterList");
  
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
  START_TIMER3(label_,"getValidParameters");

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
  Teuchos::RCP<Teuchos::StringToIntegralParameterEntryValidator<int> >
        partValidator = Teuchos::rcp(
                new Teuchos::StringToIntegralParameterEntryValidator<int>(
                    Teuchos::tuple<std::string>("Cartesian"),"Partitioner"));
    
    VPL("Preconditioner").set("Partitioner", "Cartesian",
        "Type of partitioner to be used to define the subdomains",
        partValidator);

  return validParams_;
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
                4*comm_->NumProc());
    Tools::SplitBox(nx_,ny_,nz_,numGlobalSubdomains,npx,npy,npz);
    }

  
  Teuchos::RCP<CartesianPartitioner> cartPart
        = Teuchos::rcp_dynamic_cast<CartesianPartitioner>(partitioner_);
        
  if (cartPart!=Teuchos::null) 
    {
    cartPart->Partition(npx,npy,npz);
    cartPart->SetNodeDistance((double)(sx/base_sx));
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
#ifdef TESTING
  int rank = comm_->MyPID();
  if (partitioner_==Teuchos::null) Tools::Error("partitioner zero???",__FILE__,__LINE__);
  DEBVAR(partitioner_->Partitioned());
  DEBVAR(partitioner_->Map());
  Epetra_Import test_importer(*baseMap_,partitioner_->Map());
  if ((test_importer.NumSend()!=0) || (test_importer.NumRecv()!=0))
    {
    Tools::Warning("your matrix is being repartitioned. This feature is buggy\n"
                   " and may infringe the performance of the solver.\n"
                   " Warning issued by rank "+Teuchos::toString(rank),__FILE__,__LINE__);
    DEBVAR(test_importer);
    }
#endif
  return;
  }
  
void OverlappingPartitioner::DetectSeparators()
  {
  START_TIMER2(label_,"DetectSeparators");
  //! first we import our original matrix into the ordering defined by the partitioner.

  if (Teuchos::is_null(graph_))
    {
    Tools::Error("Graph not yet constructed!",__FILE__,__LINE__);
    }

  if (Teuchos::is_null(parallelGraph_))
    {
    Tools::Error("parallelGraph not yet constructed!",__FILE__,__LINE__);
    }
  
  int np_schur=0;
  nodeType_=Teuchos::rcp(new Epetra_IntVector(partitioner_->Map()));
  nodeType_->PutValue(-1);

  this->BuildInitialNodeTypeVector(*parallelGraph_,*nodeType_,np_schur);

  // now every subdomain has marked those nodes it owns and 
  // that should become interior, separator or retained nodes.

  // import partition overlap
  p_nodeType_=Teuchos::rcp(new Epetra_IntVector(parallelGraph_->RowMap()));
  Epetra_Import import(p_nodeType_->Map(),nodeType_->Map());
  CHECK_ZERO(p_nodeType_->Import(*nodeType_,import,Insert));

DEBVAR(*p_nodeType_);

  // increase the node type of nodes that only connect to separators.         
  // A type 1 sep node that only connects to sep nodes becomes a type         
  // 2 sep node. An interior (type 0) node that only connects to sep          
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
  this->DetectIsolated(*parallelGraph_,*p_nodeType_, *nodeType_,np_schur);

  // import again to "spread the word"
  CHECK_ZERO(p_nodeType_->Import(*nodeType_,import,Insert));

  //... then do it again: this is required to get a 3 in the corners in 3D
  this->DetectIsolated(*parallelGraph_,*p_nodeType_, *nodeType_,np_schur);

  // import again to "spread the word"
  CHECK_ZERO(p_nodeType_->Import(*nodeType_,import,Insert));

  DEBVAR(*p_nodeType_);

  int global_np_schur;
  comm_->SumAll(&np_schur,&global_np_schur,1);

  if (comm_->MyPID()==0 && global_np_schur>0)
    {
    Tools::Out("Number of retained pressures: "+Teuchos::toString(global_np_schur));
    }
  
  // put them into lists and form GroupPointer

  // all nodes - for building the map
  Teuchos::Array<int> my_nodes;
 
  // nodes to be eliminated exactly in the next step  
  Teuchos::Array<int> interior_nodes;
  // separator nodes
  Teuchos::Array<int> separator_nodes;
  // nodes to be retained in the Schur complement (typically pressures)
  Teuchos::Array<int> retained_nodes;

  groupPointer_->resize(partitioner_->NumLocalParts());
  int last;
  (*groupPointer_)[0].append(0);

  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    if (sd>0)
      {
      last=(*groupPointer_)[sd-1].size()-1;
      (*groupPointer_)[sd].append((*groupPointer_)[sd-1][last]);
      }

    interior_nodes.resize(0);
    separator_nodes.resize(0);
    retained_nodes.resize(0);
    this->BuildNodeLists(sd,*parallelGraph_, *nodeType_, *p_nodeType_,
                interior_nodes, separator_nodes, retained_nodes);

    DEBVAR(sd);
    DEBVAR(interior_nodes);
    DEBVAR(separator_nodes);
    DEBVAR(retained_nodes);
    last=(*groupPointer_)[sd].size()-1;
    (*groupPointer_)[sd].append((*groupPointer_)[sd][last]+interior_nodes.size());

    last=(*groupPointer_)[sd].size()-1;
    (*groupPointer_)[sd].append((*groupPointer_)[sd][last]+separator_nodes.size());

    last=(*groupPointer_)[sd].size()-1;
    (*groupPointer_)[sd].append((*groupPointer_)[sd][last]+retained_nodes.size());
    
    std::copy(interior_nodes.begin(),interior_nodes.end(),
              std::back_inserter(my_nodes));
    std::copy(separator_nodes.begin(),separator_nodes.end(),
              std::back_inserter(my_nodes));
    std::copy(retained_nodes.begin(),retained_nodes.end(),
              std::back_inserter(my_nodes));
    
    }
  int NumMyElements = my_nodes.length();
  int *MyElements = &(my_nodes[0]);
  
  overlappingMap_=Teuchos::rcp(new  Epetra_Map
        (-1,NumMyElements, MyElements,partitioner_->Map().IndexBase(),*comm_));

  }

  //! build initial vector with 0 (interior), 1 (separator) or 2 (retained)
  void OverlappingPartitioner::BuildInitialNodeTypeVector(
        const Epetra_CrsGraph& G, Epetra_IntVector& nodeType,
        int& np_schur) const
  {
  START_TIMER3(label_,"BuildInitialNodeTypeVector");

  int MaxNumEntriesPerRow=G.MaxNumIndices();

  int *cols = new int[MaxNumEntriesPerRow];
  int len;

  Teuchos::Array<int> retain(dof_);
  Teuchos::Array<int> retained(dof_);

  np_schur=0; // count number of pressures in Schur-complement
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
          nodeType[i]=4;
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
            nodeType[i]=1;
            }
          }
        }// not retained
      }//i
    }//sd
  delete [] cols;
  }

  //! detect isolated interior nodes and mark them 'retain' (3)
  void OverlappingPartitioner::DetectIsolated(
                      const Epetra_CrsGraph& G, 
                      const Epetra_IntVector& p_nodeType,
                            Epetra_IntVector& nodeType,
                            int& np_schur) const
  {
  START_TIMER3(label_,"DetectIsolated");
  
  int MaxNumEntriesPerRow = G.MaxNumIndices();
  
  int *cols = new int[MaxNumEntriesPerRow];
  int len;
 
  const Epetra_BlockMap& map = nodeType.Map();
  const Epetra_BlockMap& p_map = p_nodeType.Map();
#ifdef TESTING
  if (!map.SameAs(partitioner_->Map()))
    {
    Tools::Error("nodeType vector based on wrong map!",__FILE__,__LINE__);
    }
#endif  
  for (int sd=0;sd<partitioner_->NumLocalParts();sd++)
    {
    for (int i=partitioner_->First(sd); i<partitioner_->First(sd+1);i++)
      {
      int row=map.GID(i);
      CHECK_ZERO(G.ExtractGlobalRowCopy(row,MaxNumEntriesPerRow,len,cols));
      int my_type = nodeType[i];
      int var_i = partitioner_->VariableType(row);
      int min_neighbor = 99;
      for (int j=0;j<len;j++)
        {
#ifdef TESTING
        if (p_map.LID(cols[j])==-1)
          {
          std::string msg="parallel map used does not contain all necessary nodes, node "
          +Teuchos::toString(cols[j])+" not found on processor "
          +Teuchos::toString(comm_->MyPID());
          Tools::Error(msg,__FILE__,__LINE__);
          }
#endif
        if (cols[j]!=row)
          {
          min_neighbor=std::min(min_neighbor,p_nodeType[p_map.LID(cols[j])]);
          }
        }//j
      if (my_type==0) 
        {
        if ((min_neighbor>0)&&retainIsolated_[var_i])
          {
          DEBUG(" isolated interior node found: "<<row);
          np_schur++;
          nodeType[i]=4;
          // all surrounding (velocity) nodes
          // are to be retained. As this is a
          // row from the 'Div' part of the  
          // matrix, there are only connec-  
          // tions to velocities.            
          for (int j=0;j<len;j++)
            {
            if (map.MyGID(cols[j]))
              {
              nodeType[map.LID(cols[j])]=4;
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
      else if (min_neighbor>=my_type)
        {
        DEBUG("increase node level of "<<row);
        nodeType[i]=min_neighbor+1;
        }
      }//i
    }//sd
    
  delete [] cols;
  }

  //! form list with interior, separator and retained nodes for subdomain
  // sd. Links separators to subdomains.
  void OverlappingPartitioner::BuildNodeLists(int sd, 
                              const Epetra_CrsGraph& G, 
                              const Epetra_IntVector& nodeType,
                              const Epetra_IntVector& p_nodeType,
                              Teuchos::Array<int>& interior,
                              Teuchos::Array<int>& separator,
                              Teuchos::Array<int>& retained) const
  {
  START_TIMER3(label_,"BuildNodeLists");

  int MaxNumEntriesPerRow = G.MaxNumIndices();
  
  int *cols = new int[MaxNumEntriesPerRow];
  int len;

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
    if (nodeType[i]==0)
      {
      interior.append(row);
      // add any separator nodes on adjacent subdomains
      CHECK_ZERO(G.ExtractGlobalRowCopy(row,MaxNumEntriesPerRow,len,cols));
      for (int j=0;j<len;j++)
        {
        int sd_j = (*partitioner_)(cols[j]);
        int var_j = partitioner_->VariableType(cols[j]);
        if ((sd_j!=sd_i) &&(var_j==var_i) )
          {
          DEBUG("include "<<cols[j]<<" from "<<row);
          if (p_nodeType[p_map.LID(cols[j])]>=4)
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
  this->FindMissingSepNodes(sd_i,G,p_nodeType,separatorL1,separatorL2);
  
  // The D-node in the corner of the 3D subdomain has an edge to an A-
  // and two C-nodes on subdomain 1 (from the perspective of SD2).
  // For all other subdomains it has an edge to two C-nodes on 
  // different subdomains.
  this->FindMissingSepNodes(sd_i,G,p_nodeType,separatorL2,separatorL3);
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
  }

void OverlappingPartitioner::FindMissingSepNodes
        (int my_sd, const Epetra_CrsGraph& G, const Epetra_IntVector& p_nodeType,
         const std::set<int>& in, std::set<int>& out) const

  {  
  START_TIMER3(label_,"FindMissingSepNodes");
  int MaxNumEntriesPerRow = G.MaxNumIndices();
  const Epetra_BlockMap& p_map=p_nodeType.Map();
  int *colsI = new int[MaxNumEntriesPerRow];
  int *colsJ = new int[MaxNumEntriesPerRow];
  int lenI,lenJ;
  for (std::set<int>::const_iterator i=in.begin(); i!=in.end();i++)
    {
    DEBVAR(*i);
    int sd_i = (*partitioner_)(*i);
    int type_i = p_nodeType[p_map.LID(*i)];
    for (std::set<int>::const_iterator j=i; j!=in.end();j++)
      {
      int sd_j = (*partitioner_)(*j);
      int type_j = p_nodeType[p_map.LID(*j)];
//      if (sd_i!=sd_j)
      if (true)
        {
        // a level 1 node can include level 2 nodes, but not level 0
        // or 1 (and the same holds on each level).
        int min_type = std::min(type_i,type_j);
        DEBUG("\t check "<<*i<<" and "<<*j);
        CHECK_ZERO(G.ExtractGlobalRowCopy(*i,MaxNumEntriesPerRow,lenI,colsI));
        CHECK_ZERO(G.ExtractGlobalRowCopy(*j,MaxNumEntriesPerRow,lenJ,colsJ));
        for (int ii=0;ii<lenI;ii++)
          {
          int sd_ii = (*partitioner_)(colsI[ii]);
          int type_ii = p_nodeType[p_map.LID(colsI[ii])];
          if ((sd_ii!=my_sd)&&(type_ii>min_type))
            {
            for (int jj=0;jj<lenJ;jj++)
              {
              if (colsI[ii]==colsJ[jj])
                {
                DEBUG("\t\t missed node "<<colsJ[jj]<<" inserted");
                out.insert(colsJ[jj]);
                }
              }//jj
            }
          }//ii
        }
      }//j
    }//i
  delete [] colsI;
  delete [] colsJ;
  }


void OverlappingPartitioner::GroupSeparators()
  {
  START_TIMER2(label_,"GroupSeparators");
  if (Teuchos::is_null(graph_))
    {
    Tools::Error("Graph not yet constructed!",__FILE__,__LINE__);
    }
  if (Teuchos::is_null(parallelGraph_))
    {
    Tools::Error("overlapping Graph not yet constructed!",__FILE__,__LINE__);
    }
  if (Teuchos::is_null(overlappingMap_))
    {
    Tools::Error("Separators not yet detected!",__FILE__,__LINE__);
    }
  const Epetra_BlockMap& p_map = p_nodeType_->Map();
      
  DEBUG("build separator lists...");

  int MaxNumElements=parallelGraph_->MaxNumIndices();

  int* cols = new int[MaxNumElements];
  int len;

    Teuchos::Array<SepNode> sepNodes;
    
    int NumMyElements = overlappingMap_->NumMyElements();
    int *MyElements=new int[NumMyElements];

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
    int new_id = partitioner_->NumLocalParts()+1;
    for (int i=0;i<numSepNodes;i++)
      {
      int row=overlappingMap_->GID((*groupPointer_)[sd][1]+i);
      int myLevel = (*p_nodeType_)[p_map.LID(row)];

      //DEBUG("Process node "<<i<<", GID "<<row);
      connectedSubs.resize(0);      
      //CHECK_ZERO(parallelGraph_->ExtractGlobalRowCopy(row,MaxNumElements,len,cols));
      int ierr=parallelGraph_->ExtractGlobalRowCopy(row,MaxNumElements,len,cols);
      if (ierr!=0)
        {
        Tools::Error("extracting global row "+Teuchos::toString(row)+
                " failed on rank "+Teuchos::toString(comm_->MyPID()),__FILE__,__LINE__);
        }
      DEBVAR(row);
      DEBVAR(myLevel);
      for (int j=0;j<len;j++)
        {
        // We only consider edges to lower-level nodes here,
        // e.g. from face separators to interior, from 
        // from edges to faces and from vertices to edges.
        if ((*p_nodeType_)[p_map.LID(cols[j])]<myLevel)
          {
          int flow = partitioner_->flow(row,cols[j]);    
          
          //TODO: check if this still makes sense!
          
          // if the row and col are not in the same subdomain, multiply
          // the partition ID by +1 or -1, depending on the "direction of
          // flow" across the separator. If we don't do this, for periodic
          // BC we can get two different separators identified as one, e.g
          //                                                              
          // | SD1 | SD2 |        (here separators s1 and s2 both connect 
          // s1    s2    s1       to subdomains SD1 and SD2)              
          //                                                              
          int sign = flow? flow:1;
          connectedSubs.append(sign*(*partitioner_)(cols[j]));
          }
        }
      if (connectedSubs.length()==0)
        {
        // this is a singleton as they appear in the corners
        // of subdomains (only connected to separators). Give
        // it a unique ID so that it isn't put in the same group
        // as other singletons around the same subdomain.
        connectedSubs.append(new_id++);
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

  }



Teuchos::RCP<const OverlappingPartitioner> OverlappingPartitioner::SpawnNextLevel
        (Teuchos::RCP<const Epetra_RowMatrix> Ared, Teuchos::RCP<Teuchos::ParameterList> newList) const
  {
  START_TIMER3(label_,"SpawnNextLevel");

  *newList = *getMyParamList();

  if (newList->sublist("Preconditioner").get("Partitioner","Cartesian")!="Cartesian")
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
  newLevel = Teuchos::rcp(new OverlappingPartitioner(Ared, newList,myLevel_+1));
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
    START_TIMER2(label_,"CreateParallelGraph");
    
    if (graph_==Teuchos::null) 
      {
      Tools::Error("graph not yet constructed",__FILE__,__LINE__);
      }
    if (partitioner_->Partitioned()==false)
      {
      Tools::Error("domain not yet partitioned",__FILE__,__LINE__);
      }

    Teuchos::RCP<Epetra_CrsGraph> G=Teuchos::null;

    Epetra_Import importRepart(partitioner_->Map(),*baseMap_);

    int MaxNumEntriesPerRow=graph_->MaxNumIndices();
    Teuchos::RCP<Epetra_CrsGraph> G_repart = Teuchos::rcp
        (new Epetra_CrsGraph(Copy,partitioner_->Map(),MaxNumEntriesPerRow,false));
    CHECK_ZERO(G_repart->Import(*graph_,importRepart,Insert));
    CHECK_ZERO(G_repart->FillComplete());
    
    if (comm_->NumProc()==1)
      {
      return G_repart;
      }
    // build a test matrix - we create an overlapping matrix and then extract its graph
    Teuchos::RCP<Epetra_CrsMatrix> Atest = 
        Teuchos::rcp(new Epetra_CrsMatrix(Copy,*G_repart));
//TODO - check how to do this with substituted graph. This is a new implementation because
//       we had seg faults in Trilinos 10.x with using the OverlapGraph directly.
    if (substituteGraph_) 
        Tools::Error("graph substitution needs fix", __FILE__,__LINE__);
    CHECK_ZERO(Atest->Import(*matrix_,importRepart,Insert));
    CHECK_ZERO(Atest->FillComplete());

    Ifpack_OverlappingRowMatrix Aov(Atest, dim_);
    Teuchos::RCP<const Epetra_Map> overlappingMap = 
        Teuchos::rcp(&(Aov.RowMatrixRowMap()),false);
    Teuchos::RCP<Epetra_Import> importOverlap =
      Teuchos::rcp(new Epetra_Import(*overlappingMap, *baseMap_));

    G = Teuchos::rcp(new Epetra_CrsGraph
      (Copy,*overlappingMap,graph_->MaxNumIndices(),false));

  CHECK_ZERO(G->Import(*graph_,*importOverlap,Insert));
  CHECK_ZERO(G->FillComplete());
    
    return G;
  }
  

void OverlappingPartitioner::DumpGraph() const
  {
  START_TIMER2(label_,"DumpGraph");
  std::string filename="matrixGraph"+Teuchos::toString(myLevel_)+".txt";
  
  Teuchos::RCP<Epetra_CrsMatrix> graph=
       Teuchos::rcp(new Epetra_CrsMatrix(Copy,*graph_));
  graph->PutScalar(1.0);
  MatrixUtils::Dump(*graph,filename);
  return;
  }


}//namespace
