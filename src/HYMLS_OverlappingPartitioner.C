#include <mpi.h>
#include <iostream>
#include <algorithm>

#include "HYMLS_OverlappingPartitioner.H"
#include "HYMLS_Tools.H"

#include "HYMLS_Tester.H"

#include "HYMLS_BaseCartesianPartitioner.H"
#include "HYMLS_CartesianPartitioner.H"

#include "HYMLS_StandardNodeClassifier.H"
#include "HYMLS_CartesianStokesClassifier.H"

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

#include "GaleriExt_Periodic.h"

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
    HYMLS_PROF3(Label(),"Constructor");

    setParameterList(params);

    // check that this is an F-matrix. Note that the actual test is quite expensive,
    // but it is only performed if -DTESTING is defined.
    Teuchos::RCP<const Epetra_CrsMatrix> Kcrs =
        Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(K);
    if (Kcrs!=Teuchos::null)
      {
      HYMLS_TEST(Label(),isFmatrix(*Kcrs,dof_,pvar_),__FILE__,__LINE__);
      }

    //TODO - partitioning before creating the graph is
    //       kind of not so nice, it just works because
    //       we use the cartesian partitioner, which does
    //       not need a graph.

    CHECK_ZERO(this->Partition());

    // construct a graph with overlap between partitions (for finding/grouping
    // separators in parallel).
    CHECK_ZERO(CreateGraph());

    // pass the graph to the cartesian partitioner so that the flow() function
    // works. TODO - see comment above, we should first make the graph and then
    // partition.
    Teuchos::RCP<CartesianPartitioner> cartPart =
        Teuchos::rcp_dynamic_cast<CartesianPartitioner>(partitioner_);

    if (cartPart!=Teuchos::null)
      {
      cartPart->SetGraph(p_graph_);
      cartPart->SetPressureVariable(pvar_);
      }

    //DEBVAR(*p_graph_);
    int nzgraph = p_graph_->NumMyNonzeros();
    REPORT_SUM_MEM(Label(),"graph with overlap",0,nzgraph,GetComm());

#ifdef STORE_MATRICES
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
    HYMLS_PROF3(Label(),"Destructor");
    }



void OverlappingPartitioner::setParameterList
        (const Teuchos::RCP<Teuchos::ParameterList>& params)
  {
  HYMLS_PROF3(Label(),"setParameterList");
  
  setMyParamList(params);

  dim_=PL().get("Dimension",2);

  dof_=PL().get("Degrees of Freedom",1);

  perio_=PL().get("Periodicity",GaleriExt::NO_PERIO);

  variableType_.resize(dof_);
  retainIsolated_.resize(dof_);

  for (int i=0;i<dof_;i++)
    {
    Teuchos::ParameterList& varList=PL().sublist("Variable "+Teuchos::toString(i));
    variableType_[i]=varList.get("Variable Type","Laplace");
    retainIsolated_[i]=varList.get("Retain Isolated",false);
    }

  pvar_=-1; int pcount=0;
  for (int i=0;i<dof_;i++)
    {
    DEBVAR(variableType_[i]);
    if (retainIsolated_[i]) {pvar_=i; pcount++;}
    }
  if (pcount>1)
    {
    Tools::Error("can only have one 'Retain Isolated' variable",
        __FILE__,__LINE__);
    }
  if (pvar_>=0 && pvar_!=dim_)
    {
    Tools::Warning("we require a certain ordering, u/v[/w]/p/...\n"
                   "(although it is not certain where, but you may\n"
                   "get problems)",__FILE__,__LINE__);
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
  classificationMethod_=PL("Preconditioner").get("Classifier","Standard");

  // TODO: the Stokes classifier gives a far more efficient algrithm, but unfortunately
  //       it does not yield grid-independent convergence unless the standard classifier
  //       is used starting from level 2. It is not clear at this point wether this is
  //       an algorithmic problem or a bug.
  if (classificationMethod_=="Hybrid")
    {
    if (Level()==1)
      {
      classificationMethod_="Stokes";
      }
    else
      {
      classificationMethod_="Standard";
      }
    }

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
  HYMLS_PROF3(Label(),"getValidParameters");
#ifdef TESTING
  VPL().set("Test F-Matrix Properties",false,"do special tests for F-matrices in TESTING mode.");
#endif
  VPL().set("Dimension",2,"physical dimension of the problem");
  VPL().set("Degrees of Freedom",1,"number of unknowns per node");

  VPL().set("Periodicity",GaleriExt::NO_PERIO,"does the problem have periodic BC?"
        " (flag constructed by Preconditioner object)");

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
                    Teuchos::tuple<std::string>("Cartesian"),"Partitioner"));

    VPL("Preconditioner").set("Partitioner", "Cartesian",
        "Type of partitioner to be used to define the subdomains",
        partValidator);

  return validParams_;
  }


int OverlappingPartitioner::Partition()
  {
  HYMLS_PROF2(Label(),"Partition");
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
  int sy,sz;
  if (solverParams.isParameter("Separator Length (x)"))
    {
    sx=solverParams.get("Separator Length (x)",4);
    sy=solverParams.get("Separator Length (y)",sx);
    sz=solverParams.get("Separator Length (z)",nz_>1?sx:1);
    }
  else if (solverParams.isParameter("Separator Length"))
    {
    sx=solverParams.get("Separator Length",4);
    sy=sx;
    sz=nz_>1 ? sx : 1;
    }
  else
    {
    Tools::Error("Separator Length not set",__FILE__,__LINE__);
    }

  if (sx>0)
    {
    npx=(nx_>1) ? nx_/sx : 1;
    npy=(ny_>1) ? ny_/sy : 1;
    npz=(nz_>1) ? nz_/sz : 1;
    }
  else
    {
    int numGlobalSubdomains=solverParams.get("Number of Subdomains",
                4*Comm().NumProc());
    Tools::SplitBox(nx_,ny_,nz_,numGlobalSubdomains,npx,npy,npz);
    }

  // npX==0 can occur on the last level
  // for some reason, in that case we
  // simply set it to 1
  npx=std::max(npx,1);
  npy=std::max(npy,1);
  npz=std::max(npz,1);

  Teuchos::RCP<BaseCartesianPartitioner> cartPart
        = Teuchos::rcp_dynamic_cast<BaseCartesianPartitioner>(partitioner_);

  if (cartPart!=Teuchos::null)
    {
    CHECK_ZERO(cartPart->Partition(npx,npy,npz, false));
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

#ifdef DEBUGGING__disabled_
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
  HYMLS_PROF2(Label(),"DetectSeparators");
  //! first we import our original matrix into the ordering defined by the partitioner.

  if (Teuchos::is_null(graph_))
    {
    Tools::Error("Graph not yet constructed!",__FILE__,__LINE__);
    }

  if (Teuchos::is_null(p_graph_))
    {
    Tools::Error("p_graph not yet constructed!",__FILE__,__LINE__);
    }

  bool isCartStokes=(partitioningMethod_=="Cartesian" && classificationMethod_=="Stokes");
  if (isCartStokes)
    {
    classifier_=Teuchos::rcp(new CartesianStokesClassifier
        (p_graph_,partitioner_,variableType_,retainIsolated_,
        perio_,dim_,Level(),nx_,ny_,nz_));
#ifdef TESTING
    // for comparison purposes - create the standard object so it
    // writes its nodeType vector to file in TESTING mode
    Teuchos::RCP<StandardNodeClassifier> tmp=Teuchos::rcp(new StandardNodeClassifier
        (p_graph_,partitioner_,variableType_,retainIsolated_,
        Level(),nx_,ny_,nz_));
    CHECK_ZERO(tmp->BuildNodeTypeVector());
#endif
    }
  else
    {
    classifier_=Teuchos::rcp(new StandardNodeClassifier
        (p_graph_,partitioner_,variableType_,retainIsolated_,
        Level(),nx_,ny_,nz_));
    }

  CHECK_ZERO(classifier_->BuildNodeTypeVector());
  nodeType_ = classifier_->GetVector();
  p_nodeType_ = classifier_->GetOverlappingVector();

  if (dim_==3 && classificationMethod_=="Stokes")
    {
    HYMLS_TEST(Label(),areTubesCorrect(*Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_),*p_nodeType_,dof_,pvar_),__FILE__,__LINE__);
    }

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
    CHECK_ZERO(this->BuildNodeLists(sd,*p_graph_, *nodeType_, *p_nodeType_,
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
  HYMLS_PROF3(Label(),"BuildNodeLists");

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
    // check for non-local separators of the subdomain
    if (nodeType[i]<=0)
      {
      // add any separator nodes on adjacent subdomains
      CHECK_ZERO(G.ExtractGlobalRowCopy(row,MaxNumEntriesPerRow,len,cols));
      for (int j=0;j<len;j++)
        {
        int sd_j = (*partitioner_)(cols[j]);
        int nt_j = (*p_nodeType_)[p_map.LID(cols[j])];
        if ((sd_j!=sd_i)&&nt_j>0)
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
    if (nodeType[i]<=0)
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

  std::set<int> tmp;

  tmp.insert(separator.begin(),separator.end());
  tmp.insert(retained.begin(),retained.end());

  separator.clear();
  retained.clear();

  for (std::set<int>::iterator i=tmp.begin();i!=tmp.end();i++)
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

// reorders the separators found in DtectSeparators()
// into groups suitable for our transformations.
int OverlappingPartitioner::GroupSeparators()
  {
  HYMLS_PROF2(Label(),"GroupSeparators");
  if (Teuchos::is_null(graph_))
    {
    Tools::Error("Graph not yet constructed!",__FILE__,__LINE__);
    }
  if (Teuchos::is_null(p_graph_))
    {
    Tools::Error("overlapping Graph not yet constructed!",__FILE__,__LINE__);
    }
  if (!Filled())
    {
    Tools::Error("Separators not yet detected!",__FILE__,__LINE__);
    }

  const Epetra_BlockMap& p_map = p_nodeType_->Map();

  // offset for creating new partition numbers in corners (2D) and edges
  // (3D) Stokes problems. Singleton subdomains start at offset, FCTs at
  // 2*offset.
  int offset = partitioner_->NumGlobalParts();

  // copy old data structures and reset base class (HierarchicalMap)
  Teuchos::RCP<Teuchos::Array<Teuchos::Array<int> > > groupPointer
        = this->GetGroupPointer();

  Teuchos::RCP<const Epetra_Map> overlappingMap =
        this->GetOverlappingMap();

  this->Reset(partitioner_->NumLocalParts());

  DEBUG("build separator lists...");

  int MaxNumElements=p_graph_->MaxNumIndices();

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
    int new_id = offset;
    for (int i=0;i<numSepNodes;i++)
      {
      int row=overlappingMap->GID((*groupPointer)[sd][1]+i);
      int type_i = (*p_nodeType_)[p_map.LID(row)];
      int var_i=partitioner_->VariableType(row);

      //DEBUG("Process node "<<row);
      connectedSubs.resize(1);
      connectedSubs[0]=(*partitioner_)(row);
      //CHECK_ZERO(p_graph_->ExtractGlobalRowCopy(row,MaxNumElements,len,cols));
      int ierr=p_graph_->ExtractGlobalRowCopy(row,MaxNumElements,len,cols);
      if (ierr!=0)
        {
        Tools::Error("extracting global row "+Teuchos::toString(row)+
                " failed on rank "+Teuchos::toString(Comm().MyPID()),__FILE__,__LINE__);
        }
      for (int j=0;j<len;j++)
        {
        // We only consider edges to lower-level nodes here,
        // e.g. from face separators to interior, from
        // edges to faces and from vertices to edges. We treat
        // subcells (full conservation tubes in Stokes) as separate
        // subdomains.
        int type_j=(*p_nodeType_)[p_map.LID(cols[j])];
        int var_j=partitioner_->VariableType(cols[j]);

        if (type_j<0 && var_i!=var_j)
          {
          // we have to create a 'new subdomain id' as the separator
          // node connects to the interior of a full conservation tube.
          //DEBUG("connected to FCT");
          int sd_id = (*partitioner_)(cols[j])
                    + (-type_j+1)*offset;
          //DEBUG("\t"<<cols[j]<<" ["<<type_j<<"], gives sd="<<sd_id);
          // flow may be 0 because the subcell is on the
          // same subdomain, I hope this is correct without
          // the sign as well (may fail in rare cases like
          // a single subdomain with periodic BC)
          connectedSubs.append(sd_id);
          }
        else if (type_j<type_i)
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
            int sign = flow/std::abs(flow);
            int sd_id = (*partitioner_)(cols[j]);
            DEBUG("\t"<<cols[j]<<" "<<sd_id);
            connectedSubs.append(sign*sd_id);
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

      //int variableType=var_i;
      int variableType= type_i*10 + var_i;

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
  HYMLS_PROF3(Label(),"SpawnNextLevel");

  *newList = *getMyParamList();

  std::string partType=newList->sublist("Preconditioner").get("Partitioner","Cartesian");
  if (partType!="Cartesian")
    {
    Tools::Error("Can currently only handle cartesian partitioners",__FILE__,__LINE__);
    }

  int dim = newList->sublist("Problem").get("Dimension",-1);
  if (dim==-1) Tools::Error("'Dimension' not set in 'Problem' subist",
        __FILE__,__LINE__);
  int cx=-1,cy,cz;
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

  // "Base Separator Length" is deprecated and "Coarsening Factor" should be
  // used instead, since it explains better what it does. "Base Separator Length"
  // is still here for backward compatibility
  if (newList->sublist("Preconditioner").isParameter("Base Separator Length (x)"))
    {
    cx = newList->sublist("Preconditioner").get("Base Separator Length (x)", -1);
    cy = newList->sublist("Preconditioner").get("Base Separator Length (y)", cx);
    cz = newList->sublist("Preconditioner").get("Base Separator Length (z)", dim>2 ? cx : 1);
    }
  else if (newList->sublist("Preconditioner").isParameter("Base Separator Length"))
    {
    cx = newList->sublist("Preconditioner").get("Base Separator Length", -1);
    cy = cx;
    cz = dim>2 ? cx : 1;
    }
  else if (newList->sublist("Preconditioner").isParameter("Coarsening Factor (x)"))
    {
    cx = newList->sublist("Preconditioner").get("Coarsening Factor (x)", -1);
    cy = newList->sublist("Preconditioner").get("Coarsening Factor (y)", cx);
    cz = newList->sublist("Preconditioner").get("Coarsening Factor (z)", dim>2 ? cx : 1);
    }
  else if (newList->sublist("Preconditioner").isParameter("Coarsening Factor"))
    {
    cx = newList->sublist("Preconditioner").get("Coarsening Factor", -1);
    cy = cx;
    cz = dim>2 ? cx : 1;
    }

  // Set the coarsening factor to be the same as the separator size
  if (cx == -1) // assume that this is the first level
    {
    cx = old_sx;
    cy = old_sy;
    cz = old_sz;
    newList->sublist("Preconditioner").set("Coarsening Factor (x)", cx);
    newList->sublist("Preconditioner").set("Coarsening Factor (y)", cy);
    newList->sublist("Preconditioner").set("Coarsening Factor (z)", cz);
    }

  int new_sx = old_sx*cx;
  int new_sy = old_sy*cy;
  int new_sz = old_sz*cz;

  if (newList->sublist("Preconditioner").isParameter("Separator Length (x)"))
    {
    newList->sublist("Preconditioner").set("Separator Length (x)", new_sx);
    newList->sublist("Preconditioner").set("Separator Length (y)", new_sy);
    newList->sublist("Preconditioner").set("Separator Length (z)", new_sz);
    }

  if (newList->sublist("Preconditioner").isParameter("Separator Length"))
    {
    newList->sublist("Preconditioner").set("Separator Length", new_sx);
    }

  DEBVAR(*newList);

  Teuchos::RCP<const OverlappingPartitioner> newLevel;
  newLevel = Teuchos::rcp(new OverlappingPartitioner(Ared, newList,Level()+1));
  return newLevel;
  }

  // constructs a graph suitable for the partitioning process, for instance
  // [A+BB' B; B' 0] for saddlepoint matrices (graph_), then
  // construct a graph with overlap between partitions (for finding/grouping
  // separators in parallel) (p_graph_).
  //
  // We need two levels of overlap because we may
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
  // ... TODO - check wether this is still true, even with the graph of [A+BB' B; B' 0]
  //     that we use now.
  int OverlappingPartitioner::CreateGraph()
    {
    HYMLS_PROF2(Label(),"CreateGraph");

    Teuchos::RCP<const Epetra_CrsMatrix> myCrsMatrix =
        Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);

    if (Teuchos::is_null(myCrsMatrix))
      {
      Tools::Error("we need a CrsMatrix here!",__FILE__,__LINE__);
      }
    // copy the graph
    graph_=Teuchos::rcp(&(myCrsMatrix->Graph()),false);

    HYMLS_TEST(Label(),noNumericalZeros(*myCrsMatrix),__FILE__,__LINE__);

    if (partitioner_->Partitioned()==false)
      {
      Tools::Error("domain not yet partitioned",__FILE__,__LINE__);
      }

   // repartition...
   Teuchos::RCP<Epetra_Import> importRepart =
     Teuchos::rcp(new Epetra_Import(partitioner_->Map(),graph_->RowMap()));

    int MaxNumEntriesPerRow=graph_->MaxNumIndices();
    Teuchos::RCP<Epetra_CrsGraph> G_repart = Teuchos::rcp
        (new Epetra_CrsGraph(Copy,partitioner_->Map(),MaxNumEntriesPerRow,false));
    CHECK_ZERO(G_repart->Import(*graph_,*importRepart,Insert));
    CHECK_ZERO(G_repart->FillComplete());

    // parallel graph setup
    p_graph_=Teuchos::null;


    if (Comm().NumProc()==1)
      {
      p_graph_=G_repart;
      importOverlap_=importRepart;
      }
    else
      {
      // build a test matrix - we create an overlapping matrix and then extract its graph
      Teuchos::RCP<Epetra_CrsMatrix> Atest =
          Teuchos::rcp(new Epetra_CrsMatrix(Copy,*G_repart));

      CHECK_ZERO(Atest->Import(*matrix_,*importRepart,Insert));
      CHECK_ZERO(Atest->FillComplete());
      //the original graph of the matrix is also distributed
      Ifpack_OverlappingRowMatrix Aov(Atest, dim_);
      Teuchos::RCP<const Epetra_Map> overlappingMap =
        Teuchos::rcp(&(Aov.RowMatrixRowMap()),false);

      importOverlap_ =
      Teuchos::rcp(new Epetra_Import(*overlappingMap, graph_->RowMap()));
      //importOverlap contains all information needed for MPI.
      p_graph_ = Teuchos::rcp(new Epetra_CrsGraph
        (Copy,*overlappingMap,graph_->MaxNumIndices(),false));
      //Definition of the Graph but it is still  empty
      //below it is filled

      CHECK_ZERO(p_graph_->Import(*graph_,*importOverlap_,Insert));
      CHECK_ZERO(p_graph_->FillComplete());// cleans everything up (removes workarrays).
      }

    if (pvar_>=0 && false) // TODO this section is disabled for now
      {
      DEBVAR(pvar_);
      // given A  B, form  C  B, with C = A + BB'
      //       B' 0        B' 0
      CHECK_ZERO(AugmentSppGraph(pvar_));
      }

  return 0;
  }


int OverlappingPartitioner::DumpGraph() const
  {
  HYMLS_PROF2(Label(),"DumpGraph");
  std::string filename="matrixGraph"+Teuchos::toString(Level())+".txt";

  Teuchos::RCP<Epetra_CrsMatrix> graph=
       Teuchos::rcp(new Epetra_CrsMatrix(Copy,*graph_));
  graph->PutScalar(1.0);
  MatrixUtils::Dump(*graph,filename);
  return 0;
  }


  //! If graph_/p_graph_ represent a saddlepoint matrix K=[A G; D 0], adds edges
  //! so that the pattern is that of [A+G*D G; D 0]. Replaces graph_ and p_graph_,
  //! which should both already be Filled().
  int OverlappingPartitioner::AugmentSppGraph(int pvar)
  {
  HYMLS_PROF2(Label(),"AugmentSppGraph");

  DEBVAR(pvar);
  DEBVAR(pvar_);
  DEBVAR(*p_graph_);

  if (!p_graph_->Filled()) Tools::Error("AugmentSppGraph() requires p_graph_ to be filled",
        __FILE__,__LINE__);

  if (!graph_->Filled()) Tools::Error("AugmentSppGraph() requires graph_ to be filled",
        __FILE__,__LINE__);

  int max_len = p_graph_->MaxNumIndices()*4;
  Teuchos::Array<int> cols;
  int len,lenK, lenD;
  int *colsD;

  Teuchos::RCP<Epetra_CrsGraph> graph = Teuchos::rcp(
        new Epetra_CrsGraph(Copy, graph_->RowMap(),graph_->MaxNumIndices()));


  for (int i=0;i<graph_->NumMyRows();i++)
    {
    cols.resize(max_len);
    int grid = graph_->GRID(i);
    CHECK_ZERO(graph_->ExtractGlobalRowCopy(grid,max_len,lenK,cols.getRawPtr()));
    len=lenK;
    if (MOD(grid,dof_)!=pvar && MOD(grid,dof_)<pvar_)
      {

      // append B'B (Grad*Div)

      // walk through row of Grad
      for (int j=0;j<lenK;j++)
        {
        if (MOD(cols[j],dof_)==pvar) // entry of Grad (=B)
          {
          // walk through row of Div to see which rows ii,jj of Grad have entries in common
          // (that's where an entry ii,jj has to be added)
          int lridD = p_graph_->LRID(cols[j]);
          DEBVAR(lridD);
          if (lridD>=0)
            {
            CHECK_ZERO(p_graph_->ExtractMyRowView(lridD,lenD,colsD));
            for (int jj=0;jj<lenD;jj++)
              {
              if (MOD(p_graph_->GCID(colsD[jj]),dof_)!=pvar_ &&
                  MOD(p_graph_->GCID(colsD[jj]),dof_)!=pvar)
                {
                if (len>=max_len)
                  {
                  max_len*=4;
                  cols.resize(max_len);
                  }
                cols[len++]=p_graph_->GCID(colsD[jj]);
                }
              }
            }
#ifdef TESTING
          else
            {
            std::stringstream msg;
            msg << "rank "<<p_graph_->Comm().MyPID()<<std::endl;
            msg << "current row (V-node): "<<grid<<std::endl;
            msg << "column in G: "<<cols[j] << std::endl;
            msg << "Row not present in K on this proc.\n";
            msg << "Case not handled, your graph should have overlap.\n";
            msg << "If you compile with -DDEBUGGING, the bad graph is stored in \n";
            msg << "debug*.txt, after the keyword AugmentSppGraph.\n";
            DEBVAR(*p_graph_);
            Tools::Error(msg.str(),__FILE__,__LINE__);
            }
#endif
          }
        }
      std::sort(cols.begin(),cols.begin()+len);
      int_i cols_end = std::unique(cols.begin(),cols.begin()+len);
      len = std::distance(cols.begin(),cols_end);

#ifdef DEBUGGING_
      DEBVAR(grid);
      for (int j=0;j<len;j++)
        {
        Tools::deb() << cols[j] << " ";
        }
      DEBUG("");
#endif
      }
    else
      {
      // copy in the div-row unchanged.
      }
    CHECK_ZERO(graph->InsertGlobalIndices(grid,len,cols.getRawPtr()));
    }

  DEBUG("call FillComplete on augmented graph");
  CHECK_ZERO(graph->FillComplete());
  graph_=graph;

  p_graph_ = Teuchos::rcp(new Epetra_CrsGraph
      (Copy,p_graph_->RowMap(),graph_->MaxNumIndices(),false));

  CHECK_ZERO(p_graph_->Import(*graph_,*importOverlap_,Insert));
  CHECK_ZERO(p_graph_->FillComplete());
  return 0;
  }

}//namespace

