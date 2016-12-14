#include <mpi.h>
#include <iostream>
#include <algorithm>

#include "HYMLS_OverlappingPartitioner.H"
#include "HYMLS_Tools.H"

#include "HYMLS_Tester.H"

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

#include "GaleriExt_Periodic.h"

#include "HYMLS_MatrixUtils.H"

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#endif

#include "Teuchos_StandardParameterEntryValidators.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

//#undef DEBUG
//#define HYMLS_DEBUG(s) std::cerr << s << std::endl;
//#undef HYMLS_DEBVAR
//#define HYMLS_DEBVAR(s) std::cerr << #s << " = "<< s << std::endl;

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

  CHECK_ZERO(this->Partition());

  CHECK_ZERO(this->DetectSeparators());
  HYMLS_DEBVAR(*this);
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

  if (validateParameters_)
    {
    this->getValidParameters();
    PL().validateParameters(VPL());
    }

  HYMLS_DEBVAR(PL());
  }

Teuchos::RCP<const Teuchos::ParameterList> OverlappingPartitioner::getValidParameters() const
  {
  if (validParams_!=Teuchos::null) return validParams_;
  HYMLS_PROF3(Label(),"getValidParameters");
#ifdef HYMLS_TESTING
  VPL().set("Test F-Matrix Properties",false,"do special tests for F-matrices in HYMLS_TESTING mode.");
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
    partitioner_=Teuchos::rcp(new CartesianPartitioner(GetMap(), nx_, ny_, nz_,
        dof_, pvar_, perio_));
    }
  else
    {
    Tools::Error("Up to now we only support Cartesian partitioning",__FILE__,__LINE__);
    }

  Teuchos::ParameterList& solverParams=PL("Preconditioner");
  HYMLS_DEBVAR(PL("Preconditioner"));
  int npx,npy,npz;
  sx_ = -1;
  if (solverParams.isParameter("Separator Length (x)"))
    {
    sx_=solverParams.get("Separator Length (x)",4);
    sy_=solverParams.get("Separator Length (y)",sx_);
    sz_=solverParams.get("Separator Length (z)",nz_>1?sx_:1);
    }
  else if (solverParams.isParameter("Separator Length"))
    {
    sx_=solverParams.get("Separator Length",4);
    sy_=sx_;
    sz_=nz_>1 ? sx_ : 1;
    }
  else
    {
    Tools::Error("Separator Length not set",__FILE__,__LINE__);
    }

  if (sx_>0)
    {
    npx=(nx_>1) ? nx_/sx_ : 1;
    npy=(ny_>1) ? ny_/sy_ : 1;
    npz=(nz_>1) ? nz_/sz_ : 1;
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

  Teuchos::RCP<CartesianPartitioner> cartPart
        = Teuchos::rcp_dynamic_cast<CartesianPartitioner>(partitioner_);

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

#ifdef HYMLS_DEBUGGING__disabled_
HYMLS_DEBUG("Partition numbers:");
for (int i=0;i<Map().NumMyElements();i++)
  {
  int gid=Map().GID(i);
  HYMLS_DEBUG(gid << " " << (*partitioner_)(gid));
  }
#endif

  // add the subdomains to the base class so we can start inserting groups of nodes
  Reset(partitioner_->NumLocalParts());

  return 0;
  }

int OverlappingPartitioner::DetectSeparators()
  {
  HYMLS_PROF2(Label(),"DetectSeparators");

  // nodes to be eliminated exactly in the next step
  Teuchos::Array<int> interior_nodes;
  // separator nodes
  Teuchos::Array<Teuchos::Array<int> > separator_nodes;
  // presure nodes that need to be retained
  Teuchos::Array<int> retained_nodes;

  for (int sd = 0; sd < partitioner_->NumLocalParts(); sd++)
    {
    interior_nodes.resize(0);
    separator_nodes.resize(0);
    retained_nodes.resize(0);

    partitioner_->GetGroups(sd, interior_nodes, separator_nodes);

    AddGroup(sd, interior_nodes);
    for (int i = 0; i < separator_nodes.size(); i++)
      {
      if (separator_nodes[i].size() > 0)
        {
        AddGroup(sd, separator_nodes[i]);
        }
      }
    }

  // and rebuild map and global groupPointer
  CHECK_ZERO(FillComplete());
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
    cx = sx_;
    cy = sy_;
    cz = sz_;
    newList->sublist("Preconditioner").set("Coarsening Factor (x)", cx);
    newList->sublist("Preconditioner").set("Coarsening Factor (y)", cy);
    newList->sublist("Preconditioner").set("Coarsening Factor (z)", cz);
    }

  int new_sx = sx_*cx;
  int new_sy = sy_*cy;
  int new_sz = sz_*cz;

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

  HYMLS_DEBVAR(*newList);

  Teuchos::RCP<const OverlappingPartitioner> newLevel;
  newLevel = Teuchos::rcp(new OverlappingPartitioner(Ared, newList,Level()+1));
  return newLevel;
  }

}//namespace

