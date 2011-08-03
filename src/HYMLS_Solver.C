//#define BLOCK_IMPLEMENTATION 1
#include "HYMLS_Solver.H"

#include "HYMLS_Tools.H"
#include "HYMLS_MatrixUtils.H"
#include "HYMLS_OverlappingPartitioner.H"

#include "HYMLS_SchurComplement.H"
#include "HYMLS_SchurPreconditioner.H"
#include <Epetra_Time.h> 
#include "Epetra_Comm.h"
#include "Epetra_SerialComm.h"
#include "Epetra_Map.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_Import.h"
#include "Ifpack_Amesos.h"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Utils.hpp"

#include "HYMLS_View_MultiVector.H"

#include "BelosBlockGmresSolMgr.hpp"
#include "BelosBlockCGSolMgr.hpp"
#include "BelosPCPGSolMgr.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include "Galeri_Periodic.h"


typedef Teuchos::Array<int>::iterator int_i;

namespace HYMLS {


  // constructor
  Solver::Solver(Teuchos::RCP<const Epetra_RowMatrix> K, 
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<const OverlappingPartitioner> hid,
      int myLevel)
      : numInitialize_(0),numCompute_(0),numApplyInverse_(0),
        flopsInitialize_(0.0),flopsCompute_(0.0),flopsApplyInverse_(0.0),
        timeInitialize_(0.0),timeCompute_(0.0),timeApplyInverse_(0.0),
        initialized_(false),computed_(false),
        matrix_(K), comm_(Teuchos::rcp(&(K->Comm()),false)), params_(Teuchos::null),
        hid_(hid),
        rangeMap_(Teuchos::rcp(&(K->RowMatrixRowMap()),false)),
        normInf_(-1.0), useTranspose_(false), 
        usePreconditioner_(true),
        myLevel_(myLevel),
        label_("HYMLS::Solver (level "+Teuchos::toString(myLevel_)+")")
    {
    DEBUG("Solver::Solver(...)");
    serialComm_=Teuchos::rcp(new Epetra_SerialComm());
    time_=Teuchos::rcp(new Epetra_Time(K->Comm()));
    SetParameters(*params);
#ifdef TESTING
    dumpVectors_=true;
#endif    
    }


  // destructor
  Solver::~Solver()
    {
    DEBUG("Solver::~Solver()");
    }

  // Ifpack_Preconditioner interface
  
  //@{

  // Sets all parameters for the preconditioner.
  int Solver::SetParameters(Teuchos::ParameterList& List)
    {
    DEBUG("Enter Solver::SetParameters()");
    if (Teuchos::is_null(params_))
      {
      params_=Teuchos::rcp(new Teuchos::ParameterList);
      }

    // this is the place where we check for
    // valid parameters for the whole
    // method

    Teuchos::ParameterList& probList_ = params_->sublist("Problem");
    Teuchos::ParameterList& probList = List.sublist("Problem");

    // general settings for all problems
    int dim=probList.get("Dimension",2);
    probList_.set("Dimension", dim);

    if (dim>=1) probList_.set("nx", probList.get("nx",16));
    if (dim>=2) probList_.set("ny", probList.get("ny",16));
    if (dim>=3) probList_.set("nz", probList.get("nz",16));

    if (dim>=1) probList_.set("x-periodic", probList.get("x-periodic",false));
    if (dim>=2) probList_.set("y-periodic", probList.get("y-periodic",false));
    if (dim>=3) probList_.set("z-periodic", probList.get("z-periodic",false));
    
    probList_.set("Visualize Solver", probList.get("Visualize Solver",false));

    Teuchos::ParameterList& solverList_ = params_->sublist("Solver");
    Teuchos::ParameterList& solverList = List.sublist("Solver");

    solverList_.set("Partitioner", solverList.get("Partitioner","Cartesian"));
    solverList_.set("No Preconditioning",solverList.get("No Preconditioning",false));

    usePreconditioner_=!(solverList_.get("No Preconditioning",false));

    solverList_.set("Scale Schur-Complement",solverList.get("Scale Schur-Complement",false));
    scaleSchur_=solverList_.get("Scale Schur-Complement",false);

    int pos=1;
    while (pos>0)
      {
      string label = "Fix GID "+Teuchos::toString(pos);
      if (solverList.isParameter(label))
        {
        solverList_.set(label,solverList.get(label,-1));
        pos++;
        }
      else
        {
        pos=-1;
        }
      }

    // for the recursive solver, store the original separator length
    int sepx, sepy, sepz;
    int base_sepx, base_sepy, base_sepz;

    // these two are alternatives:
    if (solverList.isParameter("Separator Length"))
      {
      sepx=solverList.get("Separator Length",4);
      sepy=sepx; 
      sepz=dim>2? sepx:1;
      }
    else if (solverList.isParameter("Separator Length (x)"))
      {
      sepx=solverList.get("Separator Length (x)",4);
      sepy=solverList.get("Separator Length (y)",sepx);
      sepz=solverList.get("Separator Length (z)",dim>2?sepx:1);
      }
    else
      {
      sepx=-1;
      solverList_.set("Number of Subdomains", solverList.get("Number of Subdomains",16));
      }
    if (sepx>0)
      {
      solverList_.set("Separator Length (x)", sepx);
      solverList_.set("Separator Length (y)", sepy);
      solverList_.set("Separator Length (z)", sepz);
      }

    if (solverList.isParameter("Base Separator Length"))
      {
      base_sepx=solverList.get("Base Separator Length",sepx);
      base_sepy=base_sepx;
      base_sepz=dim>2?base_sepx:1;
      }
    else if (solverList.isParameter("Base Separator Length (x)"))
      {
      base_sepx=solverList.get("Base Separator Length (x)",sepx);
      base_sepy=solverList.get("Base Separator Length (y)",sepy);
      base_sepz=solverList.get("Base Separator Length (z)",sepz);
      }
    else
      {
      base_sepx=sepx;
      base_sepy=sepy;
      base_sepz=sepz;
      }
    
    solverList_.set("Base Separator Length (x)", base_sepx);
    solverList_.set("Base Separator Length (y)", base_sepy);
    solverList_.set("Base Separator Length (z)", base_sepz);
    
    solverList_.set("Subdivide Separators",solverList.get("Subdivide Separators",false));
    solverList_.set("Subdivide based on variable",solverList.get("Subdivide based on variable",-1));

    solverList_.set("Number of Levels",solverList.get("Number of Levels",2));
    solverList_.set("Nested Iterations",solverList.get("Nested Iterations",false));

    solverList_.sublist("Subdomain Solver")
       = solverList.sublist("Subdomain Solver");

    // this typically doesn't need parameters, it's just lapack on small dense
    // matrices.
    solverList_.sublist("Dense Solver")
       = solverList.sublist("Dense Solver");

    solverList_.sublist("Coarse Solver")
       = solverList.sublist("Coarse Solver");

    solverList_.set("Krylov Method",solverList.get("Krylov Method","GMRES"));
    solverList_.set("Initial Vector",solverList.get("Initial Vector","Previous"));
    solverList_.set("Left or Right Preconditioning",solverList.get("Left or Right Preconditioning","Right"));
    // Belos parameters should be specified in this list:
    solverList_.sublist("Iterative Solver")=solverList.sublist("Iterative Solver");

    // there are two ways of defining the problem to be solved:
    // (a) simply set the "Equations" to something we know
    // ("Laplace", "Stokes-C"), or set the "Problem Definition"
    // sublist manually (for expert use only!)
    string eqn="Undefined Problem";
    

    if (probList.isParameter("Equations"))
      {
      if (probList.isSublist("Problem Definition"))
        {
        Tools::Out("you have set both 'Equations' and 'Problem Definition'");
        Tools::Out("in your parameter list. You should set only one of them.");
        return -1;
        }
      probList_.set("Complex Arithmetic",probList.get("Complex Arithmetic",false));  
      
      eqn=probList.get("Equations",eqn);
      probList_.set("Equations",eqn);
      this->SetProblemDefinition(eqn,*params_);
      probList_.remove("Equations");
      }
    else
      {
      // make sure the user has set the "Problem Definition" list
      if (!probList.isSublist("Problem Definition"))
        {
        Tools::Warning("You have not set the 'Equations' parameter or 'Problem Definition' correctly",__FILE__,__LINE__);        
        }
      probList_.sublist("Problem Definition")=probList.sublist("Problem Definition");
      if (probList.isParameter("Complex Arithmetic"))
        {
        bool complex = probList.get("Complex Arithmetic",false);
        if (complex)
          {
          // user should manually adjust the sublist instead of expecting us to do it
          Tools::Warning(std::string("You have set the 'Complex Arithmetic' parameter in combination \n")+
                       std::string("with the 'Problem Definition' sublist, it will be ignored!"),
                       __FILE__,__LINE__);
        
          }
        }
      }


    List.unused(std::cerr);
    DEBVAR(*params_);
//    std::cout << Label() << std::endl;
//    std::cout << *params_ << std::endl;
    DEBUG("Leave Solver::SetParameters()");
    return 0;
    }

  // Computes all it is necessary to initialize the preconditioner.
  int Solver::Initialize()
    {
    START_TIMER(label_,"Initialize");
    time_->ResetStartTime();
    if (hid_==Teuchos::null)
      {
      hid_=Teuchos::rcp(new HYMLS::OverlappingPartitioner(matrix_,params_));
      }

#ifdef TESTING
  this->Visualize("hid_data_deb.m",false);
  // preconditioner will do the same thing,
  // so the hid_data_deb.m file is always written,
  // even if the program crashes before the end of
  // the Compute() phase.
#endif

    // create all arrays we need
    DEBUG("allocate memory for blocks");
    int num_sd=hid_->NumMySubdomains();
    localMap1_.resize(num_sd);
    localMap2_.resize(num_sd);
    localImport1_.resize(num_sd);
    localImport2_.resize(num_sd);

    localA12_.resize(num_sd);
    localA21_.resize(num_sd);
    localA22_.resize(num_sd);

    subdomainSolver_.resize(num_sd);

    // construct maps for the 1- (subdomain-) and 2- (separator-)blocks
    DEBUG("Solver: construct maps");

    Teuchos::RCP<const RecursiveOverlappingPartitioner> interiorParts = 
        hid_->Spawn(RecursiveOverlappingPartitioner::Interior);
    Teuchos::RCP<const RecursiveOverlappingPartitioner> separatorParts = 
        hid_->Spawn(RecursiveOverlappingPartitioner::Separators);

  map1_ = interiorParts->GetMap();
  map2_ = separatorParts->GetMap();

  int *AllMyElements=new int[rangeMap_->NumMyElements()];
  
#ifdef TESTING
  if (rangeMap_->NumMyElements()!=(map1_->NumMyElements()+map2_->NumMyElements()))
    {
    std::ofstream ofs("hid_data.m",std::ios::app);
    ofs << *hid_<<std::endl;
    ofs.close();
    DEBVAR(*map1_);
    DEBVAR(*map2_);
    DEBVAR(*rangeMap_);
    Tools::Error("inconsistent maps found",__FILE__,__LINE__);
    }
#endif
  
  for (int i=0;i<map1_->NumMyElements();i++)
    {
    AllMyElements[i]=map1_->GID(i);
    }

  for (int i=0;i<map2_->NumMyElements();i++)
    {
    AllMyElements[map1_->NumMyElements()+i]=map2_->GID(i);
    }

  DEBUG("build rowMap");
  rowMap_ = Teuchos::rcp(new Epetra_Map(-1,rangeMap_->NumMyElements(),AllMyElements,
                rangeMap_->IndexBase(), *comm_));

  importer_=Teuchos::rcp(new Epetra_Import(*rowMap_,*rangeMap_));
                
  delete [] AllMyElements;
  
  DEBUG("spawn subdomain maps");
  
  for (int sd=0;sd<hid_->NumMySubdomains();sd++)
    {
    localMap1_[sd] = hid_->SpawnMap(sd,RecursiveOverlappingPartitioner::Interior);
    DEBVAR(*localMap1_[sd]);
    localMap2_[sd] = hid_->SpawnMap(sd,RecursiveOverlappingPartitioner::Separators);
    DEBVAR(*localMap2_[sd]);
    }
            
  int MaxNumEntriesPerRow=matrix_->MaxNumEntries();

  Teuchos::RCP<const Epetra_CrsMatrix> Acrs = Teuchos::null;

  Acrs=Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);

  if (Teuchos::is_null(Acrs))
    {
    Tools::Error("Currently requires an Epetra_CrsMatrix!",__FILE__,__LINE__);
    }

#ifdef STORE_MATRICES
MatrixUtils::Dump(*rangeMap_,"originalMap"+Teuchos::toString(myLevel_)+".txt");
#endif

#ifdef STORE_MATRICES
MatrixUtils::Dump(*rowMap_,"reorderedMap"+Teuchos::toString(myLevel_)+".txt");
#endif

  DEBUG("Reorder global matrix");
  reorderedMatrix_=Teuchos::rcp(new Epetra_CrsMatrix(Copy,*rowMap_,MaxNumEntriesPerRow));

  CHECK_ZERO(reorderedMatrix_->Import(*Acrs,*importer_,Insert));

try {
    EPETRA_CHK_ERR(reorderedMatrix_->FillComplete());
    } catch (...) {HYMLS::Tools::Error("caught exception in FillComplete()",__FILE__,__LINE__);}
    
    DEBUG("construct col-maps, importers and submatrices. Import");
    Teuchos::RCP<Epetra_Map> colMap1 = MatrixUtils::AllGather(*map1_);
    Teuchos::RCP<Epetra_Map> colMap2 = MatrixUtils::AllGather(*map2_);

    import1_ = Teuchos::rcp(new Epetra_Import(*map1_,*rowMap_));
    import2_ = Teuchos::rcp(new Epetra_Import(*map2_,*rowMap_));

    
    for (int sd=0;sd<hid_->NumMySubdomains();sd++)
      {
      localImport1_[sd] = Teuchos::rcp(new Epetra_Import(*localMap1_[sd],*rowMap_));
      localImport2_[sd] = Teuchos::rcp(new Epetra_Import(*localMap2_[sd],*rowMap_));
      
      //TODO: we copy a lot of data, we should actually use EpetraExt Views.  
      //      I think, however, that their View_CrsGraph class only allows    
      //      viewing the first block (A11). EpetraExt::SubCopyCrsMatrix does 
      //      the same as we do, so we should probably use that class here to 
      //      make the implementation more readable and easier to debug.      
      localA12_[sd] = Teuchos::rcp(new
          Epetra_CrsMatrix(Copy,*localMap1_[sd],*localMap2_[sd],MaxNumEntriesPerRow));

      localA21_[sd] = Teuchos::rcp(new
          Epetra_CrsMatrix(Copy,*localMap2_[sd],*localMap1_[sd],MaxNumEntriesPerRow));

      localA22_[sd] = Teuchos::rcp(new
          Epetra_CrsMatrix(Copy,*localMap2_[sd],*localMap2_[sd],MaxNumEntriesPerRow));

      CHECK_ZERO(localA12_[sd]->Import(*reorderedMatrix_,*localImport1_[sd],Insert));
      CHECK_ZERO(localA21_[sd]->Import(*reorderedMatrix_,*localImport2_[sd],Insert));
      CHECK_ZERO(localA22_[sd]->Import(*reorderedMatrix_,*localImport2_[sd],Insert));

      // we let these operators act on all separators. As localMap2_[sd] are reordered local
      // submaps of map2_, we cannot use View operations anyway and things have to be 
      // permuted when applying these operators.
      CHECK_ZERO(localA12_[sd]->FillComplete(*map2_,*localMap1_[sd]));
      CHECK_ZERO(localA21_[sd]->FillComplete(*localMap1_[sd],*localMap2_[sd]));
      CHECK_ZERO(localA22_[sd]->FillComplete(*map2_,*map2_));
      }
//TODO: we presently construct both local blocks and global blocks.
//      For the preconditioner, local blocks are useful. For i.e. 
//      applying the operators, global blocks are easier and probably
//      more efficient. Not sure, what the final implementation will be
      A12_ = Teuchos::rcp(new
          Epetra_CrsMatrix(Copy,*map1_,*colMap2,MaxNumEntriesPerRow));

      A21_ = Teuchos::rcp(new
          Epetra_CrsMatrix(Copy,*map2_,*colMap1,MaxNumEntriesPerRow));

      A22_ = Teuchos::rcp(new
          Epetra_CrsMatrix(Copy,*map2_,*colMap2,MaxNumEntriesPerRow));

      CHECK_ZERO(A12_->Import(*reorderedMatrix_,*import1_,Insert));
      CHECK_ZERO(A21_->Import(*reorderedMatrix_,*import2_,Insert));
      CHECK_ZERO(A22_->Import(*reorderedMatrix_,*import2_,Insert));

      CHECK_ZERO(A12_->FillComplete(*map2_,*map1_));
      CHECK_ZERO(A21_->FillComplete(*map1_,*map2_));
      CHECK_ZERO(A22_->FillComplete(*map2_,*map2_));

#ifdef STORE_MATRICES
MatrixUtils::Dump(*A12_, "Solver"+Teuchos::toString(myLevel_)+"_A12.txt");
MatrixUtils::Dump(*A21_, "Solver"+Teuchos::toString(myLevel_)+"_A21.txt");
MatrixUtils::Dump(*A22_, "Solver"+Teuchos::toString(myLevel_)+"_A22.txt");
#endif
      
  DEBUG("initialize subdomain solvers...");

  for (int sd=0;sd<hid_->NumMySubdomains();sd++)
    {
    int nrows = hid_->NumInteriorElements(sd);
    
    subdomainSolver_[sd] = 
        Teuchos::rcp( new ifpackSolverType_(nrows) );

    IFPACK_CHK_ERR(subdomainSolver_[sd]->SetParameters
        (params_->sublist("Solver").sublist("Subdomain Solver")));
        
    CHECK_ZERO(subdomainSolver_[sd]->Initialize());

    // flops in Initialize() will be computed on-the-fly in method InitializeFlops().

    // set "global" ID of each partitioner row
    for (int j = 0 ; j < nrows ; j++) 
      {
      int LRID = rowMap_->LID(hid_->GID(sd,0,j));
      subdomainSolver_[sd]->ID(j) = LRID;
      }                            
    }

  DEBUG("Create Schur-complement");

  // construct the Schur-complement operator (no computations, just
  // pass in pointers of the LU's)
  Schur_=Teuchos::rcp(new SchurComplement(Teuchos::rcp(this,false)));
  Teuchos::RCP<const Epetra_CrsMatrix> SC = Schur_->Matrix();
  
#ifdef TESTING
Tools::out() << "LEVEL "<< myLevel_<<std::endl;
Tools::out() << "SIZE OF A: "<< rowMap_->NumGlobalElements()<<std::endl;
Tools::out() << "SIZE OF S: "<< map2_->NumGlobalElements()<<std::endl;
#endif  

  
  if (usePreconditioner_)
    {
    DEBUG("Construct preconditioner");

    schurPrec_=Teuchos::rcp(new SchurPreconditioner(SC,hid_,
                params_, myLevel_));
        
    // now we have all the data structures, but the pattern of 
    // the Schur-complement is not available, yet (it will be in 
    // Compute()). So we cannot initialize the Schur preconditioner
    // here (Ifpack_Preconditioner::Initialize() requires the pattern
    // to be there).
    }
  
  // create Belos' view of the Schur-complement problem
  schurRhs_=Teuchos::rcp(new Epetra_Vector(*map2_));
  schurSol_=Teuchos::rcp(new Epetra_Vector(*map2_));
  schurSol_->PutScalar(0.0);
  
  schurProblemPtr_=Teuchos::rcp(new belosProblemType_(Schur_,schurSol_,schurRhs_));
  if (usePreconditioner_)
    {
    schurPrecPtr_=Teuchos::rcp(new belosPrecType_(schurPrec_));
    string lor = params_->sublist("Solver").get("Left or Right Preconditioning","Right");
    if (lor=="Left")
      {
      schurProblemPtr_->setLeftPrec(schurPrecPtr_);
      }
    else if (lor=="Right")
      {
      schurProblemPtr_->setRightPrec(schurPrecPtr_);      
      }
    else
      {
      Tools::Error("Parameter 'Left or Right Preconditioning' has an invalid value",
                __FILE__, __LINE__);
      }
    }

  Teuchos::ParameterList& belosList = params_->sublist("Solver").sublist("Iterative Solver");
  string linearSolver = params_->sublist("Solver").get("Krylov Method","GMRES");

  belosList.set("Output Style",Belos::Brief);
  belosList.set("Verbosity",Belos::Errors+Belos::Warnings
                           +Belos::IterationDetails
                           +Belos::StatusTestDetails
                           +Belos::FinalSummary);
                           //+Belos::TimingDetails

// create the solver
RCP<Teuchos::ParameterList> belosListPtr=rcp(&belosList,false);
if (linearSolver=="CG")
  {
  schurSolverPtr_ = rcp(new 
        Belos::BlockCGSolMgr<double,Epetra_MultiVector,Epetra_Operator>
        (schurProblemPtr_,belosListPtr));
  }
else if (linearSolver=="PCG")
  {
  Tools::Warning("NOT IMPLEMENTED!",__FILE__,__LINE__);
  schurSolverPtr_ = rcp(new 
        Belos::PCPGSolMgr<double,Epetra_MultiVector,Epetra_Operator>
        (schurProblemPtr_,belosListPtr));
  }
else if (linearSolver=="GMRES")
  {
  schurSolverPtr_ = rcp(new 
        Belos::BlockGmresSolMgr<double,Epetra_MultiVector,Epetra_Operator>
        (schurProblemPtr_,belosListPtr));
  }
else if (linearSolver=="None")
  {
  schurSolverPtr_=Teuchos::null;
  }
else
  {
  Tools::Error("Currently only 'GMRES' is supported as 'Belos Solver'",__FILE__,__LINE__);
  }

  // create output manager - we start by printing everything. 
  // TODO: do we have to put it into a parameter list or is it
  // somehow globally shared?
  int verbosity = Belos::Errors+Belos::Warnings+
                  Belos::IterationDetails+Belos::OrthoDetails+
                  Belos::FinalSummary+Belos::TimingDetails+
                  Belos::StatusTestDetails;
  
  Teuchos::RCP<std::ostream> os = Teuchos::rcp(&(std::cout), false);
       
  //TODO: I think this object never makes it to the solver
  schurOutputPtr_=Teuchos::rcp(new belosOutputType_(verbosity,os));  
  
  initialized_=true;
  numInitialize_++;
  timeInitialize_+=time_->ElapsedTime();

  STOP_TIMER(label_,"Initialize");
  return 0;
  }


int Solver::InitializeCompute()
  {
  START_TIMER(label_,"InitializeCompute");


  // (1) import values of matrix into local data structures.
  //     This certainly has to be done before any Compute()

    Teuchos::RCP<const Epetra_CrsMatrix> Acrs = 
        Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);

    EPETRA_CHK_ERR(reorderedMatrix_->PutScalar(0.0));
    EPETRA_CHK_ERR(reorderedMatrix_->Import(*Acrs,*importer_,Insert));
    

#ifdef STORE_MATRICES
    MatrixUtils::Dump(*Acrs,"originalMatrix"+Teuchos::toString(myLevel_)+".txt");
#endif    

  for (int sd=0;sd<hid_->NumMySubdomains();sd++)
    {
    EPETRA_CHK_ERR(localA12_[sd]->PutScalar(0.0));
    EPETRA_CHK_ERR(localA21_[sd]->PutScalar(0.0));
    EPETRA_CHK_ERR(localA22_[sd]->PutScalar(0.0));
    
    CHECK_ZERO(localA12_[sd]->Import(*reorderedMatrix_,*localImport1_[sd],Insert));
    CHECK_ZERO(localA21_[sd]->Import(*reorderedMatrix_,*localImport2_[sd],Insert));
    CHECK_ZERO(localA22_[sd]->Import(*reorderedMatrix_,*localImport2_[sd],Insert));
    }
    
    EPETRA_CHK_ERR(A12_->PutScalar(0.0));
    EPETRA_CHK_ERR(A21_->PutScalar(0.0));
    EPETRA_CHK_ERR(A22_->PutScalar(0.0));

  CHECK_ZERO(A12_->Import(*reorderedMatrix_,*import1_,Insert));
  CHECK_ZERO(A21_->Import(*reorderedMatrix_,*import2_,Insert));
  CHECK_ZERO(A22_->Import(*reorderedMatrix_,*import2_,Insert));

  // (2) re-initiailze the subdomain solvers. I don't know why this is 
  //     required, maybe some Ifpack glitch? If we don't do this before
  //     Compute(), the solver doesn't work.
  DEBUG("initialize subdomain solvers...");

  for (int sd=0;sd<hid_->NumMySubdomains();sd++)
    {
    CHECK_ZERO(subdomainSolver_[sd]->Initialize());
    }

  STOP_TIMER(label_,"InitializeCompute");
  return 0;
  }

  // Returns true if the  preconditioner has been successfully initialized, false otherwise.
  bool Solver::IsInitialized() const {return initialized_;}

  // Computes all it is necessary to apply the preconditioner.
  int Solver::Compute()
    {
    START_TIMER(label_,"Compute");
    if (!IsInitialized())
      {
      // the user should normally call Initialize before Compute
      Tools::Warning("HYMLS::Solver not initialized. I'll do it for you.",
        __FILE__,__LINE__);
      this->Initialize();
      }

  time_->ResetStartTime();

InitializeCompute();

START_TIMER(label_,"Subdomain factorization");

  for (int sd=0;sd<hid_->NumMySubdomains();sd++)
    {
    if (subdomainSolver_[sd]->NumRows()>0)
      {
      // compute subdomain factorization
      CHECK_ZERO(subdomainSolver_[sd]->Compute(*reorderedMatrix_));
      }
    }

STOP_TIMER(label_,"Subdomain factorization");
    
  START_TIMER(label_,"Construct Schur-Complement");
  CHECK_ZERO(Schur_->Construct());
  STOP_TIMER(label_,"Construct Schur-Complement");

#ifdef STORE_MATRICES
  if (Schur_->IsConstructed())
    {
    //MatrixUtils::Dump(*(Schur_->Matrix()),"SchurComplement.txt",false);
    MatrixUtils::Dump(*(Schur_->Matrix()),"SchurReindexed"+Teuchos::toString(myLevel_)+".txt",true);
    }

for (int sd=0;sd<hid_->NumMySubdomains();sd++)
  {
  if (subdomainSolver_[sd]->NumRows()>0)
    {
    Tools::out() << "Level: "<<myLevel_<<", save A11 block "<<sd<<std::endl;
    Teuchos::RCP<const Epetra_CrsMatrix> A11 = subdomainSolver_[sd]->Matrix();
    MatrixUtils::Dump(*A11, 
      "Solver"+Teuchos::toString(myLevel_)+"_A11_"+Teuchos::toString(sd)+".txt");  
    }
  else
    {
    Tools::out() << "Level: "<<myLevel_<<", subdomain "<<sd<<" has 0 rows!"<<std::endl;
    }
  }
#endif

  if (scaleSchur_)
    {
    // the scaling is somewhat adhoc right now.
  
    // TODO: this is not true in general but works for some       
    //       problems we're trying to tackle:                     
    //    Laplace - no scaling                                    
    //    Navier-Stokes with uv(w)p ording - diagonal scaling of  
    //            V-nodes not coupled to P-nodes                  
    //    THCM - doesn't work because P is variable 4 out of 6.   
    if (hid_->Partitioner().DofPerNode()>4)
      {
      Tools::Error("scaling not implemented for THCM",__FILE__,__LINE__);
      }
    int pvar=hid_->Partitioner().DofPerNode()-1;
    schurScaLeft_=Schur_->ConstructLeftScaling(pvar);
    schurScaRight_=Schur_->ConstructRightScaling();
    
    CHECK_ZERO(Schur_->Scale(schurScaLeft_,schurScaRight_));
    }

  if (usePreconditioner_)
    {
//    if (schurPrec_->IsInitialized()==false)
    if (1)
      {
      DEBUG("initialize preconditioner");
      schurPrec_->SetParameters(*params_);
      // we can do this only now where the pattern is available
      EPETRA_CHK_ERR(schurPrec_->Initialize());
      }

    CHECK_ZERO(schurPrec_->Compute());
    }

  computed_ = true;
  timeCompute_ += time_->ElapsedTime();
  numCompute_++;
#ifdef TESTING
{
Epetra_Vector test_lhs(*map1_);
Epetra_Vector test_rhs(*map1_);
MatrixUtils::Random(test_rhs);
CHECK_ZERO(this->ApplyInverseA11(test_rhs,test_lhs));
MatrixUtils::Dump(test_rhs,"Solver"+Teuchos::toString(myLevel_)+"_test_Rhs1.txt",true);
MatrixUtils::Dump(test_lhs,"Solver"+Teuchos::toString(myLevel_)+"_test_Sol1.txt",true);
}
#endif

    STOP_TIMER(label_,"Compute");

  if (params_->sublist("Problem").get("Visualize Solver",false)==true)
    {
    Tools::out() << "MATLAB file for visualizing the solver is written to hid_data.m" << std::endl;
    this->Visualize("hid_data.m");
    }

  return 0;
  }

  // Returns true if the  preconditioner has been successfully computed, false otherwise.
  bool Solver::IsComputed() const {return computed_;}

//TODO: the flops-counters currently do not include anything inside Belos

  double Solver::InitializeFlops() const
    {
    // the total number of flops is computed each time InitializeFlops() is
    // called. This is becase I also have to add the contribution from each
    // container.
    double total = flopsInitialize_;
    for (int i = 0 ; i < subdomainSolver_.size() ; i++)
      {
      if (subdomainSolver_[i]!=Teuchos::null)
        {
        total += subdomainSolver_[i]->InitializeFlops();
        }
      }
    if (usePreconditioner_)
      {
      if (schurPrec_!=Teuchos::null)
        {
        total+=schurPrec_->InitializeFlops();      
        }
      }
    return(total);
    }

  double Solver::ComputeFlops() const
    {
    double total = flopsCompute_;
    if (Schur_!=Teuchos::null)
      {
      total += Schur_->ComputeFlops();
      }
    for (int i = 0 ; i < subdomainSolver_.size() ; i++)
      {
      if (subdomainSolver_[i]!=Teuchos::null)
        {
        total += subdomainSolver_[i]->ComputeFlops();
        }
      }
    if (usePreconditioner_)
      {
      if (schurPrec_!=Teuchos::null)
        {
        total +=schurPrec_->ComputeFlops();
        }
      }
    return(total);
    }

  double Solver::ApplyInverseFlops() const
    {
    double total = flopsApplyInverse_;
    if (Schur_!=Teuchos::null)
      {
      total+=Schur_->ApplyFlops();
      }
    for (int i = 0 ; i < subdomainSolver_.size() ; i++) 
      {
      if (subdomainSolver_[i]!=Teuchos::null)
        {
        total += subdomainSolver_[i]->ApplyInverseFlops();
        }
      }
    if (usePreconditioner_)
      {
      if (schurPrec_!=Teuchos::null)
        {
        total +=schurPrec_->ApplyInverseFlops();
        }
      }
    return(total);
    }


  // Computes the condition number estimate, returns its value.
  double Solver::Condest(const Ifpack_CondestType CT,
                         const int MaxIters,
                         const double Tol,
                         Epetra_RowMatrix* Matrix)
                         {
                         Tools::Warning("not implemented!",__FILE__,__LINE__);
                         return -1.0; // not implemented.
                         }

  // Returns the computed condition number estimate, or -1.0 if not computed.
  double Solver::Condest() const
    {
    Tools::Warning("not implemented!",__FILE__,__LINE__);
    return -1.0;
    }

  // Applies the preconditioner to vector X, returns the result in Y.
  int Solver::ApplyInverse(const Epetra_MultiVector& B,
                           Epetra_MultiVector& X) const
    {
    START_TIMER(label_,"ApplyInverse");
    numApplyInverse_++;
    time_->ResetStartTime();

#ifdef TESTING
if (dumpVectors_)
  {
  MatrixUtils::Dump(*(B(0)), "Solver"+Teuchos::toString(myLevel_)+"_Rhs.txt",true);
  }
#endif    
    int numvec=X.NumVectors();   // these are used for calculating flops
    int veclen=X.GlobalLength();

    
    // create some vectors based on the map we use internally (first all internal and then 
    // all separator variables):
    Teuchos::RCP<Epetra_MultiVector> x,y,z,b;
    x = Teuchos::rcp( new Epetra_MultiVector(*rowMap_,X.NumVectors()) );
    y = Teuchos::rcp( new Epetra_MultiVector(*rowMap_,X.NumVectors()) );
    z = Teuchos::rcp( new Epetra_MultiVector(*rowMap_,X.NumVectors()) );
    b = Teuchos::rcp( new Epetra_MultiVector(*rowMap_,X.NumVectors()) );

    EPETRA_CHK_ERR(b->Import(B,*importer_,Zero)); // should just be a local reordering

    // create a view of the Schur-part of these vectors. Note that EpetraExt's version
    // doesn't work here because it assumes the submap to be the first part of the original
    HYMLS::MultiVector_View separators(*rowMap_,*map2_);

    // create appropriate views of the vectors
    Epetra_MultiVector& x2=separators(*x);
    Epetra_MultiVector& y2=separators(*y);
    Epetra_MultiVector& z2=separators(*z);
    Epetra_MultiVector& b2=separators(*b);

    
    DEBUG("solve subdomains...");
    CHECK_ZERO(ApplyInverseA11(*b, *x));
    
    DEBUG("apply A21...");    
    //CHECK_ZERO(ApplyA21(*x,z2,&flopsApplyInverse_));
    CHECK_ZERO(ApplyA21(*x,*z,&flopsApplyInverse_));
            
    schurRhs_->Update(1.0,b2,-1.0,z2,0.0);
    flopsApplyInverse_+=veclen*numvec;

    string startVec = params_->sublist("Solver").get("Initial Vector","Previous");

    if (startVec=="Random")
      {
#ifdef DEBUGGING
      int seed=42;
      MatrixUtils::Random(*schurSol_, seed);
#else
      MatrixUtils::Random(*schurSol_);
#endif      
      }
    else if (startVec=="Zero")
      {
      // set initial vector to 0
      EPETRA_CHK_ERR(schurSol_->PutScalar(0.0));
      }

  if (scaleSchur_)
    {
    // left-scale rhs with schurScaLeft_
    CHECK_ZERO(schurRhs_->Multiply(1.0, *schurScaLeft_, *schurRhs_, 0.0))
    }
    // solve Schur-complement problem
    DEBUG("solve Schur...");    
    CHECK_TRUE(schurProblemPtr_->setProblem());
    DEBVAR(*(schurProblemPtr_->getInitResVec()));
    Belos::ReturnType ret;
    int status;
    if (schurSolverPtr_!=Teuchos::null)
      {
      try {
      ret=schurSolverPtr_->solve();
      } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr, status);
      }
    else
      {
      CHECK_ZERO(schurPrecPtr_->ApplyInverse(*schurRhs_,*schurSol_));
      ret=Belos::Converged;//TODO: check preconditinoer return code
      }

  // unscale rhs with schurScaRight_
  if (scaleSchur_)
    {
    CHECK_ZERO(schurSol_->ReciprocalMultiply(1.0, *schurScaRight_, *schurSol_, 0.0))
    }
    
    //
    // Get the number of iterations for this solve.
    //
    if (schurSolverPtr_!=Teuchos::null)
      {
      int numIters = schurSolverPtr_->getNumIters();
      if (comm_->MyPID()==0)
        {
        Tools::Out("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        Tools::Out("iterations on Schur complement (level "+Teuchos::toString(myLevel_)+"): "+Teuchos::toString(numIters));
        Tools::Out("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        Tools::Out("");
        }
      }

#ifdef TESTING
// compute explicit residual of Schur problem
Epetra_Vector SchurRes(schurRhs_->Map());
CHECK_ZERO(Schur_->Apply(*schurSol_,SchurRes));
CHECK_ZERO(SchurRes.Update(1.0,*schurRhs_,-1.0));
double resNorm;
SchurRes.Norm2(&resNorm);
if (comm_->MyPID()==0)
  {
  Tools::out() << "LEVEL "<< myLevel_<< ", Residual norm of Schur problem: ";
  Tools::out() << resNorm << std::endl;
  }
#endif

    if (ret!=Belos::Converged) 
      {
      // the nature of the problem is kind of hard to determine...
      Tools::Warning("Level "+Teuchos::toString(myLevel_)+
        ": Belos returned "+Teuchos::toString((int)ret)+"!",__FILE__,__LINE__);    
#ifdef TESTING     
      Teuchos::RCP<const Epetra_CrsMatrix> Acrs =
        Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);
      MatrixUtils::Dump(*Acrs,"FailedMatrix.txt");
      MatrixUtils::Dump(B,"FailedRhs.txt");
      bool reindex=true;
      MatrixUtils::Dump(*(Schur_->Matrix()),"FailedSchurMatrix.txt",reindex);
      MatrixUtils::Dump(*schurRhs_,"FailedSchurRhs.txt",reindex);
      MatrixUtils::Dump(*schurSol_,"FailedSchurSol.txt",reindex);
#endif      
      }
      
    //TODO: avoid this copy operation
    x2=*schurSol_;
    
    // this gives z1
    DEBUG("Apply A12...");
    CHECK_ZERO(ApplyA12(x2, *z,&flopsApplyInverse_));
    // this gives y1, y2=0
    DEBUG("solve subdomains...");
    CHECK_ZERO(ApplyInverseA11(*z, *y));
    // this gives the final result [x1-y1; x2]
    CHECK_ZERO(x->Update(-1.0,*y,1.0));
    flopsApplyInverse_+=numvec*veclen;    

    DEBUG("export solution.");
    EPETRA_CHK_ERR(X.Export(*x,*importer_,Zero)); // should just be a local reordering

      /*
    if (ret!=Belos::Converged) 
      {
      Teuchos::RCP<const Epetra_CrsMatrix> Acrs
        = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);
      if (Acrs!=Teuchos::null)
        {
        MatrixUtils::Dump(*Acrs,"BadMatrix.txt");
        }
      MatrixUtils::Dump(*(B(0)),"BadRhs.txt");
      MatrixUtils::Dump(*(X(0)),"BadSol.txt");
      // we stop here to look at the system
      Tools::Error("Bad Solve!",__FILE__,__LINE__);
      }
      */

#ifdef TESTING
  if (dumpVectors_)
    {
    MatrixUtils::Dump(*(X(0)), "Solver"+Teuchos::toString(myLevel_)+"_Sol.txt",true);
    dumpVectors_=false;
    }
#endif    
    timeApplyInverse_+=time_->ElapsedTime();
    STOP_TIMER(label_,"ApplyInverse");
    return 0;
    }

  // Returns a pointer to the matrix to be preconditioned.
  const Epetra_RowMatrix& Solver::Matrix() const {return *matrix_;}

  // Returns the number of calls to Initialize().
  int Solver::NumInitialize() const {return numInitialize_;}

  // Returns the number of calls to Compute().
  int Solver::NumCompute() const {return numCompute_;}

  // Returns the number of calls to ApplyInverse().
  int Solver::NumApplyInverse() const {return numApplyInverse_;}

  // Returns the time spent in Initialize().
  double Solver::InitializeTime() const {return timeInitialize_;}

  // Returns the time spent in Compute().
  double Solver::ComputeTime() const {return timeCompute_;}

  // Returns the time spent in ApplyInverse().
  double Solver::ApplyInverseTime() const {return timeApplyInverse_;}


  // Prints basic information on iostream. This function is used by operator<<.
  ostream& Solver::Print(std::ostream& os) const
    {
    os << Label() << std::endl;
    if (IsInitialized())
      {
      os << "+++++++++++++++++++++++++++++++++"<<std::endl;
      os << "+ Domain Decomp  osition object:  +"<<std::endl;
      os << "+++++++++++++++++++++++++++++++++"<<std::endl;
      os << *hid_;
      os << "+++++++++++++++++++++++++++++++++"<<std::endl;
      }
    else
      {
      os << " ... not initialized ..."<<std::endl;
      }

    if (IsComputed())
      {
      os << "+++++++++++++++++++++++++++++++++"<<std::endl;
      os << "+ Factorization info:           +"<<std::endl;
      os << "+++++++++++++++++++++++++++++++++"<<std::endl;
      os << "+ NOT IMPLEMENTED.              +"<<std::endl;
      os << "+++++++++++++++++++++++++++++++++"<<std::endl;
      }
    else
      {
      os << " ... not computed ..."<<std::endl;
      }
    return os;
    }
  //@}

// solve a block diagonal system with A11. Vectors are based on rowMap_
int Solver::ApplyInverseA11(const Epetra_MultiVector& B, Epetra_MultiVector& X) const
  {

  if (!IsComputed())
    {
    Tools::Warning("solver not computed!",__FILE__,__LINE__);
    return -1;
    }
  
  START_TIMER(label_,"subdomain solve");
  int lid=0;

    // step 1: solve subdomain problems for temporary vector y
    for (int sd = 0 ; sd < subdomainSolver_.size() ; sd++)
      {
      // extract RHS from X
      for (int j = 0 ; j < subdomainSolver_[sd]->NumRows() ; j++)
        {
        lid = subdomainSolver_[sd]->ID(j);
        for (int k = 0 ; k < B.NumVectors() ; k++)
          {
          subdomainSolver_[sd]->RHS(j,k) = B[k][lid];
          }
        }
      // apply the inverse of each block. NOTE: flops occurred
      // in ApplyInverse() of each block are summed up in method
      // ApplyInverseFlops().
      if (subdomainSolver_[sd]->NumRows()>0)
        {
        IFPACK_CHK_ERR(subdomainSolver_[sd]->ApplyInverse());
        }

      // copy back into solution vector Y
      for (int j = 0 ; j < subdomainSolver_[sd]->NumRows() ; j++)
        {
        lid = subdomainSolver_[sd]->ID(j);
        for (int k = 0 ; k < B.NumVectors() ; k++)
          {
          X[k][lid] = subdomainSolver_[sd]->LHS(j,k);
          }
        }
      }

    STOP_TIMER(label_,"subdomain solve");
    return 0;
    }


  // apply Y=A12*X. This only works if the solver is computed. The input vector
  // Y should be based on rowMap__, X on rowMap_ or map2_.
  int Solver::ApplyA12(const Epetra_MultiVector& X, 
                             Epetra_MultiVector& Y,
                             double* flops) const
    {

    if (!IsComputed())
      {
      Tools::Warning("solver not computed!",__FILE__,__LINE__);
      return -1;
      }

    HYMLS::MultiVector_View separators(X.Map(),*map2_);
#ifdef BLOCK_IMPLEMENTATION
    for (int sd=0;sd<hid_->NumMySubdomains();sd++)
      {
      HYMLS::MultiVector_View interior(Y.Map(),*localMap1_[sd]);
      EPETRA_CHK_ERR(localA12_[sd]->Apply(separators(X),interior(Y)));
      if (flops) *flops+=2*localA12_[sd]->NumGlobalNonzeros();
      }  
#else
      HYMLS::MultiVector_View interior(Y.Map(),*map1_);
      EPETRA_CHK_ERR(A12_->Apply(separators(X),interior(Y)));
      if (flops) *flops+=2*A12_->NumGlobalNonzeros();
#endif      
    return 0;
    }

  // apply Y=A21*X. This only works if the solver is computed. The input vector
  // X should be based on rowMap_ and Y on rowMap_ or map2_.
  int Solver::ApplyA21(const Epetra_MultiVector& X, 
                             Epetra_MultiVector& Y,
                             double* flops) const
    {    
    if (!IsComputed())
      {
      Tools::Warning("solver not computed!",__FILE__,__LINE__);
      return -1;
      }

    HYMLS::MultiVector_View separators(Y.Map(),*map2_);

#ifdef BLOCK_IMPLEMENTATION_this_is_disabled
// this is hard to implement: each loc_tmp may own nodes belonging to 
// other processors, but inside the loop we can't do collective calls.
// We would have to construct a global overlapping vector first, fill it,
// and them export it to the non-overlapping one. This code fragment doesn't work!!!
    Epetra_MultiVector tmp(*map2_, Y.NumVectors());
    
    EPETRA_CHK_ERR(separators(Y).PutScalar(0.0))

    for (int sd=0;sd<hid_->NumMySubdomains();sd++)
      {
      Epetra_MultiVector loc_tmp(*localMap2_[sd], Y.NumVectors());
      //Epetra_Import import(*localMap2_[sd],*map2_);
      Epetra_Import import(*map2_,*localMap2_[sd]);
      HYMLS::MultiVector_View interior(X.Map(),*localMap1_[sd]);
      // Here Trilinos assumes A21 is mostly empty and consequently
      // zeros out the vector in each step. That's why we need a temporary
      // vector (TODO: this is a hotfix, really, we should rethink the
      // implementation).
      EPETRA_CHK_ERR(localA21_[sd]->Apply(interior(X),loc_tmp));
      tmp.PutScalar(0.0);
      EPETRA_CHK_ERR(tmp.Import(loc_tmp,import,Insert));
      EPETRA_CHK_ERR(separators(Y).Update(1.0,tmp,1.0));
      //EPETRA_CHK_ERR(separators(Y).Import(loc_tmp,import,Add));
      DEBVAR(import);
      DEBVAR(*localA21_[sd]);
      DEBVAR(loc_tmp);
      DEBVAR(interior(X));
      DEBVAR(tmp);

      if (flops) *flops+=2*localA21_[sd]->NumGlobalNonzeros();
      }
#else
      HYMLS::MultiVector_View interior(X.Map(),*map1_);
      EPETRA_CHK_ERR(A21_->Apply(interior(X),separators(Y)));
      if (flops) *flops+=2*A21_->NumGlobalNonzeros();

#endif      
    return 0;
    }

  //! apply Y=A22*X. This only works if the solver is computed. The input vectors
  //! can be based on rowMap_ or map2_ 
  int Solver::ApplyA22(const Epetra_MultiVector& X, 
                             Epetra_MultiVector& Y,
                             double *flops) const
    {
    if (!IsComputed())
      {
      Tools::Warning("solver not computed!",__FILE__,__LINE__);
      return -1;
      }
    if (!(X.Map().SameAs(Y.Map())))
      {
      Tools::Warning("incompatible maps!",__FILE__,__LINE__);
      return -2;
      }

#ifdef BLOCK_IMPLEMENTATION_this_is_disabled
//note: a block-implementation is not so straight-forward because
//      the overlap between the blocks leads to multiple addition of some
//      values. We don't have that problem for A12 and A21 because the '1'-map
//      at least doesn't overlap. This code fragment doesn't work!!!
    Epetra_MultiVector tmp(*map2_, Y.NumVectors());
    
    EPETRA_CHK_ERR(separators(Y).PutScalar(0.0))

    for (int sd=0;sd<hid_->NumMySubdomains();sd++)
      {
      // Here Trilinos assumes A22 is mostly empty and consequently
      // zeros out the vector in each step. That's why we need a temporary
      // vector (TODO: this is a hotfix, really, we should rethink the
      // implementation)
      EPETRA_CHK_ERR(localA22_[sd]->Apply(separators(X),tmp));
      EPETRA_CHK_ERR(separators(Y).Update(1.0,tmp,1.0));
      if (flops) *flops+=2*localA22_[sd]->NumGlobalNonzeros();
      }
#else
    HYMLS::MultiVector_View separators(X.Map(),*map2_);

    EPETRA_CHK_ERR(A22_->Apply(separators(X),separators(Y)));
    if (flops) *flops+=2*A22_->NumGlobalNonzeros();
#endif
    return 0;
    }


int Solver::SetProblemDefinition(string eqn, Teuchos::ParameterList& list)
  {
  Teuchos::ParameterList& probList=list.sublist("Problem");
  Teuchos::ParameterList& defList=list.sublist("Problem").sublist("Problem Definition");
  Teuchos::ParameterList& solverList=list.sublist("Solver");

  int dim=probList.get("Dimension",2);
  
  bool xperio=false;
  bool yperio=false;
  bool zperio=false;
  xperio=probList.get("x-periodic",xperio);
  if (dim>=1) yperio=probList.get("y-periodic",yperio);
  if (dim>=2) zperio=probList.get("z-periodic",zperio);
  
  Galeri::PERIO_Flag perio=Galeri::NO_PERIO;
  
  if (xperio) perio=(Galeri::PERIO_Flag)(perio|Galeri::X_PERIO);
  if (yperio) perio=(Galeri::PERIO_Flag)(perio|Galeri::Y_PERIO);
  if (zperio) perio=(Galeri::PERIO_Flag)(perio|Galeri::Z_PERIO);
  
  defList.set("Periodicity",perio);
  
  bool is_complex = probList.get("Complex Arithmetic",false);

  if (eqn=="Laplace")
    {
    defList.set("Dimension",dim);
    defList.set("Substitute Graph",false); 
    if (!is_complex)
      {
      defList.set("Degrees of Freedom",1);
      defList.set("Variable Type (0)","Laplace");
      }
    else
      {
      defList.set("Degrees of Freedom",2);
      defList.set("Variable Type (0)","Laplace");
      defList.set("Variable Type (1)","Laplace");
      }
    
    }
  else if (eqn=="Stokes-C")
    {
    defList.set("Dimension",dim);
    defList.set("Substitute Graph",false); 
    int factor = is_complex? 2 : 1;
    defList.set("Degrees of Freedom",(dim+1)*factor);
    for (int i=0;i<dim*factor;i++)
      {
      Teuchos::ParameterList& velList =
        defList.sublist("Variable "+Teuchos::toString(i));      
      velList.set("Variable Type","Laplace");
      }
    // pressure:
    for (int i=0;i<factor;i++)
      {
      Teuchos::ParameterList& presList =
        defList.sublist("Variable "+Teuchos::toString(dim*factor+i));
      presList.set("Variable Type","Retain 1");
      presList.set("Retain Isolated",true);
      }
    // we fix the singularity by inserting a Dirichlet condition for 
    // global pressure node 2 
    solverList.set("Fix GID 1",factor*dim);
    if (is_complex) solverList.set("Fix GID 2",2*dim+1);
    }
  else
    {
    Tools::Warning("'Equations' parameter not recognized, we only know 'Laplace' and 'Stokes-C' at the moment",
        __FILE__,__LINE__);
    return -1;
    }
  return 0;
  }

void Solver::Visualize(std::string mfilename, bool no_recurse) const
  {
  if ( (comm_->MyPID()==0) && (myLevel_==1))
    {
    std::ofstream ofs(mfilename.c_str(),std::ios::out);
    int dim=params_->sublist("Problem").get("Dimension",-1);
    int dof=params_->sublist("Problem").sublist("Problem Definition").get("Degrees of Freedom",1);
    int nx=params_->sublist("Problem").get("nx",-1);
    int ny=params_->sublist("Problem").get("ny",-1);
    ofs << "dim="<<dim<<";"<<std::endl;
    ofs << "dof="<<dof<<";"<<std::endl;
    ofs << "nx="<<nx<<";"<<std::endl;
    ofs << "ny="<<ny<<";"<<std::endl;
    if (dim>2)
      {
      int nz=params_->sublist("Problem").get("nz",-1);
      ofs << "nz="<<nz<<";"<<std::endl;
      }
    ofs.close();    
    }
  comm_->Barrier();
  std::ofstream ofs(mfilename.c_str(),std::ios::app);
  comm_->Barrier();
  ofs << *hid_<<std::endl;
  ofs.close();
//  hid_->DumpGraph(); //that's the same as of the original matrix right now
  Teuchos::RCP<const Epetra_CrsMatrix> A = Teuchos::rcp_dynamic_cast
        <const Epetra_CrsMatrix>(matrix_);
  if (A!=Teuchos::null) MatrixUtils::Dump(*A,"matrix"+Teuchos::toString(myLevel_)+".txt");
  if ((schurPrec_!=Teuchos::null) && (no_recurse!=true))
    {
    schurPrec_->Visualize(mfilename);
    }
  }
}
