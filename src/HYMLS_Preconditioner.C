//#define BLOCK_IMPLEMENTATION 1
//#include "HYMLS_no_debug.H"

#include "HYMLS_Preconditioner.H"

#include "HYMLS_Tools.H"
#include "HYMLS_MatrixUtils.H"
#include "HYMLS_DenseUtils.H"
#include "HYMLS_OverlappingPartitioner.H"

#include "HYMLS_SchurComplement.H"
#include "HYMLS_SchurPreconditioner.H"
#include "HYMLS_MatrixBlock.H"

#include <Epetra_Time.h> 
#include "Epetra_Comm.h"
#include "Epetra_SerialComm.h"
#include "Epetra_Map.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_Import.h"
#include "Epetra_Export.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"

#include "HYMLS_Tester.H"

#ifdef HYMLS_TESTING
#include "Epetra_FECrsMatrix.h"
#endif

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardParameterEntryValidators.hpp"
#include "Teuchos_Utils.hpp"

#include "Teuchos_StandardCatchMacros.hpp"

#include "GaleriExt_Periodic.h"

namespace HYMLS {

  // constructor
  Preconditioner::Preconditioner(Teuchos::RCP<const Epetra_RowMatrix> K, 
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<const OverlappingPartitioner> hid,
      int myLevel, Teuchos::RCP<Epetra_Vector> testVector)
      : PLA("Preconditioner"),
        comm_(Teuchos::rcp(&(K->Comm()), false)), matrix_(K),
        rangeMap_(Teuchos::rcp(&(K->RowMatrixRowMap()), false)),
        hid_(hid), myLevel_(myLevel), testVector_(testVector),
        useTranspose_(false), normInf_(-1.0),
        label_("Preconditioner"),
        initialized_(false), computed_(false),
        numInitialize_(0), numCompute_(0), numApplyInverse_(0),
        flopsInitialize_(0.0), flopsCompute_(0.0), flopsApplyInverse_(0.0),
        timeInitialize_(0.0), timeCompute_(0.0), timeApplyInverse_(0.0),
        numThreadsSD_(-1)
    {
    HYMLS_LPROF3(label_,"Constructor");
    REPORT_SUM_MEM(label_,"Matrix",K->NumMyNonzeros(),K->NumMyNonzeros(),comm_);
    serialComm_=Teuchos::rcp(new Epetra_SerialComm());
//    serialComm_=Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_SELF));
    time_=Teuchos::rcp(new Epetra_Time(K->Comm()));

    setParameterList(params);
#ifdef HYMLS_DEBUGGING
    dumpVectors_=true;
#endif
    }


  // destructor
  Preconditioner::~Preconditioner()
    {
    HYMLS_LPROF3(label_,"Destructor");
    }


  // Ifpack_Preconditioner interface
  

  // Sets all parameters for the preconditioner.
  int Preconditioner::SetParameters(Teuchos::ParameterList& List)
    {
    HYMLS_LPROF3(label_,"SetParameters");
    
    Teuchos::RCP<Teuchos::ParameterList> List_ = 
        getMyNonconstParamList();
        
   if (List_==Teuchos::null)
     {
     setMyParamList(Teuchos::rcp(&List, false));
     }
   else if (List_.get()!=&List)
     {
     List_->setParameters(List);
     }
    
    Teuchos::ParameterList& probList_ = 
        List_->sublist("Problem");

    // general settings for all problems

    // these are used for writing the partitioning to matlab files
    // ("Visualize Solver" parameter)
    dim_=probList_.get("Dimension",-1);
    nx_=probList_.get("nx",-1);
    ny_=probList_.get("ny",-1);
    if (dim_>2)
      {
      nz_=probList_.get("nz",-1);
      }
    else
      {
      nz_=1;
      }

#ifdef HYMLS_TESTING
    Tester::nx_=nx_;
    Tester::ny_=ny_;
    Tester::nz_=nz_;
    Tester::dim_=dim_;
#endif

    scaleSchur_=PL().get("Scale Schur-Complement",false);

    sdSolverType_ = PL().get("Subdomain Solver Type", "Sparse");
    numThreadsSD_ = PL().get("Subdomain Solver Num Threads", numThreadsSD_);

    bool xperio = false;
    bool yperio = false;
    bool zperio = false;
    xperio = probList_.get("x-periodic", xperio);
    if (dim_>=1) yperio = probList_.get("y-periodic", yperio);
    if (dim_>=2) zperio = probList_.get("z-periodic", zperio);

    GaleriExt::PERIO_Flag perio = GaleriExt::NO_PERIO;

    if (xperio) perio = (GaleriExt::PERIO_Flag)(perio|GaleriExt::X_PERIO);
    if (yperio) perio = (GaleriExt::PERIO_Flag)(perio|GaleriExt::Y_PERIO);
    if (zperio) perio = (GaleriExt::PERIO_Flag)(perio|GaleriExt::Z_PERIO);

    probList_.set("Periodicity", perio);
    probList_.remove("x-periodic");
    probList_.remove("y-periodic");
    probList_.remove("z-periodic");

    // the entire "Problem" list used by the overlapping partiitioner
    // is fairly complex, but we implement a set of default cases like
    // "Laplace", "Stokes-C" etc to make it easier for the user.
    // on coarser levels, the "Equation" parameter is removed and the
    // "Problem Definition" list is provided by the previous level.
    std::string eqn="Undefined Problem";
    
    if (probList_.isParameter("Equations"))
      {
      eqn = probList_.get("Equations",eqn);
#ifdef MATLAB_COMPATIBILITY_MODE
      if (eqn!="Stokes-C" || dim_!=3)
        {
        // this is because of some assumptions we make when ordering the nodes
        // in HierarchicalMap, where we don't have access to the problem definition:
        Tools::Error("MATLAB_COMPATIBILITY_MODE is defined. The ordering\n"+
                     std::string("only works for 3D Stokes-C problems in that case."),
                     __FILE__,__LINE__);
        }
#endif      
      this->SetProblemDefinition(eqn,*List_);
      // the partitioning classes will not accept this parameter,
      // to indicate the list has been processed we remove it.
      probList_.remove("Equations");
      if (probList_.isParameter("Complex Arithmetic"))
        {
        probList_.remove("Complex Arithmetic");
        }
      }
    if (probList_.isParameter("Degrees of Freedom")==false)
    {
      HYMLS::Tools::Error("At this point, the 'Problem' sublist must contain 'Degrees of Freedom'\n"
                          "If you do not set 'Equations', you have to set a (among others) this one.\n",
        __FILE__,__LINE__);
    }
HYMLS_DEBVAR(probList_);
    dof_=probList_.get("Degrees of Freedom",1);

#ifdef HYMLS_TESTING
      Tester::dof_=dof_;
      if (probList_.get("Test F-Matrix Properties",false))
        {
        Tester::doFmatTests_=true;
        Tester::pvar_=dim_;
        }
      else
        {
        Tester::doFmatTests_=false;
        }
#endif

    return 0;
    }

  //!
  void Preconditioner::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& list)
    {
    HYMLS_LPROF3(label_,"setParameterList");
    setMyParamList(list);
    this->SetParameters(*list);
    // this is the place where we check for
    // valid parameters for the preconditioner'
    // and Schur preconditioner
    if (validateParameters_)
      {
      this->getValidParameters();
      PL().validateParameters(VPL());
      }
    HYMLS_DEBVAR(PL());
    }

  //
  Teuchos::RCP<const Teuchos::ParameterList> 
  Preconditioner::getValidParameters() const
    {
    if (validParams_!=Teuchos::null) return validParams_;
    HYMLS_LPROF3(label_,"getValidParameters");

    validParams_=Teuchos::rcp(new Teuchos::ParameterList());
 

    VPL("Problem").set("Dimension", 2,"number of spatial dimensions");

    VPL("Problem").set("nx",16,"number of grid points in x-direction");
    VPL("Problem").set("ny",16,"number of grid points in y-direction");
    VPL("Problem").set("nz",1,"number of grid points in z-direction");

    VPL("Problem").set("x-periodic", false,"assume periodicity in x-direction");
    VPL("Problem").set("y-periodic", false,"assume periodicity in y-direction");
    VPL("Problem").set("z-periodic", false,"assume periodicity in z-direction");

    Teuchos::RCP<Teuchos::StringToIntegralParameterEntryValidator<int> >
        partValidator = Teuchos::rcp(
                new Teuchos::StringToIntegralParameterEntryValidator<int>(
                  Teuchos::tuple<std::string>("Cartesian", "Skew Cartesian"),"Partitioner"));
    
    VPL().set("Partitioner", "Cartesian",
        "Type of partitioner to be used to define the subdomains",
        partValidator);

    VPL().set("Scale Schur-Complement",false,
        "Apply scaling to the Schur complement before building an approximation.\n"
        "This is only intended for Navier-Stokes type problems and it is a bit \n"
        "ad-hoc right now.");

    VPL().set("Fix Pressure Level",true,
        "Put a Dirichlet condition on a single P-node on the coarsest grid");

    VPL().set("Fix GID 1",-1,"put a Dirichlet condition for node x in the last Schur \n"
                                 "complement. This is useful for e.g. fixing the pressure \n"
                                 "level.");

    VPL().set("Fix GID 2",-1,"put a Dirichlet condition for node x in the last Schur \n"
                                 "complement. This is useful for e.g. fixing the pressure \n"
                                 "level.");

    int sepx=4;
    std::string doc = "Defines the subdomain size for Cartesian partitioning";

    VPL().set("Visualize Solver", false, "write matlab files to visualize the partitioning");

    VPL().set("Separator Length", sepx,doc+" (square subdomains)");
    VPL().set("Separator Length (x)", sepx, doc);
    VPL().set("Separator Length (y)", sepx, doc);
    VPL().set("Separator Length (z)", 1, doc);

    std::string doc2 = "Defines the coarsening factor of the subdomains size at each level";

    VPL().set("Base Separator Length", sepx, doc2+" (deprecated)");
    VPL().set("Base Separator Length (x)", sepx, doc2+" (deprecated)");
    VPL().set("Base Separator Length (y)", sepx, doc2+" (deprecated)");
    VPL().set("Base Separator Length (z)", 1, doc2+" (deprecated)");

    VPL().set("Coarsening Factor", sepx, doc2);
    VPL().set("Coarsening Factor (x)", sepx, doc2);
    VPL().set("Coarsening Factor (y)", sepx, doc2);
    VPL().set("Coarsening Factor (z)", 1, doc2);

    VPL().set("Subdivide Separators",false,
        "this was implemented for the rotated B-grid and is not intended for any other "
        "problems right now");
    VPL().set("Subdivide based on variable",-1,
        "decide to which variables on a separator to apply the OT based on couplings to \n"
        " this variable (typically the pressure). Not intended for general use");

    VPL().set("Number of Levels",2,
        "number of levels - on level k a direct solver is used. k=1 is a direct method\n"
        " for the complete problem, k=2 is the standard two-level scheme, k>2 means \n"
        "recursive application of the approximation technique");

    Teuchos::RCP<Teuchos::StringToIntegralParameterEntryValidator<int> >
        sparseDenseValidator = Teuchos::rcp(
                new Teuchos::StringToIntegralParameterEntryValidator<int>(
                        Teuchos::tuple<std::string>( "Sparse","Dense"),"Subdomain Solver Type"));
    

    VPL().set("Subdomain Solver Type","Sparse",
        "Sparse or dense subdomain solver?", sparseDenseValidator);

    VPL().set("Dense Solvers on Level",99,
        "Switch to dense subdomain solver on levels larger than this value");
    
    VPL().set("Subdomain Solver Num Threads",-1,
        "Set number of OMP/MKL threads before calling subdomain solver, -1: don't "
        "(default)");  
    
    // this typically doesn't need parameters, it's just lapack on small dense
    // matrices.
    VPL().sublist("Dense Solver",false,
    "settings for serial dense solves inside the preconditioner").disableRecursiveValidation();

    VPL().sublist("Sparse Solver",false,
    "settings for serial sparse solvers (passed to Ifpack)").disableRecursiveValidation();

    VPL().sublist("Coarse Solver",false,
    "settings for serial or parallel solver used on the last level "
    " (passed to Ifpack_Amesos)").disableRecursiveValidation();      

    Teuchos::RCP<Teuchos::StringToIntegralParameterEntryValidator<int> >
        variantValidator = Teuchos::rcp(
                new Teuchos::StringToIntegralParameterEntryValidator<int>(
                    Teuchos::tuple<std::string>
                    ("Block Diagonal","Lower Triangular","Domain Decomposition","Do Nothing"),
                    "Preconditioner Variant"));
    
    VPL().set("Preconditioner Variant", "Block Diagonal",
        "Type of approximation used for the non-Vsums:\n"
        "'Block Diagonal' - one dense block per separator group (cf. SIMAX paper)\n"
        "'Block Lower Triangular' - not implemented yet\n"
        "'Domain Decomposition' - one sparse block per processor",
        variantValidator);
    
    return validParams_;    
    }


  // Computes all it is necessary to initialize the preconditioner.
  int Preconditioner::Initialize()
    {
    HYMLS_LPROF(label_,"Initialize");
    time_->ResetStartTime();
    if (hid_==Teuchos::null)
      {
      HYMLS_DEBVAR(*getMyNonconstParamList());
      // this is the partitioning step:
      // - partition domain into small subdomains
      // - find separators
      // - group them according to the needs of our algorithm
      hid_=Teuchos::rcp(new 
         HYMLS::OverlappingPartitioner(matrix_,getMyNonconstParamList(),myLevel_));
      }

    HYMLS_TEST(Label()+Teuchos::toString(myLevel_),
      isDDcorrect(*Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_),
        *hid_),__FILE__,__LINE__);

#ifdef HYMLS_TESTING
  this->Visualize("hid_data_deb.m",false);
  // schur preconditioner will do the same thing,
  // so the hid_data_deb.m file is always written,
  // even if the program crashes before the end of
  // the Compute() phase.
#endif

    // this is an idea to make the "HyperCube" object
    // statically available and using it to query things
    // like "idle CPU cores on node" dynamically

    /*
    if (HYMLS::ProcTopo==Teuchos::null)
      {
      HYMLS::Tools::Error("static object HYMLS::ProcTopo not constructed!");
      }
  int active = num_sd>0 ? 1:0;
  // this is just for figuring
  // out how many threads we can
  // use locally, it does not *change*
  // the process layout
  ProcTopo->setActive(myLevel_,active);
*/

  // Obtain a map with overlap between processors from the overlapping
  // partitioner which we need for the A12/A21 subdomain blocks
  rowMap_ = hid_->GetOverlappingMap();

  importer_=Teuchos::rcp(new Epetra_Import(*rowMap_,*rangeMap_));

  int MaxNumEntriesPerRow=matrix_->MaxNumEntries();

  Teuchos::RCP<const Epetra_CrsMatrix> Acrs = Teuchos::null;

  Acrs=Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);

  if (Teuchos::is_null(Acrs))
    {
    Tools::Error("Currently requires an Epetra_CrsMatrix!",__FILE__,__LINE__);
    }

#if defined(HYMLS_STORE_MATRICES) || defined(HYMLS_TESTING)
  MatrixUtils::Dump(*rangeMap_,"originalMap"+Teuchos::toString(myLevel_)+".txt");
  MatrixUtils::Dump(*rowMap_,"reorderedMap"+Teuchos::toString(myLevel_)+".txt");
#endif

  HYMLS_DEBUG("Reorder global matrix");
  reorderedMatrix_=Teuchos::rcp(new Epetra_CrsMatrix(Copy,*rowMap_,MaxNumEntriesPerRow));

  CHECK_ZERO(reorderedMatrix_->Import(*Acrs,*importer_,Insert));

  CHECK_ZERO(reorderedMatrix_->FillComplete());

  REPORT_SUM_MEM(label_,"reordered matrix",reorderedMatrix_->NumMyNonzeros(),
    reorderedMatrix_->NumMyNonzeros(),
    comm_);

  // Construct the matrix blocks we need for the Schur complement
  A11_ = Teuchos::rcp(new MatrixBlock(Acrs, reorderedMatrix_, hid_,
      HierarchicalMap::Interior, HierarchicalMap::Interior, myLevel_));
  A12_ = Teuchos::rcp(new MatrixBlock(Acrs, reorderedMatrix_, hid_,
      HierarchicalMap::Interior, HierarchicalMap::Separators, myLevel_));
  A21_ = Teuchos::rcp(new MatrixBlock(Acrs, reorderedMatrix_, hid_,
      HierarchicalMap::Separators, HierarchicalMap::Interior, myLevel_));
  A22_ = Teuchos::rcp(new MatrixBlock(Acrs, reorderedMatrix_, hid_,
      HierarchicalMap::Separators, HierarchicalMap::Separators, myLevel_));

  // Compute the A12, A21, A22 blocks
  CHECK_ZERO(A12_->Compute());
  CHECK_ZERO(A21_->Compute());
  CHECK_ZERO(A22_->Compute());

  // Also construct the subdomain blocks separately for A12 and A21
  CHECK_ZERO(A12_->ComputeSubdomainBlocks());
  CHECK_ZERO(A21_->ComputeSubdomainBlocks());
  CHECK_ZERO(A22_->ComputeSubdomainBlocks());

#ifdef HYMLS_STORE_MATRICES
  MatrixUtils::Dump(*A12_->Block(), "Precond"+Teuchos::toString(myLevel_)+"_A12.txt");
  MatrixUtils::Dump(*A21_->Block(), "Precond"+Teuchos::toString(myLevel_)+"_A21.txt");
  MatrixUtils::Dump(*A22_->Block(), "Precond"+Teuchos::toString(myLevel_)+"_A22.txt");
#endif

  Teuchos::RCP<Teuchos::ParameterList> sd_list = Teuchos::rcp(new
    Teuchos::ParameterList(PL().sublist("Sparse Solver")));

  // Initialize the subdomain solvers for the A11 block
  CHECK_ZERO(A11_->InitializeSubdomainSolvers(sdSolverType_, sd_list, numThreadsSD_));

  HYMLS_DEBUG("Create Schur-complement");

  Epetra_Map const &map2 = A22_->RowMap();

  // construct the Schur-complement operator (no computations, just
  // pass in pointers of the LU's)
  Schur_ = Teuchos::rcp(new SchurComplement(
      A11_, A12_, A21_, A22_, myLevel_));
Tools::out() << "=============================="<<std::endl;
Tools::out() << "LEVEL "<< myLevel_<<std::endl;
Tools::out() << "SIZE OF A: "<< matrix_->NumGlobalRows64()<<std::endl;
Tools::out() << "SIZE OF S: "<< map2.NumGlobalElements64()<<std::endl;
if (sdSolverType_=="Dense")
  {
    Tools::out() << "*** USING DENSE SUBDOMAIN SOLVERS ***"<<std::endl;
  }
  
Tools::out() << "=============================="<<std::endl;

  // we use a constant vector to generate the orthogonal transformation 
  // for each separator group on the first level, and then keep track   
  // of the coefficients by applying the OT to the vector and extracting
  // the V-sums on each level.
  if (testVector_==Teuchos::null)
    {
    testVector_=Teuchos::rcp(new Epetra_Vector(*rangeMap_));
    testVector_->PutScalar(1.0);
    }
  else
    {
    if (testVector_->Map().SameAs(*rangeMap_)==false)
      {
      Tools::Error("incompatible maps found!",__FILE__,__LINE__);
      }
    }
    
  Epetra_Vector tmpVec(*rowMap_);
  Teuchos::RCP<Epetra_Vector> testVector2 = Teuchos::rcp(new Epetra_Vector(map2));
  Teuchos::RCP<Epetra_Import> import2 = Teuchos::rcp(new Epetra_Import(map2, *rowMap_));
  CHECK_ZERO(tmpVec.Import(*testVector_,*importer_,Insert));
  CHECK_ZERO(testVector2->Import(tmpVec,*import2,Insert));

#ifdef STORE_TESTVECTOR
  Teuchos::RCP<Epetra_Import> importer = Teuchos::rcp(new Epetra_Import(Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_)->DomainMap(), *rangeMap_));
  Epetra_Vector tmpVec2(Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_)->DomainMap());
  tmpVec2.Import(*testVector_, *importer, Insert);

  Epetra_Vector out(*rangeMap_);
  matrix_->Multiply(false, tmpVec2, out);
  MatrixUtils::Dump(*testVector_, "testVec"+Teuchos::toString(myLevel_)+".txt");
  MatrixUtils::Dump(out, "multVec"+Teuchos::toString(myLevel_)+".txt");
  MatrixUtils::Dump(*Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_), "multMat"+Teuchos::toString(myLevel_)+".txt");

  MatrixUtils::Dump(*testVector2, "testVecDrop"+Teuchos::toString(myLevel_)+".txt");
#endif

  HYMLS_DEBUG("Construct schur-preconditioner");
  schurPrec_=Teuchos::rcp(new SchurPreconditioner(Schur_,hid_,
      getMyNonconstParamList(), myLevel_, testVector2));
  
  CHECK_ZERO(schurPrec_->Initialize());

  // create Belos' view of the Schur-complement problem
  schurRhs_=Teuchos::rcp(new Epetra_Vector(map2));
  schurSol_=Teuchos::rcp(new Epetra_Vector(map2));
  schurSol_->PutScalar(0.0);
    
  initialized_=true;
  numInitialize_++;
  timeInitialize_+=time_->ElapsedTime();

  return 0;
  }


int Preconditioner::InitializeCompute()
  {
  HYMLS_LPROF(label_,"InitializeCompute");


  // (1) import values of matrix into local data structures.
  //     This certainly has to be done before any Compute()

  Teuchos::RCP<const Epetra_CrsMatrix> Acrs = 
    Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);

  CHECK_ZERO(reorderedMatrix_->PutScalar(0.0));
  CHECK_ZERO(reorderedMatrix_->Import(*Acrs,*importer_,Insert));

#ifdef HYMLS_STORE_MATRICES
  MatrixUtils::Dump(*Acrs,"originalMatrix"+Teuchos::toString(myLevel_)+".txt");
#endif

  CHECK_ZERO(A11_->Recompute(Acrs, reorderedMatrix_));
  CHECK_ZERO(A12_->Recompute(Acrs, reorderedMatrix_));
  CHECK_ZERO(A21_->Recompute(Acrs, reorderedMatrix_));
  CHECK_ZERO(A22_->Recompute(Acrs, reorderedMatrix_));
  return 0;
  }

  // Returns true if the  preconditioner has been successfully initialized, false otherwise.
  bool Preconditioner::IsInitialized() const {return initialized_;}

  // Computes all it is necessary to apply the preconditioner.
  int Preconditioner::Compute()
    {
    HYMLS_LPROF(label_,"Compute");
    if (!IsInitialized())
      {
      // the user should normally call Initialize before Compute
      Tools::Warning("HYMLS::Preconditioner not initialized. I'll do it for you.",
        __FILE__,__LINE__);
      this->Initialize();
      }

  time_->ResetStartTime();

InitializeCompute();
{
HYMLS_LPROF(label_,"subdomain factorization");
A11_->ComputeSubdomainSolvers();

#ifdef HYMLS_TESTING
    Tools::out() << "Preconditioner level " << myLevel_ << ", doFmatTests=" << Tester::doFmatTests_ << std::endl;
    if (Tester::doFmatTests_)
      {
      // explicitly construct the SC and check wether it is an F-matrix
      Teuchos::RCP<Epetra_FECrsMatrix> TestSC =
        Teuchos::rcp(new Epetra_FECrsMatrix(Copy, A22_->RowMap(), Matrix().MaxNumEntries()));
      CHECK_ZERO(Schur_->Construct(TestSC));

#ifdef HYMLS_STORE_MATRICES
      HYMLS::MatrixUtils::Dump(*TestSC, "SchurComplement" + Teuchos::toString(myLevel_) + "_noDrop.txt");
#endif

      // this is usually done in Construct(), but not if we pass in a pointer ourselves.
      // We need a new pointer because DropByValue creates a CrsMatrix, not FECrsMatrix.
      Teuchos::RCP<Epetra_CrsMatrix> TestSC_crs = MatrixUtils::DropByValue(TestSC, HYMLS_SMALL_ENTRY);

      // Free some memory since these tests use huge amounts
      TestSC = Teuchos::null;

      HYMLS_TEST(Label(), isFmatrix(*TestSC_crs, dof_, dim_), __FILE__, __LINE__);

#ifdef HYMLS_STORE_MATRICES
      HYMLS::MatrixUtils::Dump(*TestSC_crs, "SchurComplement" + Teuchos::toString(myLevel_) + ".txt");
#endif
      }
#endif
}

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
      Tools::Error("scaling not implemented for dof>4",__FILE__,__LINE__);
      }
    int pvar=dim_;
    schurScaLeft_=Schur_->ConstructLeftScaling(pvar);
    schurScaRight_=Schur_->ConstructRightScaling();
    
#ifdef HYMLS_STORE_MATRICES
    MatrixUtils::Dump(*schurScaLeft_,"SchurScaLeft"+Teuchos::toString(myLevel_)+".txt");
    MatrixUtils::Dump(*schurScaRight_,"SchurScaRight"+Teuchos::toString(myLevel_)+".txt");
#endif    
    
    CHECK_ZERO(Schur_->Scale(schurScaLeft_,schurScaRight_));
    }

REPORT_SUM_MEM(label_,"before schurprec",0,0, comm_);
  CHECK_ZERO(schurPrec_->Compute());

  computed_ = true;
  timeCompute_ += time_->ElapsedTime();
  numCompute_++;

  if (PL().get("Visualize Solver",false)==true)
    {
    Tools::out() << "MATLAB file for visualizing the solver is written to hid_data.m" << std::endl;
    this->Visualize("hid_data.m");
    }

  return 0;
  }

  // Returns true if the  preconditioner has been successfully computed, false otherwise.
  bool Preconditioner::IsComputed() const {return computed_;}

  // Applies the preconditioner to vector B, returns the result in X.
  int Preconditioner::ApplyInverse(const Epetra_MultiVector& B,
                           Epetra_MultiVector& X) const
    {
    HYMLS_LPROF(label_,"ApplyInverse");
    Epetra_SerialDenseMatrix S, T;
    if (HaveBorder())
      {
      CHECK_ZERO(T.Reshape(V_->NumVectors(), B.NumVectors()));
      }

    return ApplyInverse(B, T, X, S);
    }

  // Returns a pointer to the matrix to be preconditioned.
  const Epetra_RowMatrix& Preconditioner::Matrix() const {return *matrix_;}

//TODO: the flops-counters currently do not include anything inside Belos

  double Preconditioner::InitializeFlops() const
    {
    // the total number of flops is computed each time InitializeFlops() is
    // called. This is becase I also have to add the contribution from each
    // container.
    double total = flopsInitialize_;

    if (A11_ != Teuchos::null)
      {
      total += A11_->InitializeFlops();
      total += A12_->InitializeFlops();
      total += A21_->InitializeFlops();
      total += A22_->InitializeFlops();
      }

    if (schurPrec_!=Teuchos::null)
      {
      total += schurPrec_->InitializeFlops();      
      }
    return total;
    }

  double Preconditioner::ComputeFlops() const
    {
    double total = flopsCompute_;

    if (Schur_ != Teuchos::null)
      {
      total += Schur_->ComputeFlops();
      }

    if (A11_ != Teuchos::null)
      {
      total += A11_->ComputeFlops();
      total += A12_->ComputeFlops();
      total += A21_->ComputeFlops();
      total += A22_->ComputeFlops();
      }

    if (schurPrec_ != Teuchos::null)
      {
      total += schurPrec_->ComputeFlops();
      }
    return total;
    }

  double Preconditioner::ApplyInverseFlops() const
    {
    double total = flopsApplyInverse_;

    if (Schur_!=Teuchos::null)
      {
      total += Schur_->ApplyFlops();
      }

    if (A11_ != Teuchos::null)
      {
      total += A11_->ApplyInverseFlops();
      total += A12_->ApplyFlops();
      total += A21_->ApplyFlops();
      total += A22_->ApplyFlops();
      }

    if (schurPrec_!=Teuchos::null)
      {
      total +=schurPrec_->ApplyInverseFlops();
      }
    return(total);
    }


  // Computes the condition number estimate, returns its value.
  double Preconditioner::Condest(const Ifpack_CondestType CT,
                         const int MaxIters,
                         const double Tol,
                         Epetra_RowMatrix* Matrix)
                         {
                         Tools::Warning("not implemented!",__FILE__,__LINE__);
                         return -1.0; // not implemented.
                         }

  // Returns the computed condition number estimate, or -1.0 if not computed.
  double Preconditioner::Condest() const
    {
    Tools::Warning("not implemented!",__FILE__,__LINE__);
    return -1.0;
    }


  // Returns the number of calls to Initialize().
  int Preconditioner::NumInitialize() const {return numInitialize_;}

  // Returns the number of calls to Compute().
  int Preconditioner::NumCompute() const {return numCompute_;}

  // Returns the number of calls to ApplyInverse().
  int Preconditioner::NumApplyInverse() const {return numApplyInverse_;}

  // Returns the time spent in Initialize().
  double Preconditioner::InitializeTime() const {return timeInitialize_;}

  // Returns the time spent in Compute().
  double Preconditioner::ComputeTime() const {return timeCompute_;}

  // Returns the time spent in ApplyInverse().
  double Preconditioner::ApplyInverseTime() const {return timeApplyInverse_;}


  // Prints basic information on iostream. This function is used by operator<<.
  ostream& Preconditioner::Print(std::ostream& os) const
    {
    HYMLS_LPROF2(label_,"Print");
    os << Label() << std::endl;
    if (IsInitialized())
      {
      os << "+++++++++++++++++++++++++++++++++"<<std::endl;
      os << "+ Domain Decomposition object:  +"<<std::endl;
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

int Preconditioner::SetProblemDefinition(std::string eqn, Teuchos::ParameterList& list)
  {
    HYMLS_LPROF3(label_,"SetProblemDefinition");
  Teuchos::ParameterList& probList=list.sublist("Problem");
  Teuchos::ParameterList& precList=list.sublist("Preconditioner");
  
  bool is_complex = probList.get("Complex Arithmetic",false);

  if (eqn=="Laplace")
    {
    if (!is_complex)
      {
      probList.set("Degrees of Freedom",1);      
      probList.sublist("Variable 0").set("Variable Type","Laplace");
      }
    else
      {
      probList.set("Degrees of Freedom",2);
      probList.sublist("Variable 0").set("Variable Type","Laplace");
      probList.sublist("Variable 1").set("Variable Type","Laplace");
      }
    
    }
  else if (eqn=="Stokes-C")
    {
    // rare case - only one subdomain. Do not retain a pressure point because there won't be 
    // aSchur-Complement
    bool no_SC = false;
    int sx,sy,sz;
    if (precList.isParameter("Separator Length (x)"))
      {
      sx=precList.get("Separator Length (x)",-1);
      sy=precList.get("Separator Length (y)",sx);
      sz= dim_<3? 1: precList.get("Separator Length (z)",sx);
      }
    else
      {
      sx=precList.get("Separator Length",-1);
      sy=sx;
      sz=dim_<3? 1: sx;
      }
    if (nx_==sx && ny_==sy && nz_==sz) no_SC = true;
    int factor = is_complex? 2 : 1;
    probList.set("Degrees of Freedom",(dim_+1)*factor);
    for (int i=0;i<dim_*factor;i++)
      {
      Teuchos::ParameterList& velList =
        probList.sublist("Variable "+Teuchos::toString(i));
      velList.set("Variable Type","Laplace");
      }
    // pressure:
    for (int i=0;i<factor;i++)
      {
      Teuchos::ParameterList& presList =
        probList.sublist("Variable "+Teuchos::toString(dim_*factor+i));
      if (no_SC==false)
        {
        presList.set("Variable Type","Retain 1");
        // unless the partitioner provides the correct partitioning
        // (with full conservation cells as separate subdomains),  
        // we need to locate them ourselves, which makes the       
        // finding of separators more complex.
        presList.set("Retain Isolated",true);
        }
      else
        {
        presList.set("Variable Type","Uncoupled");
        }
      }
    if (PL().get("Fix Pressure Level",true)==true)
      {
      // we fix the singularity by inserting a Dirichlet condition for 
      // global pressure node 2 
      precList.set("Fix GID 1",factor*dim_);
      if (is_complex) precList.set("Fix GID 2",2*dim_+1);
      }      
#ifdef HYMLS_TESTING
    probList.set("Test F-Matrix Properties",true);
#endif
    }
  else if (eqn=="Bous-C")
    {
    int pvar = dim_ + 1;
    // rare case - only one subdomain. Do not retain a pressure point because there won't be 
    // aSchur-Complement
    bool no_SC = false;
    int sx,sy,sz;
    if (precList.isParameter("Separator Length (x)"))
      {
      sx=precList.get("Separator Length (x)",-1);
      sy=precList.get("Separator Length (y)",sx);
      sz= dim_<3? 1: precList.get("Separator Length (z)",sx);
      }
    else
      {
      sx=precList.get("Separator Length",-1);
      sy=sx;
      sz=dim_<3? 1: sx;
      }
    if (nx_==sx && ny_==sy && nz_==sz) no_SC = true;
    int factor = is_complex? 2 : 1;
    probList.set("Degrees of Freedom",(dim_+2)*factor);

    // Velocities and temperature
    for (int i = 0; i < (dim_+2) * factor; i++)
      {
      if (i != pvar * factor)
        {
        Teuchos::ParameterList& velList =
          probList.sublist("Variable "+Teuchos::toString(i));
        velList.set("Variable Type","Laplace");
        }
      }

    // pressure:
    Teuchos::ParameterList& presList =
      probList.sublist("Variable "+Teuchos::toString(pvar*factor));
      if (!no_SC)
        {
        presList.set("Variable Type","Retain 1");
        presList.set("Retain Isolated",true);
        }
      else
        {
        presList.set("Variable Type","Uncoupled");
        }

    if (PL().get("Fix Pressure Level", true))
      {
      // we fix the singularity by inserting a Dirichlet condition for 
      // global pressure node 2 
      precList.set("Fix GID 1", factor*pvar);
      if (is_complex) precList.set("Fix GID 2", 2*pvar+1);
      }
#ifdef HYMLS_TESTING
    probList.set("Test F-Matrix Properties",true);
#endif
    }
  else if (eqn=="Stokes-B")
    {
    
/* 
   we assume the following 'augmented B-grid',
   where the @ are dummy p-nodes, * are p-nodes
   and > are v-nodes. To transform this into an
   F-matrix, one has to apply a Givvens rotation
   to the velocity field (giving an F-grid). 
   This currently has to be done manually outside
   the solver/preconditioner.

    >---->---->---->>---->---->---->
  @ | *  |  * |  * ||  * |  * | *  |
    >---->---->---->>---->---->---->
  @ | *  |  * |  * ||  * |  * | *  |
    >---->---->---->>---->---->---->
  @ |  * |  * | *  || *  |  * | *  |
    >====>====>====>>====>====>====>
  @ | *  |  * |  * ||  * |  * | *  |
    >---->---->---->>---->---->---->
  @ |  * |  * |  * || *  | *  |  * |
    >---->---->---->>---->---->---->
  @ |  * |  * | *  ||  * | *  |  * |
    >---->---->---->>---->---->---->
  @    @    @    @    @    @     @
*/    
    // case of one subdomain per partition not implemented for B-grid
    bool no_SC=false;
    if (is_complex) Tools::Error("complex Stokes-B not implemented",__FILE__,__LINE__);
    probList.set("Degrees of Freedom",(dim_+1));
    for (int i=0;i<dim_;i++)
      {
      Teuchos::ParameterList& velList =
        probList.sublist("Variable "+Teuchos::toString(i));
      velList.set("Variable Type","Laplace");
    
      // pressure:
      Teuchos::ParameterList& presList =
        probList.sublist("Variable "+Teuchos::toString(dim_+i));
      if (no_SC==false)
        {
        presList.set("Variable Type","Retain 2");
        presList.set("Retain Isolated",true);
        }
      else
        {
        presList.set("Variable Type","Uncoupled");
        }
      }
    if (PL().get("Fix Pressure Level",true)==true)
      {
      // we fix the singularity by inserting a Dirichlet condition for 
      // global pressure in cells 0 and 1, since we retain two pressures
      // per subdomain both will be retained until the coarsest grid.
      // We use +nx*dof here to skip the dummy P-nodes (@).
      precList.set("Fix GID 1",dim_+nx_*dof_);
      precList.set("Fix GID 2",2*dim_+nx_*dof_);
      }
    }
  else
    {
    Tools::Warning("'Equations' parameter not recognized, we only know 'Laplace' and 'Stokes-C' at the moment",
        __FILE__,__LINE__);
    return -1;
    }
  return 0;
  }

void Preconditioner::Visualize(std::string mfilename, bool no_recurse) const
  {
    HYMLS_LPROF2(label_,"Visualize");
  if ( (comm_->MyPID()==0) && (myLevel_==1))
    {
    std::ofstream ofs(mfilename.c_str(),std::ios::out);
    ofs << "dim="<<dim_<<";"<<std::endl;
    ofs << "dof="<<dof_<<";"<<std::endl;
    ofs << "nx="<<nx_<<";"<<std::endl;
    ofs << "ny="<<ny_<<";"<<std::endl;
    if (dim_>2)
      {
      ofs << "nz="<<nz_<<";"<<std::endl;
      }
    ofs.close();    
    }
  comm_->Barrier();
  std::ofstream ofs(mfilename.c_str(),std::ios::app);
  comm_->Barrier();
  ofs << *hid_<<std::endl;
  ofs.close();
#ifdef HYMLS_STORE_MATRICES  
  Teuchos::RCP<const Epetra_CrsMatrix> A = Teuchos::rcp_dynamic_cast
        <const Epetra_CrsMatrix>(matrix_);
  if (A!=Teuchos::null) MatrixUtils::Dump(*A,"matrix"+Teuchos::toString(myLevel_)+".txt");
#endif
  if ((schurPrec_!=Teuchos::null) && (no_recurse!=true))
    {
    schurPrec_->Visualize(mfilename);
    }
  }

//////////////////////////////////
// BORDEREDSOLVER INTERFACE     //
//////////////////////////////////

  // add a border to the preconditioner
  int Preconditioner::setBorder(
               Teuchos::RCP<const Epetra_MultiVector> V, 
               Teuchos::RCP<const Epetra_MultiVector> W,
               Teuchos::RCP<const Epetra_SerialDenseMatrix> C)
    {
    HYMLS_LPROF2(label_,"setBorder");

    V_ = Teuchos::null;
    W_ = Teuchos::null;
    C_ = Teuchos::null;

    if (!IsComputed())
      {
      // this could be done differently, for instance
      // by adding some of these computations to Compute(),
      // but I think it is OK to compute the prec first and
      // set the bordering afterwards.
      Tools::Error("setBorder: requires preconditioner to be computed",
        __FILE__,__LINE__);
      }

    if (V == Teuchos::null)
      {
      borderSchurV_ = Teuchos::null;
      borderSchurW_ = Teuchos::null;
      borderSchurC_ = Teuchos::null;
      borderQ1_ = Teuchos::null;

      CHECK_ZERO(schurPrec_->setBorder(borderSchurV_, borderSchurW_, borderSchurC_));
      return 0;
      }

    if (!V->Map().SameAs(OperatorRangeMap()) || (W != Teuchos::null &&
        !V->Map().SameAs(W->Map())))
      {
      Tools::Error("incompatible maps found", __FILE__, __LINE__);
      }

    int m = V->NumVectors();

    V_ = Teuchos::rcp(new Epetra_MultiVector(*V));

    if (W == Teuchos::null || W.get() == V.get())
      {
      W_ = V_;
      }
    else
      {
      W_ = Teuchos::rcp(new Epetra_MultiVector(*W));
      }

    if (W_->NumVectors() != m)
      {
      Tools::Error("bordering: V and W must have same number of columns",
        __FILE__, __LINE__);
      }

    if (C == Teuchos::null)
      {
      C_ = Teuchos::rcp(new Epetra_SerialDenseMatrix(m,m));
      }
    else
      {
      C_ = Teuchos::rcp(new Epetra_SerialDenseMatrix(*C));
      }

    if ((C_->N() != C_->M()) || (C_->N() != m))
      {
      Tools::Error("bordering: C block must be square and compatible with V and W",
        __FILE__, __LINE__);
      }

    Epetra_Import const &import1 = A12_->Importer();
    Epetra_Import const &import2 = A21_->Importer();

    Epetra_Map const &map1 = A12_->RowMap();
    Epetra_Map const &map2 = A21_->RowMap();

    borderV1_ = Teuchos::rcp(new Epetra_MultiVector(map1, m));
    borderV2_ = Teuchos::rcp(new Epetra_MultiVector(map2, m));

    CHECK_ZERO(borderV1_->Import(*V_, import1, Insert));
    CHECK_ZERO(borderV2_->Import(*V_, import2, Insert));

    if (V_.get() == W_.get())
      {
      borderW1_ = borderV1_;
      borderW2_ = borderV2_;
      }
    else
      {
      borderW1_ = Teuchos::rcp(new Epetra_MultiVector(map1, m));
      borderW2_ = Teuchos::rcp(new Epetra_MultiVector(map2, m));
      CHECK_ZERO(borderW1_->Import(*W_, import1, Insert));
      CHECK_ZERO(borderW2_->Import(*W_, import2, Insert));
      }

    // build the border for the Schur-complement
    borderSchurV_ = Teuchos::rcp(new Epetra_MultiVector(map2,m));
    borderQ1_= Teuchos::rcp(new Epetra_MultiVector(map1,m));

    CHECK_ZERO(A11_->ApplyInverse(*borderV1_,*borderQ1_));
    CHECK_ZERO(A21_->Apply(*borderQ1_, *borderSchurV_));
    CHECK_ZERO(borderSchurV_->Update(1.0,*borderV2_,-1.0));

    // borderSchurW is given by W2 - (A11\A12)'W1
    borderSchurW_ = Teuchos::rcp(new Epetra_MultiVector(map2,m));
    // TODO: we use the formulation W2 - A12'(A11'\W1) instead, not
    //       sure which is the more efficient implementation, but this
    //       seemed to be easier to do quickly.
    Epetra_MultiVector w1tmp(map1,m);
    A11_->SetUseTranspose(true);
    CHECK_ZERO(A11_->ApplyInverse(*borderW1_, w1tmp));
    A11_->SetUseTranspose(false);

    A12_->SetUseTranspose(true);
    CHECK_ZERO(A12_->Apply(w1tmp,*borderSchurW_));
    A12_->SetUseTranspose(false);

    CHECK_ZERO(borderSchurW_->Update(1.0,*borderW2_,-1.0));
    
    borderSchurC_ = Teuchos::rcp(new Epetra_SerialDenseMatrix(m,m));
    CHECK_ZERO(DenseUtils::MatMul(*borderW1_,*borderQ1_,*borderSchurC_));
    CHECK_ZERO(borderSchurC_->Scale(-1.0));
    *borderSchurC_ += *C_;
    
    //TODO: if the Schur-complement is left- and right-scaled,
    //      we also have to scale the borders
    if (scaleSchur_)
      {
      Tools::Error("not implemented!",__FILE__,__LINE__);
      }

    HYMLS_DEBVAR(borderSchurV_->MyLength());
    CHECK_ZERO(schurPrec_->setBorder(borderSchurV_,borderSchurW_,borderSchurC_));
    return 0;
    }

  // if a border has been added, apply [Y T]' = [K V; W' O] [X S]'
  int Preconditioner::Apply(const Epetra_MultiVector& X, const Epetra_SerialDenseMatrix& S,
                                  Epetra_MultiVector& Y,       Epetra_SerialDenseMatrix& T) const
  {
  // we don't need this for now
  Tools::Error("not implemented",__FILE__,__LINE__);
  return -99;
  }           

  // if a border has been added, apply [X S]' = [K V; W' O]\[Y T]'
  int Preconditioner::ApplyInverse(const Epetra_MultiVector& B, const Epetra_SerialDenseMatrix& T,
               Epetra_MultiVector& X,       Epetra_SerialDenseMatrix& S) const
  {
    numApplyInverse_++;
    time_->ResetStartTime();

#ifdef HYMLS_DEBUGGING
    if (dumpVectors_)
      {
      MatrixUtils::Dump(B, "Preconditioner"+Teuchos::toString(myLevel_)+"_Rhs.txt");
      }
#endif

    int numvec = X.NumVectors();

    Epetra_Import const &import1 = A12_->Importer();
    Epetra_Import const &import2 = A21_->Importer();

    Epetra_Map const &map1 = A12_->RowMap();
    Epetra_Map const &map2 = A21_->RowMap();

    Epetra_MultiVector x1(map1, numvec);
    Epetra_MultiVector x2(map2, numvec);

    Epetra_MultiVector b1(map1, numvec);
    Epetra_MultiVector b2(map2, numvec);

    Epetra_MultiVector y1(map1, numvec);
    Epetra_MultiVector y2(map2, numvec);

    // We first import B into the parts of B belonging to their blocks
    CHECK_ZERO(b1.Import(B, import1, Insert));
    CHECK_ZERO(b2.Import(B, import2, Insert));

    // We want to compute
    // A11*x1 + A12*x2 = b1
    // A21*x1 + A22*x2 = b2
    // which results in solving
    // S*x2 = b2 - A21*A11\b1
    // x1 = -A11\A12*x2 + A11\b1
    // where S is the Schur complement

    // We first compute x1 = A11\b1, which we keep for later
    CHECK_ZERO(A11_->ApplyInverse(b1, x1));

    // Now we compute y2 = A21*A11\b1
    CHECK_ZERO(A21_->Apply(x1, y2));

    // We now compute the right-hand side for the Schur complement solve
    if (schurRhs_->NumVectors() != X.NumVectors())
      {
      schurRhs_ = Teuchos::rcp(new Epetra_MultiVector(map2, X.NumVectors()));
      schurSol_ = Teuchos::rcp(new Epetra_MultiVector(map2, X.NumVectors()));
      schurSol_->PutScalar(0.0);
      }
    CHECK_ZERO(schurRhs_->Update(1.0, b2, -1.0, y2, 0.0));

    // We now compute the border in case it is present
    Epetra_SerialDenseMatrix q;
    if (W_!=Teuchos::null)
      {
      CHECK_ZERO(DenseUtils::MatMul(*borderW1_, x1, q));
      CHECK_ZERO(q.Scale(-1));
      S.Reshape(q.M(), q.N());
      q += T;
      }

    // And now we solve the Schur complement system to compute x2
    if (scaleSchur_)
      {
      // left-scale rhs with schurScaLeft_
      CHECK_ZERO(schurRhs_->Multiply(1.0, *schurScaLeft_, *schurRhs_, 0.0));
      }

    CHECK_ZERO(schurPrec_->ApplyInverse(*schurRhs_, q, *schurSol_, S));

    if (scaleSchur_)
      {
      // unscale rhs with schurScaRight_
      CHECK_ZERO(schurSol_->ReciprocalMultiply(1.0, *schurScaRight_, *schurSol_, 0.0))
      }

    x2 = *schurSol_;

    // We have x2 now, so now we can compute x1. Remember that part of the solution
    // is already in there. We first compute y1=A12*x2
    CHECK_ZERO(A12_->Apply(x2, y1));

    // And now b1=A11\A12*x2. Note that we just use b1 as tmp variable here
    CHECK_ZERO(A11_->ApplyInverse(y1, b1));

    // And finally we update x1
    // this gives the final result [x1-y1; x2]
    CHECK_ZERO(x1.Update(-1.0, b1, 1.0));

    // Bordered stuff again
    if (borderQ1_!=Teuchos::null)
      {
      Teuchos::RCP<Epetra_MultiVector> ss = DenseUtils::CreateView(S);
      CHECK_ZERO(x1.Multiply('N', 'N', -1.0, *borderQ1_, *ss, 1.0));
      }

    // And now we import the result into X
    //'Zero' would disable repartitioning here (some
    // ranks may have a part of the vector but not  
    // of the preconditioner), and
    //'Insert' would put the empty overlap nodes into
    // the other subdomains, so we need to zero out X
    // and 'Add' instead.
    CHECK_ZERO(X.PutScalar(0.0));
    CHECK_ZERO(X.Export(x1, import1, Add));
    CHECK_ZERO(X.Export(x2, import2, Add));
    
#ifdef HYMLS_DEBUGGING
    if (dumpVectors_)
      {
      MatrixUtils::Dump(X, "Preconditioner"+Teuchos::toString(myLevel_)+"_Sol.txt");
      dumpVectors_=numApplyInverse_>1?false: true;
      }
#endif
    timeApplyInverse_+=time_->ElapsedTime();
    return 0;
  }

}//namespace
