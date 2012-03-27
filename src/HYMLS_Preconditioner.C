//#define BLOCK_IMPLEMENTATION 1
#include "HYMLS_Preconditioner.H"

#include "HYMLS_Tools.H"
#include "HYMLS_MatrixUtils.H"
#include "HYMLS_DenseUtils.H"
#include "HYMLS_OverlappingPartitioner.H"

#include "HYMLS_SchurComplement.H"
#include "HYMLS_SchurPreconditioner.H"
#include "HYMLS_BorderedLU.H"

#include <Epetra_Time.h> 
#include "Epetra_Comm.h"
#include "Epetra_SerialComm.h"
#include "Epetra_LocalMap.h"
#include "Epetra_Map.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_Import.h"
#include "Epetra_InvOperator.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"

#include "Ifpack_SparseContainer.h"
#include "Ifpack_DenseContainer.h"
#include "Ifpack_Amesos.h"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardParameterEntryValidators.hpp"
#include "Teuchos_Utils.hpp"

#include "HYMLS_View_MultiVector.H"

#include "Teuchos_StandardCatchMacros.hpp"

#include "GaleriExt_Periodic.h"


typedef Teuchos::Array<int>::iterator int_i;

namespace HYMLS {

  // constructor
  Preconditioner::Preconditioner(Teuchos::RCP<const Epetra_RowMatrix> K, 
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<const OverlappingPartitioner> hid,
      int myLevel, Teuchos::RCP<Epetra_Vector> testVector)
      : numInitialize_(0),numCompute_(0),numApplyInverse_(0),
        flopsInitialize_(0.0),flopsCompute_(0.0),flopsApplyInverse_(0.0),
        timeInitialize_(0.0),timeCompute_(0.0),timeApplyInverse_(0.0),
        initialized_(false),computed_(false),
        matrix_(K), comm_(Teuchos::rcp(&(K->Comm()),false)),
        hid_(hid),
        rangeMap_(Teuchos::rcp(&(K->RowMatrixRowMap()),false)),
        normInf_(-1.0), useTranspose_(false), 
        myLevel_(myLevel), testVector_(testVector),
        label_("HYMLS::Preconditioner (level "+Teuchos::toString(myLevel_)+")"),
        PLA("Preconditioner")
    {
    START_TIMER2(label_,"Constructor");
    serialComm_=Teuchos::rcp(new Epetra_SerialComm());
    time_=Teuchos::rcp(new Epetra_Time(K->Comm()));

    setParameterList(params);
#ifdef DEBUGGING
    dumpVectors_=true;
#endif
    }


  // destructor
  Preconditioner::~Preconditioner()
    {
    START_TIMER3(label_,"Destructor");
    }


  // Ifpack_Preconditioner interface
  

  // Sets all parameters for the preconditioner.
  int Preconditioner::SetParameters(Teuchos::ParameterList& List)
    {
    START_TIMER3(label_,"SetParameters");
    
    Teuchos::RCP<Teuchos::ParameterList> List_ = 
        getMyNonconstParamList();
        
   if (List_==Teuchos::null)
     {
     setMyParamList(Teuchos::rcp(&List, false));
     }
   else if (List_.get()!=&List);
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
        
    scaleSchur_=PL().get("Scale Schur-Complement",false);

    sdSolverType_=PL().get("Subdomain Solver Type","Sparse");

    // the entire "Problem" list used by the overlapping partiitioner
    // is fairly complex, but we implement a set of default cases like
    // "Laplace", "Stokes-C" etc to make it easier for the user.
    string eqn="Undefined Problem";
    
    if (probList_.isParameter("Equations"))
      {
      eqn = probList_.get("Equations",eqn);
      this->SetProblemDefinition(eqn,*List_);
      // the partitioning classes will not accept this parameter,
      // to indicate the list has been processed we remove it.
      probList_.remove("Equations");
      if (probList_.isParameter("Complex Arithmetic"))
        {
        probList_.remove("Complex Arithmetic");
        }
      }

    dof_=probList_.get("Degrees of Freedom",1);

    return 0;
    }

  //!
  void Preconditioner::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& list)
    {
    START_TIMER3(label_,"setParameterList");
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
    DEBVAR(PL());
    }

  //
  Teuchos::RCP<const Teuchos::ParameterList> 
  Preconditioner::getValidParameters() const
    {
    if (validParams_!=Teuchos::null) return validParams_;
    START_TIMER3(label_,"getValidParameters");

    validParams_=Teuchos::rcp(new Teuchos::ParameterList());
 

    VPL("Problem").set("Dimension", 2,"number of spatial dimensions");

    VPL("Problem").set("nx",16,"number of grid points in x-direction");
    VPL("Problem").set("ny",16,"number of grid points in y-direction");
    VPL("Problem").set("nz",1,"number of grid points in z-direction");

    VPL("Problem").set("x-periodic", false,"assume periodicity in x-direction");
    VPL("Problem").set("y-periodic", false,"assume periodicity in y-direction");
    VPL("Problem").set("z-periodic", false,"assume periodicity in z-direction");

    //TODO: use validators everywhere

    Teuchos::RCP<Teuchos::StringToIntegralParameterEntryValidator<int> >
        partValidator = Teuchos::rcp(
                new Teuchos::StringToIntegralParameterEntryValidator<int>(
                    Teuchos::tuple<std::string>("Cartesian"),"Partitioner"));
    
    VPL().set("Partitioner", "Cartesian",
        "Type of partitioner to be used to define the subdomains",
        partValidator);
    
    VPL().set("Scale Schur-Complement",false,
        "Apply scaling to the Schur complement before building an approximation.\n"
        "This is only intended for Navier-Stokes type problems and it is a bit \n"
        "ad-hoc right now.");

    VPL().set("Fix GID 1",-1,"put a Dirichlet condition for node x in the last Schur \n"
                                 "complement. This is useful for e.g. fixing the pressure \n"
                                 "level.");

    VPL().set("Fix GID 2",-1,"put a Dirichlet condition for node x in the last Schur \n"
                                 "complement. This is useful for e.g. fixing the pressure \n"
                                 "level.");

    int sepx=4;
    std::string doc = "defines the subdomain size for cortesian partitioning";

    VPL().set("Visualize Solver", false, "write matlab files to visualize the partitioning");

    VPL().set("Separator Length", sepx,doc+" (square subdomains)");
    VPL().set("Separator Length (x)", sepx,doc);
    VPL().set("Separator Length (y)", sepx,doc);
    VPL().set("Separator Length (z)", 1,doc);

    std::string doc2 = "this is an internal parameter that should not be set by the user";

    VPL().set("Base Separator Length", sepx, doc2);
    VPL().set("Base Separator Length (x)", sepx,doc2);
    VPL().set("Base Separator Length (y)", sepx,doc2);
    VPL().set("Base Separator Length (z)", 1,doc2);

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
    START_TIMER(label_,"Initialize");
    time_->ResetStartTime();
    if (hid_==Teuchos::null)
      {
      hid_=Teuchos::rcp(new 
         HYMLS::OverlappingPartitioner(matrix_,getMyNonconstParamList(),myLevel_));
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

#if defined(STORE_MATRICES) || defined(TESTING)
MatrixUtils::Dump(*rangeMap_,"originalMap"+Teuchos::toString(myLevel_)+".txt");
#endif

#if defined(TESTING) || defined(STORE_MATRICES)
MatrixUtils::Dump(*rowMap_,"reorderedMap"+Teuchos::toString(myLevel_)+".txt");
#endif

    // this object can be used to create a vector view of the interior nodes:
    interior_=Teuchos::rcp(new HYMLS::MultiVector_View(*rowMap_,*map1_));

    // create a view of the Schur-part of vectors. Note that EpetraExt's version
    // doesn't work here because it assumes the submap to be the first part of the original
    separators_=Teuchos::rcp(new HYMLS::MultiVector_View(*rowMap_,*map2_));


  DEBUG("Reorder global matrix");
  reorderedMatrix_=Teuchos::rcp(new Epetra_CrsMatrix(Copy,*rowMap_,MaxNumEntriesPerRow));

  CHECK_ZERO(reorderedMatrix_->Import(*Acrs,*importer_,Insert));

try {
    CHECK_ZERO(reorderedMatrix_->FillComplete());
    } catch (...) {HYMLS::Tools::Error("caught exception in FillComplete()",__FILE__,__LINE__);}
    
    DEBUG("construct col-maps, importers and submatrices. Import");
    colMap1_ = MatrixUtils::AllGather(*map1_);
    colMap2_ = MatrixUtils::AllGather(*map2_);

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
          Epetra_CrsMatrix(Copy,*map1_,*colMap2_,MaxNumEntriesPerRow));

      A21_ = Teuchos::rcp(new
          Epetra_CrsMatrix(Copy,*map2_,*colMap1_,MaxNumEntriesPerRow));

      A22_ = Teuchos::rcp(new
          Epetra_CrsMatrix(Copy,*map2_,*colMap2_,MaxNumEntriesPerRow));

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
    
    if (sdSolverType_=="Dense")
      {
      subdomainSolver_[sd] = 
        Teuchos::rcp( new Ifpack_DenseContainer(nrows) );
      }
    else if (sdSolverType_=="Sparse")
      {
      subdomainSolver_[sd] = 
        Teuchos::rcp( new Ifpack_SparseContainer<Ifpack_Amesos>(nrows) );
      }        
    else
      {
      Tools::Error("invalid 'Subdomain Solver Type' in 'Solver' sublist",
        __FILE__,__LINE__);
      }

    IFPACK_CHK_ERR(subdomainSolver_[sd]->SetParameters
        (PL().sublist("Sparse Solver")));
        
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
  Schur_=Teuchos::rcp(new SchurComplement(Teuchos::rcp(this,false),myLevel_));
  Teuchos::RCP<const Epetra_CrsMatrix> SC = Schur_->Matrix();
  
#ifdef TESTING
Tools::out() << "LEVEL "<< myLevel_<<std::endl;
Tools::out() << "SIZE OF A: "<< rowMap_->NumGlobalElements()<<std::endl;
Tools::out() << "SIZE OF S: "<< map2_->NumGlobalElements()<<std::endl;
#endif  

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
  Teuchos::RCP<Epetra_Vector> testVector2
        = Teuchos::rcp(new Epetra_Vector(*map2_));
  CHECK_ZERO(tmpVec.Import(*testVector_,*importer_,Insert));
  CHECK_ZERO(testVector2->Import(tmpVec,*import2_,Insert));

  DEBUG("Construct preconditioner");

  schurPrec_=Teuchos::rcp(new SchurPreconditioner(SC,hid_,
                getMyNonconstParamList(), myLevel_, testVector2));

  // now we have all the data structures, but the pattern of 
  // the Schur-complement is not available, yet (it will be in 
  // Compute()). So we cannot initialize the Schur preconditioner
  // here (Ifpack_Preconditioner::Initialize() requires the pattern
  // to be there).
  
  // create Belos' view of the Schur-complement problem
  schurRhs_=Teuchos::rcp(new Epetra_Vector(*map2_));
  schurSol_=Teuchos::rcp(new Epetra_Vector(*map2_));
  schurSol_->PutScalar(0.0);
    
  initialized_=true;
  numInitialize_++;
  timeInitialize_+=time_->ElapsedTime();

  return 0;
  }


int Preconditioner::InitializeCompute()
  {
  START_TIMER(label_,"InitializeCompute");


  // (1) import values of matrix into local data structures.
  //     This certainly has to be done before any Compute()

    Teuchos::RCP<const Epetra_CrsMatrix> Acrs = 
        Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);

    CHECK_ZERO(reorderedMatrix_->PutScalar(0.0));
    CHECK_ZERO(reorderedMatrix_->Import(*Acrs,*importer_,Insert));
    

#ifdef STORE_MATRICES
    MatrixUtils::Dump(*Acrs,"originalMatrix"+Teuchos::toString(myLevel_)+".txt");
#endif    

  for (int sd=0;sd<hid_->NumMySubdomains();sd++)
    {
    CHECK_ZERO(localA12_[sd]->PutScalar(0.0));
    CHECK_ZERO(localA21_[sd]->PutScalar(0.0));
    CHECK_ZERO(localA22_[sd]->PutScalar(0.0));
    
    CHECK_ZERO(localA12_[sd]->Import(*reorderedMatrix_,*localImport1_[sd],Insert));
    CHECK_ZERO(localA21_[sd]->Import(*reorderedMatrix_,*localImport2_[sd],Insert));
    CHECK_ZERO(localA22_[sd]->Import(*reorderedMatrix_,*localImport2_[sd],Insert));
    }
    
    CHECK_ZERO(A12_->PutScalar(0.0));
    CHECK_ZERO(A21_->PutScalar(0.0));
    CHECK_ZERO(A22_->PutScalar(0.0));

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

  return 0;
  }

  // Returns true if the  preconditioner has been successfully initialized, false otherwise.
  bool Preconditioner::IsInitialized() const {return initialized_;}

  // Computes all it is necessary to apply the preconditioner.
  int Preconditioner::Compute()
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

{
START_TIMER(label_,"Subdomain factorization");

  for (int sd=0;sd<hid_->NumMySubdomains();sd++)
    {
    if (subdomainSolver_[sd]->NumRows()>0)
      {
      // compute subdomain factorization
      CHECK_ZERO(subdomainSolver_[sd]->Compute(*reorderedMatrix_));
      }
    }
}

  CHECK_ZERO(Schur_->Construct());

#ifdef STORE_MATRICES
    MatrixUtils::Dump(*(Schur_->Matrix()),"SchurComplement"+Teuchos::toString(myLevel_)+".txt");
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
    
#ifdef STORE_MATRICES
    MatrixUtils::Dump(*schurScaLeft_,"SchurScaLeft"+Teuchos::toString(myLevel_)+".txt");
    MatrixUtils::Dump(*schurScaRight_,"SchurScaRight"+Teuchos::toString(myLevel_)+".txt");
#endif    
    
    CHECK_ZERO(Schur_->Scale(schurScaLeft_,schurScaRight_));
    }

//  if (schurPrec_->IsInitialized()==false)
  if (1)
    {
    DEBUG("initialize preconditioner");
    schurPrec_->setParameterList(getMyNonconstParamList());
    // we can do this only now where the pattern is available
    CHECK_ZERO(schurPrec_->Initialize());
    }

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

  // Applies the preconditioner to vector X, returns the result in Y.
  int Preconditioner::ApplyInverse(const Epetra_MultiVector& B,
                           Epetra_MultiVector& X) const
    {
    START_TIMER(label_,"ApplyInverse");
    numApplyInverse_++;
    time_->ResetStartTime();

#ifdef DEBUGGING
if (dumpVectors_)
  {
  MatrixUtils::Dump(B, "Preconditioner"+Teuchos::toString(myLevel_)+"_Rhs.txt");
  }
#endif    
    int numvec=X.NumVectors();   // these are used for calculating flops
    int veclen=X.GlobalLength();

    
    // create some vectors based on the map we use internally (first all internal and then 
    // all separator variables):
    Teuchos::RCP<Epetra_MultiVector> x,y,z,b;
    // views of the interior nodes:
    Teuchos::RCP<Epetra_MultiVector> x1,z1;
    // views of the vsum nodes:
    Teuchos::RCP<Epetra_MultiVector> x2,y2,z2,b2;
    
    // only reconstruct temporary vectors and views if
    // first call or number of rhs changed:
    bool realloc=false; 
    if (tmpVec_[0]!=Teuchos::null)
      {
      realloc = (tmpVec_[0]->NumVectors()!=numvec);
      }
    else
      {
      realloc = true;
      }

    if (realloc)
      {
      for (int i=0;i<4;i++)
        {
        tmpVec_[i] = Teuchos::rcp( new Epetra_MultiVector(*rowMap_,X.NumVectors()) );
        tmpVec_[4+i] = (*interior_)(tmpVec_[i]);
        tmpVec_[8+i] = (*separators_)(tmpVec_[i]);
        }
      }
      
    x = tmpVec_[0];
    y = tmpVec_[1];
    z = tmpVec_[2];
    b = tmpVec_[3];
    x1= tmpVec_[4];
    z1= tmpVec_[6];
    x2= tmpVec_[8];
    y2= tmpVec_[9];
    z2= tmpVec_[10];
    b2= tmpVec_[11];
    
    CHECK_ZERO(b->Import(B,*importer_,Zero)); // should just be a local reordering
    
    DEBUG("solve subdomains...");
    CHECK_ZERO(ApplyInverseA11(*b, *x));
    
    DEBUG("apply A21...");    
    //CHECK_ZERO(ApplyA21(*x,z2,&flopsApplyInverse_));
    CHECK_ZERO(ApplyA21(*x,*z,&flopsApplyInverse_));

    if (schurRhs_->NumVectors()!=X.NumVectors())
      {
      schurRhs_=Teuchos::rcp(new Epetra_MultiVector(*map2_,X.NumVectors()));
      schurSol_=Teuchos::rcp(new Epetra_MultiVector(*map2_,X.NumVectors()));
      schurSol_->PutScalar(0.0);
      }        
    schurRhs_->Update(1.0,*b2,-1.0,*z2,0.0);
    flopsApplyInverse_+=veclen*numvec;

    Epetra_SerialDenseMatrix q,s;

    if (borderW_!=Teuchos::null)
      {
      CHECK_ZERO(DenseUtils::MatMul(*borderW1_,*z1,q));
      CHECK_ZERO(q.Scale(-1));
      s.Reshape(q.M(),q.N());
      }

  if (scaleSchur_)
    {
    // left-scale rhs with schurScaLeft_
    CHECK_ZERO(schurRhs_->Multiply(1.0, *schurScaLeft_, *schurRhs_, 0.0))
    }
  if (borderV_!=Teuchos::null)
    {
    //TODO: for recursive application we should probably 
    //      use the next level HYMLS::Preconditioner as the
    //      bordered Solver instead of a borderedLU.
    if (borderedSchurSolver_==Teuchos::null)
      {
      Tools::Error("cannot handle bordered Schur system!",__FILE__,__LINE__);
      }
    else
      {
      CHECK_ZERO(borderedSchurSolver_->ApplyInverse(*schurRhs_,q,*schurSol_,s));  
      }
    }
  else
    {
    CHECK_ZERO(schurPrec_->ApplyInverse(*schurRhs_,*schurSol_));  
    }
  // unscale rhs with schurScaRight_
  if (scaleSchur_)
    {
    CHECK_ZERO(schurSol_->ReciprocalMultiply(1.0, *schurScaRight_, *schurSol_, 0.0))
    }

  //TODO: avoid this copy operation
  *x2=*schurSol_;
    
  // this gives z1
  DEBUG("Apply A12...");
  CHECK_ZERO(ApplyA12(*x2, *z,&flopsApplyInverse_));
  // this gives y1, y2=0   
  DEBUG("solve subdomains...");
  CHECK_ZERO(ApplyInverseA11(*z, *y));
  // this gives the final result [x1-y1; x2]
  CHECK_ZERO(x->Update(-1.0,*y,1.0));
  flopsApplyInverse_+=numvec*veclen;
  
  if (borderQ1_!=Teuchos::null)
    {
    Teuchos::RCP<Epetra_MultiVector> ss = DenseUtils::CreateView(s);
    CHECK_ZERO(x1->Multiply('N','N',-1.0,*borderQ1_,*ss,1.0));
    }   

  DEBUG("export solution.");
  CHECK_ZERO(X.Export(*x,*importer_,Zero)); // should just be a local reordering

#ifdef DEBUGGING
  if (dumpVectors_)
    {
    MatrixUtils::Dump(X, "Preconditioner"+Teuchos::toString(myLevel_)+"_Sol.txt");
    dumpVectors_=false;
    }
#endif    
    timeApplyInverse_+=time_->ElapsedTime();
    return 0;
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
    for (int i = 0 ; i < subdomainSolver_.size() ; i++)
      {
      if (subdomainSolver_[i]!=Teuchos::null)
        {
        total += subdomainSolver_[i]->InitializeFlops();
        }
      }
    if (schurPrec_!=Teuchos::null)
      {
      total+=schurPrec_->InitializeFlops();      
      }
    return(total);
    }

  double Preconditioner::ComputeFlops() const
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
    if (schurPrec_!=Teuchos::null)
      {
      total +=schurPrec_->ComputeFlops();
      }
    return(total);
    }

  double Preconditioner::ApplyInverseFlops() const
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
    START_TIMER2(label_,"Print");
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

// solve a block diagonal system with A11. Vectors are based on rowMap_
int Preconditioner::ApplyInverseA11(const Epetra_MultiVector& B, Epetra_MultiVector& X) const
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
    return 0;
    }

// solve a block diagonal system with A11^T. Vectors are based on rowMap_
int Preconditioner::ApplyInverseA11T(const Epetra_MultiVector& B, Epetra_MultiVector& X) const
  {
  START_TIMER3(label_,"ApplyInverseA11T");
  int ierr=0;
  int nsd=subdomainSolver_.size();
  Teuchos::Array<bool> prevtrans(nsd); // remember if the solvers were already transposed
  Teuchos::RCP<const Ifpack_SparseContainer<Ifpack_Amesos> > sparseLU=Teuchos::null;
  for (int sd=0;sd<nsd;sd++)
    {
    sparseLU=Teuchos::rcp_dynamic_cast
        <const Ifpack_SparseContainer<Ifpack_Amesos> >(subdomainSolver_[sd]);
    if (sparseLU!=Teuchos::null)
      {
      prevtrans[sd]=sparseLU->Inverse()->UseTranspose();
      CHECK_ZERO(Teuchos::rcp_const_cast<Ifpack_Amesos>(sparseLU->Inverse())->SetUseTranspose(true));
      }
    else
      {
      Tools::Error("Transpose not implemented for dense subdomain solver!",__FILE__,__LINE__);
      }
    sparseLU=Teuchos::null;
    }

  ierr = this->ApplyInverseA11(B,X);

  for (int sd=0;sd<nsd;sd++)
    {
    sparseLU=Teuchos::rcp_dynamic_cast
        <const Ifpack_SparseContainer<Ifpack_Amesos> >(subdomainSolver_[sd]);
    if (sparseLU!=Teuchos::null)
      {
      CHECK_ZERO(Teuchos::rcp_const_cast<Ifpack_Amesos>(sparseLU->Inverse())->SetUseTranspose(prevtrans[sd]));
      }
    else
      {
      Tools::Error("Transpose not implemented for dense subdomain solver!",__FILE__,__LINE__);
      }
    sparseLU=Teuchos::null;
    }
  return ierr;
  }


  // apply Y=A12*X. This only works if the solver is computed. The input vector
  // Y should be based on rowMap_ or map1_, X on rowMap_ or map2_.
  int Preconditioner::ApplyA12(const Epetra_MultiVector& X, 
                             Epetra_MultiVector& Y,
                             double* flops) const
    {
  START_TIMER3(label_,"ApplyA12");

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
      CHECK_ZERO(localA12_[sd]->Apply(*(separators(X)),*(interior(Y))));
      if (flops) *flops+=2*localA12_[sd]->NumGlobalNonzeros();
      }  
#else
      HYMLS::MultiVector_View interior(Y.Map(),*map1_);
      CHECK_ZERO(A12_->Apply(*separators(X),*interior(Y)));
      if (flops) *flops+=2*A12_->NumGlobalNonzeros();
#endif      
    return 0;
    }

  // apply Y=A12*X. This only works if the solver is computed. The input vector
  // Y should be based on rowMap_ or map2_, X on rowMap_ or map1_.
  int Preconditioner::ApplyA12T(const Epetra_MultiVector& X, 
                             Epetra_MultiVector& Y,
                             double* flops) const
  {
  START_TIMER3(label_,"ApplyA12T");
    if (!IsComputed())
      {
      Tools::Warning("solver not computed!",__FILE__,__LINE__);
      return -1;
      }

    HYMLS::MultiVector_View interior(X.Map(),*map1_);
    HYMLS::MultiVector_View separators(Y.Map(),*map2_);
    CHECK_ZERO(A12_->Multiply(true,*interior(X),*separators(Y)));
    if (flops) *flops+=2*A12_->NumGlobalNonzeros();
  return 0;
  }
  // apply Y=A21*X. This only works if the solver is computed. The input vector
  // X should be based on rowMap_ or map1_, and Y on rowMap_ or map2_.
  int Preconditioner::ApplyA21(const Epetra_MultiVector& X, 
                             Epetra_MultiVector& Y,
                             double* flops) const
    {
    START_TIMER3(label_,"ApplyA21");    
    
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
    
    CHECK_ZERO(separators(Y)->PutScalar(0.0))

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
      CHECK_ZERO(localA21_[sd]->Apply(*(interior(X)),loc_tmp));
      tmp.PutScalar(0.0);
      CHECK_ZERO(tmp.Import(loc_tmp,import,Insert));
      CHECK_ZERO(separators(Y)->Update(1.0,tmp,1.0));
      //CHECK_ZERO(separators(Y)->Import(loc_tmp,import,Add));
      DEBVAR(import);
      DEBVAR(*localA21_[sd]);
      DEBVAR(loc_tmp);
      DEBVAR(*interior(X));
      DEBVAR(tmp);

      if (flops) *flops+=2*localA21_[sd]->NumGlobalNonzeros();
      }
#else
      HYMLS::MultiVector_View interior(X.Map(),*map1_);
      CHECK_ZERO(A21_->Apply(*interior(X),*separators(Y)));
      if (flops) *flops+=2*A21_->NumGlobalNonzeros();

#endif      
    return 0;
    }

  // apply Y=A21^T*X. This only works if the solver is computed. The input vector
  // X should be based on rowMap_ or map2_, and Y on rowMap_ or map1_.
  int Preconditioner::ApplyA21T(const Epetra_MultiVector& X, 
                             Epetra_MultiVector& Y,
                             double* flops) const
    {
    START_TIMER3(label_,"ApplyA21T");
    if (!IsComputed())
      {
      Tools::Warning("solver not computed!",__FILE__,__LINE__);
      return -1;
      }

    HYMLS::MultiVector_View separators(X.Map(),*map2_);
    HYMLS::MultiVector_View interior(Y.Map(),*map1_);
    
    CHECK_ZERO(A21_->Multiply(true,*separators(X),*interior(Y)));
    if (flops) *flops+=2*A21_->NumGlobalNonzeros();

    return 0;
    }

  //! apply Y=A22*X. This only works if the solver is computed. The input vectors
  //! can be based on rowMap_ or map2_ 
  int Preconditioner::ApplyA22(const Epetra_MultiVector& X, 
                             Epetra_MultiVector& Y,
                             double *flops) const
    {
    START_TIMER3(label_,"ApplyA22");
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
    
    CHECK_ZERO(separators(Y).PutScalar(0.0))

    for (int sd=0;sd<hid_->NumMySubdomains();sd++)
      {
      // Here Trilinos assumes A22 is mostly empty and consequently
      // zeros out the vector in each step. That's why we need a temporary
      // vector (TODO: this is a hotfix, really, we should rethink the
      // implementation)
      CHECK_ZERO(localA22_[sd]->Apply(*separators(X),tmp));
      CHECK_ZERO(separators(Y)->Update(1.0,tmp,1.0));
      if (flops) *flops+=2*localA22_[sd]->NumGlobalNonzeros();
      }
#else
    HYMLS::MultiVector_View separators(X.Map(),*map2_);

    CHECK_ZERO(A22_->Apply(*separators(X),*separators(Y)));
    if (flops) *flops+=2*A22_->NumGlobalNonzeros();
#endif
    return 0;
    }

  //! apply Y=A22^T*X. This only works if the solver is computed. The input vectors
  //! can be based on rowMap_ or map2_ 
  int Preconditioner::ApplyA22T(const Epetra_MultiVector& X, 
                             Epetra_MultiVector& Y,
                             double *flops) const
    {
    START_TIMER3(label_,"ApplyA22T");
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

    HYMLS::MultiVector_View separators(X.Map(),*map2_);

    CHECK_ZERO(A22_->Multiply(true,*separators(X),*separators(Y)));
    if (flops) *flops+=2*A22_->NumGlobalNonzeros();
    return 0;
    }


int Preconditioner::SetProblemDefinition(string eqn, Teuchos::ParameterList& list)
  {
    START_TIMER3(label_,"SetProblemDefinition");
  Teuchos::ParameterList& probList=list.sublist("Problem");
  Teuchos::ParameterList& precList=list.sublist("Preconditioner");

  bool xperio=false;
  bool yperio=false;
  bool zperio=false;
  xperio=probList.get("x-periodic",xperio);
  if (dim_>=1) yperio=probList.get("y-periodic",yperio);
  if (dim_>=2) zperio=probList.get("z-periodic",zperio);
  
  GaleriExt::PERIO_Flag perio=GaleriExt::NO_PERIO;
  
  if (xperio) perio=(GaleriExt::PERIO_Flag)(perio|GaleriExt::X_PERIO);
  if (yperio) perio=(GaleriExt::PERIO_Flag)(perio|GaleriExt::Y_PERIO);
  if (zperio) perio=(GaleriExt::PERIO_Flag)(perio|GaleriExt::Z_PERIO);
  
  probList.set("Periodicity",perio);
  probList.remove("x-periodic");
  probList.remove("y-periodic");
  probList.remove("z-periodic");
  
  bool is_complex = probList.get("Complex Arithmetic",false);

  if (eqn=="Laplace")
    {
    probList.set("Substitute Graph",false); 
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
    probList.set("Substitute Graph",false);
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
      presList.set("Variable Type","Retain 1");
      presList.set("Retain Isolated",true);
      }
    // we fix the singularity by inserting a Dirichlet condition for 
    // global pressure node 2 
    precList.set("Fix GID 1",factor*dim_);
    if (is_complex) precList.set("Fix GID 2",2*dim_+1);
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
    START_TIMER2(label_,"Visualize");
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
//  hid_->DumpGraph(); //that's the same as of the original matrix right now
  Teuchos::RCP<const Epetra_CrsMatrix> A = Teuchos::rcp_dynamic_cast
        <const Epetra_CrsMatrix>(matrix_);
  if (A!=Teuchos::null) MatrixUtils::Dump(*A,"matrix"+Teuchos::toString(myLevel_)+".txt");
  if ((schurPrec_!=Teuchos::null) && (no_recurse!=true))
    {
    schurPrec_->Visualize(mfilename);
    }
  }

//////////////////////////////////
// BORDEREDSOLVER INTERFACE     //
//////////////////////////////////

  // add a border to the preconditioner
  int Preconditioner::SetBorder(
               Teuchos::RCP<const Epetra_MultiVector> V, 
               Teuchos::RCP<const Epetra_MultiVector> W,
               Teuchos::RCP<const Epetra_SerialDenseMatrix> C)
    {
    START_TIMER2(label_,"SetBorder");

    Teuchos::RCP<const Epetra_MultiVector> _V=V;
    Teuchos::RCP<const Epetra_MultiVector> _W=W;
    Teuchos::RCP<const Epetra_SerialDenseMatrix> _C=C;

    if (!IsComputed())
      {
      // this could be done differently, for instance
      // by adding some of these computations to Compute(),
      // but I think it is OK to compute the prec first and
      // set the bordering afterwards.
      Tools::Error("SetBorder: requires preconditioner to be computed",
        __FILE__,__LINE__);
      }
    if (_V==Teuchos::null)
      {
      Tools::Error("SetBorder: V can't be null",__FILE__,__LINE__);
      }
    int m = _V->NumVectors();
    if (_W==Teuchos::null)
      {
      _W=_V;
      }
    if (_C==Teuchos::null)
      {
      _C=Teuchos::rcp(new Epetra_SerialDenseMatrix(m,m));
      }
    
    if (!(_V->Map().SameAs(OperatorRangeMap())&&_V->Map().SameAs(_W->Map())))
      {
      Tools::Error("incompatible maps found",__FILE__,__LINE__);
      }
    if (_W->NumVectors()!=m)
      {
      Tools::Error("bordering: V and W must have same number of columns",
        __FILE__,__LINE__); 
      }
    if ((_C->N()!=_C->M())||(_C->N()!=m))
      {
      Tools::Error("bordering: C block must be square and compatible with V and W",
        __FILE__,__LINE__);
      }
    
    borderV_ = Teuchos::rcp(new Epetra_MultiVector(*rowMap_, m));
    CHECK_ZERO(borderV_->Import(*_V, *importer_, Insert));
    
    if (_W.get()==_V.get())
      {
      borderW_=borderV_;
      }
    else
      {
      borderW_ = Teuchos::rcp(new Epetra_MultiVector(*rowMap_, m));
      CHECK_ZERO(borderW_->Import(*_W, *importer_, Insert));
      }

    borderC_ = _C;
    
    borderV1_ = Teuchos::rcp(new Epetra_MultiVector(*map1_,m));
    borderV2_ = Teuchos::rcp(new Epetra_MultiVector(*map2_,m));

    CHECK_ZERO(borderV1_->Import(*borderV_,*import1_,Insert));
    CHECK_ZERO(borderV2_->Import(*borderV_,*import2_,Insert));

    if (borderV_.get()==borderW_.get())
      {
      borderW1_=borderV1_;
      borderW2_=borderV2_;
      }
    else
      {
      borderW1_ = Teuchos::rcp(new Epetra_MultiVector(*map1_,m));
      borderW2_ = Teuchos::rcp(new Epetra_MultiVector(*map2_,m));
      CHECK_ZERO(borderW1_->Import(*borderW_,*import1_,Insert));
      CHECK_ZERO(borderW2_->Import(*borderW_,*import2_,Insert));
      }

    // build the border for the Schur-complement
    borderSchurV_ = Teuchos::rcp(new Epetra_MultiVector(*map2_,m));
    borderQ1_= Teuchos::rcp(new Epetra_MultiVector(*map1_,m));

    CHECK_ZERO(ApplyInverseA11(*borderV1_,*borderQ1_));
    CHECK_ZERO(ApplyA21(*borderQ1_, *borderSchurV_));
    CHECK_ZERO(borderSchurV_->Update(1.0,*borderV2_,-1.0));

    // borderSchurW is given by W2 - (A11\A12)'W1
    borderSchurW_ = Teuchos::rcp(new Epetra_MultiVector(*map2_,m));
    // TODO: we use the formulation W2 - A12'(A11'\W1) instead, not
    //       sure which is the more efficient implementation, but this
    //       seemed to be easier to do quickly.
    Epetra_MultiVector w1tmp(*map1_,m);
    CHECK_ZERO(this->ApplyInverseA11T(*borderW1_,w1tmp));
    CHECK_ZERO(this->ApplyA12T(w1tmp,*borderSchurW_));
    CHECK_ZERO(borderSchurW_->Update(1.0,*borderW2_,-1.0));
    
    borderSchurC_ = Teuchos::rcp(new Epetra_SerialDenseMatrix(m,m));
    CHECK_ZERO(DenseUtils::MatMul(*borderW1_,*borderQ1_,*borderSchurC_));
    CHECK_ZERO(borderSchurC_->Scale(-1.0));
    *borderSchurC_ += *borderC_;
    
    //TODO: if the Schur-complement is left- and right-scaled,
    //      we also have to scale the borders
    if (scaleSchur_)
      {
      Tools::Error("not implemented!",__FILE__,__LINE__);
      }
    /*
    I don't think this is necessary/correct, the ApplyInverse() function of the
    preconditioner takes care of the OT.
    // TODO: no flops counted - can pass in a counter here if we like.
    CHECK_ZERO(schurPrec_->ApplyOT(true,*borderSchurV_));
    CHECK_ZERO(schurPrec_->ApplyOT(true,*borderSchurW_));
    */      
    borderedSchurSolver_ = Teuchos::rcp
        (new BorderedLU(schurPrec_,borderSchurV_,borderSchurW_,borderSchurC_));

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
  int Preconditioner::ApplyInverse
        (const Epetra_MultiVector& Y, const Epetra_SerialDenseMatrix& T,
               Epetra_MultiVector& X,       Epetra_SerialDenseMatrix& S) const
  {
  Tools::Error("not implemented",__FILE__,__LINE__);
  return -99;
  }

}//namespace
