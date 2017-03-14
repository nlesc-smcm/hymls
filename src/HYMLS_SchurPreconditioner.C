#define RESTRICT_ON_COARSE_LEVEL

#include "HYMLS_no_debug.H"

#include "HYMLS_SchurPreconditioner.H"

#include "HYMLS_Macros.H" 
#include "HYMLS_Tools.H" 
#include "HYMLS_MatrixUtils.H"

#include "HYMLS_OverlappingPartitioner.H" 
#include "HYMLS_SchurComplement.H" 
#include "HYMLS_Preconditioner.H" 
#include "HYMLS_Householder.H" 
#include "HYMLS_SepNode.H"
#include "HYMLS_RestrictedOT.H"

#include "Epetra_Comm.h" 
#include "Epetra_Map.h" 
#include "Epetra_RowMatrix.h" 
#include "Epetra_Import.h" 
#include "Epetra_IntVector.h" 
#include "Epetra_MultiVector.h" 
#include "Epetra_FECrsMatrix.h" 
#include "Epetra_SerialDenseVector.h" 
#include "Ifpack_Amesos.h" 
#include "Teuchos_RCP.hpp" 
#include "Teuchos_ArrayView.hpp" 
#include "Teuchos_ParameterList.hpp" 
#include "Teuchos_StandardCatchMacros.hpp"

#include "Ifpack_DenseContainer.h" 
#include "Ifpack_SparseContainer.h" 
#include "Ifpack_Amesos.h"

#include "EpetraExt_Reindex_CrsMatrix.h" 
#include "EpetraExt_Reindex_MultiVector.h" 
#include "EpetraExt_RestrictedCrsMatrixWrapper.h"
#include "EpetraExt_RestrictedMultiVectorWrapper.h"
#include "EpetraExt_MatrixMatrix.h"

#include "HYMLS_AugmentedMatrix.H"
#include "HYMLS_Tester.H"

namespace HYMLS {


  // private constructor
  SchurPreconditioner::SchurPreconditioner(
                Teuchos::RCP<const Epetra_Operator> SC, 
                Teuchos::RCP<const OverlappingPartitioner> hid, 
                Teuchos::RCP<Teuchos::ParameterList> params,
                int level,
                Teuchos::RCP<Epetra_Vector> testVector)
      : PLA("Preconditioner"),
        comm_(Teuchos::rcp(&(SC->Comm()),false)),
        SchurMatrix_(Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(SC)),
        tmpMatrix_(Teuchos::null),
        SchurComplement_(Teuchos::rcp_dynamic_cast<const HYMLS::SchurComplement>(SC)),
        myLevel_(level), amActive_(true),
        variant_("Block Diagonal"),
        denseSwitch_(99),
        hid_(hid), map_(Teuchos::rcp(&(SC->OperatorDomainMap()),false)),
        testVector_(testVector),
        sparseMatrixOT_(Teuchos::null),
        matrix_(Teuchos::null),
        nextLevelHID_(Teuchos::null),
        linearRhs_(Teuchos::null), linearSol_(Teuchos::null),
        useTranspose_(false), haveBorder_(false), normInf_(-1.0),
        label_("SchurPreconditioner"),
        initialized_(false),computed_(false),
        numInitialize_(0),numCompute_(0),numApplyInverse_(0),
        flopsInitialize_(0.0),flopsCompute_(0.0),flopsApplyInverse_(0.0),
        timeInitialize_(0.0),timeCompute_(0.0),timeApplyInverse_(0.0)
    {
    HYMLS_LPROF3(label_,"Constructor (1)");
    time_=Teuchos::rcp(new Epetra_Time(*comm_));
    
    if (SchurMatrix_==Teuchos::null && SchurComplement_==Teuchos::null)
      {
      HYMLS::Tools::Error("need either a CrsMatrix or a SchurComplement operator",
        __FILE__,__LINE__);
      }
    
    setParameterList(params);
          
    HYMLS_DEBVAR(myLevel_);
    HYMLS_DEBVAR(maxLevel_);

    if (myLevel_==maxLevel_)
      {
      // reindex the reduced system, this seems to be a good idea when
      // solving it using Ifpack_Amesos
      linearMap_ = Teuchos::rcp(new Epetra_Map(map_->NumGlobalElements(),
                                                  map_->NumMyElements(),
                                                  0, map_->Comm()) );

      reindexA_ = Teuchos::rcp(new ::EpetraExt::CrsMatrix_Reindex(*linearMap_));
      reindexX_ = Teuchos::rcp(new ::EpetraExt::MultiVector_Reindex(*linearMap_));
      reindexB_ = Teuchos::rcp(new ::EpetraExt::MultiVector_Reindex(*linearMap_));      
      }
    else
      {
      if (hid_==Teuchos::null)
        {
        Tools::Error("not on coarsest level and no HID available!",
                __FILE__,__LINE__);
        }
      }

  isEmpty_ = (map_->NumGlobalElements()==0);

  OT=Teuchos::rcp(new Householder(myLevel_));
  dumpVectors_=false;
#ifdef HYMLS_DEBUGGING
  dumpVectors_=true;
#endif
  return;
  }
  
  // destructor
  SchurPreconditioner::~SchurPreconditioner()
    {
    HYMLS_LPROF3(label_,"Destructor");
    }

  // Ifpack_Preconditioner interface
  
  void SchurPreconditioner::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& list)
    {
    HYMLS_LPROF3(label_,"setParameterList");
    setMyParamList(list);
    this->SetParameters(*list);
    // note - this class gets a few parameters from the big "Preconditioner"
    // list, which has been validated by the Preconditioner class already. So
    // we don't validate anything here.
    HYMLS_DEBVAR(PL());
    }

  // Sets all parameters for the preconditioner.
  int SchurPreconditioner::SetParameters(Teuchos::ParameterList& List)
    {
    HYMLS_LPROF3(label_,"SetParameters");
    Teuchos::RCP<Teuchos::ParameterList> myPL = getMyNonconstParamList();
    
    if (myPL.get()!=&List)
      {
      setMyParamList(Teuchos::rcp(&List,false));
      }
    
    maxLevel_=PL().get("Number of Levels",myLevel_);
    variant_ = PL().get("Preconditioner Variant","Block Diagonal");
    denseSwitch_=PL().get("Dense Solvers on Level",denseSwitch_);
    subdivideSeparators_=PL().get("Subdivide Separators",false);
    int pos=1;
    
    fix_gid_.resize(0);
    
    while (pos>0)
      {
      std::string label="Fix GID "+Teuchos::toString(pos);
      if (PL().isParameter(label))
        {
        fix_gid_.append(PL().get(label,0));
        pos++;
        }
      else
        {
        pos=0;
        }
      }
      
    HYMLS_DEBVAR(fix_gid_);
    
    return 0;
    }

  // Sets all parameters for the preconditioner.
  Teuchos::RCP<const Teuchos::ParameterList> SchurPreconditioner::getValidParameters() const
    {
    /*
    if (validParams_!=Teuchos::null) return validParams_;
    HYMLS_LPROF3(label_,"getValidParameters");
    
    VPL().set("Number of Levels",2,"If larger than 2, the method is applied recursively.");
    VPL().set("Subdivide Separators",false,"this has been implemented for the rotated "
        "B-grid and is not intended for general use right now.");
    VPL().set("Fix GID 1",-1,"enforce dirichlet condition on the coarsest level "
    "(for fixing the pressure, mainly)");
    VPL().set("Fix GID 2",-1,"enforce dirichlet condition on the coarsest level "
    "(for fixing the pressure, mainly)");
    */
    Tools::Warning("The SchurPreconditioner should not be used for validating the parameter list!",__FILE__,__LINE__);
    VPL().disableRecursiveValidation();
    return validParams_;
    }

  // force Compute() to re-initialize by deleting stuff
  int SchurPreconditioner::Initialize()
    {
    HYMLS_LPROF2(label_,"Initialize");
    time_->ResetStartTime();

    // force next Compute/InitializeCompute to rebuild everything
    sparseMatrixOT_=Teuchos::null;
    matrix_=Teuchos::null;
    vsumMap_=Teuchos::null;
    reducedSchurSolver_=Teuchos::null;
    if (myLevel_!=maxLevel_)
      {
      CHECK_ZERO(InitializeSeparatorGroups());
      CHECK_ZERO(InitializeOT());
      if (variant_=="Block Diagonal"||
          variant_=="Lower Triangular")
        {
        CHECK_ZERO(InitializeBlocks());
        }
      else if (variant_=="Domain Decomposition")
        {
        CHECK_ZERO(InitializeSingleBlock());
        }
      else if (variant_=="Do Nothing")
        {
        blockSolver_.resize(0);
        }
      else
        {
        Tools::Error("Variant '"+variant_+"'not implemented",
                __FILE__,__LINE__);
        }
      }
    numInitialize_++;
    initialized_=true;
    timeInitialize_+=time_->ElapsedTime();
    return 0;
    }


  int SchurPreconditioner::InitializeCompute()
    {
    if (isEmpty_)
      {
      return 0;
      }
    
    HYMLS_LPROF(label_,"InitializeCompute");

    if (myLevel_==maxLevel_)
      {
      // this should happen only if HYMLS is used as a direct method
      // ("Number of Levels" is 1) because then the Preconditioner  
      // class will construct a SchurPreconditioner on the coarsest 
      // level, otherwise a SchurPreconditioner constructs it with  
      // a sparse matrix directly.
      if (SchurComplement_==Teuchos::null && SchurMatrix_==Teuchos::null)
        {
        HYMLS::Tools::Error("no matrix and no SC object available on coarsest level",
              __FILE__,__LINE__);
        }
      if (SchurComplement_!=Teuchos::null)
        {
        HYMLS_DEBUG("This is probably a one-level method.");
        if (tmpMatrix_!=Teuchos::null) 
          {
          HYMLS_DEBUG("This is not the first call to InitializeCompute()");
          // Check that the SchurMatrix_ was previously constructed 
          // by this same function. As it is never passed out of the
          // object we can adjust it in that case.
          if (SchurMatrix_.get()!=tmpMatrix_.get())
            {
            Tools::Error("we seem to have both a SchurComplement object and a sparse\n"
                         " matrix representation. This case is not allowed!",
                         __FILE__,__LINE__);
            }
          }
        else
          {
          HYMLS_DEBUG("This is the first call to InitializeCompute()");
          tmpMatrix_ = Teuchos::rcp(new Epetra_FECrsMatrix(Copy,*map_,32));
          }
        SchurComplement_->Construct(tmpMatrix_);
        if (SchurMatrix_.get()!=tmpMatrix_.get())
          {
          HYMLS_DEBUG("Set Pointer.");
          SchurMatrix_ = tmpMatrix_;
          }
        }
#ifdef HYMLS_STORE_MATRICES
      HYMLS::MatrixUtils::Dump(*SchurMatrix_,"FinalSC.txt");
#endif

#if defined(HYMLS_STORE_MATRICES) || defined(HYMLS_TESTING)
      HYMLS::MatrixUtils::Dump(SchurMatrix_->RowMap(),"finalMap.txt");
#endif
      }
    else
      {
      if (variant_=="Block Diagonal"||
          variant_=="Lower Triangular")
        {
        CHECK_ZERO(InitializeBlocks());
        }
      else if (variant_=="Domain Decomposition")
        {
        CHECK_ZERO(InitializeSingleBlock());
        }
      else if (variant_=="Do Nothing")
        {
        blockSolver_.resize(0);
        }
      else
        {
        Tools::Error("Variant '"+variant_+"'not implemented",
                __FILE__,__LINE__);
        }

      if (SchurMatrix_!=Teuchos::null)
        {
        CHECK_ZERO(TransformAndDrop());
        }
      else if (SchurComplement_!=Teuchos::null)
        {
        CHECK_ZERO(AssembleTransformAndDrop());
        }
      else
        {
        Tools::Error("SchurComplement not accessible",__FILE__,__LINE__);
        }

      CHECK_ZERO(InitializeNextLevel())
      }    

    return 0;
    }


  // Computes all it is necessary to apply the preconditioner.
  int SchurPreconditioner::Compute()
    {
    if (isEmpty_) 
      {
      computed_=true;
      return 0;      
      }
    HYMLS_LPROF(label_,"Compute");
    CHECK_ZERO(this->InitializeCompute());

  time_->ResetStartTime();

  if (myLevel_==maxLevel_)
    {
    // drop numerical zeros. We need to copy the matrix anyway because
    // we may want to put in some artificial Dirichlet conditions and 
    // scale the matrix.
#ifdef HYMLS_TESTING
      Tools::Out("drop on coarsest level");
#endif

    reducedSchur_ = MatrixUtils::DropByValue(SchurMatrix_, 
        HYMLS_SMALL_ENTRY, MatrixUtils::RelFullDiag);

    HYMLS_TEST(Label(),
            isFmatrix(*reducedSchur_),
            __FILE__,__LINE__);

  reducedSchur_->SetLabel(("Coarsest Matrix (level "+Teuchos::toString(myLevel_+1)+")").c_str());

  if (!HaveBorder())
    {
    for (int i=0;i<fix_gid_.length();i++)
      {
      HYMLS_DEBUG("set Dirichlet node "<<fix_gid_[i]);
      CHECK_ZERO(MatrixUtils::PutDirichlet(*reducedSchur_,fix_gid_[i]));
      }
    }

    // compute scaling for reduced Schur
    CHECK_ZERO(ComputeScaling(*reducedSchur_,reducedSchurScaLeft_,reducedSchurScaRight_));
    
    HYMLS_DEBUG("scale matrix");
    CHECK_ZERO(reducedSchur_->LeftScale(*reducedSchurScaLeft_));
    CHECK_ZERO(reducedSchur_->RightScale(*reducedSchurScaRight_));

    HYMLS_DEBUG("reindex matrix to linear indexing");
    linearMatrix_ = Teuchos::rcp(&((*reindexA_)(*reducedSchur_)),false);

    // passed to direct solver - depends on what exactly we do
    Teuchos::RCP<Epetra_RowMatrix> S2 = Teuchos::null;

#ifdef RESTRICT_ON_COARSE_LEVEL
    // restrict the matrix to the active processors
    if (restrictA_==Teuchos::null)
      {
      restrictA_ = Teuchos::rcp(new ::EpetraExt::RestrictedCrsMatrixWrapper());
      restrictX_ = Teuchos::rcp(new ::EpetraExt::RestrictedMultiVectorWrapper());
      restrictB_ = Teuchos::rcp(new ::EpetraExt::RestrictedMultiVectorWrapper());
      }
      // we have to restrict_comm again because the pointer is no longer
      // valid, it seems
#ifdef OLD_TRILINOS      
    // bug in Trilinos 10.10, uninitialized return
    restrictA_->restrict_comm(linearMatrix_);
#else
    CHECK_ZERO(restrictA_->restrict_comm(linearMatrix_));      
#endif
    amActive_=restrictA_->RestrictedProcIsActive();
    restrictX_->SetMPISubComm(restrictA_->GetMPISubComm());
    restrictB_->SetMPISubComm(restrictA_->GetMPISubComm());

    restrictedMatrix_=restrictA_->RestrictedMatrix();
    int reducedNumProc = -1;
      if (restrictA_->RestrictedProcIsActive())
        {
        reducedNumProc=restrictA_->RestrictedComm().NumProc();
        }

    // if we do not set this, Amesos may try to think of its own strategy
    // to reduce the number of procs, which in my experience leads to MPI
    // errors in MPI_Comm_free (as of Trilinos 10.0)
    if (PL().sublist("Coarse Solver").isParameter("MaxProcs")==false)
      {
      PL().sublist("Coarse Solver").set("MaxProcs",reducedNumProc);
      }
    HYMLS_DEBUG("next SC defined as restricted linear-index matrix");
    S2=restrictedMatrix_;
#else
    HYMLS_DEBUG("next SC defined as linear-index matrix");
    S2=linearMatrix_;
    amActive_=true;
#endif

////////////////////////////////////////////////////////////////////////////
// this next section is just for the bordered case                        //
////////////////////////////////////////////////////////////////////////////
  if (HaveBorder())
    {
    if (borderV_==Teuchos::null || borderW_==Teuchos::null || borderC_==Teuchos::null)
      {
      Tools::Error("border not set correctly",__FILE__,__LINE__);
      }
#ifndef RESTRICT_ON_COARSE_LEVEL
    // we use the variant with restricting the number of ranks in comm usually
    Tools::Error("not implemented",__FILE__,__LINE__);
#endif

HYMLS_DEBVAR(*borderV_);
HYMLS_DEBVAR(*borderW_);
HYMLS_DEBVAR(*borderC_);

    // we need to create views of the vectors here because the
    // map is different for the solver (linear restricted map)
    Teuchos::RCP<const Epetra_MultiVector> Vprime =
        Teuchos::rcp(new Epetra_MultiVector(View,restrictedMatrix_->RowMap(),
                borderV_->Values(),borderV_->Stride(),borderV_->NumVectors()));
    Teuchos::RCP<const Epetra_MultiVector> Wprime =
        Teuchos::rcp(new Epetra_MultiVector(View,restrictedMatrix_->RowMap(),
                borderW_->Values(),borderW_->Stride(),borderW_->NumVectors()));

    // create AugmentedMatrix, refactor reducedSchurSolver_
    augmentedMatrix_ = Teuchos::rcp
        (new HYMLS::AugmentedMatrix(restrictedMatrix_,Vprime,Wprime,borderC_));
    S2=augmentedMatrix_;
    }
    
////////////////////////////////////////////////////////////////////////////
// end bordered case section                                              //
////////////////////////////////////////////////////////////////////////////

    Teuchos::ParameterList& amesosList=PL().sublist("Coarse Solver");                    
    if (amActive_)
      {
      if (S2==Teuchos::null)
        {
        Tools::Error("failed to select matrix for coarsest level",__FILE__,__LINE__);
        }
      reducedSchurSolver_= Teuchos::rcp(new Ifpack_Amesos(S2.get()));
      CHECK_ZERO(reducedSchurSolver_->SetParameters(amesosList));
      HYMLS_DEBUG("Initialize direct solver");
      CHECK_ZERO(reducedSchurSolver_->Initialize());
      }
    }
  else // not on coarsest level
    {
#if defined(HYMLS_STORE_MATRICES)
    // dump a reordering for the Schur-complement (for checking in MATLAB)
    Teuchos::RCP<const HierarchicalMap>
        sepObject = hid_->Spawn(HierarchicalMap::LocalSeparators);
    std::string postfix = "_"+Teuchos::toString(myLevel_)+".txt";
    std::ofstream ofs(("pS"+postfix).c_str());
    std::ofstream ofs1(("pS1"+postfix).c_str());
    std::ofstream ofs2(("pS2"+postfix).c_str());
    std::ofstream begI(("begI"+postfix).c_str());
    std::ofstream begS(("begS"+postfix).c_str());
          
    bool linear_indices=true;
    
    Teuchos::RCP<const Epetra_Map> newMap=map_;
    
    if (linear_indices)
      {
      int myLength = map_->NumMyElements();
      newMap=Teuchos::rcp(new Epetra_Map(-1,myLength,0,*comm_));
      }
      
    int off = 0;
    for (int sd=0;sd<hid_->NumMySubdomains();sd++)
      {
      begI << off << std::endl;
      off = off + hid_->NumInteriorElements(sd);
      }
    begI << off << std::endl;
    
    int offset=0;
          
    for (int sep=0;sep<sepObject->NumMySubdomains();sep++)
      {
      for (int grp=0;grp<sepObject->NumGroups(sep);grp++)
        {
        begS << offset << std::endl;
        offset = offset + sepObject->NumElements(sep,grp);
        //begS << sepObject->LID(sep,grp,0)<<std::endl;
        
        // V-sum nodes
        ofs << newMap->GID(map_->LID(sepObject->GID(sep,grp,0))) << std::endl;
        ofs2 << newMap->GID(map_->LID(sepObject->GID(sep,grp,0))) << std::endl;
        // non-Vsum nodes
        for (int j=1;j<sepObject->NumElements(sep,grp);j++)
          {
          ofs << newMap->GID(map_->LID(sepObject->GID(sep,grp,j))) << std::endl;
          ofs1 << newMap->GID(map_->LID(sepObject->GID(sep,grp,j))) << std::endl;
          }
        }
      }
    
    begS << matrix_->NumMyRows()<<std::endl;
    
    ofs.close();
    ofs1.close();
    ofs2.close();
    begI.close();
    begS.close();

    MatrixUtils::Dump(*matrix_,"SchurPreconditioner"+Teuchos::toString(myLevel_)+".txt");
#endif

  // compute LU decompositions of blocks...
  {
  HYMLS_LPROF(label_,"factor blocks");
  for (int i=0;i<blockSolver_.size();i++)
    {
    CHECK_ZERO(blockSolver_[i]->Compute(*matrix_));
    }
  }
  
  // extract the Vsum part of the preconditioner (reduced Schur)
  CHECK_ZERO(reducedSchur_->Import(*matrix_, *vsumImporter_, Insert));

  //TODO: compute actual Schur complement rather than just using the part K22
  // ...

  CHECK_ZERO(reducedSchur_->FillComplete(*vsumMap_,*vsumMap_));

  //  DropByValue before going to the next level. Careful about
  //  the pointer, though, which is shared with the next level 
  //  solver...
#ifdef HYMLS_STORE_MATRICES
    MatrixUtils::Dump(*reducedSchur_,"ReducedSchurBeforeDropping"+Teuchos::toString(myLevel_)+".txt");
#endif  

#ifdef HYMLS_TESTING
  Tools::Out("drop before going to next level");
#endif
  Teuchos::RCP<Epetra_CrsMatrix> tmp = MatrixUtils::DropByValue(reducedSchur_,
        HYMLS_SMALL_ENTRY, MatrixUtils::RelDropDiag);
  *reducedSchur_ = *tmp; 
  tmp=Teuchos::null;
  
#ifdef HYMLS_STORE_MATRICES
    MatrixUtils::Dump(*reducedSchur_,"ReducedSchur"+Teuchos::toString(myLevel_)+".txt");
#endif  

    }
  
  if (amActive_)
    {
    // compute solver for reduced Schur
    HYMLS_DEBUG("compute coarse solver");
    int ierr=reducedSchurSolver_->Compute();

    if (ierr!=0)
      {
#ifdef HYMLS_STORE_MATRICES
      Teuchos::RCP<const Epetra_CrsMatrix> dumpMatrix
        = reducedSchur_;
      if (myLevel_==maxLevel_)
        {
        dumpMatrix = linearMatrix_;
        }
        MatrixUtils::Dump(*dumpMatrix,"BadMatrix"+Teuchos::toString(myLevel_)+".txt");
#endif      
      Tools::Error("factorization returned value "+Teuchos::toString(ierr)+
          " on level "+Teuchos::toString(myLevel_),__FILE__,__LINE__);
      }
    }

  computed_ = true;
  timeCompute_ += time_->ElapsedTime();
  numCompute_++;
  return 0;
  }

/// private init/compute functions /////

int SchurPreconditioner::InitializeBlocks()
  {
  HYMLS_LPROF2(label_,"InitializeBlocks");
  // get an object with only local separators and remote connected separators:
  Teuchos::RCP<const HierarchicalMap> sepObject
      = hid_->Spawn(HierarchicalMap::LocalSeparators);
    
    // number of blocks in this preconditioner (except next Schur-complement).
    // Some blocks may ultimately have 0 rows if they had only one element which
    // is retained as a 'Vsum'-node. That doesn't bother the solver, though.
    int numBlocks=0;
    for (int i=0;i<sepObject->NumMySubdomains();i++)
      {
      numBlocks+=sepObject->NumGroups(i);
      }

#ifdef HYMLS_TESTING
    for (int i=0;i<sepObject->NumMySubdomains();i++)
      {
      for (int grp=0;grp<sepObject->NumGroups(i);grp++)
        {
        if (sepObject->NumElements(i,grp)==0)
          {
          HYMLS::Tools::Warning("there is an empty separator, which is probably dangerous",
            __FILE__,__LINE__);
          }
        }
      }
#endif
        
  // create an array of solvers for all the diagonal blocks
  blockSolver_.resize(numBlocks);
  double nnz=0.0;
  int blk=0;
  for (int sep=0;sep<sepObject->NumMySubdomains();sep++)
    {
    for (int grp=0;grp<sepObject->NumGroups(sep);grp++)
      {
      // in the spawned sepObject, each local separator is a group of a subdomain.
      // -1 because we remove one Vsum node from each block
      int numRows=std::max(sepObject->NumElements(sep,grp)-1,0);
      nnz+=numRows*numRows; 
      blockSolver_[blk]=Teuchos::rcp(new 
             Ifpack_DenseContainer(numRows));
      CHECK_ZERO(blockSolver_[blk]->SetParameters(
              PL().sublist("Dense Solver")));

      CHECK_ZERO(blockSolver_[blk]->Initialize());

      for (int j=0; j<numRows; j++)
        {
        int gid=sepObject->GID(sep,grp,j+1); // skip first element, which is a Vsum
        int LRID = map_->LID(gid);
        blockSolver_[blk]->ID(j) = LRID;
        }
      blk++;
      }
    }
  REPORT_SUM_MEM(label_,"dense diagonal blocks",nnz,0,comm_);
  return 0;  
  }

int SchurPreconditioner::InitializeSingleBlock()
  {
  HYMLS_LPROF2(label_,"InitializeSingleBlock");
  // get an object with only local separators and remote connected separators:
  Teuchos::RCP<const HierarchicalMap> sepObject
      = hid_->Spawn(HierarchicalMap::LocalSeparators);
    
    // count the number of owned elements and vsums
    int numMyVsums=0;
   int numMyElements = 0;
    for (int i=0;i<sepObject->NumMySubdomains();i++)
      {
      numMyElements+=sepObject->NumElements(i);
      numMyVsums+=sepObject->NumGroups(i);
      }
   // we actually need the number of owned non-Vsums:
   int numRows = numMyElements - numMyVsums;
  
  // create a single solver for all the non-Vsums
  blockSolver_.resize(1);
  blockSolver_[0]=Teuchos::rcp
        (new Ifpack_SparseContainer<Ifpack_Amesos>(numRows));
  CHECK_ZERO(blockSolver_[0]->SetParameters(
              PL().sublist("Sparse Solver")));
  CHECK_ZERO(blockSolver_[0]->Initialize());
  int pos=0;
  for (int sep=0;sep<sepObject->NumMySubdomains();sep++)
    {
    for (int grp=0;grp<sepObject->NumGroups(sep);grp++)
      {      
      // skip first element, which is a Vsum
      for (int j=1; j<sepObject->NumElements(sep,grp); j++)
        {
        int gid=sepObject->GID(sep,grp,j);
        int LRID = map_->LID(gid);
        blockSolver_[0]->ID(pos++) = LRID;
        }
      }
    }
  REPORT_SUM_MEM(label_,"single diagonal block (not counted)",0.0,0,comm_);
  return 0;  
  }


int SchurPreconditioner::InitializeSeparatorGroups()
  {
  HYMLS_LPROF2(label_,"InitializeSeparatorGroups");
  if (subdivideSeparators_)
    {
    if (SchurMatrix_==Teuchos::null)
      {
      Tools::Error("for splitting separators we need an assembled\n"
                   "Schur-Complement",__FILE__,__LINE__);
      }
    int dof=PL("Problem").sublist("Partitioner")
                  .get("Degrees of Freedom",-1);
    if (dof==-1)
      {
      HYMLS::Tools::Error("'Degrees of Freedom' parameter not set!",
              __FILE__,__LINE__);
      }
    int pressure=PL().get("Subdivide based on variable",-1);
    if (pressure==-1)
      {
      HYMLS::Tools::Error("'Subdivide based on variable' parameter not set!",
              __FILE__,__LINE__);
      }
    Teuchos::RCP<const HierarchicalMap> sepObject
        = hid_->Spawn(HierarchicalMap::Separators);
    Teuchos::RCP<Teuchos::Array<HYMLS::SepNode> > sepList 
        = Teuchos::rcp(new Teuchos::Array<HYMLS::SepNode>(sepObject->NumMyElements()));
    Teuchos::Array<int> connectedPs(2); // typically there are exactly two p-couplings
    for (int sep=0;sep<sepObject->NumMySubdomains();sep++)
      {
      int grp=0; // the standard Separator object has local separators as group 0
                 // of its 'subdomains'
      for (int j=0;j<sepObject->NumElements(sep,grp);j++)
        {
        int lsid = sepObject->LID(sep,grp,j);// local separator ID
        int gid = sepObject->GID(sep,grp,j); // global ID
        int lrid = SchurMatrix_->LRID(gid); // local row ID
        int* indices;
        double* values;
        int len;
        int type = -1;
        CHECK_ZERO(SchurMatrix_->ExtractMyRowView(lrid,len,values,indices));
        int pos=0;
        connectedPs[0]=-1;
        connectedPs[1]=-1;// will remain there if not connected to P-nodes -> own group
        for (int k=0;k<len;k++)
          {
          int gcid = SchurMatrix_->GCID(indices[k]);
          if (MOD(gcid,dof)==pressure)
            {
            if (std::abs(values[k])>1.0e-8)
              {
              if (pos==0)
                {
                // distinguish between [+1 -1] and [-1 +1] type p-couplings
                type = values[k]>0? 1:0;
                }
              connectedPs[pos++]=gcid;
              }
            if (pos>=2)
              {
              break;
              }
            }
          }
        SepNode S(gid,connectedPs,type);
        (*sepList)[lsid] = S;
        }
      }
      
    hid_->Spawn(HierarchicalMap::LocalSeparators,sepList);
    }
  else
    {
    hid_->Spawn(HierarchicalMap::LocalSeparators);
    }

#ifdef HYMLS_DEBUGGING
    std::ofstream ofs1,ofs2;
    if (myLevel_==1)
      {
      ofs1.open("sep_data.m",std::ios::out);
      ofs2.open("lsep_data.m",std::ios::out);
      }
    else
      {
      ofs1.open("sep_data.m",std::ios::app);
      ofs2.open("lsep_data.m",std::ios::app);
      }
    ofs1<<*(hid_->Spawn(HierarchicalMap::Separators));
    ofs1.close();
    ofs2<<*(hid_->Spawn(HierarchicalMap::LocalSeparators));
    ofs2.close();
#endif      

  return 0;
  }

int SchurPreconditioner::InitializeOT()
  {
  HYMLS_LPROF2(label_,"InitializeOT");
  
  // create orthogonal transform as a sparse matrix representation
  if (sparseMatrixOT_==Teuchos::null)
    {

    // Get an object with only local separators and remote connected separators.
    Teuchos::RCP<const HierarchicalMap> sepObject
        = hid_->Spawn(HierarchicalMap::LocalSeparators);
        
    // import our test vector into the map of this object (to get the off-processor
    // separators connected to local subdomains). The separators are unique in this object, 
    // so the Map() and OverlappingMap() are the same.
    const Epetra_Map& sepMap = sepObject->Map();
    Epetra_Import import(sepMap,*map_);
    Epetra_Vector localTestVector(sepMap);
    CHECK_ZERO(localTestVector.Import(*testVector_,import,Insert));

    Epetra_IntSerialDenseVector inds;
    Epetra_SerialDenseVector vec;
    
    int nnzPerRow = sepObject->NumMySubdomains()>0? sepObject->NumInteriorElements(0) : 
    0;

    sparseMatrixOT_ = Teuchos::rcp(new
        Epetra_CrsMatrix(Copy,*map_,nnzPerRow));

    // loop over all separators connected to a local subdomain
    for (int sep=0;sep<sepObject->NumMySubdomains();sep++)
      {
      HYMLS_DEBVAR(sep);
      // the LocalSeparator object has only local separators, but it may
      // have several groups due to splitting of groups (i.e. for the B-grid,
      // where velocities are grouped depending on how they connect ot the pressures)
      for (int grp=0;grp<sepObject->NumGroups(sep);grp++)
        {
//        HYMLS_DEBVAR(grp);
        int len = sepObject->NumElements(sep,grp);
        if ((inds.Length()!=len) && (len>0))
          {
          inds.Size(len);
          vec.Size(len);
          }
        for (int j=0;j<len;j++)
          {
          inds[j] = sepObject->GID(sep,grp,j);
          vec[j] = localTestVector[sepMap.LID(inds[j])];
          }
        if (len>0)
          {
//          HYMLS_DEBVAR(inds);
//          HYMLS_DEBVAR(vec);
          int ierr=OT->Construct(*sparseMatrixOT_,inds,vec);
          if (ierr)
            {
            Tools::Warning("Error code "+Teuchos::toString(ierr)+" returned from Epetra call!",
                        __FILE__, __LINE__);                        
            return ierr;
            }
          }
        }    
      }
    
    CHECK_ZERO(sparseMatrixOT_->FillComplete())
    }
#ifdef HYMLS_STORE_MATRICES
  MatrixUtils::Dump(*sparseMatrixOT_, 
        "Householder"+Teuchos::toString(myLevel_)+".txt");
#endif  
  return 0;
  }
    
  int SchurPreconditioner::InitializeNextLevel()
    {
    HYMLS_LPROF2(label_,"InitializeNextLevel");

    Teuchos::RCP<const HierarchicalMap>
        sepObject = hid_->Spawn(HierarchicalMap::LocalSeparators);
    
    if (vsumMap_==Teuchos::null)
      {
      int numBlocks = 0;
      for (int i=0;i<sepObject->NumMySubdomains();i++)
        {
        for (int j=0;j<sepObject->NumGroups(i);j++)
          {
          if (sepObject->NumElements(i,j)>0) numBlocks++;
          }
        }
    
      HYMLS_DEBVAR(numBlocks);            

      // create a map for the reduced Schur-complement. Note that this is a distributed
      // matrix, in contrast to the other diagonal blocks, so we can't use an Ifpack 
      // container.
      int *MyVsumElements = new int[numBlocks]; // one Vsum per block
      int pos=0;
      for (int sep=0;sep<sepObject->NumMySubdomains();sep++)
        {
        HYMLS_DEBVAR(sep)
        for (int grp = 0 ; grp < sepObject->NumGroups(sep) ; grp++)
          {
          if (sepObject->NumElements(sep,grp)>0)
            {
            MyVsumElements[pos++] = sepObject->GID(sep,grp,0);
            }
          }
        }
        
      vsumMap_=Teuchos::rcp(new Epetra_Map(-1,numBlocks,MyVsumElements,
                                map_->IndexBase(), map_->Comm()));
      
      delete [] MyVsumElements;
      HYMLS_DEBUG(label_);
      HYMLS_DEBVAR(*vsumMap_);

      vsumRhs_ = Teuchos::rcp(new Epetra_MultiVector(*vsumMap_,1));
      vsumSol_ = Teuchos::rcp(new Epetra_MultiVector(*vsumMap_,1));

      // the vsums are still distributed and we must
      // form a correct col map
      vsumColMap_ = MatrixUtils::CreateColMap(*matrix_,*vsumMap_,*vsumMap_);

      reducedSchur_=Teuchos::rcp(new 
            Epetra_CrsMatrix(Copy,*vsumMap_,*vsumColMap_,matrix_->MaxNumEntries()));
            
      vsumImporter_=Teuchos::rcp(new Epetra_Import(*vsumMap_,*map_));
      }
      
    if (reducedSchur_->Filled()) reducedSchur_->PutScalar(0.0);
      
    // import sparsity pattern for S2
    // extract the Vsum part of the preconditioner (reduced Schur)
    CHECK_ZERO(reducedSchur_->Import(*matrix_, *vsumImporter_, Insert));
      
    //TODO: actual Schur Complement
    CHECK_ZERO(reducedSchur_->FillComplete(*vsumMap_,*vsumMap_));

    // drop numerical zeros so that the domain decomposition works
#ifdef HYMLS_TESTING
    Tools::Out("drop because of next DD");
#endif
    reducedSchur_ = MatrixUtils::DropByValue(reducedSchur_, HYMLS_SMALL_ENTRY,
                        MatrixUtils::RelDropDiag);
  
    reducedSchur_->SetLabel(("Matrix (level "+Teuchos::toString(myLevel_+1)+")").c_str());
  
#ifdef HYMLS_TESTING
  this->Visualize("hid_data_deb.m",false);
#endif


  HYMLS_DEBUG("Create solver for reduced Schur");

  nextLevelParams_ = Teuchos::rcp(new Teuchos::ParameterList(*getMyParamList()));

  Teuchos::RCP<Epetra_Vector> nextTestVector = Teuchos::null;
  
  if (myLevel_+1!=maxLevel_)
    {
    if (nextLevelHID_==Teuchos::null)
      {
      bool stat=true;
      try {
      nextLevelHID_ = hid_->SpawnNextLevel(reducedSchur_,nextLevelParams_);
      } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr,stat);
      if (!stat) Tools::Fatal("Failed to create next level ordering",__FILE__,__LINE__);
      }
    if (reducedSchurSolver_==Teuchos::null)
      {
      Epetra_Vector transformedTestVector(*map_);

      CHECK_ZERO(OT->Apply(transformedTestVector, *sparseMatrixOT_, *testVector_))
    
      nextTestVector = Teuchos::rcp(new Epetra_Vector(*vsumMap_));

      CHECK_ZERO(nextTestVector->Import(transformedTestVector, *vsumImporter_, Insert));    

      // create another level of HYMLS::Preconditioner,
      if (myLevel_>=denseSwitch_-1)
        {
        nextLevelParams_->sublist("Preconditioner").set("Subdomain Solver Type","Dense");
        }
  
      //TODO: move the direct solver thing to the Preconditioner class and rename
      //      the SchurPreconditioner SchurApproximation. Then this call can be put
      //      outside the if statement because we will always create a Preconditioner
      //      object for the reduced problem.
      reducedSchurSolver_= Teuchos::rcp(new
          Preconditioner(reducedSchur_,nextLevelParams_,nextLevelHID_,
          myLevel_+1, nextTestVector));
      }
    else
      {
      Teuchos::RCP<Preconditioner> prec=Teuchos::rcp_dynamic_cast<Preconditioner>
        (reducedSchurSolver_);
      if (prec==Teuchos::null) Tools::Error("dynamic cast failed",__FILE__,__LINE__);
      prec->SetMatrix(reducedSchur_);
      }
    }
  else
    {
    // fix pressure on coarsest level:
    for (int i=0;i<fix_gid_.length();i++)
      {
      CHECK_ZERO(MatrixUtils::PutDirichlet(*reducedSchur_,fix_gid_[i]));
      }
    reducedSchurSolver_= Teuchos::rcp(new
        SchurPreconditioner(reducedSchur_,nextLevelHID_,nextLevelParams_,
        myLevel_+1, nextTestVector));
    }
    
  HYMLS_DEBUG("Initialize solver for reduced Schur");
  CHECK_ZERO(reducedSchurSolver_->Initialize());
  return 0;
  }

  int SchurPreconditioner::TransformAndDrop()
    {
    HYMLS_LPROF2(label_,"TransformAndDrop");

    // currently we simply compute T'*S*T using a sparse matmul.
    // I tried more fancy block-variants, but they were quite tedious
    // to implement and also slower than this.
    // We do not actually drop anything here, that happens automatically
    // by the reducedSchur import and the block solvers.
    if (matrix_==Teuchos::null || OT->SaveMemory())
      {
      matrix_=OT->Apply(*sparseMatrixOT_, *SchurMatrix_);
      }
    else
      {
      OT->Apply(*matrix_,*sparseMatrixOT_, *SchurMatrix_);
      }

    CHECK_ZERO(matrix_->FillComplete());
    REPORT_SUM_MEM(label_,"Transformed SC",matrix_->NumMyNonzeros(),
                                       matrix_->NumMyNonzeros(),
                                       comm_);
    HYMLS_TEST(Label(),
            noPcouplingsDropped(*matrix_,*hid_->Spawn(HierarchicalMap::LocalSeparators)),
            __FILE__,__LINE__);
    return 0;
    }


  // alternative implementation without previously assembling the SC
  // (saves some memory)
  int SchurPreconditioner::AssembleTransformAndDrop()
    {
    std::string timerLabel="AssembleTransformAndDrop";

    if (SchurComplement_==Teuchos::null) Tools::Error("SC not available in unassembled form", __FILE__,__LINE__);

    Teuchos::RCP<Epetra_FECrsMatrix> matrix = 
        Teuchos::rcp_dynamic_cast<Epetra_FECrsMatrix>(matrix_);

    if (matrix==Teuchos::null)
      {
      timerLabel=timerLabel+" (first call)";
      }

    HYMLS_LPROF2(label_,timerLabel);
        
    if (matrix==Teuchos::null)
      {
      int nzest = 0;
      if (hid_->NumMySubdomains()>0)
        {
        nzest = hid_->NumElements(0);
        if (hid_->NumGroups(0)>0) nzest -= hid_->NumElements(0,0);
        }
      matrix = Teuchos::rcp(new Epetra_FECrsMatrix(Copy,*map_,nzest));
      matrix_=matrix;
      }
    
    Epetra_IntSerialDenseVector indices;
    Epetra_IntSerialDenseVector sep;
    Epetra_SerialDenseVector v;
    Epetra_SerialDenseMatrix Sk;
        
    // part remaining after dropping
    Epetra_SerialDenseMatrix Spart;
    Epetra_IntSerialDenseVector indsPart;

    Teuchos::RCP<Epetra_CrsMatrix> transformedA22 =
    OT->Apply(*sparseMatrixOT_,SchurComplement_->A22());
    
#ifdef HYMLS_DEBUGGING
  std::string s1 = "SchurPrecond"+Teuchos::toString(myLevel_)+"_";
/*
  MatrixUtils::Dump(*matrix_,s1+"Pattern.txt");
  // not Filled yet
  //MatrixUtils::Dump(*transformedA22,s1+"TransformedA22.txt");
*/
#endif

    if (transformedA22->RowMap().SameAs(matrix->RowMap())==false)
      {
      HYMLS::Tools::Error("mismatched maps",__FILE__,__LINE__);
      }

    int len;
    int maxlen = transformedA22->GlobalMaxNumEntries();
    int* cols = new int[maxlen];
    double* values = new double[maxlen];

    // put H'*A22*H into matrix_
    if (!matrix->Filled())
      {
      // start out by just putting the structure together.
      // I do this because the SumInto function will fail 
      // unless the values have been put in already. On the
      // other hand, the Insert function will overwrite stuff
      // we put in previously.
      //
      // NOTE: this is where the 'dropping' occurs, so if we
      //       want to implement different schemes we have  
      //       to adjust this loop in the first place.      
      for (int sd=0;sd<hid_->NumMySubdomains();sd++)
        {
        // put in the Vsum-Vsum couplings
        int numVsums = hid_->NumGroups(sd);
        indsPart.Resize(numVsums);
        if (numVsums>Spart.N()) Spart.Reshape(2*numVsums,2*numVsums);
        numVsums=0;
        for (int grp=1;grp<hid_->NumGroups(sd);grp++)
          {
          if (hid_->NumElements(sd,grp)>0)
            {
            indsPart[numVsums++]=hid_->GID(sd,grp,0);
            }
          }
        indsPart.Resize(numVsums);
        //HYMLS_DEBVAR(sd);
        //HYMLS_DEBVAR(indsPart);
        CHECK_NONNEG(matrix->InsertGlobalValues(indsPart.Length(),
                indsPart.Values(), Spart.A()));
        // now the non-Vsums
        for (int grp=1;grp<hid_->NumGroups(sd);grp++)
          {
          int len = hid_->NumElements(sd,grp)-1;
          indsPart.Resize(len);
          if (Spart.N()<len) Spart.Reshape(2*len,2*len);
          for (int j=0;j<len;j++)
            {
            indsPart[j]=hid_->GID(sd,grp,1+j);
            }//j
          //HYMLS_DEBVAR(indsPart);
          CHECK_NONNEG(matrix->InsertGlobalValues(indsPart.Length(),
                        indsPart.Values(),Spart.A()));
          }
        }
      // assemble with all zeros
      HYMLS_DEBVAR("assemble pattern of transformed SC");
      CHECK_ZERO(matrix->GlobalAssemble(false));

      // now fill with H'*A22*H
      for (int i=0;i<matrix_->NumMyRows();i++)
        {
        //global row id
        int grid = transformedA22->GRID(i);
        CHECK_ZERO(transformedA22->ExtractGlobalRowCopy(grid,maxlen,len,values,cols));

#ifdef HYMLS_TESTING
        //before we would sum the value because of duplicate entries,
        //but here we check that they don't exist and instead just insert them
        std::vector<int> colvec(len);
        colvec.assign(cols, cols+len);
        std::sort(colvec.begin(), colvec.end());
        for (int j = 0; j < len - 1; j++)
          {
          if (colvec[j] == colvec[j + 1])
            {
            Tools::Error("Duplicate entries on row "+Teuchos::toString(grid)+ " column "+Teuchos::toString(colvec[j]),__FILE__,__LINE__);
            }
          }
#endif
        CHECK_NONNEG(matrix_->InsertGlobalValues(grid,len,values,cols));
        }
      CHECK_ZERO(matrix_->FillComplete());
      }
    else
      {
      CHECK_ZERO(matrix->PutScalar(0.0));

      for (int i=0;i<matrix_->NumMyRows();i++)
        {
        //global row id
        int grid = transformedA22->GRID(i);
        CHECK_ZERO(transformedA22->ExtractGlobalRowCopy(grid,maxlen,len,values,cols));

#ifdef HYMLS_TESTING
        //check that the pattern didn't change
        int len2;
        int maxlen2 = matrix_->GlobalMaxNumEntries();
        int* cols2 = new int[maxlen2];
        double* values2 = new double[maxlen2];
        CHECK_ZERO(matrix_->ExtractGlobalRowCopy(grid,maxlen2,len2,values2,cols2));

        std::vector<int> colvec(len);
        colvec.assign(cols, cols+len);
        std::sort(colvec.begin(), colvec.end());

        std::vector<int> colvec2(len2);
        colvec2.assign(cols2, cols2+len2);
        std::sort(colvec2.begin(), colvec2.end());

        int j2 = 0;
        for (int j = 0; j < len; j++)
          {
          for (; j2 < len2; j2++)
            {
            if (colvec[j] == colvec2[j2])
              break;

            // We iterated past the number in colvec[j] so it is not in colvec2
            if (colvec2[j2] > colvec[j])
              break;
            }
          // colvec[j] is not anywhere before the last number in colvec2
          if (j2 == len2 || colvec2[j2] > colvec[j])
            {
            // Find the index of the value
            for (int j3 = 0; j3 < len; j3++)
              if (colvec[j] == cols[j3])
                {
                Tools::Warning("Pattern is different on row "
                  + Teuchos::toString(grid) + " column "
                  + Teuchos::toString(colvec[j]) + " value "
                  + Teuchos::toString(values[j3]), __FILE__, __LINE__);
                break;
                }
            }
          }

        delete [] cols2;
        delete [] values2;
#endif
        CHECK_NONNEG(matrix_->ReplaceGlobalValues(grid,len,values,cols));
        }
      }

    // free temporary storage
    delete [] values;
    delete [] cols;
    transformedA22=Teuchos::null;
#ifdef HYMLS_STORE_MATRICES
//  HYMLS::MatrixUtils::Dump(*matrix_,s1+"_TransDroppedA22.txt");
#endif

    // Get an object with all separators connected to local subdomains
    Teuchos::RCP<const HierarchicalMap> sepObject
        = hid_->Spawn(HierarchicalMap::Separators);
        
    // import our test vector into the map of this object (to get the off-processor
    // separators connected to local subdomains). The separators are unique in this object, 
    // so the Map() and OverlappingMap() are the same.
    const Epetra_Map& sepMap = sepObject->OverlappingMap();
    Epetra_Import import(sepMap,*map_);
    Epetra_Vector localTestVector(sepMap);
    CHECK_ZERO(localTestVector.Import(*testVector_,import,Insert));

    // now for each subdomain construct the SC part A21*A11\A12 for the      
    // surrounding separators, apply orthogonal transforms to each separator 
    // group and sum them into the pattern defined above, dropping everything
    // that is not defined in the matrix pattern.
     
    // loop over all subdomains
    for (int sd=0;sd<hid_->NumMySubdomains();sd++)
      {
      // construct the local contribution of the SC
      // (for all separators around the subdomain) 

      // Get the global indices of the separators
      CHECK_ZERO(hid_->getSeparatorGIDs(sd, indices));

      // Construct the local -A21*A11\A12
      CHECK_ZERO(SchurComplement_->Construct(sd, Sk, indices));
      CHECK_ZERO(Sk.Scale(-1.0));

      // Get the part of the testvector that belongs to the
      // separators
      v.Resize(indices.Length());
      for (int i = 0; i < indices.Length(); i++)
        {
        int gid = indices[i];
        v[i] = localTestVector[sepMap.LID(gid)];
        }

      int pos = 0;
      // Loop over all separators of the subdomain sd
      for (int grp = 1; grp < hid_->NumGroups(sd); grp++)
        {
        int len = hid_->NumElements(sd, grp);
        Epetra_SerialDenseVector vView(View, &v[pos], len);

        // Apply the orthogonal transformation for each group
        // separately
        RestrictedOT::Apply(Sk, pos, *OT, vView);

        pos += len;
        }

      CHECK_NONNEG(matrix->SumIntoGlobalValues(indices,Sk));
      }//sd

    HYMLS_DEBUG("assemble transformed/dropped SC");
    CHECK_ZERO(matrix->GlobalAssemble(true));

#ifdef HYMLS_STORE_MATRICES
    MatrixUtils::Dump(*matrix_,"SchurPreconditioner"+Teuchos::toString(myLevel_)+".txt");
#endif
    REPORT_SUM_MEM(label_,"Transformed SC",matrix_->NumMyNonzeros(),
                                           matrix_->NumMyNonzeros(),comm_);

    HYMLS_TEST(Label(),
            noPcouplingsDropped(*matrix_,*hid_->Spawn(HierarchicalMap::LocalSeparators)),
            __FILE__,__LINE__);
    return 0;
    }


  // Returns true if the  preconditioner has been successfully initialized, false otherwise.
  bool SchurPreconditioner::IsInitialized() const {return initialized_;}
  
  // Returns true if the  preconditioner has been successfully computed, false otherwise.
  bool SchurPreconditioner::IsComputed() const {return computed_;}


  int SchurPreconditioner::Apply(const Epetra_MultiVector& X,
                           Epetra_MultiVector& Y) const
    {
    Tools::Warning("not implemented",__FILE__,__LINE__);
    return -99;
    }                           

  // Applies the preconditioner to vector X, returns the result in Y.
  int SchurPreconditioner::ApplyInverse(const Epetra_MultiVector& X,
                           Epetra_MultiVector& Y) const
    {
    if (isEmpty_) return 0;
    HYMLS_LPROF(label_,"ApplyInverse");

    if (HaveBorder())
      {
      // this is the expected behavior for standard ApplyInverse() of
      // a BorderedPreconditioner in HYMLS: solve with rhs 0 for border.
      int m = borderV_->NumVectors();
      int n = X.NumVectors();
      Epetra_SerialDenseMatrix C(m,n);
      Epetra_SerialDenseMatrix D(m,n);
      return ApplyInverse(X,C,Y,D);
      }
    
    numApplyInverse_++;
    time_->ResetStartTime();
    if (!IsComputed())
      {
      return -1;
      }

#ifdef HYMLS_TESTING
if (dumpVectors_)
  {
  MatrixUtils::Dump(*(X(0)),"SchurPreconditioner"+Teuchos::toString(myLevel_)+"_Rhs.txt");
  }
#endif

    if (myLevel_==maxLevel_)
      {
#ifdef HYMLS_TESTING
      if (Teuchos::is_null(reducedSchurScaLeft_) ||
          Teuchos::is_null(reducedSchurScaRight_)  )
        {
        Tools::Error("Scaling not available (should be created in Compute())",
                        __FILE__,__LINE__);
        }
#endif      

      bool realloc_vectors = (linearRhs_==Teuchos::null);
      if (!realloc_vectors) realloc_vectors = (linearRhs_->NumVectors()!=X.NumVectors());
      if (realloc_vectors)
        {
        linearRhs_=Teuchos::rcp(new Epetra_MultiVector(*linearMap_,X.NumVectors()));
        linearSol_=Teuchos::rcp(new Epetra_MultiVector(*linearMap_,X.NumVectors()));
        }
      for (int j=0;j<X.NumVectors();j++)
        {
        for (int i=0;i<X.MyLength();i++)
          {
          (*linearRhs_)[j][i]=X[j][i] * (*reducedSchurScaLeft_)[i];
          }
        }

      for (int i=0;i<fix_gid_.length();i++)
        {
        int lid = X.Map().LID(fix_gid_[i]);
        if (lid>0)
          {
          for (int k=0;k<X.NumVectors();k++)
            {
            (*linearRhs_)[k][lid]=0.0;
            }
          }
        }
      if (realloc_vectors)
        {
#ifdef RESTRICT_ON_COARSE_LEVEL
        // TODO - CHECK_ZERO at next Trilinos release
//        CHECK_ZERO(restrictB_->restrict_comm(linearRhs_));
//        CHECK_ZERO(restrictX_->restrict_comm(linearSol_));
        restrictB_->restrict_comm(linearRhs_);
        restrictX_->restrict_comm(linearSol_);
        restrictedRhs_ = restrictB_->RestrictedMultiVector();
        restrictedSol_ = restrictX_->RestrictedMultiVector();
#else
        restrictedRhs_=linearRhs_;
        restrictedSol_=linearSol_;
#endif        
        }
      if (amActive_)
        {
        CHECK_ZERO(reducedSchurSolver_->ApplyInverse(*restrictedRhs_,*restrictedSol_));
        }
      // unscale the solution
      for (int j=0;j<X.NumVectors();j++)
        {
        for (int i=0;i<X.MyLength();i++)
          {
          Y[j][i]=(*linearSol_)[j][i] * (*reducedSchurScaRight_)[i];
          }
        }
      }
    else
      {
      // (1) Transform right-hand side, B=OT'*X
      Epetra_MultiVector B(X);
    
      ApplyOT(true,B,&flopsApplyInverse_);

#ifdef HYMLS_TESTING
if (dumpVectors_)
  {
  MatrixUtils::Dump(*(B(0)),"SchurPreconditioner"+Teuchos::toString(myLevel_)+"_TransformedRhs.txt");
  }
#endif

      
      if (variant_=="Block Diagonal")
        {
        CHECK_ZERO(this->ApplyBlockDiagonal(B,Y));
        }
      else if (variant_=="Lower Triangular")
        {
        CHECK_ZERO(this->ApplyBlockLowerTriangular(B,Y));
        }
      else if (variant_=="Upper Triangular")
        {
        CHECK_ZERO(this->ApplyBlockUpperTriangular(B,Y));
        }

      CHECK_ZERO(this->UpdateVsumRhs(B,Y));
        
      // solve reduced Schur-complement problem
      if (X.NumVectors()!=vsumRhs_->NumVectors())
        {
        vsumRhs_ = Teuchos::rcp(new Epetra_MultiVector(*vsumMap_,X.NumVectors()));
        vsumSol_ = Teuchos::rcp(new Epetra_MultiVector(*vsumMap_,X.NumVectors()));
        }

      CHECK_ZERO(vsumRhs_->Import(Y,*vsumImporter_,Insert));
      if (reducedSchurScaLeft_!=Teuchos::null)
        {
        CHECK_ZERO(vsumRhs_->Multiply(1.0,*reducedSchurScaLeft_,*vsumRhs_,0.0));
        }
      CHECK_ZERO(reducedSchurSolver_->ApplyInverse(*vsumRhs_,*vsumSol_));
      if (reducedSchurScaRight_!=Teuchos::null)
        {
        CHECK_ZERO(vsumSol_->Multiply(1.0,*reducedSchurScaRight_,*vsumSol_,0.0));
        }
      CHECK_ZERO(Y.Export(*vsumSol_,*vsumImporter_,Insert));
      
      // transform back
      ApplyOT(false,Y,&flopsApplyInverse_);
      }

#ifdef HYMLS_TESTING
if (dumpVectors_)
  {
  MatrixUtils::Dump(*(Y(0)),"SchurPreconditioner"+Teuchos::toString(myLevel_)+"_Sol.txt");
//  dumpVectors_=false;
  }
#endif
      
    timeApplyInverse_+=time_->ElapsedTime();
    return 0;
    }

  // Returns a pointer to the matrix to be preconditioned.
  // will cause an error if the Schur-complement has not been
  // explicitly constructed!
  const Epetra_RowMatrix& SchurPreconditioner::Matrix() const 
        {
        if (SchurMatrix_==Teuchos::null) Tools::Error("SchurPreconditioner has no matrix",
                __FILE__,__LINE__);
        return *SchurMatrix_;
        }

  double SchurPreconditioner::InitializeFlops() const
    {
    // the total number of flops is computed each time InitializeFlops() is
    // called. This is becase I also have to add the contribution from each
    // container.
    double total = flopsInitialize_;
    for (int i=0;i<blockSolver_.size();i++)
      {
      if (blockSolver_[i]!=Teuchos::null)
        {
        total+=blockSolver_[i]->InitializeFlops();
        }
      }
    if (reducedSchurSolver_!=Teuchos::null)
      {
      total += reducedSchurSolver_->InitializeFlops();
      }
    return(total);
    }

  double SchurPreconditioner::ComputeFlops() const
    {
    double total = flopsCompute_;
    for (int i=0;i<blockSolver_.size();i++)
      {
      if (blockSolver_[i]!=Teuchos::null)
        {
        total+=blockSolver_[i]->ComputeFlops();
        }
      }
    if (reducedSchurSolver_!=Teuchos::null)
      {
      total += reducedSchurSolver_->ComputeFlops();
      }
    return(total);
    }

  double SchurPreconditioner::ApplyInverseFlops() const
    {
    double total = flopsApplyInverse_;
    for (int i=0;i<blockSolver_.size();i++)
      {
      if (blockSolver_[i]!=Teuchos::null)
        {
        total+=blockSolver_[i]->ApplyInverseFlops();
        }
      }
    if (reducedSchurSolver_!=Teuchos::null)
      {
      total += reducedSchurSolver_->ApplyInverseFlops();
      }
    return(total);
    }


  // Computes the condition number estimate, returns its value.
  double SchurPreconditioner::Condest(const Ifpack_CondestType CT,
                         const int MaxIters,
                         const double Tol,
                         Epetra_RowMatrix* Matrix)
                         {
                         Tools::Warning("not implemented!",__FILE__,__LINE__);
                         return -1.0; // not implemented.
                         }

  // Returns the computed condition number estimate, or -1.0 if not computed.
  double SchurPreconditioner::Condest() const
    {
    Tools::Warning("not implemented!",__FILE__,__LINE__);
    return -1.0;
    }

  // Returns the number of calls to Initialize().
  int SchurPreconditioner::NumInitialize() const {return numInitialize_;}

  // Returns the number of calls to Compute().
  int SchurPreconditioner::NumCompute() const {return numCompute_;}

  // Returns the number of calls to ApplyInverse().
  int SchurPreconditioner::NumApplyInverse() const {return numApplyInverse_;}

  // Returns the time spent in Initialize().
  double SchurPreconditioner::InitializeTime() const {return timeInitialize_;}

  // Returns the time spent in Compute().
  double SchurPreconditioner::ComputeTime() const {return timeCompute_;}

  // Returns the time spent in ApplyInverse().
  double SchurPreconditioner::ApplyInverseTime() const {return timeApplyInverse_;}
  
  // infinity-norm
  double SchurPreconditioner::NormInf() const 
    {
    if (IsComputed())
      {
      // note: this is not really correct anymore, the 
      // matrix_ has all nonzeros (even those that are 
      // ignored in ApplyInverse();
      //return matrix_->NormInf();
      return -1.0;
      }
    else 
      {
      return -1.0;      
      }
    }

  // Prints basic information on iostream. This function is used by operator<<.
  ostream& SchurPreconditioner::Print(std::ostream& os) const
    {
    os << label_ << std::endl;
    return os;
    }

  // apply orthogonal transforms to a vector v
int SchurPreconditioner::ApplyOT(bool trans, Epetra_MultiVector& v, double* flops) const
  {
  HYMLS_LPROF2(label_,"ApplyOT");

//  if ((!subdivideSeparators_))
  if (false)
    {
    if (maxLevel_>2)
      {
      Tools::Warning("this variant of ApplyOT is not valid for multi-level case",      
        __FILE__,__LINE__);
      }
    // implementation 1: apply OT to views of blocks of the vector. This is 
    // potentially very fast, but works only if the separators are contiguous
    // in the ordering of S (which they are unless you use "Subdivide Separators").
    
    // We currently always use the second implementation because it allows more general
    // transformations, which is required for the multi-level method (the vector v in the 
    // Householder transform is always assumed to contain only ones in case of the first
    // implementation). If there is a performance gain we can implement this in the
    // 'view' implementation, too.

    // this object has each local separator as a group (without overlap),
    // and remote separators connecting to local subdomains as separators
    // (we only access the local separators here)
    Teuchos::RCP<const HierarchicalMap> sepObject
          = hid_->Spawn(HierarchicalMap::LocalSeparators);
  
    if (trans)
      {
      for (int sep=0;sep<sepObject->NumMySubdomains();sep++)
        {
        int len = sepObject->NumInteriorElements(sep);
        
        int gid = sepObject->GID(sep,0,0);
        int lid = map_->LID(gid);
        for (int k=0;k<v.NumVectors();k++)
          {
          Epetra_SerialDenseVector block(View,&(v[k][lid]),len);
          OT->ApplyInverse(block);
          }
        }
      }
    else
      {
      for (int sep=0;sep<sepObject->NumMySubdomains();sep++)
        {
        int len = sepObject->NumInteriorElements(sep);
        int gid = sepObject->GID(sep,0,0);
        int lid = map_->LID(gid);
        for (int k=0;k<v.NumVectors();k++)
          {
          Epetra_SerialDenseVector block(View,&(v[k][lid]),len);
          OT->Apply(block);
          }
        }
      }
              
    //TODO: implement flops-counter
    if (flops!=NULL)
      {
      }
    }
  else
    {
    // implementation 2: just apply the constructed sparse matrix.
    // This makes sure that the OT for the matrix and the vectors are
    // consistent.
    if (sparseMatrixOT_==Teuchos::null)
      {
      HYMLS::Tools::Error("orth. transform not available as matrix!",
                  __FILE__, __LINE__);
      }
    Epetra_MultiVector tmp=v;
    if (trans)
      {
      CHECK_ZERO(OT->ApplyInverse(v,*sparseMatrixOT_,tmp));
      }
    else
      {
      CHECK_ZERO(OT->Apply(v,*sparseMatrixOT_,tmp));
      }
    if (flops!=NULL)
      {
      //TODO: make this general for all OTs
      *flops += sparseMatrixOT_->NumGlobalNonzeros() * 4 + v.MyLength();
      }
    }
  return 0;
  }

// attempt to scale P-couplings to 1. If row not coupled to any P-node,
// scale diagonal to 1 unless diagonal entry zero.
int SchurPreconditioner::ComputeScaling(const Epetra_CrsMatrix& A,
                                        Teuchos::RCP<Epetra_Vector>& sca_left,
                                        Teuchos::RCP<Epetra_Vector>& sca_right)
  {
  HYMLS_LPROF2(label_,"ComputeScaling");
  // TODO: not general!
  if (Teuchos::is_null(sca_left))
    {
    sca_left = Teuchos::rcp(new Epetra_Vector(A.RowMap()));
    }

  if (Teuchos::is_null(sca_right))
    {
    sca_right = Teuchos::rcp(new Epetra_Vector(A.RowMap()));
    }
    
  sca_left->PutScalar(1.0);
  sca_right->PutScalar(1.0);
  return 0; // this causes problems with the 1-level method for Stokes at 128x128,
            // the scaling should be looked at (TODO)
  Epetra_Vector diagA(A.RowMap());
  
  CHECK_ZERO(A.ExtractDiagonalCopy(diagA));
  CHECK_ZERO(diagA.Abs(diagA));
  double dmax;
  CHECK_ZERO(diagA.MaxValue(&dmax));
  // for saddle point matrices, this gets us 
  // similarly sized entries in the A and B part.
  for (int i=0;i<diagA.MyLength();i++)
    {
    if (diagA[i]<dmax*HYMLS_SMALL_ENTRY)
      {
      (*sca_left)[i] = dmax;
      (*sca_right)[i] = dmax;
      }
    }
// this is old stuff
#if 0
  if (hid_!=Teuchos::null)
    {
    const BasePartitioner& BP = hid_->Partitioner();
    int p_node = BP.DofPerNode()-1;
    double p_entry;
    double* val;
    int* ind;
    int len;

    for (int i=0;i<diagA.MyLength();i++)
      {
      CHECK_ZERO(A.ExtractMyRowView(i,len,val,ind));
      p_entry=0.0;
      for (int j=0;j<len;j++)
        {
        if (BP.VariableType(A.GCID(ind[j]))==p_node)
          {
          p_entry=std::abs(val[j]);
          }
        }
      if (p_entry>1.0e-8)
        {
        (*sca_left)[i] = 1.0/p_entry;
        (*sca_right)[i] = 1.0/p_entry;
        }
      }
    }
#endif    
  //MatrixUtils::Dump(*sca_left, "left_scale.txt");
  //MatrixUtils::Dump(*sca_right, "right_scale.txt");
  
  return 0;
  } 

int SchurPreconditioner::ApplyBlockDiagonal
        (const Epetra_MultiVector& B, Epetra_MultiVector& Y) const
  {
  HYMLS_LPROF2(label_,"Block Diagonal Solve");
  int numBlocks=blockSolver_.size(); // will be 0 on coarsest level
  for (int blk=0;blk<numBlocks;blk++)
    {
    if (Y.NumVectors()!=blockSolver_[blk]->NumVectors())
      {
      CHECK_ZERO(blockSolver_[blk]->SetNumVectors(Y.NumVectors()));
      }
    for (int j = 0 ; j < blockSolver_[blk]->NumRows() ; j++)
      {
      int lid = blockSolver_[blk]->ID(j);
      for (int k = 0 ; k < Y.NumVectors() ; k++)
        {
        blockSolver_[blk]->RHS(j,k) = B[k][lid];
        }
      }
    // apply the inverse of each block. NOTE: flops occurred
    // in ApplyInverse() of each block are summed up in method
    // ApplyInverseFlops().
    CHECK_ZERO(blockSolver_[blk]->ApplyInverse());

    // copy back into solution vector Y
    for (int j = 0 ; j < blockSolver_[blk]->NumRows() ; j++)  
      {
      int lid = blockSolver_[blk]->ID(j);
      for (int k = 0 ; k < Y.NumVectors() ; k++)
        {
        Y[k][lid] = blockSolver_[blk]->LHS(j,k);
        }
      }
    }   
  return 0;
  }

// approximate Y=S\B in non-Vsum points using Block lower triangular
// factor. The V-sum part of Y is not touched.
int SchurPreconditioner::ApplyBlockLowerTriangular
        (const Epetra_MultiVector& B, Epetra_MultiVector& Y) const
  {
  HYMLS_LPROF2(label_,"Block Lower Triangular Solve");
  int numBlocks=blockSolver_.size(); // will be 0 on coarsest level
    
  int ierr = this->BlockTriangularSolve(B,Y,0,numBlocks,+1);
  return ierr;
  }

// approximate Y=S\B in non-Vsum points using Block upper triangular
// factor. The V-sum part of Y is not touched.
int SchurPreconditioner::ApplyBlockUpperTriangular
        (const Epetra_MultiVector& B, Epetra_MultiVector& Y) const
  {
  HYMLS_LPROF2(label_,"Block Lower Triangular Solve");
  int numBlocks=blockSolver_.size(); // will be 0 on coarsest level
    
  int ierr = this->BlockTriangularSolve(B,Y,numBlocks-1,-1,-1);
  return ierr;
  }

// upper or lower triangular solve for non-Vsums
int SchurPreconditioner::BlockTriangularSolve
        (const Epetra_MultiVector& B, Epetra_MultiVector& Y,
        int start, int end, int incr) const
  {
  HYMLS_LPROF3(label_,"General Triangular Solve");
  int *indices;
  double* values;
  int len;

  // zero out Y so that we can use a matrix vector product for the
  // lower triangular blocks without checking column indices
  CHECK_ZERO(Y.PutScalar(0.0));
  
  // for each block row ...
  for (int blk=start;blk!=end;blk+=incr)
    {
    if (Y.NumVectors()!=blockSolver_[blk]->NumVectors())
      {
      blockSolver_[blk]->SetNumVectors(Y.NumVectors());
      }
    for (int i = 0 ; i < blockSolver_[blk]->NumRows() ; i++)
      {
      int lid = blockSolver_[blk]->ID(i);
      for (int k = 0 ; k < Y.NumVectors() ; k++)
        {
        blockSolver_[blk]->RHS(i,k) = B[k][lid];
        }
      // do a matrix vector product with the transformed S. Since
      // all Y entries are zeroed out from the start, this gives
      // RHS=B-L*Y, where L is the strict (block-)lower triangular 
      // part of the matrix.
      CHECK_ZERO(matrix_->ExtractMyRowView(lid,len,values,indices));
      for (int j=0;j<len;j++)
        {
        for (int k=0;k<Y.NumVectors();k++)
          {
          blockSolver_[blk]->RHS(i,k) -= values[j]*Y[k][indices[j]];
          }
        }
      }
    
    //TODO: flop count
    
    // apply the inverse of each block. NOTE: flops occurred
    // in ApplyInverse() of each block are summed up in method
    // ApplyInverseFlops().
    CHECK_ZERO(blockSolver_[blk]->ApplyInverse());

    // copy back into solution vector Y
    for (int i = 0 ; i < blockSolver_[blk]->NumRows() ; i++)  
      {
      int lid = blockSolver_[blk]->ID(i);
      for (int k = 0 ; k < Y.NumVectors() ; k++)         
        {
        Y[k][lid] = blockSolver_[blk]->LHS(i,k);
        }
      }
    }
  return 0;
  }

// update rhs for the next solve (currently does nothing)
int SchurPreconditioner::UpdateVsumRhs(const Epetra_MultiVector& B, Epetra_MultiVector& Y) const
  {
  HYMLS_LPROF3(label_,"Update Vsum RHS");

  // update the RHS for the V-sum solve
  for (int i=0;i<vsumMap_->NumMyElements();i++)
    {
    int lid = Y.Map().LID(vsumMap_->GID(i));
    for (int k=0;k<Y.NumVectors();k++)
      {
      Y[k][lid] = B[k][lid];
      }
    /*
    CHECK_ZERO(matrix_->ExtractMyRowView(lid,len,values,indices));
    for (int j=0;j<len;j++)
      {
      for (int k=0;k<Y.NumVectors();k++)
        {
        Y[k][lid] -= values[j]*Y[k][indices[j]];
        }
      }
    */
    }
  return 0;
  }

////////////////////////////////////////////////////
// implementation of the BorderedPreconditioner interface //
////////////////////////////////////////////////////

    // set the operators V, W and C to solve systems with
    // | M11 M12 V1 |
    // | M21 M22 V2 |
    // | W1  W2   C |. We already have M11 and M22 facored, but
    // we need to add a border to M22 and factor it again on the
    // coarsest level. On intermediate levelswe just need to compute
    // the border for M22 and pass it to the next level. M12 and M21
    // are currently assumed to be zero (block diagonal preconditioner)
    //                                          
    int SchurPreconditioner::setBorder(Teuchos::RCP<const Epetra_MultiVector> V,
                          Teuchos::RCP<const Epetra_MultiVector> W,
                          Teuchos::RCP<const Epetra_SerialDenseMatrix> C)
  {
  HYMLS_LPROF(label_,"setBorder");
  int ierr=0;
  if (V==Teuchos::null) 
    {
    //unset
    haveBorder_=false;
    return 0;
    }
  borderV_=Teuchos::rcp(new Epetra_MultiVector(*V));
  if (W!=Teuchos::null)
    {
    borderW_=Teuchos::rcp(new Epetra_MultiVector(*W));
    }
  else
    {
    borderW_=Teuchos::rcp(new Epetra_MultiVector(*V));    
    }
  if (C!=Teuchos::null)
    {
    borderC_=Teuchos::rcp(new Epetra_SerialDenseMatrix(*C));
    }
  else
    {
    int n=V->NumVectors();
    borderC_=Teuchos::rcp(new Epetra_SerialDenseMatrix(n,n));
    }

  if (!IsInitialized())
    {
    Tools::Error("SchurPreconditioner not yet initialized",__FILE__,__LINE__);
    }

  if (myLevel_==maxLevel_)
    {
    if (amActive_)
      {
      // we need to create views of the vectors here because the
      // map is different for the solver (linear restricted map)
      Teuchos::RCP<const Epetra_MultiVector> Vprime = 
          Teuchos::rcp(new Epetra_MultiVector(View,restrictedMatrix_->RowMap(),
                borderV_->Values(),borderV_->Stride(),borderV_->NumVectors()));
      Teuchos::RCP<const Epetra_MultiVector> Wprime =
          Teuchos::rcp(new Epetra_MultiVector(View,restrictedMatrix_->RowMap(),
                borderW_->Values(),borderW_->Stride(),borderW_->NumVectors()));

      // create AugmentedMatrix, refactor reducedSchurSolver_
      bool status=true;
      try {
      augmentedMatrix_ = Teuchos::rcp
          (new HYMLS::AugmentedMatrix(restrictedMatrix_,Vprime,Wprime,borderC_));
      } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr,status);
      if (!status) {Tools::Fatal("caught an exception when constructing final bordered system",__FILE__,__LINE__);}
      Teuchos::ParameterList& amesosList=PL().sublist("Coarse Solver");                    
#ifdef HYMLS_STORE_MATRICES
      status=true;
      try {
      ::EpetraExt::RowMatrixToMatrixMarketFile("FinalBorderedSchur.mtx",*augmentedMatrix_);
      } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr,status);
      if (status==false) Tools::Warning("caught exception when trying to dump final bordered SC",__FILE__,__LINE__);
#endif
      reducedSchurSolver_= Teuchos::rcp(new Ifpack_Amesos(augmentedMatrix_.get()));
      CHECK_ZERO(reducedSchurSolver_->SetParameters(amesosList));
      HYMLS_DEBUG("re-initialize direct solver for augmented system");
      CHECK_ZERO(reducedSchurSolver_->Initialize());
      HYMLS_DEBUG("re-compute direct solver for augmented system");
      CHECK_ZERO(reducedSchurSolver_->Compute());
      }
    }
  else
    {
    // transform V and W
    CHECK_ZERO(this->ApplyOT(false,*borderV_));
    CHECK_ZERO(this->ApplyOT(true,*borderW_));
    // form V_2 and W_2 by import operations (V_1 and W_1 are views of V_ and W_)
    borderV2_=Teuchos::rcp(new Epetra_MultiVector(*vsumMap_,borderV_->NumVectors()));
    borderW2_=Teuchos::rcp(new Epetra_MultiVector(*vsumMap_,borderW_->NumVectors()));
    CHECK_ZERO(borderV2_->Import(*borderV_,*vsumImporter_,Insert));
    CHECK_ZERO(borderW2_->Import(*borderW_,*vsumImporter_,Insert));
    // set border in next level problem
    Teuchos::RCP<HYMLS::BorderedPreconditioner> borderedNextLevel =
        Teuchos::rcp_dynamic_cast<HYMLS::BorderedPreconditioner>(reducedSchurSolver_);
    if (Teuchos::is_null(borderedNextLevel))
      {
      HYMLS::Tools::Error("next level solver can't handle border!",__FILE__,__LINE__);
      }
    HYMLS_DEBUG("call setBorder in next level precond");
    borderedNextLevel->setBorder(borderV2_,borderW2_,borderC_);
    }
  haveBorder_=true;
  return ierr;
  }

    // compute [Y T]' = [K V;W' C]*[X S]'
    int SchurPreconditioner::Apply(const Epetra_MultiVector& X, const Epetra_SerialDenseMatrix& S,
                    Epetra_MultiVector& Y,       Epetra_SerialDenseMatrix& T) const
  {
  return -99; // not implemented
  }                    

    // compute [X S]' = [K V;W' C]\[Y T]'
    int SchurPreconditioner::ApplyInverse(const Epetra_MultiVector& X, 
                const Epetra_SerialDenseMatrix& T,
                            Epetra_MultiVector& Y,
                      Epetra_SerialDenseMatrix& S) const
  {
  HYMLS_LPROF2(label_,"ApplyInverse (bordered)");

    // so the procedure to solve the system is  
    //                                          
    // |M22 V2| x2   = f2                       
    // |W2  C |  v   = g - W1'M11\f1            
    //                                          
    // x1 = M11\(f1 -V1 v)                      
    //                                          
    // where [f1' f2'] are the Householder-transformed RHS of the original problem
    // and g is the rhs contribution for the border.
    // This is like standard ApplyInverse(), only there we do the coarse solve
    // last.
    if (isEmpty_) return 0;

    numApplyInverse_++;
    time_->ResetStartTime();
    if (!IsComputed())
      {
      return -1;
      }
      
    if (!HaveBorder())
      {
      Tools::Warning("border not set!",__FILE__,__LINE__);
      return this->ApplyInverse(X,Y);
      }

#ifdef HYMLS_TESTING
if (dumpVectors_)
  {
  MatrixUtils::Dump(*(X(0)),"SchurPreconditioner"+Teuchos::toString(myLevel_)+"_Rhs.txt");
  }
#endif

    CHECK_ZERO(Y.PutScalar(0.0));

    if (myLevel_==maxLevel_)
      {
      if (amActive_)
        {
        // on the coarsest level we have put the border explicitly into an
        // AugmentedMatrix so we need to form the complete RHS
        bool realloc_vectors = (linearRhs_==Teuchos::null);
        if (!realloc_vectors) realloc_vectors = (linearRhs_->NumVectors()!=X.NumVectors());
        if (!realloc_vectors) realloc_vectors = 
          (linearRhs_->Map().SameAs(augmentedMatrix_->Map())==false);
        if (realloc_vectors)
          {
          HYMLS_DEBUG("(re-)allocate tmp vectors");
          linearRhs_=Teuchos::rcp(new 
          Epetra_MultiVector(augmentedMatrix_->Map(),Y.NumVectors()));
          linearSol_=Teuchos::rcp(new 
          Epetra_MultiVector(augmentedMatrix_->Map(),Y.NumVectors()));
          }
        for (int j=0;j<X.NumVectors();j++)
          {
          for (int i=0;i<X.MyLength();i++)
            {
            (*linearRhs_)[j][i]=X[j][i] * (*reducedSchurScaLeft_)[i];
            }
          for (int i=X.MyLength();i<linearRhs_->MyLength();i++)
            {
            int k = i-X.MyLength();
            (*linearRhs_)[j][i]=T[j][k];
            }
          }
/* in augmented systems we do not fix any GIDs, I guess...
      for (int i=0;i<fix_gid_.length();i++)
        {
        int lid = X.Map().LID(fix_gid_[i]);
        if (lid>0)
          {
          for (int k=0;k<X.NumVectors();k++)
            {
            (*linearRhs_)[k][lid]=0.0;
            }
          }
        }
*/
        if (realloc_vectors)
          {
#ifdef RESTRICT_ON_COARSE_LEVEL
#ifndef OLD_TRILINOS
          CHECK_ZERO(restrictB_->restrict_comm(linearRhs_));
          CHECK_ZERO(restrictX_->restrict_comm(linearSol_));
#else
          restrictB_->restrict_comm(linearRhs_);
          restrictX_->restrict_comm(linearSol_);
#endif
          restrictedRhs_ = restrictB_->RestrictedMultiVector();
          restrictedSol_ = restrictX_->RestrictedMultiVector();
#else
          restrictedRhs_=linearRhs_;
          restrictedSol_=linearSol_;
#endif        
          }
        HYMLS_DEBUG("coarse level solve");
        CHECK_ZERO(reducedSchurSolver_->ApplyInverse(*restrictedRhs_,*restrictedSol_));

        // unscale the solution and split into X and S
        for (int j=0;j<X.NumVectors();j++)
          {
          for (int i=0;i<X.MyLength();i++)
            {
            Y[j][i]=(*linearSol_)[j][i] * (*reducedSchurScaRight_)[i];
            }
          for (int i=X.MyLength();i<linearRhs_->MyLength();i++)
            {
            int k = i-X.MyLength();
            S[j][k]=(*linearSol_)[j][i];
            }
          }
#ifdef HYMLS_DEBUGGING
HYMLS::MatrixUtils::Dump(*linearRhs_,"CoarseLevelRhs.txt");
HYMLS::MatrixUtils::Dump(*linearSol_,"CoarseLevelSol.txt");
#endif      
        }
      }
    else
      {
      // (1) Transform right-hand side, B=OT'*X
      Epetra_MultiVector B(X);

      ApplyOT(true,B,&flopsApplyInverse_);

      // compute x1 = M11\f1
      if (variant_=="Block Diagonal")
        {
        CHECK_ZERO(this->ApplyBlockDiagonal(B,Y));
        }
      else
        {
        Tools::Error("not implemented",__FILE__,__LINE__);
        // did not look at this
        }


      // solve reduced Schur-complement problem.
      // We do not have to form the augmented vectors here as
      // we use the BorderedPreconditioner interface's ApplyInverse()
      // function recursively.
      if (X.NumVectors()!=vsumRhs_->NumVectors())
        {
        vsumRhs_ = Teuchos::rcp(new
          Epetra_MultiVector(*vsumMap_,X.NumVectors()));
        vsumSol_ = Teuchos::rcp(new
          Epetra_MultiVector(*vsumMap_,X.NumVectors()));
        }

      CHECK_ZERO(vsumRhs_->Import(B,*vsumImporter_,Insert));
      Epetra_SerialDenseMatrix Tcopy(T);
      // compute W1'(M11\F1). note zeros in X2
      for (int j=0;j<T.N();j++)
        {
        for (int i=0;i<T.M();i++)
          {
          double WdotX;
          CHECK_ZERO((*borderW_)(i)->Dot(*(Y(j)),&WdotX));
          // | T - W1'M11\f1
          Tcopy[j][i]-= WdotX;
          }
        }

      if (reducedSchurScaLeft_!=Teuchos::null)
        {
        CHECK_ZERO(vsumRhs_->Multiply(1.0,*reducedSchurScaLeft_,*vsumRhs_,0.0));
        }
      Teuchos::RCP<const HYMLS::BorderedPreconditioner> borderedNextLevel =
        Teuchos::rcp_dynamic_cast<const HYMLS::BorderedPreconditioner>(reducedSchurSolver_);
      if (Teuchos::is_null(borderedNextLevel))
        {
        Tools::Error("cannot handle next level bordered system!",__FILE__,__LINE__);
        }
      CHECK_ZERO(borderedNextLevel->ApplyInverse(*vsumRhs_,Tcopy,*vsumSol_,S));
      if (reducedSchurScaRight_!=Teuchos::null)
        {
        CHECK_ZERO(vsumSol_->Multiply(1.0,*reducedSchurScaRight_,*vsumSol_,0.0));
        }
      // copy into X
      for (int j=0;j<Y.NumVectors();j++)
        for (int i=0;i<vsumSol_->MyLength();i++)
          {
          int gid = vsumMap_->GID(i);
          int lid = Y.Map().LID(gid);
#ifdef HYMLS_TESTING
              // something's fishy, should just be a copy operation.
              if (lid<0) Tools::Error("inconsistent maps",__FILE__,__LINE__);
#endif          
          Y[j][lid] = (*vsumSol_)[j][i];
          }
      
      // transform back
      ApplyOT(false,Y,&flopsApplyInverse_);
      }

#ifdef HYMLS_TESTING
if (dumpVectors_)
  {
  MatrixUtils::Dump(*(Y(0)),"SchurPreconditioner"+Teuchos::toString(myLevel_)+"_Sol.txt");
  dumpVectors_=false;
  }
#endif
      
    timeApplyInverse_+=time_->ElapsedTime();

  return 0;
  }


////////////////////////////////////////////////////

void SchurPreconditioner::Visualize(std::string mfilename,bool recurse) const
  {
  HYMLS_LPROF3(label_,"Visualize");
  if (myLevel_<maxLevel_)
    {
    std::ofstream ofs(mfilename.c_str(),std::ios::app);
    for (int i=0;i<comm_->NumProc();i++)
      {
      if (comm_->MyPID()==i)
        {
        ofs << "% rank "<<comm_->MyPID() << ": "<<Label()<<std::endl;
        ofs << "p{"<<myLevel_<<"}{"<<1+i<<"}.vsums=[";
        for (int j=0;j<vsumMap_->NumMyElements();j++)
          {
          ofs << vsumMap_->GID(j) << " ";
          }
        ofs << "];"<<std::endl;
        }
      comm_->Barrier();
      }
    ofs.close();
    if (recurse)
      {
      Teuchos::RCP<const HYMLS::Preconditioner> hymls =
          Teuchos::rcp_dynamic_cast<const HYMLS::Preconditioner>(reducedSchurSolver_);
      if (!Teuchos::is_null(hymls)) hymls->Visualize(mfilename);
      }
    }
  }
}// namespace
