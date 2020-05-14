#define RESTRICT_ON_COARSE_LEVEL

#include "HYMLS_SchurPreconditioner.hpp"

#include "HYMLS_config.h"

#include "HYMLS_Macros.hpp"
#include "HYMLS_Tools.hpp"
#include "HYMLS_MatrixUtils.hpp"
#include "HYMLS_OverlappingPartitioner.hpp"
#include "HYMLS_SchurComplement.hpp"
#include "HYMLS_Preconditioner.hpp"
#include "HYMLS_Householder.hpp"
#include "HYMLS_RestrictedOT.hpp"
#include "HYMLS_SeparatorGroup.hpp"
#include "HYMLS_CoarseSolver.hpp"

#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_Import.h"
#include "Epetra_MultiVector.h"
#include "Epetra_FECrsMatrix.h"
#include "Epetra_SerialDenseVector.h"
#include "Epetra_BlockMap.h"
#include "Epetra_CrsMatrix.h"
#ifdef HYMLS_LONG_LONG
#include "Epetra_LongLongSerialDenseVector.h"
#else
#include "Epetra_IntSerialDenseVector.h"
#endif
#include "Epetra_MpiComm.h"
#include "Epetra_Operator.h"
#include "Epetra_Vector.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include "Ifpack_Container.h"
#include "Ifpack_DenseContainer.h"
#include "Ifpack_SparseContainer.h"
#include "Ifpack_Amesos.h"

#ifdef HYMLS_STORE_MATRICES
#include "EpetraExt_RowMatrixOut.h"
#endif

#include "HYMLS_Tester.hpp"
#include "HYMLS_Epetra_Time.h"
#include "HYMLS_HierarchicalMap.hpp"
#include "HYMLS_OrthogonalTransform.hpp"

#include <fstream>
#include <algorithm>
#include <iostream>

namespace HYMLS {


// private constructor
SchurPreconditioner::SchurPreconditioner(
  Teuchos::RCP<const Epetra_Operator> SC,
  Teuchos::RCP<const OverlappingPartitioner> hid,
  Teuchos::RCP<Teuchos::ParameterList> params,
  int level,
  Teuchos::RCP<Epetra_Vector> testVector)
  : PLA("Preconditioner"),
    comm_(Teuchos::rcp(SC->Comm().Clone())),
    SchurMatrix_(Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(SC)),
    SchurComplement_(Teuchos::rcp_dynamic_cast<const HYMLS::SchurComplement>(SC)),
    myLevel_(level),
    variant_("Block Diagonal"),
    denseSwitch_(99), applyDropping_(true),
    applyOT_(true),
    hid_(hid), map_(Teuchos::rcp(&(SC->OperatorDomainMap()),false)),
    testVector_(testVector),
    sparseMatrixOT_(Teuchos::null),
    matrix_(Teuchos::null),
    nextLevelHID_(Teuchos::null),
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

  if (myLevel_ != maxLevel_ && hid_ == Teuchos::null)
    {
    Tools::Error("no HID available!", __FILE__, __LINE__);
    }

  isEmpty_ = (map_->NumGlobalElements64()==0);

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
  applyDropping_ = PL().get("Apply Dropping", true);
  applyOT_ = PL().get("Apply Orthogonal Transformation", applyDropping_);
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

  if (reducedSchurSolver_!=Teuchos::null)
    {
    CHECK_ZERO(reducedSchurSolver_->SetParameters(List));
    }

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
    CHECK_ZERO(InitializeOT());
    if (variant_ == "Do Nothing" || !applyDropping_)
      {
      blockSolver_.resize(0);
      }
    else if (variant_ == "Block Diagonal" ||
      variant_ == "Lower Triangular" ||
      variant_ == "No Dropping")
      {
      CHECK_ZERO(InitializeBlocks());
      }
    else if (variant_=="Domain Decomposition")
      {
      CHECK_ZERO(InitializeSingleBlock());
      }
    else
      {
      Tools::Error("Variant '"+variant_+"'not implemented",
        __FILE__,__LINE__);
      }
    }
  else
    applyOT_ = false;

  numInitialize_++;
  initialized_ = true;
  timeInitialize_ += time_->ElapsedTime();

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
    if (SchurComplement_ != Teuchos::null)
      {
      CHECK_ZERO(Assemble());
      SchurMatrix_ = matrix_;
      }
    }
  else
    {
    if (variant_ == "Do Nothing" || !applyDropping_)
      {
      blockSolver_.resize(0);
      }
    else if (variant_=="Block Diagonal"||
      variant_=="Lower Triangular")
      {
      CHECK_ZERO(InitializeBlocks());
      }
    else if (variant_=="Domain Decomposition")
      {
      CHECK_ZERO(InitializeSingleBlock());
      }
    else
      {
      Tools::Error("Variant '"+variant_+"'not implemented",
        __FILE__,__LINE__);
      }

    if (!applyDropping_ && SchurComplement_ != Teuchos::null)
      {
      CHECK_ZERO(Assemble());
      }
    else if (SchurMatrix_!=Teuchos::null)
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

    CHECK_ZERO(InitializeNextLevel());
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

  if (myLevel_ == maxLevel_)
    {
    reducedSchurSolver_ = Teuchos::rcp(new CoarseSolver(SchurMatrix_, fix_gid_, myLevel_));
    CHECK_ZERO(reducedSchurSolver_->SetParameters(PL()));
    HYMLS_DEBUG("Initialize direct solver");
    CHECK_ZERO(reducedSchurSolver_->Initialize());
    HYMLS_DEBUG("Compute direct solver");
    CHECK_ZERO(reducedSchurSolver_->Compute());
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
      newMap = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1),myLength,0,*comm_));
      }

    int off = 0;
    for (int sd = 0; sd < hid_->NumMySubdomains(); sd++)
      {
      begI << off << std::endl;
      off = off + hid_->NumInteriorElements(sd);
      }
    begI << off << std::endl;

    int offset=0;

    for (int sd = 0; sd < sepObject->NumMySubdomains(); sd++)
      {
      for (SeparatorGroup const &group: sepObject->GetSeparatorGroups(sd))
        {
        begS << offset << std::endl;
        offset = offset + group.length();
        //begS << sepObject->LID(sep,grp,0)<<std::endl;

        // V-sum nodes
        ofs << newMap->GID64(map_->LID(group[0])) << std::endl;
        ofs2 << newMap->GID64(map_->LID(group[0])) << std::endl;
        // non-Vsum nodes
        for (int j = 1; j < group.length(); j++)
          {
          ofs << newMap->GID64(map_->LID(group[j])) << std::endl;
          ofs1 << newMap->GID64(map_->LID(group[j])) << std::endl;
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
    reducedSchur_ = MatrixUtils::DropByValue(reducedSchur_,
      HYMLS_SMALL_ENTRY, MatrixUtils::RelDropDiag);

#ifdef HYMLS_STORE_MATRICES
    MatrixUtils::Dump(*reducedSchur_,"ReducedSchur"+Teuchos::toString(myLevel_)+".txt");
#endif

    }

  // compute solver for reduced Schur
  HYMLS_DEBUG("compute coarse solver");
  int ierr=reducedSchurSolver_->Compute();

  if (ierr!=0)
    {
#ifdef HYMLS_STORE_MATRICES
    MatrixUtils::Dump(*reducedSchur_,"BadMatrix"+Teuchos::toString(myLevel_)+".txt");
#endif
    Tools::Error("factorization returned value "+Teuchos::toString(ierr)+
      " on level "+Teuchos::toString(myLevel_),__FILE__,__LINE__);
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

  // create an array of solvers for all the diagonal blocks
  blockSolver_.resize(0);
  for (int sd = 0; sd < sepObject->NumMySubdomains(); sd++)
    {
    for (auto const &linked_groups: sepObject->GetLinkedSeparatorGroups(sd))
      {
      int numRows = 0;
      for (SeparatorGroup const &group: linked_groups)
        {
        if (group.length() == 0)
          HYMLS::Tools::Error("there is an empty separator, which is probably dangerous", __FILE__, __LINE__);

        // in the spawned sepObject, each local separator is a group of a subdomain.
        // -1 because we remove one Vsum node from each block
        numRows += std::max(group.length() - 1, 0);
        }

      blockSolver_.append(Teuchos::rcp(new Ifpack_DenseContainer(numRows)));
      CHECK_ZERO(blockSolver_.back()->SetParameters(PL().sublist("Dense Solver")));
      CHECK_ZERO(blockSolver_.back()->Initialize());

      int k = 0;
      for (SeparatorGroup const &group: linked_groups)
        for (int j = 1; j < group.length(); j++)
        {
        // skip first element, which is a Vsum
        int LRID = map_->LID(group[j]);
        blockSolver_.back()->ID(k++) = LRID;
        }
      }
    }
  return 0;
  }

int SchurPreconditioner::InitializeSingleBlock()
  {
  HYMLS_LPROF2(label_,"InitializeSingleBlock");
  // get an object with only local separators and remote connected separators:
  Teuchos::RCP<const HierarchicalMap> sepObject
    = hid_->Spawn(HierarchicalMap::LocalSeparators);

  // count the number of owned elements and vsums
  int numMyVsums = 0;
  int numMyElements = 0;
  for (int sd = 0; sd < sepObject->NumMySubdomains(); sd++)
    {
    numMyElements += sepObject->NumSeparatorElements(sd);
    numMyVsums += sepObject->NumSeparatorGroups(sd);
    }
  // we actually need the number of owned non-Vsums:
  int numRows = numMyElements - numMyVsums;

  // create a single solver for all the non-Vsums
  blockSolver_.resize(1);
  blockSolver_[0] = Teuchos::rcp
    (new Ifpack_SparseContainer<Ifpack_Amesos>(numRows));
  CHECK_ZERO(blockSolver_[0]->SetParameters(
      PL().sublist("Sparse Solver")));
  CHECK_ZERO(blockSolver_[0]->Initialize());

  int pos = 0;
  for (int sd = 0; sd < sepObject->NumMySubdomains(); sd++)
    {
    for (SeparatorGroup const &group: sepObject->GetSeparatorGroups(sd))
      {
      // skip first element, which is a Vsum
      for (int j = 1; j < group.length(); j++)
        {
        int LRID = map_->LID(group[j]);
        blockSolver_[0]->ID(pos++) = LRID;
        }
      }
    }
  return 0;
  }

int SchurPreconditioner::InitializeOT()
  {
  HYMLS_LPROF2(label_,"InitializeOT");

  if (!applyOT_)
    return 0;

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

#ifdef HYMLS_LONG_LONG
    Epetra_LongLongSerialDenseVector inds;
#else
    Epetra_IntSerialDenseVector inds;
#endif
    Epetra_SerialDenseVector vec;

    int nnzPerRow = sepObject->NumMySubdomains()>0? sepObject->NumSeparatorElements(0) : 0;

    sparseMatrixOT_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy,*map_,nnzPerRow));

    // loop over all separators connected to a local subdomain
    for (int sd = 0; sd < sepObject->NumMySubdomains(); sd++)
      {
      HYMLS_DEBVAR(sd);
      // The LocalSeparator object has only local separators, but it may
      // have several groups due to splitting of groups (i.e. for the B-grid,
      // where velocities are grouped depending on how they connect to the pressures)
      for (SeparatorGroup const &group: sepObject->GetSeparatorGroups(sd))
        {
        int len = group.length();
        if (inds.Length() != len && len > 0)
          {
          inds.Size(len);
          vec.Size(len);
          }

        int pos = 0;
        for (hymls_gidx gid: group.nodes())
          {
          int lid = sepMap.LID(gid);
          if (lid != -1)
            {
            inds[pos] = gid;
            vec[pos++] = localTestVector[lid];
            }
          }
        inds.Resize(pos);
        vec.Resize(pos);
        if (pos > 0)
          {
//          HYMLS_DEBVAR(inds);
//          HYMLS_DEBVAR(vec);
          int ierr = OT->Construct(*sparseMatrixOT_, inds, vec);
          if (ierr)
            {
            Tools::Warning("Error code "+Teuchos::toString(ierr)+" returned from Epetra call!",
              __FILE__, __LINE__);
            return ierr;
            }
          }
        }
      }
    CHECK_ZERO(sparseMatrixOT_->FillComplete());
    }
#ifdef HYMLS_STORE_MATRICES
  MatrixUtils::Dump(*sparseMatrixOT_,
    "Householder"+Teuchos::toString(myLevel_)+".txt");
#endif
  return 0;
  }

Teuchos::RCP<const Epetra_Map> SchurPreconditioner::CreateVSumMap(
  Teuchos::RCP<const HierarchicalMap> &sepObject) const
  {
  int numBlocks = 0;
  for (int sd = 0; sd < sepObject->NumMySubdomains(); sd++)
    {
    for (SeparatorGroup const &group: sepObject->GetSeparatorGroups(sd))
      {
      if (applyDropping_)
        {
        if (group.length() > 0)
          numBlocks++;
        }
      else
        {
        numBlocks += group.length();
        }
      }
    }

  HYMLS_DEBVAR(numBlocks);

  // create a map for the reduced Schur-complement. Note that this is a distributed
  // matrix, in contrast to the other diagonal blocks, so we can't use an Ifpack
  // container.
  hymls_gidx *MyVsumElements = new hymls_gidx[numBlocks]; // one Vsum per block
  int pos = 0;
  for (int sd = 0; sd < sepObject->NumMySubdomains(); sd++)
    {
    for (SeparatorGroup const &group: sepObject->GetSeparatorGroups(sd))
      {
      if (group.length() > 0)
        {
        if (applyDropping_)
          MyVsumElements[pos++] = group[0];
        else
          for (hymls_gidx gid: group.nodes())
            MyVsumElements[pos++] = gid;
        }
      }
    }

  Teuchos::RCP<const Epetra_Map> ret = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1),
      numBlocks, MyVsumElements,
      (hymls_gidx)map_->IndexBase64(), map_->Comm()));

  delete[] MyVsumElements;

  return ret;
  }

int SchurPreconditioner::InitializeNextLevel()
  {
  HYMLS_LPROF2(label_,"InitializeNextLevel");

  if (vsumMap_==Teuchos::null)
    {
    Teuchos::RCP<const HierarchicalMap> localSepObject =
      hid_->Spawn(HierarchicalMap::LocalSeparators);
    vsumMap_ = CreateVSumMap(localSepObject);

    Teuchos::RCP<const HierarchicalMap> sepObject =
      hid_->Spawn(HierarchicalMap::Separators);
    overlappingVsumMap_ = CreateVSumMap(sepObject);

    HYMLS_DEBUG(label_);
    HYMLS_DEBVAR(*vsumMap_);

    vsumRhs_ = Teuchos::rcp(new Epetra_MultiVector(*vsumMap_,1));
    vsumSol_ = Teuchos::rcp(new Epetra_MultiVector(*vsumMap_,1));

    // the vsums are still distributed and we must
    // form a correct col map
    vsumColMap_ = MatrixUtils::CreateColMap(*matrix_,*vsumMap_,*vsumMap_);

    reducedSchur_ = Teuchos::rcp(new
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

  Teuchos::RCP<Epetra_Vector> nextTestVector = Teuchos::null;

  if (myLevel_+1!=maxLevel_)
    {
    if (nextLevelHID_==Teuchos::null)
      {
      bool stat=true;
      try {
        nextLevelHID_ = hid_->SpawnNextLevel(vsumMap_, overlappingVsumMap_);
        } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr,stat);
      if (!stat) Tools::Fatal("Failed to create next level ordering",__FILE__,__LINE__);
      }
    if (reducedSchurSolver_==Teuchos::null)
      {
      nextTestVector = Teuchos::rcp(new Epetra_Vector(*vsumMap_));

      Epetra_Vector transformedTestVector(*testVector_);
      CHECK_ZERO(ApplyOT(false, transformedTestVector, &flopsCompute_));
      CHECK_ZERO(nextTestVector->Import(transformedTestVector, *vsumImporter_, Insert));

      // create another level of HYMLS::Preconditioner
      Teuchos::RCP<Teuchos::ParameterList> nextLevelParams =
        Teuchos::rcp(new Teuchos::ParameterList(*getMyParamList()));
      if (myLevel_>=denseSwitch_-1)
        {
        nextLevelParams->sublist("Preconditioner").set("Subdomain Solver Type","Dense");
        }

      //TODO: move the direct solver thing to the Preconditioner class and rename
      //      the SchurPreconditioner SchurApproximation. Then this call can be put
      //      outside the if statement because we will always create a Preconditioner
      //      object for the reduced problem.
      reducedSchurSolver_= Teuchos::rcp(new
        Preconditioner(reducedSchur_, nextLevelParams,
          nextTestVector, myLevel_+1, nextLevelHID_));
      }
    else
      {
      Teuchos::RCP<Preconditioner> prec =
        Teuchos::rcp_dynamic_cast<Preconditioner>(reducedSchurSolver_);

      if (prec == Teuchos::null)
        Tools::Error("dynamic cast failed", __FILE__, __LINE__);

      prec->SetMatrix(reducedSchur_);
      }
    }
  else
    {
    Teuchos::RCP<Teuchos::ParameterList> nextLevelParams =
      Teuchos::rcp(new Teuchos::ParameterList(*getMyParamList()));

    // fix pressure on coarsest level:
    for (int i = 0; i < fix_gid_.length(); i++)
      {
      CHECK_ZERO(MatrixUtils::PutDirichlet(*reducedSchur_, fix_gid_[i]));
      }

    reducedSchurSolver_= Teuchos::rcp(new
      SchurPreconditioner(reducedSchur_,nextLevelHID_,nextLevelParams,
        myLevel_+1, nextTestVector));
    }

  HYMLS_DEBUG("Initialize solver for reduced Schur");
  CHECK_ZERO(reducedSchurSolver_->Initialize());
  return 0;
  }

int SchurPreconditioner::Assemble()
  {
  HYMLS_LPROF2(label_,"Assemble");

  Teuchos::RCP<Epetra_FECrsMatrix> matrix =
    Teuchos::rcp_dynamic_cast<Epetra_FECrsMatrix>(matrix_);
  if (matrix == Teuchos::null)
    {
    int nzest = 32;
    if (hid_ != Teuchos::null && hid_->NumMySubdomains() > 0)
      nzest = hid_->NumSeparatorElements(0);
    matrix = Teuchos::rcp(new
      Epetra_FECrsMatrix(Copy, SchurComplement_->OperatorDomainMap(), nzest));
    }
  CHECK_ZERO(SchurComplement_->Construct(matrix));

  if (applyOT_)
    matrix_ = OT->Apply(*sparseMatrixOT_, *matrix);
  else
    matrix_ = matrix;

  matrix_ = MatrixUtils::DropByValue(matrix_, HYMLS_SMALL_ENTRY);

#ifdef HYMLS_STORE_MATRICES
  MatrixUtils::Dump(*matrix_,"SchurPreconditioner"+Teuchos::toString(myLevel_)+".txt");
#endif

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
    if (hid_->NumMySubdomains() > 0)
      nzest = hid_->NumSeparatorElements(0);
    matrix = Teuchos::rcp(new Epetra_FECrsMatrix(Copy,*map_,nzest));
    matrix_=matrix;
    }

  Epetra_SerialDenseVector v;
  Epetra_SerialDenseMatrix Sk;

  // part remaining after dropping
  Epetra_SerialDenseMatrix Spart;
#ifdef HYMLS_LONG_LONG
  Epetra_LongLongSerialDenseVector indices;
  Epetra_LongLongSerialDenseVector indsPart;
#else
  Epetra_IntSerialDenseVector indices;
  Epetra_IntSerialDenseVector indsPart;
#endif

  // put the pattern into matrix_
  if (!matrix->Filled())
    {
    HYMLS_LPROF3(label_,"Fill matrix");
    // start out by just putting the structure together.
    // I do this because the SumInto function will fail
    // unless the values have been put in already. On the
    // other hand, the Insert function will overwrite stuff
    // we put in previously.
    //
    // NOTE: this is where the 'dropping' occurs, so if we
    //       want to implement different schemes we have
    //       to adjust this loop in the first place.
    for (int sd = 0; sd < hid_->NumMySubdomains(); sd++)
      {
      // put in the Vsum-Vsum couplings
      int numVsums = hid_->NumSeparatorGroups(sd);
      indsPart.Size(numVsums);
      if (numVsums > Spart.N())
        Spart.Shape(2 * numVsums, 2 * numVsums);

      numVsums = 0;
      for (SeparatorGroup const &group: hid_->GetSeparatorGroups(sd))
        {
        if (group.length() > 0)
          indsPart[numVsums++] = group[0];
        }
      CHECK_NONNEG(matrix->InsertGlobalValues(numVsums, indsPart.Values(), Spart.A()));

      // now the non-Vsums
      for (auto const &linked_groups: hid_->GetLinkedSeparatorGroups(sd))
        {
        int len = 0;
        for (SeparatorGroup const &group: linked_groups)
          len += group.length() - 1;

        indsPart.Size(len);
        if (Spart.N() < len)
          Spart.Shape(2 * len, 2 * len);

        int i = 0;
        for (SeparatorGroup const &group: linked_groups)
          for (int j = 1; j < group.length(); j++)
            indsPart[i++] = group[j];

        CHECK_NONNEG(matrix->InsertGlobalValues(len, indsPart.Values(), Spart.A()));
        }
      }
    // assemble with all zeros
    HYMLS_DEBUG("assemble pattern of transformed SC");
    CHECK_ZERO(matrix->GlobalAssemble());
    }

  CHECK_ZERO(matrix->PutScalar(0.0));

#ifdef HYMLS_DEBUGGING
  // std::string s1 = "SchurPrecond"+Teuchos::toString(myLevel_)+"_";
  // MatrixUtils::Dump(*matrix_,s1+"Pattern.txt");
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

  // now for each subdomain construct the SC part A22 and A21*A11\A12 for the
  // surrounding separators, apply orthogonal transforms to each separator
  // group and sum them into the pattern defined above, dropping everything
  // that is not defined in the matrix pattern.

  HYMLS_DEBUG("Add A22 part");
  for (int sd = 0; sd < hid_->NumMySubdomains(); sd++)
    {
    HYMLS_LPROF3(label_, "Add A22 part");
    // construct the local contribution of the SC
    // (for all separators around the subdomain)
    HYMLS_DEBVAR(sd);

    // Construct the local A22
    CHECK_ZERO(SchurComplement_->Construct22(sd, Sk, indices));

    Teuchos::Array<Teuchos::RCP<Epetra_SerialDenseMatrix> > SkArray;
#ifdef HYMLS_LONG_LONG
    Teuchos::Array<Teuchos::RCP<Epetra_LongLongSerialDenseVector> > indicesArray;
#else
    Teuchos::Array<Teuchos::RCP<Epetra_IntSerialDenseVector> > indicesArray;
#endif
    CHECK_ZERO(ConstructSCPart(sd, localTestVector, Sk, indices, SkArray, indicesArray));

    for (int i = 0; i < SkArray.length(); i++)
      {
      HYMLS_DEBVAR(i);
      CHECK_ZERO(matrix->ReplaceGlobalValues(*indicesArray[i], *SkArray[i]));
      }
    }//sd
  CHECK_ZERO(matrix->GlobalAssemble(false, Insert));

  HYMLS_DEBUG("-A21*A11\\A12 part");
  for (int sd = 0; sd < hid_->NumMySubdomains(); sd++)
    {
    HYMLS_LPROF3(label_, "Add -A21*A11\\A12 part");
    // construct the local contribution of the SC
    // (for all separators around the subdomain)
    HYMLS_DEBVAR(sd);

    // Construct the local -A21*A11\A12
    CHECK_ZERO(SchurComplement_->Construct11(sd, Sk, indices));

    Teuchos::Array<Teuchos::RCP<Epetra_SerialDenseMatrix> > SkArray;
#ifdef HYMLS_LONG_LONG
    Teuchos::Array<Teuchos::RCP<Epetra_LongLongSerialDenseVector> > indicesArray;
#else
    Teuchos::Array<Teuchos::RCP<Epetra_IntSerialDenseVector> > indicesArray;
#endif
    CHECK_ZERO(ConstructSCPart(sd, localTestVector, Sk, indices, SkArray, indicesArray));

    for (int i = 0; i < SkArray.length(); i++)
      {
      HYMLS_DEBVAR(i);
      CHECK_ZERO(matrix->SumIntoGlobalValues(*indicesArray[i], *SkArray[i]));
      }
    }//sd
  CHECK_ZERO(matrix->GlobalAssemble());

#ifdef HYMLS_STORE_MATRICES
  MatrixUtils::Dump(*matrix_,"SchurPreconditioner"+Teuchos::toString(myLevel_)+".txt");
#endif

  HYMLS_TEST(Label(),
    noPcouplingsDropped(*matrix_,*hid_->Spawn(HierarchicalMap::LocalSeparators)),
    __FILE__,__LINE__);
  return 0;
  }

int SchurPreconditioner::ConstructSCPart(int sd, Epetra_Vector const &localTestVector,
  Epetra_SerialDenseMatrix & Sk,
#ifdef HYMLS_LONG_LONG
  Epetra_LongLongSerialDenseVector &indices,
#else
  Epetra_IntSerialDenseVector &indices,
#endif
  Teuchos::Array<Teuchos::RCP<Epetra_SerialDenseMatrix> > &SkArray,
#ifdef HYMLS_LONG_LONG
  Teuchos::Array<Teuchos::RCP<Epetra_LongLongSerialDenseVector> > &indicesArray
#else
  Teuchos::Array<Teuchos::RCP<Epetra_IntSerialDenseVector> > &indicesArray
#endif
  ) const
  {
  Epetra_SerialDenseVector v;

  // Get the part of the testvector that belongs to the
  // separators
  const Epetra_BlockMap& sepMap = localTestVector.Map();
  v.Resize(indices.Length());
  for (int i = 0; i < indices.Length(); i++)
    v[i] = localTestVector[sepMap.LID(indices[i])];

  const int numVSums = hid_->NumSeparatorGroups(sd);

  SkArray.append(Teuchos::rcp(new Epetra_SerialDenseMatrix(numVSums, numVSums)));
  Epetra_SerialDenseMatrix &VSumSk = *SkArray.back();
#ifdef HYMLS_LONG_LONG
  indicesArray.append(Teuchos::rcp(new Epetra_LongLongSerialDenseVector(numVSums)));
  Epetra_LongLongSerialDenseVector &VSumIndices = *indicesArray.back();
#else
  indicesArray.append(Teuchos::rcp(new Epetra_IntSerialDenseVector(numVSums)));
  Epetra_IntSerialDenseVector &VSumIndices = *indicesArray.back();
#endif

  int i = 0, j = 0, pos = 0;
  // Loop over all separators of the subdomain sd
  for (SeparatorGroup const &group: hid_->GetSeparatorGroups(sd))
    {
    HYMLS_LPROF3(label_,"Apply OT");
    const int len = group.length();
    Epetra_SerialDenseVector vView(View, &v[pos], len);

    // Apply the orthogonal transformation for each group
    // separately
    RestrictedOT::Apply(Sk, pos, *OT, vView);

    VSumIndices[i++] = indices[pos];

    pos += len;
    }

  Teuchos::RCP<const Epetra_Map> map = hid_->SpawnMap(sd, HierarchicalMap::Separators);

  // Only add Vsum-Vsum couplings and non-Vsums. This is way faster than
  // than trying to add all the values and letting SumIntoGlobalValues
  // decide which ones to drop.
  i = 0;
  for (SeparatorGroup const &group1: hid_->GetSeparatorGroups(sd))
    {
    HYMLS_LPROF3(label_, "Compute non-dropped Vsum part");

    j = 0;
    const int lid1 = map->LID(group1[0]);
    for (SeparatorGroup const &group2: hid_->GetSeparatorGroups(sd))
      {
      const int lid2 = map->LID(group2[0]);
      VSumSk(i, j++) = Sk(lid1, lid2);
      }
    i++;
    }

  for (auto const &linked_groups: hid_->GetLinkedSeparatorGroups(sd))
    {
    HYMLS_LPROF3(label_, "Compute non-Vsum part");

    int len = 0;
    for (SeparatorGroup const &group: linked_groups)
      len += group.length() - 1;

    SkArray.append(Teuchos::rcp(new Epetra_SerialDenseMatrix(len, len)));
    Epetra_SerialDenseMatrix &localSk = *SkArray.back();
#ifdef HYMLS_LONG_LONG
    indicesArray.append(Teuchos::rcp(new Epetra_LongLongSerialDenseVector(len)));
    Epetra_LongLongSerialDenseVector &globalIndices = *indicesArray.back();
#else
    indicesArray.append(Teuchos::rcp(new Epetra_IntSerialDenseVector(len)));
    Epetra_IntSerialDenseVector &globalIndices = *indicesArray.back();
#endif
    Teuchos::Array<int> localIndices(len);

    int i = 0;
    for (SeparatorGroup const &group: linked_groups)
      for (int j = 1; j < group.length(); j++)
        {
        hymls_gidx gid = group[j];
        HYMLS_DEBUG(i << " " << j << " " << gid << " " << map->LID(gid));
        globalIndices[i] = gid;
        localIndices[i] = map->LID(gid);
        i++;
        }

    for (int i = 0; i < len; i++)
      for (int j = 0; j < len; j++)
        localSk(i, j) = Sk(localIndices[i], localIndices[j]);
    }

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
    // a BorderedOperator in HYMLS: solve with rhs 0 for border.
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
    CHECK_ZERO(reducedSchurSolver_->ApplyInverse(X, Y));
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
    CHECK_ZERO(reducedSchurSolver_->ApplyInverse(*vsumRhs_,*vsumSol_));
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
std::ostream& SchurPreconditioner::Print(std::ostream& os) const
  {
  os << label_ << std::endl;
  return os;
  }

// apply orthogonal transforms to a vector v
int SchurPreconditioner::ApplyOT(bool trans, Epetra_MultiVector& v, double* flops) const
  {
  HYMLS_LPROF2(label_,"ApplyOT");

  if (!applyOT_)
    return 0;

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
    *flops += sparseMatrixOT_->NumGlobalNonzeros64() * 4 + v.MyLength();
    }
  // }
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
    int lid = Y.Map().LID(vsumMap_->GID64(i));
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
// implementation of the BorderedOperator interface //
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
    haveBorder_ = true;
    Teuchos::RCP<HYMLS::BorderedOperator> borderedSolver =
      Teuchos::rcp_dynamic_cast<HYMLS::BorderedOperator>(reducedSchurSolver_);
    if (Teuchos::is_null(borderedSolver))
      {
      HYMLS::Tools::Error("next level solver can't handle border!",__FILE__,__LINE__);
      }
    HYMLS_DEBUG("call setBorder in next level precond");
    borderedSolver->setBorder(borderV_,borderW_,borderC_);
    }
  else
    {
    // transform V and W
    CHECK_ZERO(this->ApplyOT(false,*borderV_));
    CHECK_ZERO(this->ApplyOT(true,*borderW_));
    // form V_2 and W_2 by import operations (V_1 and W_1 are views of V_ and W_)
    Teuchos::RCP<Epetra_MultiVector> borderV2 = Teuchos::rcp(
      new Epetra_MultiVector(*vsumMap_, borderV_->NumVectors()));
    Teuchos::RCP<Epetra_MultiVector> borderW2 = Teuchos::rcp(
      new Epetra_MultiVector(*vsumMap_, borderW_->NumVectors()));
    CHECK_ZERO(borderV2->Import(*borderV_,*vsumImporter_,Insert));
    CHECK_ZERO(borderW2->Import(*borderW_,*vsumImporter_,Insert));
    // set border in next level problem
    Teuchos::RCP<HYMLS::BorderedOperator> borderedNextLevel =
      Teuchos::rcp_dynamic_cast<HYMLS::BorderedOperator>(reducedSchurSolver_);
    if (Teuchos::is_null(borderedNextLevel))
      {
      HYMLS::Tools::Error("next level solver can't handle border!",__FILE__,__LINE__);
      }
    HYMLS_DEBUG("call setBorder in next level precond");
    borderedNextLevel->setBorder(borderV2, borderW2, borderC_);
    }
  haveBorder_ = true;
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
    HYMLS_DEBUG("border not set!");
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
    HYMLS_DEBUG("coarse level solve");
    Teuchos::RCP<const HYMLS::BorderedOperator> borderedSolver =
      Teuchos::rcp_dynamic_cast<const HYMLS::BorderedOperator>(reducedSchurSolver_);
    if (Teuchos::is_null(borderedSolver))
      {
      Tools::Error("cannot handle next level bordered system!", __FILE__, __LINE__);
      }
    CHECK_ZERO(borderedSolver->ApplyInverse(X, T, Y, S));
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
    // we use the BorderedOperator interface's ApplyInverse()
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

    Teuchos::RCP<const HYMLS::BorderedOperator> borderedNextLevel =
      Teuchos::rcp_dynamic_cast<const HYMLS::BorderedOperator>(reducedSchurSolver_);
    if (Teuchos::is_null(borderedNextLevel))
      {
      Tools::Error("cannot handle next level bordered system!",__FILE__,__LINE__);
      }
    CHECK_ZERO(borderedNextLevel->ApplyInverse(*vsumRhs_,Tcopy,*vsumSol_,S));
    // copy into X
    for (int j=0;j<Y.NumVectors();j++)
      for (int i=0;i<vsumSol_->MyLength();i++)
        {
        int lid = Y.Map().LID(vsumMap_->GID64(i));
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
          ofs << vsumMap_->GID64(j) << " ";
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
