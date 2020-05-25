#include "HYMLS_CoarseSolver.hpp"

#define RESTRICT_ON_COARSE_LEVEL

#include "HYMLS_config.h"

#include "HYMLS_Macros.hpp"
#include "HYMLS_Tools.hpp"
#include "HYMLS_MatrixUtils.hpp"

#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_Import.h"
#include "Epetra_MultiVector.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_MpiComm.h"
#include "Epetra_Operator.h"
#include "Epetra_Vector.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include "Ifpack_Amesos.h"

#include "EpetraExt_Reindex_CrsMatrix.h"
#include "./EpetraExt_RestrictedCrsMatrixWrapper.h"
#include "./EpetraExt_RestrictedMultiVectorWrapper.h"
#ifdef HYMLS_STORE_MATRICES
#include "EpetraExt_RowMatrixOut.h"
#endif

#include "HYMLS_Tester.hpp"
#include "HYMLS_AugmentedMatrix.hpp"

#include <iostream>

namespace HYMLS
  {

CoarseSolver::CoarseSolver(
  Teuchos::RCP<const Epetra_CrsMatrix> matrix,
  int level)
  :
  PLA("Coarse Solver"),
  comm_(Teuchos::rcp(matrix->Comm().Clone())),
  myLevel_(level),
  amActive_(true),
  matrix_(matrix),
  linearRhs_(Teuchos::null), linearSol_(Teuchos::null),
  haveBorder_(false),
  label_("CoarseSolver"),
  isEmpty_(false),
  initialized_(false), computed_(false)
  {
  }

void CoarseSolver::setParameterList(
  const Teuchos::RCP<Teuchos::ParameterList>& list)
  {
  HYMLS_LPROF3(label_, "setParameterList");
  setMyParamList(list);
  SetParameters(*list);
  }

// Sets all parameters for the preconditioner.
int CoarseSolver::SetParameters(Teuchos::ParameterList& List)
  {
  HYMLS_LPROF3(label_, "SetParameters");
  Teuchos::RCP<Teuchos::ParameterList> myPL = getMyNonconstParamList();

  if (myPL.get() != &List)
    {
    setMyParamList(Teuchos::rcp(&List, false));
    }

  fix_gid_.resize(0);

  int pos = 1;
  while (pos > 0)
    {
    std::string label = "Fix GID " + Teuchos::toString(pos);
    if (getMyParamList()->isParameter(label))
      {
      fix_gid_.append(getMyParamList()->get<int>(label));
      pos++;
      }
    else
      {
      pos = 0;
      }
    }

  HYMLS_DEBVAR(fix_gid_);

  return 0;
  }

int CoarseSolver::Initialize()
  {
  HYMLS_LPROF2(label_, "Initialize");

  // reindex the reduced system, this seems to be a good idea when
  // solving it using Ifpack_Amesos
  Epetra_BlockMap const &map = matrix_->Map();
  linearMap_ = Teuchos::rcp(new Epetra_Map((hymls_gidx)map.NumGlobalElements64(),
      map.NumMyElements(), 0, map.Comm()));

  isEmpty_ = map.NumGlobalElements64() == 0;

  reindexA_ = Teuchos::rcp(new ::EpetraExt::CrsMatrix_Reindex(*linearMap_));

  restrictA_ = Teuchos::rcp(new ::HYMLS::EpetraExt::RestrictedCrsMatrixWrapper());
  restrictX_ = Teuchos::rcp(new ::HYMLS::EpetraExt::RestrictedMultiVectorWrapper());
  restrictB_ = Teuchos::rcp(new ::HYMLS::EpetraExt::RestrictedMultiVectorWrapper());

  initialized_ = true;
  computed_ = false;
  haveBorder_ = false;

  return 0;
  }

bool CoarseSolver::IsInitialized() const
  {
  return initialized_;
  }

int CoarseSolver::Compute()
  {
  HYMLS_LPROF(label_, "Compute");

  // drop numerical zeros. We need to copy the matrix anyway because
  // we may want to put in some artificial Dirichlet conditions.
#ifdef HYMLS_TESTING
  Tools::Out("drop on coarsest level");
#endif

  reducedSchur_ = MatrixUtils::DropByValue(matrix_,
    HYMLS_SMALL_ENTRY, MatrixUtils::RelFullDiag);

  HYMLS_TEST(Label(), isFmatrix(*reducedSchur_), __FILE__, __LINE__);

  reducedSchur_->SetLabel(("Coarsest Matrix (level " + Teuchos::toString(myLevel_ + 1) + ")").c_str());

  if (!HaveBorder())
    {
    for (int i = 0; i < fix_gid_.length(); i++)
      {
      HYMLS_DEBUG("set Dirichlet node " << fix_gid_[i]);
      CHECK_ZERO(MatrixUtils::PutDirichlet(*reducedSchur_, fix_gid_[i]));
      }
    }

  HYMLS_DEBUG("reindex matrix to linear indexing");
  linearMatrix_ = Teuchos::rcp(&((*reindexA_)(*reducedSchur_)), false);

  // passed to direct solver - depends on what exactly we do
  Teuchos::RCP<Epetra_RowMatrix> S2 = Teuchos::null;

#ifdef RESTRICT_ON_COARSE_LEVEL
  int reducedNumProc = -1;
  if (Teuchos::rcp_dynamic_cast<const Epetra_MpiComm>(comm_) != Teuchos::null)
    {
    // restrict the matrix to the active processors
    // we have to restrict_comm again because the pointer is no longer
    // valid, it seems
    CHECK_ZERO(restrictA_->restrict_comm(linearMatrix_));
    amActive_ = restrictA_->RestrictedProcIsActive();
    restrictX_->SetMPISubComm(restrictA_->GetMPISubComm());
    restrictB_->SetMPISubComm(restrictA_->GetMPISubComm());
    restrictedMatrix_ = restrictA_->RestrictedMatrix();
    if (restrictA_->RestrictedProcIsActive())
      {
      reducedNumProc = restrictA_->RestrictedComm().NumProc();
      }
    }
  else
    {
    restrictedMatrix_ = Teuchos::rcp(new Epetra_CrsMatrix(*linearMatrix_));
    }

  // if we do not set this, Amesos may try to think of its own strategy
  // to reduce the number of procs, which in my experience leads to MPI
  // errors in MPI_Comm_free (as of Trilinos 10.0)
  if (PL().sublist("Coarse Solver").isParameter("MaxProcs") == false)
    {
    PL().sublist("Coarse Solver").set("MaxProcs", reducedNumProc);
    }
  HYMLS_DEBUG("next SC defined as restricted linear-index matrix");
  S2 = restrictedMatrix_;
#else
  HYMLS_DEBUG("next SC defined as linear-index matrix");
  S2 = linearMatrix_;
  amActive_ = true;
#endif

  ////////////////////////////////////////////////////////////////////////////
  // this next section is just for the bordered case                        //
  ////////////////////////////////////////////////////////////////////////////
  if (HaveBorder() && amActive_)
    {
    if (V_ == Teuchos::null || W_ == Teuchos::null || C_ == Teuchos::null)
      {
      Tools::Error("border not set correctly", __FILE__, __LINE__);
      }
#ifndef RESTRICT_ON_COARSE_LEVEL
    // we use the variant with restricting the number of ranks in comm usually
    Tools::Error("not implemented", __FILE__, __LINE__);
#endif

    HYMLS_DEBVAR(*V_);
    HYMLS_DEBVAR(*W_);
    HYMLS_DEBVAR(*C_);

    // we need to create views of the vectors here because the
    // map is different for the solver (linear restricted map)
    Teuchos::RCP<const Epetra_MultiVector> Vprime =
      Teuchos::rcp(new Epetra_MultiVector(View, restrictedMatrix_->RowMap(),
          V_->Values(), V_->Stride(), V_->NumVectors()));
    Teuchos::RCP<const Epetra_MultiVector> Wprime =
      Teuchos::rcp(new Epetra_MultiVector(View, restrictedMatrix_->RowMap(),
          W_->Values(), W_->Stride(), W_->NumVectors()));

    // create AugmentedMatrix, refactor reducedSchurSolver_
    augmentedMatrix_ = Teuchos::rcp
      (new HYMLS::AugmentedMatrix(restrictedMatrix_, Vprime, Wprime, C_));
    S2 = augmentedMatrix_;
    }

  ////////////////////////////////////////////////////////////////////////////
  // end bordered case section                                              //
  ////////////////////////////////////////////////////////////////////////////

  Teuchos::ParameterList &amesosList = PL().sublist("Coarse Solver");
  if (amActive_)
    {
    if (S2 == Teuchos::null)
      {
      Tools::Error("failed to select matrix for coarsest level", __FILE__, __LINE__);
      }
    reducedSchurSolver_ = Teuchos::rcp(new Ifpack_Amesos(S2.get()));
    CHECK_ZERO(reducedSchurSolver_->SetParameters(amesosList));
    HYMLS_DEBUG("Initialize direct solver");
    CHECK_ZERO(reducedSchurSolver_->Initialize());
    HYMLS_DEBUG("Compute direct solver");
    CHECK_ZERO(reducedSchurSolver_->Compute());
    }

  computed_ = true;

  return 0;
  }

bool CoarseSolver::IsComputed() const
  {
  return computed_;
  }

double CoarseSolver::Condest(const Ifpack_CondestType CT,
  const int MaxIters,
  const double Tol,
  Epetra_RowMatrix* Matrix)
  {
  return reducedSchurSolver_->Condest(CT, MaxIters, Tol, Matrix);
  }

double CoarseSolver::Condest() const
  {
  return reducedSchurSolver_->Condest();
  }

int CoarseSolver::ApplyInverse(const Epetra_MultiVector &X,
  Epetra_MultiVector &Y) const
  {
  HYMLS_LPROF(label_, "ApplyInverse");

  bool realloc_vectors = (linearRhs_ == Teuchos::null);
  if (!realloc_vectors) realloc_vectors = (linearRhs_->NumVectors() != X.NumVectors());
  if (realloc_vectors)
    {
    linearRhs_ = Teuchos::rcp(new Epetra_MultiVector(*linearMap_, X.NumVectors()));
    linearSol_ = Teuchos::rcp(new Epetra_MultiVector(*linearMap_, X.NumVectors()));
    }

  // Put the RHS in a vector with a linear map, which is required
  // for Ifpack_Amesos
  *linearRhs_ = X;

  // Add the boundary conditions
  for (int i = 0; i < fix_gid_.length(); i++)
    {
    int lid = X.Map().LID(fix_gid_[i]);
    if (lid > 0)
      {
      for (int k = 0; k < X.NumVectors(); k++)
        {
        (*linearRhs_)[k][lid] = 0.0;
        }
      }
    }

  if (realloc_vectors)
    {
#ifdef RESTRICT_ON_COARSE_LEVEL
    if (Teuchos::rcp_dynamic_cast<const Epetra_MpiComm>(comm_) != Teuchos::null)
      {
      CHECK_ZERO(restrictB_->restrict_comm(linearRhs_));
      CHECK_ZERO(restrictX_->restrict_comm(linearSol_));
      restrictedRhs_ = restrictB_->RestrictedMultiVector();
      restrictedSol_ = restrictX_->RestrictedMultiVector();
      }
    else
#endif
      {
      restrictedRhs_ = linearRhs_;
      restrictedSol_ = linearSol_;
      }
    }
  if (amActive_)
    {
    CHECK_ZERO(reducedSchurSolver_->ApplyInverse(*restrictedRhs_, *restrictedSol_));
    }
  // Put the solution back into the vector with the original map
  Y = *linearSol_;

  return 0;
  }

const Epetra_RowMatrix& CoarseSolver::Matrix() const
  {
  return *matrix_;
  }

int CoarseSolver::NumInitialize() const
  {
  return reducedSchurSolver_->NumInitialize();
  }

int CoarseSolver::NumCompute() const
  {
  return reducedSchurSolver_->NumCompute();
  }

int CoarseSolver::NumApplyInverse() const
  {
  return reducedSchurSolver_->NumApplyInverse();
  }

double CoarseSolver::InitializeTime() const
  {
  return reducedSchurSolver_->InitializeTime();
  }

double CoarseSolver::ComputeTime() const
  {
  return reducedSchurSolver_->ComputeTime();
  }

double CoarseSolver::ApplyInverseTime() const
  {
  return reducedSchurSolver_->ApplyInverseTime();
  }

double CoarseSolver::InitializeFlops() const
  {
  return reducedSchurSolver_->InitializeFlops();
  }

double CoarseSolver::ComputeFlops() const
  {
  return reducedSchurSolver_->ComputeFlops();
  }

double CoarseSolver::ApplyInverseFlops() const
  {
  return reducedSchurSolver_->ApplyInverseFlops();
  }

std::ostream& CoarseSolver::Print(std::ostream& os) const
  {
  return reducedSchurSolver_->Print(os);
  }

int CoarseSolver::SetUseTranspose(bool UseTranspose)
  {
  return -1;
  }

int CoarseSolver::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
  {
  return matrix_->Apply(X, Y);
  }

double CoarseSolver::NormInf() const
  {
  return reducedSchurSolver_->NormInf();
  }

const char * CoarseSolver::Label() const
  {
  return label_.c_str();
  }

bool CoarseSolver::UseTranspose() const
  {
  return false;
  }

bool CoarseSolver::HasNormInf() const
  {
  return false;
  }

const Epetra_Comm & CoarseSolver::Comm() const
  {
  return *comm_;
  }

const Epetra_Map & CoarseSolver::OperatorDomainMap() const
  {
  return matrix_->OperatorDomainMap();
  }

const Epetra_Map & CoarseSolver::OperatorRangeMap() const
  {
  return matrix_->OperatorRangeMap();
  }

int CoarseSolver::SetBorder(Teuchos::RCP<const Epetra_MultiVector> V,
  Teuchos::RCP<const Epetra_MultiVector> W,
  Teuchos::RCP<const Epetra_SerialDenseMatrix> C)
  {
  HYMLS_LPROF(label_, "SetBorder");

  if (V == Teuchos::null)
    {
    //unset
    haveBorder_ = false;
    return 0;
    }

  V_ = V;
  W_ = W;
  C_ = C;

  computed_ = false;
  haveBorder_ = true;
  return 0;
  }

int CoarseSolver::Apply(const Epetra_MultiVector & B, const Epetra_SerialDenseMatrix & C,
  Epetra_MultiVector& X, Epetra_SerialDenseMatrix & Y) const
  {
  return -1;
  }

// compute [X S]' = [K V;W' C]\[Y T]'
int CoarseSolver::ApplyInverse(const Epetra_MultiVector &X,
  const Epetra_SerialDenseMatrix &T,
  Epetra_MultiVector &Y,
  Epetra_SerialDenseMatrix &S) const
  {
  HYMLS_LPROF2(label_, "ApplyInverse (bordered)");

  if (isEmpty_) return 0;

  if (!IsComputed())
    {
    return -1;
    }

  if (!HaveBorder())
    {
    HYMLS_DEBUG("border not set!");
    return ApplyInverse(X, Y);
    }

  CHECK_ZERO(Y.PutScalar(0.0));
  if (amActive_)
    {
    // on the coarsest level we have put the border explicitly into an
    // AugmentedMatrix so we need to form the complete RHS
    bool realloc_vectors = (linearRhs_ == Teuchos::null);
    if (!realloc_vectors) realloc_vectors = (linearRhs_->NumVectors() != X.NumVectors());
    if (!realloc_vectors) realloc_vectors =
                            (linearRhs_->Map().SameAs(augmentedMatrix_->Map()) == false);
    if (realloc_vectors)
      {
      HYMLS_DEBUG("(re-)allocate tmp vectors");
      linearRhs_ = Teuchos::rcp(new
        Epetra_MultiVector(augmentedMatrix_->Map(), Y.NumVectors()));
      linearSol_ = Teuchos::rcp(new
        Epetra_MultiVector(augmentedMatrix_->Map(), Y.NumVectors()));
      }
    for (int j = 0; j < X.NumVectors(); j++)
      {
      for (int i = 0; i < X.MyLength(); i++)
        {
        (*linearRhs_)[j][i] = X[j][i];
        }
      for (int i = X.MyLength(); i < linearRhs_->MyLength(); i++)
        {
        int k = i - X.MyLength();
        (*linearRhs_)[j][i] = T[j][k];
        }
      }
/* in augmented systems we do not fix any GIDs, I guess...
   Wfor (int i=0;i<fix_gid_.length();i++)
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
      if (Teuchos::rcp_dynamic_cast<const Epetra_MpiComm>(comm_) != Teuchos::null)
        {
        CHECK_ZERO(restrictB_->restrict_comm(linearRhs_));
        CHECK_ZERO(restrictX_->restrict_comm(linearSol_));
        restrictedRhs_ = restrictB_->RestrictedMultiVector();
        restrictedSol_ = restrictX_->RestrictedMultiVector();
        }
      else
#endif
        {
        restrictedRhs_ = linearRhs_;
        restrictedSol_ = linearSol_;
        }
      }
    HYMLS_DEBUG("coarse level solve");
    CHECK_ZERO(reducedSchurSolver_->ApplyInverse(*restrictedRhs_, *restrictedSol_));

    // unscale the solution and split into X and S
    for (int j = 0; j < X.NumVectors(); j++)
      {
      for (int i = 0; i < X.MyLength(); i++)
        {
        Y[j][i] = (*linearSol_)[j][i];
        }
      for (int i = X.MyLength(); i < linearRhs_->MyLength(); i++)
        {
        int k = i - X.MyLength();
        S[j][k] = (*linearSol_)[j][i];
        }
      }
#ifdef HYMLS_DEBUGGING
    HYMLS::MatrixUtils::Dump(*linearRhs_, "CoarseLevelRhs.txt");
    HYMLS::MatrixUtils::Dump(*linearSol_, "CoarseLevelSol.txt");
#endif
    }

  return 0;
  }

  }
