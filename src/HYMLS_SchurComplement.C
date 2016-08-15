#include "HYMLS_no_debug.H"

#include "HYMLS_SchurComplement.H"
#include "HYMLS_OverlappingPartitioner.H"
#include "HYMLS_SparseDirectSolver.H"
#include "HYMLS_MatrixUtils.H"

#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_FECrsMatrix.h"
#include "Epetra_Import.h"

#include "Ifpack_Container.h"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Array.hpp"

#include "EpetraExt_MatrixMatrix.h"
#include "EpetraExt_Reindex_MultiVector.h"


namespace HYMLS {

// operator representation of our Schur complement.
// allows applying the Schur complement of our factorization
// to a vector without actually constructing it.
// Also provides functionality to explicitly construct parts
// of the SC or the whole thing as sparse or dense matrix.

SchurComplement::SchurComplement(
  Teuchos::RCP<const MatrixBlock> A11,
  Teuchos::RCP<const MatrixBlock> A12,
  Teuchos::RCP<const MatrixBlock> A21,
  Teuchos::RCP<const MatrixBlock> A22,
  int lev)
  : A11_(A11), A12_(A12), A21_(A21), A22_(A22),
    myLevel_(lev),
    sparseMatrixRepresentation_(Teuchos::null),
    useTranspose_(false), normInf_(-1.0),
    label_("SchurComplement"),
    flopsApply_(0.0), flopsCompute_(0.0)
  {
  HYMLS_LPROF3(label_, "Constructor");
  isConstructed_ = false;
  // we do a finite-element style assembly of the full matrix
  const Epetra_Map &map = A22_->RowMap();
  sparseMatrixRepresentation_ = Teuchos::rcp(new
    Epetra_FECrsMatrix(Copy, map, A22_->Matrix()->MaxNumEntries()));
  Scrs_ = sparseMatrixRepresentation_;
  sca_left_ = Teuchos::rcp(new Epetra_Vector(map));
  sca_right_ = Teuchos::rcp(new Epetra_Vector(map));
  sca_left_->PutScalar(1.0);
  sca_right_->PutScalar(1.0);
  }

// destructor
SchurComplement::~SchurComplement()
  {
  HYMLS_LPROF3(label_, "Destructor");
  }

// Applies the operator. Here X and Y are based on the rowmap of A22
int SchurComplement::Apply(const Epetra_MultiVector &X,
  Epetra_MultiVector &Y) const
  {
  HYMLS_LPROF2(label_, "Apply");
  int ierr = 0;
  if (IsConstructed())
    {
    CHECK_ZERO(Scrs_->Apply(X, Y));
#ifdef FLOPS_COUNT
    flopsApply_ += 2 * Scrs_->NumGlobalNonzeros();
#endif
    }
  else
    {
    // we now have overlap in the rowMap of the Preconditioner class
    // and I can't oversee if this would still work.
    Tools::Error("distributed SC currently disabled", __FILE__, __LINE__);
#if 0
    // The Schur-complement is given by A22-A21*A11\A12
    CHECK_ZERO(A22_->Apply(X, Y));

    // 2) compute y2 = A21*A11\A12*X
    Epetra_MultiVector Y1(A11_->RowMap(), Y.NumVectors());
    Epetra_MultiVector Z1(A11_->RowMap(), Y.NumVectors());
    Epetra_MultiVector Y2(A22_->RowMap(), Y.NumVectors());

    CHECK_ZERO(A12_->Apply(X, Y1));

    CHECK_ZERO(A11_->ApplyInverse(Y1, Z1));

    CHECK_ZERO(A21_->Apply(Z1, Y2));

    // 3) compute Y = Y-Y2
    CHECK_ZERO(Y.Update(-1.0, Y2, 1.0));
#ifdef FLOPS_COUNT
    flopsApply_ += Y.GlobalLength() *Y.NumVectors();
#endif
#endif
    }
  return ierr;
  }

// Apply inverse operator - not implemented.
int SchurComplement::ApplyInverse(const Epetra_MultiVector &X,
  Epetra_MultiVector &Y) const
  {
  Tools::Warning("ApplyInverse() not available!", __FILE__, __LINE__);
  return -1;
  }

// construct complete Schur complement as a sparse matrix
int SchurComplement::Construct()
  {
  HYMLS_LPROF3(label_, "Construct (1)");

  isConstructed_ = true;
  CHECK_ZERO(this->Construct(sparseMatrixRepresentation_));
  Scrs_ = MatrixUtils::DropByValue(sparseMatrixRepresentation_,
    HYMLS_SMALL_ENTRY);
  REPORT_MEM(label_, "SchurComplement", Scrs_->NumGlobalNonzeros(),
    Scrs_->NumGlobalNonzeros() +
    Scrs_->NumGlobalRows());
  return 0;
  }

int SchurComplement::Construct(Teuchos::RCP<Epetra_FECrsMatrix> S) const
  {
  HYMLS_LPROF3(label_, "Construct FEC");
  Epetra_IntSerialDenseVector indices;
  Epetra_SerialDenseMatrix Sk;

  const Epetra_Map &map = A22_->RowMap();
  const OverlappingPartitioner &hid = A22_->Partitioner();

  if (map.NumGlobalElements() == 0) return 0; // empty SC

  if (!S->Filled())
    {

    // start out by just putting the structure together.
    // I do this because the SumInto function will fail
    // unless the values have been put in already. On the
    // other hand, the Insert function will overwrite stuff
    // we put in previously.

    for (int k = 0; k < hid.NumMySubdomains(); k++)
      {
      CHECK_ZERO(hid.getSeparatorGIDs(k, indices));
      HYMLS_DEBVAR(k);
      HYMLS_DEBVAR(indices);
      if (indices.Length() != Sk.N())
        {
        Sk.Shape(indices.Length(), indices.Length());
        }
      int ierr = S->InsertGlobalValues(indices, Sk);
      if (ierr < 0)
        {
        Tools::Warning("error " + Teuchos::toString(ierr) + " returned from call S->InsertGlobalValues",
          __FILE__, __LINE__);
        return ierr;
        }
      }

    HYMLS_DEBUG("SchurComplement: Assembly with all zeros...");
    //assemble without calling FillComplete because we
    // still miss A22 in the pattern
    CHECK_ZERO(S->GlobalAssemble(false));
    }
  else
    {
    CHECK_ZERO(S->PutScalar(0.0));
    }

  for (int k = 0; k < hid.NumMySubdomains(); k++)
    {
    // construct values for separators around subdomain k
    CHECK_ZERO(hid.getSeparatorGIDs(k, indices));
    CHECK_ZERO(Construct(k, Sk, indices, &flopsCompute_));

    CHECK_ZERO(S->SumIntoGlobalValues(indices, Sk));
    }
  HYMLS_DEBUG("SchurComplement - GlobalAssembly");
  CHECK_ZERO(S->GlobalAssemble(false));
  CHECK_ZERO(EpetraExt::MatrixMatrix::Add(*A22_->Block(), false, 1.0,
      *S, -1.0));
  // finish construction by creating local IDs:
  CHECK_ZERO(S->FillComplete());
  return 0;
  }

int SchurComplement::Construct(int sd, Epetra_SerialDenseMatrix &Sk,
  const Epetra_IntSerialDenseVector &inds,
  double *count_flops) const
  {
  HYMLS_LPROF3(label_, "Construct SDM");
#ifdef FLOPS_COUNT
  double flops = 0;
#endif
  const OverlappingPartitioner &hid = A22_->Partitioner();
  const Epetra_CrsMatrix &A12 = *A12_->SubBlock(sd);
  const Epetra_CrsMatrix &A21 = *A21_->SubBlock(sd);
  Ifpack_Container &A11 = *A11_->SubdomainSolver(sd);

  if (sd < 0 || sd > hid.NumMySubdomains())
    {
    Tools::Warning("Subdomain index out of range!", __FILE__, __LINE__);
    return -1;
    }

#ifdef HYMLS_TESTING
  // verify that the ID array of the subdomain solver is sorted
  // in ascending order, I think we assume that...
  for (int i = 1; i < A11.NumRows(); i++)
    {
    if (A11.ID(i) < A11.ID(i - 1))
      {
      Tools::Warning("re-indexing of blocks is not supported!", __FILE__, __LINE__);
      }
    }
#endif

  int nrows = hid.NumSeparatorElements(sd);

  if (inds.Length() != nrows)
    {
    return -1; // caller probably did not call Construct(indices)
    }

  if (Sk.M() != nrows || Sk.N() != nrows)
    {
    CHECK_ZERO(Sk.Shape(nrows, nrows));
    }
  
  if (A11.NumRows() == 0)
    {
    return 0; // has only an A22-contribution (no interior elements)
    }

  A11.SetNumVectors(nrows);

  HYMLS_DEBVAR(sd);
  HYMLS_DEBVAR(inds);
  HYMLS_DEBVAR(nrows);

  int int_elems = hid.NumInteriorElements(sd);
  int *len = new int[int_elems];
  int **indices = new int*[int_elems];
  double **values = new double*[int_elems];
  // loop over all rows in this subdomain
  for (int i = 0; i < int_elems; i++)
    {
    // get a view of the matrix row (with all separator couplings)
    CHECK_ZERO(A12.ExtractMyRowView(i, len[i], values[i], indices[i]));
    }

  // Loop over all GIDs of separators around this subdomain
  for (int pos = 0; pos < nrows; pos++)
    {
    int gcid = inds[pos];
    // loop over all rows in this subdomain
    for (int i = 0; i < int_elems; i++)
      {
      // A11 ID stores local indices of the original matrix
      const int lrid = A12.LRID(A11_->ExtendedMatrix()->GRID(A11.ID(i)));
      // loop over the matrix row and look for matching entries
      for (int k = 0 ; k < len[lrid]; k++)
        {
        if (gcid == A12.GCID(indices[lrid][k]))
          A11.RHS(i, pos) = values[lrid][k];
        }
      }
    }

  delete[] len;
  delete[] indices;
  delete[] values;

//    HYMLS_DEBUG("Apply A11 inverse...");
#ifdef FLOPS_COUNT
  double flopsOld = A11.ApplyInverseFlops();
#endif
  IFPACK_CHK_ERR(A11.ApplyInverse());
#ifdef FLOPS_COUNT
  double flopsNew = A11.ApplyInverseFlops();
  //TODO: these flops are counted twice: in Solver->ApplyInverse() they shouldn't
  //      contribute!
  flops += flopsNew - flopsOld;
#endif

  // get the solution, B=A11\A12, as a MultiVector in the domain map of operator A21
  Epetra_MultiVector B(A12.RowMap(), nrows);
  Epetra_MultiVector Aloc(A21.RowMap(), B.NumVectors());
  for (int j = 0; j < B.MyLength(); j++)
    {
    for (int k = 0; k < nrows; k++)
      {
      const int lrid = A12.LRID(A11_->ExtendedMatrix()->GRID(A11.ID(j)));
      B[k][lrid] = A11.LHS(j, k);
      }
    }

  // multiply by A21, giving A21*(A11\A12) in a vector based on Map2 (i.e. with a row
  // for each separator element) and a column for each separator node connected to this
  // subdomain. Some separators may not be on this CPU: those need to be imported
  // manually later on.

  CHECK_ZERO(A21.Multiply(false, B, Aloc));

#ifdef FLOPS_COUNT
  flops += 2 * B.NumVectors() *A21.NumGlobalNonzeros();
#endif
  // re-index and put into final block

//    HYMLS_DEBUG("Copy into Sk matrix");
  for (int i = 0; i < nrows; i++)
    {
    int lrid = A21.RowMap().LID(inds[i]);
    for (int j = 0; j < nrows; j++)
      {
      Sk(i, j) = Aloc[j][lrid];
      }
    }

  A11.SetNumVectors(1);

//    HYMLS_DEBUG("Block constructed successfully!");
#ifdef FLOPS_COUNT
  if (count_flops != NULL) *count_flops += flops;
#endif
  return 0;
  }

Teuchos::RCP<Epetra_Vector> SchurComplement::ConstructLeftScaling(int p_variable)
  {
  HYMLS_LPROF2(label_, "ConstructLeftScaling");
  sca_left_->PutScalar(1.0);
  double *val;
  int *ind;
  int len;
  bool has_pcol;
  double diag;
  const OverlappingPartitioner &hid = A22_->Partitioner();
  const BasePartitioner &BP = hid.Partitioner();
  if (!IsConstructed())
    {
    Tools::Warning("Schur-complement not constructed, using default scaling", __FILE__, __LINE__);
    }
  else
    {
    for (int i = 0; i < Scrs_->NumMyRows(); i++)
      {
      diag = 1.0;
      has_pcol = false;
      CHECK_ZERO(Scrs_->ExtractMyRowView(i, len, val, ind));
      for (int j = 0; j < len; j++)
        {
        if (Scrs_->GRID(i) == Scrs_->GCID(ind[j]))
          {
          diag = std::abs(val[j]);
          }
        if (BP.VariableType(Scrs_->GCID(ind[j])) == p_variable)
          {
          if (std::abs(val[j]) > 1.0e-8) has_pcol = true;
          }
        }
      if ((has_pcol == false) && (diag > 1.0e-10))
        {
        (*sca_left_)[i] = 1.0 / diag;
        }
      }
    }
  return sca_left_;
  }

int SchurComplement::Scale(Teuchos::RCP<Epetra_Vector> sca_left, Teuchos::RCP<Epetra_Vector> sca_right)
  {
  HYMLS_LPROF3(label_, "Scale");
  int ierr = 0;
  if (!IsConstructed())
    {
    ierr = 1;
    }
  else
    {
    ierr = sparseMatrixRepresentation_->LeftScale(*sca_left);
    if (ierr == 0)
      {
      ierr = sparseMatrixRepresentation_->RightScale(*sca_right);
      }
    }
  return ierr;
  }

int SchurComplement::Unscale(Teuchos::RCP<Epetra_Vector> sca_left, Teuchos::RCP<Epetra_Vector> sca_right)
  {
  HYMLS_LPROF3(label_, "Unscale");

  int ierr = 0;
  if (!IsConstructed())
    {
    ierr = 1;
    }
  else
    {
    Epetra_Vector left(sca_left->Map());
    left.Reciprocal(*sca_left);
    Epetra_Vector right(sca_right->Map());
    right.Reciprocal(*sca_right);
    ierr = sparseMatrixRepresentation_->LeftScale(left);
    if (ierr == 0)
      {
      ierr = sparseMatrixRepresentation_->RightScale(right);
      }
    }
  return ierr;
  }

  }
