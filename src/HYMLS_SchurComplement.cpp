#include "HYMLS_config.h"

#include "HYMLS_SchurComplement.hpp"
#include "HYMLS_OverlappingPartitioner.hpp"
#include "HYMLS_Macros.hpp"
#include "HYMLS_Tools.hpp"

#include "Epetra_Map.h"
#include "Epetra_MultiVector.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_FECrsMatrix.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_IntSerialDenseVector.h"
#include "Epetra_LongLongSerialDenseVector.h"

#include "Ifpack_ConfigDefs.h"
#include "Ifpack_Container.h"

#include "EpetraExt_MatrixMatrix.h"

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
    useTranspose_(false), normInf_(-1.0),
    label_("SchurComplement"),
    flopsApply_(0.0), flopsCompute_(0.0)
  {
  HYMLS_LPROF3(label_, "Constructor");
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
  Tools::Warning("Apply() not available!", __FILE__, __LINE__);
  return -1;
  }

// Apply inverse operator - not implemented.
int SchurComplement::ApplyInverse(const Epetra_MultiVector &X,
  Epetra_MultiVector &Y) const
  {
  Tools::Warning("ApplyInverse() not available!", __FILE__, __LINE__);
  return -1;
  }

int SchurComplement::Construct(Teuchos::RCP<Epetra_FECrsMatrix> S) const
  {
  HYMLS_LPROF3(label_, "Construct FEC");
#ifdef HYMLS_LONG_LONG
  Epetra_LongLongSerialDenseVector indices;
#else
  Epetra_IntSerialDenseVector indices;
#endif
  Epetra_SerialDenseMatrix Sk;

  const Epetra_Map &map = A22_->RowMap();
  const OverlappingPartitioner &hid = A22_->Partitioner();

  if (map.NumGlobalElements64() == 0) return 0; // empty SC

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

  // Add A21*A11\A12
  for (int k = 0; k < hid.NumMySubdomains(); k++)
    {
    // construct values for separators around subdomain k
    CHECK_ZERO(hid.getSeparatorGIDs(k, indices));
    CHECK_ZERO(Construct11(k, Sk, indices, &flopsCompute_));

    CHECK_ZERO(S->SumIntoGlobalValues(indices, Sk));
    }
  CHECK_ZERO(S->GlobalAssemble(false));
  CHECK_ZERO(::EpetraExt::MatrixMatrix::Add(*A22_->Block(), false, 1.0,
      *S, 1.0));

  // finish construction by creating local IDs:
  CHECK_ZERO(S->FillComplete());

  HYMLS_DEBUG("SchurComplement - GlobalAssembly");
  // CHECK_ZERO(S->GlobalAssemble());
  return 0;
  }

int SchurComplement::Construct11(int sd, Epetra_SerialDenseMatrix &Sk,
#ifdef HYMLS_LONG_LONG
  const Epetra_LongLongSerialDenseVector &inds,
#else
  const Epetra_IntSerialDenseVector &inds,
#endif
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

  int len;
  int *indices;
  double *values;
  int int_elems = hid.NumInteriorElements(sd);

  // Loop over all interior elements
  for (int i = 0; i < int_elems; i++)
    {
    // Get a view of the matrix row (with all separator couplings)
    CHECK_ZERO(A12.ExtractMyRowView(i, len, values, indices));

    // A11 ID stores local indices of the original matrix
    // loop over the matrix row and look for matching entries
    for (int k = 0 ; k < len; k++)
      {
      const hymls_gidx gcid = A12.GCID64(indices[k]);

      // Loop over all GIDs of separators around this subdomain
      for (int j = 0; j < nrows; j++)
        {
        if (gcid == inds[j])
          {
          A11.RHS(i, j) = values[k];
          break;
          }
        }
      }
    }

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
    const int lrid = A12.LRID(hid.OverlappingMap().GID64(A11.ID(j)));
    for (int k = 0; k < nrows; k++)
      {
      B[k][lrid] = A11.LHS(j, k);
      }
    }

  // multiply by A21, giving A21*(A11\A12) in a vector based on Map2 (i.e. with a row
  // for each separator element) and a column for each separator node connected to this
  // subdomain. Some separators may not be on this CPU: those need to be imported
  // manually later on.

  CHECK_ZERO(A21.Multiply(false, B, Aloc));

#ifdef FLOPS_COUNT
  flops += 2 * B.NumVectors() *A21.NumGlobalNonzeros64();
#endif
  // re-index and put into final block

//    HYMLS_DEBUG("Copy into Sk matrix");
  for (int i = 0; i < nrows; i++)
    {
    // A21*A11\A12 part
    int lrid = A21.RowMap().LID(inds[i]);
    for (int j = 0; j < nrows; j++)
      {
      Sk(i, j) = -Aloc[j][lrid];
      }
    }

  A11.SetNumVectors(1);

//    HYMLS_DEBUG("Block constructed successfully!");
#ifdef FLOPS_COUNT
  if (count_flops != NULL) *count_flops += flops;
#endif
  return 0;
  }

int SchurComplement::Construct22(int sd, Epetra_SerialDenseMatrix &Sk,
#ifdef HYMLS_LONG_LONG
  const Epetra_LongLongSerialDenseVector &inds,
#else
  const Epetra_IntSerialDenseVector &inds,
#endif
  double *count_flops) const
  {
  HYMLS_LPROF3(label_, "Construct SDM");

  const OverlappingPartitioner &hid = A22_->Partitioner();
  const Epetra_CrsMatrix &A22 = *A22_->SubBlock(sd);

  if (sd < 0 || sd > hid.NumMySubdomains())
    {
    Tools::Warning("Subdomain index out of range!", __FILE__, __LINE__);
    return -1;
    }

  int nrows = hid.NumSeparatorElements(sd);

  if (inds.Length() != nrows)
    {
    return -1; // caller probably did not call Construct(indices)
    }

  if (Sk.M() != nrows || Sk.N() != nrows)
    {
    CHECK_ZERO(Sk.Shape(nrows, nrows));
    }

  CHECK_ZERO(Sk.Scale(0.0));

  int len;
  int *indices;
  double *values;

  for (int i = 0; i < nrows; i++)
    {
    // A22 part
    int lrid = A22.RowMap().LID(inds[i]);
    CHECK_ZERO(A22.ExtractMyRowView(lrid, len, values, indices));
    for (int k = 0; k < len; k++)
      {
      const hymls_gidx gcid = A22.GCID64(indices[k]);
      for (int j = 0; j < nrows; j++)
        {
        if (gcid == inds[j])
          {
          Sk(i, j) = values[k];
          break;
          }
        }
      }
    }
  return 0;
  }

  }
