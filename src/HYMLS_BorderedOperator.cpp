#include "HYMLS_BorderedOperator.hpp"

#include "HYMLS_config.h"

#include "HYMLS_Macros.hpp"
#include "HYMLS_Tools.hpp"
#include "HYMLS_BorderedVector.hpp"
#include "HYMLS_DenseUtils.hpp"

#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_Operator.h"
#include "Epetra_Import.h"
#include "Epetra_MultiVector.h"
#include "Epetra_SerialDenseMatrix.h"

#include "BelosEpetraAdapter.hpp"

namespace HYMLS
  {
BorderedOperator::BorderedOperator()
  :
  A_(Teuchos::null), V_(Teuchos::null), W_(Teuchos::null), C_(Teuchos::null),
  label_("BorderedOperator")
  {
  HYMLS_PROF3("BorderedOperator", "Constructor");
  }

BorderedOperator::BorderedOperator(Teuchos::RCP<const Epetra_Operator> A,
  Teuchos::RCP<const Epetra_MultiVector> V,
  Teuchos::RCP<const Epetra_MultiVector> W,
  Teuchos::RCP<const Epetra_SerialDenseMatrix> C)
  :
  A_(A), V_(V), W_(W), C_(C),
  label_("BorderedOperator")
  {
  HYMLS_PROF3("BorderedOperator", "Constructor");

  if (A_->OperatorRangeMap().SameAs(A_->OperatorDomainMap())==false)
    {
    Tools::Error("operator must be 'square'",__FILE__,__LINE__);
    }

  if (A_->OperatorRangeMap().SameAs(V_->Map())==false)
    {
    Tools::Error("operator and vector space must have compatible maps",
      __FILE__,__LINE__);
    }

  if (W_ == Teuchos::null)
    {
    W_ = V_;
    }
  }

int BorderedOperator::Apply(const BorderedVector& X, BorderedVector& Y) const
  {
  HYMLS_PROF3("BorderedOperator", "Apply");
  return Apply(*X.Vector(), *X.Border(), *Y.Vector(), *Y.Border());
  }

int BorderedOperator::ApplyInverse(const BorderedVector& X, BorderedVector& Y) const
  {
  HYMLS_PROF3("BorderedOperator", "ApplyInverse");
  return ApplyInverse(*X.Vector(), *X.Border(), *Y.Vector(), *Y.Border());
  }

int BorderedOperator::SetBorder(Teuchos::RCP<const Epetra_MultiVector> V,
  Teuchos::RCP<const Epetra_MultiVector> W,
  Teuchos::RCP<const Epetra_SerialDenseMatrix> C)
  {
  V_ = V;
  W_ = W;
  C_ = C;

  if (W_ == Teuchos::null)
    W_ = V_;

  return 0;
  }

//! compute [Y T]' = [K V;W' C]*[X S]'
int BorderedOperator::Apply(
  const Epetra_MultiVector& X, const Epetra_SerialDenseMatrix& S,
  Epetra_MultiVector& Y, Epetra_SerialDenseMatrix& T) const
  {
  if (A_ == Teuchos::null)
    return -1;

  CHECK_ZERO(A_->Apply(X, Y));

  if (V_ == Teuchos::null)
    return 1;

  CHECK_ZERO(Y.Multiply('N', 'N', 1.0, *V_, *DenseUtils::CreateView(S), 1.0));

  Epetra_SerialDenseMatrix tmp;
  DenseUtils::MatMul(*W_, X, tmp);

  if (T.M() != S.M() && T.N() != Y.NumVectors())
    {
    Tools::Error("T has the wrong shape", __FILE__, __LINE__);
    }

  // Set T to zero. First randomize it to make sure values are not uninitialized
  CHECK_ZERO(T.Random())
  CHECK_ZERO(T.Scale(0.0));

  if (!C_.is_null())
    {
    // Apply for a SerialDenseMatrix is non-const. No clue why
    // CHECK_ZERO(C_->Apply(S, T));
    Epetra_SerialDenseMatrix C(*C_);
    CHECK_ZERO(C.Apply(S, T));
    }
  T += tmp;

  return 0;
  }

//! compute [Y T]' = [K V;W' C]\[X S]'
int BorderedOperator::ApplyInverse(
  const Epetra_MultiVector& X, const Epetra_SerialDenseMatrix& S,
  Epetra_MultiVector& Y, Epetra_SerialDenseMatrix& T) const
  {
  return -99;
  }

  } // namespace HYMLS

namespace Belos
  {

void OperatorTraits<double, HYMLS::BorderedVector, HYMLS::BorderedOperator>::Apply(
  HYMLS::BorderedOperator const &Op, HYMLS::BorderedVector const &x,
  HYMLS::BorderedVector &y, int trans)
  {
  int ierr = Op.ApplyInverse(x, y);
  if (ierr == -99)
    {
    ierr = Op.Apply(x, y);

    TEUCHOS_TEST_FOR_EXCEPTION(ierr != 0, EpetraOpFailure,
      "Belos::OperatorTraits::Apply: Calling Apply() on the "
      "underlying BorderedOperator object failed, returning a "
      "nonzero error code of " << ierr << ".");
    return;
    }

  TEUCHOS_TEST_FOR_EXCEPTION(ierr != 0, EpetraOpFailure,
    "Belos::OperatorTraits::Apply: Calling ApplyInverse() on the "
    "underlying BorderedOperator object failed, returning a "
    "nonzero error code of " << ierr << ". This probably means "
    "that the underlying BorderedOperator object doesn't know "
    "how to apply.");
  }

  } // namespace Belos
