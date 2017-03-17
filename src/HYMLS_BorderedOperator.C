#include "HYMLS_BorderedOperator.H"

#include "HYMLS_Macros.H"
#include "HYMLS_Tools.H"
#include "HYMLS_BorderedVector.H"
#include "HYMLS_DenseUtils.H"

#include "Epetra_MultiVector.h"
#include "Epetra_SerialDenseMatrix.h"

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

int BorderedOperator::setBorder(Teuchos::RCP<const Epetra_MultiVector> V,
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

  Teuchos::RCP<Epetra_MultiVector> Tview = DenseUtils::CreateView(T);
  if (!C_.is_null())
    {
    // Apply for a SerialDenseMatrix is non-const. No clue why
    // CHECK_ZERO(C_->Apply(S, T));
    Epetra_SerialDenseMatrix C(*C_);
    CHECK_ZERO(C.Apply(S, T));
    }
  else
    {
    CHECK_ZERO(Tview->PutScalar(0.0));
    }
  CHECK_ZERO(Tview->Multiply('T', 'N', 1.0, *W_, X, 1.0));

  return 0;
  }

//! compute [Y T]' = [K V;W' C]\[X S]'
int BorderedOperator::ApplyInverse(
  const Epetra_MultiVector& X, const Epetra_SerialDenseMatrix& S,
  Epetra_MultiVector& Y, Epetra_SerialDenseMatrix& T) const
  {
  return -99;
  }

  }//namespace HYMLS
