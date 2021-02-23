#include "HYMLS_ComplexOperator.hpp"

#include "BelosEpetraAdapter.hpp"

#include "HYMLS_BorderedOperator.hpp"
#include "HYMLS_BorderedVector.hpp"
#include "HYMLS_ComplexVector.hpp"
#include "HYMLS_Tools.hpp"
#include "HYMLS_Macros.hpp"

template class HYMLS::ComplexOperator<Epetra_Operator, Epetra_MultiVector>;
template class HYMLS::ComplexOperator<HYMLS::BorderedOperator, HYMLS::BorderedVector>;
template class Belos::OperatorTraits<std::complex<double>, HYMLS::ComplexVector<Epetra_MultiVector>,
                                     HYMLS::ComplexOperator<Epetra_Operator, Epetra_MultiVector> >;
template class Belos::OperatorTraits<std::complex<double>, HYMLS::ComplexVector<HYMLS::BorderedVector>,
                                     HYMLS::ComplexOperator<HYMLS::BorderedOperator, HYMLS::BorderedVector> >;

namespace HYMLS
  {

template<class Operator, class MultiVector>
ComplexOperator<Operator, MultiVector>::ComplexOperator()
  :
  A_(Teuchos::null),
  label_("ComplexOperator")
  {
  HYMLS_PROF3("ComplexOperator", "Constructor");
  }

template<class Operator, class MultiVector>
ComplexOperator<Operator, MultiVector>::ComplexOperator(Teuchos::RCP<const Operator> A, bool isPreconditioner)
  :
  A_(A),
  isPreconditioner_(isPreconditioner),
  label_("ComplexOperator")
  {
  HYMLS_PROF3("ComplexOperator", "Constructor");
  }


template<class Operator, class MultiVector>
bool ComplexOperator<Operator, MultiVector>::IsPreconditioner() const
  {
  return isPreconditioner_;
  }

template<class Operator, class MultiVector>
int ComplexOperator<Operator, MultiVector>::Apply(const ComplexVector<MultiVector>& X, ComplexVector<MultiVector>& Y) const
  {
  HYMLS_PROF3("ComplexOperator", "Apply");

  if (X.NumVectors() != 1)
    Tools::Error("Only one complex vector supported", __FILE__, __LINE__);

  Teuchos::RCP<MultiVector> XMV = Belos::MultiVecTraits<double, MultiVector>::Clone(*X.Real(), 2);
  Teuchos::RCP<MultiVector> YMV = Belos::MultiVecTraits<double, MultiVector>::Clone(*X.Real(), 2);

  MultiVector XMV_real(View, *XMV, 0, 1);
  MultiVector XMV_imag(View, *XMV, 1, 1);

  XMV_real = *X.Real();
  XMV_imag = *X.Imag();

  int ierr = A_->Apply(*XMV, *YMV);

  MultiVector YMV_real(View, *YMV, 0, 1);
  MultiVector YMV_imag(View, *YMV, 1, 1);

  *Y.Real() = YMV_real;
  *Y.Imag() = YMV_imag;

  return ierr;
  }

template<class Operator, class MultiVector>
int ComplexOperator<Operator, MultiVector>::ApplyInverse(const ComplexVector<MultiVector>& X, ComplexVector<MultiVector>& Y) const
  {
  HYMLS_PROF3("ComplexOperator", "ApplyInverse");

  if (X.NumVectors() != 1)
    Tools::Error("Only one complex vector supported", __FILE__, __LINE__);

  Teuchos::RCP<MultiVector> XMV = Belos::MultiVecTraits<double, MultiVector>::Clone(*X.Real(), 2);
  Teuchos::RCP<MultiVector> YMV = Belos::MultiVecTraits<double, MultiVector>::Clone(*X.Real(), 2);

  MultiVector XMV_real(View, *XMV, 0, 1);
  MultiVector XMV_imag(View, *XMV, 1, 1);

  XMV_real = *X.Real();
  XMV_imag = *X.Imag();

  int ierr = A_->ApplyInverse(*XMV, *YMV);

  MultiVector YMV_real(View, *YMV, 0, 1);
  MultiVector YMV_imag(View, *YMV, 1, 1);

  *Y.Real() = YMV_real;
  *Y.Imag() = YMV_imag;

  return ierr;
  }

  } // namespace HYMLS

namespace Belos
  {

template<class Operator, class MultiVector>
void OperatorTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector>, HYMLS::ComplexOperator<Operator, MultiVector> >::Apply(
  HYMLS::ComplexOperator<Operator, MultiVector> const &Op, HYMLS::ComplexVector<MultiVector> const &x,
  HYMLS::ComplexVector<MultiVector> &y, int trans)
  {
  if (Op.IsPreconditioner())
    {
    int ierr = Op.ApplyInverse(x, y);

    TEUCHOS_TEST_FOR_EXCEPTION(ierr != 0, EpetraOpFailure,
      "Belos::OperatorTraits::Apply: Calling ApplyInverse() on the "
      "underlying ComplexOperator object failed, returning a "
      "nonzero error code of " << ierr << ". This probably means "
      "that the underlying ComplexOperator object doesn't know "
      "how to apply.");

    return;
    }

  int ierr = Op.Apply(x, y);

  TEUCHOS_TEST_FOR_EXCEPTION(ierr != 0, EpetraOpFailure,
    "Belos::OperatorTraits::Apply: Calling Apply() on the "
    "underlying ComplexOperator object failed, returning a "
    "nonzero error code of " << ierr << ".");
  }

  } // namespace Belos
