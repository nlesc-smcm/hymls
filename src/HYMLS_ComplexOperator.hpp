#ifndef HYMLS_COMPLEX_OPERATOR_H
#define HYMLS_COMPLEX_OPERATOR_H

#include "Teuchos_RCP.hpp"

#include "BelosConfigDefs.hpp"
#include "BelosTypes.hpp"
#include "BelosOperator.hpp"

#include "BelosEpetraAdapter.hpp"

#include "HYMLS_Tools.hpp"
#include "HYMLS_Macros.hpp"

#include "Epetra_DataAccess.h"

class Epetra_MultiVector;
class Epetra_Operator;
class Epetra_SerialDenseMatrix;

namespace HYMLS
  {
template<typename>
class ComplexVector;

template<class Operator, class MultiVector>
class ComplexOperator
  {
public:
  ComplexOperator();

  ComplexOperator(Teuchos::RCP<const Operator> A, bool isPreconditioner=false);

  virtual ~ComplexOperator() {};

  bool IsPreconditioner() const;

  int Apply(const ComplexVector<MultiVector>& X, ComplexVector<MultiVector>& Y) const;

  int ApplyInverse(const ComplexVector<MultiVector>& X, ComplexVector<MultiVector>& Y) const;

protected:
  //! original operator
  Teuchos::RCP<const Operator> A_;

private:
  bool isPreconditioner_;

  //! label
  std::string label_;

  };

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
class OperatorTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector>, HYMLS::ComplexOperator<Operator, MultiVector> >
  {
public:
  static void
  Apply (HYMLS::ComplexOperator<Operator, MultiVector> const &Op, HYMLS::ComplexVector<MultiVector> const &x,
    HYMLS::ComplexVector<MultiVector> &y, int trans = 0)
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

  static bool
  HasApplyTranspose (const HYMLS::ComplexOperator<Operator, MultiVector> &Op) { return false; }
  };

  } // namespace Belos



#endif
