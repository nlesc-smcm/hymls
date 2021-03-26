#ifndef HYMLS_COMPLEX_OPERATOR_H
#define HYMLS_COMPLEX_OPERATOR_H

#include <complex>

#include "Teuchos_RCP.hpp"

#include "BelosOperator.hpp"

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

  bool isPreconditioner_;

private:
  //! label
  std::string label_;

  };

  } // namespace HYMLS

namespace Belos
  {
template<class Operator, class MultiVector>
class OperatorTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector>, HYMLS::ComplexOperator<Operator, MultiVector> >
  {
public:
  static void
  Apply (HYMLS::ComplexOperator<Operator, MultiVector> const &Op, HYMLS::ComplexVector<MultiVector> const &x,
    HYMLS::ComplexVector<MultiVector> &y, int trans = 0);

  static bool
  HasApplyTranspose (const HYMLS::ComplexOperator<Operator, MultiVector> &Op) { return false; }
  };

  } // namespace Belos

#endif
