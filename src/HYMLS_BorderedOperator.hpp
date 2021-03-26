#ifndef HYMLS_BORDERED_OPERATOR_H
#define HYMLS_BORDERED_OPERATOR_H

#include "Teuchos_RCP.hpp"

#include "BelosOperator.hpp"

class Epetra_MultiVector;
class Epetra_Operator;
class Epetra_SerialDenseMatrix;

namespace HYMLS
  {
class BorderedVector;

//! Given an operator A and a border, this class applies the
//! borderded matrix
class BorderedOperator
  {
public:
  BorderedOperator();

  BorderedOperator(Teuchos::RCP<const Epetra_Operator> A,
    bool isPreconditioner=false);

  BorderedOperator(Teuchos::RCP<const Epetra_Operator> A,
    Teuchos::RCP<const Epetra_MultiVector> V,
    Teuchos::RCP<const Epetra_MultiVector> W=Teuchos::null,
    Teuchos::RCP<const Epetra_SerialDenseMatrix> C=Teuchos::null,
    bool isPreconditioner=false);

  virtual ~BorderedOperator() {};

  bool IsPreconditioner() const;

  int Apply(const BorderedVector& X, BorderedVector& Y) const;

  int ApplyInverse(const BorderedVector& X, BorderedVector& Y) const;

  virtual int SetBorder(Teuchos::RCP<const Epetra_MultiVector> V,
    Teuchos::RCP<const Epetra_MultiVector> W=Teuchos::null,
    Teuchos::RCP<const Epetra_SerialDenseMatrix> C=Teuchos::null);

  //! compute [Y T]' = [K V;W' C]*[X S]'
  virtual int Apply(const Epetra_MultiVector& X, const Epetra_SerialDenseMatrix& S,
    Epetra_MultiVector& Y, Epetra_SerialDenseMatrix& T) const;

  //! compute [Y T]' = [K V;W' C]\[X S]'
  virtual int ApplyInverse(const Epetra_MultiVector& X, const Epetra_SerialDenseMatrix& S,
    Epetra_MultiVector& Y, Epetra_SerialDenseMatrix& T) const;

protected:
  //! original operator
  Teuchos::RCP<const Epetra_Operator> A_;

  //! borders
  Teuchos::RCP<const Epetra_MultiVector> V_, W_;

  Teuchos::RCP<const Epetra_SerialDenseMatrix> C_;

  bool isPreconditioner_;

private:
  //! label
  std::string label_;

  };

  } // namespace HYMLS

namespace Belos
  {
template<>
class OperatorTraits<double, HYMLS::BorderedVector, HYMLS::BorderedOperator>
  {
public:
  static void
  Apply (HYMLS::BorderedOperator const &Op, HYMLS::BorderedVector const &x,
    HYMLS::BorderedVector &y, int trans = 0);

  static bool
  HasApplyTranspose (const HYMLS::BorderedOperator &Op) { return false; }
  };

  } // namespace Belos

#endif
