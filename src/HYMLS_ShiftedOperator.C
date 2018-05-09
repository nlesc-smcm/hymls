#include "HYMLS_ShiftedOperator.H"

#include "Epetra_MultiVector.h"

namespace HYMLS
  {
int ShiftedOperator::SetUseTranspose(bool UseTranspose)
  {
  useTranspose_ = UseTranspose;
  return 0;
  }

int ShiftedOperator::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
  {
  bool trans;
  if (shiftB_ != 0.0)
    {
    if (B_ == Teuchos::null)
      {
      Y=X;
      }
    else
      {
      Teuchos::RCP<Epetra_Operator> B =
        Teuchos::rcp_const_cast<Epetra_Operator>(B_);
      trans = B->UseTranspose();
      B->SetUseTranspose(useTranspose_);
      B->Apply(X,Y);
      B->SetUseTranspose(trans);
      }
    }
  else
    {
    Y.PutScalar(0.0);
    }
  Teuchos::RCP<Epetra_Operator> A =
    Teuchos::rcp_const_cast<Epetra_Operator>(A_);
  Epetra_MultiVector AX(X.Map(), X.NumVectors());
  trans = A->UseTranspose();
  A->SetUseTranspose(useTranspose_);
  A->Apply(X,AX);
  A->SetUseTranspose(trans);
  Y.Update(shiftA_, AX, shiftB_);
  return 0;
  }

  }//namespace HYMLS
