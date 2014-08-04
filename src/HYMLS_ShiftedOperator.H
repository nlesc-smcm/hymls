#ifndef HYMLS_SHIFTED_OPERATOR_H
#define HYMLS_SHIFTED_OPERATOR_H

#include "Teuchos_RCP.hpp"
#include "Epetra_Operator.h"
#include "HYMLS_Tools.H"

namespace HYMLS 
  {

  //! given operators A, B and scalars shiftA, shiftB, implements the
  //! action Y=(shiftA*A+shiftB*B)X
  class ShiftedOperator : public Epetra_Operator 
    {
    
    public:
    
    //!constructor
    ShiftedOperator(Teuchos::RCP<const Epetra_Operator> A, 
                      Teuchos::RCP<const Epetra_Operator> B,
                      double shiftA, double shiftB)
        : A_(A), B_(B), shiftA_(shiftA), shiftB_(shiftB), useTranspose_(false) {}

   //! @name Destructor
  //@{ 
    //! Destructor
   ~ShiftedOperator() {};
  //@}
  
  //! @name Atribute set methods
  //@{ 

    //! If set true, transpose of this operator will be applied.
    /*! This flag allows the transpose of the given operator to be used implicitly.  Setting this flag
        affects only the Apply() and ApplyInverse() methods.  If the implementation of this interface 
        does not support transpose use, this method should return a value of -1.
      
    \param In
           UseTranspose -If true, multiply by the transpose of operator, otherwise just use operator.

    \return Integer error code, set to 0 if successful.  Set to -1 if this implementation 
does not support transpose.
  */
    int SetUseTranspose(bool UseTranspose)
        {
        // not implemented
        if (UseTranspose)
          {
          Tools::Error("transpose not implemented",__FILE__,__LINE__);
          }
/*
        useTranspose_=UseTranspose;
        labelT_= useTranspose_? "^T":"";
*/
        return 0;
        }
  //@}
  
  //! @name Mathematical functions
  //@{ 

    //! Returns the result of a Epetra_Operator applied to a Epetra_MultiVector X in Y.
    /*! 
    \param In
           X - A Epetra_MultiVector of dimension NumVectors to multiply with matrix.
    \param Out
           Y -A Epetra_MultiVector of dimension NumVectors containing result.

    \return Integer error code, set to 0 if successful.
  */
  int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
    {
    if (shiftB_!=0.0)
      {
      if (B_==Teuchos::null)
        {
        Y=X;
        }
      else
        {
        //bool trans=B_->UseTranspose();
        //B_->SetUseTranspose(useTranspose_);
        B_->Apply(X,Y);
        //B_->SetUseTranspose(trans);
        }
      }
    else
      {
        Y.PutScalar(0.0);
      }
    Epetra_MultiVector AX(X.Map(),X.NumVectors());
    //trans=A_->UseTranspose();
    //A_->SetUseTranspose(useTranspose);
    A_->Apply(X,AX);
    //A_->SetUseTranspose(trans);
    Y.Update(shiftA_,AX,shiftB_);
    return 0;
    }

    //! Returns the result of a Epetra_Operator inverse applied to an Epetra_MultiVector X in Y.
    /*! 
    \param In
           X - A Epetra_MultiVector of dimension NumVectors to solve for.
    \param Out
           Y -A Epetra_MultiVector of dimension NumVectors containing result.

    \return Integer error code, set to 0 if successful.

    \warning In order to work with AztecOO, any implementation of this method must 
              support the case where X and Y are the same object.
  */
   int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
     {
     return -1;
     }

    //! Returns the infinity norm of the global matrix.
    /* Returns the quantity \f$ \| A \|_\infty\f$ such that
       \f[\| A \|_\infty = \max_{1\lei\lem} \sum_{j=1}^n |a_{ij}| \f].

       \warning This method must not be called unless HasNormInf() returns true.
    */ 
   double NormInf() const {return -1.0;}
  //@}
  
  //! @name Atribute access functions
  //@{ 

    //! Returns a character string describing the operator
   const char * Label() const 
     {
//     return ("["+labelV_+"^T"+labelA_+labelV_+"]"+labelT_).c_str();
     return "shiftA*A+shiftB*B";
     }

    //! Returns the current UseTranspose setting.
   bool UseTranspose() const {return useTranspose_;}

    //! Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
   bool HasNormInf() const {return false;}

    //! Returns a pointer to the Epetra_Comm communicator associated with this operator.
   const Epetra_Comm & Comm() const 
      {
      return A_->Comm();
      }

    //! Returns the Epetra_Map object associated with the domain of this operator.
   const Epetra_Map & OperatorDomainMap() const 
      {
      return A_->OperatorDomainMap();
      }

    //! Returns the Epetra_Map object associated with the range of this operator.
   const Epetra_Map & OperatorRangeMap() const 
      {
      return A_->OperatorRangeMap();
      }
  //@}

    
    private:

    //! label
    std::string labelA_,labelT_;

    //! original operators A, B
    Teuchos::RCP<const Epetra_Operator> A_, B_;
    
    //! scalars
    double shiftA_, shiftB_;
    
    //! use transposed operator?
    bool useTranspose_;
    
    };
    
  }//namespace HYMLS


#endif
