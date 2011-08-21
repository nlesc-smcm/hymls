#ifndef HYMLS_BORDERED_SOLVER_H
#define HYMLS_BORDERED_SOLVER_H

class Epetra_MultiVector;
class Epetra_SerialDenseMatrix;

namespace HYMLS
  {

  //!                                                             
  //! virtual base class to implement an operator of the form     
  //!                                                             
  //! |K  V|                                                      
  //! |W' C|.                                                     
  //!                                                             
  //! Once a border has been added, the standard ApplyInverse()   
  //! function will solve                                         
  //!                                                             
  //! |K  V| |x|   |b|                                            
  //! |W' C| |s| = |0|.                                           
  //!                                                             
  //! (if Epetra_Operator is implemented, which is not mandatory).
  //!                                                             
  //! An additional function ApplyInverse() with four arguments   
  //! can be used to solve                                        
  //!                                                             
  //! |K  V| |x|   |b|                                            
  //! |W' C| |s| = |c|.                                           
  //!                                                             
  class BorderedSolver
    {
    public:
    
    //! set the operators V and W. Currently we only allow one border 
    //! to be set (one could think of a more general interface that   
    //! allows adding several borders, i.e. for doing pseudo-arclength
    //! continuation with a projected preconditioner).                
    virtual int SetBorder(Teuchos::RCP<const Epetra_MultiVector> V,
                          Teuchos::RCP<const Epetra_MultiVector> W,
                          Teuchos::RCP<const Epetra_SerialDenseMatrix> C) = 0;

    //! compute [Y T]' = [K V;W' C]*[X S]'
    virtual int Apply(const Epetra_MultiVector& X, const Epetra_SerialDenseMatrix& S,
                    Epetra_MultiVector& Y,       Epetra_SerialDenseMatrix& T) const = 0;

    //! compute [X S]' = [K V;W' C]\[Y T]'
    virtual int ApplyInverse(const Epetra_MultiVector& Y, const Epetra_SerialDenseMatrix& T,
                           Epetra_MultiVector& X,       Epetra_SerialDenseMatrix& S) const = 0;
    
    };
  
  }

#endif
