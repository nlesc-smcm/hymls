#ifndef HYMLS_BORDERED_LU_H
#define HYMLS_BORDERED_LU_H

#include "HYMLS_config.h"

#include "Teuchos_RCP.hpp"
#include "HYMLS_BorderedSolver.H"
#include "Epetra_Operator.h"
#include "Epetra_SerialDenseSVD.h"

class Epetra_SerialDenseMatrix;
class Epetra_MultiVector;

namespace HYMLS
  {

  //! given an operator A that implements ApplyInverse(), this    
  //! class implements an LU-decomposition                        
  //!                                                             
  //! |A  V|   | A,     0    | |I A\V|                            
  //! |W' C| = | W', C-W'A\V | |0  I |                            
  //!                                                             
  //! The class implements an Epetra_Operator acting on the same  
  //! space as K and a HYMLS::BorderedSolver for setting the V,W  
  //! and C blocks and for solving the complete system.           
  class BorderedLU: public BorderedSolver, public Epetra_Operator
    {
    public:

  //! constructor from an operator K (and optionally a border V,W,C)
  BorderedLU(Teuchos::RCP<const Epetra_Operator> K, 
             Teuchos::RCP<const Epetra_MultiVector> V = Teuchos::null,
             Teuchos::RCP<const Epetra_MultiVector> W = Teuchos::null,
             Teuchos::RCP<const Epetra_SerialDenseMatrix> C = Teuchos::null
             );

  //! destructor
  ~BorderedLU();

//! \name Epetra_Operator interface

//@{
    //!
    int SetUseTranspose(bool UseTranspose);
    //! just applies X=AY
    int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;
    //! solves [A V; W' C][X;0]=[Y;0] if a a border is present, and AX=Y otherwise
    int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;
    //!
    double NormInf() const {return -1.0;}
    //!
    const char * Label() const {return label_.c_str();}
    //!
    bool UseTranspose() const {return useTranspose_;}
    //!
    bool HasNormInf() const {return false;}
    //!
    const Epetra_Comm & Comm() const {return A_->Comm();}
    //! returns domain map of A (map of X in A*X)
    const Epetra_Map & OperatorDomainMap() const {return A_->OperatorDomainMap();}
    //! returns range map of A (map of Y in Y'A)
    const Epetra_Map & OperatorRangeMap() const {return A_->OperatorRangeMap();}

//@}

//! \name BorderedOperator interface

//@{
    
    //! set the operators V and W. Currently we only allow one border 
    //! to be set (one could think of a more general interface that   
    //! allows adding several borders, i.e. for doing pseudo-arclength
    //! continuation with a projected preconditioner).                
    int setBorder(Teuchos::RCP<const Epetra_MultiVector> V,
                  Teuchos::RCP<const Epetra_MultiVector> W,
                  Teuchos::RCP<const Epetra_SerialDenseMatrix> C);

    //! compute [Y T]' = [K V;W' C]*[X S]'
    int Apply(const Epetra_MultiVector& X, const Epetra_SerialDenseMatrix& S,
                    Epetra_MultiVector& Y,       Epetra_SerialDenseMatrix& T) const;

    //! compute [X S]' = [K V;W' C]\[Y T]'
    int ApplyInverse(const Epetra_MultiVector& Y, const Epetra_SerialDenseMatrix& T,
                           Epetra_MultiVector& X,       Epetra_SerialDenseMatrix& S) const;
//@}

  protected:
    
    //! operator in the upper left corner (11-block)
    Teuchos::RCP<const Epetra_Operator> A_;
    
    //! border: V and W blocks (may point to the same object)
    Teuchos::RCP<const Epetra_MultiVector> V_,W_;
    
    //! C (22)block
    Teuchos::RCP<const Epetra_SerialDenseMatrix> C_;
    
    //! block A \ V
    Teuchos::RCP<Epetra_MultiVector> Q_;
    
    //! Schur-complement (LU-factored in place)
    Teuchos::RCP<Epetra_SerialDenseMatrix> S_;
    
    //! solver for the Schur-complement (emtpy lapack interface)
    mutable Epetra_SerialDenseSVD SVD_;
    
    //! use transposed operator?
    bool useTranspose_;
    
    //! object label
    std::string label_;
  
  protected:
  
    //! setup the LU decomposition
    int Compute();
    
    };
  }//namespace

#endif
