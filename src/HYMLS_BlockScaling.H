#ifndef HYMLS_BLOCKSCALING_H
#define HYMLS_BLOCKSCALING_H

#include "HYMLS_config.h"

#include "BelosConfigDefs.hpp"
#include "NOX_Epetra_Scaling.H"


class Epetra_MultiVector;
class Epetra_Operator;


namespace HYMLS {


//! class for block 2x2 scaling of a linear system

//!                                                             
//! this class extends the NOX diagonal scaling facility to     
//! a block 2x2 scaling. After constructing the object, the     
//! user has to pass in the scaling coefficients via the        
//! 'setCoefficients' function. The 'computeScaling' does       
//! *not* compute the coefficients from the matrix!!!           
//!                                                             
//! If the optional 'diagOnly' parameter to the constructor     
//! is true, the scaling is applied only to the 2x2 diagonal    
//! block of the matrix A. This may be sufficient if there      
//! are only constant diagonals in the matrix and Sl*Sr=I.      
//!                                                             
//! The scaled linear system is defined in MATLAB syntax as     
//!                                                             
//! Sl * A * Sr (Sr\x) = Sl*b, with                             
//!                                                             
//! Sl = kron(I,[Sl11 Sl12; Sl21 Sl22])                         
//! Sr = kron(I,[Sr11 Sr12; Sr21 Sr22])                         
//!                                                             
class BlockScaling : public NOX::Epetra::Scaling
{
public:

  //!Constructor
  BlockScaling(bool diagOnly=false);
   //!Destuctor
  ~BlockScaling();


  //! Computes Row Sum scaling diagonal vectors.  Only needs to be called if a row or column sum scaling has been requested.
  void computeScaling(const Epetra_LinearProblem& problem);

  //get the scaling coeefcients from the model
  void setCoefficients(double Sl11, double Sl12, double Sl21, double Sl22,
                       double Sr11, double Sr12, double Sr21, double Sr22, double factor);
  //! Scales the linear system.
  void scaleLinearSystem(Epetra_LinearProblem& problem);

  //! Remove the scaling from the linear system.
  void unscaleLinearSystem(Epetra_LinearProblem& problem);

  //! apply left and right scaling Sl to a matrix (in place, A<-Sl*A*Sr)
  int applyScaling(Epetra_RowMatrix& A) const;

  //! undo left and right scaling Sl of a matrix (in place, A<-Sl\A/Sr)
  int applyInverseScaling(Epetra_RowMatrix& x) const;

  //! apply left scaling Sl to a (block)vector (in place, x<-Sl*x)
  int applyLeftScaling(Epetra_MultiVector& x) const;

  //! apply right scaling Sr to a (block)vector (in place, x<-Sr*x)
  int applyRightScaling(Epetra_MultiVector& x) const;

  //! apply inverse of left scaling Sl to a (block)vector (in place, x<-Sl\x)
  int applyInverseLeftScaling(Epetra_MultiVector& x) const;

  //! apply inverse of right scaling Sr to a (block)vector (in place, x<-Sr\x)
  int applyInverseRightScaling(Epetra_MultiVector& x) const;


protected: 
  //! sacling one factor to the second variable 
  double factor_; 
  //! 2x2 right scaling Sr
  double Sr11_, Sr12_, Sr21_, Sr22_;
  //! inverse of Sr
  double iSr11_, iSr12_, iSr21_, iSr22_;


  
  //! 2x2 left scaling Sl=inv(Sr)*some factor
  double Sl11_, Sl12_, Sl21_, Sl22_;
  //! inverse of Sl
  double iSl11_, iSl12_, iSl21_, iSl22_;

  //! indicates wether to scale all of A (if false) or only the block diagonal part.
  bool diagOnly_;
  
  private:
  
  //! apply x <- S*x, S=kron(I,[s11 s12; s21 s22])
  int apply(Epetra_MultiVector& x, 
            double s11, double s12,
            double s21, double s22) const;

  //! internal helper function to implement scaling and unscaling of a matrix in one place
  int apply(Epetra_RowMatrix& A, 
            double sL11, double sL12,
            double sL21, double sL22,
            double sR11, double sR12,
            double sR21, double sR22,
            double factor) const;

};




}
#endif
