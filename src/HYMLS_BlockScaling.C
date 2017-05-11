#include "HYMLS_config.h"
#include "HYMLS_Tools.H"
#include "HYMLS_Macros.H"
#include "HYMLS_BlockScaling.H"
#include "BelosConfigDefs.hpp"
#include "NOX_Epetra_Scaling.H"

#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_LinearProblem.h"


namespace HYMLS {

  //!Constructor
  BlockScaling::BlockScaling(bool diagOnly)
    {
    diagOnly_=diagOnly;

    Sl11_=1.0;
    Sl12_=0.0;
    Sl21_=0.0;
    Sl22_=1.0;
    Sr11_=1.0;
    Sr12_=0.0;
    Sr21_=0.0;
    Sr22_=1.0;
    factor_=1.0;
    
    iSl11_=1.0;
    iSl12_=0.0;
    iSl21_=0.0;
    iSl22_=1.0;
    iSr11_=1.0;
    iSr12_=0.0;
    iSr21_=0.0;
    iSr22_=1.0;
    }
    
 //!Destuctor
  BlockScaling::~BlockScaling()
    {
    }

 //! Computes Row Sum scaling diagonal vectors.  Only needs to be called if a row or column sum scaling has been requested.
  void BlockScaling::computeScaling(const Epetra_LinearProblem& problem)
    {
    }

  //! set the coefficients for the left scaling [Sl11 Sl12; Sl21 Sl22] and right scaling
  //! [Sr11 Sr12; Sr21 Sr22]. Formally they are placed on the block 2x2 diagonal of matrices
  //! Sl and Sr, and the linear system is scaled as Sl*A*Sr (Sr\x) = Sl*b
  void BlockScaling::setCoefficients(double Sl11, double Sl12, double Sl21, double Sl22,
                       double Sr11, double Sr12, double Sr21, double Sr22, double factor)
    {
    Sl11_=Sl11; Sl12_=Sl12;
    Sl21_=Sl21; Sl22_=Sl22;
    
    Sr11_=Sr11; Sr12_=Sr12;
    Sr21_=Sr21; Sr22_=Sr22;
    
    factor_=factor;
    
    // compute inverse of Sl and Sr
    double detSl, detSr;
    detSl=Sl11_*Sl22_-Sl12_*Sl21_;
    iSl11_=Sl22_/detSl; iSl12_=-Sl12_/detSl;
    iSl21_=-Sl21_/detSl; iSl22_=Sl11_/detSl;
     
    detSr=Sr11_*Sr22_-Sr12_*Sr21_;
    iSr11_=Sr22_/detSr; iSr12_=-Sr12_/detSr;
    iSr21_=-Sr21_/detSr; iSr22_=Sr11_/detSr;
    }
    
  //! Scales the linear system.
  void BlockScaling::scaleLinearSystem(Epetra_LinearProblem& problem) 
    {
    if (diagOnly_==false)
      {
      HYMLS::Tools::Error("general 2x2 scaling is not implemented,\n"
                          "currently it only works for 'diagOnly==true'.",
                          __FILE__,__LINE__);
      }
    //Teuchos::RCP<Epetra_RowMatrix> A = Teuchos::rcp(problem.GetMatrix());
    Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(dynamic_cast<Epetra_CrsMatrix *>(problem.GetMatrix()),false); 
    CHECK_TRUE(A!=Teuchos::null);
     
    Teuchos::RCP<Epetra_MultiVector> rhs = Teuchos::rcp(problem.GetRHS(),false);
    Teuchos::RCP<Epetra_MultiVector> sol = Teuchos::rcp(problem.GetLHS(),false);

    CHECK_TRUE(rhs!=Teuchos::null);
  
    // again, loop over all elements with stride 2, apply left scaling to rhs
    // and sol, and left and right scaling to the matrix
    for (int i=0; i< A->NumMyRows(); i+=2)
      {
      //get matrixblock
      int row1=A->GRID(i);
      int row2=A->GRID(i+1);
      int len1, len2;
      double *values1, *values2;
      int *cols1, *cols2;
      CHECK_ZERO(A->ExtractMyRowView(i,len1,values1,cols1));
      CHECK_ZERO(A->ExtractMyRowView(i+1,len2,values2,cols2));
      CHECK_ZERO(len1-len2);

      double *a11,*a12,*a21,*a22;
      double t11,t12,t21,t22;
      for (int j=0; j<len1; j++)
        {
        //get global column index
        int col1=A->GCID(cols1[j]);
        int col2=A->GCID(cols2[j]);
        values2[j]=values2[j]*factor_;
        if (col1==row1)
          {
          a11=&values1[j];
          }
        if (col1==row1+1)
          {
          a12=&values1[j];
          }
        if (col2==row2-1)
          {
          a21=&values2[j];
          *a21=(*a21)/factor_;
          }
        if (col2==row2)
          {
          a22=&values2[j];
          *a22=(*a22)/factor_;
          }
        }
      //perform Sl*A*Sr
      //1. T=A*Sr
      t11=(*a11)*Sr11_+(*a12)*Sr21_;
      t12=(*a11)*Sr12_+(*a12)*Sr22_;
      t21=(*a21)*Sr11_+(*a22)*Sr21_;
      t22=(*a21)*Sr12_+(*a22)*Sr22_;
      //2. A=Sl*T
      *a11=Sl11_*t11+Sl12_*t21;
      *a12=Sl11_*t12+Sl12_*t22;
      *a21=Sl21_*t11+Sl22_*t21;
      *a22=Sl21_*t12+Sl22_*t22;
      }

    //left scale the rhs by rhs=Sl*rhs
    this->apply(*sol,Sl11_,Sl12_,Sl21_,Sl22_);

    // scale the current solution, sol=inv(Sr)*sol
    this->apply(*sol,iSr11_,iSr12_,iSr21_,iSr22_);
    }

  //! Remove the scaling from the linear system.
  void BlockScaling::unscaleLinearSystem(Epetra_LinearProblem& problem) 
    {
    if (diagOnly_==false)
      {
      HYMLS::Tools::Error("general 2x2 scaling is not implemented,\n"
                          "currently it only works for 'diagOnly==true'.",
                          __FILE__,__LINE__);
      }
    Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(dynamic_cast<Epetra_CrsMatrix *>(problem.GetMatrix()),false);
    CHECK_TRUE (A!=Teuchos::null);
    
    Teuchos::RCP<Epetra_MultiVector> rhs = Teuchos::rcp(problem.GetRHS(),false);
    Teuchos::RCP<Epetra_MultiVector> sol = Teuchos::rcp(problem.GetLHS(),false);
    // again, loop over all elements with stride 2, remove left scaling to rhs
    // and sol, and left and right scaling to the matrix
    for (int i=0; i< A->NumMyRows(); i+=2)
      {
      //get matrixblock
      int row1=A->GRID(i);
      int row2=A->GRID(i+1);
      int len1, len2;
      double *values1, *values2;
      int *cols1, *cols2;
      CHECK_ZERO(A->ExtractMyRowView(i,len1,values1,cols1));
      CHECK_ZERO(A->ExtractMyRowView(i+1,len2,values2,cols2));
      CHECK_ZERO(len1-len2);

      double *a11,*a12,*a21,*a22;
      double t11,t12,t21,t22;
      for (int j=0; j<len1; j++)
        {
        //get global column index
        int col1=A->GCID(cols1[j]);
        int col2=A->GCID(cols2[j]);

        if (col1==row1)
          {
          a11=&values1[j];
          }
        if (col1==row1+1)
          {
          a12=&values1[j];
          }
        if (col2==row2-1)
          {
          a21=&values2[j];
          }
        else
          {
          if (col2==row2)
            {
            a22=&values2[j];
            }
          else 
            { 
            values2[j]=values2[j]/factor_;
            }
          }
        }
      //Now the matrix is A=Sl*A*Sr, to scale back first right multiply by inv(Sr)
      //1. T=A*inv(Sr)
      t11=(*a11)*iSr11_+(*a12)*iSr21_;
      t12=(*a11)*iSr12_+(*a12)*iSr22_;
      t21=(*a21)*iSr11_+(*a22)*iSr21_;
      t22=(*a21)*iSr12_+(*a22)*iSr22_;
      //2. then left multiply by inv(Sl)
      *a11=iSl11_*t11+iSl12_*t21;
      *a12=iSl11_*t12+iSl12_*t22;
      *a21=iSl21_*t11+iSl22_*t21;
      *a22=iSl21_*t12+iSl22_*t22;
    }

    // unscale the rhs by rhs=inv(Sl)*rhs
    this->apply(*rhs,iSl11_,iSl12_,iSl21_,iSl22_);

    // unscale  sol=Sr*sol
    this->apply(*sol,Sr11_,Sr12_,Sr21_,Sr22_);
    }







  // private helper function:
  // apply x <- S*x, S=kron(I,[s11 s12; s21 s22])
  int BlockScaling::apply(Epetra_MultiVector& x, 
            double s11, double s12,
            double s21, double s22)
    {
    for (int j=0; j< x.NumVectors(); j++)
    {
    for (int i=0; i< x.MyLength(); i+=2)
      {
      double tmp = x[j][i];
      x[j][i]=s11*x[j][i]+s12*x[j][i+1];
      x[j][i+1]=s21*tmp+s22*x[j][i+1];
      }
    }
    }

  //! apply left scaling Sl to a (block)vector (in place, x<-Sl*x)
  int BlockScaling::applyLeftScaling(Epetra_MultiVector& x)
    {
    this->apply(x,Sl11_,Sl12_,Sl21_,Sl22_);
    }

  //! apply right scaling Sr to a (block)vector (in place, x<-Sr*x)
  int BlockScaling::applyRightScaling(Epetra_MultiVector& x)
    {
    this->apply(x,Sr11_,Sr12_,Sr21_,Sr22_);
    }

}
