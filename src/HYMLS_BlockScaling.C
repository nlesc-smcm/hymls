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
    Teuchos::RCP<Epetra_RowMatrix> A = Teuchos::rcp(problem.GetMatrix(),false);
    Teuchos::RCP<Epetra_MultiVector> rhs = Teuchos::rcp(problem.GetRHS(),false);
    Teuchos::RCP<Epetra_MultiVector> sol = Teuchos::rcp(problem.GetLHS(),false);

    //left and right scale the matrix
    if (A!=Teuchos::null) CHECK_ZERO(this->applyScaling(*A));

    //left scale the rhs by rhs=Sl*rhs
    if (rhs!=Teuchos::null) CHECK_ZERO(this->applyLeftScaling(*rhs));

    // scale the current solution, sol=inv(Sr)*sol
    if (sol!=Teuchos::null) CHECK_ZERO(this->applyInverseRightScaling(*sol));

    }

  //! Remove the scaling from the linear system.
  void BlockScaling::unscaleLinearSystem(Epetra_LinearProblem& problem) 
    {
    Teuchos::RCP<Epetra_RowMatrix> A = Teuchos::rcp(problem.GetMatrix(),false);
    Teuchos::RCP<Epetra_MultiVector> rhs = Teuchos::rcp(problem.GetRHS(),false);
    Teuchos::RCP<Epetra_MultiVector> sol = Teuchos::rcp(problem.GetLHS(),false);

    //unscale the matrix, A<- iSl * (Sl*A*Sr) * iSr
    if (A!=Teuchos::null) CHECK_ZERO(this->applyInverseScaling(*A));

    //unscale the rhs by rhs=Sl\(Sl*rhs)
    if (rhs!=Teuchos::null) CHECK_ZERO(this->applyInverseLeftScaling(*rhs));

    // scale the current solution, sol=Sr*(Sr\sol)
    if (sol!=Teuchos::null) CHECK_ZERO(this->applyRightScaling(*sol));

    }

  // apply left scaling Sl to a (block)vector (in place, x<-Sl*x)
  int BlockScaling::applyLeftScaling(Epetra_MultiVector& x) const
    {
    return this->apply(x,Sl11_,Sl12_,Sl21_,Sl22_);
    }

  // apply right scaling Sr to a (block)vector (in place, x<-Sr*x)
  int BlockScaling::applyRightScaling(Epetra_MultiVector& x) const
    {
    return this->apply(x,Sr11_,Sr12_,Sr21_,Sr22_);
    }

  // apply inverse of left scaling Sl to a (block)vector (in place, x<-Sl\x)
  int BlockScaling::applyInverseLeftScaling(Epetra_MultiVector& x) const
    {
    return this->apply(x,iSl11_,iSl12_,iSl21_,iSl22_);
    }

  // apply inverse of right scaling Sr to a (block)vector (in place, x<-Sr\x)
  int BlockScaling::applyInverseRightScaling(Epetra_MultiVector& x) const
    {
    return this->apply(x,iSr11_,iSr12_,iSr21_,iSr22_);
    }

  //! apply left and right scaling Sl to a matrix (in place, A<-Sl*A*Sr)
  int BlockScaling::applyScaling(Epetra_RowMatrix& A) const
    {
    return this->apply(A, Sl11_, Sl12_, Sl21_, Sl22_,
                          Sr11_, Sr12_, Sr21_, Sr22_, factor_);
    }

  //! undo left and right scaling Sl of a matrix (in place, A<-Sl\A/Sr)
  int BlockScaling::applyInverseScaling(Epetra_RowMatrix& A) const
    {
    return this->apply(A, iSl11_, iSl12_, iSl21_, iSl22_,
                          iSr11_, iSr12_, iSr21_, iSr22_, 1.0/factor_);
    }


  // private helper function:
  // apply x <- S*x, S=kron(I,[s11 s12; s21 s22])
  int BlockScaling::apply(Epetra_MultiVector& x, 
            double s11, double s12,
            double s21, double s22) const
    {
      if (x.MyLength()%2!=0)
        {
        HYMLS::Tools::out() << "bad vector: "<<x<<std::endl;
        HYMLS::Tools::Error("our 2x2 scaling object assumes consistent interleaved storage of variables in the vector!",
              __FILE__,__LINE__);
        }
      for (int j=0; j< x.NumVectors(); j++)
      {
      for (int i=0; i< x.MyLength(); i+=2)
        {
#ifdef HYMLS_TESTING
        if (x.Map().GID(i+1)!=x.Map().GID(i)+1)
          {
          HYMLS::Tools::out() << "bad entries local row "<<i<<" (GID "<<x.Map().GID(i)<<") and "<<i+1<<" (GID "<<x.Map().GID(i+1)<<")\n";
          HYMLS::Tools::out() << "bad map: "<<x.Map()<<std::endl;
          HYMLS::Tools::out() << "bad vector: "<<x<<std::endl;
          HYMLS::Tools::Error("our 2x2 scaling object assumes consistent interleaved storage of variables in the vector!",
                __FILE__,__LINE__);
          }
#endif
        double tmp = x[j][i];
        x[j][i]=s11*x[j][i]+s12*x[j][i+1];
        x[j][i+1]=s21*tmp+s22*x[j][i+1];
        }
      }
    return 0;
    }
    
    
  // internal helper function to implement scaling and unscaling of a matrix in one place
  int BlockScaling::apply(Epetra_RowMatrix& A_row_ref, 
            double Sl11, double Sl12,
            double Sl21, double Sl22,
            double Sr11, double Sr12,
            double Sr21, double Sr22,
            double factor) const
    {
    // currently only implemented for CrsMatrices
    Teuchos::RCP<Epetra_RowMatrix> A_row=Teuchos::rcpFromRef(A_row_ref);
    Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(A_row);
    if (A==Teuchos::null) HYMLS::Tools::Error("only implemented for Epetra_CrsMatrix objects right now",__FILE__,__LINE__);

    if (diagOnly_==false)
      {
      HYMLS::Tools::Error("general 2x2 scaling is not implemented,\n"
                          "currently it only works for 'diagOnly==true'.",
                          __FILE__,__LINE__);
      }
    
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
      // make sure we're not missing any entries, we assume that
      // both rows have the same number of entries and throw an 
      // error if not.
      CHECK_ZERO(len1-len2);

      double *a11=NULL,*a12=NULL,*a21=NULL,*a22=NULL;
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
            values2[j]=values2[j]*factor;
            }
          }
        }
      //perform Sl*A*Sr
      //1. T=A*Sr
      t11=(*a11)*Sr11+(*a12)*Sr21;
      t12=(*a11)*Sr12+(*a12)*Sr22;
      t21=(*a21)*Sr11+(*a22)*Sr21;
      t22=(*a21)*Sr12+(*a22)*Sr22;
      //2. A=Sl*T
      *a11=Sl11*t11+Sl12*t21;
      *a12=Sl11*t12+Sl12*t22;
      *a21=Sl21*t11+Sl22*t21;
      *a22=Sl21*t12+Sl22*t22;
      }
    return 0;
    }
}
