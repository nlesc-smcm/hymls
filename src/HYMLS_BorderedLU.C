#include "HYMLS_BorderedLU.H"
#include "HYMLS_Tools.H"
#include "Epetra_MultiVector.h"
#include "Epetra_Map.h"
#include "Epetra_BlockMap.h"
#include "Epetra_SerialDenseMatrix.h"

#include "HYMLS_DenseUtils.H"
#include "HYMLS_MatrixUtils.H"

namespace HYMLS
  {

  //! constructor from an operator K (and optionally a border V,W,C)
  BorderedLU::BorderedLU(Teuchos::RCP<const Epetra_Operator> A, 
             Teuchos::RCP<const Epetra_MultiVector> V,
             Teuchos::RCP<const Epetra_MultiVector> W,
             Teuchos::RCP<const Epetra_SerialDenseMatrix> C)
  : A_(A),useTranspose_(false),label_("BorderedLU(A="+std::string(A->Label())+")")
  {
  HYMLS_PROF2(label_,"Constructor");
  CHECK_ZERO(this->setBorder(V,W,C));
  }

HYMLS::BorderedLU::~BorderedLU()
  {
  HYMLS_PROF3(label_,"Destructor");
  }

  int BorderedLU::SetUseTranspose(bool UseTranspose)
    {
    useTranspose_ = UseTranspose;
    // we don't implement this at the moment
    return -1;
    }

    int BorderedLU::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
      {
      HYMLS_PROF3(label_,"Apply (1)");
      return A_->Apply(X,Y);
      }
      
    int BorderedLU::ApplyInverse(const Epetra_MultiVector& Y, Epetra_MultiVector& X) const
      {
      HYMLS_PROF3(label_, "ApplyInverse (1)");
      int m=V_->NumVectors();
      int k=X.NumVectors();
      Epetra_SerialDenseMatrix S(m,k);
      Epetra_SerialDenseMatrix T(m,k);
      int ierr = this->ApplyInverse(Y,T,X,S);
      return ierr;
      }

    int BorderedLU::setBorder(Teuchos::RCP<const Epetra_MultiVector> V,
                  Teuchos::RCP<const Epetra_MultiVector> W,
                  Teuchos::RCP<const Epetra_SerialDenseMatrix> C)
  {
  HYMLS_PROF2(label_,"setBorder");
  V_=V; W_=W; C_=C;
  if (V==Teuchos::null) Tools::Error("V is null",__FILE__,__LINE__);
  if (W==Teuchos::null) W_=V;
  if (C==Teuchos::null) 
    {
    C_=Teuchos::rcp(new Epetra_SerialDenseMatrix(V_->NumVectors(), V_->NumVectors()));
    }
  return this->Compute();
  }
  
    // compute [Y;T]=[A V; W' C]*[X;S]
    int BorderedLU::Apply(const Epetra_MultiVector& X, const Epetra_SerialDenseMatrix& S,
                    Epetra_MultiVector& Y,       Epetra_SerialDenseMatrix& T) const
  {
  HYMLS_PROF2(label_,"Apply");
  CHECK_ZERO(A_->Apply(X,Y));
  Teuchos::RCP<const Epetra_MultiVector> s = DenseUtils::CreateView(S);
  CHECK_ZERO(Y.Multiply('N','N',1.0,*V_,*s,1.0));
  CHECK_ZERO(DenseUtils::MatMul(*W_,X,T));
  CHECK_ZERO(T.Multiply('N','N',1.0,*C_,S,1.0));
  return 0;
  }                    

    //! compute [X S]' = [K V;W' C]\[Y T]'
    int BorderedLU::ApplyInverse(const Epetra_MultiVector& Y, const Epetra_SerialDenseMatrix& T,
                           Epetra_MultiVector& X,       Epetra_SerialDenseMatrix& S) const
  {
  HYMLS_PROF2(label_,"ApplyInverse");
  CHECK_ZERO(A_->ApplyInverse(Y,X));
  if (V_==Teuchos::null)
    {
    HYMLS_DEBUG("no border set");
    return 1; // border not set
    }
  int m = V_->NumVectors();
  HYMLS_DEBVAR(m);
  int k = Y.NumVectors();
  HYMLS_DEBVAR(k);
  // W'x
  Epetra_SerialDenseMatrix B(m, k);
  CHECK_ZERO(DenseUtils::MatMul(*W_,X,B));
  B.Scale(-1.0);
  B+=T;
  // s = (C-W'S\V) \ (T-W'y)
  HYMLS_DEBUG("solve tiny system");
  CHECK_ZERO(SVD_.SetVectors(S,B));
//  CHECK_ZERO(SVD_.Solve());
int ierr=SVD_.Solve();
// ierr=1 is a warning that the matrix should be equilibrated,
// but that is based on our possibly singular matrix
if (ierr!=0 && ierr!=1) return ierr;
  
  Teuchos::RCP<Epetra_MultiVector> s = DenseUtils::CreateView(S);
  
  Epetra_MultiVector Qs(Q_->Map(),k);
  CHECK_ZERO(Qs.Multiply('N','N',1.0,*Q_,*s,0.0));

  CHECK_ZERO(X.Update(-1.0,Qs,1.0));        
  return 0;
  }

int BorderedLU::Compute()
  {
  HYMLS_PROF2(label_,"Compute");
  if (V_==Teuchos::null || W_==Teuchos::null || C_==Teuchos::null)
    {
    Tools::Error("border not set in BorderedLU::Compute",__FILE__,__LINE__);
    }
  if (!(OperatorRangeMap().SameAs(V_->Map())&&V_->Map().SameAs(W_->Map())))
    {
    Tools::Error("incompatible maps found",__FILE__,__LINE__);
    }
  int m = V_->NumVectors();
  if (W_->NumVectors()!=m)
    {     
    Tools::Error("bordering: V and W must have same number of columns",
        __FILE__,__LINE__); 
    }
  if ((C_->N()!=C_->M())||(C_->N()!=m))
    {
    Tools::Error("bordering: C block must be square and compatible with V and W",
        __FILE__,__LINE__);
    }
    
  // compute Q =A\borderV (W_A in Fred's projections document)
  Q_ = Teuchos::rcp(new Epetra_MultiVector(V_->Map(),m));
  CHECK_ZERO(A_->ApplyInverse(*V_,*Q_));
  
#ifdef HYMLS_STORE_MATRICES
  HYMLS::MatrixUtils::Dump(*V_,"BorderedLU_V.txt");
  HYMLS::MatrixUtils::Dump(*Q_,"BorderedLU_Q.txt");
#endif  

  //Schur-complement
  S_ = Teuchos::rcp(new Epetra_SerialDenseMatrix(m,m));
  CHECK_ZERO(DenseUtils::MatMul(*W_,*Q_,*S_));
  CHECK_ZERO(S_->Scale(-1.0));
  *S_ += *C_;

HYMLS_DEBVAR(*S_);

  // factor it using LAPACK
  SVD_.SetMatrix(*S_);
  CHECK_ZERO(SVD_.Invert());
  return 0;
  }

  }//namespace
