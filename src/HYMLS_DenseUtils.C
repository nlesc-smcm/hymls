#include "HYMLS_DenseUtils.H"
#include "HYMLS_MatrixUtils.H"
#include "HYMLS_Tools.H"

#include "Epetra_Operator.h"
#include "Epetra_MultiVector.h"
#include "Epetra_LAPACK.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"

#include "Epetra_SerialComm.h"

#include "AnasaziBlockKrylovSchurSolMgr.hpp"
#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziEpetraAdapter.hpp"

#include "HYMLS_Tools.H"

namespace HYMLS {


int DenseUtils::Eig(const Epetra_SerialDenseMatrix& A,
                     Epetra_SerialDenseVector& lambda_r,
                     Epetra_SerialDenseVector& lambda_i,
                     Epetra_SerialDenseMatrix& right_evecs,
                     Epetra_SerialDenseMatrix& left_evecs)
  {
  HYMLS_PROF2(Label(),"Eig");
  
  Epetra_LAPACK lapack;
  
  int lwork = 8*A.N() + 20*A.N(); // some extra space
  double* work = new double[lwork];
  
  int info;
  
  lapack.GEEV('V','V',A.N(),A.A(),A.LDA(),
        lambda_r.Values(), lambda_i.Values(),
        left_evecs.A(), left_evecs.LDA(),
        right_evecs.A(), right_evecs.LDA(),
        work, lwork, &info);
  if (info!=0)
    {
    Tools::Warning("LAPACK call DGEEV returned non-zero info code "+Teuchos::toString(info),
        __FILE__,__LINE__);
    }
  delete [] work;
    
  return info;
  }                     

int DenseUtils::Eig(const Epetra_SerialDenseMatrix& A,
               const Epetra_SerialDenseMatrix& B,
                     Epetra_SerialDenseVector& alpha_r,
                     Epetra_SerialDenseVector& alpha_i,
                     Epetra_SerialDenseVector& beta,
                     Epetra_SerialDenseMatrix& right_evecs,
                     Epetra_SerialDenseMatrix& left_evecs)
  {
  HYMLS_PROF2(Label(),"Eig");
  
  Epetra_LAPACK lapack;
  
  int lwork = 8*A.N() + 20*A.N(); // some extra space
  double* work = new double[lwork];
  
  int info;
  
  lapack.GGEV('N','V',A.N(),A.A(),A.LDA(),B.A(),B.LDA(),
        alpha_r.Values(), alpha_i.Values(), beta.Values(),
         left_evecs.A(), left_evecs.LDA(),
         right_evecs.A(), right_evecs.LDA(),
         work, lwork, &info);
  if (info!=0)
    {
    Tools::Warning("LAPACK call DGGEV returned non-zero info code "+Teuchos::toString(info),
        __FILE__,__LINE__);
    }

  delete [] work;
    
  return info;
  }

int DenseUtils::MatMul(const Epetra_MultiVector& V, const Epetra_MultiVector& W,
                       Epetra_SerialDenseMatrix& C)
  {
  HYMLS_PROF3(Label(),"MatMul");
  if (!(W.Map().SameAs(V.Map())))
    {
    HYMLS_DEBUG("DenseUtils::MatMul(V,W) failed because the maps are not the same");
    HYMLS_DEBVAR(V.Map());
    HYMLS_DEBVAR(W.Map());
    return -1;
    }
  int m = V.NumVectors();
  int n = W.NumVectors();
  if ((C.N()!=n)||(C.M()!=m))
    {
    C.Reshape(m,n);
    }

  // this object is replicated on all procs because of the LocalMap:
  Epetra_SerialDenseMatrix tmp = C;
  Teuchos::RCP<Epetra_MultiVector> VW = CreateView(tmp);

  CHECK_ZERO(VW->Multiply('T','N',1.0,V,W,0.0));
  CHECK_ZERO(V.Comm().SumAll(tmp.A(),C.A(),m*n));

  return 0;
  }

// given two multivectors V and W, computes V_orth*W and returns the result
// as a new MultiVector Z. V, W and Z should have the same maps and numbers
// of vectors (columns). The product is computed as Z=(I-VV')W.                
int DenseUtils::ApplyOrth(const Epetra_MultiVector& V, const Epetra_MultiVector& W,
                           Epetra_MultiVector& Z, Teuchos::RCP<const Epetra_MultiVector> BV,
                           bool reverse)
  {
  HYMLS_PROF3(Label(),"ApplyOrth");
  if (W.Map().SameAs(V.Map())==false)
    {
    //input args not correctly shaped
    return -1;
    }
  if (!(W.Map().SameAs(Z.Map()) && (W.NumVectors()==Z.NumVectors())))
    {
    //output arg not correctly shaped
    return -2;
    }
  int m=V.NumVectors();
  int k=W.NumVectors();
  Epetra_SerialDenseMatrix C(m,k);
  // this object is replicated on all procs because of the LocalMap:
  Teuchos::RCP<Epetra_MultiVector> VW=CreateView(C);
  //Z=W-VV'W
  if (BV == Teuchos::null) 
    {
    //VW=V'W
    CHECK_ZERO(MatMul(V,W,C));
    Z=W;
    CHECK_ZERO(Z.Multiply('N','N',-1.0,V,*VW,1.0));
    }
  else if (reverse)
    {
    //VV'BW
    CHECK_ZERO(MatMul(*BV,W,C));
    Z=W;
    CHECK_ZERO(Z.Multiply('N','N',-1.0,V,*VW,1.0));
    }
  else
    {
    //VW=V'W
    CHECK_ZERO(MatMul(V,W,C));
    //BVV'W
    CHECK_ZERO(Z.Multiply('N','N',1.0,*BV,*VW,0.0));
    CHECK_ZERO(Z.Update(1.0,W,-1.0));
    }
  return 0;
  }

//! returns orthogonal basis for the columns of A.
int DenseUtils::Orthogonalize(Epetra_SerialDenseMatrix& A)
  {
  HYMLS_PROF2(Label(),"Orthogonalize");
  int n=A.N();
  int m=A.M();
  Epetra_LAPACK lapack;
  int info;
  Epetra_SerialDenseMatrix Tau(n,m);
  Epetra_SerialDenseMatrix Work(n,m);
  
  int lwork=n*m;

  lapack.GEQRF(m, n, A.A(),A.LDA(),Tau.A(), Work.A(), n*m, &info);

  if (info!=0)
    {
    Tools::Warning("LAPACK call DGEQRF returned non-zero info code "+Teuchos::toString(info),
        __FILE__,__LINE__);
    }
  int k = std::min(m,n);
  lapack.ORGQR(m, n, k, A.A(), A.LDA(), Tau.A(), Work.A(), lwork, &info);
  if (info!=0)
    {
    Tools::Warning("LAPACK call DORQR returned non-zero info code "+Teuchos::toString(info),
        __FILE__,__LINE__);
    }
  return 0;
  }

//! create a multivector view of a dense matrix
Teuchos::RCP<Epetra_MultiVector> DenseUtils::CreateView(Epetra_SerialDenseMatrix& A)
  {
  HYMLS_PROF3(Label(),"CreateView");
  int nrows = A.M();
  int ncols = A.N();
  Epetra_SerialComm comm;
  Epetra_LocalMap tinyMap(nrows,0,comm);
  Teuchos::RCP<Epetra_MultiVector> MV = 
        Teuchos::rcp(new Epetra_MultiVector(View,tinyMap,A.A(),A.LDA(),ncols));
  return MV;
  }

//! create a multivector view of a dense matrix
Teuchos::RCP<const Epetra_MultiVector> DenseUtils::CreateView
        (const Epetra_SerialDenseMatrix& A)
  {
  HYMLS_PROF3(Label(),"CreateView");
  int n = A.N();
  Epetra_SerialComm comm;
  Epetra_LocalMap tinyMap(n,0,comm);
  Teuchos::RCP<const Epetra_MultiVector> MV = 
        Teuchos::rcp(new Epetra_MultiVector(View,tinyMap,A.A(),A.LDA(),n));
  return MV;
  }

//! create a dense matrix view of a multivector
Teuchos::RCP<Epetra_SerialDenseMatrix> DenseUtils::CreateView(Epetra_MultiVector& A)
  {
  HYMLS_PROF3(Label(),"CreateView");
  int n = A.NumVectors();
  if (A.DistributedGlobal())
    {
    Tools::Error("Cannot convert this MV to a serial dense matrix!",
                __FILE__,__LINE__);
    }
  Teuchos::RCP<Epetra_SerialDenseMatrix> DM = 
        Teuchos::rcp(new 
        Epetra_SerialDenseMatrix(View,A.Values(),A.MyLength(),n,n));
  return DM;
  }

//! create a dense matrix view of a multivector
Teuchos::RCP<const Epetra_SerialDenseMatrix> DenseUtils::CreateView
        (const Epetra_MultiVector& A)
  {
  HYMLS_PROF3(Label(),"CreateView");
  int n = A.NumVectors();
  if (A.DistributedGlobal())
    {
    Tools::Error("Cannot convert this MV to a serial dense matrix!",
                __FILE__,__LINE__);
    }
  Teuchos::RCP<const Epetra_SerialDenseMatrix> DM = 
        Teuchos::rcp(new 
        Epetra_SerialDenseMatrix(View,A.Values(),A.MyLength(),n,n));
  return DM;
  }


}

