#include "HYMLS_no_debug.H"

#include "HYMLS_Householder.H"
#include "HYMLS_Tools.H"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"
#include "Epetra_CrsMatrix.h"
#include "EpetraExt_MatrixMatrix.h"
#include "HYMLS_MatrixUtils.H"
#include "Epetra_RowMatrixTransposer.h"
#include "Epetra_MultiVector.h"
#include "Trilinos_version.h"

namespace HYMLS {

 
  // constructor
  Householder::Householder(int lev) : label_("Householder"),
                               myLevel_(lev),
                               Wmat_(Teuchos::null),
                               WTmat_(Teuchos::null),
                               Cmat_(Teuchos::null)
    {
    }
   
   Householder::~Householder() 
     {
     }
   
  //! compute X=Q*X in place
  int Householder::Apply(Epetra_SerialDenseMatrix& X) const
    {
    HYMLS_PROF3(label_, "Apply (1)");
    // X = (2vv'/v'v-I)Y
    // can be written as X = Z-Y, Z= (2/nrmv^2 v'Y)v
    // (we use a vector v which is 1 everywhere except that the first entry is 1+sqrt(n))
    int n=X.M();
    double sqn=sqrt((double)n);
    double v1=1+sqn; // first vector element, all others are 1
    double fac1 = 1.0/(n+sqn); // 2/v'v
    for (int k=0;k<X.N();k++)
      {
      // v'x
      double fac2 = sqn*X[k][0];
      for (int i=0;i<n;i++) 
        {
        fac2+= X[k][i];
        }
      double fac=fac1*fac2;
      X[k][0]=v1*fac-X[k][0];
      for (int i=1;i<n;i++) 
        {
        X[k][i]=fac-X[k][i];
        }
      }
    return 0;
    }

  //! compute X=X*Q' in place
  int Householder::ApplyR(Epetra_SerialDenseMatrix& X) const
    {
    HYMLS_PROF3(label_, "ApplyR (1)");
    int n=X.N();
    double sqn=sqrt((double)n);
    double v1=1+sqn; // first vector element, all others are 1
    double fac1 = 1.0/(n+sqn); // 2/v'v
    for (int k=0;k<X.M();k++)
      {
      // v'x
      double fac2 = sqn*X[0][k];
      for (int i=0;i<n;i++) 
        {
        fac2+= X[i][k];
        }
      double fac=fac1*fac2;
      X[0][k]=v1*fac-X[0][k];
      for (int i=1;i<n;i++) 
        {
        X[i][k]=fac-X[i][k];
        }
      }
    return 0;
    }

  //! compute X=Q*Y*Q' in place
  int Householder::ApplyLR(Epetra_SerialDenseMatrix& Y) const
    {
    HYMLS_PROF3(label_, "ApplyLR (1)");
    // let Y \in R^{m x n}
    // Q = (1-alpha*vv')=Q', alpha = 2/(v'*v)
    // Y = (I-alpha_m*vmvm') Y (1-alpha_n*vnvn')
    //   = Y - alpha_n Avnvn' - alpha_m vmvm'A + alpha_n*alpha_m vmvm'Avnvn'
    int n=Y.N();
    int m=Y.M();
    double sqn=sqrt((double)n);
    double sqm=sqrt((double)m);
    double alpha_n = 1.0/(n+sqn); // 2/v'v
    double alpha_m = 1.0/(m+sqm); // 2/v'v

    // our implementation is very unoptimized right now, we 
    // simply use matrix-vector products and do not make use
    // of the special structure of the vectors or the operations
    Epetra_SerialDenseVector vm(m);
    Epetra_SerialDenseVector vn(n);
    
    vn(0)=1.0+sqn;
    for (int i=1;i<n;i++) vn(i)=1.0;
    vm(0)=1.0+sqm;
    for (int i=1;i<m;i++) vm(i)=1.0;
    
    // compute A*vn, an m-vector
    Epetra_SerialDenseVector Avn(m);
    // for some reason the 'const' attribute is missing on this function
    CHECK_ZERO(Y.Multiply(false,vn,Avn));
    
    // compute vm'*A, an n-vector
    Epetra_SerialDenseVector vmTA(n);
    CHECK_ZERO(Y.Multiply(true,vm,vmTA));
    
    double vTAv = vmTA.Dot(vn);
        
    // perform rank 1 updates
    // (TODO: optimize this by avoiding all the multiplies with 1)
    double factor = alpha_m*alpha_n*vTAv;
    for (int j=0;j<n;j++)
      for (int i=0;i<m;i++)
        {
        // X = Y - alpha_n Avnvn' - alpha_m vmvm'A + factor*vn*vm'
        Y(i,j)=Y(i,j) - alpha_n*Avn(i)*vn(j) - alpha_m*vm(i)*vmTA(j)
                      + factor*vm(i)*vn(j);
        }
    return 0;
    }
    

  int Householder::ApplyInverse(Epetra_SerialDenseMatrix& X) const
    {
    return Apply(X);
    }
    
  //! compute X=Q*X in place
  int Householder::Apply(Epetra_SerialDenseMatrix& X,
                   const Epetra_SerialDenseVector& v) const
    {
    HYMLS_PROF3(label_, "Apply (2)");
    // X = (2vv'/v'v-I)Y
    // can be written as X = Z-Y, Z= (2/nrmv^2 v'Y)v
    const int n=X.M();
#ifdef TESTING
    if (v.Length()!=n)
      {
      return -1;
      }
#endif
    const double nrmv=v.Norm2();
    const double v1=v(0)+nrmv; // first vector element, all others are those in v
    // this is 2/(v'v) with the adjusted v
    const double fac1 = 1.0/(nrmv*v1);
#pragma omp parallel
{
    double *Xrow;
    const double *vvalues = v.Values();
#pragma omp for
    for (int k=0;k<X.N();k++)
      {
      // v'x
      Xrow = X[k];
      double fac2 = nrmv*Xrow[0];
      for (int i=0;i<n;i++)
        {
        fac2+= Xrow[i]*vvalues[i];
        }
      const double fac=fac1*fac2;
      Xrow[0]=v1*fac-Xrow[0];
      for (int i=1;i<n;i++)
        {
        Xrow[i]=vvalues[i]*fac-Xrow[i];
        }
      }
}
    return 0;
    }

  //! compute X=X*Q' in place
  int Householder::ApplyR(Epetra_SerialDenseMatrix& X,
                    const Epetra_SerialDenseVector& v) const
    {
    HYMLS_PROF3(label_, "ApplyR (2)");
    // X = (2vv'/v'v-I)Y
    // can be written as X = Z-X, Z= (2/nrmv^2 v'Y)v
    int n=X.N();
#ifdef TESTING
    if (v.Length()!=n)
      {
      return -1;
      }
#endif
    // to zero out all but the first entry in v, use v
    const double nrmv=v.Norm2();
    const double v1=v(0)+nrmv; // first vector element, all others are those in v
    // this is 2/(v'v) with the adjusted v
    const double fac1 = 1.0/(nrmv*v1);
#pragma omp parallel
{
    const double *vvalues = v.Values();
#pragma omp for
    for (int k=0;k<X.M();k++)
      {
      // v'x
      double fac2 = nrmv*X[0][k];
      for (int i=0;i<n;i++)
        {
        fac2+= X[i][k]*vvalues[i];
        }
      const double fac=fac1*fac2;
      X[0][k] = v1*fac-X[0][k];
      for (int i=1;i<n;i++)
        {
        X[i][k] = vvalues[i]*fac-X[i][k];
        }
      }
}
    return 0;
    }

  int Householder::ApplyInverse(Epetra_SerialDenseMatrix& X,
                           const Epetra_SerialDenseVector& v) const
    {
    return Apply(X,v);
    }

  //! explicitly form the OT as a dense matrix. The dimension is given by
  //! the size of the output matrix.
  int Householder::Construct(Epetra_SerialDenseMatrix& M) const
    {
    HYMLS_PROF3(label_, "Construct (1)");
    int n=M.N();
#ifdef TESTING
    if (M.M()!=n)
      {
      return -1;
      }
#endif

    double sqn=sqrt((double)n);

    double alpha_n = 1.0/(n+sqn); // 2/v'v

    Epetra_SerialDenseVector vn(n);
    
    vn(0)=1.0+sqn;
    for (int i=1;i<n;i++) vn(i)=1.0;

    for (int i=0;i<n;i++)
      {
      for (int j=0;j<n;j++)
        {
        M(i,j)=vn(i)*vn(j)*alpha_n;
        }
      M(i,i)-=1.0;
      }
    return 0;
    }


  // explicitly form the OT as a sparse matrix. We only put in a sparse
  // matrix representation of the vector w=v'/sqrt(v'v), so when applying the operator
  // we have to apply I-2w'w.
  int Householder::Construct(Epetra_CrsMatrix& H, 
            const Epetra_IntSerialDenseVector& inds) const
    {
    HYMLS_PROF3(label_, "Construct (2)");
    int n=inds.Length();
    Epetra_SerialDenseVector vec(n);
    for (int i=0;i<n;i++) vec[i]=1.0;
    return this->Construct(H,inds,vec);
    }

  int Householder::Construct(Epetra_CrsMatrix& H, 
            const Epetra_IntSerialDenseVector& inds,
            const Epetra_SerialDenseVector& vec) const
    {
    HYMLS_PROF3(label_, "Construct (3)");
    // vec is the test vector to be zeroed out by this transform,
    // construct the according v for the Householder reflection: 
    Epetra_SerialDenseVector v = vec;
    int n=vec.Length();
    int row=inds[0];
    double nrm=vec.Norm2();
    v[0]=v[0]+nrm;
    nrm=v.Norm2();
    CHECK_ZERO(v.Scale(1.0/nrm));
          
    if (H.Filled())
      {
      CHECK_ZERO(H.ReplaceGlobalValues(row,n,v.A(),const_cast<int*>(&(inds[0]))));
      }
    else
      {
      CHECK_NONNEG(H.InsertGlobalValues(row,n,v.A(),const_cast<int*>(&(inds[0]))));
      }
    return 0;
    }

  //! apply a sparse matrix representation of a set of transforms from the left
  //! and right to a sparse matrix.
  // H is a matrix representation of the vector w=v'/sqrt(v'v), so 
  // we have to apply (2w'w-I)A(2w'w-I)=A-2w'wA-2Aw'w+4w'wAw'w
  //                                   =A-2Aw'w-2w'w(A-2Aw'w)
  Teuchos::RCP<Epetra_CrsMatrix> Householder::Apply
        (const Epetra_CrsMatrix& T, const Epetra_CrsMatrix& A) const
    {
    HYMLS_PROF2(label_,"H^TAH (1)");
        
    
    if (A.Filled()==false || T.Filled()==false)
      {
      Tools::Error("For at least one of the matrices FillComplete() has not been called!",
        __FILE__,__LINE__);
      }
    
    Wmat_=Teuchos::rcp(&T,false);  
    
#ifdef STORE_MATRICES
    MatrixUtils::Dump(A, "HOUSE_A.txt");
    MatrixUtils::Dump(*Wmat_, "HOUSE_W.txt");
#endif

    // Aw'
    Teuchos::RCP<Epetra_CrsMatrix> AwT = Teuchos::rcp(new
        Epetra_CrsMatrix(Copy,A.RowMap(),A.MaxNumEntries()) );

    DEBUG("compute A*wT...");
#if (1 || TRILINOS_MAJOR_MINOR_VERSION<101000)
#define MATMUL_BUG 1
#endif
#ifndef MATMUL_BUG
    // buggy in older Trilinos versions in parallel, slow in new versions
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(A,false,T,true,*AwT,true));
#else
    Transp_=Teuchos::rcp(new Epetra_RowMatrixTransposer(const_cast<Epetra_CrsMatrix*>(&T)));
    Epetra_CrsMatrix* tmp;
    Transp_->CreateTranspose(false,tmp,const_cast<Epetra_Map*>(&(T.RowMap())));
    WTmat_=Teuchos::rcp(tmp,true);
    CHECK_ZERO(WTmat_->FillComplete());
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(A,false,*WTmat_,false,*AwT,true));
#endif
    // Aw'w
    Teuchos::RCP<Epetra_CrsMatrix> AwTw = Teuchos::rcp(new 
        Epetra_CrsMatrix(Copy,A.RowMap(),AwT->MaxNumEntries()) );

    DEBUG("compute A*wT*w...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(*AwT,false,T,false,*AwTw,false));

#ifdef STORE_MATRICES
    MatrixUtils::Dump(*AwT,"HOUSE_AwT.txt");
    // not filled yet
    //MatrixUtils::Dump(*AwTw,"HOUSE_AwTw.txt");
#endif

    // C=A-2Aw'w
    Cmat_ = AwTw;
    if (SaveMemory()) 
      {
      AwT=Teuchos::null;
      AwTw=Teuchos::null;
      }
    
    DEBUG("compute C=A(2wTw-I)...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Add(A,false,-1.0,*Cmat_,2.0));
    CHECK_ZERO(Cmat_->FillComplete());

#ifdef STORE_MATRICES
    MatrixUtils::Dump(*Cmat_,"HOUSE_C.txt");
#endif

    // wC
    WCmat_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy,A.RowMap(),Cmat_->MaxNumEntries()) );

    DEBUG("compute wC...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(T,false,*Cmat_,false,*WCmat_,true));

    // wTwC
    Teuchos::RCP<Epetra_CrsMatrix> wTwC = Teuchos::rcp(new 
        Epetra_CrsMatrix(Copy,A.RowMap(),WTmat_->MaxNumEntries()) );
  
    DEBUG("compute wTwC...");
#ifndef MATMUL_BUG
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(*Wmat_,true,*WCmat_,false,*wTwC,false));
#else
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(*WTmat_,false,*WCmat_,false,*wTwC,false));
#endif
#ifdef STORE_MATRICES
    MatrixUtils::Dump(*WCmat_,"HOUSE_wC.txt");
    // not filled yet
    //MatrixUtils::Dump(*wTwC,"HOUSE_wTwC.txt");
#endif
    if (SaveMemory())
      {
      WTmat_=Teuchos::null;
      WCmat_=Teuchos::null;
      Wmat_=Teuchos::null;
      Transp_=Teuchos::null;
      }

    DEBUG("compute TAT=2wTwC-C...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Add(*Cmat_,false,-1.0,*wTwC,2.0));
    wTwC->FillComplete();

#ifdef STORE_MATRICES
    // not filled yet
    //MatrixUtils::Dump(*wTwC,"HOUSE_HAH.txt");
#endif

    if (SaveMemory())
      {
      Cmat_=Teuchos::null;
      }

    
  if (!SaveMemory())
    {    
    double my_nnz = Wmat_->NumMyNonzeros() + 
                    Cmat_->NumMyNonzeros() +
                    WCmat_->NumMyNonzeros() +
                    WTmat_->NumMyNonzeros();
    REPORT_SUM_MEM(label_,"intermediate results",my_nnz,my_nnz,&wTwC->Comm());
    }
    DEBUG("done!");
    return wTwC;                
    }

  //! apply a sparse matrix representation of a set of transforms from the left
  //! and right to a sparse matrix. This variant is to be preferred if the 
  //! sparsity pattern of the transformed matrix TAT is already known.
  //  As above, we compute the product as A-2Aw'w-2w'w(A-2Aw'w)
  int Householder::Apply
    (Epetra_CrsMatrix& TAT, const Epetra_CrsMatrix& T, const Epetra_CrsMatrix& A) const
    {
    HYMLS_PROF2(label_,"H^TAH (2)");

    if (SaveMemory())
      {
      return -1; // this variant cannot be called if SaveMemory() is true
                 // because intermediate results are discared.
      }

    Cmat_->PutScalar(0.0);
    WCmat_->PutScalar(0.0);
    TAT.PutScalar(0.0);
    
    if (Wmat_.get()!=&T)
      {
      Tools::Error("version 1 of Apply() must be called at least once!",
        __FILE__,__LINE__);
      }

    
    if (A.Filled()==false)
      {
      Tools::Error("A not filled!",__FILE__,__LINE__);
      }


    if (Wmat_->Filled()==false)
      {
      Tools::Error("W not filled!",__FILE__,__LINE__);
      }


    if (WTmat_->Filled()==false)
      {
      Tools::Error("WT not filled!",__FILE__,__LINE__);
      }
    
    // Aw'
    DEBUG("compute A*wT...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(A,false,*WTmat_,false,TAT));


    // Aw'w
    DEBUG("compute A*wT*w...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(TAT,false,*Wmat_,false,*Cmat_));

    // C=2Aw'w-A
    DEBUG("compute C=A(I-2wTw)...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Add(A,false,1.0,*Cmat_,-2.0));

    // wC
    DEBUG("compute wC...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(*Wmat_,false,*Cmat_,false,*WCmat_));

    // wTwC
    DEBUG("compute wTwC...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(*WTmat_,false,*WCmat_,false,TAT));

    DEBUG("compute TAT=C-2wTwC...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Add(*Cmat_,false,1.0,TAT,-2.0));
    DEBUG("done!");

    
    return 0;
    }

  //! apply a sparse matrix representation of a set of transforms from the left
  //! and right to a sparse matrix. This variant is to be preferred if the 
  //! sparsity pattern of the transformed matrix TAT is already known.
  int Householder::Apply
    (Epetra_MultiVector& Tv, const Epetra_CrsMatrix& T, const Epetra_MultiVector& v) const
    {
    HYMLS_PROF2(label_,"H*v");
    Epetra_MultiVector tmp = v;
    CHECK_ZERO(T.Multiply(false,v,tmp));
    CHECK_ZERO(T.Multiply(true,tmp,Tv));
    CHECK_ZERO(Tv.Update(-1.0,v,2.0));
    
    return 0;
    }

  int Householder::ApplyInverse
    (Epetra_MultiVector& Tv, const Epetra_CrsMatrix& T, const Epetra_MultiVector& v) const
    {
    return this->Apply(Tv,T,v);
    }


  

}
