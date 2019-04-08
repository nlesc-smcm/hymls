#include "HYMLS_Householder.H"

#include "HYMLS_config.h"

#include "HYMLS_Tools.H"
#include "HYMLS_MatrixUtils.H"

#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_RowMatrixTransposer.h"
#include "Epetra_MultiVector.h"
#include "EpetraExt_MatrixMatrix.h"

double sign(double x)
  {
  return (x < 0) ? -1 : (x > 0);
  }

namespace HYMLS {

// constructor
Householder::Householder(int lev)
  :
  label_("Householder"),
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
int Householder::Apply(Epetra_SerialDenseMatrix& X,
  Epetra_SerialDenseVector v) const
  {
  HYMLS_PROF3(label_, "Apply (2)");

  // X = (2vv'/v'v-I)Y
  // can be written as X = Z-Y, Z= (2/nrmv^2 v'Y)v

  const int n = X.M();
#ifdef HYMLS_TESTING
  if (v.Length() != n)
    return -1;
#endif

  // Scale with the first element of the test vector to assure
  // that the first element is always positive
  CHECK_ZERO(v.Scale(sign(v(0))));

  const double nrmv = v.Norm2();
  const double v1 = v(0) + nrmv; // first vector element, all others are those in v

  // this is 2/(v'v) with the adjusted v
  const double fac1 = 1.0 / (nrmv * v1);
  for (int k = 0; k < X.N(); k++)
    {
    // v'x
    double fac2 = nrmv * X(0, k);
    for (int i = 0; i < n; i++)
      {
      fac2 += X(i, k) * v(i);
      }
    const double fac = fac1 * fac2;
    X(0, k) = v1 * fac - X(0, k);
    for (int i = 1; i < n; i++)
      {
      X(i, k) = v(i) * fac - X(i, k);
      }
    }
  return 0;
  }

//! compute X=X*Q' in place
int Householder::ApplyR(Epetra_SerialDenseMatrix& X,
  Epetra_SerialDenseVector v) const
  {
  HYMLS_PROF3(label_, "ApplyR (2)");

  // X = (2vv'/v'v-I)Y
  // can be written as X = Z-X, Z= (2/nrmv^2 v'Y)v

  int n = X.N();
#ifdef HYMLS_TESTING
  if (v.Length() != n)
    return -1;
#endif

  // Scale with the first element of the test vector to assure
  // that the first element is always positive
  CHECK_ZERO(v.Scale(sign(v(0))));

  // to zero out all but the first entry in v, use v
  const double nrmv = v.Norm2();
  const double v1 = v(0) + nrmv; // first vector element, all others are those in v

  // this is 2/(v'v) with the adjusted v
  const double fac1 = 1.0 / (nrmv * v1);
  for (int k = 0; k < X.M(); k++)
    {
    // v'x
    double fac2 = nrmv * X(k, 0);
    for (int i = 0; i < n; i++)
      {
      fac2 += X(k, i) * v(i);
      }
    const double fac = fac1 * fac2;
    X(k, 0) = v1 * fac - X(k, 0);
    for (int i = 1; i < n; i++)
      {
      X(k, i) = v(i) * fac - X(k, i);
      }
    }
  return 0;
  }

int Householder::Construct(Epetra_CrsMatrix& H,
#ifdef HYMLS_LONG_LONG
  const Epetra_LongLongSerialDenseVector& inds,
#else
  const Epetra_IntSerialDenseVector& inds,
#endif
  Epetra_SerialDenseVector v) const
  {
  HYMLS_PROF3(label_, "Construct (3)");
  // v is the test vector to be zeroed out by this transform,
  // construct the according v for the Householder reflection:
  int n = v.Length();
  hymls_gidx row = inds[0];
  double nrm = v.Norm2();
  
  // Scale with the first element of the test vector to assure
  // that the first element is always positive
  CHECK_ZERO(v.Scale(sign(v[0])));
  v[0] = v[0] + nrm;
  nrm = v.Norm2();
  CHECK_ZERO(v.Scale(1.0 / nrm));

  if (H.Filled())
    {
    CHECK_ZERO(H.ReplaceGlobalValues(row,n,v.A(),const_cast<hymls_gidx*>(&(inds[0]))));
    }
  else
    {
    CHECK_NONNEG(H.InsertGlobalValues(row,n,v.A(),const_cast<hymls_gidx*>(&(inds[0]))));
    }
  return 0;
  }

//! apply a sparse matrix representation of a set of transforms from the left
//! and right to a sparse matrix.
// H is a matrix representation of the vector w=v'/sqrt(v'v), so
// we have to apply (2w'w-I)A(2w'w-I)=A-2w'wA-2Aw'w+4w'wAw'w
//                                   =A-2Aw'w-2w'w(A-2Aw'w)
Teuchos::RCP<Epetra_CrsMatrix> Householder::Apply(
  const Epetra_CrsMatrix& T, const Epetra_CrsMatrix& A) const
  {
  HYMLS_PROF2(label_,"H^TAH (1)");

  if (!A.Filled() || !T.Filled())
    {
    Tools::Error("For at least one of the matrices FillComplete() has not been called!",
      __FILE__,__LINE__);
    }

  Wmat_=Teuchos::rcp(&T,false);

#ifdef HYMLS_STORE_MATRICES
  MatrixUtils::Dump(A, "HOUSE_A.txt");
  MatrixUtils::Dump(*Wmat_, "HOUSE_W.txt");
#endif

  // Aw'
  Teuchos::RCP<Epetra_CrsMatrix> AwT = Teuchos::rcp(new
    Epetra_CrsMatrix(Copy,A.RowMap(),A.MaxNumEntries()) );

  HYMLS_DEBUG("compute A*wT...");
#if 0
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

  HYMLS_DEBUG("compute A*wT*w...");
  CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(*AwT,false,T,false,*AwTw,false));

#ifdef HYMLS_STORE_MATRICES
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

  HYMLS_DEBUG("compute C=A(2wTw-I)...");
  CHECK_ZERO(EpetraExt::MatrixMatrix::Add(A,false,-1.0,*Cmat_,2.0));
  CHECK_ZERO(Cmat_->FillComplete());

#ifdef HYMLS_STORE_MATRICES
  MatrixUtils::Dump(*Cmat_,"HOUSE_C.txt");
#endif

  // wC
  WCmat_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy,A.RowMap(),Cmat_->MaxNumEntries()) );

  HYMLS_DEBUG("compute wC...");
  CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(T,false,*Cmat_,false,*WCmat_,true));

  // wTwC
  Teuchos::RCP<Epetra_CrsMatrix> wTwC = Teuchos::rcp(new
    Epetra_CrsMatrix(Copy,A.RowMap(),Wmat_->MaxNumEntries()));

  HYMLS_DEBUG("compute wTwC...");
#ifndef MATMUL_BUG
  CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(*Wmat_,true,*WCmat_,false,*wTwC,false));
#else
  CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(*WTmat_,false,*WCmat_,false,*wTwC,false));
#endif
#ifdef HYMLS_STORE_MATRICES
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

  HYMLS_DEBUG("compute TAT=2wTwC-C...");
  CHECK_ZERO(EpetraExt::MatrixMatrix::Add(*Cmat_,false,-1.0,*wTwC,2.0));
  wTwC->FillComplete();

#ifdef HYMLS_STORE_MATRICES
  // not filled yet
  //MatrixUtils::Dump(*wTwC,"HOUSE_HAH.txt");
#endif

  if (SaveMemory())
    {
    Cmat_=Teuchos::null;
    }

  HYMLS_DEBUG("done!");
  return wTwC;
  }

//! apply a sparse matrix representation of a set of transforms from the left
//! and right to a sparse matrix. This variant is to be preferred if the
//! sparsity pattern of the transformed matrix TAT is already known.
//  As above, we compute the product as A-2Aw'w-2w'w(A-2Aw'w)
int Householder::Apply(
  Epetra_CrsMatrix& TAT, const Epetra_CrsMatrix& T, const Epetra_CrsMatrix& A) const
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

  if (!A.Filled())
    {
    Tools::Error("A not filled!",__FILE__,__LINE__);
    }

  if (!Wmat_->Filled())
    {
    Tools::Error("W not filled!",__FILE__,__LINE__);
    }

  if (!WTmat_->Filled())
    {
    Tools::Error("WT not filled!",__FILE__,__LINE__);
    }

  // Aw'
  HYMLS_DEBUG("compute A*wT...");
  CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(A,false,*WTmat_,false,TAT));

  // Aw'w
  HYMLS_DEBUG("compute A*wT*w...");
  CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(TAT,false,*Wmat_,false,*Cmat_));

  // C=2Aw'w-A
  HYMLS_DEBUG("compute C=A(I-2wTw)...");
  CHECK_ZERO(EpetraExt::MatrixMatrix::Add(A,false,1.0,*Cmat_,-2.0));

  // wC
  HYMLS_DEBUG("compute wC...");
  CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(*Wmat_,false,*Cmat_,false,*WCmat_));

  // wTwC
  HYMLS_DEBUG("compute wTwC...");
  CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(*WTmat_,false,*WCmat_,false,TAT));

  HYMLS_DEBUG("compute TAT=C-2wTwC...");
  CHECK_ZERO(EpetraExt::MatrixMatrix::Add(*Cmat_,false,1.0,TAT,-2.0));
  HYMLS_DEBUG("done!");

  return 0;
  }

//! apply a sparse matrix representation of a set of transforms from the left
//! and right to a sparse matrix. This variant is to be preferred if the
//! sparsity pattern of the transformed matrix TAT is already known.
int Householder::Apply(
  Epetra_MultiVector& Tv, const Epetra_CrsMatrix& T, const Epetra_MultiVector& v) const
  {
  HYMLS_PROF2(label_,"H*v");
  Epetra_MultiVector tmp = v;
  CHECK_ZERO(T.Multiply(false,v,tmp));
  CHECK_ZERO(T.Multiply(true,tmp,Tv));
  CHECK_ZERO(Tv.Update(-1.0,v,2.0));

  return 0;
  }

int Householder::ApplyInverse(
  Epetra_MultiVector& Tv, const Epetra_CrsMatrix& T, const Epetra_MultiVector& v) const
  {
  return this->Apply(Tv,T,v);
  }

  }
