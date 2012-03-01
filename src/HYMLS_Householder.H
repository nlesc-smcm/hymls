#ifndef HYMLS_HOUSEHOLDER_H
#define HYMLS_HOUSEHOLDER_H

#include "HYMLS_OrthogonalTransform.H"

class Epetra_RowMatrixTransposer;

namespace HYMLS {

//! Householder transform I-2vv'/v'v, where v_1=1+sqrt(n) and v_j=1 for 1<j<=n
class Householder : public OrthogonalTransform
  {
  
  public:
 
  //! constructor. The level parameter is just to get the object label right.
  Householder(int lev=0);
  
  //!
  ~Householder();

  //! compute X=H*Y
  int Apply(const Epetra_SerialDenseVector& Y, Epetra_SerialDenseVector& X) const;

  //! compute X=H*X in place
  int Apply(Epetra_SerialDenseVector& X) const;

  //! compute X=H*Y*H'
  int Apply(const Epetra_SerialDenseMatrix& Y, Epetra_SerialDenseMatrix& X) const;

  //! compute X=H*Y*H' in place
  int Apply(Epetra_SerialDenseMatrix& X) const;

  //! compute X=H'*Y
  int ApplyInverse(const Epetra_SerialDenseVector& Y, Epetra_SerialDenseVector& X) const;

  //! compute X=H'*X in place
  int ApplyInverse(Epetra_SerialDenseVector& X) const;

  //! compute X=H'*Y*H
  int ApplyInverse(const Epetra_SerialDenseMatrix& Y, Epetra_SerialDenseMatrix& X) const;

  //! explicitly form the OT as a dense matrix. The dimension is given by
  //! the size of the output matrix.
  int Construct(Epetra_SerialDenseMatrix& M) const;

  //! explicitly form the OT as a sparse matrix. The dimension and indices
  //! of the entries to be transformed are given by 
  //! the size of the input vector. The function may be called repeatedly
  //! for different sets of indices (separator groups) to construct a matrix
  //! for simultaneously applying many transforms. Always use the corresponding
  //! Apply() functions to apply the transform rather than sparse matrix-matrix
  //! products.
  int Construct(Epetra_CrsMatrix& H, const Epetra_IntSerialDenseVector& inds) const;
  int Construct(Epetra_CrsMatrix& H, const Epetra_IntSerialDenseVector& inds, 
        const Epetra_SerialDenseVector& vec) const;

  //! apply a sparse matrix representation of a set of transforms from the left
  //! and right to a sparse matrix.
  Teuchos::RCP<Epetra_CrsMatrix> Apply(const Epetra_CrsMatrix& T, const Epetra_CrsMatrix A) const ;

  //! apply a sparse matrix representation of a set of transforms from the left
  //! and right to a sparse matrix. This variant is to be preferred if the 
  //! sparsity pattern of the transformed matrix TAT is already known.
  int Apply(Epetra_CrsMatrix& TAT, const Epetra_CrsMatrix& T, const Epetra_CrsMatrix& A) const;

  //! apply a sparse matrix representation of a set of transforms from the left
  //! to a vector. 
  int Apply(Epetra_MultiVector& Tv, const Epetra_CrsMatrix& T, const Epetra_MultiVector& v) const;

  //! apply a sparse matrix representation of a set of transforms from the left
  //! to a vector. 
  int ApplyInverse(Epetra_MultiVector& Tv, const Epetra_CrsMatrix& T, const Epetra_MultiVector& v) const;

protected:

  //! object label
  std::string label_;
  
  //! we store pointers to sparse matrices so that we can
  //! apply a series of transforms as T'AT more efficiently
  mutable Teuchos::RCP<Epetra_RowMatrixTransposer> Transp_;
  mutable Teuchos::RCP<const Epetra_CrsMatrix> Wmat_;
  mutable Teuchos::RCP<Epetra_CrsMatrix> WTmat_,Cmat_, WCmat_;
  
  };

}
#endif
