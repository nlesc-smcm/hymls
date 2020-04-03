#ifndef HYMLS_HOUSEHOLDER_H
#define HYMLS_HOUSEHOLDER_H

#include "HYMLS_config.h"
#include "HYMLS_IndexVector.hpp"
#include "HYMLS_OrthogonalTransform.hpp"

class Epetra_RowMatrixTransposer;

namespace HYMLS {

//! Householder transform 2vv'/(v'v)-I, where v=z+e_1||z||_2 and z is a 'test vetor'.
//! This transformation eliminates all but the first element of z when applied to that
//! test vector, H*z=e_1*||z||_2
class Householder : public OrthogonalTransform
  {

public:

  //! constructor. The level parameter is just to get the object label right.
  Householder(int lev=0);

  //!
  virtual ~Householder();

  //! compute X=H*X in place, with H=2vv'/(v'v)-I, and v=z+e_1*||z||_2
  int Apply(Epetra_SerialDenseMatrix& X,
    const Epetra_SerialDenseVector& z) const;

  //! compute X=X*H' in place, with H=2vv'-I, v=z+e_1||z||_2
  int ApplyR(Epetra_SerialDenseMatrix& X,
    const Epetra_SerialDenseVector& z) const;

  //! explicitly form the structurally orthogonal operator V as a sparse matrix. 
  //! For every separator group it may contain exactly one non-zero column (stored
  //! as row of V' here). We can then use the V matrix to apply VV'/(V'V)-I for all
  //! separator groups simultaneously. The columns of V (stored as rows of V') will
  //! be normalized so that applying the operator can be implemented by matrix-vector
  //! and matrix-matrix products.
  //! The dimension and indices of the entries z to be transformed are given by
  //! the size of the input vector. The function may be called repeatedly
  //! for different sets of indices (separator groups) to construct a matrix
  //! for simultaneously applying many transforms. Always use the corresponding
  //! Apply() functions to apply the transform rather than sparse matrix-matrix
  //! products.
  virtual int Construct(Epetra_CrsMatrix& V,
    const IndexVector& inds,
    const Epetra_SerialDenseVector& vec) const;

  //! apply a sparse matrix representation V of a set of transforms from the left
  //! and right to a sparse matrix: result <- (2V'V-I)*A*(2V'V-I)
  Teuchos::RCP<Epetra_CrsMatrix> Apply(
    const Epetra_CrsMatrix& V, const Epetra_CrsMatrix& A) const ;

  //! apply a sparse matrix representation of a set of transforms from the left
  //! and right to a sparse matrix. This variant is to be preferred if the
  //! sparsity pattern of the transformed matrix TAT is already known.
  int Apply(
    Epetra_CrsMatrix& TAT, const Epetra_CrsMatrix& T, const Epetra_CrsMatrix& A) const;

  //! apply a sparse matrix representation V of a set of transforms from the left
  //! to a vector, result <- (2VV'-I)x.
  int Apply(
    Epetra_MultiVector& Tv, const Epetra_CrsMatrix& T, const Epetra_MultiVector& x) const;

  //! apply a sparse matrix representation of a set of transforms from the left
  //! to a vector.
  int ApplyInverse(
    Epetra_MultiVector& Tv, const Epetra_CrsMatrix& T, const Epetra_MultiVector& v) const;

  bool SaveMemory() const {return save_mem_;}

protected:

  //! object label
  std::string label_;

  //!
  int myLevel_;

  //! we store pointers to sparse matrices so that we can
  //! apply a series of transforms as T'AT more efficiently
  mutable Teuchos::RCP<Epetra_RowMatrixTransposer> Transp_;
  mutable Teuchos::RCP<const Epetra_CrsMatrix> Wmat_;
  mutable Teuchos::RCP<Epetra_CrsMatrix> WTmat_,Cmat_, WCmat_;

private:

  // if false, intermediate results are stored in the class that make
  // subsequent computations of T*A*T a bit faster. Since this costs a
  // lot of memory, we disable this feature until it becomes more of a
  // performance problem.
  static const bool save_mem_ = true;

  };

  }
#endif
