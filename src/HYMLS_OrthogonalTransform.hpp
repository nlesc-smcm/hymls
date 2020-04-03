#ifndef HYMLS_ORTHOGONAL_TRANSFORM_H
#define HYMLS_ORTHOGONAL_TRANSFORM_H

#include "HYMLS_config.h"
#include "Teuchos_RCP.hpp"
#include "HYMLS_IndexVector.hpp"

class Epetra_SerialDenseMatrix;
class Epetra_SerialDenseVector;
class Epetra_CrsMatrix;
class Epetra_MultiVector;

namespace HYMLS {

//! abstract definition of the kind of serial orthogonal transformations
//! we need in HYMLS. Supports application to dense vectors and matrices.
class OrthogonalTransform
  {

public:

  //! compute X=Q*X in place
  //! with a vector passed in to define the transformation
  virtual int Apply(Epetra_SerialDenseMatrix& X,
    const Epetra_SerialDenseVector& v) const = 0;

  //! compute X=X*Q' in place
  //! with a vector passed in to define the transformation
  virtual int ApplyR(Epetra_SerialDenseMatrix& X,
    const Epetra_SerialDenseVector& v) const = 0;

  //! explicitly form the OT as a sparse matrix. The dimension and indices
  //! of the entries to be transformed are given by
  //! the size of the input vector. The function may be called repeatedly
  //! for different sets of indices (separator groups) to construct a matrix
  //! for simultaneously applying many transforms. Always use the corresponding
  //! Apply() functions to apply the transform rather than sparse matrix-matrix
  //! products. If vec is omitted, it is set to all ones. Otherwise it is used
  //! as a test vector from which all but one entries should be eliminated.
  virtual int Construct(Epetra_CrsMatrix& H,
    const IndexVector& inds,
    const Epetra_SerialDenseVector& vec) const = 0;

  //! apply a sparse matrix representation of a set of transforms from the left
  //! and right to a sparse matrix.
  virtual Teuchos::RCP<Epetra_CrsMatrix> Apply
    (const Epetra_CrsMatrix& T, const Epetra_CrsMatrix& A) const = 0 ;

  //! apply a sparse matrix representation of a set of transforms from the left
  //! and right to a sparse matrix. This variant is to be preferred if the
  //! sparsity pattern of the transformed matrix TAT is already known.
  virtual int Apply
    (Epetra_CrsMatrix& TAT, const Epetra_CrsMatrix& T, const Epetra_CrsMatrix& A) const = 0;

  //! apply a sparse matrix representation of a set of transforms from the left
  //! to a vector.
  virtual int Apply
    (Epetra_MultiVector& vT, const Epetra_CrsMatrix& T, const Epetra_MultiVector& v) const = 0;

  //! apply a sparse matrix representation of a set of transforms from the left
  //! to a vector.
  virtual int ApplyInverse
    (Epetra_MultiVector& vT, const Epetra_CrsMatrix& T, const Epetra_MultiVector& v) const = 0;

  //! this can be used to indicate that no memory is stored inside the class and thus
  //! you always have to call the variant of Apply which returns a sparse matrix
  virtual bool SaveMemory() const {return false;}

  };

  }
#endif
