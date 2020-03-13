#ifndef HYMLS_SCHUR_COMPLEMENT_H
#define HYMLS_SCHUR_COMPLEMENT_H

#include "HYMLS_config.h"

#include "Teuchos_RCP.hpp"

#include "Epetra_Operator.h"

#include <string>

// forward declarations

class Epetra_Comm;
class Epetra_Map;
class Epetra_CrsMatrix;
class Epetra_FECrsMatrix;
class Epetra_IntSerialDenseVector;
class Epetra_LongLongSerialDenseVector;
class Epetra_SerialDenseMatrix;

namespace HYMLS {

class MatrixBlock;
class OverlappingPartitioner;

//! efficient implementation of the original Schur-complement of our solver

//! operator representation of our Schur complement.
//! allows applying the Schur complement of our factorization
//! to a vector without actually constructing it.
//! Also provides functionality to explicitly construct parts
//! of the SC or the whole thing as sparse or dense matrix.
class SchurComplement : public Epetra_Operator
  {

public:

  friend class SchurPreconditioner;

  //! constructor. The level parameter is just to get the
  //! label right, the mother defines A11, A12, A21 and A22
  //! so that this class represents A22 - A21 A11\A12
  SchurComplement(
    Teuchos::RCP<const MatrixBlock> A11,
    Teuchos::RCP<const MatrixBlock> A12,
    Teuchos::RCP<const MatrixBlock> A21,
    Teuchos::RCP<const MatrixBlock> A22,
    int lev = 0);

  //! destructor
  virtual ~SchurComplement();

  //!\name Epetra_Operator interface

  //@{

  //! Applies the operator
  int Apply(const Epetra_MultiVector & X,
    Epetra_MultiVector &Y) const;

  //! Apply inverse operator - not implemented.
  int ApplyInverse(const Epetra_MultiVector & X,
    Epetra_MultiVector &Y) const;

  int SetUseTranspose(bool UseTranspose)
    {
    useTranspose_ = false; // not implemented.
    return - 1;
    }

  //! not implemented.
  bool HasNormInf() const {return false; }

  //! infinity norm
  double NormInf() const {return normInf_; }

  //! label
  const char *Label() const {return label_.c_str(); }

  //! use transpose?
  bool UseTranspose() const {return useTranspose_; }

  //! communicator
  const Epetra_Comm &Comm() const;

  //! Returns the Epetra_Map object associated with the domain of this operator.
  const Epetra_Map &OperatorDomainMap() const;

  //! Returns the Epetra_Map object associated with the range of this operator.
  const Epetra_Map &OperatorRangeMap() const;

  //@}

  //! \name functionality to create parts of the Schur complement explicitly
  //@{

  //! const version - construct in user provided data structure (should be created with
  //! correct map and not filled)
  int Construct(Teuchos::RCP<Epetra_FECrsMatrix> S) const;

  //@}

  //! get number of flops during Apply()
  double ApplyFlops() const {return flopsApply_; }

  //! get number of flops during Construct()
  double ComputeFlops() const {return flopsCompute_; }

protected:

  //! Matrix blocks of the original matrix
  Teuchos::RCP<const MatrixBlock> A11_, A12_, A21_, A22_;

  //!
  int myLevel_;

  //! use transposed operator?
  bool useTranspose_;

  //! infinity norm
  double normInf_;

  //! label
  std::string label_;

  //! flops performed during Apply()
  mutable double flopsApply_;

  //! flops performed during Construct()
  mutable double flopsCompute_;

protected:

  //! construct the partial Schur-complement A21*A11\A12 associated with local subdomain k
  //! as a dense matrix. This is a very small matrix as each subdomain
  //! only connects to few separators. The matrix has hid_->NumSeparatorNodes(sd)
  //! rows and columns and should be preallocated by the user. The global row-
  //! and column indices of the dense submatrix should be given in 'indices',
  //! which can be found by the Construct() function above.
  //! If the matrix passed in is too small, it is resized. This is
  //! in principle OK, but may be inefficient if it happens very often.
  //!
  int Construct11(int k, Epetra_SerialDenseMatrix & Sk,
#ifdef HYMLS_LONG_LONG
    Epetra_LongLongSerialDenseVector &inds,
#else
    Epetra_IntSerialDenseVector &inds,
#endif
    double *flops = NULL) const;

  int Construct22(int k, Epetra_SerialDenseMatrix & Sk,
#ifdef HYMLS_LONG_LONG
    Epetra_LongLongSerialDenseVector &inds,
#else
    Epetra_IntSerialDenseVector &inds,
#endif
    double *flops = NULL) const;

  //! to allow the preconditioner access to parts of the unassembled Schur complement:
  const Epetra_CrsMatrix &A22() const;

  //! get the OverlappingPartitioner object
  const OverlappingPartitioner &Partitioner() const;

private:

  }; 

  }

#endif
