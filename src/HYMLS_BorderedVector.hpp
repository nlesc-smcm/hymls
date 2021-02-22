#ifndef HYMLS_BORDERED_VECTOR_H
#define HYMLS_BORDERED_VECTOR_H

#include "HYMLS_config.h"

#include <vector>

#include <Trilinos_version.h>

#include <Teuchos_RCP.hpp>

#include "Epetra_DataAccess.h"

#include "BelosMultiVec.hpp"

class Epetra_Comm;
class Epetra_BlockMap;
class Epetra_MultiVector;
class Epetra_SerialDenseMatrix;

namespace HYMLS {

class BorderedVector
  {
  //! Pointers to multivector and matrix
  Teuchos::RCP<Epetra_MultiVector> first_;
  Teuchos::RCP<Epetra_MultiVector> second_;

public:
  // default constructor
  BorderedVector() = delete;

  BorderedVector(const Epetra_BlockMap &map1, const Epetra_BlockMap &map2,
    int numVectors, bool zeroOut = true);

  // Copy constructor
  BorderedVector(const BorderedVector &source);

  BorderedVector(Epetra_DataAccess CV, const Epetra_MultiVector &first,
    const Epetra_MultiVector &second);

  BorderedVector(Epetra_DataAccess CV, const Epetra_MultiVector &first,
    const Epetra_SerialDenseMatrix &second);

  // const
  BorderedVector(Epetra_DataAccess CV, const BorderedVector &source,
    int *indices, int numVectors);

  // nonconst
  BorderedVector(Epetra_DataAccess CV, BorderedVector &source,
    int *indices, int numVectors);

  // const
  BorderedVector(Epetra_DataAccess CV, const BorderedVector &source,
    int startIndex, int numVectors);

  // nonconst
  BorderedVector(Epetra_DataAccess CV, BorderedVector &source,
    int startIndex, int numVectors);

  virtual ~BorderedVector() {}

  // Assignment operator
  BorderedVector &operator=(const BorderedVector &Source);

  // Insertion operator
  friend std::ostream &operator<<(std::ostream &out, const BorderedVector &mv);

  // Get the rcpointers
  Teuchos::RCP<Epetra_MultiVector> First();
  Teuchos::RCP<Epetra_MultiVector> Second();
  Teuchos::RCP<Epetra_MultiVector> Vector();
  Teuchos::RCP<Epetra_SerialDenseMatrix> Border();

  Teuchos::RCP<Epetra_MultiVector> First() const;
  Teuchos::RCP<Epetra_MultiVector> Second() const;
  Teuchos::RCP<Epetra_MultiVector> Vector() const;
  Teuchos::RCP<Epetra_SerialDenseMatrix> Border() const;

  // Method to set the border since the assignment operator does not work
  // properly with the views that we use
  int SetBorder(const Epetra_SerialDenseMatrix &mv2);

  // Get number of vectors in each multivector
  int NumVectors() const;

  // Get the global length of the combined multivector
  int GlobalLength() const;

  // Get the global length of the combined multivector
  long long GlobalLength64() const;

  // Get the local length of the combined multivector
  int MyLength() const;

  // Query the stride
  bool ConstantStride() const;

  bool DistributedGlobal() const;

  const Epetra_Comm& Comm() const;

  // this = alpha*A*B + scalarThis*this
  int Multiply(char transA, char transB, double scalarAB,
    const BorderedVector &A, const Epetra_MultiVector &B,
    double scalarThis);

  // this = scalarA*A + scalarThis*this
  int Update(double scalarA, const BorderedVector &A, double scalarThis);

  // this = scalarA*A + scalarB*B + scalarThis*this
  int Update(double scalarA, const BorderedVector &A,
    double scalarB, const BorderedVector &B, double scalarThis);

  // b[j] := this[j]^T * A[j]
  int Dot(const BorderedVector& A, std::vector<double> &b1) const;

  // result[j] := this[j]^T * A[j]
  int Dot(const BorderedVector& A, double *result) const;

  int Scale(double scalarValue);

  int Norm1(std::vector<double> &result) const;

  int Norm2(double *result) const;

  int Norm2(std::vector<double> &result) const;

  int NormInf(std::vector<double> &result) const;

  int Random();

  int PutScalar(double alpha);

  void Print(std::ostream &os) const;
  };

  } // namespace HYMLS

//------------------------------------------------------------------
// Specialization of MultiVectorTraits for Belos,
//  adapted from BelosEpetraAdapter.hpp, for better documentation go there.
//------------------------------------------------------------------

namespace Belos
  {
template<>
class MultiVecTraits<double, HYMLS::BorderedVector>
  {

public:

  static Teuchos::RCP<HYMLS::BorderedVector>
  Clone (const HYMLS::BorderedVector &mv, const int numVecs);

  static Teuchos::RCP<HYMLS::BorderedVector>
  CloneCopy (const HYMLS::BorderedVector &mv);

  static Teuchos::RCP<HYMLS::BorderedVector>
  CloneCopy (const HYMLS::BorderedVector &mv, const std::vector<int> &index);

  static Teuchos::RCP<HYMLS::BorderedVector>
  CloneCopy (const HYMLS::BorderedVector &mv, const Teuchos::Range1D &index);

  static Teuchos::RCP<HYMLS::BorderedVector>
  CloneViewNonConst (HYMLS::BorderedVector &mv, const std::vector<int> &index);

  static Teuchos::RCP<HYMLS::BorderedVector>
  CloneViewNonConst (HYMLS::BorderedVector& mv, const Teuchos::Range1D& index);

  static Teuchos::RCP<const HYMLS::BorderedVector>
  CloneView (const HYMLS::BorderedVector& mv, const std::vector<int>& index);

  static Teuchos::RCP<HYMLS::BorderedVector>
  CloneView (const HYMLS::BorderedVector &mv, const Teuchos::Range1D &index);

  static int
  GetVecLength( const HYMLS::BorderedVector& mv);

  static int
  GetNumberVecs (const HYMLS::BorderedVector& mv);

  static bool
  HasConstantStride (const HYMLS::BorderedVector& mv);

  static ptrdiff_t
  GetGlobalLength (const HYMLS::BorderedVector& mv);

  // Epetra style (we should compare this with just a bunch of updates)
  static void MvTimesMatAddMv (const double alpha,
    const HYMLS::BorderedVector& A,
    const Teuchos::SerialDenseMatrix<int,double>& B,
    const double beta,
    HYMLS::BorderedVector& mv);

  static void
  MvAddMv (const double alpha,
    const HYMLS::BorderedVector& A,
    const double beta,
    const HYMLS::BorderedVector& B,
    HYMLS::BorderedVector& mv);

  static void
  MvScale (HYMLS::BorderedVector& mv,
    const double alpha);

  //! For all columns j of  mv, set mv[j] = alpha[j] * mv[j].
  static void
  MvScale (HYMLS::BorderedVector &mv,
    const std::vector<double> &alpha);

  //! B := alpha * A^T * mv.
  //! Epetra style
  static void MvTransMv(const double alpha, const HYMLS::BorderedVector &A,
    const HYMLS::BorderedVector &mv, Teuchos::SerialDenseMatrix<int,double> &B);

  //! For all columns j of mv, set b[j] := mv[j]^T * A[j].
  static void
  MvDot (const HYMLS::BorderedVector &mv,
    const HYMLS::BorderedVector &A,
    std::vector<double> &b);

  //! For all columns j of mv, set normvec[j] = norm(mv[j]).
  static void
  MvNorm (const HYMLS::BorderedVector &mv,
    std::vector<double> &normvec,
    NormType type = TwoNorm);

  static void
  SetBlock (const HYMLS::BorderedVector &A,
    const std::vector<int> &index,
    HYMLS::BorderedVector &mv);

  static void
  SetBlock (const HYMLS::BorderedVector &A,
    const Teuchos::Range1D &index,
    HYMLS::BorderedVector &mv);

  static void
  Assign (const HYMLS::BorderedVector& A,
    HYMLS::BorderedVector& mv);

  static void
  MvRandom (HYMLS::BorderedVector& mv);

  static void
  MvInit (HYMLS::BorderedVector& mv,
    double alpha = Teuchos::ScalarTraits<double>::zero());

  static void
  MvPrint(const HYMLS::BorderedVector& mv, std::ostream& os);

  }; // end of specialization

#if TRILINOS_MAJOR_VERSION<12
template<>
class MultiVecTraitsExt<double, HYMLS::BorderedVector>
  {
public:
  //! @name New attribute methods
  //@{

  //! Obtain the vector length of \c mv.
  //! \note This method supersedes GetVecLength, which will be deprecated.
  static ptrdiff_t GetGlobalLength (const HYMLS::BorderedVector& mv);
  };
#endif

  } // namespace Belos

#endif
