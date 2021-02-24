#ifndef HYMLS_COMPLEX_VECTOR_H
#define HYMLS_COMPLEX_VECTOR_H

#include <complex>
#include <vector>

#include <Trilinos_version.h>

#include <Teuchos_RCP.hpp>

#include "Epetra_DataAccess.h"

#include "BelosMultiVec.hpp"

class Epetra_Comm;

namespace HYMLS {
template<class MultiVector>
class ComplexVector
  {
  //! Pointers to imaginary and real parts
  Teuchos::RCP<MultiVector> real_;
  Teuchos::RCP<MultiVector> imag_;

public:
  // default constructor
  ComplexVector() = delete;

  // Copy constructor
  ComplexVector(const ComplexVector &source);

  ComplexVector(const Teuchos::RCP<MultiVector> &real,
    const Teuchos::RCP<MultiVector> &imag);

  ComplexVector(Epetra_DataAccess CV, const MultiVector &source);

  ComplexVector(Epetra_DataAccess CV, const MultiVector &real,
    const MultiVector &imag);

  // const
  ComplexVector(Epetra_DataAccess CV, const ComplexVector &source,
    const std::vector<int> &index);

  // nonconst
  ComplexVector(Epetra_DataAccess CV, ComplexVector &source,
    const std::vector<int> &index);

  // const
  ComplexVector(Epetra_DataAccess CV, const ComplexVector &source,
    int startIndex, int numVectors);

  // nonconst
  ComplexVector(Epetra_DataAccess CV, ComplexVector &source,
    int startIndex, int numVectors);

  virtual ~ComplexVector() {}

  // Assignment operator
  ComplexVector &operator=(const ComplexVector &Source);

  // Insertion operator
  template<class T>
  friend std::ostream &operator<<(std::ostream &out, const ComplexVector<T> &mv);

  // Get the rcpointers
  Teuchos::RCP<MultiVector> Real();
  Teuchos::RCP<MultiVector> Imag();

  Teuchos::RCP<MultiVector> Real() const;
  Teuchos::RCP<MultiVector> Imag() const;

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
  int Multiply(char transA, char transB, std::complex<double> scalarAB,
    const ComplexVector &A, const ComplexVector &B,
    std::complex<double> scalarThis);

  // this = scalarA*A + scalarThis*this
  int Update(std::complex<double> scalarA, const ComplexVector &A, std::complex<double> scalarThis);

  // this = scalarA*A + scalarB*B + scalarThis*this
  int Update(std::complex<double> scalarA, const ComplexVector &A,
    std::complex<double> scalarB, const ComplexVector &B, std::complex<double> scalarThis);

  // result[j] := this[j]^T * A[j]
  int Dot(const ComplexVector& A, std::complex<double> *result) const;

  int Scale(std::complex<double> scalarValue);

  int Norm1(std::vector<double> &result) const;

  int Norm2(double *result) const;

  int Norm2(std::vector<double> &result) const;

  int NormInf(std::vector<double> &result) const;

  int Random();

  int PutScalar(std::complex<double> alpha);

  void Print(std::ostream &os) const;
  };

  } // namespace HYMLS

//------------------------------------------------------------------
// Specialization of MultiVectorTraits for Belos,
//  adapted from BelosEpetraAdapter.hpp, for better documentation go there.
//------------------------------------------------------------------

namespace Belos
  {
template<class MultiVector>
class MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >
  {

public:

  static Teuchos::RCP<HYMLS::ComplexVector<MultiVector> >
  Clone (const HYMLS::ComplexVector<MultiVector> &mv, const int numVecs);

  static Teuchos::RCP<HYMLS::ComplexVector<MultiVector> >
  CloneCopy (const HYMLS::ComplexVector<MultiVector> &mv);

  static Teuchos::RCP<HYMLS::ComplexVector<MultiVector> >
  CloneCopy (const HYMLS::ComplexVector<MultiVector> &mv, const std::vector<int> &index);

  static Teuchos::RCP<HYMLS::ComplexVector<MultiVector> >
  CloneCopy (const HYMLS::ComplexVector<MultiVector> &mv, const Teuchos::Range1D &index);

  static Teuchos::RCP<HYMLS::ComplexVector<MultiVector> >
  CloneViewNonConst (HYMLS::ComplexVector<MultiVector> &mv, const std::vector<int> &index);

  static Teuchos::RCP<HYMLS::ComplexVector<MultiVector> >
  CloneViewNonConst (HYMLS::ComplexVector<MultiVector>& mv, const Teuchos::Range1D& index);

  static Teuchos::RCP<const HYMLS::ComplexVector<MultiVector> >
  CloneView (const HYMLS::ComplexVector<MultiVector>& mv, const std::vector<int>& index);

  static Teuchos::RCP<HYMLS::ComplexVector<MultiVector> >
  CloneView (const HYMLS::ComplexVector<MultiVector> &mv, const Teuchos::Range1D &index);

  static int
  GetVecLength( const HYMLS::ComplexVector<MultiVector>& mv);

  static int
  GetNumberVecs (const HYMLS::ComplexVector<MultiVector>& mv);

  static bool
  HasConstantStride (const HYMLS::ComplexVector<MultiVector>& mv);

  static ptrdiff_t
  GetGlobalLength (const HYMLS::ComplexVector<MultiVector>& mv);

  // Epetra style (we should compare this with just a bunch of updates)
  static void MvTimesMatAddMv (const std::complex<double> alpha,
    const HYMLS::ComplexVector<MultiVector>& A,
    const Teuchos::SerialDenseMatrix<int,std::complex<double> >& B,
    const std::complex<double> beta,
    HYMLS::ComplexVector<MultiVector>& mv);

  static void
  MvAddMv (const std::complex<double> alpha,
    const HYMLS::ComplexVector<MultiVector>& A,
    const std::complex<double> beta,
    const HYMLS::ComplexVector<MultiVector>& B,
    HYMLS::ComplexVector<MultiVector>& mv);

  static void
  MvScale (HYMLS::ComplexVector<MultiVector>& mv,
    const std::complex<double> alpha);

  //! For all columns j of  mv, set mv[j] = alpha[j] * mv[j].
  static void
  MvScale (HYMLS::ComplexVector<MultiVector> &mv,
    const std::vector<std::complex<double> > &alpha);

  //! B := alpha * A^T * mv.
  //! Epetra style
  static void MvTransMv(const std::complex<double> alpha, const HYMLS::ComplexVector<MultiVector> &A,
    const HYMLS::ComplexVector<MultiVector> &mv, Teuchos::SerialDenseMatrix<int,std::complex<double> > &B);

  //! For all columns j of mv, set b[j] := mv[j]^T * A[j].
  static void
  MvDot (const HYMLS::ComplexVector<MultiVector> &mv,
    const HYMLS::ComplexVector<MultiVector> &A,
    std::vector<std::complex<double> > &b);

  //! For all columns j of mv, set normvec[j] = norm(mv[j]).
  static void
  MvNorm (const HYMLS::ComplexVector<MultiVector> &mv,
    std::vector<double> &normvec,
    NormType type = TwoNorm);

  static void
  SetBlock (const HYMLS::ComplexVector<MultiVector> &A,
    const std::vector<int> &index,
    HYMLS::ComplexVector<MultiVector> &mv);

  static void
  SetBlock (const HYMLS::ComplexVector<MultiVector> &A,
    const Teuchos::Range1D &index,
    HYMLS::ComplexVector<MultiVector> &mv);

  static void
  Assign (const HYMLS::ComplexVector<MultiVector>& A,
    HYMLS::ComplexVector<MultiVector>& mv);

  static void
  MvRandom (HYMLS::ComplexVector<MultiVector>& mv);

  static void
  MvInit (HYMLS::ComplexVector<MultiVector>& mv,
    std::complex<double> alpha = Teuchos::ScalarTraits<std::complex<double> >::zero());

  static void
  MvPrint(const HYMLS::ComplexVector<MultiVector>& mv, std::ostream& os);

  }; // end of specialization

#if TRILINOS_MAJOR_VERSION<12
template<class MultiVector>
class MultiVecTraitsExt<std::complex<double>, HYMLS::ComplexVector<MultiVector> >
  {
public:
  //! @name New attribute methods
  //@{

  //! Obtain the vector length of \c mv.
  //! \note This method supersedes GetVecLength, which will be deprecated.
  static ptrdiff_t GetGlobalLength (const HYMLS::ComplexVector<MultiVector>& mv);
  };
#endif

  } // namespace Belos

#endif
