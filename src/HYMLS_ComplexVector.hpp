#ifndef HYMLS_COMPLEX_VECTOR_H
#define HYMLS_COMPLEX_VECTOR_H

#include "HYMLS_config.h"

#include <vector>

#include <Trilinos_version.h>

#include <Teuchos_RCP.hpp>

#include "Epetra_DataAccess.h"
#include "Epetra_LocalMap.h"
#include "Epetra_MultiVector.h"

#include "BelosMultiVec.hpp"
#include "BelosEpetraAdapter.hpp"

class Epetra_BlockMap;
class MultiVector;

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
    const ComplexVector &A, const ComplexVector<Epetra_MultiVector> &B,
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

namespace HYMLS {

// Copy constructor
template<class MultiVector>
ComplexVector<MultiVector>::ComplexVector(const ComplexVector &source)
  {
  real_ = Teuchos::rcp(new MultiVector(*source.Real()));
  imag_ = Teuchos::rcp(new MultiVector(*source.Imag()));
  }

template<class MultiVector>
ComplexVector<MultiVector>::ComplexVector(
  const Teuchos::RCP<MultiVector> &mv1, const Teuchos::RCP<MultiVector> &mv2)
  {
  real_ = mv1;
  imag_ = mv2;
  }

template<class MultiVector>
ComplexVector<MultiVector>::ComplexVector(Epetra_DataAccess CV, const MultiVector &source)
  {
  if (source.NumVectors() != 2)
    {
    Tools::Error("Only supported with two vectors", __FILE__, __LINE__);
    }

  real_ = Teuchos::rcp(new MultiVector(CV, source, 0, 1));
  imag_ = Teuchos::rcp(new MultiVector(CV, source, 1, 1));
  }

template<class MultiVector>
ComplexVector<MultiVector>::ComplexVector(Epetra_DataAccess CV, const MultiVector &mv1,
  const MultiVector &mv2)
  {
  if (mv1.NumVectors() != mv2.NumVectors())
    {
    Tools::Error("Incompatible vectors", __FILE__, __LINE__);
    }

  real_ = Teuchos::rcp(new MultiVector(CV, mv1, 0, mv1.NumVectors()));
  imag_ = Teuchos::rcp(new MultiVector(CV, mv2, 0, mv2.NumVectors()));
  }

// const
template<class MultiVector>
ComplexVector<MultiVector>::ComplexVector(Epetra_DataAccess CV, const ComplexVector &source,
  const std::vector<int> &index)
  {
  // cast to nonconst for MultiVector
  std::vector<int> &tmpInd = const_cast< std::vector<int>& >(index);
  real_ = Teuchos::rcp
    (new MultiVector(CV, *source.Real(), &tmpInd[0], index.size()));
  imag_ = Teuchos::rcp
    (new MultiVector(CV, *source.Imag(), &tmpInd[0], index.size()));
  }

// nonconst
template<class MultiVector>
ComplexVector<MultiVector>::ComplexVector(Epetra_DataAccess CV, ComplexVector &source,
  const std::vector<int> &index)
  {
  // cast to nonconst for MultiVector
  std::vector<int> &tmpInd = const_cast< std::vector<int>& >(index);
  real_ = Teuchos::rcp
    (new MultiVector(CV, *source.Real(), &tmpInd[0], index.size()));
  imag_ = Teuchos::rcp
    (new MultiVector(CV, *source.Imag(), &tmpInd[0], index.size()));
  }

// const
template<class MultiVector>
ComplexVector<MultiVector>::ComplexVector(Epetra_DataAccess CV, const ComplexVector &source,
  int startIndex, int numVectors)
  {
  real_ = Teuchos::rcp
    (new MultiVector(CV, *source.Real(), startIndex, numVectors));
  imag_ = Teuchos::rcp
    (new MultiVector(CV, *source.Imag(), startIndex, numVectors));
  }

// nonconst
template<class MultiVector>
ComplexVector<MultiVector>::ComplexVector(Epetra_DataAccess CV, ComplexVector &source,
  int startIndex, int numVectors)
  {
  real_ = Teuchos::rcp
    (new MultiVector(CV, *source.Real(), startIndex, numVectors));
  imag_ = Teuchos::rcp
    (new MultiVector(CV, *source.Imag(), startIndex, numVectors));
  }

// Assignment operator
template<class MultiVector>
ComplexVector<MultiVector> &ComplexVector<MultiVector>::operator=(const ComplexVector &Source)
  {
  *real_ = *Source.Real();
  *imag_ = *Source.Imag();
  return *this;
  }

// Insertion operator
template<class MultiVector>
std::ostream &operator<<(std::ostream &out, const ComplexVector<MultiVector> &mv)
  {
  out << *mv.Real() << std::endl;
  out << *mv.Imag() << std::endl;
  return out;
  }

// Get the rcpointers
template<class MultiVector>
Teuchos::RCP<MultiVector> ComplexVector<MultiVector>::Real()
  {
  return real_;
  }

template<class MultiVector>
Teuchos::RCP<MultiVector> ComplexVector<MultiVector>::Imag()
  {
  return imag_;
  }

template<class MultiVector>
Teuchos::RCP<MultiVector> ComplexVector<MultiVector>::Real() const
  {
  return real_;
  }

template<class MultiVector>
Teuchos::RCP<MultiVector> ComplexVector<MultiVector>::Imag() const
  {
  return imag_;
  }

// Get number of vectors in each multivector
template<class MultiVector>
int ComplexVector<MultiVector>::NumVectors() const
  {
  if (real_.is_null())
    return 0;

  return real_->NumVectors();
  }

// Get the global length of the combined multivector
template<class MultiVector>
int ComplexVector<MultiVector>::GlobalLength() const
  {
  return real_->GlobalLength();
  }

// Get the global length of the combined multivector
template<class MultiVector>
long long ComplexVector<MultiVector>::GlobalLength64() const
  {
  return real_->GlobalLength64();
  }

// Get the local length of the combined multivector
template<class MultiVector>
int ComplexVector<MultiVector>::MyLength() const
  {
  return real_->MyLength();
  }

// Query the stride
template<class MultiVector>
bool ComplexVector<MultiVector>::ConstantStride() const
  {
  return real_->ConstantStride() && imag_->ConstantStride();
  }

template<class MultiVector>
bool ComplexVector<MultiVector>::DistributedGlobal() const
  {
  return real_->DistributedGlobal();
  }

template<class MultiVector>
const Epetra_Comm& ComplexVector<MultiVector>::Comm() const
  {
  return real_->Comm();
  }

// this = alpha*A*B + scalarThis*this
template<class MultiVector>
int ComplexVector<MultiVector>::Multiply(char transA, char transB, std::complex<double> scalarAB,
  const ComplexVector<MultiVector> &A, const ComplexVector<Epetra_MultiVector> &B,
  std::complex<double> scalarThis)
  {
  double conjA = transA == 'T' ? -1.0 : 1.0;
  double conjB = transB == 'T' ? -1.0 : 1.0;

  int info = Scale(scalarThis);

  MultiVector tmp(*real_);
  info += tmp.Multiply(transA, transB, 1.0, *A.Real(), *B.Real(), 0.0);
  info += real_->Update(scalarAB.real(), tmp, 1.0);
  info += imag_->Update(scalarAB.imag(), tmp, 1.0);

  info += tmp.Multiply(transA, transB, -conjA * conjB, *A.Imag(), *B.Imag(), 0.0);
  info += real_->Update(scalarAB.real(), tmp, 1.0);
  info += imag_->Update(scalarAB.imag(), tmp, 1.0);

  info += tmp.Multiply(transA, transB, conjB, *A.Real(), *B.Imag(), 0.0);
  info += real_->Update(-scalarAB.imag(), tmp, 1.0);
  info += imag_->Update(scalarAB.real(), tmp, 1.0);

  info += tmp.Multiply(transA, transB, conjA, *A.Imag(), *B.Real(), 0.0);
  info += real_->Update(-scalarAB.imag(), tmp, 1.0);
  info += imag_->Update(scalarAB.real(), tmp, 1.0);

  return info;
  }

// this = scalarA*A + scalarThis*this
template<class MultiVector>
int ComplexVector<MultiVector>::Update(std::complex<double> scalarA, const ComplexVector &A, std::complex<double> scalarThis)
  {
  // Make a copy so we don't overwrite values before using them
  Teuchos::RCP<const ComplexVector<MultiVector> > Acopy = Teuchos::rcp(&A, false);
  if (real_->Pointers() == A.Real()->Pointers())
    Acopy = Teuchos::rcp(new ComplexVector<MultiVector>(A));

  int info = Scale(scalarThis);

  info += real_->Update(scalarA.real(), *Acopy->Real(), 1.0);
  info += real_->Update(-scalarA.imag(), *Acopy->Imag(), 1.0);
  info += imag_->Update(scalarA.imag(), *Acopy->Real(), 1.0);
  info += imag_->Update(scalarA.real(), *Acopy->Imag(), 1.0);

  return info;
  }

// this = scalarA*A + scalarB*B + scalarThis*this
template<class MultiVector>
int ComplexVector<MultiVector>::Update(std::complex<double> scalarA, const ComplexVector &A,
  std::complex<double> scalarB, const ComplexVector &B, std::complex<double> scalarThis)
  {
  // Make copies so we don't overwrite values before using them
  Teuchos::RCP<const ComplexVector<MultiVector> > Acopy = Teuchos::rcp(&A, false);
  if (real_->Pointers() == A.Real()->Pointers())
    Acopy = Teuchos::rcp(new ComplexVector<MultiVector>(A));

  Teuchos::RCP<const ComplexVector<MultiVector> > Bcopy = Teuchos::rcp(&B, false);
  if (real_->Pointers() == B.Real()->Pointers())
    Bcopy = Teuchos::rcp(new ComplexVector<MultiVector>(B));

  int info = 0;
  info =  Update(scalarA, *Acopy, scalarThis);
  info += Update(scalarB, *Bcopy, 1.0);
  return info;
  }

// result[j] := this[j]^T * A[j]
template<class MultiVector>
int ComplexVector<MultiVector>::Dot(const ComplexVector& A, std::complex<double> *result) const
  {
  std::vector<double> tmp(NumVectors(), 0.0);
  std::vector<double> real(NumVectors(), 0.0);
  std::vector<double> imag(NumVectors(), 0.0);

  int info = 0;
  info += real_->Dot(*A.Real(), &real[0]);
  info += imag_->Dot(*A.Imag(), &tmp[0]);
  for (int i = 0; i < NumVectors(); ++i)
    real[i] += tmp[i];

  info += real_->Dot(*A.Imag(), &imag[0]);
  info += imag_->Dot(*A.Real(), &tmp[0]);
  for (int i = 0; i < NumVectors(); ++i)
    imag[i] -= tmp[i];

  // combine the results
  for (int i = 0; i != NumVectors(); ++i)
    result[i] = std::complex<double>(real[i], imag[i]);

  return info;
  }

template<class MultiVector>
int ComplexVector<MultiVector>::Scale(std::complex<double> scalarValue)
  {
  int info = 0;

  if (scalarValue == 0.0)
    {
    info =  real_->PutScalar(0.0);
    info += imag_->PutScalar(0.0);
    return info;
    }

  if (scalarValue.imag() == 0.0)
    {
    info =  real_->Scale(scalarValue.real());
    info += imag_->Scale(scalarValue.real());
    return info;
    }

  MultiVector tmp(*real_);
  info =  real_->Update(-scalarValue.imag(), *imag_, scalarValue.real());
  info += imag_->Update(scalarValue.imag(), tmp, scalarValue.real());
  return info;
  }

template<class MultiVector>
int ComplexVector<MultiVector>::Norm1(std::vector<double> &result) const
  {
  std::vector<double> localresult(result.size(), 0.0);
  for (int i = 0; i != NumVectors(); ++i)
    {
    double *real = (*real_)[i];
    double *imag = (*imag_)[i];
    for (int j = 0; j < real_->MyLength(); ++j)
      localresult[i] += sqrt(real[j] * real[j] + imag[j] * imag[j]);
    }

  if (DistributedGlobal())
    Comm().MaxAll(&localresult[0], &result[0], NumVectors());
  else
    result = localresult;

  return 0;
  }

template<class MultiVector>
int ComplexVector<MultiVector>::Norm2(double *result) const
  {
  std::vector<double> tmp(NumVectors(), 0.0);
  int info = Norm2(tmp);
  for (int i = 0; i != NumVectors(); ++i)
    result[i] = tmp[i];
  return info;
  }

template<class MultiVector>
int ComplexVector<MultiVector>::Norm2(std::vector<double> &result) const
  {
  // copy result vector
  std::vector<double> tmp(result.size(), 0.0);

  int info = 0;
  info =  real_->Norm2(&result[0]);
  info += imag_->Norm2(&tmp[0]);

  // combine results
  for (int i = 0; i != NumVectors(); ++i)
    result[i] = sqrt(result[i] * result[i] + tmp[i] * tmp[i]);

  return info;
  }

template<class MultiVector>
int ComplexVector<MultiVector>::NormInf(std::vector<double> &result) const
  {
  std::vector<double> localresult(result.size(), 0.0);
  for (int i = 0; i != NumVectors(); ++i)
    {
    double *real = (*real_)[i];
    double *imag = (*imag_)[i];
    for (int j = 0; j < real_->MyLength(); ++j)
      localresult[i] = std::max(sqrt(real[j] * real[j] + imag[j] * imag[j]), localresult[i]);
    }

  if (DistributedGlobal())
    Comm().MaxAll(&localresult[0], &result[0], NumVectors());
  else
    result = localresult;

  return 0;
  }

template<class MultiVector>
int ComplexVector<MultiVector>::Random()
  {
  int info = 0;
  info += real_->Random();
  info += imag_->Random();
  return info;
  }

template<class MultiVector>
int ComplexVector<MultiVector>::PutScalar(std::complex<double> alpha)
  {
  int info = 0;
  info += real_->PutScalar(alpha.real());
  info += imag_->PutScalar(alpha.imag());
  return info;
  }

template<class MultiVector>
void ComplexVector<MultiVector>::Print(std::ostream &os) const
  {
  real_->Print(os);
  imag_->Print(os);
  }

  } // namespace HYMLS


namespace Belos
  {

template<class MultiVector>
Teuchos::RCP<HYMLS::ComplexVector<MultiVector> >
MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::Clone(
  const HYMLS::ComplexVector<MultiVector> &mv, const int numVecs)
  {
  TEUCHOS_TEST_FOR_EXCEPTION(
    numVecs <= 0, std::invalid_argument,
    "Belos::MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::"
    "Clone(mv, numVecs = " << numVecs << "): "
    "outNumVecs must be positive.");

  return Teuchos::rcp(
    new HYMLS::ComplexVector<MultiVector>(
      MultiVecTraits<double, MultiVector>::Clone(*mv.Real(), numVecs),
      MultiVecTraits<double, MultiVector>::Clone(*mv.Imag(), numVecs)));
  }

template<class MultiVector>
Teuchos::RCP<HYMLS::ComplexVector<MultiVector> >
MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::CloneCopy(
  const HYMLS::ComplexVector<MultiVector> &mv)
  {
  return Teuchos::rcp(new HYMLS::ComplexVector<MultiVector>(mv));
  }

template<class MultiVector>
Teuchos::RCP<HYMLS::ComplexVector<MultiVector> >
MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::CloneCopy(
  const HYMLS::ComplexVector<MultiVector> &mv, const std::vector<int> &index)
  {
  const int inNumVecs  = mv.NumVectors();
  const int outNumVecs = index.size();
  TEUCHOS_TEST_FOR_EXCEPTION(outNumVecs == 0, std::invalid_argument,
    "Belos::MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::"
    "CloneCopy(mv, index = {}): At least one vector must be"
    " cloned from mv.");

  if (outNumVecs > inNumVecs)
    {
    std::ostringstream os;
    os << "Belos::MultiVecTraits<std::complex<double>, Combined_Operator>::"
      "CloneCopy(mv, index = {";
    for (int k = 0; k < outNumVecs - 1; ++k)
      os << index[k] << ", ";
    os << index[outNumVecs-1] << "}): There are " << outNumVecs
       << " indices to copy, but only " << inNumVecs << " columns of mv.";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, os.str());
    }

  return Teuchos::rcp(new HYMLS::ComplexVector<MultiVector>(Copy, mv, index));
  }

template<class MultiVector>
Teuchos::RCP<HYMLS::ComplexVector<MultiVector> >
MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::CloneCopy(
  const HYMLS::ComplexVector<MultiVector> &mv, const Teuchos::Range1D &index)
  {
  const int inNumVecs   = mv.NumVectors();
  const int outNumVecs  = index.size();
  const bool validRange = outNumVecs > 0 && index.lbound() >= 0 &&
    index.ubound() < inNumVecs;

  if (! validRange)
    {
    std::ostringstream os;
    os << "Belos::MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::Clone(mv,"
      "index=[" << index.lbound() << ", " << index.ubound() << "]): ";
    TEUCHOS_TEST_FOR_EXCEPTION(outNumVecs == 0, std::invalid_argument,
      os.str() << "Column index range must be nonempty.");
    TEUCHOS_TEST_FOR_EXCEPTION(index.lbound() < 0, std::invalid_argument,
      os.str() << "Column index range must be nonnegative.");
    TEUCHOS_TEST_FOR_EXCEPTION(index.ubound() >= inNumVecs, std::invalid_argument,
      os.str() << "Column index range must not exceed "
      "number of vectors " << inNumVecs << " in the "
      "input multivector.");
    }

  return Teuchos::rcp(
    new HYMLS::ComplexVector<MultiVector>(Copy, mv, index.lbound(), index.size()));
  }

template<class MultiVector>
Teuchos::RCP<HYMLS::ComplexVector<MultiVector> >
MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::CloneViewNonConst(
  HYMLS::ComplexVector<MultiVector> &mv, const std::vector<int> &index)
  {
  const int inNumVecs  = mv.NumVectors();
  const int outNumVecs = index.size();
  // Simple, inexpensive tests of the index vector.

  TEUCHOS_TEST_FOR_EXCEPTION(outNumVecs == 0, std::invalid_argument,
    "Belos::MultiVecTraits<std::complex<double>,HYMLS::ComplexVector<MultiVector> >::"
    "CloneViewNonConst(mv, index = {}): The output view "
    "must have at least one column.");
  if (outNumVecs > inNumVecs)
    {
    std::ostringstream os;
    os << "Belos::MultiVecTraits<std::complex<double>,HYMLS::ComplexVector<MultiVector> >::"
      "CloneViewNonConst(mv, index = {";
    for (int k = 0; k < outNumVecs - 1; ++k)
      os << index[k] << ", ";
    os << index[outNumVecs-1] << "}): There are " << outNumVecs
       << " indices to view, but only " << inNumVecs << " columns of mv.";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, os.str());
    }
  return Teuchos::rcp(
    new HYMLS::ComplexVector<MultiVector>(View, mv, index));
  }

template<class MultiVector>
Teuchos::RCP<HYMLS::ComplexVector<MultiVector> >
MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::CloneViewNonConst(
  HYMLS::ComplexVector<MultiVector>& mv, const Teuchos::Range1D& index)
  {
  const bool validRange = index.size() > 0 &&
    index.lbound() >= 0 &&
    index.ubound() < mv.NumVectors();

  if (! validRange)
    {
    std::ostringstream os;
    os << "Belos::MultiVecTraits<std::complex<double>,HYMLS::ComplexVector<MultiVector> >::CloneView"
      "NonConst(mv,index=[" << index.lbound() << ", " << index.ubound()
       << "]): ";
    TEUCHOS_TEST_FOR_EXCEPTION(index.size() == 0, std::invalid_argument,
      os.str() << "Column index range must be nonempty.");
    TEUCHOS_TEST_FOR_EXCEPTION(index.lbound() < 0, std::invalid_argument,
      os.str() << "Column index range must be nonnegative.");
    TEUCHOS_TEST_FOR_EXCEPTION(index.ubound() >= mv.NumVectors(),
      std::invalid_argument,
      os.str() << "Column index range must not exceed "
      "number of vectors " << mv.NumVectors() << " in "
      "the input multivector.");
    }
  return Teuchos::rcp(
    new HYMLS::ComplexVector<MultiVector>(View, mv, index.lbound(), index.size()));
  }

template<class MultiVector>
Teuchos::RCP<const HYMLS::ComplexVector<MultiVector> >
MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::CloneView(
  const HYMLS::ComplexVector<MultiVector>& mv, const std::vector<int>& index)
  {
  const int inNumVecs  = mv.NumVectors();
  const int outNumVecs = index.size();

  // Simple, inexpensive tests of the index vector.
  TEUCHOS_TEST_FOR_EXCEPTION(outNumVecs == 0, std::invalid_argument,
    "Belos::MultiVecTraits<std::complex<double>,HYMLS::ComplexVector<MultiVector> >::"
    "CloneView(mv, index = {}): The output view "
    "must have at least one column.");
  if (outNumVecs > inNumVecs)
    {
    std::ostringstream os;
    os << "Belos::MultiVecTraits<std::complex<double>,HYMLS::ComplexVector<MultiVector> >::"
      "CloneView(mv, index = {";
    for (int k = 0; k < outNumVecs - 1; ++k)
      os << index[k] << ", ";
    os << index[outNumVecs-1] << "}): There are " << outNumVecs
       << " indices to view, but only " << inNumVecs << " columns of mv.";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, os.str());
    }

  return Teuchos::rcp(new HYMLS::ComplexVector<MultiVector>(View, mv, index));
  }

template<class MultiVector>
Teuchos::RCP<HYMLS::ComplexVector<MultiVector> >
MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::CloneView(
  const HYMLS::ComplexVector<MultiVector> &mv, const Teuchos::Range1D &index)
  {
  const bool validRange = index.size() > 0 &&
    index.lbound() >= 0 &&
    index.ubound() < mv.NumVectors();
  if (! validRange)
    {
    std::ostringstream os;
    os << "Belos::MultiVecTraits<std::complex<double>,HYMLS::ComplexVector<MultiVector> >::CloneView"
      "(mv,index=[" << index.lbound() << ", " << index.ubound()
       << "]): ";
    TEUCHOS_TEST_FOR_EXCEPTION(index.size() == 0, std::invalid_argument,
      os.str() << "Column index range must be nonempty.");
    TEUCHOS_TEST_FOR_EXCEPTION(index.lbound() < 0, std::invalid_argument,
      os.str() << "Column index range must be nonnegative.");
    TEUCHOS_TEST_FOR_EXCEPTION(index.ubound() >= mv.NumVectors(),
      std::invalid_argument,
      os.str() << "Column index range must not exceed "
      "number of vectors " << mv.NumVectors() << " in "
      "the input multivector.");
    }
  return Teuchos::rcp(new HYMLS::ComplexVector<MultiVector>(View, mv, index.lbound(), index.size()));
  }

template<class MultiVector>
int MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::GetVecLength(
  const HYMLS::ComplexVector<MultiVector>& mv)
  {
  return mv.GlobalLength();
  }

template<class MultiVector>
int MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::GetNumberVecs(
  const HYMLS::ComplexVector<MultiVector>& mv)
  {
  return mv.NumVectors();
  }

template<class MultiVector>
bool MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::HasConstantStride(
  const HYMLS::ComplexVector<MultiVector>& mv)
  {
  return mv.ConstantStride();
  }

template<class MultiVector>
ptrdiff_t MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::GetGlobalLength(
  const HYMLS::ComplexVector<MultiVector>& mv)
  {
  if ( mv.Real()->Map().GlobalIndicesLongLong() )
    return static_cast<ptrdiff_t>( mv.GlobalLength64() );
  else
    return static_cast<ptrdiff_t>( mv.GlobalLength() );
  }

// Epetra style (we should compare this with just a bunch of updates)
template<class MultiVector>
void MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::MvTimesMatAddMv(
  const std::complex<double> alpha,
  const HYMLS::ComplexVector<MultiVector>& A,
  const Teuchos::SerialDenseMatrix<int,std::complex<double> >& B,
  const std::complex<double> beta,
  HYMLS::ComplexVector<MultiVector>& mv)
  {
  // Create Epetra_Multivectors from SerialDenseMatrix
  Epetra_LocalMap LocalMap(B.numRows(), 0, mv.Comm());
  Epetra_MultiVector B_real(LocalMap, B.numCols());
  Epetra_MultiVector B_imag(LocalMap, B.numCols());
  for (int i = 0; i < B.numRows(); ++i)
    for (int j = 0; j < B.numCols(); ++j)
      {
      B_real[j][i] = B(i, j).real();
      B_imag[j][i] = B(i, j).imag();
      }

  HYMLS::ComplexVector<Epetra_MultiVector> B_Pvec(View, B_real, B_imag);
  const int info = mv.Multiply('N', 'N', alpha, A, B_Pvec, beta);

  TEUCHOS_TEST_FOR_EXCEPTION(
    info != 0, EpetraMultiVecFailure,
    "Belos::MultiVecTraits<std::complex<double>,HYMLS::ComplexVector<MultiVector> >::MvTimesMatAddMv: "
    "HYMLS::ComplexVector<MultiVector>::Multiply() returned a nonzero value info=" << info
    << ".");
  }

template<class MultiVector>
void MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::MvAddMv(
  const std::complex<double> alpha,
  const HYMLS::ComplexVector<MultiVector>& A,
  const std::complex<double> beta,
  const HYMLS::ComplexVector<MultiVector>& B,
  HYMLS::ComplexVector<MultiVector>& mv)
  {
  const int info = mv.Update(alpha, A, beta, B, 0.0);

  TEUCHOS_TEST_FOR_EXCEPTION(info != 0, EpetraMultiVecFailure,
    "Belos::MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::MvAddMv: Call to "
    "update() returned a nonzero value " << info << ".");
  }

template<class MultiVector>
void MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::MvScale(
  HYMLS::ComplexVector<MultiVector>& mv, const std::complex<double> alpha)
  {
  const int info = mv.Scale(alpha);

  TEUCHOS_TEST_FOR_EXCEPTION(info != 0, EpetraMultiVecFailure,
    "Belos::MultiVecTraits<std::complex<double>,HYMLS::ComplexVector<MultiVector> >::MvScale: "
    "HYMLS::ComplexVector<MultiVector>::Scale() returned a nonzero value info="
    << info << ".");
  }

//! For all columns j of  mv, set mv[j] = alpha[j] * mv[j].
template<class MultiVector>
void MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::MvScale(
  HYMLS::ComplexVector<MultiVector> &mv, const std::vector<std::complex<double> > &alpha)
  {
  // Check to make sure the vector has the same number of entries
  // as the multivector has columns.
  const int numvecs = mv.NumVectors();

  TEUCHOS_TEST_FOR_EXCEPTION(
    (int) alpha.size () != numvecs, EpetraMultiVecFailure,
    "Belos::MultiVecTraits<std::complex<double>,HYMLS::ComplexVector<MultiVector> >::MvScale: "
    "Array alpha of scaling coefficients has " << alpha.size ()
    << " entries, which is not the same as the number of columns "
    << numvecs << " in the input multivector mv.");

  int info = 0;
  int startIndex = 0;
  for (int i = 0; i < numvecs; ++i)
    {
    HYMLS::ComplexVector<MultiVector> temp_vec(::View, mv, startIndex, 1);
    info = temp_vec.Scale(alpha[i]);

    TEUCHOS_TEST_FOR_EXCEPTION(info != 0, EpetraMultiVecFailure,
      "Belos::MultiVecTraits<std::complex<double>,HYMLS::ComplexVector<MultiVector> >::MvScale: "
      "On column " << (i+1) << " of " << numvecs << ", Epetra_Multi"
      "Vector::Scale() returned a nonzero value info=" << info << ".");
    startIndex++;
    }
  }

//! B := alpha * A^T * mv.
//! Epetra style
template<class MultiVector>
void MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::MvTransMv(
  const std::complex<double> alpha, const HYMLS::ComplexVector<MultiVector> &A,
  const HYMLS::ComplexVector<MultiVector> &mv, Teuchos::SerialDenseMatrix<int,std::complex<double> > &B)
  {
  // Create MultiVector from SerialDenseMatrix
  Epetra_LocalMap LocalMap(B.numRows(), 0, mv.Comm());
  Epetra_MultiVector B_real(LocalMap, B.numCols());
  Epetra_MultiVector B_imag(LocalMap, B.numCols());

  HYMLS::ComplexVector<Epetra_MultiVector> B_Pvec(View, B_real, B_imag);

  int info = B_Pvec.Multiply('T', 'N', alpha, A, mv, 0.0);

  for (int i = 0; i < B.numRows(); ++i)
    for (int j = 0; j < B.numCols(); ++j)
      B(i, j) = std::complex<double>(B_real[j][i], B_imag[j][i]);

  TEUCHOS_TEST_FOR_EXCEPTION(info != 0, EpetraMultiVecFailure,
    "Belos::MultiVecTraits<std::complex<double>,HYMLS::ComplexVector<MultiVector> >::MvTransMv: "
    "HYMLS::ComplexVector<MultiVector>::Multiply() returned a nonzero value info="
    << info << ".");
  }

//! For all columns j of mv, set b[j] := mv[j]^T * A[j].
template<class MultiVector>
void MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::MvDot(
  const HYMLS::ComplexVector<MultiVector> &mv,
  const HYMLS::ComplexVector<MultiVector> &A,
  std::vector<std::complex<double> > &b)
  {
  const int info = mv.Dot(A, &b[0]);

  TEUCHOS_TEST_FOR_EXCEPTION(info != 0, EpetraMultiVecFailure,
    "Belos::MultiVecTraits<std::complex<double>,HYMLS::ComplexVector<MultiVector> >::MvDot: "
    "HYMLS::ComplexVector<MultiVector>::Dot() returned a nonzero value info="
    << info << ".");
  }

//! For all columns j of mv, set normvec[j] = norm(mv[j]).
template<class MultiVector>
void MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::MvNorm(
  const HYMLS::ComplexVector<MultiVector> &mv,
  std::vector<double> &normvec,
  NormType type)
  {
  if ((int) normvec.size() >= mv.NumVectors())
    {
    int info = 0;
    switch( type )
      {
      case ( OneNorm ) :
        info = mv.Norm1(normvec);
        break;
      case ( TwoNorm ) :
        info = mv.Norm2(normvec);
        break;
      case ( InfNorm ) :
        info = mv.NormInf(normvec);
        break;
      default:
        break;
      }
    TEUCHOS_TEST_FOR_EXCEPTION(info != 0, EpetraMultiVecFailure,
      "Belos::MultiVecTraits<std::complex<double>,HYMLS::ComplexVector<MultiVector> >::MvNorm: "
      "HYMLS::ComplexVector<MultiVector>::Norm() returned a nonzero value info="
      << info << ".");
    }
  }

template<class MultiVector>
void MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::SetBlock(
  const HYMLS::ComplexVector<MultiVector> &A,
  const std::vector<int> &index,
  HYMLS::ComplexVector<MultiVector> &mv)
  {
  const int inNumVecs  = GetNumberVecs(A);
  const int outNumVecs = index.size();

  if (inNumVecs < outNumVecs)
    {
    std::ostringstream os;
    os << "Belos::MultiVecTraits<std::complex<double>,HYMLS::ComplexVector<MultiVector> >::"
      "SetBlock(A, mv, index = {";
    if (outNumVecs > 0)
      {
      for (int k = 0; k < outNumVecs - 1; ++k)
        os << index[k] << ", ";
      os << index[outNumVecs-1];
      }
    os << "}): A has only " << inNumVecs << " columns, but there are "
       << outNumVecs << " indices in the index vector.";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, os.str());
    }

  // Make a view of the columns of mv indicated by the index std::vector.
  Teuchos::RCP<HYMLS::ComplexVector<MultiVector> > mv_view = CloneViewNonConst(mv, index);

  // View of columns [0, outNumVecs-1] of the source multivector A.
  // If A has fewer columns than mv_view, then create a view of
  // the first outNumVecs columns of A.
  Teuchos::RCP<const HYMLS::ComplexVector<MultiVector> > A_view;
  if (outNumVecs == inNumVecs)
    A_view = Teuchos::rcpFromRef(A); // Const, non-owning RCP
  else
    A_view = CloneView(A, Teuchos::Range1D(0, outNumVecs - 1));

  *mv_view = *A_view;
  }

template<class MultiVector>
void MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::SetBlock(
  const HYMLS::ComplexVector<MultiVector> &A,
  const Teuchos::Range1D &index,
  HYMLS::ComplexVector<MultiVector> &mv)
  {
  const int numColsA  = A.NumVectors();
  const int numColsMv = mv.NumVectors();

  // 'index' indexes into mv; it's the index set of the target.
  const bool validIndex = index.lbound() >= 0 && index.ubound() < numColsMv;

  // We can't take more columns out of A than A has.
  const bool validSource = index.size() <= numColsA;

  if (! validIndex || ! validSource)
    {
    std::ostringstream os;
    os << "Belos::MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::SetBlock"
      "(A, index=[" << index.lbound() << ", " << index.ubound() << "], "
      "mv): ";
    TEUCHOS_TEST_FOR_EXCEPTION(index.lbound() < 0, std::invalid_argument,
      os.str() << "Range lower bound must be nonnegative.");
    TEUCHOS_TEST_FOR_EXCEPTION(index.ubound() >= numColsMv, std::invalid_argument,
      os.str() << "Range upper bound must be less than "
      "the number of columns " << numColsA << " in the "
      "'mv' output argument.");
    TEUCHOS_TEST_FOR_EXCEPTION(index.size() > numColsA, std::invalid_argument,
      os.str() << "Range must have no more elements than"
      " the number of columns " << numColsA << " in the "
      "'A' input argument.");
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Should never get here!");
    }

  // View of columns [index.lbound(), index.ubound()] of the
  // target multivector mv.  We avoid view creation overhead by
  // only creating a view if the index range is different than [0,
  // (# columns in mv) - 1].
  Teuchos::RCP<HYMLS::ComplexVector<MultiVector> > mv_view;
  if (index.lbound() == 0 && index.ubound()+1 == numColsMv)
    mv_view = Teuchos::rcpFromRef(mv); // Non-const, non-owning RCP
  else
    mv_view = CloneViewNonConst(mv, index);

  // View of columns [0, index.size()-1] of the source multivector
  // A.  If A has fewer columns than mv_view, then create a view
  // of the first index.size() columns of A.
  Teuchos::RCP<const HYMLS::ComplexVector<MultiVector> > A_view;
  if (index.size() == numColsA)
    A_view = Teuchos::rcpFromRef(A); // Const, non-owning RCP
  else
    A_view = CloneView(A, Teuchos::Range1D(0, index.size()-1));

  *mv_view = *A_view;
  }

template<class MultiVector>
void MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::Assign(
  const HYMLS::ComplexVector<MultiVector>& A,
  HYMLS::ComplexVector<MultiVector>& mv)
  {
  const int numColsA  = GetNumberVecs(A);
  const int numColsMv = GetNumberVecs(mv);
  if (numColsA > numColsMv)
    {
    std::ostringstream os;
    os << "Belos::MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::Assign"
      "(A, mv): ";
    TEUCHOS_TEST_FOR_EXCEPTION(numColsA > numColsMv, std::invalid_argument,
      os.str() << "Input multivector 'A' has "
      << numColsA << " columns, but output multivector "
      "'mv' has only " << numColsMv << " columns.");
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Should never get here!");
    }

  // View of the first [0, numColsA-1] columns of mv.
  Teuchos::RCP<HYMLS::ComplexVector<MultiVector> > mv_view;
  if (numColsMv == numColsA)
    mv_view = Teuchos::rcpFromRef(mv); // Non-const, non-owning RCP
  else // numColsMv > numColsA
    mv_view = CloneView(mv, Teuchos::Range1D(0, numColsA - 1));

  *mv_view = A;
  }

template<class MultiVector>
void MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::MvRandom(HYMLS::ComplexVector<MultiVector>& mv)
  {
  TEUCHOS_TEST_FOR_EXCEPTION( mv.Random()!= 0, EpetraMultiVecFailure,
    "Belos::MultiVecTraits<std::complex<double>,HYMLS::ComplexVector<MultiVector> >::"
    "MvRandom() call to Random() returned a nonzero value.");
  }

template<class MultiVector>
void MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::MvInit(
  HYMLS::ComplexVector<MultiVector>& mv, std::complex<double> alpha)
  {
  TEUCHOS_TEST_FOR_EXCEPTION( mv.PutScalar(alpha) != 0, EpetraMultiVecFailure,
    "Belos::MultiVecTraits<std::complex<double>,HYMLS::ComplexVector<MultiVector> >::"
    "MvInit() call to PutScalar() returned a nonzero value.");
  }

template<class MultiVector>
void MultiVecTraits<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::MvPrint(
  const HYMLS::ComplexVector<MultiVector>& mv, std::ostream& os)
  {
  os << mv << std::endl;
  }

#if TRILINOS_MAJOR_VERSION<12
//! Obtain the vector length of \c mv.
//! \note This method supersedes GetVecLength, which will be deprecated.
template<class MultiVector>
ptrdiff_t MultiVecTraitsExt<std::complex<double>, HYMLS::ComplexVector<MultiVector> >::GetGlobalLength(const HYMLS::ComplexVector<MultiVector>& mv)
  {
  if (mv.Real()->Map().GlobalIndicesLongLong())
    return static_cast<ptrdiff_t>( mv.GlobalLength64() );
  else
    return static_cast<ptrdiff_t>( mv.GlobalLength() );
  }
#endif

  } // namespace Belos

#endif
