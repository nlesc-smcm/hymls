#include "HYMLS_BorderedVector.hpp"

#include "HYMLS_Tools.hpp"
#include "HYMLS_DenseUtils.hpp"

#include "Epetra_Comm.h"
#include "Epetra_SerialComm.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_MultiVector.h"
#include "Epetra_LocalMap.h"

#include "BelosEpetraAdapter.hpp"

#include <math.h>
#include <vector>

namespace HYMLS {

BorderedVector::BorderedVector(const Epetra_BlockMap &map1, const Epetra_BlockMap &map2,
  int numVectors, bool zeroOut)
  {
  first_   = Teuchos::rcp(new Epetra_MultiVector(map1, numVectors, zeroOut));
  second_  = Teuchos::rcp(new Epetra_MultiVector(map2, numVectors, zeroOut));
  }

// Copy constructor
BorderedVector::BorderedVector(const BorderedVector &source)
  {
  first_   = Teuchos::rcp(new Epetra_MultiVector(*source.First()));
  second_  = Teuchos::rcp(new Epetra_MultiVector(*source.Second()));
  }

BorderedVector::BorderedVector(Epetra_DataAccess CV, const Epetra_MultiVector &mv1,
  const Epetra_MultiVector &mv2)
  {
  if (mv1.NumVectors() != mv2.NumVectors())
    {
    Tools::Error("Incompatible vectors", __FILE__, __LINE__);
    }

  if (CV == Copy)
    {
    first_  = Teuchos::rcp(new Epetra_MultiVector(mv1));
    second_ = Teuchos::rcp(new Epetra_MultiVector(mv2));
    }
  else
    {
    first_  = Teuchos::rcp(new Epetra_MultiVector(View, mv1.Map(), mv1.Pointers(), mv1.NumVectors()));
    second_ = Teuchos::rcp(new Epetra_MultiVector(View, mv2.Map(), mv2.Pointers(), mv2.NumVectors()));
    }
  }

BorderedVector::BorderedVector(Epetra_DataAccess CV, const Epetra_MultiVector &mv1,
  const Epetra_SerialDenseMatrix &mv2)
  {
  if (mv1.NumVectors() != mv2.N())
    {
    Tools::Error("Incompatible vectors", __FILE__, __LINE__);
    }

  Epetra_SerialComm comm;
  Epetra_LocalMap map(mv2.M(), 0, comm);
  if (CV == Copy)
    {
    first_  = Teuchos::rcp(new Epetra_MultiVector(mv1));
    second_ = Teuchos::rcp(new Epetra_MultiVector(Copy, map, mv2.A(), mv2.LDA(), mv2.N()));
    }
  else
    {
    first_  = Teuchos::rcp(new Epetra_MultiVector(View, mv1.Map(), mv1.Pointers(), mv1.NumVectors()));
    second_ = Teuchos::rcp(new Epetra_MultiVector(View, map, mv2.A(), mv2.LDA(), mv2.N()));
    }
  }

// const
BorderedVector::BorderedVector(Epetra_DataAccess CV, const BorderedVector &source,
  int *indices, int numVectors)
  {
  first_  = Teuchos::rcp
    (new Epetra_MultiVector(CV, *source.First(),  indices, numVectors));
  second_ = Teuchos::rcp
    (new Epetra_MultiVector(CV, *source.Second(), indices, numVectors));
  }

// nonconst
BorderedVector::BorderedVector(Epetra_DataAccess CV, BorderedVector &source,
  int *indices, int numVectors)
  {
  first_  = Teuchos::rcp
    (new Epetra_MultiVector(CV, *source.First(),  indices, numVectors));
  second_ = Teuchos::rcp
    (new Epetra_MultiVector(CV, *source.Second(), indices, numVectors));
  }

// const
BorderedVector::BorderedVector(Epetra_DataAccess CV, const BorderedVector &source,
  int startIndex, int numVectors)
  {
  first_  = Teuchos::rcp
    (new Epetra_MultiVector(CV, *source.First(),  startIndex, numVectors));
  second_ = Teuchos::rcp
    (new Epetra_MultiVector(CV, *source.Second(), startIndex, numVectors));
  }

// nonconst
BorderedVector::BorderedVector(Epetra_DataAccess CV, BorderedVector &source,
  int startIndex, int numVectors)
  {
  first_  = Teuchos::rcp
    (new Epetra_MultiVector(CV, *source.First(),  startIndex, numVectors));
  second_ = Teuchos::rcp
    (new Epetra_MultiVector(CV, *source.Second(), startIndex, numVectors));
  }


BorderedVector::BorderedVector(Epetra_DataAccess CV, const Epetra_BlockMap &map, double *A,
  int myLDA, int numVectors)
  {
  first_  = Teuchos::rcp(
    new Epetra_MultiVector(CV, map, A, myLDA, numVectors));

  Epetra_LocalMap map2(0, 0, map.Comm());
  second_  = Teuchos::rcp(new Epetra_MultiVector(map2, numVectors));
  }

// Assignment operator
BorderedVector &BorderedVector::operator=(const BorderedVector &Source)
  {
  *first_  = *Source.First();
  *second_ = *Source.Second();
  return *this;
  }

// Insertion operator
std::ostream &operator<<(std::ostream &out, const BorderedVector &mv)
  {
  out << *mv.First()  << std::endl;
  out << *mv.Second() << std::endl;
  return out;
  }

// Get the rcpointers
Teuchos::RCP<Epetra_MultiVector> BorderedVector::First()
  {
  return first_;
  }

Teuchos::RCP<Epetra_MultiVector> BorderedVector::Second()
  {
  return second_;
  }

Teuchos::RCP<Epetra_MultiVector> BorderedVector::Vector()
  {
  return first_;
  }

Teuchos::RCP<Epetra_SerialDenseMatrix> BorderedVector::Border()
  {
  if (!second_->ConstantStride())
    Tools::Error("No constant stride!", __FILE__, __LINE__);

  return DenseUtils::CreateView(*second_);
  }

Teuchos::RCP<Epetra_MultiVector> BorderedVector::First() const
  {
  return first_;
  }

Teuchos::RCP<Epetra_MultiVector> BorderedVector::Second() const
  {
  return second_;
  }

Teuchos::RCP<Epetra_MultiVector> BorderedVector::Vector() const
  {
  return first_;
  }

Teuchos::RCP<Epetra_SerialDenseMatrix> BorderedVector::Border() const
  {
  if (!second_->ConstantStride())
    Tools::Error("No constant stride!", __FILE__, __LINE__);

  return DenseUtils::CreateView(*second_);
  }

int BorderedVector::SetBorder(const Epetra_SerialDenseMatrix &mv2)
  {
  if (NumVectors() != mv2.N())
    {
    Tools::Error("Incompatible vectors", __FILE__, __LINE__);
    }

  // second_ = Teuchos::rcp(new Epetra_MultiVector(Copy, map, mv2.A(), mv2.LDA(), mv2.N()));
  Epetra_MultiVector second(Copy, second_->Map(), mv2.A(), mv2.LDA(), mv2.N());
  second_->PutScalar(0.0);
  second_->Update(1.0, second, 0.0);
  return 0;
  }

// Get number of vectors in each multivector
int BorderedVector::NumVectors() const
  {
  if (first_.is_null())
    return 0;

  return first_->NumVectors();
  }

// Get the global length of the combined multivector
int BorderedVector::GlobalLength() const
  {
  return first_->GlobalLength() + second_->GlobalLength();
  }

// Get the global length of the combined multivector
long long BorderedVector::GlobalLength64() const
  {
  return first_->GlobalLength64() + second_->GlobalLength64();
  }

// Get the local length of the combined multivector
int BorderedVector::MyLength() const
  {
  return first_->MyLength() + second_->MyLength();
  }

// Query the stride
bool BorderedVector::ConstantStride() const
  {
  return first_->ConstantStride() && second_->ConstantStride();
  }

bool BorderedVector::DistributedGlobal() const
  {
  return first_->DistributedGlobal();
  }

const Epetra_Comm& BorderedVector::Comm() const
  {
  return first_->Comm();
  }

// this = alpha*A*B + scalarThis*this
int BorderedVector::Multiply(char transA, char transB, double scalarAB,
  const BorderedVector &A, const BorderedVector &B,
  double scalarThis)
  {
  int info = 0;
  if (transA == 'T')
    {
    info =  first_->Multiply(transA, transB, scalarAB, *A.First(), *B.First(), scalarThis);
    info += first_->Multiply(transA, transB, scalarAB, *A.Second(), *B.Second(), 1.0);
    return info;
    }

  info =  first_->Multiply(transA, transB, scalarAB, *A.First(), *B.First(), scalarThis);
  info += second_->Multiply(transA, transB, scalarAB, *A.Second(), *B.First(), scalarThis);

  return info;
  }

// this = scalarA*A + scalarThis*this
int BorderedVector::Update(double scalarA, const BorderedVector &A, double scalarThis)
  {
  int info = 0;
  info =  first_->Update(scalarA,  *A.First(),  scalarThis);
  info += second_->Update(scalarA, *A.Second(), scalarThis);
  return info;
  }

// this = scalarA*A + scalarB*B + scalarThis*this
int BorderedVector::Update(double scalarA, const BorderedVector &A,
  double scalarB, const BorderedVector &B, double scalarThis)
  {
  int info = 0;
  info =  first_->Update(scalarA,  *A.First(),  scalarB, *B.First(),  scalarThis);
  info += second_->Update(scalarA, *A.Second(), scalarB, *B.Second(), scalarThis);
  return info;
  }

// b[j] := this[j]^T * A[j]
int BorderedVector::Dot(const BorderedVector& A, std::vector<double> &b1) const
  {
  // we need two arrays storing results
  std::vector<double> b2 = b1;

  int info = 0;
  info += first_->Dot(*A.First(), &b1[0]);
  info += second_->Dot(*A.Second(), &b2[0]);

  // combine the results
  for (int i = 0; i != NumVectors(); ++i)
    b1[i] += b2[i];

  return info;
  }

// result[j] := this[j]^T * A[j]
int BorderedVector::Dot(const BorderedVector& A, double *result) const
  {
  std::vector<double> tmp(NumVectors(), 0.0);
  int info = Dot(A, tmp);
  for (int i = 0; i != NumVectors(); ++i)
    result[i] = tmp[i];
  return info;
  }

int BorderedVector::Scale(double scalarValue)
  {
  int info = 0;
  info =  first_->Scale(scalarValue);
  info += second_->Scale(scalarValue);
  return info;
  }

int BorderedVector::Norm1(std::vector<double> &result) const
  {
  // copy result vector
  std::vector<double> result_tmp = result;

  int info = 0;
  info =  first_->Norm1(&result[0]);
  info += second_->Norm1(&result_tmp[0]);

  // combine results
  for (int i = 0; i != NumVectors(); ++i)
    result[i] += result_tmp[i];
  return info;
  }

int BorderedVector::Norm2(double *result) const
  {
  std::vector<double> tmp(NumVectors(), 0.0);
  int info = Norm2(tmp);
  for (int i = 0; i != NumVectors(); ++i)
    result[i] = tmp[i];
  return info;
  }

int BorderedVector::Norm2(std::vector<double> &result) const
  {
  // copy result vector
  std::vector<double> result_tmp = result;

  int info = 0;
  info =  first_->Norm2(&result[0]);
  info += second_->Norm2(&result_tmp[0]);

  // combine results
  for (int i = 0; i != NumVectors(); ++i)
    result[i] = sqrt(pow(result[i],2) + pow(result_tmp[i],2));

  return info;
  }

int BorderedVector::NormInf(std::vector<double> &result) const
  {
  std::vector<double> result_tmp = result;

  int info = 0;
  info =  first_->NormInf(&result[0]);
  info += second_->NormInf(&result_tmp[0]);

  // combine results
  for (int i = 0; i != NumVectors(); ++i)
    result[i] = std::max(result[i], result_tmp[i]);

  return info;
  }

int BorderedVector::Random()
  {
  int info = 0;
  info += first_->Random();

  if (!second_->ConstantStride())
    Tools::Error("No constant stride!", __FILE__, __LINE__);

  // Broadcast random numbers from processor 0 so they are the same on every processor.
  info += second_->Random();
  info += first_->Comm().Broadcast(second_->Values(), second_->Stride() * second_->NumVectors(), 0);

  return info;
  }

int BorderedVector::PutScalar(double alpha)
  {
  int info = 0;
  info += first_->PutScalar(alpha);
  info += second_->PutScalar(alpha);
  return info;
  }

void BorderedVector::Print(std::ostream &os) const
  {
  first_->Print(os);
  second_->Print(os);
  }

  } // namespace HYMLS


namespace Belos
  {

Teuchos::RCP<HYMLS::BorderedVector>
MultiVecTraits<double, HYMLS::BorderedVector>::Clone(
  const HYMLS::BorderedVector &mv, const int numVecs)
  {
  TEUCHOS_TEST_FOR_EXCEPTION(
    numVecs <= 0, std::invalid_argument,
    "Belos::MultiVecTraits<double, HYMLS::BorderedVector>::"
    "Clone(mv, numVecs = " << numVecs << "): "
    "outNumVecs must be positive.");

  return Teuchos::rcp(
    new HYMLS::BorderedVector(mv.First()->Map(),
      mv.Second()->Map(), numVecs, false));
  }

Teuchos::RCP<HYMLS::BorderedVector>
MultiVecTraits<double, HYMLS::BorderedVector>::CloneCopy(
  const HYMLS::BorderedVector &mv)
  {
  return Teuchos::rcp(new HYMLS::BorderedVector(mv));
  }

Teuchos::RCP<HYMLS::BorderedVector>
MultiVecTraits<double, HYMLS::BorderedVector>::CloneCopy(
  const HYMLS::BorderedVector &mv, const std::vector<int> &index)
  {
  const int inNumVecs  = mv.NumVectors();
  const int outNumVecs = index.size();
  TEUCHOS_TEST_FOR_EXCEPTION(outNumVecs == 0, std::invalid_argument,
    "Belos::MultiVecTraits<double, HYMLS::BorderedVector>::"
    "CloneCopy(mv, index = {}): At least one vector must be"
    " cloned from mv.");

  if (outNumVecs > inNumVecs)
    {
    std::ostringstream os;
    os << "Belos::MultiVecTraits<double, Combined_Operator>::"
      "CloneCopy(mv, index = {";
    for (int k = 0; k < outNumVecs - 1; ++k)
      os << index[k] << ", ";
    os << index[outNumVecs-1] << "}): There are " << outNumVecs
       << " indices to copy, but only " << inNumVecs << " columns of mv.";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, os.str());
    }

  std::vector<int> &tmpInd = const_cast< std::vector<int>& >(index);
  return Teuchos::rcp(new HYMLS::BorderedVector(Copy, mv, &tmpInd[0], outNumVecs));
  }

Teuchos::RCP<HYMLS::BorderedVector>
MultiVecTraits<double, HYMLS::BorderedVector>::CloneCopy(
  const HYMLS::BorderedVector &mv, const Teuchos::Range1D &index)
  {
  const int inNumVecs   = mv.NumVectors();
  const int outNumVecs  = index.size();
  const bool validRange = outNumVecs > 0 && index.lbound() >= 0 &&
    index.ubound() < inNumVecs;

  if (! validRange)
    {
    std::ostringstream os;
    os << "Belos::MultiVecTraits<double, HYMLS::BorderedVector>::Clone(mv,"
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
    new HYMLS::BorderedVector(Copy, mv, index.lbound(), index.size()));
  }

Teuchos::RCP<HYMLS::BorderedVector>
MultiVecTraits<double, HYMLS::BorderedVector>::CloneViewNonConst(
  HYMLS::BorderedVector &mv, const std::vector<int> &index)
  {
  const int inNumVecs  = mv.NumVectors();
  const int outNumVecs = index.size();
  // Simple, inexpensive tests of the index vector.

  TEUCHOS_TEST_FOR_EXCEPTION(outNumVecs == 0, std::invalid_argument,
    "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::"
    "CloneViewNonConst(mv, index = {}): The output view "
    "must have at least one column.");
  if (outNumVecs > inNumVecs)
    {
    std::ostringstream os;
    os << "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::"
      "CloneViewNonConst(mv, index = {";
    for (int k = 0; k < outNumVecs - 1; ++k)
      os << index[k] << ", ";
    os << index[outNumVecs-1] << "}): There are " << outNumVecs
       << " indices to view, but only " << inNumVecs << " columns of mv.";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, os.str());
    }
  
  std::vector<int> &tmpInd = const_cast< std::vector<int>& >(index);
  return Teuchos::rcp(new HYMLS::BorderedVector(View, mv, &tmpInd[0], outNumVecs));
  }

Teuchos::RCP<HYMLS::BorderedVector>
MultiVecTraits<double, HYMLS::BorderedVector>::CloneViewNonConst(
  HYMLS::BorderedVector& mv, const Teuchos::Range1D& index)
  {
  const bool validRange = index.size() > 0 &&
    index.lbound() >= 0 &&
    index.ubound() < mv.NumVectors();

  if (! validRange)
    {
    std::ostringstream os;
    os << "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::CloneView"
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
    new HYMLS::BorderedVector(View, mv, index.lbound(), index.size()));
  }

Teuchos::RCP<const HYMLS::BorderedVector>
MultiVecTraits<double, HYMLS::BorderedVector>::CloneView(
  const HYMLS::BorderedVector& mv, const std::vector<int>& index)
  {
  const int inNumVecs  = mv.NumVectors();
  const int outNumVecs = index.size();

  // Simple, inexpensive tests of the index vector.
  TEUCHOS_TEST_FOR_EXCEPTION(outNumVecs == 0, std::invalid_argument,
    "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::"
    "CloneView(mv, index = {}): The output view "
    "must have at least one column.");
  if (outNumVecs > inNumVecs)
    {
    std::ostringstream os;
    os << "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::"
      "CloneView(mv, index = {";
    for (int k = 0; k < outNumVecs - 1; ++k)
      os << index[k] << ", ";
    os << index[outNumVecs-1] << "}): There are " << outNumVecs
       << " indices to view, but only " << inNumVecs << " columns of mv.";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, os.str());
    }

  std::vector<int> &tmpInd = const_cast< std::vector<int>& >(index);
  return Teuchos::rcp(new HYMLS::BorderedVector(Copy, mv, &tmpInd[0], outNumVecs));
  }

Teuchos::RCP<HYMLS::BorderedVector>
MultiVecTraits<double, HYMLS::BorderedVector>::CloneView(
  const HYMLS::BorderedVector &mv, const Teuchos::Range1D &index)
  {
  const bool validRange = index.size() > 0 &&
    index.lbound() >= 0 &&
    index.ubound() < mv.NumVectors();
  if (! validRange)
    {
    std::ostringstream os;
    os << "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::CloneView"
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
  return Teuchos::rcp(new HYMLS::BorderedVector(View, mv, index.lbound(), index.size()));
  }

int MultiVecTraits<double, HYMLS::BorderedVector>::GetVecLength(
  const HYMLS::BorderedVector& mv)
  {
  return mv.GlobalLength();
  }

int MultiVecTraits<double, HYMLS::BorderedVector>::GetNumberVecs(
  const HYMLS::BorderedVector& mv)
  {
  return mv.NumVectors();
  }

bool MultiVecTraits<double, HYMLS::BorderedVector>::HasConstantStride(
  const HYMLS::BorderedVector& mv)
  {
  return mv.ConstantStride();
  }

ptrdiff_t MultiVecTraits<double, HYMLS::BorderedVector>::GetGlobalLength(
  const HYMLS::BorderedVector& mv)
  {
  if ( mv.First()->Map().GlobalIndicesLongLong() )
    return static_cast<ptrdiff_t>( mv.GlobalLength64() );
  else
    return static_cast<ptrdiff_t>( mv.GlobalLength() );
  }

// Epetra style (we should compare this with just a bunch of updates)
void MultiVecTraits<double, HYMLS::BorderedVector>::MvTimesMatAddMv(
  const double alpha,
  const HYMLS::BorderedVector& A,
  const Teuchos::SerialDenseMatrix<int,double>& B,
  const double beta,
  HYMLS::BorderedVector& mv)
  {
  // Create Epetra_Multivector from SerialDenseMatrix
  Epetra_LocalMap LocalMap(B.numRows(), 0, mv.Second()->Map().Comm());
  Epetra_MultiVector Pvec(View, LocalMap, B.values(), B.stride(), B.numCols());
  HYMLS::BorderedVector B_Pvec(View, Pvec, Pvec);

  const int info = mv.Multiply('N', 'N', alpha, A, B_Pvec, beta);

  TEUCHOS_TEST_FOR_EXCEPTION(
    info != 0, EpetraMultiVecFailure,
    "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::MvTimesMatAddMv: "
    "HYMLS::BorderedVector::Multiply() returned a nonzero value info=" << info
    << ".");
  }

void MultiVecTraits<double, HYMLS::BorderedVector>::MvAddMv(
  const double alpha,
  const HYMLS::BorderedVector& A,
  const double beta,
  const HYMLS::BorderedVector& B,
  HYMLS::BorderedVector& mv)
  {
  const int info = mv.Update(alpha, A, beta, B, 0.0);

  TEUCHOS_TEST_FOR_EXCEPTION(info != 0, EpetraMultiVecFailure,
    "Belos::MultiVecTraits<double, HYMLS::BorderedVector>::MvAddMv: Call to "
    "update() returned a nonzero value " << info << ".");
  }

void MultiVecTraits<double, HYMLS::BorderedVector>::MvScale(
  HYMLS::BorderedVector& mv, const double alpha)
  {
  const int info = mv.Scale(alpha);

  TEUCHOS_TEST_FOR_EXCEPTION(info != 0, EpetraMultiVecFailure,
    "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::MvScale: "
    "HYMLS::BorderedVector::Scale() returned a nonzero value info="
    << info << ".");
  }

//! For all columns j of  mv, set mv[j] = alpha[j] * mv[j].
void MultiVecTraits<double, HYMLS::BorderedVector>::MvScale(
  HYMLS::BorderedVector &mv, const std::vector<double> &alpha)
  {
  // Check to make sure the vector has the same number of entries
  // as the multivector has columns.
  const int numvecs = mv.NumVectors();

  TEUCHOS_TEST_FOR_EXCEPTION(
    (int) alpha.size () != numvecs, EpetraMultiVecFailure,
    "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::MvScale: "
    "Array alpha of scaling coefficients has " << alpha.size ()
    << " entries, which is not the same as the number of columns "
    << numvecs << " in the input multivector mv.");

  int info = 0;
  int startIndex = 0;
  for (int i = 0; i < numvecs; ++i)
    {
    HYMLS::BorderedVector temp_vec(::View, mv, startIndex, 1);
    info = temp_vec.Scale(alpha[i]);

    TEUCHOS_TEST_FOR_EXCEPTION(info != 0, EpetraMultiVecFailure,
      "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::MvScale: "
      "On column " << (i+1) << " of " << numvecs << ", Epetra_Multi"
      "Vector::Scale() returned a nonzero value info=" << info << ".");
    startIndex++;
    }
  }

//! B := alpha * A^T * mv.
//! Epetra style
void MultiVecTraits<double, HYMLS::BorderedVector>::MvTransMv(
  const double alpha, const HYMLS::BorderedVector &A,
  const HYMLS::BorderedVector &mv, Teuchos::SerialDenseMatrix<int,double> &B)
  {
  // Create Epetra_MultiVector from SerialDenseMatrix
  Epetra_LocalMap LocalMap(B.numRows(), 0, mv.First()->Map().Comm());
  Epetra_MultiVector Pvec(View, LocalMap, B.values(), B.stride(), B.numCols());
  HYMLS::BorderedVector B_Pvec(View, Pvec, Pvec);

  const int info = B_Pvec.Multiply('T', 'N', alpha, A, mv, 0.0);

  TEUCHOS_TEST_FOR_EXCEPTION(info != 0, EpetraMultiVecFailure,
    "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::MvTransMv: "
    "HYMLS::BorderedVector::Multiply() returned a nonzero value info="
    << info << ".");
  }

//! For all columns j of mv, set b[j] := mv[j]^T * A[j].
void MultiVecTraits<double, HYMLS::BorderedVector>::MvDot(
  const HYMLS::BorderedVector &mv,
  const HYMLS::BorderedVector &A,
  std::vector<double> &b)
  {
  const int info = mv.Dot(A, b);

  TEUCHOS_TEST_FOR_EXCEPTION(info != 0, EpetraMultiVecFailure,
    "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::MvDot: "
    "HYMLS::BorderedVector::Dot() returned a nonzero value info="
    << info << ".");
  }

//! For all columns j of mv, set normvec[j] = norm(mv[j]).
void MultiVecTraits<double, HYMLS::BorderedVector>::MvNorm(
  const HYMLS::BorderedVector &mv,
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
      "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::MvNorm: "
      "HYMLS::BorderedVector::Norm() returned a nonzero value info="
      << info << ".");
    }
  }

void MultiVecTraits<double, HYMLS::BorderedVector>::SetBlock(
  const HYMLS::BorderedVector &A,
  const std::vector<int> &index,
  HYMLS::BorderedVector &mv)
  {
  const int inNumVecs  = GetNumberVecs(A);
  const int outNumVecs = index.size();

  if (inNumVecs < outNumVecs)
    {
    std::ostringstream os;
    os << "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::"
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
  Teuchos::RCP<HYMLS::BorderedVector> mv_view = CloneViewNonConst(mv, index);

  // View of columns [0, outNumVecs-1] of the source multivector A.
  // If A has fewer columns than mv_view, then create a view of
  // the first outNumVecs columns of A.
  Teuchos::RCP<const HYMLS::BorderedVector> A_view;
  if (outNumVecs == inNumVecs)
    A_view = Teuchos::rcpFromRef(A); // Const, non-owning RCP
  else
    A_view = CloneView(A, Teuchos::Range1D(0, outNumVecs - 1));

  *mv_view = *A_view;
  }

void MultiVecTraits<double, HYMLS::BorderedVector>::SetBlock(
  const HYMLS::BorderedVector &A,
  const Teuchos::Range1D &index,
  HYMLS::BorderedVector &mv)
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
    os << "Belos::MultiVecTraits<double, HYMLS::BorderedVector>::SetBlock"
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
  Teuchos::RCP<HYMLS::BorderedVector> mv_view;
  if (index.lbound() == 0 && index.ubound()+1 == numColsMv)
    mv_view = Teuchos::rcpFromRef(mv); // Non-const, non-owning RCP
  else
    mv_view = CloneViewNonConst(mv, index);

  // View of columns [0, index.size()-1] of the source multivector
  // A.  If A has fewer columns than mv_view, then create a view
  // of the first index.size() columns of A.
  Teuchos::RCP<const HYMLS::BorderedVector> A_view;
  if (index.size() == numColsA)
    A_view = Teuchos::rcpFromRef(A); // Const, non-owning RCP
  else
    A_view = CloneView(A, Teuchos::Range1D(0, index.size()-1));

  *mv_view = *A_view;
  }

void MultiVecTraits<double, HYMLS::BorderedVector>::Assign(
  const HYMLS::BorderedVector& A,
  HYMLS::BorderedVector& mv)
  {
  const int numColsA  = GetNumberVecs(A);
  const int numColsMv = GetNumberVecs(mv);
  if (numColsA > numColsMv)
    {
    std::ostringstream os;
    os << "Belos::MultiVecTraits<double, HYMLS::BorderedVector>::Assign"
      "(A, mv): ";
    TEUCHOS_TEST_FOR_EXCEPTION(numColsA > numColsMv, std::invalid_argument,
      os.str() << "Input multivector 'A' has "
      << numColsA << " columns, but output multivector "
      "'mv' has only " << numColsMv << " columns.");
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Should never get here!");
    }

  // View of the first [0, numColsA-1] columns of mv.
  Teuchos::RCP<HYMLS::BorderedVector> mv_view;
  if (numColsMv == numColsA)
    mv_view = Teuchos::rcpFromRef(mv); // Non-const, non-owning RCP
  else // numColsMv > numColsA
    mv_view = CloneView(mv, Teuchos::Range1D(0, numColsA - 1));

  *mv_view = A;
  }

void MultiVecTraits<double, HYMLS::BorderedVector>::MvRandom(HYMLS::BorderedVector& mv)
  {
  TEUCHOS_TEST_FOR_EXCEPTION( mv.Random()!= 0, EpetraMultiVecFailure,
    "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::"
    "MvRandom() call to Random() returned a nonzero value.");
  }

void MultiVecTraits<double, HYMLS::BorderedVector>::MvInit(
  HYMLS::BorderedVector& mv, double alpha)
  {
  TEUCHOS_TEST_FOR_EXCEPTION( mv.PutScalar(alpha) != 0, EpetraMultiVecFailure,
    "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::"
    "MvInit() call to PutScalar() returned a nonzero value.");
  }

void MultiVecTraits<double, HYMLS::BorderedVector>::MvPrint(
  const HYMLS::BorderedVector& mv, std::ostream& os)
  {
  os << mv << std::endl;
  }

#if TRILINOS_MAJOR_VERSION<12
//! Obtain the vector length of \c mv.
//! \note This method supersedes GetVecLength, which will be deprecated.
ptrdiff_t MultiVecTraitsExt<double, HYMLS::BorderedVector>::GetGlobalLength(const HYMLS::BorderedVector& mv)
  {
  if (mv.First()->Map().GlobalIndicesLongLong())
    return static_cast<ptrdiff_t>( mv.GlobalLength64() );
  else
    return static_cast<ptrdiff_t>( mv.GlobalLength() );
  }
#endif

  } // namespace Belos
