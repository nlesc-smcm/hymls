#include "HYMLS_BorderedVector.hpp"

#include "HYMLS_Tools.hpp"

#include "Epetra_Comm.h"
#include "Epetra_SerialDenseMatrix.h"

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

BorderedVector::BorderedVector(const Teuchos::RCP<Epetra_MultiVector> &mv1,
  const Teuchos::RCP<Epetra_MultiVector> &mv2)
  :
  first_(mv1),
  second_(mv2)
  {
  if (mv1->NumVectors() != mv2->NumVectors())
    {
    Tools::Error("Incompatible vectors", __FILE__, __LINE__);
    }
  }

BorderedVector::BorderedVector(const Teuchos::RCP<Epetra_MultiVector> &mv1,
  const Teuchos::RCP<Epetra_SerialDenseMatrix> &mv2)
  :
  first_(mv1)
  {
  if (mv1->NumVectors() != mv2->N())
    {
    Tools::Error("Incompatible vectors", __FILE__, __LINE__);
    }

  // This seems like a hack but is how it is done in the preconditioner
  int num = 0;
  if (first_->Comm().MyPID() == first_->Comm().NumProc()-1)
    {
    num = mv2->M();
    }

  Epetra_Map map((hymls_gidx)mv2->M(), num, (hymls_gidx)0, first_->Comm());
  second_ = Teuchos::rcp(new Epetra_MultiVector(View, map, mv2->A(), mv2->LDA(), mv2->N()));
  }

BorderedVector::BorderedVector(const Epetra_MultiVector &mv1, const Epetra_MultiVector &mv2)
  {
  if (mv1.NumVectors() != mv2.NumVectors())
    {
    Tools::Error("Incompatible vectors", __FILE__, __LINE__);
    }

  first_  = Teuchos::rcp(new Epetra_MultiVector(mv1));
  second_ = Teuchos::rcp(new Epetra_MultiVector(mv2));
  }

BorderedVector::BorderedVector(const Epetra_MultiVector &mv1,
  const Epetra_SerialDenseMatrix &mv2)
  {
  if (mv1.NumVectors() != mv2.N())
    {
    Tools::Error("Incompatible vectors", __FILE__, __LINE__);
    }

  first_  = Teuchos::rcp(new Epetra_MultiVector(mv1));

  // This seems like a hack but is how it is done in the preconditioner
  int num = 0;
  if (first_->Comm().MyPID() == first_->Comm().NumProc()-1)
    {
    num = mv2.M();
    }

  Epetra_Map map((hymls_gidx)mv2.M(), num, (hymls_gidx)0, first_->Comm());
  second_ = Teuchos::rcp(new Epetra_MultiVector(Copy, map, mv2.A(), mv2.LDA(), mv2.N()));
  }

// const
BorderedVector::BorderedVector(Epetra_DataAccess CV, const BorderedVector &source,
  const std::vector<int> &index)
  {
  // cast to nonconst for Epetra_MultiVector
  std::vector<int> &tmpInd = const_cast< std::vector<int>& >(index);
  first_  = Teuchos::rcp
    (new Epetra_MultiVector(CV, *source.First(),  &tmpInd[0], index.size()));
  second_ = Teuchos::rcp
    (new Epetra_MultiVector(CV, *source.Second(), &tmpInd[0], index.size()));
  }

// nonconst
BorderedVector::BorderedVector(Epetra_DataAccess CV, BorderedVector &source,
  const std::vector<int> &index)
  {
  // cast to nonconst for Epetra_MultiVector
  std::vector<int> &tmpInd = const_cast< std::vector<int>& >(index);
  first_  = Teuchos::rcp
    (new Epetra_MultiVector(CV, *source.First(),  &tmpInd[0], index.size()));
  second_ = Teuchos::rcp
    (new Epetra_MultiVector(CV, *source.Second(), &tmpInd[0], index.size()));
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

  if (first_->Comm().MyPID() == first_->Comm().NumProc()-1)
    {
    return Teuchos::rcp(new
      Epetra_SerialDenseMatrix(View, second_->Values(),
        second_->Stride(), second_->MyLength(), second_->NumVectors()));
    }
  else
    {
    return Teuchos::rcp(new
      Epetra_SerialDenseMatrix(second_->GlobalLength64(), second_->NumVectors()));
    }
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

  if (first_->Comm().MyPID() == first_->Comm().NumProc()-1)
    {
    return Teuchos::rcp(new
      Epetra_SerialDenseMatrix(View, second_->Values(),
        second_->Stride(), second_->MyLength(), second_->NumVectors()));
    }
  else
    {
    return Teuchos::rcp(new
      Epetra_SerialDenseMatrix(second_->GlobalLength64(), second_->NumVectors()));
    }
  }

int BorderedVector::SetBorder(const Epetra_SerialDenseMatrix &mv2)
  {
  if (first_->NumVectors() != mv2.N())
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
int BorderedVector::NumVecs() const
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

// this = alpha*A*B + scalarThis*this
int BorderedVector::Multiply(char transA, char transB, double scalarAB,
  const BorderedVector &A, const Epetra_MultiVector &B,
  double scalarThis)
  {
  int info = 0;
  info =  first_->Multiply(transA, transB, scalarAB, *A.First(), B, scalarThis);
  info += second_->Multiply(transA, transB, scalarAB, *A.Second(), B, scalarThis);
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
  for (int i = 0; i != first_->NumVectors(); ++i)
    b1[i] += b2[i];

  return info;
  }

// result[j] := this[j]^T * A[j]
int BorderedVector::Dot(const BorderedVector& A, double *result) const
  {
  std::vector<double> tmp(first_->NumVectors(), 0.0);
  int info = Dot(A, tmp);
  for (int i = 0; i != first_->NumVectors(); ++i)
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
  for (int i = 0; i != first_->NumVectors(); ++i)
    result[i] += result_tmp[i];
  return info;
  }

int BorderedVector::Norm2(double *result) const
  {
  std::vector<double> tmp(first_->NumVectors(), 0.0);
  int info = Norm2(tmp);
  for (int i = 0; i != first_->NumVectors(); ++i)
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
  for (int i = 0; i != first_->NumVectors(); ++i)
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
  for (int i = 0; i != first_->NumVectors(); ++i)
    result[i] = std::max(result[i], result_tmp[i]);

  return info;
  }

int BorderedVector::Random()
  {
  int info = 0;
  info += first_->Random();
  info += second_->Random();
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

}
