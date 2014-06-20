#include "HYMLS_no_debug.H"
#include "HYMLS_ParallelSparseDirectSolver.H"
#include "HYMLS_SolverContainer.H"

#include "Ifpack_Condest.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_Time.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"
#include "Teuchos_ParameterList.hpp"

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "HYMLS_Tools.H"
#include "HYMLS_MatrixUtils.H"
#include "HYMLS_View_MultiVector.H"

#include "Teuchos_StrUtils.hpp"

#include "Ifpack_Partitioner.h"
#include "Ifpack_GreedyPartitioner.h"
#include "Ifpack_LinearPartitioner.h"
#include "Ifpack_METISPartitioner.h"
#include "Ifpack_Graph.h"
#include "Ifpack_Graph_Epetra_RowMatrix.h"

namespace HYMLS {

//==============================================================================
ParallelSparseDirectSolver::ParallelSparseDirectSolver(Epetra_RowMatrix* Matrix_in) :
  Matrix_(Teuchos::rcp( Matrix_in, false )),
  label_("ParallelSparseDirectSolver"), 
  IsEmpty_(false),
  IsInitialized_(false),
  IsComputed_(false),
  UseTranspose_(false),
  NumInitialize_(0),
  NumCompute_(0),
  NumApplyInverse_(0),
  InitializeTime_(0.0),
  ComputeTime_(0.0),
  ApplyInverseTime_(0.0),
  ComputeFlops_(0),
  ApplyInverseFlops_(0),
  Condest_(-1.0)
{
START_TIMER3(label_,"Constructor");
#ifdef HAVE_MPI
  Comm_ = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_SELF));
#else
  Comm_ = Teuchos::rcp(new Epetra_SerialComm);
#endif
}

//==============================================================================
ParallelSparseDirectSolver::ParallelSparseDirectSolver(const ParallelSparseDirectSolver& rhs) :
  Matrix_(Teuchos::rcp( &rhs.Matrix(), false )),
  label_(rhs.Label()),
  IsEmpty_(false),
  IsInitialized_(false),
  IsComputed_(false),
  NumInitialize_(rhs.NumInitialize()),
  NumCompute_(rhs.NumCompute()),
  NumApplyInverse_(rhs.NumApplyInverse()),
  InitializeTime_(rhs.InitializeTime()),
  ComputeTime_(rhs.ComputeTime()),
  ApplyInverseTime_(rhs.ApplyInverseTime()),
  ComputeFlops_(rhs.ComputeFlops()),
  ApplyInverseFlops_(rhs.ApplyInverseFlops()),
  Condest_(rhs.Condest())
{
Tools::Error("not implemented!",__FILE__,__LINE__);
}

ParallelSparseDirectSolver::~ParallelSparseDirectSolver()
  {
  }

//==============================================================================
int ParallelSparseDirectSolver::SetParameters(Teuchos::ParameterList& List_in)
  {
START_TIMER3(label_,"SetParameters");
  List_ = List_in;

  if (A11_Solver_ == Teuchos::null)
    {
    return 0;
    }

  A11_Solver_->SetParameters(List_in);
  A33_Solver_->SetParameters(List_in);

#ifndef DENSE_PARALLEL_SOLVER
  Schur_Solver_->SetParameters(List_in);
#endif

  return 0;
  }

int ParallelSparseDirectSolver::CreateSolvers()
  {
START_TIMER2(label_,"CreateSolvers");
  Teuchos::RCP<Ifpack_Graph> Graph = Teuchos::rcp(new Ifpack_Graph_Epetra_RowMatrix(Matrix_));

  Teuchos::RCP<Ifpack_Partitioner> partitioner = Teuchos::rcp( new Ifpack_GreedyPartitioner(&*Graph));

  Teuchos::ParameterList List;
  List.set("partitioner: local parts", 2);
  List.set("partitioner: overlap", 0);

  CHECK_ZERO(partitioner->SetParameters(List));

  CHECK_ZERO(partitioner->Compute());

  int NumParts = partitioner->NumLocalParts();

  Teuchos::Array<int> firstPart;
  Teuchos::Array<int> secondPart;
  Teuchos::Array<int> schurPart;

  for (int i = 0 ; i < partitioner->NumRowsInPart(1) ; ++i)
    {
    secondPart.push_back((*partitioner)(1, i));
    }

  int numEntries;
  int maxNumEntries = Matrix_->MaxNumEntries();
  Teuchos::Array<int> intersection(maxNumEntries);
  Teuchos::Array<int> indices(maxNumEntries);
  Teuchos::Array<double> values(maxNumEntries);

  // extact the other two parts that weren't computed by the partitioner
  // TODO: Check if there isn't a faster way...
  for (int i = 0; i < Matrix_->NumMyRows(); ++i)
    {
    if (std::find(secondPart.begin(), secondPart.end(), i) != secondPart.end())
      {
      continue;
      }

    indices.resize(maxNumEntries);
    Matrix_->ExtractMyRowCopy(i, maxNumEntries, numEntries, &values[0], &indices[0]);
    indices.resize(numEntries);
    Teuchos::Array<int>::iterator end = std::set_intersection(indices.begin(), indices.end(),
                      secondPart.begin(), secondPart.end(), intersection.begin());
    if (intersection.begin() == end)
      {
      firstPart.push_back(i);
      }
    else
      {
      schurPart.push_back(i);
      }
    }


  A11_Solver_ = Teuchos::rcp(new SolverContainer(SolverContainer::SPARSE, firstPart.length()));
  A33_Solver_ = Teuchos::rcp(new SolverContainer(SolverContainer::SPARSE, secondPart.length()));
#ifdef DENSE_PARALLEL_SOLVER
  Schur_Solver_ = Teuchos::rcp(new Epetra_SerialDenseSolver());
#else
  Schur_Solver_ = Teuchos::rcp(new SolverContainer(SolverContainer::DENSE, schurPart.length()));
#endif

  ID_A11_ = firstPart;
  ID_A33_ = secondPart;
  ID_Schur_ = schurPart;
  
  std::cout << ID_A11_ << std::endl;
  std::cout << ID_A33_ << std::endl;
  std::cout << ID_Schur_ << std::endl;

  int base = Matrix_->RowMatrixRowMap().IndexBase();

  Teuchos::RCP<Epetra_Map> map1 = Teuchos::rcp(new Epetra_Map(firstPart.length(), firstPart.length(), &firstPart[0], base, *Comm_));
  Teuchos::RCP<Epetra_Map> map2 = Teuchos::rcp(new Epetra_Map(schurPart.length(), schurPart.length(), &schurPart[0], base, *Comm_));
  Teuchos::RCP<Epetra_Map> map3 = Teuchos::rcp(new Epetra_Map(secondPart.length(), secondPart.length(), &secondPart[0], base, *Comm_));

  A12_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map1, *map2, maxNumEntries));
  A21_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map2, *map1, maxNumEntries));
  A22_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map2, *map2, maxNumEntries));
  A23_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map2, *map3, maxNumEntries));
  A32_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map3, *map2, maxNumEntries));
#ifdef DENSE_PARALLEL_SOLVER
  Schur_Matrix_ = Teuchos::rcp(new Epetra_SerialDenseMatrix(schurPart.length(), schurPart.length()));
#endif

  CHECK_ZERO(MatrixUtils::ExtractLocalBlock(*Matrix_, *A12_));
  CHECK_ZERO(MatrixUtils::ExtractLocalBlock(*Matrix_, *A21_));
  CHECK_ZERO(MatrixUtils::ExtractLocalBlock(*Matrix_, *A22_));
  CHECK_ZERO(MatrixUtils::ExtractLocalBlock(*Matrix_, *A23_));
  CHECK_ZERO(MatrixUtils::ExtractLocalBlock(*Matrix_, *A32_));

  CHECK_ZERO(A12_->FillComplete(*map2, *map1));
  CHECK_ZERO(A21_->FillComplete(*map1, *map2));
  CHECK_ZERO(A22_->FillComplete(*map2, *map2));
  CHECK_ZERO(A23_->FillComplete(*map3, *map2));
  CHECK_ZERO(A32_->FillComplete(*map2, *map3));
  return 0;
  }

//==============================================================================
int ParallelSparseDirectSolver::Initialize()
  {
START_TIMER2(label_,"Initialize");

  IsEmpty_ = false;
  IsInitialized_ = false;
  IsComputed_ = false;

  if (Matrix_ == Teuchos::null)
    {
    Tools::Error("null matrix",__FILE__,__LINE__);
    }

  // only square matrices
  if (Matrix_->NumGlobalRows() != Matrix_->NumGlobalCols())
    {
    Tools::Error("non-square matrix",__FILE__,__LINE__);
    }

  // if the matrix has a dimension of 0, this is an empty preconditioning object.
  if (Matrix_->NumGlobalRows() == 0) {
    IsEmpty_ = true;
    IsInitialized_ = true;
    ++NumInitialize_;
    return(0);
  }

  // create timer, which also starts it.
  if (Time_ == Teuchos::null)
    Time_ = Teuchos::rcp( new Epetra_Time(Comm()) );

  CreateSolvers();

  CHECK_ZERO(A11_Solver_->Initialize());
  CHECK_ZERO(A33_Solver_->Initialize());
#ifndef DENSE_PARALLEL_SOLVER
  CHECK_ZERO(Schur_Solver_->Initialize());
#endif

  CHECK_ZERO(A11_Solver_->SetParameters(List_));
  CHECK_ZERO(A33_Solver_->SetParameters(List_));
#ifndef DENSE_PARALLEL_SOLVER
  CHECK_ZERO(Schur_Solver_->SetParameters(List_));
#endif

  for (int i = 0; i < ID_A11_.length(); ++i)
    {
    A11_Solver_->ID(i) = ID_A11_[i];
    }

  for (int i = 0; i < ID_A33_.length(); ++i)
    {
    A33_Solver_->ID(i) = ID_A33_[i];
    }

#ifndef DENSE_PARALLEL_SOLVER
  for (int i = 0; i < ID_Schur_.length(); ++i)
    {
    Schur_Solver_->ID(i) = i;
    }
#else
  CHECK_ZERO(Schur_Solver_->SetMatrix(*Schur_Matrix_));
#endif

  IsInitialized_ = true;
  ++NumInitialize_;
  InitializeTime_ += Time_->ElapsedTime();

  return 0;
  }

//==============================================================================
int ParallelSparseDirectSolver::Compute()
{
START_TIMER2(label_,"Compute");
  if (!IsInitialized())
    CHECK_ZERO(Initialize());

  if (IsEmpty_) {
    IsComputed_ = true;
    ++NumCompute_;
    return(0);
  }

  IsComputed_ = false;
  Time_->ResetStartTime();

  CHECK_ZERO(A11_Solver_->Compute(*Matrix_));
  CHECK_ZERO(A33_Solver_->Compute(*Matrix_));

  if (A22_->NumMyRows() > 0)
    {
    Epetra_MultiVector A11_part(A22_->RowMap(), A22_->NumMyCols());
    Epetra_MultiVector A33_part(A22_->RowMap(), A22_->NumMyCols());

#pragma omp parallel
{
    int numEntries;
    double *values;
    int *indices;

#pragma omp task
{
    A11_Solver_->SetNumVectors(A12_->NumMyCols());
    for (int i = 0; i < A12_->NumMyRows(); ++i)
      {
      A12_->ExtractMyRowView(i, numEntries, values, indices);
      for (int j = 0; j < numEntries; ++j)
        {
        A11_Solver_->RHS(i, indices[j]) = values[j];
        }
      }

    CHECK_ZERO(A11_Solver_->ApplyInverse());

    CHECK_ZERO(A21_->Multiply(false, *A11_Solver_->LHS(), A11_part));

    A11_Solver_->SetNumVectors(1);
}
#pragma omp task
{
    A33_Solver_->SetNumVectors(A32_->NumMyCols());
    for (int i = 0; i < A32_->NumMyRows(); ++i)
      {
      CHECK_ZERO(A32_->ExtractMyRowView(i, numEntries, values, indices));
      for (int j = 0; j < numEntries; ++j)
        {
        A33_Solver_->RHS(i, indices[j]) = values[j];
        }
      }

    CHECK_ZERO(A33_Solver_->ApplyInverse());

    CHECK_ZERO(A23_->Multiply(false, *A33_Solver_->LHS(), A33_part));

    A33_Solver_->SetNumVectors(1);
}
#ifdef DENSE_PARALLEL_SOLVER
#pragma omp barrier
    int N = A22_->NumMyRows();
#pragma omp for
    for (int i = 0; i < N; ++i)
      {
      CHECK_ZERO(A22_->ExtractMyRowView(i, numEntries, values, indices));
      for (int j = 0; j < numEntries; ++j)
        {
        (*Schur_Matrix_)[indices[j]][i] = values[j];
        }
      for (int j = 0; j < N; ++j)
        {
        (*Schur_Matrix_)[j][i] -= A11_part[j][i] + A33_part[j][i];
        }
      }
#endif
}

#ifndef DENSE_PARALLEL_SOLVER
    int N = A22_->NumMyRows();
    Schur_Matrix_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy, A22_->RowMap(), A22_->ColMap(), N));

    int *newIndices = new int[N];
    double *newValues = new double[N];

    int numEntries;
    double *values;
    int *indices;

    for (int i = 0; i < N; ++i)
      {
      int idx = 0;
      int idx2 = 0;
      CHECK_ZERO(A22_->ExtractMyRowView(i, numEntries, values, indices));
      for (int j = 0; j < N; ++j)
        {
        double val = -(A11_part[j][i] + A33_part[j][i]);
        if (idx2 < numEntries && indices[idx2] == j)
          {
          val += values[idx2];
          idx2++;
          }
        if (val)
          {
          newValues[idx] = val;
          newIndices[idx] = j;
          idx++;
          }
        }
      CHECK_ZERO(Schur_Matrix_->InsertMyValues(i, idx, newValues, newIndices));
      }
    delete[] newIndices;
    delete[] newValues;

    CHECK_ZERO(Schur_Matrix_->FillComplete());

    CHECK_ZERO(Schur_Solver_->Compute(*Schur_Matrix_));
#else
    CHECK_ZERO(Schur_Solver_->Factor());
#endif
    }

  IsComputed_ = true;
  ++NumCompute_;
  ComputeTime_ += Time_->ElapsedTime();
  return(0);
}

//==============================================================================
int ParallelSparseDirectSolver::SetUseTranspose(bool UseTranspose_in)
{
UseTranspose_ = UseTranspose_in;
return(0);
}

//==============================================================================
int ParallelSparseDirectSolver::
Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  // check for maps? check UseTranspose_?
  return(-99); // not implemented
  CHECK_ZERO(Matrix_->Apply(X,Y));
}

//==============================================================================
int ParallelSparseDirectSolver::
ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  if (IsEmpty_) {
    ++NumApplyInverse_;
    return(0);
  }

  if (IsComputed() == false) 
        {return -1;}

  if (X.NumVectors() != Y.NumVectors())
        {return -2;}

  Time_->ResetStartTime();

  int numVectors = X.NumVectors();

  Epetra_MultiVector z1(A12_->RowMap(), numVectors);
  Epetra_MultiVector z3(A32_->RowMap(), numVectors);
  Epetra_MultiVector p1(A22_->RowMap(), numVectors);
  Epetra_MultiVector p2(A22_->RowMap(), numVectors);
  Epetra_MultiVector p3(A22_->RowMap(), numVectors);

#pragma omp parallel
{
#pragma omp task
{
  A11_Solver_->SetNumVectors(numVectors);
  for (int i = 0; i < A12_->NumMyRows(); ++i)
    {
    for (int j = 0; j < numVectors; ++j)
      {
      A11_Solver_->RHS(i, j) = X[j][ID_A11_[i]];
      }
    }
  CHECK_ZERO(A11_Solver_->ApplyInverse());
  for (int i = 0; i < A12_->NumMyRows(); ++i)
    {
    for (int j = 0; j < numVectors; ++j)
      {
      z1[j][i] = A11_Solver_->LHS(i, j);
      }
    }
  CHECK_ZERO(A21_->Multiply(false, z1, p1));
}
#pragma omp task
{
  A33_Solver_->SetNumVectors(numVectors);
  for (int i = 0; i < A32_->NumMyRows(); ++i)
    {
    for (int j = 0; j < numVectors; ++j)
      {
      A33_Solver_->RHS(i, j) = X[j][ID_A33_[i]];
      }
    }
  CHECK_ZERO(A33_Solver_->ApplyInverse());
  for (int i = 0; i < A32_->NumMyRows(); ++i)
    {
    for (int j = 0; j < numVectors; ++j)
      {
      z3[j][i] = A33_Solver_->LHS(i, j);
      }
    }
  CHECK_ZERO(A23_->Multiply(false, z3, p3));
}
}
  // Why doesn't MultiVector_View work here?
  //~ MultiVector_View view(X.Map(), A22_->RowMap());
  //~ Teuchos::RCP<const Epetra_MultiVector> X2 = view(X);
  for (int i = 0; i < A22_->NumMyRows(); ++i)
    {
    for (int j = 0; j < numVectors; ++j)
      {
      p2[j][i] = X[j][ID_Schur_[i]];
      }
    }

  CHECK_ZERO(p1.Update(1.0, p2, -1.0, p3, -1.0));

  if (A22_->NumMyRows() > 0)
    {
#ifndef DENSE_PARALLEL_SOLVER
    Schur_Solver_->SetNumVectors(numVectors);
#else
    Epetra_SerialDenseMatrix LHS(Schur_Matrix_->N(), numVectors);
    Epetra_SerialDenseMatrix RHS(Schur_Matrix_->M(), numVectors);
    CHECK_ZERO(Schur_Solver_->SetVectors(LHS, RHS));
#endif
    for (int i = 0; i < A22_->NumMyRows(); ++i)
      {
      for (int j = 0; j < numVectors; ++j)
        {
#ifndef DENSE_PARALLEL_SOLVER
        Schur_Solver_->RHS(i, j)= p1[j][i];
#else
        RHS[j][i] = p1[j][i];
#endif
        }
      }
#ifndef DENSE_PARALLEL_SOLVER
    CHECK_ZERO(Schur_Solver_->ApplyInverse());
#else
    CHECK_ZERO(Schur_Solver_->Solve());
#endif
    for (int i = 0; i < A22_->NumMyRows(); ++i)
      {
      for (int j = 0; j < numVectors; ++j)
        {
#ifndef DENSE_PARALLEL_SOLVER
        Y[j][ID_Schur_[i]] = Schur_Solver_->LHS(i, j);
#else
        Y[j][ID_Schur_[i]] = LHS[j][i];
#endif
        }
      }

#ifndef DENSE_PARALLEL_SOLVER
    Teuchos::RCP<const Epetra_MultiVector> Schur_LHS = Schur_Solver_->LHS();
    const Epetra_MultiVector &B = *Schur_LHS;
#else
    Epetra_MultiVector B(Copy, A22_->RowMap(), LHS.A(), LHS.LDA(), LHS.N());
#endif

#pragma omp parallel
{
#pragma omp task
{
    Epetra_MultiVector q1(A12_->RowMap(), numVectors);
    CHECK_ZERO(A12_->Multiply(false, B, q1));

    A11_Solver_->SetNumVectors(numVectors);
    for (int i = 0; i < A12_->NumMyRows(); ++i)
      {
      for (int j = 0; j < numVectors; ++j)
        {
        A11_Solver_->RHS(i, j) = q1[j][i];
        }
      }
    CHECK_ZERO(A11_Solver_->ApplyInverse());

    CHECK_ZERO(z1.Update(-1.0, *A11_Solver_->LHS(), 1.0));
}
#pragma omp task
{
    Epetra_MultiVector q3(A32_->RowMap(), numVectors);
    CHECK_ZERO(A32_->Multiply(false, B, q3));

    A33_Solver_->SetNumVectors(numVectors);
    for (int i = 0; i < A32_->NumMyRows(); ++i)
      {
      for (int j = 0; j < numVectors; ++j)
        {
        A33_Solver_->RHS(i, j) = q3[j][i];
        }
      }
    CHECK_ZERO(A33_Solver_->ApplyInverse());

    CHECK_ZERO(z3.Update(-1.0, *A33_Solver_->LHS(), 1.0));
}
}
    }

#pragma omp parallel
{
#pragma omp task
{
  for (int i = 0; i < A12_->NumMyRows(); ++i)
    {
    for (int j = 0; j < numVectors; ++j)
      {
      Y[j][ID_A11_[i]] = z1[j][i];
      }
    }
  A11_Solver_->SetNumVectors(1);
}
#pragma omp task
{
  for (int i = 0; i < A32_->NumMyRows(); ++i)
    {
    for (int j = 0; j < numVectors; ++j)
      {
      Y[j][ID_A33_[i]] = z3[j][i];
      }
    }
  A33_Solver_->SetNumVectors(1);
}
}

  ++NumApplyInverse_;
  ApplyInverseTime_ += Time_->ElapsedTime();

  return(0);
}

//==============================================================================
double ParallelSparseDirectSolver::NormInf() const
{
  return(-1.0);
}

//==============================================================================
const char* ParallelSparseDirectSolver::Label() const
{
  return((char*)label_.c_str());
}

//==============================================================================
bool ParallelSparseDirectSolver::UseTranspose() const
{
  return(UseTranspose_);
}

//==============================================================================
bool ParallelSparseDirectSolver::HasNormInf() const
{
  return(false);
}

//==============================================================================
const Epetra_Comm & ParallelSparseDirectSolver::Comm() const
{
  return(Matrix_->Comm());
}

//==============================================================================
const Epetra_Map & ParallelSparseDirectSolver::OperatorDomainMap() const
{
  return(Matrix_->OperatorDomainMap());
}

//==============================================================================
const Epetra_Map & ParallelSparseDirectSolver::OperatorRangeMap() const
{
  return(Matrix_->OperatorRangeMap());
}

//==============================================================================
double ParallelSparseDirectSolver::Condest(const Ifpack_CondestType CT,
                              const int MaxIters, const double Tol,
			      Epetra_RowMatrix* Matrix_in)
{

  if (!IsComputed()) // cannot compute right now
    return(-1.0);

  if (Condest_ == -1.0)
    Condest_ = Ifpack_Condest(*this, CT, MaxIters, Tol, Matrix_in);

  return(Condest_);
}

//==============================================================================
std::ostream& ParallelSparseDirectSolver::Print(std::ostream& os) const
{
  if (!Comm().MyPID()) {
    os << endl;
    os << "================================================================================" << endl;
    os << "ParallelSparseDirectSolver: " << Label () << endl << endl;
    os << "Condition number estimate = " << Condest() << endl;
    os << "Global number of rows            = " << Matrix_->NumGlobalRows() << endl;
    os << endl;
    os << "Phase           # calls   Total Time (s)       Total MFlops     MFlops/s" << endl;
    os << "-----           -------   --------------       ------------     --------" << endl;
    os << "Initialize()    "   << std::setw(5) << NumInitialize_ 
       << "  " << std::setw(15) << InitializeTime_ 
       << "              0.0              0.0" << endl;
    os << "Compute()       "   << std::setw(5) << NumCompute_ 
       << "  " << std::setw(15) << ComputeTime_
       << "  " << std::setw(15) << 1.0e-6 * ComputeFlops_;
    if (ComputeTime_ != 0.0) 
      os << "  " << std::setw(15) << 1.0e-6 * ComputeFlops_ / ComputeTime_ << endl;
    else
      os << "  " << std::setw(15) << 0.0 << endl;
    os << "ApplyInverse()  "   << std::setw(5) << NumApplyInverse_ 
       << "  " << std::setw(15) << ApplyInverseTime_
       << "  " << std::setw(15) << 1.0e-6 * ApplyInverseFlops_;
    if (ApplyInverseTime_ != 0.0) 
      os << "  " << std::setw(15) << 1.0e-6 * ApplyInverseFlops_ / ApplyInverseTime_ << endl;
    else
      os << "  " << std::setw(15) << 0.0 << endl;
    os << "================================================================================" << endl;
    os << endl;
  }

  return(os);
}

}//namespace HYMLS
