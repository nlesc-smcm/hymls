#include "HYMLS_Preconditioner.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Epetra_SerialComm.h>
#include <Epetra_MpiComm.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Util.h>
#include <Epetra_Import.h>
#include <Epetra_SerialDenseMatrix.h>

#include "HYMLS_Macros.hpp"
#include "HYMLS_DenseUtils.hpp"
#include "HYMLS_MatrixBlock.hpp"
#include "HYMLS_SchurComplement.hpp"
#include "HYMLS_CartesianPartitioner.hpp"
#include "HYMLS_SkewCartesianPartitioner.hpp"

#include "Galeri_CrsMatrices.h"
#include "GaleriExt_CrsMatrices.h"

#include "HYMLS_FakeComm.hpp"
#include "HYMLS_UnitTests.hpp"

class TestableSchurComplement: public HYMLS::SchurComplement
  {
public:
    TestableSchurComplement(HYMLS::SchurComplement SC)
    :
      HYMLS::SchurComplement(SC)
    {}

  Epetra_CrsMatrix const &A22()
    {
    return *A22_->Block();
    }
  };
class TestablePreconditioner: public HYMLS::Preconditioner
  {
public:
  TestablePreconditioner(Teuchos::RCP<Epetra_CrsMatrix> A,
    Teuchos::RCP<Teuchos::ParameterList> params)
    :
    HYMLS::Preconditioner(A, params)
    {}

  Epetra_CrsMatrix const &A22()
    {
    return *A22_->Block();
    }

  HYMLS::SchurComplement const &SchurComplement()
    {
    return *Schur_;
    }

  Teuchos::RCP<const Epetra_MultiVector> V()
    {
    return V_;
    }
  };

Teuchos::RCP<TestablePreconditioner> createPreconditioner(
  Teuchos::RCP<Teuchos::ParameterList> &params,
  Teuchos::RCP<Epetra_Comm> const &comm)
  {
  Teuchos::ParameterList &problemList = params->sublist("Problem");
  problemList.set("Degrees of Freedom", 4);
  problemList.set("Dimension", 3);
  problemList.set("nx", 8);
  problemList.set("ny", 4);
  problemList.set("nz", 4);

  Teuchos::ParameterList &precList = params->sublist("Preconditioner");
  precList.set("Separator Length", 4);
  precList.set("Number of Levels", 0);

  HYMLS::CartesianPartitioner part(Teuchos::null, params, *comm);
  part.Partition(true);

  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, part.Map(), 2));

  Epetra_Util util;
  for (hymls_gidx i = 0; i < A->NumGlobalRows64(); i++) {
    // int A_idx = util.RandomInt() % n;
    // double A_val = -std::abs(util.RandomDouble());
    double A_val2 = std::abs(util.RandomDouble());

    // Check if we own the index
    if (A->LRID(i) == -1)
      continue;

    // CHECK_ZERO(A->InsertGlobalValues(i, 1, &A_val, &A_idx));
    CHECK_ZERO(A->InsertGlobalValues(i, 1, &A_val2, &i));
  }
  CHECK_ZERO(A->FillComplete());

  Teuchos::RCP<TestablePreconditioner> prec =
    Teuchos::rcp(new TestablePreconditioner(A, params));
  return prec;
  }

TEUCHOS_UNIT_TEST(Preconditioner, Blocks)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::RCP<TestablePreconditioner> prec = createPreconditioner(params, comm);
  prec->Initialize();
  prec->Compute();

  Teuchos::RCP<Epetra_CrsMatrix> B = Teuchos::rcp(new Epetra_CrsMatrix(
      dynamic_cast<Epetra_CrsMatrix const&>(prec->Matrix())));
  prec->SetMatrix(B);
  prec->Initialize();
  prec->Compute();

  // Make sure the pointers on the preconditioner and Schur complement are the same
  TEST_EQUALITY(&prec->A22(), &TestableSchurComplement(prec->SchurComplement()).A22());
  }

TEUCHOS_UNIT_TEST(Preconditioner, SerialComm)
  {
  // Check if HYMLS can work without MPI
  Teuchos::RCP<Epetra_SerialComm> comm = Teuchos::rcp(new Epetra_SerialComm);
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::RCP<TestablePreconditioner> prec = createPreconditioner(params, comm);
  prec->Initialize();
  prec->Compute();
  }

TEUCHOS_UNIT_TEST(Preconditioner, SetBorder)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::RCP<TestablePreconditioner> prec = createPreconditioner(params, comm);
  prec->Initialize();
  prec->Compute();

  Epetra_Map const &map = prec->OperatorRangeMap();
  Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(new Epetra_MultiVector(map, 1));
  V->Random();

  prec->SetBorder(V);

  // Import the vector since the matrix might have a different map
  Teuchos::RCP<const Epetra_MultiVector> precV = prec->V();
  Epetra_MultiVector importedV(map, 1);
  Epetra_Import importer(map, precV->Map());
  importedV.Import(*precV, importer, Insert);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(importedV, *V), <, 1e-12);
  }

TEUCHOS_UNIT_TEST(Preconditioner, SetBorderNull)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::RCP<TestablePreconditioner> prec = createPreconditioner(params, comm);
  prec->Initialize();
  prec->Compute();

  // Set the border to null
  prec->SetBorder(Teuchos::null);
  TEST_EQUALITY(prec->V(), Teuchos::null);

  // Set the border to something else
  Epetra_Map const &map = prec->OperatorRangeMap();
  Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(new Epetra_MultiVector(map, 1));
  V->Random();

  prec->SetBorder(V);
  TEST_INEQUALITY(prec->V(), Teuchos::null);

  // Check if we can set the border to null again
  prec->SetBorder(Teuchos::null);
  TEST_EQUALITY(prec->V(), Teuchos::null);
  }


Teuchos::RCP<TestablePreconditioner> create2DStokesPreconditioner(
  Teuchos::RCP<Teuchos::ParameterList> &params,
  Teuchos::RCP<Epetra_Comm> const &comm)
  {
  Teuchos::ParameterList &problemList = params->sublist("Problem");
  problemList.set("nx", 8);
  problemList.set("ny", 8);
  problemList.set("nz", 1);
  problemList.set("Degrees of Freedom", 3);
  problemList.set("Dimension", 2);

  Teuchos::ParameterList &solverList = params->sublist("Preconditioner");
  solverList.set("Separator Length", 4);
  solverList.set("Coarsening Factor", 2);
  solverList.set("Partitioner", "Skew Cartesian");
  solverList.set("Number of Levels", 2);

  for (int i = 0; i < 2; i++)
    {
    Teuchos::ParameterList& velList =
      problemList.sublist("Variable " + Teuchos::toString(i));
    velList.set("Variable Type", "Velocity");
    }

  Teuchos::ParameterList& presList =
    problemList.sublist("Variable "+Teuchos::toString(2));
  presList.set("Variable Type", "Pressure");

  Teuchos::ParameterList &ssolverList = solverList.sublist("Sparse Solver");
  ssolverList.set("amesos: solver type", "KLU");
  ssolverList.set("Custom Ordering", true);

  Teuchos::RCP<HYMLS::SkewCartesianPartitioner> part = Teuchos::rcp(
    new HYMLS::SkewCartesianPartitioner(Teuchos::null, params, *comm));
  part->Partition(true);

  Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::rcp(
    GaleriExt::CreateCrsMatrix("Stokes2D", &part->Map(), problemList));

  Teuchos::RCP<TestablePreconditioner> prec =
    Teuchos::rcp(new TestablePreconditioner(matrix, params));
  return prec;
  }

TEUCHOS_UNIT_TEST(Preconditioner, 2DStokes)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::RCP<TestablePreconditioner> prec = create2DStokesPreconditioner(params, comm);
  prec->Initialize();
  prec->Compute();
  }

TEUCHOS_UNIT_TEST(Preconditioner, ApplyInverse)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::RCP<TestablePreconditioner> prec = createPreconditioner(params, comm);
  prec->Initialize();
  prec->Compute();

  Epetra_Map const &map = prec->OperatorRangeMap();

  Teuchos::RCP<Epetra_MultiVector> X = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  X->Random();

  Teuchos::RCP<Epetra_MultiVector> X_EX = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  X_EX->Random();

  Teuchos::RCP<Epetra_MultiVector> B = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  prec->Matrix().Multiply('N', *X_EX, *B);

  prec->ApplyInverse(*B, *X);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*X, *X_EX), <, 1e-12);
  }

TEUCHOS_UNIT_TEST(Preconditioner, BorderedApplyInverse)
  {
  Teuchos::RCP<Epetra_MpiComm> comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  DISABLE_OUTPUT;

  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::RCP<TestablePreconditioner> prec = createPreconditioner(params, comm);
  prec->Initialize();
  prec->Compute();

  Epetra_Map const &map = prec->OperatorRangeMap();
  Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  V->Random();
  Teuchos::RCP<Epetra_MultiVector> W = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  W->Random();
  Teuchos::RCP<Epetra_SerialDenseMatrix> C = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));
  C->Random();

  prec->SetBorder(V, W, C);

  Teuchos::RCP<Epetra_MultiVector> X = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  X->Random();

  Teuchos::RCP<Epetra_MultiVector> X_EX = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  X_EX->Random();

  Teuchos::RCP<Epetra_SerialDenseMatrix> X2 = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));
  X2->Random();

  Teuchos::RCP<Epetra_SerialDenseMatrix> X_EX2 = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));
  X_EX2->Random();

  Teuchos::RCP<Epetra_MultiVector> B = Teuchos::rcp(new Epetra_MultiVector(map, 2));
  prec->Matrix().Multiply('N', *X_EX, *B);
  B->Multiply('N', 'N', 1.0, *V, *HYMLS::DenseUtils::CreateView(*X_EX2), 1.0);

  Teuchos::RCP<Epetra_SerialDenseMatrix> B2 = Teuchos::rcp(new Epetra_SerialDenseMatrix(2, 2));
  HYMLS::DenseUtils::CreateView(*B2)->Multiply('T', 'N', 1.0, *W, *X_EX, 0.0);
  B2->Multiply('N', 'N', 1.0, *C, *X_EX2, 1.0);

  prec->ApplyInverse(*B, *B2, *X, *X2);

  // Check if they are the same
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*X, *X_EX), <, 1e-12);
  TEST_COMPARE(HYMLS::UnitTests::NormInfAminusB(*X2, *X_EX2), <, 1e-12);
  }
