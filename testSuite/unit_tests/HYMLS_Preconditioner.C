#include "HYMLS_Preconditioner.H"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Epetra_SerialComm.h>
#include <Epetra_MpiComm.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Util.h>
#include <Epetra_Import.h>

#include "HYMLS_SchurPreconditioner.H"
#include "HYMLS_SchurComplement.H"
#include "HYMLS_CartesianPartitioner.H"

#include "HYMLS_UnitTests.H"

class TestableSchurComplement: public HYMLS::SchurComplement
  {
public:
    TestableSchurComplement(HYMLS::SchurComplement SC)
    :
      HYMLS::SchurComplement(SC)
    {}

  Epetra_CrsMatrix const &A22()
    {
    return HYMLS::SchurComplement::A22();
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

  HYMLS::SchurPreconditioner const &SchurPreconditioner()
    {
    return *schurPrec_;
    }

  Teuchos::RCP<const Epetra_MultiVector> V()
    {
    return V_;
    }
  };

TestablePreconditioner createPreconditioner(Epetra_Comm &Comm)
  {
  Teuchos::RCP<Teuchos::ParameterList> params = Teuchos::rcp(new Teuchos::ParameterList());
  Teuchos::ParameterList &problemList = params->sublist("Problem");
  problemList.set("Degrees of Freedom", 4);
  problemList.set("Dimension", 3);
  problemList.set("nx", 8);
  problemList.set("ny", 4);
  problemList.set("nz", 4);
  Teuchos::ParameterList &precList = params->sublist("Preconditioner");
  precList.set("Separator Length", 4);
  precList.set("Number of Levels", 1);

  int n = problemList.get<int>("Degrees of Freedom") * problemList.get<int>("nx") *
    problemList.get<int>("ny") * problemList.get<int>("nz");
  Epetra_Map map(n, 0, Comm);

  HYMLS::CartesianPartitioner part(Teuchos::rcp(&map, false),
    problemList.get<int>("nx"), problemList.get<int>("ny"), problemList.get<int>("nz"),
    problemList.get<int>("Degrees of Freedom"));
  part.Partition(2, true);
  map = *part.GetMap();

  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, map, 2));

  Epetra_Util util;
  for (int i = 0; i < n; i++) {
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

  TestablePreconditioner prec(A, params);
  return prec;
  }

TEUCHOS_UNIT_TEST(Preconditioner, Blocks)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  TestablePreconditioner prec = createPreconditioner(Comm);
  prec.Initialize();
  prec.Compute();

  Teuchos::RCP<Epetra_CrsMatrix> B = Teuchos::rcp(new Epetra_CrsMatrix(
      dynamic_cast<Epetra_CrsMatrix const&>(prec.Matrix())));
  prec.SetMatrix(B);
  prec.Initialize();
  prec.Compute();

  // Make sure the pointers on the preconditioner and Schur complement are the same
  TEST_EQUALITY(&prec.A22(), &TestableSchurComplement(prec.SchurComplement()).A22());
  }

TEUCHOS_UNIT_TEST(Preconditioner, SerialComm)
  {
  // Check if HYMLS can work without MPI
  Epetra_SerialComm Comm;
  TestablePreconditioner prec = createPreconditioner(Comm);
  prec.Initialize();
  prec.Compute();
  }

TEUCHOS_UNIT_TEST(Preconditioner, setBorder)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  TestablePreconditioner prec = createPreconditioner(Comm);
  prec.Initialize();
  prec.Compute();

  Epetra_Map const &map = prec.OperatorDomainMap();
  Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(new Epetra_MultiVector(map, 1));
  V->Random();

  prec.setBorder(V);

  // Import the vector since the matrix might have a different map
  Teuchos::RCP<const Epetra_MultiVector> precV = prec.V();
  Epetra_MultiVector importedV(map, 1);
  Epetra_Import importer(map, precV->Map());
  importedV.Import(*precV, importer, Insert);

  // Check if they are the same
  TEST_FLOATING_EQUALITY(HYMLS::UnitTests::NormInfAminusB(importedV, *V), 0.0, 1e-12);
  }

TEUCHOS_UNIT_TEST(Preconditioner, setBorderNull)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  TestablePreconditioner prec = createPreconditioner(Comm);
  prec.Initialize();
  prec.Compute();

  // Set the border to null
  prec.setBorder(Teuchos::null);
  TEST_EQUALITY(prec.V(), Teuchos::null);

  // Set the border to something else
  Epetra_Map const &map = prec.OperatorDomainMap();
  Teuchos::RCP<Epetra_MultiVector> V = Teuchos::rcp(new Epetra_MultiVector(map, 1));
  V->Random();

  prec.setBorder(V);
  TEST_INEQUALITY(prec.V(), Teuchos::null);

  // Check if we can set the border to null again
  prec.setBorder(Teuchos::null);
  TEST_EQUALITY(prec.V(), Teuchos::null);
  }
