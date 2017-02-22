#include "HYMLS_Preconditioner.H"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Epetra_MpiComm.h>
#include <Epetra_Map.h>
#include <Epetra_MultiVector.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Util.h>

#include "HYMLS_SchurPreconditioner.H"
#include "HYMLS_SchurComplement.H"

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
  };

TEUCHOS_UNIT_TEST(Preconditioner, Blocks)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);

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

  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(new Epetra_CrsMatrix(Copy, map, 2));
  Epetra_Util util;
  for (int i = 0; i < n; i++) {
    int A_idx = util.RandomInt() % n;
    double A_val = util.RandomDouble();
    A->InsertGlobalValues(i, 1, &A_val, &A_idx);
    A->InsertGlobalValues(i, 1, &A_val, &i);
  }
  A->FillComplete();

  TestablePreconditioner prec(A, params);
  prec.Initialize();
  prec.Compute();

  Teuchos::RCP<Epetra_CrsMatrix> B = Teuchos::rcp(new Epetra_CrsMatrix(*A));
  prec.SetMatrix(B);
  prec.Initialize();
  prec.Compute();

  // Make sure the pointers on the preconditioner and Schur complement are the same
  TEST_EQUALITY(&prec.A22(), &TestableSchurComplement(prec.SchurComplement()).A22());
  }
