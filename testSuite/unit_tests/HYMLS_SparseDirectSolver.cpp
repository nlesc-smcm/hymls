#include "HYMLS_SparseDirectSolver.hpp"

#include "Epetra_SerialComm.h"
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"

#include "GaleriExt_Stokes2D.h"

#include "HYMLS_Macros.hpp"
#include "HYMLS_MatrixUtils.hpp"
#include "HYMLS_UnitTests.hpp"

Teuchos::RCP<Epetra_CrsMatrix> createStokesMatrix(int nx, char grid_type='C')
  {
  int ny = nx, dof = 3;
  int n = nx * ny * dof;
  Teuchos::Array<int> pvars;
  pvars.append(2);
  if (grid_type == 'B')
    pvars.append(5);

  Epetra_SerialComm comm;
  Epetra_Map map(n, 0, comm);

  Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp(GaleriExt::Matrices::Stokes2D(
      &map, nx, ny, nx*nx, 1.0, GaleriExt::NO_PERIO, grid_type));

  A = HYMLS::MatrixUtils::DropByValue(A, HYMLS_SMALL_ENTRY, HYMLS::MatrixUtils::RelFullDiag);

  // Fix one p to make the matrix nonsingular
  int maxlen = A->MaxNumEntries();
  Teuchos::Array<int> indices(maxlen);
  Teuchos::Array<double> values(maxlen);
  // Boundary condition
  for (int i = 0; i < A->NumMyRows(); i++)
    {
    int len;
    CHECK_ZERO(A->ExtractGlobalRowCopy(i, maxlen, len, &values[0], &indices[0]));
    if (std::find(pvars.begin(), pvars.end(), i) != pvars.end())
      {
      for (int j = 0; j < len; j++)
        {
        if (indices[j] == i)
          values[j] = 1.0;
        else
          values[j] = 0.0;
        }
      }
    else
      {
      for (int j = 0; j < len; j++)
        {
        if (std::find(pvars.begin(), pvars.end(), indices[j]) != pvars.end())
          values[j] = 0.0;
        }
      }
    CHECK_ZERO(A->ReplaceGlobalValues(i, len, &values[0], &indices[0]));
    }
  return HYMLS::MatrixUtils::DropByValue(A, HYMLS_SMALL_ENTRY, HYMLS::MatrixUtils::RelDropDiag);
  }

TEUCHOS_UNIT_TEST(SparseDirectSolver, NoCustomOrdering)
  {
  DISABLE_OUTPUT;
  Teuchos::RCP<Epetra_CrsMatrix> A;
  Teuchos::RCP<HYMLS::SparseDirectSolver> solver;

  A = createStokesMatrix(3);
  solver = Teuchos::rcp(new HYMLS::SparseDirectSolver(A.get()));

  CHECK_ZERO(solver->Initialize());
  CHECK_ZERO(solver->Compute());

  TEST_EQUALITY(solver->NumGlobalNonzerosL(), 89);

  A = createStokesMatrix(5);
  solver = Teuchos::rcp(new HYMLS::SparseDirectSolver(A.get()));

  CHECK_ZERO(solver->Initialize());
  CHECK_ZERO(solver->Compute());

  TEST_EQUALITY(solver->NumGlobalNonzerosL(), 522);

  A = createStokesMatrix(9);
  solver = Teuchos::rcp(new HYMLS::SparseDirectSolver(A.get()));

  CHECK_ZERO(solver->Initialize());
  CHECK_ZERO(solver->Compute());

  TEST_EQUALITY(solver->NumGlobalNonzerosL(), 2768);

  A = createStokesMatrix(3, 'B');
  solver = Teuchos::rcp(new HYMLS::SparseDirectSolver(A.get()));

  CHECK_ZERO(solver->Initialize());
  CHECK_ZERO(solver->Compute());

  TEST_EQUALITY(solver->NumGlobalNonzerosL(), 65);

  A = createStokesMatrix(5, 'B');
  solver = Teuchos::rcp(new HYMLS::SparseDirectSolver(A.get()));

  CHECK_ZERO(solver->Initialize());
  CHECK_ZERO(solver->Compute());

  TEST_EQUALITY(solver->NumGlobalNonzerosL(), 489);

  A = createStokesMatrix(9, 'B');
  solver = Teuchos::rcp(new HYMLS::SparseDirectSolver(A.get()));

  CHECK_ZERO(solver->Initialize());
  CHECK_ZERO(solver->Compute());

  TEST_EQUALITY(solver->NumGlobalNonzerosL(), 3317);
  }

TEUCHOS_UNIT_TEST(SparseDirectSolver, CustomOrdering)
  {
  DISABLE_OUTPUT;
  Teuchos::RCP<Epetra_CrsMatrix> A;
  Teuchos::RCP<HYMLS::SparseDirectSolver> solver;

  Teuchos::ParameterList params;
  params.set("Custom Ordering", true);

  A = createStokesMatrix(3);
  solver = Teuchos::rcp(new HYMLS::SparseDirectSolver(A.get()));

  CHECK_ZERO(solver->SetParameters(params));
  CHECK_ZERO(solver->Initialize());
  CHECK_ZERO(solver->Compute());

  TEST_EQUALITY(solver->NumGlobalNonzerosL(), 78); // 82 in the paper

  A = createStokesMatrix(5);
  solver = Teuchos::rcp(new HYMLS::SparseDirectSolver(A.get()));

  CHECK_ZERO(solver->SetParameters(params));
  CHECK_ZERO(solver->Initialize());
  CHECK_ZERO(solver->Compute());

  TEST_EQUALITY(solver->NumGlobalNonzerosL(), 397); // 403 in the paper

  A = createStokesMatrix(9);
  solver = Teuchos::rcp(new HYMLS::SparseDirectSolver(A.get()));

  CHECK_ZERO(solver->SetParameters(params));
  CHECK_ZERO(solver->Initialize());
  CHECK_ZERO(solver->Compute());

  TEST_EQUALITY(solver->NumGlobalNonzerosL(), 2033); // 2134 in the paper
  }
