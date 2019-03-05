#include "HYMLS_MatrixUtils.H"
#include "GaleriExt_Periodic.h"
#include "GaleriExt_Darcy3D.h"
#include "HYMLS_Macros.H"

#include "Epetra_MpiComm.h"
#include "EpetraExt_CrsMatrixIn.h"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"

#include "HYMLS_UnitTests.H"

TEUCHOS_UNIT_TEST(GaleriExt, Darcy3D)
{
  Epetra_MpiComm Comm(MPI_COMM_WORLD);

  // create the 32x32 C-grid Stokes matrix using the function to be tested
  const int nx = 10, ny = 10, nz = 10, dof = 4;
  hymls_gidx n = nx*ny*nz*dof;
  Teuchos::RCP<Epetra_Map> map_ptr = HYMLS::UnitTests::create_random_map(Comm, n, dof);
  const Epetra_Map& map = *map_ptr;
  double a = 42.0;

  Teuchos::RCP<Epetra_CrsMatrix> A_func = Teuchos::rcp(
    GaleriExt::Matrices::Darcy3D(&map, nx, ny, nz, a, 1.0, GaleriExt::NO_PERIO));
  // test if the diagonal has <a> in the u/v and 0 in the p rows
  Epetra_Vector d(map);
  A_func->ExtractDiagonalCopy(d);

  double max_dev_diag = 0.0;
  for (int i = 0; i < d.MyLength(); i += dof)
  {
    max_dev_diag = std::max(max_dev_diag, std::abs(d[i]-a));
    max_dev_diag = std::max(max_dev_diag, std::abs(d[i+1]-a));
    max_dev_diag = std::max(max_dev_diag, std::abs(d[i+2]-a));
    max_dev_diag = std::max(max_dev_diag, std::abs(d[i+3]));
  }

  // test that if we apply the operator to a vector with constant P it gives the original scaled by a
  Epetra_Vector v1(map);
  v1.Random();
  // put nice numbers in the vector to avoid too much round-off
  for (int i = 0; i < v1.MyLength(); i++)
    v1[i] = (double)((int)(100.0*v1[i]));
  double b = 7.0;
  for (int i = dof-1; i < v1.MyLength(); i += dof)
    v1[i] = b;

  Epetra_Vector v2(map);
  v2.Random();

  int ierr;
  ierr = A_func->Multiply(false, v1, v2);
  TEST_EQUALITY(0, ierr);

  double norm_should_be_small = 0;
  double div_should_not_be_small = 10.0;
  for (int i = 0; i < v1.MyLength(); i += dof)
  {
    for (int j = 0; j < dof-1; j++)
    {
      norm_should_be_small = std::max(std::abs(v1[i+j]*a - v2[i+j]), norm_should_be_small);
    }
    div_should_not_be_small = std::min(std::abs(v1[i+dof-1]), div_should_not_be_small);
  }
  // std::cout  << "diff: " << v2 << std::endl;
  TEST_FLOATING_EQUALITY(1.0, 1.0+norm_should_be_small, 1e-14);
  TEST_EQUALITY(div_should_not_be_small > 1.0, true);
}

TEUCHOS_UNIT_TEST(GaleriExt, Darcy3DSymmetry)
{
  Epetra_MpiComm Comm(MPI_COMM_WORLD);

  // create the 32x32 C-grid Stokes matrix using the function to be tested
  const int nx = 10, ny = 10, nz = 10, dof = 4;
  hymls_gidx n = nx*ny*nz*dof;
  Teuchos::RCP<Epetra_Map> map_ptr = HYMLS::UnitTests::create_random_map(Comm, n, dof);
  const Epetra_Map& map = *map_ptr;

  Teuchos::RCP<Epetra_CrsMatrix> A_func = Teuchos::rcp(
    GaleriExt::Matrices::Darcy3D(&map, nx, ny, nz, 0.0, 1.0, GaleriExt::NO_PERIO));

  Epetra_Vector v1(map);
  v1.Random();
  Epetra_Vector v2(map);
  v2.Random();
  Epetra_Vector v3(map);
  v3.Random();

  int ierr;
  ierr = A_func->Multiply(false, v1, v2);
  TEST_EQUALITY(0, ierr);

  ierr = A_func->Multiply(true, v1, v3);
  TEST_EQUALITY(0, ierr);

  // Test skew-symmetry
  double norm_should_be_small = 0.0;
  v3.Update(1.0, v2, 1.0);
  v3.Norm2(&norm_should_be_small);
  TEST_FLOATING_EQUALITY(1.0, 1.0+norm_should_be_small, 1e-13);
}

TEUCHOS_UNIT_TEST(GaleriExt, DarcyB3DSymmetry)
{
  Epetra_MpiComm Comm(MPI_COMM_WORLD);

  // create the 32x32 C-grid Stokes matrix using the function to be tested
  const int nx = 10, ny = 10, nz = 10, dof = 4;
  hymls_gidx n = nx*ny*nz*dof;
  Teuchos::RCP<Epetra_Map> map_ptr = HYMLS::UnitTests::create_random_map(Comm, n, dof);
  const Epetra_Map& map = *map_ptr;

  Teuchos::RCP<Epetra_CrsMatrix> A_func = Teuchos::rcp(
    GaleriExt::Matrices::Darcy3D(&map, nx, ny, nz, 0.0, 1.0, GaleriExt::NO_PERIO, 'B'));

  Epetra_Vector v1(map);
  v1.Random();
  Epetra_Vector v2(map);
  v2.Random();
  Epetra_Vector v3(map);
  v3.Random();

  int ierr;
  ierr = A_func->Multiply(false, v1, v2);
  TEST_EQUALITY(0, ierr);

  ierr = A_func->Multiply(true, v1, v3);
  TEST_EQUALITY(0, ierr);

  // Test skew-symmetry
  double norm_should_be_small = 0.0;
  v3.Update(1.0, v2, 1.0);
  v3.Norm2(&norm_should_be_small);
  TEST_FLOATING_EQUALITY(1.0, 1.0+norm_should_be_small, 1e-13);
}

