#include "HYMLS_MatrixUtils.H"
#include "GaleriExt_Periodic.h"
#include "GaleriExt_Darcy2D.h"
#include "HYMLS_Macros.H"

#include "Epetra_MpiComm.h"
#include "EpetraExt_CrsMatrixIn.h"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"

#include "HYMLS_UnitTests.H"

TEUCHOS_UNIT_TEST(GaleriExt, Darcy2D)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);

  // create the 32x32 C-grid Stokes matrix using the function to be tested
  const int nx = 32,ny=32,dof=3;
  hymls_gidx n = nx*ny*dof;
  Teuchos::RCP<Epetra_Map> map_ptr=HYMLS::UnitTests::create_random_map(Comm,n, dof);
  const Epetra_Map& map=*map_ptr;
  double a=42.0;
  
  Teuchos::RCP<Epetra_CrsMatrix> A_func = Teuchos::rcp(GaleriExt::Matrices::Darcy2D
        (&map,nx,ny,a,1.0,GaleriExt::NO_PERIO));
  
  // test if the diagonal has <a> in the u/v and 0 in the p rows
  Epetra_Vector d(map);
  A_func->ExtractDiagonalCopy(d);
  
  double max_dev_diag=0.0;
  for (int i=0; i<d.MyLength(); i+=3)
  {
    max_dev_diag = std::max(max_dev_diag, std::abs(d[i]-a));
    max_dev_diag = std::max(max_dev_diag, std::abs(d[i+1]-a));
    max_dev_diag = std::max(max_dev_diag, std::abs(d[i+2]));
  }
  
  // test that if we apply the operator to a vector with constant P it gives the original scaled by a
  Epetra_Vector v1(map);
  v1.Random();
  // put nice numbers in the vector to avoid too much round-off
  for (int i=0; i<v1.MyLength(); i++) v1[i]=(double)((int)(100.0*v1[i]));
  double b=7.0;
  for (int i=dof-1; i< v1.MyLength(); i+=dof) v1[i]=b;
  Epetra_Vector v2(map);
  v2.Random();
  
  int ierr;  
  ierr=A_func->Multiply(false,v1,v2);
  TEST_EQUALITY(0,ierr);
  double norm_should_be_small=0;
  for (int i=0; i<v1.MyLength(); i+=dof)
  {
    for (int j=0; j<dof-1; j++)
    {
      norm_should_be_small = std::max(std::abs(v1[i+j]*a - v2[i+j]),norm_should_be_small);
    }
  }
//  std::cout << "diff: "<<v2<<std::endl;
  TEST_FLOATING_EQUALITY(1.0,1.0+norm_should_be_small,1e-14);
  }

