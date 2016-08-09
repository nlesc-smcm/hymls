#include "HYMLS_MatrixUtils.H"
#include "GaleriExt_Periodic.h"
#include "GaleriExt_Stokes2D.h"
#include "HYMLS_Macros.H"

#include "Epetra_MpiComm.h"
#include "EpetraExt_CrsMatrixIn.h"

#include "HYMLS_UnitTests.H"

TEUCHOS_UNIT_TEST(Stokes2D, CompareWithFile)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);

  // create the 32x32 C-grid Stokes matrix using the function to be tested
  const int nx = 32,ny=32,dof=3;
  std::string filename="stokes32x32.mtx";
  int n = nx*ny*dof;
  Epetra_Map map(n, 0, Comm);
  double dx = 1.0/nx, dy=1.0/ny;
  
  Teuchos::RCP<Epetra_CrsMatrix> A_func = Teuchos::rcp(GaleriExt::Matrices::Stokes2D
        (&map,nx,ny,1.0/(dx*dx),1.0,GaleriExt::NO_PERIO));
  
  // read matrix from file for comparing
  Epetra_CrsMatrix* tmp=NULL;
  EpetraExt::MatrixMarketFileToCrsMatrix(filename.c_str(),map,tmp);
  Teuchos::RCP<Epetra_CrsMatrix> A_file=Teuchos::rcp(tmp,true);

//  std::cout << "A_file="<<*A_file<<std::endl;
 // std::cout << "A_func="<<*A_func<<std::endl;

  // test if the matrices are the same by doing a matvec with a random vector
  Epetra_Vector v1(map);
  v1.Random();
  Epetra_Vector v2(map),v3(map);
  v2.Random(); v3.Random();
  // put nice numbers in the vector to avoid too much round-off
  for (int i=0; i<v1.MyLength(); i++) v1[i]=(double)((int)(100.0*v2[i]));
  int ierr;  
  ierr=A_func->Multiply(false,v1,v2);
  TEST_EQUALITY(0,ierr);
  ierr=A_file->Multiply(false,v1,v3);
  TEST_EQUALITY(0,ierr);
  v2.Update(-1.0,v3,1.0);
  double norm_should_be_small;
  v2.Norm2(&norm_should_be_small);
//  std::cout << "diff: "<<v2<<std::endl;
  TEST_FLOATING_EQUALITY(1.0,1.0+norm_should_be_small,1e-14);
  }

