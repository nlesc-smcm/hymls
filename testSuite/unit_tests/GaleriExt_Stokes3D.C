#include "HYMLS_MatrixUtils.H"
#include "GaleriExt_Periodic.h"
#include "GaleriExt_Stokes3D.h"
#include "HYMLS_Macros.H"

#include "Epetra_MpiComm.h"
#include "EpetraExt_CrsMatrixIn.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Vector.h"

#include "HYMLS_UnitTests.H"

TEUCHOS_UNIT_TEST(GaleriExt, Stokes3D_CompareWithFile)
  {
  Epetra_MpiComm Comm(MPI_COMM_WORLD);

  // create the 32x32 C-grid Stokes matrix using the function to be tested
  const int nx = 16,ny=16,nz=16,dof=4;
  std::string filename="stokes16x16x16.mtx";
  int n = nx*ny*nz*dof;
  Teuchos::RCP<Epetra_Map> map_ptr=HYMLS::UnitTests::create_random_map(Comm,n, dof);
  const Epetra_Map& map=*map_ptr;
  double dx = 1.0/nx;
  
  // note: the matrix file we use for comparison comes from our FVM package,
  // where the equations are scaled by dx*dy*dz, hence the scaling factors
  // for the A and B part (the function puts b in the B part and a*[-1 6 -1 ...] in the A part)
  Teuchos::RCP<Epetra_CrsMatrix> A_func = Teuchos::rcp(GaleriExt::Matrices::Stokes3D
        (&map,nx,ny,nz,dx,(dx*dx),GaleriExt::NO_PERIO));
  
  // read matrix from file for comparing
  Epetra_CrsMatrix* tmp=NULL;
  EpetraExt::MatrixMarketFileToCrsMatrix(filename.c_str(),map,tmp);
  Teuchos::RCP<Epetra_CrsMatrix> A_file=Teuchos::rcp(tmp,true);

  // note: in the file we compare with, we have A B; B' 0], but our function creates A B; -B' 0], so
  // scale all the p-rows of the comparison matrix by -1 beforehand
  Epetra_Vector scale(map);
  scale.PutScalar(1.0);
  for (int i=dof-1; i<scale.MyLength(); i+=dof) scale[i]=-1.0;
  A_file->RightScale(scale);

//  std::cout << "A_file="<<*A_file<<std::endl;
//  std::cout << "A_func="<<*A_func<<std::endl;

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
  //std::cout << "diff: "<<v2<<std::endl;
  TEST_FLOATING_EQUALITY(1.0,1.0+norm_should_be_small,1e-14);
  }

