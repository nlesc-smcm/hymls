#include <mpi.h>
#include <iostream>

#include "HYMLS_OrthogonalTransform.H"
#include "HYMLS_Householder.H"
#include "Epetra_SerialDenseVector.h"
#include "Epetra_SerialDenseMatrix.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "Epetra_CrsMatrix.h"
#include "Epetra_SerialComm.h"
#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"

#include "HYMLS_Tools.H"
#include "Galeri_Periodic.h"

using namespace Galeri;

int main(int argc, char** argv)
  {
  std::cout << "true: " << std::endl;
  std::cout << (X_PERIO&X_PERIO) << std::endl;
  std::cout << (XY_PERIO&X_PERIO) << std::endl;
  std::cout << (XZ_PERIO&X_PERIO) << std::endl;
  std::cout << (XYZ_PERIO&X_PERIO) << std::endl;
  std::cout << std::endl;

  std::cout << (Y_PERIO&Y_PERIO) << std::endl;
  std::cout << (XY_PERIO&Y_PERIO) << std::endl;
  std::cout << (YZ_PERIO&Y_PERIO) << std::endl;
  std::cout << (XYZ_PERIO&Y_PERIO) << std::endl;
  std::cout << std::endl;

  std::cout << (Z_PERIO&Z_PERIO) << std::endl;
  std::cout << (XZ_PERIO&Z_PERIO) << std::endl;
  std::cout << (YZ_PERIO&Z_PERIO) << std::endl;
  std::cout << (XYZ_PERIO&Z_PERIO) << std::endl;
  std::cout << std::endl;
  

  std::cout << "false: " << std::endl;
  std::cout << (Y_PERIO&X_PERIO) << std::endl;
  std::cout << (Z_PERIO&X_PERIO) << std::endl;
  std::cout << (YZ_PERIO&X_PERIO) << std::endl;
  std::cout << std::endl;

  std::cout << (X_PERIO&Y_PERIO) << std::endl;
  std::cout << (Z_PERIO&Y_PERIO) << std::endl;
  std::cout << (XZ_PERIO&Y_PERIO) << std::endl;
  std::cout << std::endl;

  std::cout << (X_PERIO&Z_PERIO) << std::endl;
  std::cout << (Y_PERIO&Z_PERIO) << std::endl;
  std::cout << (XY_PERIO&Z_PERIO) << std::endl;
  std::cout << std::endl;

  int nx=2; 
  int ny=2;
  int nz=2;
  int left,right,lower,upper,below,above;
  PERIO_Flag perio=Y_PERIO;
  int center=6;
  GetNeighboursCartesian3d(center,nx,ny,nz,
        left,right,lower,upper,below,above,perio);

  std::cout << std::endl;
  std::cout << "\t"<< above << std::endl;
  std::cout << std::endl;

  std::cout << std::endl;

  std::cout << " stencil: "<<std::endl;
  std::cout << "\t"<<upper<<std::endl;
  std::cout << left << "\t"<<center<<"\t"<<right<<std::endl;
  std::cout << "\t"<<lower<<std::endl;

  std::cout << std::endl;

  std::cout << std::endl;
  std::cout << "\t"<< below << std::endl;
  std::cout << std::endl;

  }
