#include "HYMLS_Tools.H"
#include "Galeri_Utils.h"
#include "GaleriExt_Periodic.h"
#include "Galeri_Exception.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Comm.h"
#include "Epetra_BlockMap.h"
#include "Epetra_Map.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_LinearProblem.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"
#include "Epetra_LAPACK.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_IntSerialDenseMatrix.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Teuchos_ParameterList.hpp"

namespace GaleriExt {

// ============================================================================ 
void GetNeighboursCartesian2d(const int i, const int nx, const int ny,
                              int & left, int & right, 
                              int & lower, int & upper,
                              PERIO_Flag perio) 
{
  Galeri::GetNeighboursCartesian2d(i,nx,ny,left,right,lower,upper);
  int ix, iy, ixx, iyy;
  ix = i % nx;
  iy = (i - ix) / nx;

  if (perio&X_PERIO)
    {
    left =  iy*nx + (MOD(ix-1,nx));
    right=  iy*nx + (MOD(ix+1,nx));
    }

  if (perio&Y_PERIO)
    {
    lower = (MOD(iy-1,ny))*nx+ix;
    upper=  (MOD(iy+1,ny))*nx+ix;
    }
}

// ============================================================================ 
void GetNeighboursCartesian3d(const int i, 
                              const int nx, const int ny, const int nz,
                              int& left, int& right, int& lower, int& upper,
                              int& below, int& above, PERIO_Flag perio) 
{
  int ixy, iz;
  ixy = i % (nx * ny);
    
  iz = (i - ixy) / (nx * ny);
  if ((perio&Z_PERIO)==0)
    {
    if (iz == 0)      below = -1;
    else              below = i - nx * ny;
    if (iz == nz - 1) above = -1;
    else              above = i + nx * ny;
    }
  else
    {
    below = MOD(i-nx*ny,nx*ny*nz);
    above = MOD(i+nx*ny,nx*ny*nz);
    }
  GetNeighboursCartesian2d(ixy, nx, ny, left, right, lower, upper, perio);
    
  if (left  != -1) left  += iz * (nx * ny);
  if (right != -1) right += iz * (nx * ny);
  if (lower != -1) lower += iz * (nx * ny);
  if (upper != -1) upper += iz * (nx * ny);
}

}//namespace
