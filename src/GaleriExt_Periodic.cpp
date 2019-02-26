#include "Galeri_Utils.h"
#include "GaleriExt_Periodic.h"

namespace GaleriExt {

#define MOD(a, b) (((a) % (b) + (b)) % (b))

// ============================================================================ 
void GetNeighboursCartesian2d(const int i, const int nx, const int ny,
                              int & left, int & right, 
                              int & lower, int & upper,
                              PERIO_Flag perio) 
{
  Galeri::GetNeighboursCartesian2d(i,nx,ny,left,right,lower,upper);
  int ix, iy;
  ix = i % nx;
  iy = (i - ix) / nx;

  if (perio & X_PERIO)
    {
    left = iy*nx + MOD(ix-1, nx);
    right = iy*nx + MOD(ix+1, nx);
    }

  if (perio & Y_PERIO)
    {
    lower = MOD(iy-1, ny)*nx+ix;
    upper = MOD(iy+1, ny)*nx+ix;
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
  if (!(perio & Z_PERIO))
    {
    if (iz == 0)      below = -1;
    else              below = i - nx * ny;
    if (iz == nz - 1) above = -1;
    else              above = i + nx * ny;
    }
  else
    {
    below = MOD(i-nx*ny, nx*ny*nz);
    above = MOD(i+nx*ny, nx*ny*nz);
    }
  GetNeighboursCartesian2d(ixy, nx, ny, left, right, lower, upper, perio);
    
  if (left  != -1) left  += iz * (nx * ny);
  if (right != -1) right += iz * (nx * ny);
  if (lower != -1) lower += iz * (nx * ny);
  if (upper != -1) upper += iz * (nx * ny);
}

void GetNeighboursCartesian3d(const int i,
                              const int nx, const int ny, const int nz,
                              int& above_upper_left, int& above_upper, int& above_upper_right,
                              int& above_left, int& above, int& above_right,
                              int& above_lower_left, int& above_lower, int& above_lower_right,
                              int& upper_left, int& upper, int& upper_right,
                              int& left, int& base, int& right,
                              int& lower_left, int& lower, int& lower_right,
                              int& below_upper_left, int& below_upper, int& below_upper_right,
                              int& below_left, int& below, int& below_right,
                              int& below_lower_left, int& below_lower, int& below_lower_right,
                              PERIO_Flag perio)
{
  int dummy;

  above_upper_left = -1;
  above_upper = -1;
  above_upper_right = -1;

  above_left = -1;
  above = -1;
  above_right = -1;

  above_lower_left = -1;
  above_lower = -1;
  above_lower_right = -1;

  upper_left = -1;
  upper = -1;
  upper_right = -1;

  left = -1;
  base = i;
  right = -1;

  lower_left = -1;
  lower = -1;
  lower_right = -1;

  below_upper_left = -1;
  below_upper = -1;
  below_upper_right = -1;

  below_left = -1;
  below = -1;
  below_right = -1;

  below_lower_left = -1;
  below_lower = -1;
  below_lower_right = -1;

  GetNeighboursCartesian3d(i, nx, ny, nz, left, right, lower, upper, below, above, perio);
  if (above != -1)
    GetNeighboursCartesian3d(above, nx, ny, nz, above_left, above_right,
                             above_lower, above_upper, dummy, dummy, perio);
  if (above_left != -1)
    GetNeighboursCartesian3d(above_left, nx, ny, nz, dummy, dummy,
                             above_lower_left, above_upper_left, dummy, dummy, perio);
  if (above_right != -1)
    GetNeighboursCartesian3d(above_right, nx, ny, nz, dummy, dummy,
                             above_lower_right, above_upper_right, dummy, dummy, perio);
  if (below != -1)
    GetNeighboursCartesian3d(below, nx, ny, nz, below_left, below_right,
                             below_lower, below_upper, dummy, dummy, perio);
  if (below_left != -1)
    GetNeighboursCartesian3d(below_left, nx, ny, nz, dummy, dummy,
                             below_lower_left, below_upper_left, dummy, dummy, perio);
  if (below_right != -1)
    GetNeighboursCartesian3d(below_right, nx, ny, nz, dummy, dummy,
                             below_lower_right, below_upper_right, dummy, dummy, perio);
  if (left != -1)
    GetNeighboursCartesian3d(left, nx, ny, nz, dummy, dummy,
                             lower_left, upper_left, dummy, dummy, perio);
  if (right != -1)
    GetNeighboursCartesian3d(right, nx, ny, nz, dummy, dummy,
                             lower_right, upper_right, dummy, dummy, perio);
}

}//namespace
