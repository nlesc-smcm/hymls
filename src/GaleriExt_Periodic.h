#ifndef GALERIEXT_PERIODIC_H
#define GALERIEXT_PERIODIC_H

#include "Galeri_ConfigDefs.h"

namespace GaleriExt {

typedef enum {NO_PERIO=0x0,
              Z_PERIO=0x1,
              Y_PERIO=0x2,
              YZ_PERIO=0x3,
              X_PERIO=0x4, 
              XZ_PERIO=0x5,
              XY_PERIO=0x6,
              XYZ_PERIO=0x7} PERIO_Flag;

void 
GetNeighboursCartesian2d(const int i, const int nx, const int ny,
                         int& left, int& right, int& lower, int& upper,
                         PERIO_Flag perio);

void 
GetNeighboursCartesian3d(const int i, const int nx, const int ny, const int nz,
                         int& left, int& right, int& lower, int& upper,
                         int& below, int& above,PERIO_Flag perio);

void
GetNeighboursCartesian3d(const int i,
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
                         PERIO_Flag perio);

} // namespace GaleriExt

#endif
