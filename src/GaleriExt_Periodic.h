#ifndef GALERIEXT_PERIODIC_H
#define GALERIEXT_PERIODIC_H

#include "Galeri_ConfigDefs.h"
#include "Galeri_Exception.h"

class Epetra_BlockMap;
class Epetra_RowMatrix;
class Epetra_CrsMatrix;
class Epetra_MultiVector;
class Epetra_LinearProblem;
namespace Teuchos {
  class ParameterList;
}

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
                         int& left2, int& right2, int& lower2, int& upper2,
                         PERIO_Flag perio);

void 
GetNeighboursCartesian3d(const int i, const int nx, const int ny, const int nz,
                         int& left, int& right, int& lower, int& upper,
                         int& below, int& above,PERIO_Flag perio);

} // namespace GaleriExt

#endif
