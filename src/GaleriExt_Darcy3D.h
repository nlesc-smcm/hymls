// @HEADER
// ************************************************************************
//
//           Galeri: Finite Element and Matrix Generation Package
//                 Copyright (2006) ETHZ/Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
//
// Questions about Galeri? Contact Marzio Sala (marzio.sala _AT_ gmail.com)
//
// ************************************************************************
// @HEADER

#ifndef GALERIEXT_DARCY3D_H
#define GALERIEXT_DARCY3D_H

#include "Galeri_Exception.h"
#include "Galeri_Utils.h"
#include "Epetra_Comm.h"
#include "Epetra_BlockMap.h"
#include "Epetra_CrsMatrix.h"

#include "GaleriExt_Periodic.h"

namespace GaleriExt {
namespace Matrices {

//! generate an F-matrix with A = diag(a) and +b -b in the B part
template<typename int_type>
inline
Epetra_CrsMatrix*
Darcy3D(const Epetra_Map* Map,
        const int nx, const int ny, const int nz,
        const double a, const double b,
        PERIO_Flag perio=NO_PERIO)
{
  Epetra_CrsMatrix* Matrix = new Epetra_CrsMatrix(Copy, *Map,  6);

  int NumMyElements = Map->NumMyElements();
  int_type* MyGlobalElements = 0;
  Map->MyGlobalElementsPtr(MyGlobalElements);

  int left, right, lower, upper, below, above;

  std::vector<double> Values(6);
  std::vector<int_type> Indices(6);

  double c = -b; // c==b => [A B'; B 0]. c==-b => A B'; -B 0]

  int dof = 4;

  if (dof*nx*ny*nz!=Map->NumGlobalElements64())
  {
    throw("bad input map for GaleriExt::Darcy3D. Should have 4 dof/node");
  }

  for (int i = 0 ; i < NumMyElements ; i++)
  {
    int NumEntries = 0;

    int_type ibase = std::floor(MyGlobalElements[i]/dof);
    int_type ivar   = MyGlobalElements[i]-ibase*dof;
    // first the regular 7-point stencil
    GetNeighboursCartesian3d(ibase, nx, ny, nz,
                             left, right, lower, upper, below, above,
                             perio);

    if (ivar!=3)
    {
      NumEntries=1;
      Values[0]=a;
      Indices[0]=MyGlobalElements[i];
      if (right != -1 && ivar==0)
      {
        Indices[NumEntries] = ibase*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = right*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
      }
      if (upper != -1 && ivar==1)
      {
        Indices[NumEntries] = ibase*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = upper*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
      }
      if (above != -1 && ivar==2)
      {
        Indices[NumEntries] = ibase*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = above*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
      }
    }
    else // P
    {
      // div-rows
      if (right!=-1)
      {
        Indices[NumEntries]=ibase*dof+0;
        Values[NumEntries]=-c;
        NumEntries++;
      }
      if (upper!=-1)
      {
        Indices[NumEntries]=ibase*dof+1;
        Values[NumEntries]=-c;
        NumEntries++;
      }
      if (above!=-1)
      {
        Indices[NumEntries]=ibase*dof+2;
        Values[NumEntries]=-c;
        NumEntries++;
      }
      if (left!=-1)
      {
        Indices[NumEntries]=left*dof+0;
        Values[NumEntries]=c;
        NumEntries++;
      }
      if (lower!=-1)
      {
        Indices[NumEntries]=lower*dof+1;
        Values[NumEntries]=c;
        NumEntries++;
      }
      if (below!=-1)
      {
        Indices[NumEntries]=below*dof+2;
        Values[NumEntries]=c;
        NumEntries++;
      }
    }

#ifdef DEBUGGING_
    std::cerr << i << " " << MyGlobalElements[i] << " " << ibase << " " << ivar << std::endl;
    for (int jj=0;jj<NumEntries;jj++)
    {
      std::cerr << Indices[jj] << " ";
    }
    std::cerr << std::endl;
#endif

    // put the off-diagonal entries
    Matrix->InsertGlobalValues(MyGlobalElements[i], NumEntries,
                               &Values[0], &Indices[0]);

  }
  Matrix->FillComplete();
  Matrix->OptimizeStorage();

  return(Matrix);
}

//! generate an B-grid discretization with A = diag(a) and +b -b in the B part
template<typename int_type>
inline
Epetra_CrsMatrix*
DarcyB3D(const Epetra_Map* Map,
        const int nx, const int ny, const int nz,
        const double a, const double b,
        PERIO_Flag perio=NO_PERIO)
{
  Epetra_CrsMatrix* Matrix = new Epetra_CrsMatrix(Copy, *Map,  24);

  int NumMyElements = Map->NumMyElements();
  int_type* MyGlobalElements = 0;
  Map->MyGlobalElementsPtr(MyGlobalElements);

  int above_upper_left, above_upper, above_upper_right,
    above_left, above, above_right,
    above_lower_left, above_lower, above_lower_right,
    upper_left, upper, upper_right,
    left, base, right,
    lower_left, lower, lower_right,
    below_upper_left, below_upper, below_upper_right,
    below_left, below, below_right,
    below_lower_left, below_lower, below_lower_right;

  std::vector<double> Values(24);
  std::vector<int_type> Indices(24);

  double c = -b; // c==b => [A B'; B 0]. c==-b => A B'; -B 0]

  int dof = 4;

  if (dof*nx*ny*nz!=Map->NumGlobalElements64())
  {
    throw("bad input map for GaleriExt::Darcy3D. Should have 4 dof/node");
  }

  for (int i = 0 ; i < NumMyElements ; i++)
  {
    int NumEntries = 0;

    int_type ibase = std::floor(MyGlobalElements[i]/dof);
    int_type ivar   = MyGlobalElements[i]-ibase*dof;
    // first the regular 27-point stencil
    GetNeighboursCartesian3d(ibase, nx, ny, nz,
                             above_upper_left, above_upper, above_upper_right,
                             above_left, above, above_right,
                             above_lower_left, above_lower, above_lower_right,
                             upper_left, upper, upper_right,
                             left, base, right,
                             lower_left, lower, lower_right,
                             below_upper_left, below_upper, below_upper_right,
                             below_left, below, below_right,
                             below_lower_left, below_lower, below_lower_right,
                             perio);

    if (ivar!=3)
    {
      NumEntries=1;
      Values[0]=a;
      Indices[0]=MyGlobalElements[i];
      if (above_upper_right != -1 && ivar == 0)
      {
        Indices[NumEntries] = ibase*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = upper*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = right*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
        Indices[NumEntries] = upper_right*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
        Indices[NumEntries] = above*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = above_upper*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = above_right*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
        Indices[NumEntries] = above_upper_right*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
      }
      else if (above_upper_right != -1 && ivar == 1)
      {
        Indices[NumEntries] = ibase*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = upper*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
        Indices[NumEntries] = right*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = upper_right*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
        Indices[NumEntries] = above*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = above_upper*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
        Indices[NumEntries] = above_right*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = above_upper_right*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
      }
      else if (above_upper_right != -1 && ivar == 2)
      {
        Indices[NumEntries] = ibase*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = upper*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = right*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = upper_right*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = above*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
        Indices[NumEntries] = above_upper*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
        Indices[NumEntries] = above_right*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
        Indices[NumEntries] = above_upper_right*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
      }
    }
    else // P
    {
      // div-rows
      if (above_upper_right != -1)
      {
        Indices[NumEntries] = ibase*dof+0;
        Values[NumEntries] = -c;
        NumEntries++;
        Indices[NumEntries] = ibase*dof+1;
        Values[NumEntries] = -c;
        NumEntries++;
        Indices[NumEntries] = ibase*dof+2;
        Values[NumEntries] = -c;
        NumEntries++;
      }
      if (above_upper_left != -1)
      {
        Indices[NumEntries] = left*dof+0;
        Values[NumEntries] = c;
        NumEntries++;
        Indices[NumEntries] = left*dof+1;
        Values[NumEntries] = -c;
        NumEntries++;
        Indices[NumEntries] = left*dof+2;
        Values[NumEntries] = -c;
        NumEntries++;
      }
      if (above_lower_right != -1)
      {
        Indices[NumEntries] = lower*dof+0;
        Values[NumEntries] = -c;
        NumEntries++;
        Indices[NumEntries] = lower*dof+1;
        Values[NumEntries] = c;
        NumEntries++;
        Indices[NumEntries] = lower*dof+2;
        Values[NumEntries] = -c;
        NumEntries++;
      }
      if (above_lower_left != -1)
      {
        Indices[NumEntries] = lower_left*dof+0;
        Values[NumEntries] = c;
        NumEntries++;
        Indices[NumEntries] = lower_left*dof+1;
        Values[NumEntries] = c;
        NumEntries++;
        Indices[NumEntries] = lower_left*dof+2;
        Values[NumEntries] = -c;
        NumEntries++;
      }
      if (below_upper_right != -1)
      {
        Indices[NumEntries] = below*dof+0;
        Values[NumEntries] = -c;
        NumEntries++;
        Indices[NumEntries] = below*dof+1;
        Values[NumEntries] = -c;
        NumEntries++;
        Indices[NumEntries] = below*dof+2;
        Values[NumEntries] = c;
        NumEntries++;
      }
      if (below_upper_left != -1)
      {
        Indices[NumEntries] = below_left*dof+0;
        Values[NumEntries] = c;
        NumEntries++;
        Indices[NumEntries] = below_left*dof+1;
        Values[NumEntries] = -c;
        NumEntries++;
        Indices[NumEntries] = below_left*dof+2;
        Values[NumEntries] = c;
        NumEntries++;
      }
      if (below_lower_right != -1)
      {
        Indices[NumEntries] = below_lower*dof+0;
        Values[NumEntries] = -c;
        NumEntries++;
        Indices[NumEntries] = below_lower*dof+1;
        Values[NumEntries] = c;
        NumEntries++;
        Indices[NumEntries] = below_lower*dof+2;
        Values[NumEntries] = c;
        NumEntries++;
      }
      if (below_lower_left != -1)
      {
        Indices[NumEntries] = below_lower_left*dof+0;
        Values[NumEntries] = c;
        NumEntries++;
        Indices[NumEntries] = below_lower_left*dof+1;
        Values[NumEntries] = c;
        NumEntries++;
        Indices[NumEntries] = below_lower_left*dof+2;
        Values[NumEntries] = c;
        NumEntries++;
      }
    }

#ifdef DEBUGGING_
    std::cerr << i << " " << MyGlobalElements[i] << " " << ibase << " " << ivar << std::endl;
    for (int jj=0;jj<NumEntries;jj++)
    {
      std::cerr << Indices[jj] << " ";
    }
    std::cerr << std::endl;
#endif

    // put the off-diagonal entries
    Matrix->InsertGlobalValues(MyGlobalElements[i], NumEntries,
                               &Values[0], &Indices[0]);

  }
  Matrix->FillComplete();
  Matrix->OptimizeStorage();

  return(Matrix);
}

//! generate an THCM grid discretization with A = diag(a) and +b -b in the B part
template<typename int_type>
inline
Epetra_CrsMatrix*
DarcyTHCM3D(const Epetra_Map* Map,
            const int nx, const int ny, const int nz,
            const double a, const double b,
            PERIO_Flag perio=NO_PERIO)
{
  Epetra_CrsMatrix* Matrix = new Epetra_CrsMatrix(Copy, *Map,  24);

  int NumMyElements = Map->NumMyElements();
  int_type* MyGlobalElements = 0;
  Map->MyGlobalElementsPtr(MyGlobalElements);

  int above_upper_left, above_upper, above_upper_right,
    above_left, above, above_right,
    above_lower_left, above_lower, above_lower_right,
    upper_left, upper, upper_right,
    left, base, right,
    lower_left, lower, lower_right,
    below_upper_left, below_upper, below_upper_right,
    below_left, below, below_right,
    below_lower_left, below_lower, below_lower_right;

  std::vector<double> Values(24);
  std::vector<int_type> Indices(24);

  double c = -b; // c==b => [A B'; B 0]. c==-b => A B'; -B 0]

  int dof = 4;

  if (dof*nx*ny*nz!=Map->NumGlobalElements64())
  {
    throw("bad input map for GaleriExt::Darcy3D. Should have 4 dof/node");
  }

  for (int i = 0 ; i < NumMyElements ; i++)
  {
    int NumEntries = 0;

    int_type ibase = std::floor(MyGlobalElements[i]/dof);
    int_type ivar   = MyGlobalElements[i]-ibase*dof;
    // first the regular 27-point stencil
    GetNeighboursCartesian3d(ibase, nx, ny, nz,
                             above_upper_left, above_upper, above_upper_right,
                             above_left, above, above_right,
                             above_lower_left, above_lower, above_lower_right,
                             upper_left, upper, upper_right,
                             left, base, right,
                             lower_left, lower, lower_right,
                             below_upper_left, below_upper, below_upper_right,
                             below_left, below, below_right,
                             below_lower_left, below_lower, below_lower_right,
                             perio);

    if (ivar!=3)
    {
      NumEntries=1;
      Values[0]=a;
      Indices[0]=MyGlobalElements[i];
      if (upper_right != -1 && ivar == 0)
      {
        Indices[NumEntries] = ibase*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = upper*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = right*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
        Indices[NumEntries] = upper_right*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
      }
      else if (upper_right != -1 && ivar == 1)
      {
        Indices[NumEntries] = ibase*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = upper*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
        Indices[NumEntries] = right*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = upper_right*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
      }
      else if (above != -1 && ivar == 2)
      {
        Indices[NumEntries] = ibase*dof+3;
        Values[NumEntries] = -b;
        ++NumEntries;
        Indices[NumEntries] = above*dof+3;
        Values[NumEntries] = b;
        ++NumEntries;
      }
    }
    else // P
    {
      // div-rows
      if (upper_right != -1)
      {
        Indices[NumEntries] = ibase*dof+0;
        Values[NumEntries] = -c;
        NumEntries++;
        Indices[NumEntries] = ibase*dof+1;
        Values[NumEntries] = -c;
        NumEntries++;
      }
      if (above != -1)
      {
        Indices[NumEntries] = ibase*dof+2;
        Values[NumEntries] = -c;
        NumEntries++;
      }
      if (upper_left != -1)
      {
        Indices[NumEntries] = left*dof+0;
        Values[NumEntries] = c;
        NumEntries++;
        Indices[NumEntries] = left*dof+1;
        Values[NumEntries] = -c;
        NumEntries++;
      }
      if (lower_right != -1)
      {
        Indices[NumEntries] = lower*dof+0;
        Values[NumEntries] = -c;
        NumEntries++;
        Indices[NumEntries] = lower*dof+1;
        Values[NumEntries] = c;
        NumEntries++;
      }
      if (lower_left != -1)
      {
        Indices[NumEntries] = lower_left*dof+0;
        Values[NumEntries] = c;
        NumEntries++;
        Indices[NumEntries] = lower_left*dof+1;
        Values[NumEntries] = c;
        NumEntries++;
      }
      if (below != -1)
      {
        Indices[NumEntries] = below*dof+2;
        Values[NumEntries] = c;
        NumEntries++;
      }
    }

#ifdef DEBUGGING_
    std::cerr << i << " " << MyGlobalElements[i] << " " << ibase << " " << ivar << std::endl;
    for (int jj=0;jj<NumEntries;jj++)
    {
      std::cerr << Indices[jj] << " ";
    }
    std::cerr << std::endl;
#endif

    // put the off-diagonal entries
    Matrix->InsertGlobalValues(MyGlobalElements[i], NumEntries,
                               &Values[0], &Indices[0]);

  }
  Matrix->FillComplete();
  Matrix->OptimizeStorage();

  return(Matrix);
}

inline
Epetra_CrsMatrix*
Darcy3D(const Epetra_Map* Map,
        const int nx, const int ny, const int nz,
        const double a, const double b,
        PERIO_Flag perio=NO_PERIO,
        char grid_type='C')
{
#ifndef EPETRA_NO_32BIT_GLOBAL_INDICES
  if(Map->GlobalIndicesInt()) {
    if (grid_type == 'C')
      return Darcy3D<int>(Map, nx, ny, nz, a, b, perio);
    else if (grid_type == 'B')
      return DarcyB3D<int>(Map, nx, ny, nz, a, b, perio);
    else if (grid_type == 'T')
      return DarcyTHCM3D<int>(Map, nx, ny, nz, a, b, perio);
    else
      throw "GaleriExt::Matrices::Darcy3D: Unknown grid type";
  }
  else
#endif
#ifndef EPETRA_NO_64BIT_GLOBAL_INDICES
  if(Map->GlobalIndicesLongLong()) {
    if (grid_type == 'C')
      return Darcy3D<long long>(Map, nx, ny, nz, a, b, perio);
    else if (grid_type == 'B')
      return DarcyB3D<long long>(Map, nx, ny, nz, a, b, perio);
    else if (grid_type == 'T')
      return DarcyTHCM3D<long long>(Map, nx, ny, nz, a, b, perio);
    else
      throw "GaleriExt::Matrices::Darcy3D: Unknown grid type";
  }
  else
#endif
    throw "GaleriExt::Matrices::Darcy3D: GlobalIndices type unknown";
}

} // namespace Matrices
} // namespace GaleriExt
#endif
