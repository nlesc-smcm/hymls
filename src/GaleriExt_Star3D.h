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

#ifndef GALERIEXT_STAR3D_H
#define GALERIEXT_STAR3D_H

#include "Galeri_Exception.h"
#include "Galeri_Utils.h"
#include "Epetra_Comm.h"
#include "Epetra_BlockMap.h"
#include "Epetra_CrsMatrix.h"

#include "GaleriExt_Periodic.h"

namespace GaleriExt {
namespace Matrices {

inline
Epetra_CrsMatrix* 
Star3D(const Epetra_Map* Map, 
        const int nx, const int ny, const int nz,
        const double a, const double b, const double c, const double d,
        PERIO_Flag perio=NO_PERIO)
{
  Epetra_CrsMatrix* Matrix = new Epetra_CrsMatrix(Copy, *Map,  27);

  int NumMyElements = Map->NumMyElements();
  int* MyGlobalElements = Map->MyGlobalElements();

  int left, right, lower, upper, below, above;
  // lower left (lole), upper right (upri) etc.
  int lole, lori, uple, upri;
  // above left (al), below right (bl) etc.
  int able,abri,bele,beri;
  // above upper (abup), below lower (belo) etc.
  int abup,beup,ablo,belo;
  // below lower left (belole) etc.
  int belole,belori,beuple,beupri;
  // above lower left (ablole) etc.
  int ablole,ablori,abuple,abupri;
  
  int dummy[4];
  
  vector<double> Values(26);
  vector<int> Indices(26);

  //  c b c
  //  b a b
  //  c b c
  // 
  // d c d
  // c b c below and above
  // d c d
  
  for (int i = 0 ; i < NumMyElements ; ++i) 
  {
    int NumEntries = 0;
    
    // first the regular 7-point stencil
    GetNeighboursCartesian3d(MyGlobalElements[i], nx, ny, nz,
			     left, right, lower, upper, below, above,
			     perio);

    if (left != -1) 
      {
      Indices[NumEntries] = left;
      Values[NumEntries] = b;
      ++NumEntries;
      }
    if (right != -1) 
      {
      Indices[NumEntries] = right;
      Values[NumEntries] = b;
      ++NumEntries;
      }
    if (lower != -1) 
      {
      Indices[NumEntries] = lower;
      Values[NumEntries] = b;
      ++NumEntries;
      }
    if (upper != -1) 
      {
      Indices[NumEntries] = upper;
      Values[NumEntries] = b;
      ++NumEntries;
      }
    if (below != -1) 
      {
      Indices[NumEntries] = below;
      Values[NumEntries] = b;
      ++NumEntries;
      }
    if (above != -1) 
      {
      Indices[NumEntries] = above;
      Values[NumEntries] = b;
      ++NumEntries;
      }
    
    // now the 'edges' (c-values)
    lole=-1; lori=-1; uple=-1; upri=-1;
    able=-1; abri=-1; bele=-1; beri=-1;
    ablo=-1; abup=-1; belo=-1; beup=-1;
    
    if (lower!=-1)
      {
      GetNeighboursCartesian3d(lower, nx, ny, nz,
			     lole, lori, dummy[0], dummy[1], dummy[2],dummy[3],
			     perio);
      }
    if (upper!=-1)
      {
      GetNeighboursCartesian3d(upper, nx, ny, nz,
			     uple, upri, dummy[0], dummy[1], dummy[2],dummy[3],
			     perio);
      }
    if (above!=-1)
      {
      GetNeighboursCartesian3d(above, nx, ny, nz,
			     able, abri, ablo, abup, dummy[0],dummy[1],
			     perio);
      }
    if (below!=-1)
      {
      GetNeighboursCartesian3d(below, nx, ny, nz,
			     bele, beri, belo, beup, dummy[0],dummy[1],
			     perio);
      }
  
    if (lole != -1) 
      {
      Indices[NumEntries] = lole;
      Values[NumEntries] = c;
      ++NumEntries;
      }
    if (lori != -1) 
      {
      Indices[NumEntries] = lori;
      Values[NumEntries] = c;
      ++NumEntries;
      }
    if (uple != -1) 
      {
      Indices[NumEntries] = uple;
      Values[NumEntries] = c;
      ++NumEntries;
      }
    if (upri != -1) 
      {
      Indices[NumEntries] = upri;
      Values[NumEntries] = c;
      ++NumEntries;
      }
    if (able != -1) 
      {
      Indices[NumEntries] = able;
      Values[NumEntries] = c;
      ++NumEntries;
      }
    if (abri != -1) 
      {
      Indices[NumEntries] = abri;
      Values[NumEntries] = c;
      ++NumEntries;
      }
    if (ablo != -1) 
      {
      Indices[NumEntries] = ablo;
      Values[NumEntries] = c;
      ++NumEntries;
      }
    if (abup != -1) 
      {
      Indices[NumEntries] = abup;
      Values[NumEntries] = c;
      ++NumEntries;
      }
    if (bele != -1) 
      {
      Indices[NumEntries] = bele;
      Values[NumEntries] = c;
      ++NumEntries;
      }
    if (beri != -1) 
      {
      Indices[NumEntries] = beri;
      Values[NumEntries] = c;
      ++NumEntries;
      }
    if (belo != -1) 
      {
      Indices[NumEntries] = belo;
      Values[NumEntries] = c;
      ++NumEntries;
      }
    if (beup != -1) 
      {
      Indices[NumEntries] = beup;
      Values[NumEntries] = c;
      ++NumEntries;
      }
    
    // and finally the corners (d-values)
    belori=-1; belole=-1; beupri=-1; beuple=-1;
    ablori=-1; ablole=-1; abupri=-1; abuple=-1;

    if (belo!=-1)
      {
      GetNeighboursCartesian3d(belo, nx, ny, nz,
			     belole, belori, dummy[0], dummy[1], dummy[2],dummy[3],
			     perio);
      }
    if (beup!=-1)
      {
      GetNeighboursCartesian3d(beup, nx, ny, nz,
			     beuple, beupri, dummy[0], dummy[1], dummy[2],dummy[3],
			     perio);
      }
    if (ablo!=-1)
      {
      GetNeighboursCartesian3d(ablo, nx, ny, nz,
			     ablole, ablori, dummy[0], dummy[1], dummy[2],dummy[3],
			     perio);
      }
    if (abup!=-1)
      {
      GetNeighboursCartesian3d(abup, nx, ny, nz,
			     abuple, abupri, dummy[0], dummy[1], dummy[2],dummy[3],
			     perio);
      }
  
    if (belole != -1) 
      {
      Indices[NumEntries] = belole;
      Values[NumEntries] = d;
      ++NumEntries;
      }
    if (belori != -1) 
      {
      Indices[NumEntries] = belori;
      Values[NumEntries] = d;
      ++NumEntries;
      }
    if (ablole != -1) 
      {
      Indices[NumEntries] = ablole;
      Values[NumEntries] = d;
      ++NumEntries;
      }
    if (ablori != -1) 
      {
      Indices[NumEntries] = ablori;
      Values[NumEntries] = d;
      ++NumEntries;
      }
    if (beuple != -1) 
      {
      Indices[NumEntries] = beuple;
      Values[NumEntries] = d;
      ++NumEntries;
      }
    if (beupri != -1) 
      {
      Indices[NumEntries] = beupri;
      Values[NumEntries] = d;
      ++NumEntries;
      }
    if (abuple != -1) 
      {
      Indices[NumEntries] = abuple;
      Values[NumEntries] = d;
      ++NumEntries;
      }
    if (abupri != -1) 
      {
      Indices[NumEntries] = abupri;
      Values[NumEntries] = d;
      ++NumEntries;
      }
    
    // put the off-diagonal entries
    Matrix->InsertGlobalValues(MyGlobalElements[i], NumEntries, 
                               &Values[0], &Indices[0]);
    // Put in the diagonal entry
    double diag = a;
	
    Matrix->InsertGlobalValues(MyGlobalElements[i], 1, 
                               &diag, MyGlobalElements + i);
  }
  Matrix->FillComplete();
  Matrix->OptimizeStorage();

  return(Matrix);
}

} // namespace Matrices
} // namespace Galeri
#endif
