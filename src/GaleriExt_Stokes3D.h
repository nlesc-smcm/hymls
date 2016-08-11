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

#ifndef GALERIEXT_STOKES3D_H
#define GALERIEXT_STOKES3D_H

#include "Galeri_Exception.h"
#include "Galeri_Utils.h"
#include "Epetra_Comm.h"
#include "Epetra_BlockMap.h"
#include "Epetra_CrsMatrix.h"

#include "GaleriExt_Periodic.h"
#include "Galeri_Cross3D.h"
#include "GaleriExt_Darcy3D.h"

namespace GaleriExt {
namespace Matrices {

//! generate Stokes problem on a C grid with
//! the A part scaled by a and the B part scaled
//! by b, K=[A B; B' 0];
inline Epetra_CrsMatrix* 
Stokes3D(const Epetra_Map* Map, 
        const int nx, const int ny, const int nz,
        const double a, const double b, 
        PERIO_Flag perio=NO_PERIO
        )
{
  Teuchos::RCP<Epetra_CrsMatrix> Matrix = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *Map,  9));
  Teuchos::RCP<Epetra_CrsMatrix> Darcy  = Teuchos::rcp(Darcy3D(Map,nx,ny,nz,0.0,-b,perio));
  
  
  int dof=4;
  int NumMyElements1 = Map->NumMyElements()/dof;
  int NumGlobalElements1=Map->NumGlobalElements()/dof;
  Teuchos::ArrayRCP<int> MyGlobalElements1;
  MyGlobalElements1.resize(NumMyElements1);
  int pos=0;
  
  
  for (int i=0; i<Map->NumMyElements();i++)
  {
    int gid=Map->GID(i);
    if (gid%dof==0) MyGlobalElements1[pos++]=gid/dof;
  }
  Teuchos::RCP<Epetra_Map> Map1=Teuchos::rcp(new Epetra_Map(NumGlobalElements1,NumMyElements1,&MyGlobalElements1[0],Map->IndexBase(),Map->Comm()));
 
  Teuchos::RCP<Epetra_CrsMatrix> Laplace=Teuchos::null;
  if (perio==NO_PERIO)
  {
    Laplace=Teuchos::rcp(Galeri::Matrices::Cross3D(Map1.get(),nx,ny,nz,4,-1,-1,-1,-1,-1,-1));
  }
  else
  {
    throw "GaleriExt::Stokes3D doesn't work with periodic BC yet";
    //Laplace=Teuchos::rcp(Cross3DN(Map1.get(),nx,ny,4,-1,-1,-1,-1,-1,-1));
  }
  // now create the combined Stokes matrix [A B'; B 0] from A=[Laplace 0; 0 Laplace] and Darcy=[I B'; B 0];
  for (int i=0; i<Map->NumMyElements(); i++)
  {
    int row = Map->GID(i);
    const int max_len=9;
    int cols[max_len], cols_laplace[max_len];
    double vals[max_len],vals_laplace[max_len];
    int lenDarcy,lenLaplace;
    Darcy->ExtractGlobalRowCopy(row,max_len,lenDarcy,vals,cols);
    // u or v node? add Laplace row entries
    int lenTotal=lenDarcy;
    if ((row+1)%dof)
    {
      int row0 = row/dof;
      Laplace->ExtractGlobalRowCopy(row0,max_len,lenLaplace,vals_laplace,cols_laplace);
      // compensation for missing nodes at the boundary (gives Dirichlet boundary conditions)
      double add_to_diag = 0.0;
      int left,right,lower,upper,below,above;
      int dum1,dum2,dum3,dum4,dum5,rightright,upup,toptop;
      GetNeighboursCartesian3d(row0, nx, ny, nz, left, right, lower, upper, below, above, perio);

      if (right>0) GetNeighboursCartesian3d(right, nx, ny, nz, dum1, rightright, dum2, dum3, dum4, dum5, perio);
      if (upper>0) GetNeighboursCartesian3d(upper, nx, ny, nz, dum1, dum2, dum3, upup, dum4, dum5, perio);
      if (above>0) GetNeighboursCartesian3d(above, nx, ny, nz, dum1, dum2, dum3, dum4, dum5, toptop, perio);

        // the left and border look like this:
        //   
        // /+-v-+               +-v-+/
        // /|   |               |   |/
        // /| p u       ....    u p u/
        // /|   |               |   |/
        // /+---+               +---+/

      if ((row+1)%dof==1) // u-variable
      {
        if (right==-1)
        {
          lenLaplace=1;
          vals_laplace[0]=b/(a*a); 
          cols_laplace[0]=row0;
        }
        else 
        {
        if (lower==-1 || upper==-1) add_to_diag+=a;
        if (below==-1 || above==-1) add_to_diag+=a;
        }
        if (right>=0 && rightright==-1)
        {
          // remove coupling to velocity on boundary
          for (int j=0; j<lenLaplace; j++)
          {
            if (cols_laplace[j]==right) vals_laplace[j]=0.0;
          }
        }
      }
      else if ((row+1)%dof==2) // v-variable
      {
        if (upper==-1)
        {
          lenLaplace=1;
          vals_laplace[0]=b/(a*a);
          cols_laplace[0]=row0;
        }
        else
        {
        if (left==-1 || right==-1) add_to_diag+=a;
        if (below==-1 || above==-1) add_to_diag+=a;
        }
        if (upper>0 && upup==-1)
        {
          // remove coupling to velocity on boundary
          for (int j=0; j<lenLaplace; j++)
          {
            if (cols_laplace[j]==upper) vals_laplace[j]=0.0;
          }
        }
      }
      else if ((row+1)%dof==3) // w-variable
      {
        if (above==-1)
        {
          lenLaplace=1;
          vals_laplace[0]=b/(a*a);
          cols_laplace[0]=row0;
        }
        else
        {
        if (left==-1 || right==-1) add_to_diag+=a;
        if (lower==-1 || upper==-1) add_to_diag+=a;
        }
        if (above>0 && toptop==-1)
        {
          // remove coupling to velocity on boundary
          for (int j=0; j<lenLaplace; j++)
          {
            if (cols_laplace[j]==above) vals_laplace[j]=0.0;
          }
        }
      }
      
      for (int j=0; j<lenLaplace; j++)
      {
        // column index in final Stokes matrix
        int c=cols_laplace[j]*dof + row%dof;
        if (c==row)
        {
          // find entry in existing values and replace it
          for (int k=0; k<lenDarcy; k++)
          {
            if (cols[k]==c) vals[k]=-(vals_laplace[j]*a + add_to_diag);
          }
        }
        else
        {
          // add entry at the ent of the row
          cols[lenTotal]=c;
          vals[lenTotal]=-vals_laplace[j]*a;
          lenTotal++;
        }
      }
    }
    Matrix->InsertGlobalValues(row,lenTotal,vals,cols);
  }
  Matrix->FillComplete();
  return Matrix.release().get();
}

} // namespace Matrices
} // namespace Galeri
#endif
