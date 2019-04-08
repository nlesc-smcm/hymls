// @HEADER
// ************************************************************************
//
//           Galeri: Finite Element and Matrix Generation Package
//                 Copyright (2006) ETHZ/Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions about Galeri? Contact Marzio Sala (marzio.sala _AT_ gmail.com)
//
// ************************************************************************
// @HEADER

#include "GaleriExt_CrsMatrices.h"
#include "Galeri_Exception.h"
#include "Teuchos_ParameterList.hpp"

#include "GaleriExt_Darcy2D.h"
#include "GaleriExt_Darcy3D.h"
#include "GaleriExt_Stokes2D.h"
#include "GaleriExt_Stokes3D.h"

class Epetra_CrsMatrix;
class Epetra_Map;

namespace GaleriExt {

Epetra_CrsMatrix*
CreateCrsMatrix(std::string const &MatrixType, const Epetra_Map* Map,
                Teuchos::ParameterList& List)
{
  // =============== //
  // MATLAB MATRICES //
  // =============== //
  //

  // ========================== //
  // FINITE DIFFERENCE MATRICES //
  // ========================== //
  if (MatrixType == "Darcy2D")
  {
    int nx = List.get("nx", -1);
    int ny = List.get("ny", -1);

    double a = List.get("a", 1.0);
    double b = List.get("b", 1.0);

    return(Matrices::Darcy2D(Map, nx, ny, a, b));
  }
  else if (MatrixType == "Stokes2D")
  {
    int nx = List.get("nx", -1);
    int ny = List.get("ny", -1);

    double a = List.get("a", (double)nx);
    double b = List.get("b", 1.0);

    return(Matrices::Stokes2D(Map, nx, ny, a, b));
  }
  else if (MatrixType == "Darcy3D")
  {
    int nx = List.get("nx", -1);
    int ny = List.get("ny", -1);
    int nz = List.get("nz", -1);

    double a = List.get("a", 1.0);
    double b = List.get("b", 1.0);

    return(Matrices::Darcy3D(Map, nx, ny, nz, a, b));
  }
  else if (MatrixType == "Stokes3D")
  {
    int nx = List.get("nx", -1);
    int ny = List.get("ny", -1);
    int nz = List.get("nz", -1);

    double a = List.get("a", (double)nx);
    double b = List.get("b", 1.0);

    return(Matrices::Stokes3D(Map, nx, ny, nz, a, b));
  }
  else
  {
    throw(Galeri::Exception(__FILE__, __LINE__,
                    "`MatrixType' has incorrect value (" + MatrixType + ")",
                    "in input to function CreateMatrix()",
                    "Check the documentation for a list of valid choices"));
  }
} // CreateMatrix()

} // namespace Galeri
