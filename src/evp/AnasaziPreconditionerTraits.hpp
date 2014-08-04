// @HEADER
// ***********************************************************************
//
//                 Anasazi: Block Eigensolvers Package
//                 Copyright (2004) Sandia Corporation
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
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ***********************************************************************
// @HEADER

#ifndef ANASAZI_PRECONDITIONER_TRAITS_HPP
#define ANASAZI_PRECONDITIONER_TRAITS_HPP

/*!     \file AnasaziPreconditionerTraits.hpp
        \brief Virtual base class which defines basic traits for the operation 
        $(I-QQ^T)(beta*A-alpha*B)^{~1}$
*/

#include "AnasaziConfigDefs.hpp"
#include "AnasaziTypes.hpp"
#include "Teuchos_RCP.hpp"

namespace Anasazi {

  /*! \brief This is the default struct used by PreconditionerTraits<ScalarType, MV, OP> class to produce a
      compile time error when the specialization does not exist for operator type <tt>OP</tt>.
  */
  template< class ScalarType, class MV, class OP >
  struct UndefinedPreconditionerTraits
  {
    //! This function should not compile if there is an attempt to instantiate!
    /*! \note Any attempt to compile this function results in a compile time error.  This means
      that the template specialization of Anasazi::PreconditionerTraits class does not exist for type
      <tt>OP</tt>, or is not complete.
    */
    static inline void notDefined() { return OP::this_type_is_missing_a_specialization(); };
  };


  /*!  \brief Virtual base class which defines basic traits for an operator that acts as the 
  inverse of (I-VV')(A-sigma*B)

       An adapter for this traits class must exist for the <tt>MV</tt> and <tt>OP</tt> types.
       If not, this class will produce a compile-time error.

       \ingroup anasazi_opvec_interfaces
  */
  template <class ScalarType, class MV, class OP>
  class PreconditionerTraits 
  {
  public:
    
    //! set the shift $\sigma$ to be used in the preconditioner. The matrix is assumed
    //! to be fixed during the course of an Anasazi solve (and known to the preconditioner),
    //! the preconditioner may decide to which extend it makes use of updated shifts in a
    //! Jacobi-Davidson process. The shift is given as two factors, the system to be 
    //! solved will be (shiftA*A+shiftB*B)X=RHS
    static void SetShift(OP& Op, const ScalarType& shiftA, const ScalarType& shiftB)
    {
      UndefinedPreconditionerTraits<ScalarType,MV,OP>::notDefined();
    }
    
    //! set projection vectors V: operator should act as inverse of (I-VV')(A-sigma*B)
    static void SetProjectionVectors(OP& Op, const Teuchos::RCP<const MV>& Q)
    {
      UndefinedPreconditionerTraits<ScalarType,MV,OP>::notDefined();
    }
    
    //! @name Operator application method.
    //@{ 
    
    //! Application method which performs operation <b>y ~= (I-VV')*(A-sigmaB)\x</b>. 
    static void ApplyInverse ( const OP& Op, 
                        const MV& x, 
                        MV& y )
    {
      UndefinedPreconditionerTraits<ScalarType,MV,OP>::notDefined();
    }
    
    //@}
    
  };
  
} // end Anasazi namespace

#endif // ANASAZI_PRECONDITIONER_TRAITS_HPP

// end of file AnasaziPreconditionerTraits.hpp
