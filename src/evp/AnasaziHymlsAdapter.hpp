#ifndef ANASAZI_HYMLS_ADAPTER_HPP
#define ANASAZI_HYMLS_ADAPTER_HPP


#include "AnasaziPreconditionerTraits.hpp"
#include "Epetra_MultiVector.h"
#include "HYMLS_Solver.H"

namespace Anasazi {

  //!
  template <>
  class PreconditionerTraits<double,Epetra_MultiVector, HYMLS::Solver> 
  {
  public:
    
    typedef double ST;
    typedef Epetra_MultiVector MV;
    typedef HYMLS::Solver OP;
    
    //! setup HYMLS for using the operator (shiftA*A+shiftB*B)
    static void SetShift(OP& Op, const ST& shiftA,
                                 const ST& shiftB)
    {
      Op.setShift(shiftA,shiftB);
    }
    
    //! set projection vectors V: operator should act as inverse of (I-VV')(beta*A-alpha*B)
    static void SetProjectionVectors(OP& Op, const Teuchos::RCP<const MV>& Q)
    {
      // I'm experimenting with what to do here

#define VARIANT_B
#ifdef VARIANT_A
      // this function sets V as border in the ILU preconditioner
      // and replaces (A-sigma*I) by (I-VV')(A-sigma*I):
      CHECK_ZERO(Op.setProjectionVectors(Q));
#elif defined(VARIANT_B)      
      // this function adds V to the border, which may contain an exact null space
      // already (as for Navier-Stokes)
      CHECK_ZERO(Op.setNullSpace(Q));
      // this function 
      // * adds V as border to the ILU preconditioner
      // * replaces the system to be solved by a block ILU,
      //   where the top left part is (I-VV')(A-\sigmaI)
      //   and the possibly ill-conditioned Schur-complement
      //   is solved by an SVD (I think...)
      CHECK_ZERO(Op.SetupDeflation());
#endif
    }
    
    //! @name Operator application method.
    //@{ 
    
    //! Application method which performs operation <b>y ~= (I-VV')*(A-sigmaB)\x</b>. 
    static void ApplyInverse ( const OP& Op, 
                        const MV& x, 
                        MV& y )
    {
      int ierr=Op.ApplyInverse(x,y);
      if (ierr<0)
        {
        throw Anasazi::AnasaziError("Error code "+Teuchos::toString(ierr)+" returned from Op.ApplyInverse()");
        }
    }
    
    //@}
    
  };
  
} // end Anasazi namespace

#endif // ANASAZI_PRECONDITIONER_TRAITS_HPP

// end of file AnasaziPreconditionerTraits.hpp
