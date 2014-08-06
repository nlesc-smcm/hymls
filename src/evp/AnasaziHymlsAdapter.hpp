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
      Op.SetShift(shiftA,shiftB);
    }
    
    //! set projection vectors V: operator should act as inverse of (I-VV')(beta*A-alpha*B)
    static void SetProjectionVectors(OP& Op, const Teuchos::RCP<const MV>& Q)
    {
      CHECK_ZERO(Op.setNullSpace(Q));
      CHECK_ZERO(Op.SetupDeflation());
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