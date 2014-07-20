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
    
    //! setup HYMLS for using the operator (beta*A-alpha*B)
    static void SetShift(OP& Op, const ST& beta,
                                 const ST& alpha)
    {
      //TODO
    }
    
    //! set projection vectors V: operator should act as inverse of (I-VV')(beta*A-alpha*B)
    static void SetProjectionVectors(OP& Op, const Teuchos::RCP<const MV>& Q)
    {
      //TODO
    }
    
    //! @name Operator application method.
    //@{ 
    
    //! Application method which performs operation <b>y ~= (I-VV')*(A-sigmaB)\x</b>. 
    static void ApplyInverse ( const OP& Op, 
                        const MV& x, 
                        MV& y )
    {
      //TODO
    }
    
    //@}
    
  };
  
} // end Anasazi namespace

#endif // ANASAZI_PRECONDITIONER_TRAITS_HPP

// end of file AnasaziPreconditionerTraits.hpp
