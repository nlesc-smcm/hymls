#ifndef HYMLS_PLA_H
#define HYMLS_PLA_H

#include "HYMLS_Tools.H"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterListAcceptorDefaultBase.hpp"
#include "Teuchos_Tuple.hpp"

namespace Teuchos
  {
  class ParameterList;
  }

namespace HYMLS 
  {

  //! base class to be used instead of Teuchos::ParameterListAcceptorDefaultBase.
  //! That name is a bit longish, and we add a simple function PL() to get the   
  //! parameter list, plus a default implementation of getValidParameters() that 
  //! calls SetParameterList().
  class PLA : public Teuchos::ParameterListAcceptorDefaultBase
    {
    
  public:
  
    Teuchos::RCP<const Teuchos::ParameterList> getValidParameters();
    
  protected:
  
  PLA();
  inline Teuchos::ParameterList& PL()
    {
    if (Teuchos::is_null(getMyNonconstParamList()))
      {
      Tools::Error("parameter list is null",__FILE__,__LINE__);
      }
    return *getMyNonconstParamList();
    }
  inline const Teuchos::ParameterList& PL() const 
    {
    if (Teuchos::is_null(getMyParamList()))
      {
      Tools::Error("parameter list is null",__FILE__,__LINE__);
      }
    return *getMyParamList();
    }
     
  };


}

#endif
