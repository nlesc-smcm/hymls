#include "HYMLS_PLA.H"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

namespace HYMLS {  

PLA::PLA()
  {
  setMyParamList(Teuchos::rcp(new Teuchos::ParameterList()));
  }

Teuchos::RCP<const Teuchos::ParameterList> PLA::getValidParameters()
  {
  Teuchos::RCP<Teuchos::ParameterList> saveParams = getMyNonconstParamList();
  Teuchos::RCP<Teuchos::ParameterList> validParams = 
        Teuchos::rcp(new Teuchos::ParameterList());
  setMyParamList(validParams);
  setParameterList(validParams);
  if (saveParams!=Teuchos::null)
    {
    setMyParamList(saveParams);
    setParameterList(saveParams);
    }
  return validParams;
  }
}
