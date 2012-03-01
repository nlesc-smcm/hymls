#include "HYMLS_PLA.H"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "HYMLS_Tools.H"

namespace HYMLS {  

PLA::PLA(std::string my_sublist)
  {
  START_TIMER3("PLA","Constructor");

  default_sublist_=my_sublist;  
  validateParameters_=true; // this statement determines wether
                            // parameter validation is applied 
                            // anywhere in HYMLS
  validParams_=Teuchos::null;
  }

PLA::~PLA()
  {
  START_TIMER3("PLA","Destructor");
  }

}
