#include "HYMLS_PLA.H"

#include "HYMLS_Macros.H"

namespace HYMLS {

PLA::PLA(std::string my_sublist)
  {
  HYMLS_PROF3("PLA","Constructor");

  default_sublist_=my_sublist;  
  validateParameters_=true; // this statement determines wether
                            // parameter validation is applied 
                            // anywhere in HYMLS
  validParams_=Teuchos::null;
  }

PLA::~PLA()
  {
  HYMLS_PROF3("PLA","Destructor");
  }

}
