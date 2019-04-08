#include "HYMLS_PLA.H"

#include "HYMLS_Macros.H"
#include "HYMLS_Tools.H"

#include "Teuchos_ParameterList.hpp"

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

Teuchos::ParameterList& PLA::PL(std::string sublist)
  {
  if (Teuchos::is_null(getMyNonconstParamList()))
    Tools::Error("parameter list not set",__FILE__,__LINE__);

  if (sublist != "")
    return getMyNonconstParamList()->sublist(sublist);

  if (default_sublist_ != "")
    return getMyNonconstParamList()->sublist(default_sublist_);

  return *getMyNonconstParamList();
  }

const Teuchos::ParameterList& PLA::PL(std::string sublist) const
  {
  if (Teuchos::is_null(getMyParamList()))
    Tools::Error("parameter list is null",__FILE__,__LINE__);

  if (sublist != "")
    return getMyParamList()->sublist(sublist);

  if (default_sublist_ != "")
    return getMyParamList()->sublist(default_sublist_);

  return *getMyParamList();
  }

Teuchos::ParameterList& PLA::VPL(std::string sublist) const
  {
  if (Teuchos::is_null(validParams_))
    {
    validParams_ = Teuchos::rcp(new Teuchos::ParameterList());
    }
  if (sublist != "")
    return validParams_->sublist(sublist);

  if (default_sublist_ != "")
    return validParams_->sublist(default_sublist_);

  return *validParams_;
  }

}
