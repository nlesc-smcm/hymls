#ifndef HYMLS_PLA_H
#define HYMLS_PLA_H

#include "HYMLS_config.h"

#include "HYMLS_Tools.H"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterListAcceptorDefaultBase.hpp"
#include "Teuchos_Tuple.hpp"
#include "Teuchos_ParameterList.hpp"

namespace HYMLS 
  {

  //! base class to be used instead of Teuchos::ParameterListAcceptorDefaultBase. 
  //! That name is a bit longish, and we add a member parameter list for valid    
  //! parameters, plus a number of handy access functions to the internal lists.  
  //!                                                                             
  //! Guidelines for treatment of Parameters in HYMLS                             
  //! +++++++++++++++++++++++++++++++++++++++++++++++                             
  //!                                                                             
  //! Each class is passed the global parameter list but is responsible for vali- 
  //! dating only a certain sublist. So, for instance:                            
  //!   - HYMLS::Solver validates "Solver"                                        
  //!   - HYMLS::Preconditioner validates "Preconditioner" and modifies "Problem".
  //!   - HYMLS::OverlappingPartitioner validates the "Problem" sublist.          
  class PLA : public virtual Teuchos::ParameterListAcceptorDefaultBase
    {
    
  public:
  
    virtual ~PLA();
    
  protected:

  //! constructor - if a sublist name is passed in , PL() will return that 
  //! sublist rather then the global list.
  PLA(std::string default_sublist="");

  //! get ref to internal list  
  inline Teuchos::ParameterList& PL(std::string sublist="")
    {
    if (Teuchos::is_null(getMyNonconstParamList()))
      {
      Tools::Error("parameter list not set",__FILE__,__LINE__);
      }
    if (sublist!="") return getMyNonconstParamList()->sublist(sublist);
    if (default_sublist_!="") return getMyNonconstParamList()->sublist(default_sublist_);
    return *getMyNonconstParamList();
    }
  //!
  inline const Teuchos::ParameterList& PL(std::string sublist="") const 
    {
    if (Teuchos::is_null(getMyParamList()))
      {
      Tools::Error("parameter list is null",__FILE__,__LINE__);
      }
    if (sublist!="") return getMyParamList()->sublist(sublist);
    if (default_sublist_!="") return getMyParamList()->sublist(default_sublist_);
    return *getMyParamList();
    }

  //!
  inline Teuchos::ParameterList& VPL(std::string sublist="") const 
    {
    if (Teuchos::is_null(validParams_))
      {
      validParams_=Teuchos::rcp(new Teuchos::ParameterList());
      }
    if (sublist!="") return validParams_->sublist(sublist);
    if (default_sublist_!="") return validParams_->sublist(default_sublist_);
    return *validParams_;
    }

  mutable Teuchos::RCP<Teuchos::ParameterList> validParams_;
  
  std::string default_sublist_;

  bool validateParameters_;     
  };


}

#endif
