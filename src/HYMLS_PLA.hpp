#ifndef HYMLS_PLA_H
#define HYMLS_PLA_H

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterListAcceptorDefaultBase.hpp"

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
  Teuchos::ParameterList& PL(std::string sublist="");

  //!
  const Teuchos::ParameterList& PL(std::string sublist="") const;

  //!
  Teuchos::ParameterList& VPL(std::string sublist="") const;

  mutable Teuchos::RCP<Teuchos::ParameterList> validParams_;

  std::string default_sublist_;

  bool validateParameters_;
  };

  }

#endif
