#ifndef HYMLS_INTERIOR_GROUP_H
#define HYMLS_INTERIOR_GROUP_H

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "HYMLS_SeparatorGroup.hpp"

namespace HYMLS
  {

class InteriorGroup: public SeparatorGroup
  {
public:
  InteriorGroup()
    :
    SeparatorGroup()
    {}

  };

  }
#endif
