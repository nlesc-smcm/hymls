#include "HYMLS_SeparatorGroup.hpp"

namespace HYMLS
  {

SeparatorGroup::SeparatorGroup()
  :
  nodes_(new Teuchos::Array<hymls_gidx>())
  {}

Teuchos::Array<hymls_gidx> &SeparatorGroup::nodes()
  {
  return *nodes_;
  }

  }
