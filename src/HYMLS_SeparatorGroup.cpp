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

Teuchos::Array<hymls_gidx> const &SeparatorGroup::nodes() const
  {
  return *nodes_;
  }

int SeparatorGroup::length() const
  {
  return nodes_->length();
  }

Teuchos::Array<hymls_gidx> &SeparatorGroup::append(hymls_gidx gid)
  {
  return nodes_->append(gid);
  }

  }
