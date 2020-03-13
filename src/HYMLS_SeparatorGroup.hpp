#ifndef HYMLS_SEPARATOR_GROUP_H
#define HYMLS_SEPARATOR_GROUP_H

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "HYMLS_config.h"

namespace HYMLS
  {

class SeparatorGroup
  {
  Teuchos::RCP<Teuchos::Array<hymls_gidx> > nodes_;

  int type_;

public:
  SeparatorGroup();

  hymls_gidx const &operator[](int i) const;

  Teuchos::Array<hymls_gidx> &nodes();

  Teuchos::Array<hymls_gidx> const &nodes() const;

  int length() const;

  Teuchos::Array<hymls_gidx> &append(hymls_gidx gid);

  void sort();

  int type() const;

  void set_type(int type);
  };

  }
#endif
