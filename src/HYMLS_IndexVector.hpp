#ifndef HYMLS_INDEXVECTOR_HPP
#define HYMLS_INDEXVECTOR_HPP

#include "HYMLS_config.h"
#include "Epetra_IntSerialDenseVector.h"
#include "Epetra_LongLongSerialDenseVector.h"

namespace HYMLS {
#ifdef HYMLS_LONG_LONG
typedef Epetra_LongLongSerialDenseVector IndexVector;
#else
typedef Epetra_IntSerialDenseVector IndexVector;
#endif
}

#endif
