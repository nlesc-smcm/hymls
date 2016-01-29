#ifndef HYMLS_RESTRICTED_OT_H
#define HYMLS_RESTRICTED_OT_H

#include "HYMLS_Tools.H"
#include "HYMLS_OrthogonalTransform.H"
#include "Epetra_SerialDenseVector.h"
#include "Epetra_SerialDenseMatrix.h"

namespace HYMLS {

//! this class contains some static member functions for
//! doing operations like H'*X(idx:idx+length(v),idx:idx+length(v))*H
//! with a given HYMLS::OrthogonalTransform H and
//! a given starting index idx
class RestrictedOT {

public:

//! in place transformation of some rows and cols of X
inline static int Apply(Epetra_SerialDenseMatrix& X,
  int idx,
  const HYMLS::OrthogonalTransform& OT,
  const Epetra_SerialDenseVector& v)
  {
  int n = v.Length();
  if (n <= 0) return 0;

  Epetra_SerialDenseMatrix Cols(View, X[idx], X.LDA(), X.M(), n);
  Epetra_SerialDenseMatrix Rows(View, &X[0][idx], X.LDA(), n, X.N());

  CHECK_ZERO(OT.Apply(Rows, v));
  CHECK_ZERO(OT.ApplyR(Cols, v));

  return 0;
  }

};

}

#endif
