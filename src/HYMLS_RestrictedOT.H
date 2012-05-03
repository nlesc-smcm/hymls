#ifndef HYMLS_RESTRICTED_OT_H
#define HYMLS_RESTRICTED_OT_H

#include "HYMLS_Tools.H"
#include "HYMLS_OrthogonalTransform.H"
#include "Epetra_IntSerialDenseVector.h"
#include "Epetra_SerialDenseMatrix.h"

namespace HYMLS {

//! this class contains some static member functions for
//! doing operations like H'X(idx,idx)H with a given    
//! HYMLS::OrthogonalTransform H
class RestrictedOT {

public:

//! in place transformation of some rows and cols of X
inline static int Apply(Epetra_SerialDenseMatrix& X,
                 const Epetra_IntSerialDenseVector& idx,
                 const HYMLS::OrthogonalTransform& OT,
                 const Epetra_SerialDenseVector& v)
  {
  int n=idx.Length();
  if (n<=0) return 0;
  // for now, the indices must be contiguous. This is typically the case
  // in the way we use this class
#ifdef TESTING
  for (int i=1;i<n;i++)
    if (idx[i]-idx[i-1]!=1)
      {
      Tools::Error("indices must be contiguous",__FILE__,__LINE__);
      }
#endif  
  Epetra_SerialDenseMatrix Cols(View, X[idx[0]], X.LDA(),X.M(),n);
  Epetra_SerialDenseMatrix Rows(View, &X[0][idx[0]], X.LDA(),n,X.N());
  
  CHECK_ZERO(OT.Apply(Rows,v));
  CHECK_ZERO(OT.ApplyR(Cols,v));

  return 0;
  }                 

};

}

#endif
