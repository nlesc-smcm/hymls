#ifndef HYMLS_RESTRICTED_OT_H
#define HYMLS_RESTRICTED_OT_H

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
                 const HYMLS::OrthogonalTransform& OT)
  {
  int n=idx.Length();
  if (n<=0) return 0;
  Epetra_SerialDenseMatrix Y(n,n);
  for (int i=0;i<n;i++) 
    for (int j=0;j<n;j++)
      {
      Y(j,i) = X(idx[j],idx[i]);
      }
  OT.Apply(Y);
  for (int i=0;i<n;i++) 
    for (int j=0;j<n;j++)
      {
      X(idx[j],idx[i])=Y(j,i);
      }
  return 0;
  }                 

};

}

#endif
