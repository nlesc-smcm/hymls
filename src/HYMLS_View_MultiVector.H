#ifndef HYMLS_MULTIVECTOR_VIEW_H
#define HYMLS_MULTIVECTOR_VIEW_H

#include <EpetraExt_Transform.h>
#include "Epetra_BlockMap.h"

class Epetra_MultiVector;

namespace HYMLS {

//! Generates a sub-block view of a Epetra_MultiVector. This is an adaptation of
//! EpetraExt::MultiVector_View which allows to view parts of an MV with an offset.
//! If I understand the original class correctly, it could only reindex MV's starting
//! at the first element. This class still requires the new map to be a contiguous subset
//! of the old map, but it may start anywhere inside the old map. A typical use of this
//! would be to view a block of a vector, i.e. v(5:18).
class MultiVector_View : public EpetraExt::ViewTransform<Epetra_MultiVector> {

  const Epetra_BlockMap & OrigMap_;
  const Epetra_BlockMap & NewMap_;

  const int NumVec_;
  int Offset_;  

 public:

  //! Destructor
  ~MultiVector_View();

  //! Constructor
  MultiVector_View( const Epetra_BlockMap & orig_map,
                    const Epetra_BlockMap & new_map,
                    const int num_vec = -1 )
  : OrigMap_(orig_map),
    NewMap_(new_map),
    NumVec_(num_vec)
  {
  Offset_=0;
  while (OrigMap_.GID(Offset_)!=NewMap_.GID(0) && Offset_<OrigMap_.NumMyElements())
    {
    Offset_++;
    }
  if (Offset_+NewMap_.NumMyElements()>OrigMap_.NumMyElements())
    {
    throw("HYMLS::MultiVector_View: bad submap!");
    }
  }

  //! Transformation Operator
  NewTypeRef operator()( OriginalTypeRef orig );

  //! Transformation Operator (const version)
  const Epetra_MultiVector& operator()(const Epetra_MultiVector& orig );

};

} //namespace HYMLS

#endif //HYMLS_MULTIVECTOR_VIEW_H
