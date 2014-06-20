#ifndef HYMLS_MULTIVECTOR_VIEW_H
#define HYMLS_MULTIVECTOR_VIEW_H

#include "Epetra_BlockMap.h"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "HYMLS_Tools.H"
#include "HYMLS_Macros.H"

class Epetra_MultiVector;

namespace HYMLS {

//! Generates a sub-block view of a Epetra_MultiVector. This is an adaptation of
//! EpetraExt::MultiVector_View which allows to view parts of an MV with an offset.
//! If I understand the original class correctly, it could only reindex MV's starting
//! at the first element. This class still requires the new map to be a contiguous subset
//! of the old map, but it may start anywhere inside the old map. A typical use of this
//! would be to view a block of a vector, i.e. v(5:18).
//! In contrast to the EpetraExt view classes, this object is memory safe and returns RCP's
//! rather than references to internal raw pointers.
class MultiVector_View 
  {
  const Epetra_BlockMap & OrigMap_;
  const Epetra_BlockMap & NewMap_;

  int Offset_;  

 public:

  //! Destructor
  ~MultiVector_View();

  //! Constructor
  MultiVector_View( const Epetra_BlockMap & orig_map,
                    const Epetra_BlockMap & new_map)
  : OrigMap_(orig_map),
    NewMap_(new_map)
  {
      START_TIMER2("MultiVector_View", "Constructor");
  Offset_=0;
  while (OrigMap_.GID(Offset_)!=NewMap_.GID(0) && Offset_<OrigMap_.NumMyElements())
    {
    Offset_++;
    }
  if (Offset_+NewMap_.NumMyElements()>OrigMap_.NumMyElements())
    {
    std::cout << "MultiVector_View: size of original map: "<<OrigMap_.NumMyElements()<<std::endl;
    std::cout << "MultiVector_View: size of view: "<<NewMap_.NumMyElements()<<std::endl;
    std::cout << "MultiVector_View: offset = "<< Offset_<<std::endl;
    throw("HYMLS::MultiVector_View: bad submap!");
    }
  }

  //! Transformation Operator
  Teuchos::RCP<Epetra_MultiVector> operator()
        (Teuchos::RCP<Epetra_MultiVector> orig );

  //! Transformation Operator (const version)
  Teuchos::RCP<const Epetra_MultiVector> operator()
        (Teuchos::RCP<const Epetra_MultiVector> orig );

  //! Transformation Operator
  Teuchos::RCP<Epetra_MultiVector> operator()
        (Epetra_MultiVector& orig );

  //! Transformation Operator (const version)
  Teuchos::RCP<const Epetra_MultiVector> operator()
        (const Epetra_MultiVector& orig );

private:

//! store an RCP to each object that the user wants to view so the original
//! can't be deleted before the view.
Teuchos::Array<Teuchos::RCP<const Epetra_MultiVector> > origObj_;

};

} //namespace HYMLS

#endif //HYMLS_MULTIVECTOR_VIEW_H
