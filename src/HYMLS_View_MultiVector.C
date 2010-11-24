
#include "HYMLS_View_MultiVector.H"

#include <Epetra_MultiVector.h>
#include <Epetra_BlockMap.h>

namespace HYMLS {

MultiVector_View::
~MultiVector_View()
{
  if( newObj_ ) delete newObj_;
}

MultiVector_View::NewTypeRef
MultiVector_View::
operator()( OriginalTypeRef orig )
{
  origObj_ = &orig;

  int numVec = NumVec_;
  if( numVec == -1 ) numVec = orig.NumVectors();
  
  int origLDA;
  double * ptr;
  orig.ExtractView( &ptr, &origLDA );

  Epetra_MultiVector * newMV = new Epetra_MultiVector( View, NewMap_, 
        ptr+Offset_, origLDA, numVec);

  newObj_ = newMV;

  return *newMV;
}

const Epetra_MultiVector&
MultiVector_View::
operator()(const Epetra_MultiVector& orig )
  {
  origObj_ = const_cast<Epetra_MultiVector*>(&orig);

  int numVec = NumVec_;
  if( numVec == -1 ) numVec = orig.NumVectors();
  
  int origLDA;
  double * ptr;
  orig.ExtractView( &ptr, &origLDA );

  Epetra_MultiVector * newMV = new Epetra_MultiVector( View, NewMap_, 
        ptr+Offset_, origLDA, numVec);

  newObj_ = newMV;

  return *newMV;
  }

} // namespace EpetraExt

