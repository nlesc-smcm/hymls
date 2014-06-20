
#include "HYMLS_View_MultiVector.H"

#include <Epetra_MultiVector.h>
#include <Epetra_BlockMap.h>

namespace HYMLS {

MultiVector_View::
~MultiVector_View()
{
}

Teuchos::RCP<Epetra_MultiVector>
MultiVector_View::
operator()( Teuchos::RCP<Epetra_MultiVector> orig )
{
  START_TIMER3("MultiVector_View", "operator (1)");
  // make sure the original is not deleted before this view object is
  origObj_.append(orig);

  int numVec = orig->NumVectors();
  
  int origLDA;
  double * ptr;
  orig->ExtractView( &ptr, &origLDA );

  Teuchos::RCP<Epetra_MultiVector> newMV = 
        Teuchos::rcp(new Epetra_MultiVector( View, NewMap_, 
        ptr+Offset_, origLDA, numVec));

  return newMV;
}

Teuchos::RCP<const Epetra_MultiVector>
MultiVector_View::
operator()(Teuchos::RCP<const Epetra_MultiVector> orig )
  {
  START_TIMER3("MultiVector_View", "operator (2)");
  // make sure the original is not deleted before this view object is
  origObj_.append(orig);

  int numVec = orig->NumVectors();
  
  int origLDA;
  double * ptr;
  orig->ExtractView( &ptr, &origLDA );

  Teuchos::RCP<const Epetra_MultiVector> newMV = Teuchos::rcp(new Epetra_MultiVector( View, NewMap_, 
        ptr+Offset_, origLDA, numVec));

  return newMV;
  }

Teuchos::RCP<Epetra_MultiVector>
MultiVector_View::operator()(Epetra_MultiVector& orig )
  {
  START_TIMER3("MultiVector_View", "operator (3)");
  return this->operator()(Teuchos::rcp(&orig,false));
  }

Teuchos::RCP<const Epetra_MultiVector>
MultiVector_View::operator()(const Epetra_MultiVector& orig )
  {
  START_TIMER3("MultiVector_View", "operator (4)");
  return this->operator()(Teuchos::rcp(&orig,false));
  }


} // namespace HYMLS

