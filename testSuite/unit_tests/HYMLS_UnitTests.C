#include "HYMLS_UnitTests.H"
#include "Galeri_Random.h"
#include "Epetra_Map.h"
#include "Epetra_Comm.h"

namespace HYMLS {
namespace UnitTests {

// create a Galeri random map with n global IDs and ndof consecutive
// nodes always on the same partition.
Teuchos::RCP<Epetra_Map> create_random_map(const Epetra_Comm& comm, int n, int ndof)
{
  int n1=n/ndof;
  if (n1*ndof!=n) 
  {
    throw "n must be a multiple of ndof!";
  }
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(Galeri::Maps::Random(comm,n1));
  int nloc=map->NumMyElements()*ndof;
  int ibase=map->IndexBase();
  int my_gids[map->NumMyElements()*ndof];
  for (int i=0; i<map->NumMyElements(); i++)
  {
    for (int j=0; j<ndof; j++)
    {
      my_gids[i*ndof+j]=map->GID(i)*ndof+j;
    }
  }
  map=Teuchos::rcp(new Epetra_Map(n,nloc,my_gids,ibase,comm));
  return map;
}

}} // namespaces HYMLS::UnitTests
