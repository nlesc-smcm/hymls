#include "HYMLS_UnitTests.H"
#include "Galeri_Random.h"
#include "Epetra_Map.h"
#include "Epetra_Comm.h"
#include "Epetra_IntVector.h"
#include "Epetra_MultiVector.h"

#include "HYMLS_Tools.H"

namespace HYMLS {
namespace UnitTests {

DisableOutput::DisableOutput()
  :
  no_output(Teuchos::rcp(new Teuchos::oblackholestream()))
  {
  HYMLS::Tools::InitializeIO_std(Teuchos::null, no_output, no_output);
  }

DisableOutput::~DisableOutput()
  {
  HYMLS::Tools::InitializeIO(Teuchos::null);
  }

void DisableOutput::EnableOutput()
  {
  HYMLS::Tools::InitializeIO(Teuchos::null);
  }

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
  int *my_gids = new int[nloc];
  for (int i=0; i<map->NumMyElements(); i++)
  {
    for (int j=0; j<ndof; j++)
    {
      my_gids[i*ndof+j] = map->GID(i)*ndof+j;
    }
  }
  map = Teuchos::rcp(new Epetra_Map(n,nloc,my_gids,ibase,comm));
  delete[] my_gids;
  return map;
}

Teuchos::RCP<Epetra_Map> create_random_map(const Epetra_Comm& comm, long long n, int ndof)
{
  int n1=n/ndof;
  if (n1*ndof!=n) 
  {
    throw "n must be a multiple of ndof!";
  }
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(Galeri::Maps::Random(comm,n1));
  int nloc=map->NumMyElements()*ndof;
  long long ibase=map->IndexBase64();
  long long *my_gids = new long long[nloc];
  for (int i=0; i<map->NumMyElements(); i++)
  {
    for (int j=0; j<ndof; j++)
    {
      my_gids[i*ndof+j] = map->GID64(i)*ndof+j;
    }
  }
  map = Teuchos::rcp(new Epetra_Map(n,nloc,my_gids,ibase,comm));
  delete[] my_gids;
  return map;
}

// helper function for comparing Epetra_IntVectors
int NormInfAminusB(const Epetra_IntVector& A, const Epetra_IntVector& B)
{
  if (!A.Map().SameAs(B.Map()))
    return -1;

  int value=0;
  for (int i=0; i<A.MyLength(); i++)
  {
    value=std::max(value,std::abs(A[i]-B[i]));
  }
  int global_value;
  A.Map().Comm().MaxAll(&value,&global_value,1);
  return global_value;
}

// helper function for comparing Epetra_MultiVectors
double NormInfAminusB(const Epetra_MultiVector& A, const Epetra_MultiVector& B)
{
  if (!A.Map().SameAs(B.Map()))
    return 1;

  double value = 0;
  for (int i = 0; i < A.MyLength(); i++)
    {
    for (int j = 0; j < A.NumVectors(); j++)
      {
      value = std::max(value, std::abs(A[j][i]-B[j][i]));
      }
    }

  double global_value;
  A.Map().Comm().MaxAll(&value, &global_value, 1);
  return global_value;
}


}} // namespaces HYMLS::UnitTests
