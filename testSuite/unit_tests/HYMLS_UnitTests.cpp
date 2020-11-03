#include "HYMLS_UnitTests.hpp"
#include "Galeri_Random.h"
#include "Epetra_Map.h"
#include "Epetra_Comm.h"
#include "Epetra_IntVector.h"
#include "Epetra_MultiVector.h"
#include "Epetra_SerialDenseMatrix.h"

#include "HYMLS_DenseUtils.hpp"
#include "HYMLS_Tools.hpp"

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

// helper function for comparing Epetra_MultiVectors
double NormInfAminusB(const Epetra_SerialDenseMatrix& A, const Epetra_SerialDenseMatrix& B)
{
    return NormInfAminusB(*DenseUtils::CreateView(A), *DenseUtils::CreateView(B));
}

Teuchos::RCP<Epetra_SerialDenseMatrix> RandomSerialDenseMatrix(int m, int n, const Epetra_Comm& comm)
{
    Teuchos::RCP<Epetra_SerialDenseMatrix> A = Teuchos::rcp(new Epetra_SerialDenseMatrix(m, n));
    A->Random();

    Teuchos::RCP<Epetra_SerialDenseMatrix> A_max = Teuchos::rcp(new Epetra_SerialDenseMatrix(m, n));

    comm.MaxAll(A->A(), A_max->A(), m * n);

    return A_max;
}

}} // namespaces HYMLS::UnitTests
