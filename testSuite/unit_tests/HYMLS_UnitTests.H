#ifndef HYMLS_UNITTESTS_H
#define HYMLS_UNITTESTS_H

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_RCP.hpp"

class Epetra_IntVector;

#define TODO_TEST_EQUALITY( v1, v2 ) \
  TEST_INEQUALITY( v1, v2 )

class Epetra_Comm;
class Epetra_Map;

namespace HYMLS {
namespace UnitTests {

//! create a Galeri random map with n global IDs and ndof consecutive
//! nodes always on the same partition.
Teuchos::RCP<Epetra_Map> create_random_map(const Epetra_Comm& comm, int n, int ndof);

//! compute the inf-norm of the difference of two IntVectors
int NormInfAminusB(const Epetra_IntVector& A, const Epetra_IntVector& B);

}} // namespaces HYMLS::UnitTests
#endif
