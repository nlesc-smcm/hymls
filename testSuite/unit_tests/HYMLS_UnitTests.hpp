#ifndef HYMLS_UNITTESTS_H
#define HYMLS_UNITTESTS_H

#include "HYMLS_config.h"

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_RCP.hpp"

#define TODO_TEST_EQUALITY( v1, v2 ) \
  TEST_INEQUALITY( v1, v2 )

#define TEST_MAYTHROW(code)                     \
    {                                           \
        try {                                   \
            code;                               \
        }                                       \
        catch (...) {                           \
        }                                       \
    }

class Epetra_Comm;
class Epetra_Map;

class Epetra_IntVector;
class Epetra_MultiVector;
class Epetra_CrsMatrix;
class Epetra_SerialDenseMatrix;

namespace Teuchos {
class ParameterList;
  }

namespace HYMLS {
namespace UnitTests {

class DisableOutput
  {
  Teuchos::RCP<std::ostream> no_output;

public:
  DisableOutput();
  ~DisableOutput();

  void EnableOutput();
  };

#define DISABLE_OUTPUT HYMLS::UnitTests::DisableOutput \
  Error_You_are_trying_to_disable_output_multiple_times_in_one_scope;
#define ENABLE_OUTPUT \
  Error_You_are_trying_to_disable_output_multiple_times_in_one_scope.EnableOutput();

//! create a Galeri random map with n global IDs and ndof consecutive
//! nodes always on the same partition.
Teuchos::RCP<Epetra_Map> create_random_map(const Epetra_Comm& comm, int n, int ndof);
Teuchos::RCP<Epetra_Map> create_random_map(const Epetra_Comm& comm, long long n, int ndof);

//! compute the inf-norm of the difference of two IntVectors
int NormInfAminusB(const Epetra_IntVector& A, const Epetra_IntVector& B);

//! compute the inf-norm of the difference of two MultiVectors
double NormInfAminusB(const Epetra_MultiVector& A, const Epetra_MultiVector& B);

//! compute the inf-norm of the difference of two SerialDenseMatrices
double NormInfAminusB(const Epetra_SerialDenseMatrix& A, const Epetra_SerialDenseMatrix& B);

Teuchos::RCP<Epetra_SerialDenseMatrix> RandomSerialDenseMatrix(int m, int n, const Epetra_Comm& comm);

Teuchos::RCP<Teuchos::ParameterList> CreateTestParameterList();

Teuchos::RCP<Epetra_CrsMatrix> CreateTestMatrix(Teuchos::RCP<Teuchos::ParameterList> &params, const Epetra_Comm& comm);

}} // namespaces HYMLS::UnitTests
#endif
