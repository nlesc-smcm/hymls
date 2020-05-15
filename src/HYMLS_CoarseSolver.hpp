#ifndef HYMLS_COARSE_SOLVER_H
#define HYMLS_COARSE_SOLVER_H

#include "HYMLS_config.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "Ifpack_CondestType.h"
#include "Ifpack_Preconditioner.h"

#include "HYMLS_BorderedOperator.hpp"
#include "HYMLS_PLA.hpp"

#include <string>

// forward declarations
class Epetra_Comm;
class Epetra_Map;
class Epetra_RowMatrix;
class Epetra_CrsMatrix;
class Epetra_SerialDensematrix;
class Epetra_MultiVector;
class Epetra_Vector;

namespace EpetraExt
  {
class MultiVector_Reindex;
class CrsMatrix_Reindex;
  }

namespace Teuchos
  {
class ParameterList;
  }

namespace HYMLS {

namespace EpetraExt
  {
class RestrictedCrsMatrixWrapper;
class RestrictedMultiVectorWrapper;
  }

class CoarseSolver: public Ifpack_Preconditioner,
                    public BorderedOperator,
                    public PLA
  {
public:
  CoarseSolver() = delete;

  CoarseSolver(
    Teuchos::RCP<const Epetra_CrsMatrix> matrix,
    int level);

  virtual ~CoarseSolver() {}

  //! \name ParameterListAcceptor interface
  //@{

  //! Set the ParameterList RCP directly
  void setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& list);

  //@}

  //! \name Ifpack_Preconditioner interface
  //@{

  //! Sets all parameters for the preconditioner.
  int SetParameters(Teuchos::ParameterList& List);

  //! Computes all it is necessary to initialize the preconditioner.
  int Initialize();

  //! Returns true if the  preconditioner has been successfully initialized, false otherwise.
  bool IsInitialized() const;

  //! Computes all it is necessary to apply the preconditioner.
  int Compute();

  //! Returns true if the  preconditioner has been successfully computed, false otherwise.
  bool IsComputed() const;

  //! Computes the condition number estimate, returns its value.
  double Condest(const Ifpack_CondestType CT = Ifpack_Cheap,
    const int MaxIters = 1550,
    const double Tol = 1e-9,
    Epetra_RowMatrix* Matrix = 0);

  //! Returns the computed condition number estimate, or -1.0 if not computed.
  double Condest() const;

  //! Applies the preconditioner to vector X, returns the result in Y.
  int ApplyInverse(const Epetra_MultiVector& X,
    Epetra_MultiVector& Y) const;

  //! Returns a pointer to the matrix to be preconditioned.
  const Epetra_RowMatrix& Matrix() const;

  //! Returns the number of calls to Initialize().
  int NumInitialize() const;

  //! Returns the number of calls to Compute().
  int NumCompute() const;

  //! Returns the number of calls to ApplyInverse().
  int NumApplyInverse() const;

  //! Returns the time spent in Initialize().
  double InitializeTime() const;

  //! Returns the time spent in Compute().
  double ComputeTime() const;

  //! Returns the time spent in ApplyInverse().
  double ApplyInverseTime() const;

  //! Returns the number of flops in the initialization phase.
  double InitializeFlops() const;

  //! Returns the number of flops in the computation phase.
  double ComputeFlops() const;

  //! Returns the number of flops in the application of the preconditioner.
  double ApplyInverseFlops() const;

  //! Prints basic information on iostream. This function is used by operator<<.
  std::ostream& Print(std::ostream& os) const;

  //@}

  //! \name Epetra_Operator interface
  //@{

  //! If set true, transpose of this operator will be applied.
  int SetUseTranspose(bool UseTranspose);

  //! Returns the result of a Epetra_Operator applied to a Epetra_MultiVector X in Y.
  int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

  //! Returns the infinity norm of the global matrix.
  double NormInf() const;

  //! Returns a character string describing the operator
  const char * Label() const;

  //! Returns the current UseTranspose setting.
  bool UseTranspose() const;

  //! Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
  bool HasNormInf() const;

  //! Returns a pointer to the Epetra_Comm communicator associated with this operator.
  const Epetra_Comm & Comm() const;

  //! Returns the Epetra_Map object associated with the domain of this operator.
  const Epetra_Map & OperatorDomainMap() const;

  //! Returns the Epetra_Map object associated with the range of this operator.
  const Epetra_Map & OperatorRangeMap() const;

  //@}

  //! \name HYMLS BorderedOperator interface
  //@{

  //!
  int setBorder(Teuchos::RCP<const Epetra_MultiVector> V,
    Teuchos::RCP<const Epetra_MultiVector> W,
    Teuchos::RCP<const Epetra_SerialDenseMatrix> C);

  //!
  bool HaveBorder() const {return haveBorder_;}

  //!
  int Apply(const Epetra_MultiVector & B, const Epetra_SerialDenseMatrix & C,
    Epetra_MultiVector& X, Epetra_SerialDenseMatrix & Y) const;

  //! Compute [X S]' = [K V;W' C] \ [Y T]'
  int ApplyInverse(const Epetra_MultiVector& X,
    const Epetra_SerialDenseMatrix& T,
    Epetra_MultiVector& Y,
    Epetra_SerialDenseMatrix& S) const;

  //@}

protected:

  //! communicator
  Teuchos::RCP<const Epetra_Comm> comm_;

  //! my level ID
  int myLevel_;

  //! if the processor has no rows in the present SC, this is false.
  bool amActive_;

  //! input matrix
  Teuchos::RCP<const Epetra_CrsMatrix> matrix_;

  //! linear map for the reduced SC
  Teuchos::RCP<Epetra_Map> linearMap_;

  //! this is to reindex the reduced SC, which is
  //! important when using a direct solver (I think)
  Teuchos::RCP< ::EpetraExt::CrsMatrix_Reindex> reindexA_;

  //! reindex corresponding vectors
  Teuchos::RCP< ::EpetraExt::MultiVector_Reindex> reindexX_, reindexB_;

  //! this is to restrict the reduced Schur problem on the
  //! coarsest level to only the active procs so that
  //! independently of the Amesos solver, the number of
  //! procs participating in the factorization is determi-
  //! ned by our own algorithm, e.g. max(np, nsd).
  Teuchos::RCP< ::HYMLS::EpetraExt::RestrictedCrsMatrixWrapper> restrictA_;

  //! restrict corresponding vectors
  Teuchos::RCP< ::HYMLS::EpetraExt::RestrictedMultiVectorWrapper> restrictX_, restrictB_;

  //! sparse matrix representation of the reduced Schur-complement
  //! (associated with Vsum nodes)
  Teuchos::RCP<Epetra_CrsMatrix> reducedSchur_;

  // View of SC2 with linear map
  Teuchos::RCP<Epetra_CrsMatrix> linearMatrix_;

  // View of SC2 with linear map and no empty partitions (restricted Comm)
  Teuchos::RCP<Epetra_CrsMatrix> restrictedMatrix_;

  //! Views and copies of vectors used in ApplyInverse(), mutable temporary data
  mutable Teuchos::RCP<Epetra_MultiVector> linearRhs_, linearSol_, restrictedRhs_, restrictedSol_;

  //! solver for the reduced Schur complement. Note that Ifpack_Preconditioner
  //! is implemented by both Amesos (direct solver) and our HYMLS::Solver,
  //! so we don't have to make a choice at this point.
  Teuchos::RCP<Ifpack_Preconditioner> reducedSchurSolver_;

  //! true if addBorder() has been called with non-null args
  bool haveBorder_;

  //! label
  std::string label_;

  //! true if the Schur complement has 0 global rows
  bool isEmpty_;

  //! has Initialize() been called?
  bool initialized_;

  //! has Compute() been called?
  bool computed_;

  //! we can replace a number of rows and cols of the reduced SC
  //! by Dirichlet conditions. This is used to fix the pressure
  //! level
  Teuchos::Array<hymls_gidx> fix_gid_;

  //! \name data structures for bordering

  //! augmented matrix for V-sums, [M22 V2; W2 C]
  Teuchos::RCP<Epetra_RowMatrix> augmentedMatrix_;

  };

  }

#endif
