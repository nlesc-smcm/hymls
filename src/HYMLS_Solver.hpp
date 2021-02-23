#ifndef HYMLS_SOLVER_H
#define HYMLS_SOLVER_H

#include "Teuchos_RCP.hpp"
#include "Epetra_Operator.h"

#include "HYMLS_PLA.hpp"

#include <string>

// forward declarations
class Epetra_MultiVector;
class Epetra_Comm;
class Epetra_Map;
class Epetra_SerialDenseMatrix;
class Epetra_RowMatrix;

namespace Teuchos
  {
class ParameterList;
  }

namespace HYMLS {

class BaseSolver;

/*! iterative solver class, basically
  an Epetra wrapper for Belos extended with
  some bordering and deflation functionality.
*/
class Solver : public Epetra_Operator,
               public PLA
  {
public:
  //!
  //! Constructor
  //!
  //! arguments: matrix, preconditioner and belos params.
  //!
  Solver(Teuchos::RCP<const Epetra_Operator> K,
    Teuchos::RCP<Epetra_Operator> P,
    Teuchos::RCP<Teuchos::ParameterList> params);

  //! destructor
  virtual ~Solver();

  //! set solver parameters (the list is the "HYMLS"->"Solver" sublist)
  void setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params);

  //! get a list of valid parameters for this object
  Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;

  //! set matrix for solve
  void SetOperator(Teuchos::RCP<const Epetra_Operator> A);

  //! set preconditioner for solve
  void SetPrecond(Teuchos::RCP<Epetra_Operator> P);

  //! for eigenvalue computations - set mass matrix
  void SetMassMatrix(Teuchos::RCP<const Epetra_RowMatrix> B);

  //! Applies the operator
  int Apply(const Epetra_MultiVector& X,
    Epetra_MultiVector& Y) const;

  //! Applies the matrix
  int ApplyMatrix(const Epetra_MultiVector& X,
    Epetra_MultiVector& Y) const;

  //! Applies the preconditioner
  int ApplyPrec(const Epetra_MultiVector& X,
    Epetra_MultiVector& Y) const;

  //! Applies the mass matrix
  int ApplyMass(const Epetra_MultiVector& X,
    Epetra_MultiVector& Y) const;

  //! Applies the preconditioner to vector X, returns the result in Y.
  int ApplyInverse(const Epetra_MultiVector& X,
    Epetra_MultiVector& Y) const;

  int SetUseTranspose(bool UseTranspose);

  //! not implemented.
  bool HasNormInf() const;

  //! infinity norm
  double NormInf() const;

  //! label
  const char* Label() const;

  //! use transpose?
  bool UseTranspose() const;

  //! communicator
  const Epetra_Comm & Comm() const;

  //! Returns the Epetra_Map object associated with the domain of this operator.
  const Epetra_Map & OperatorDomainMap() const;

  //! Returns the Epetra_Map object associated with the range of this operator.
  const Epetra_Map & OperatorRangeMap() const;
  //@}

  void SetTolerance(double tol);

  //! get number of iterations performed in last ApplyInverse() call
  int getNumIter() const;

  //! For singular problems with a known null space, add the null space
  //! as a border so that in fact the linear system
  //!
  //! |A   V||x |   |b|
  //! |W'  C||x0| = |0|
  //!
  //! is being solved with W=V and C=0. This means that the solution will
  //! be perpendicular to V. If the function is called repeatedly,
  //! the 'old' vectors are replaced.
  int SetBorder(Teuchos::RCP<const Epetra_MultiVector> const &V,
    Teuchos::RCP<const Epetra_MultiVector> const &W = Teuchos::null,
    Teuchos::RCP<const Epetra_SerialDenseMatrix> const &C = Teuchos::null);

  //! use same preconditioner but operator (I-VV')A
  int setProjectionVectors(Teuchos::RCP<const Epetra_MultiVector> V,
    Teuchos::RCP<const Epetra_MultiVector> W = Teuchos::null);

  //! computes the modes closest to 0 and sets up the solver's deflation
  //! capabilities. If maxEigs==-2, the value "Deflated Subspace Dimension"
  //! from the "Solver" sublist is used.
  //!
  //! If maxEigs==0, this function adds the null space as a border of the
  //! preconditioner to make it non-singular.
  //!
  //! \todo see todo in addBorder
  int SetupDeflation();

protected:
  //! Actual HYMLS solver without extra functionality
  Teuchos::RCP<BaseSolver> solver_;

  //! label
  std::string label_;

  //! Used to determine if we want to use the complex HYMLS solver class
  bool isComplex_;

  //! Used to determine if we want to use the deflated HYMLS solver class
  bool useDeflation_;

  //! Used to determine if we want to use the bordered HYMLS solver class
  bool useBordering_;

  };

  }

#endif
