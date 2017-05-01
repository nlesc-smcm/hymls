#ifndef HYMLS_SOLVER_H
#define HYMLS_SOLVER_H

#include "HYMLS_config.h"

#include "Teuchos_RCP.hpp"
#include "Epetra_Operator.h"
#include "Epetra_RowMatrix.h"

#include "HYMLS_PLA.H"
#include "HYMLS_BaseSolver.H"

  // forward declarations
class Epetra_MultiVector;
class Epetra_Comm;
class Epetra_Map;
class Epetra_SerialDenseMatrix;

namespace Teuchos
  {
  class ParameterList;
  }

namespace HYMLS {

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
  Solver(Teuchos::RCP<const Epetra_RowMatrix> K,
    Teuchos::RCP<Epetra_Operator> P,
    Teuchos::RCP<Teuchos::ParameterList> params,
    int numRhs = 1);

  //! destructor
  ~Solver();

  //! set solver parameters (the list is the "HYMLS"->"Solver" sublist)
  void setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params);

  //! get a list of valid parameters for this object
  Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;

  //! set matrix for solve
  void SetMatrix(Teuchos::RCP<const Epetra_RowMatrix> A)
    {
    solver_->SetMatrix(A);
    }

  //! set preconditioner for solve
  void SetPrecond(Teuchos::RCP<Epetra_Operator> P)
    {
    solver_->SetPrecond(P);
    }

  //! for eigenvalue computations - set mass matrix
  void SetMassMatrix(Teuchos::RCP<const Epetra_RowMatrix> B)
    {
    solver_->SetMassMatrix(B);
    }

  //! Applies the operator
  int Apply(const Epetra_MultiVector& X,
    Epetra_MultiVector& Y) const
    {
    return solver_->Apply(X, Y);
    }

  //! Applies the matrix
  int ApplyMatrix(const Epetra_MultiVector& X,
    Epetra_MultiVector& Y) const
    {
    return solver_->ApplyMatrix(X, Y);
    }

  //! Applies the preconditioner
  int ApplyPrec(const Epetra_MultiVector& X,
    Epetra_MultiVector& Y) const
    {
    return solver_->ApplyPrec(X, Y);
    }

  //! Applies the mass matrix
  int ApplyMass(const Epetra_MultiVector& X,
    Epetra_MultiVector& Y) const
    {
    return solver_->ApplyMass(X, Y);
    }

  //! Applies the preconditioner to vector X, returns the result in Y.
  int ApplyInverse(const Epetra_MultiVector& X,
    Epetra_MultiVector& Y) const
    {
    return solver_->ApplyInverse(X, Y);
    }

  int SetUseTranspose(bool UseTranspose)
    {
    return solver_->SetUseTranspose(UseTranspose);
    }

  //! not implemented.
  bool HasNormInf() const
    {
    return solver_->HasNormInf();
    }

  //! infinity norm
  double NormInf() const
    {
    return solver_->NormInf();
    }

  //! label
  const char* Label() const
    {
    return solver_->Label();
    }

  //! use transpose?
  bool UseTranspose() const
    {
    return solver_->UseTranspose();
    }

  //! communicator
  const Epetra_Comm & Comm() const
    {
    return solver_->Comm();
    }

  //! Returns the Epetra_Map object associated with the domain of this operator.
  const Epetra_Map & OperatorDomainMap() const
    {
    return solver_->OperatorDomainMap();
    }

  //! Returns the Epetra_Map object associated with the range of this operator.
  const Epetra_Map & OperatorRangeMap() const
    {
    return solver_->OperatorRangeMap();
    }
  //@}

  //! setup the solver to solve (shiftA*A+shiftB*B)x=b
  void setShift(double shiftA, double shiftB)
    {
    solver_->setShift(shiftA, shiftB);
    }

  void SetTolerance(double tol)
    {
    solver_->SetTolerance(tol);
    }

  //! get number of iterations performed in last ApplyInverse() call
  int getNumIter() const
    {
    return solver_->getNumIter();
    }

  //! For singular problems with a known null space, add the null space
  //! as a border so that in fact the linear system
  //!
  //! |A   V||x |   |b|
  //! |W'  C||x0| = |0|
  //!
  //! is being solved with W=V and C=0. This means that the solution will
  //! be perpendicular to V. If the function is called repeatedly,
  //! the 'old' vectors are replaced.
  int setBorder(Teuchos::RCP<const Epetra_MultiVector> const &V,
    Teuchos::RCP<const Epetra_MultiVector> const &W=Teuchos::null,
    Teuchos::RCP<const Epetra_SerialDenseMatrix> const &C=Teuchos::null)
    {
    return solver_->setBorder(V, W, C);
    }

  //! use same preconditioner but operator (I-VV')A
  int setProjectionVectors(Teuchos::RCP<const Epetra_MultiVector> V,
    Teuchos::RCP<const Epetra_MultiVector> W = Teuchos::null)
    {
    return solver_->setProjectionVectors(V, W);
    }

  //! computes the modes closest to 0 and sets up the solver's deflation
  //! capabilities. If maxEigs==-2, the value "Deflated Subspace Dimension"
  //! from the "Solver" sublist is used.
  //!
  //! If maxEigs==0, this function adds the null space as a border of the
  //! preconditioner to make it non-singular.
  //!
  //! \todo see todo in addBorder
  int SetupDeflation()
    {
    return solver_->SetupDeflation();
    }

protected:
  //! Actual HYMLS solver without extra functionality
  Teuchos::RCP<BaseSolver> solver_;

  //! label
  std::string label_;

  //! Used to determine if we want to use the deflated HYMLS solver class
  bool useDeflation_;

  //! Used to determine if we want to use the bordered HYMLS solver class
  bool useBordering_;

  };

}

#endif
