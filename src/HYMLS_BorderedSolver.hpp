#ifndef HYMLS_BORDERED_SOLVER_H
#define HYMLS_BORDERED_SOLVER_H

#include "Teuchos_RCP.hpp"

#include "HYMLS_BaseSolver.hpp"

// forward declarations
class Epetra_MultiVector;
class Epetra_SerialDenseMatrix;
class Epetra_Operator;

namespace Belos
  {
template<typename, typename, typename>
class LinearProblem;
template<typename, typename, typename>
class SolverManager;
  }

namespace Teuchos
  {
class ParameterList;
  }

namespace HYMLS {

class BorderedOperator;
class BorderedVector;

/*! iterative solver class, basically
   an Epetra wrapper for Belos extended with
   some bordering and deflation functionality.
*/
class BorderedSolver : public virtual BaseSolver
  {

  using BelosProblemType = Belos::LinearProblem<
    double, BorderedVector, BorderedOperator>;
  using BelosSolverType = Belos::SolverManager<
    double, BorderedVector, BorderedOperator>;

public:

  //!
  //! Constructor
  //!
  //! arguments: matrix, preconditioner and belos params.
  //!
  BorderedSolver(Teuchos::RCP<const Epetra_Operator> K,
    Teuchos::RCP<Epetra_Operator> P,
    Teuchos::RCP<Teuchos::ParameterList> params,
    bool validate=true);

  //! destructor
  virtual ~BorderedSolver();

  //! set solver parameters (the list is the "HYMLS"->"Solver" sublist)
  virtual void setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params);

  //! set solver parameters (the list is the "HYMLS"->"Solver" sublist)
  //! The extra argument is so it can be used by the actual Solver class
  virtual void setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params,
    bool validateParameters);

  //! get a list of valid parameters for this object
  virtual Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;

  //! set preconditioner for solve
  virtual void SetPrecond(Teuchos::RCP<Epetra_Operator> P);

  virtual void SetTolerance(double tol);

  //! Applies the solver to vector X, returns the result in Y.
  virtual int ApplyInverse(const Epetra_MultiVector& X,
    Epetra_MultiVector& Y) const;

  //! Applies the solver to [Y T]' = [K V;W' C]\[X S]'
  virtual int ApplyInverse(const Epetra_MultiVector& X, const Epetra_SerialDenseMatrix& S,
    Epetra_MultiVector& Y, Epetra_SerialDenseMatrix& T) const;

  //! For singular problems with a known null space, add the null space
  //! as a border so that in fact the linear system
  //!
  //! |A   V0||x |   |b|
  //! |V0'  0||x0| = |0|
  //!
  //! is being solved. This means that the solution will be perpendicular to V0
  //! If the function is called repeatedly, the 'old' vectors are replaced.
  virtual int SetBorder(Teuchos::RCP<const Epetra_MultiVector> const &V,
    Teuchos::RCP<const Epetra_MultiVector> const &W=Teuchos::null,
    Teuchos::RCP<const Epetra_SerialDenseMatrix> const &C=Teuchos::null);

protected:

//@}

//!\name data structures to monitor and deflate unstable modes

//@{

//! Borders of the matrix, which might be the eigenspace
  Teuchos::RCP<const Epetra_MultiVector> V_, W_;

  Teuchos::RCP<const Epetra_SerialDenseMatrix> C_;

  //! block A \ V
  Teuchos::RCP<Epetra_MultiVector> Q_;

  //! Schur-complement (LU-factored in place)
  Teuchos::RCP<Epetra_SerialDenseMatrix> S_;

//@}

  //! label
  std::string label_;

  //! Belos preconditioner interface
  Teuchos::RCP<BorderedOperator> belosPrecPtr_;

  //! Belos linear problem interface
  Teuchos::RCP<BelosProblemType> belosProblemPtr_;

  //! Belos solver
  Teuchos::RCP<BelosSolverType> belosSolverPtr_;

  };


}

#endif
