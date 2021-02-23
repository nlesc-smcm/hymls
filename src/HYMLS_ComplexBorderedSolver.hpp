#ifndef HYMLS_COMPLEX_BORDERED_SOLVER_H
#define HYMLS_COMPLEX_BORDERED_SOLVER_H

#include "Teuchos_RCP.hpp"

#include "HYMLS_BorderedSolver.hpp"

// forward declarations
class Epetra_MultiVector;

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

class BorderedVector;
class BorderedOperator;

template<typename>
class ComplexVector;
template<typename, typename>
class ComplexOperator;

/*! iterative solver class, basically
   an Epetra wrapper for Belos extended with
   some bordering and deflation functionality.
*/
class ComplexBorderedSolver : public virtual BorderedSolver
  {

  using BelosMultiVectorType = ComplexVector<BorderedVector>;
  using BelosOperatorType = ComplexOperator<BorderedOperator, BorderedVector>;
  using BelosProblemType = Belos::LinearProblem<
    std::complex<double>, BelosMultiVectorType, BelosOperatorType>;
  using BelosSolverType = Belos::SolverManager<
    std::complex<double>, BelosMultiVectorType, BelosOperatorType>;

public:

  //!
  //! Constructor
  //!
  //! arguments: matrix, preconditioner and belos params.
  //!
  ComplexBorderedSolver(Teuchos::RCP<const Epetra_Operator> K,
    Teuchos::RCP<Epetra_Operator> P,
    Teuchos::RCP<Teuchos::ParameterList> params,
    bool validate=true);

  //! destructor
  virtual ~ComplexBorderedSolver();

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

protected:

//@}

  //! label
  std::string label_;

  //! Belos preconditioner interface
  Teuchos::RCP<BelosOperatorType> belosPrecPtr_;

  //! Belos linear problem interface
  Teuchos::RCP<BelosProblemType> belosProblemPtr_;

  //! Belos solver
  Teuchos::RCP<BelosSolverType> belosSolverPtr_;

  };


}

#endif
