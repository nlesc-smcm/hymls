#ifndef HYMLS_BORDERED_DEFLATED_SOLVER_H
#define HYMLS_BORDERED_DEFLATED_SOLVER_H

#include "Teuchos_RCP.hpp"

#include "HYMLS_BorderedSolver.H"
#include "HYMLS_DeflatedSolver.H"

class Epetra_MultiVector;
class Epetra_Operator;
class Epetra_RowMatrix;

namespace Teuchos { class ParameterList; }

namespace HYMLS {

/*! iterative solver class, basically
   an Epetra wrapper for Belos extended with
   some bordering and deflation functionality.
*/
class BorderedDeflatedSolver : public DeflatedSolver, public BorderedSolver
  {

public:

  //!
  //! Constructor
  //!
  //! arguments: matrix, preconditioner and belos params.
  //!
  BorderedDeflatedSolver(Teuchos::RCP<const Epetra_RowMatrix> K,
    Teuchos::RCP<Epetra_Operator> P,
    Teuchos::RCP<Teuchos::ParameterList> params,
    int numRhs=1, bool validate=true);

  //! destructor
  virtual ~BorderedDeflatedSolver();

  //! set solver parameters (the list is the "HYMLS"->"Solver" sublist)
  virtual void setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params);

  //! set solver parameters (the list is the "HYMLS"->"Solver" sublist)
  //! The extra argument is so it can be used by the actual Solver class
  virtual void setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params,
    bool validateParameters);

  //! get a list of valid parameters for this object
  virtual Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;

  //! Applies the preconditioner to vector X, returns the result in Y.
  virtual int ApplyInverse(const Epetra_MultiVector& X,
    Epetra_MultiVector& Y) const;

  virtual int SetupDeflation();

protected:

//@}

  //! label
  std::string label_;
  };


}

#endif
