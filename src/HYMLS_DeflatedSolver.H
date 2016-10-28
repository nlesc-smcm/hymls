#ifndef HYMLS_DEFLATED_SOLVER_H
#define HYMLS_DEFLATED_SOLVER_H

#include "HYMLS_config.h"

#include "Teuchos_RCP.hpp"
#include "Epetra_Operator.h"
#include "Epetra_RowMatrix.h"
#include "AnasaziTypes.hpp"

#include "HYMLS_PLA.H"
#include "HYMLS_BaseSolver.H"

  // forward declarations
class Epetra_MultiVector;
class Epetra_SerialDenseMatrix;
class Epetra_SerialDenseSolver;
class Epetra_Comm;
class Epetra_Map;

namespace Teuchos
  {
class ParameterList;
  }

namespace HYMLS {

/*! iterative solver class, basically
   an Epetra wrapper for Belos extended with
   some bordering and deflation functionality.
*/
class DeflatedSolver : public BaseSolver
  {
public:
  //!
  //! Constructor
  //!
  //! arguments: matrix, preconditioner and belos params.
  //!
  DeflatedSolver(Teuchos::RCP<const Epetra_RowMatrix> K,
    Teuchos::RCP<Epetra_Operator> P,
    Teuchos::RCP<Teuchos::ParameterList> params,
    int numRhs = 1, bool validate = true);

  //! destructor
  virtual ~DeflatedSolver();

  //! set solver parameters (the list is the "HYMLS"->"Solver" sublist)
  virtual void setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params);

  //! set solver parameters (the list is the "HYMLS"->"Solver" sublist)
  //! The extra argument is so it can be used by the actual Solver class
  virtual void setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params,
    bool validateParameters);

  //! get a list of valid parameters for this object
  virtual Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;

  //! Applies the preconditioner to vector X, returns the result in Y.
  int ApplyInverse(const Epetra_MultiVector& X,
    Epetra_MultiVector& Y) const;

  //! See SetupDeflation in the base solver. Maybe this one works
  virtual int SetupDeflation(int maxEigs=-2);

  //! computes dominant eigenpairs of inv(P)
  Teuchos::RCP<Anasazi::Eigensolution<double, Epetra_MultiVector> > EigsPrec(int numEigs) const;

private:

  //! label
  std::string label_;

  // TODO: Make all this stuff non-mutable
  //! number of eigenvalues computed initially
  int numEigs_;

  //! number of deflated eigenmodes
  int numDeflated_;

  //! tolerance to remove a mode. An eigenmode            
  //! is deflated if |lambda(P^{-1}A)-1|>tol (set by      
  //! "Deflation Threshold" in the "Solver" sublist).     
  double deflThres_;

  //! tells if deflation vectors were already computed
  bool deflationComputed_;

  //! tells if deflation vectors were computed after the mass matrix was set
  mutable bool deflationWithMassMatrix_;

  //! deflation vectors
  Teuchos::RCP<Epetra_MultiVector> deflationVectors_;

  //! WA from the SingSys tex document
  Teuchos::RCP<Epetra_MultiVector> deflationRhs_;

  //! WA from the SingSys tex document
  Teuchos::RCP<Epetra_SerialDenseMatrix> deflationMatrix_;

  //! WA from the SingSys tex document
  Teuchos::RCP<Epetra_SerialDenseSolver> deflationMatrixSolver_;

  //! eigen-pairs of the preconditioner
  Teuchos::RCP<Anasazi::Eigensolution<double, Epetra_MultiVector> > precEigs_;

  };

}

#endif
