#ifndef HYMLS_DEFLATED_SOLVER_H
#define HYMLS_DEFLATED_SOLVER_H

#include "Teuchos_RCP.hpp"

#include "HYMLS_BaseSolver.hpp"

// forward declarations
class Epetra_MultiVector;
class Epetra_SerialDenseMatrix;
class Epetra_SerialDenseSolver;
class Epetra_Operator;
class Epetra_RowMatrix;

namespace Anasazi
  {
template <class ScalarType, class MV> struct Eigensolution;
  }

namespace Teuchos
  {
class ParameterList;
  }

namespace HYMLS {

/*! iterative solver class, basically
   an Epetra wrapper for Belos extended with
   some bordering and deflation functionality.
*/
class DeflatedSolver : virtual public BaseSolver
  {
public:
  //!
  //! Constructor
  //!
  //! arguments: matrix, preconditioner and belos params.
  //!
  DeflatedSolver(Teuchos::RCP<const Epetra_Operator> K,
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
  virtual int SetupDeflation();

  //! computes dominant eigenpairs of inv(P)
  Teuchos::RCP<Anasazi::Eigensolution<double, Epetra_MultiVector> > EigsPrec(int numEigs) const;

  //! use same preconditioner but operator (I-VV')A
  virtual int setProjectionVectors(Teuchos::RCP<const Epetra_MultiVector> V,
    Teuchos::RCP<const Epetra_MultiVector> W = Teuchos::null);

private:

  //! label
  std::string label_;

protected:
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

  //! deflation vectors
  Teuchos::RCP<Epetra_MultiVector> massDeflationVectors_;

  //! deflation vectors
  Teuchos::RCP<Epetra_MultiVector> ATV_;

  //! WA from the SingSys tex document
  Teuchos::RCP<Epetra_MultiVector> deflationRhs_;

  //! V'AV -V'AWA from the SingSys tex document
  Teuchos::RCP<Epetra_SerialDenseMatrix> deflationMatrix_;

  //! V'AV -V'AWA from the SingSys tex document
  Teuchos::RCP<Epetra_SerialDenseMatrix> deflationMatrixFactors_;

  //! solver for the above matrix
  Teuchos::RCP<Epetra_SerialDenseSolver> deflationMatrixSolver_;

  //! eigenpairs of the preconditioner
  Teuchos::RCP<Anasazi::Eigensolution<double, Epetra_MultiVector> > precEigs_;

  //! Projection vectors other than the deflation vectors
  Teuchos::RCP<const Epetra_MultiVector> deflationV_, deflationW_;

  //! Projection vectors other than the deflation vectors
  Teuchos::RCP<Epetra_MultiVector> AinvDeflationV_, AinvDeflationW_;

  };

}

#endif
