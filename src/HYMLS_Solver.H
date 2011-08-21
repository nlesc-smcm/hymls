#ifndef HYMLS_SOLVER_H
#define HYMLS_SOLVER_H

#include "Teuchos_RCP.hpp"
#include "Ifpack_Preconditioner.h"

#include "BelosLinearProblem.hpp"
#include "BelosSolverManager.hpp"
#include "BelosEpetraAdapter.hpp"
#include "BelosOutputManager.hpp"

#include "Epetra_RowMatrix.h"

#include "HYMLS_MatrixUtils.H"

// forward declarations
class Epetra_MultiVector;
class Epetra_Comm;
class Epetra_Map;
class Epetra_Import;


namespace Teuchos 
  {
  class ParameterList;
  }

namespace HYMLS {

/*! iterative solver class, currently a very simple 
   Epetra wrapper for Belos.
*/
class Solver : public Epetra_Operator
  {
  
public:

  typedef double ST;
  typedef Epetra_MultiVector MV;
  typedef Epetra_Operator OP;
  typedef Belos::EpetraPrecOp belosPrecType_;
  typedef Belos::LinearProblem<ST,MV,OP> belosProblemType_;
  typedef Belos::SolverManager<ST,MV,OP> belosSolverType_;


  //!                                                     
  //! Constructor                                         
  //!                                                     
  //! arguments: matrix, preconditioner and belos params. 
  //!                                                     
  Solver(Teuchos::RCP<const Epetra_RowMatrix> K, 
      Teuchos::RCP<Epetra_Operator> P,
      Teuchos::RCP<Teuchos::ParameterList> params,
      int numRhs=1);

  //! destructor
  ~Solver();

  //! set solver parameters (the list is the "HYMLS" sublist,
  //! only the "Solver" sublist is accessed by this class).
  int SetParameters(Teuchos::ParameterList& params);

  //! set matrix for solve
  void SetMatrix(Teuchos::RCP<const Epetra_RowMatrix> A);

  //! set preconditioner for solve
  void SetPrecond(Teuchos::RCP<Epetra_Operator> P);

  //! for eigenvalue computations - set mass matrix
  void SetMassMatrix(Teuchos::RCP<const Epetra_RowMatrix> B);
  
  //! Applies the operator (not implemented)
  int Apply(const Epetra_MultiVector& X,
                           Epetra_MultiVector& Y) const 
    {
    return matrix_->Apply(X,Y);
    }

  //! Applies the preconditioner to vector X, returns the result in Y.
  int ApplyInverse(const Epetra_MultiVector& X,
                           Epetra_MultiVector& Y) const;

  int SetUseTranspose(bool UseTranspose)
    {
    useTranspose_=false; // not implemented.
    return -1;
    }
  //! not implemented.
  bool HasNormInf() const {return false;}

  //! infinity norm
  double NormInf() const {return normInf_;}

  //! label
  const char* Label() const {return label_.c_str();}
  
  //! use transpose?
  bool UseTranspose() const {return useTranspose_;}
  
  //! communicator
  const Epetra_Comm & Comm() const {return *comm_;}
  
    //! Returns the Epetra_Map object associated with the domain of this operator.
    const Epetra_Map & OperatorDomainMap() const {return matrix_->OperatorDomainMap();}

    //! Returns the Epetra_Map object associated with the range of this operator.
    const Epetra_Map & OperatorRangeMap() const {return matrix_->OperatorRangeMap();}
  

  

  //@}

  //! computes the modes closest to 0 and sets up the solver's deflation
  //! capabilities. If maxEigs==-2, the value "Deflated Subspace Dimension"
  //! from the "Solver" sublist is used.
  int SetupDeflation(int maxEigs=-2);

protected: 

  //! communicator
  Teuchos::RCP<const Epetra_Comm> comm_;

  //! matrix
  Teuchos::RCP<const Epetra_RowMatrix> matrix_;

  //! preconditioner
  Teuchos::RCP<Epetra_Operator> precond_;
  
  //! Parameter List
  Teuchos::RCP<Teuchos::ParameterList> params_;
  
  //! rhs and sol vectors
  Teuchos::RCP<Epetra_MultiVector> belosRhs_,belosSol_;

  //! Belos preconditioner interface
  Teuchos::RCP<belosPrecType_> belosPrecPtr_;

  //! Belos linear problem interface
  Teuchos::RCP<belosProblemType_> belosProblemPtr_;

  //! Belos solver
  Teuchos::RCP<belosSolverType_> belosSolverPtr_;

//@}

//!\name data structures to monitor and deflate unstable modes
//@{

//! mass matrix - can be set using SetmassMatrix(). If not set, the 
//! standard eigenproblem is solved.
Teuchos::RCP<const Epetra_RowMatrix> massMatrix_;

//! number of eigenvalues computed initially
int numEigs_;

//! number of deflated eigenmodes
int numDeflated_;

//! tolerance to remove a mode. An eigenmode            
//! is deflated if |lambda(P^{-1}A)-1|>tol (set by      
//! "Deflation Threshold" in the "Solver" sublist).     
double deflThres_;

//! eigen-pairs of the preconditioner
Teuchos::RCP<MatrixUtils::Eigensolution> precEigs_;

//! basis for the eigenspace to be deflated
Teuchos::RCP<Epetra_MultiVector> V_;

//! W1rhs = V_orth'AV
Teuchos::RCP<Epetra_MultiVector> W1rhs_;

//! W1 = (V_orth'AV_orth)\W1rhs (W_A)
Teuchos::RCP<Epetra_MultiVector> W1_;

//! projected operator (V_orth' A V_orth)
Teuchos::RCP<Epetra_Operator> Aorth_;

//@}

  //! use transposed operator?
  bool useTranspose_;
  
  //! infinity norm
  double normInf_;
  
  //! label
  std::string label_;

private:
     
  };


}

#endif
