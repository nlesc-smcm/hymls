#ifndef HYMLS_BASE_SOLVER_H
#define HYMLS_BASE_SOLVER_H

#include "HYMLS_config.h"
#include "HYMLS_Macros.H"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Epetra_Operator.h"

#include "BelosLinearProblem.hpp"
#include "BelosSolverManager.hpp"
#include "BelosEpetraAdapter.hpp"
#include "BelosOutputManager.hpp"

#include "Epetra_RowMatrix.h"

#include "HYMLS_PLA.H"

// forward declarations
class Epetra_MultiVector;
class Epetra_Comm;
class Epetra_SerialDenseMatrix;
class Epetra_Map;
class Epetra_Import;


namespace HYMLS {

namespace Belos {
class EpetraOperator;
}
/*! iterative solver class, basically         
   an Epetra wrapper for Belos extended with  
   some bordering and deflation functionality.
*/
class BaseSolver : public Epetra_Operator,
                   public PLA
  {
  
public:

  typedef double ST;
  typedef Epetra_MultiVector MV;
  typedef Epetra_Operator OP;
  typedef ::Belos::EpetraPrecOp belosPrecType_;
  typedef ::Belos::LinearProblem<ST,MV,OP> belosProblemType_;
  typedef ::Belos::SolverManager<ST,MV,OP> belosSolverType_;


  //!                                                     
  //! Constructor                                         
  //!                                                     
  //! arguments: matrix, preconditioner and belos params. 
  //!                                                     
  BaseSolver(Teuchos::RCP<const Epetra_RowMatrix> K, 
    Teuchos::RCP<Epetra_Operator> P,
    Teuchos::RCP<Teuchos::ParameterList> params,
    int numRhs=1, bool validate=true);

  //! destructor
  virtual ~BaseSolver();

  //! set solver parameters (the list is the "HYMLS"->"Solver" sublist)
  virtual void setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params);

  //! set solver parameters (the list is the "HYMLS"->"Solver" sublist)
  //! The extra argument is so it can be used by the actual Solver class
  virtual void setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params,
    bool validateParameters);

  //! get a list of valid parameters for this object
  virtual Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;
  
  //! set matrix for solve
  virtual void SetMatrix(Teuchos::RCP<const Epetra_RowMatrix> A);

  //! set preconditioner for solve
  virtual void SetPrecond(Teuchos::RCP<Epetra_Operator> P);

  //! for eigenvalue computations - set mass matrix
  virtual void SetMassMatrix(Teuchos::RCP<const Epetra_RowMatrix> B);

  //! Applies the operator
  int Apply(const Epetra_MultiVector& X,
                           Epetra_MultiVector& Y) const 
    {
    return operator_->Apply(X,Y);
    }

  //! Applies the matrix
  int ApplyMatrix(const Epetra_MultiVector& X,
                           Epetra_MultiVector& Y) const 
    {
    return matrix_->Apply(X,Y);
    }

  //! Applies the preconditioner
  int ApplyPrec(const Epetra_MultiVector& X,
                           Epetra_MultiVector& Y) const 
    {
    return precond_->ApplyInverse(X,Y);
    }

  //! Applies the mass matrix
  int ApplyMass(const Epetra_MultiVector& X,
                           Epetra_MultiVector& Y) const 
    {
    return massMatrix_->Apply(X,Y);
    }

  //! Applies the preconditioner to vector X, returns the result in Y.
  virtual int ApplyInverse(const Epetra_MultiVector& X,
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

  //! setup the solver to solve (shiftA*A+shiftB*B)x=b
  virtual void setShift(double shiftA, double shiftB);

  //! set convergence tolerance for Krylov solver
  virtual void SetTolerance(double tol);

  //! get number of iterations performed in last ApplyInverse() call
  inline int getNumIter() const {return numIter_;}

  //! For singular problems with a known null space, add the null space
  //! as a border so that in fact the linear system
  //!
  //! |A   V||x |   |b|
  //! |W'  C||x0| = |0|
  //!
  //! is being solved with W=V and C=0. This means that the solution will
  //! be perpendicular to V. If the function is called repeatedly,
  //! the 'old' vectors are replaced.
  virtual int setBorder(Teuchos::RCP<const Epetra_MultiVector> const &V,
    Teuchos::RCP<const Epetra_MultiVector> const &W=Teuchos::null,
    Teuchos::RCP<const Epetra_SerialDenseMatrix> const &C=Teuchos::null)
    {
    return -99;
    }

  //! use same preconditioner but operator (I-VV')A
  virtual int setProjectionVectors(Teuchos::RCP<const Epetra_MultiVector> V,
    Teuchos::RCP<const Epetra_MultiVector> W = Teuchos::null);

  //! Method to setup the deflation in the deflated solvers
  virtual int SetupDeflation()
    {
    return -99;
    }

protected: 

  //! communicator
  Teuchos::RCP<const Epetra_Comm> comm_;

  //! matrix
  Teuchos::RCP<const Epetra_RowMatrix> matrix_;

 //! operator for which we solve OP*x=b, typically same as matrix_ or
 //! beta*A+alpha*B (if SetShift was called)
 Teuchos::RCP<const Epetra_Operator> operator_;

  //! preconditioner
  Teuchos::RCP<Epetra_Operator> precond_;

  //! We solve (beta*A+alpha*B)x=b
  double shiftA_, shiftB_;

  //! solver type and start vector type
  std::string solverType_, startVec_;
  
  //! rhs and sol vectors. I'm not sure if we need these vectors
  //! in the object or could just pass on the input of ApplyInverse,
  //! but since it is currently implemented like this we make them
  //! mutable so they can be re-allocated in ApplyInverse() if the 
  //! number of RHS changes
  mutable Teuchos::RCP<Epetra_MultiVector> belosRhs_,belosSol_;

  //! Belos preconditioner interface
  Teuchos::RCP<belosPrecType_> belosPrecPtr_;

  //! Belos linear problem interface
  Teuchos::RCP<belosProblemType_> belosProblemPtr_;

  //! Belos solver
  Teuchos::RCP<belosSolverType_> belosSolverPtr_;

//@}

//@{

//! mass matrix - can be set using SetmassMatrix(). If not set, the 
//! standard eigenproblem is solved.
Teuchos::RCP<const Epetra_RowMatrix> massMatrix_;

//! Borders of the matrix, which might be the eigenspace
  Teuchos::RCP<const Epetra_MultiVector> V_, W_;

//! projected operator (V_orth' A V_orth)
Teuchos::RCP<Epetra_Operator> Aorth_;

//! a belos solver for Aorth_, disguised as an Epetra_Operator
Teuchos::RCP<Belos::EpetraOperator> AorthSolver_;

//@}

  //! use transposed operator?
  bool useTranspose_;
  
  //! infinity norm
  double normInf_;
  
  //! number of iterations performed in last ApplyInverse() call
  mutable int numIter_;
  
  //! label
  std::string label_;

  //! default value for the "Left or Right Preconditioning" parameter
  const std::string lor_default_;
  
  };


}

#endif
