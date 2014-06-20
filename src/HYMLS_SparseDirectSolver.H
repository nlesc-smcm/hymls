#ifndef HYMLS_SPARSE_DIRECT_SOLVER_H
#define HYMLS_SPARSE_DIRECT_SOLVER_H

#include "Amesos_config.h"

// use Amesos interfaces - no custom ordering or scaling,
// just Amesos options are accepted (in fact this whole  
// class turns into a 'typedef')
//#define USE_AMESOS 1

#ifdef USE_AMESOS
#include "Ifpack_Amesos.h"
namespace HYMLS {
typedef Ifpack_Amesos SparseDirectSolver;
}
#else

#include "Ifpack_Preconditioner.h"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

class Epetra_Map;
class Epetra_Time;
class Epetra_Comm;
class Epetra_MultiVector;
class Epetra_Vector;
class Epetra_RowMatrix;
class Epetra_CrsMatrix;
class Epetra_Import;

class KluWrapper;

namespace HYMLS {

//! this is our own interface to some serial sparse direct
//! solvers. Using our own interface gives us access to more
//! settings of the methods and allows us to use own ordering
//! and scaling more easily and consistently.
//!
//! This class accepts the following parameters:
//! "amesos: solver type" can be "KLU" (default) or "UMFPACK"
//!             (if HAVE_SUITESPARSE is defined). "Cholmod" is
//!             intended to be added in the future. For consistency
//!             with Amesos, you can also set Amesos_Klu etc, and  
//!             the option is case insensitive.
//! "Custom Ordering" (bool) if false, we leave it to the method to
//!             decide. Otherwise we use the ordering function from
//!             MatrixUtils and set the pivot tol to something tiny
//! "Custom Scaling" (bool) If true we construct our own row and col
//!             scaling, otherwise we leave it to the method.
//! "OutputLevel" (int) controls the verbosity of the method.
//!
class SparseDirectSolver : public Ifpack_Preconditioner 
{
      
public:

  typedef enum {KLU,UMFPACK,CHOLMOD,PARDISO} SolverType;

  //! \name Constructors/Destructors.
  //!@{
  //! Constructor.
  SparseDirectSolver(Epetra_RowMatrix* Matrix);

  //! Copy constructor.
  SparseDirectSolver(const SparseDirectSolver& rhs);

  //! Operator=.
  SparseDirectSolver& operator=(const SparseDirectSolver& rhs);

  //! Destructor
  virtual ~SparseDirectSolver();

  //@}

  //@{ \name Attribute set methods.

   //! If set true, transpose of this operator will be applied (not implemented).
    /*! This flag allows the transpose of the given operator to be used 
     * implicitly.  
      
    \param 
	   UseTranspose_in - (In) If true, multiply by the transpose of operator, 
	   otherwise just use operator.

    \return Integer error code, set to 0 if successful.  Set to -1 if this implementation does not support transpose.
  */

  virtual int SetUseTranspose(bool UseTranspose_in);
  //@}
  
  //@{ \name Mathematical functions.

    //! Applies the matrix to an Epetra_MultiVector.
  /*! 
    \param
    X - (In) A Epetra_MultiVector of dimension NumVectors to multiply with matrix.
    \param 
    Y - (Out) A Epetra_MultiVector of dimension NumVectors containing the result.

    \return Integer error code, set to 0 if successful.
    */
    virtual int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

    //! Applies the preconditioner to X, returns the result in Y.
  /*! 
    \param 
    X - (In) A Epetra_MultiVector of dimension NumVectors to be preconditioned.
    \param 
    Y - (Out) A Epetra_MultiVector of dimension NumVectors containing result.

    \return Integer error code, set to 0 if successful.

    \warning In order to work with AztecOO, any implementation of this method 
    must support the case where X and Y are the same object.
    */
    virtual int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

    //! Returns the infinity norm of the global matrix (not implemented)
    virtual double NormInf() const;
  //@}
  
  //@{ \name Attribute access functions

    //! Returns a character string describing the operator
    virtual const char * Label() const;

  //! Sets the label. We allow setting the label in otherwise
  //! const objects because it has no effect on the object's
  //! behavior.
  inline void SetLabel(const char* Label_in)
  {
    label_ = Label_in;
  }

    //! Returns the current UseTranspose setting.
    virtual bool UseTranspose() const;

    //! Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
    virtual bool HasNormInf() const;

    //! Returns a pointer to the Epetra_Comm communicator associated with this operator.
    virtual const Epetra_Comm & Comm() const;

    //! Returns the Epetra_Map object associated with the domain of this operator.
    virtual const Epetra_Map & OperatorDomainMap() const;

    //! Returns the Epetra_Map object associated with the range of this operator.
    virtual const Epetra_Map & OperatorRangeMap() const;

  //@}

  //@{ \name Construction and application methods.
 
  //! Returns \c true is the preconditioner has been successfully initialized.
  virtual bool IsInitialized() const
  {
    return(IsInitialized_);
  }

  //! Initializes the preconditioners.
  /*! \return
   * 0 if successful, 1 if problems occurred.
   */
  virtual int Initialize();

  //! Returns \c true if the preconditioner has been successfully computed.
  virtual bool IsComputed() const
  {
    return(IsComputed_);
  }

  //! Computes the preconditioners.
  /*! \return
   * 0 if successful, 1 if problems occurred.
   */
  virtual int Compute();

  //! Sets all the parameters for the preconditioner.
  /*! Parameters currently supported:
   * - \c "amesos: solver type" : Specifies the solver type
   *   for Amesos. Default: \c Amesos_Klu.
   *
   * The input list will be copied, then passed to the Amesos
   * object through Amesos::SetParameters().
   */   
  virtual int SetParameters(Teuchos::ParameterList& List);

  //@}

  //@{ \name Query methods.

  //! Returns a const reference to the internally stored matrix.
  virtual const Epetra_RowMatrix& Matrix() const
  {
    return(*Matrix_);
  }

  //! Returns the estimated condition number, computes it if necessary.
  virtual double Condest(const Ifpack_CondestType CT = Ifpack_Cheap,
                         const int MaxIters = 1550,
                         const double Tol = 1e-9,
			 Epetra_RowMatrix* Matrix_in= 0);
  
  //! Returns the estimated condition number, never computes it.
  virtual double Condest() const
  {
    return(Condest_);
  }

  //! Returns the number of calls to Initialize().
  virtual int NumInitialize() const
  {
    return(NumInitialize_);
  }

  //! Returns the number of calls to Compute().
  virtual int NumCompute() const
  {
    return(NumCompute_);
  }

  //! Returns the number of calls to ApplyInverse().
  virtual int NumApplyInverse() const
  {
    return(NumApplyInverse_);
  }

  //! Returns the total time spent in Initialize().
  virtual double InitializeTime() const
  {
    return(InitializeTime_);
  }

  //! Returns the total time spent in Compute().
  virtual double ComputeTime() const
  {
    return(ComputeTime_);
  }

  //! Returns the total time spent in ApplyInverse().
  virtual double ApplyInverseTime() const
  {
    return(ApplyInverseTime_);
  }

  //! Returns the number of flops in the initialization phase.
  virtual double InitializeFlops() const
  {
    return(0.0);
  }

  //! Returns the total number of flops to computate the preconditioner.
  virtual double ComputeFlops() const
  {
    return(ComputeFlops_);
  }

  //! Returns the total number of flops to apply the preconditioner.
  virtual double ApplyInverseFlops() const
  {
    return(ApplyInverseFlops_);
  }

  // Returns a constant reference to the internally stored 
  virtual const Teuchos::ParameterList& List() const 
  {
    return(List_);
  }

  //! Prints on ostream basic information about \c this object.
  virtual std::ostream& Print(std::ostream& os) const;

  //@}
  
  //! return number of nonzeros in original matrix
  double NumGlobalNonzerosA() const;

  //! return number of nonzeros in factorization
  double NumGlobalNonzerosLU() const; 

#ifdef STORE_SD_LU
public:
#else
private:
#endif

  //! for debugging - writes a number of .txt and .m files to look at the reordered
  //! matrix and optionally the rhs and solution of a system
  //! if overwrite==true, the function only dumps files if <filePrefix>.m doesn't
  //! exist yet.
  void DumpSolverStatus(std::string filePrefix,
                  bool overwrite=true,
                  Teuchos::RCP<const Epetra_MultiVector> X=Teuchos::null, 
                  Teuchos::RCP<const Epetra_MultiVector> B=Teuchos::null) const;

  

protected:
  
  //@{ \name Methods to get/set private data

  //! Sets \c IsInitialized_.
  inline void SetIsInitialized(const bool IsInitialized_in)
  {
    IsInitialized_ = IsInitialized_in;
  }

  //! Sets \c IsComputed_.
  inline void SetIsComputed(const int IsComputed_in)
  {
    IsComputed_ = IsComputed_in;
  }

  //! Sets \c NumInitialize_.
  inline void SetNumInitialize(const int NumInitialize_in)
  {
    NumInitialize_ = NumInitialize_in;
  }

  //! Sets \c NumCompute_.
  inline void SetNumCompute(const int NumCompute_in)
  {
    NumCompute_ = NumCompute_in;
  }

  //! Sets \c NumApplyInverse_.
  inline void SetNumApplyInverse(const int NumApplyInverse_in)
  {
    NumApplyInverse_ = NumApplyInverse_in;
  }

  //! Sets \c InitializeTime_.
  inline void SetInitializeTime(const double InitializeTime_in)
  {
    InitializeTime_ = InitializeTime_in;
  }

  //! Sets \c ComputeTime_.
  inline void SetComputeTime(const double ComputeTime_in)
  {
    ComputeTime_ = ComputeTime_in;
  }

  //! Sets \c ApplyInverseTime_.
  inline void SetApplyInverseTime(const double ApplyInverseTime_in)
  {
    ApplyInverseTime_ = ApplyInverseTime_in;
  }

  //! Sets \c ComputeFlops_.
  inline void SetComputeFlops(const double ComputeFlops_in)
  {
    ComputeFlops_ = ComputeFlops_in;
  }

  //! Sets \c ComputeFlops_.
  inline void SetApplyInverseFlops(const double ApplyInverseFlops_in)
  {
    ApplyInverseFlops_ = ApplyInverseFlops_in;
  }

  //! Set \c List_.
  inline void SetList(const Teuchos::ParameterList& List_in)
  {
    List_ = List_in;
  }
  //@}

private:

  //! Pointers to the matrix to be preconditioned.
  Teuchos::RCP<const Epetra_RowMatrix> Matrix_;

  //! Contains a copy of the input parameter list.
  Teuchos::ParameterList List_;
  
  //! indicates which solver package is being used
  SolverType method_;

  //! Contains the label of \c this object.
  std::string label_;
  //! If true, the linear system on this processor is empty, thus the preconditioner is null operation.
  bool IsEmpty_;
  //! If true, the preconditioner has been successfully initialized.
  bool IsInitialized_;
  //! If true, the preconditioner has been successfully computed.
  bool IsComputed_;
  //! If true, the preconditioner solves for the transpose of the matrix.
  bool UseTranspose_;

  //! Contains the number of successful calls to Initialize().
  int NumInitialize_;
  //! Contains the number of successful call to Compute().
  int NumCompute_;
  //! Contains the number of successful call to ApplyInverse().
  mutable int NumApplyInverse_;

  //! Contains the time for all successful calls to Initialize().
  double InitializeTime_;
  //! Contains the time for all successful calls to Compute().
  double ComputeTime_;
  //! Contains the time for all successful calls to ApplyInverse().
  mutable double ApplyInverseTime_;
  //! Time object.
  Teuchos::RCP<Epetra_Time> Time_;

  //! Contains the number of flops for Compute().
  double ComputeFlops_;
  //! Contain sthe number of flops for ApplyInverse().
  double ApplyInverseFlops_;

  //! Contains the estimated condition number.
  double Condest_;
  
  //!
  int MyPID_;

  //! serial matrix
  Teuchos::RCP<const Epetra_RowMatrix> serialMatrix_;
  
  //! diagonal scaling
  Teuchos::RCP<Epetra_Vector> scaLeft_,scaRight_;
  
  //! import to serial
  Teuchos::RCP<Epetra_Import> serialImport_;
  
  //! use Umfpack or our own ordering
  bool ownOrdering_;

  //! use Umfpack or our own scaling
  bool ownScaling_;

  //! \name SuiteSparse interface, reordering etc
  //@{

    //! Umfpack internal opaque object
    void *umf_Symbolic_;
    //! Umfpack) internal opaque object
    void *umf_Numeric_;
    //! options for Umfpack
    Teuchos::Array<double> umf_Control_;
    //! Umfpack output
    Teuchos::Array<double> umf_Info_;
    
    mutable int *pardiso_pt_[64];
    mutable int pardiso_iparam_[64];
    mutable int pardiso_mtype_;
    mutable Teuchos::Array<int> pardiso_perm_;
    bool pardiso_initialized_;
    
    //! KLU objects wrapped up so we don't need to include the header
    KluWrapper *klu_;
    
    //! row and column permutations
    Teuchos::Array<int> row_perm_, col_perm_;
    //!  Ap, Ai, Aval form the compressed row storage used by Umfpack
    mutable Teuchos::Array<int> Ap_;
    mutable Teuchos::Array<int> Ai_;
    mutable Teuchos::Array<double> Aval_;
    //@}

private:  

  //! gather matrix on rank 0 if necessary (creates serialMatrix_ and serialImport_)
  int ConvertToSerial();

  //! compute fill-reducing ordering before creating Umfpack matrix
  int FillReducingOrdering();
  
  //! compute scaling for the matrix
  int ComputeScaling();
  
    /*
    ConvertToCRS - Convert matirx to form expected by Umfpack
    and KLU: Ai, Ap, Aval
    Preconditions:
      ConvertToSerial() should have been called
      FillReducingOrdering() (optional)
      ComputeScaling (optional)
    Postconditions:
      Ai, Ap, and Aval are resized and populated with a compresses row storage 
      version of the input matrix A.
  */
  int ConvertToCRS();

  /*! symbolic factorization using Umfpack
  */      
  int UmfpackSymbolic();

  /*! numeric factorization using Umfpack
  */
  int UmfpackNumeric();

  /*! perform solve using Umfpack */
  int UmfpackSolve(const Epetra_MultiVector& B, Epetra_MultiVector& X) const;

  /*! symbolic factorization using KLU
  */      
  int KluSymbolic();

  /*! numeric factorization using KLU
  */
  int KluNumeric();

  /*! perform solve using KLU */
  int KluSolve(const Epetra_MultiVector& B, Epetra_MultiVector& X) const;

  /*! symbolic factorization using Pardiso
  */      
  int PardisoSymbolic();

  /*! numeric factorization using Pardiso
  */
  int PardisoNumeric();

  /*! perform solve using Pardiso */
  int PardisoSolve(const Epetra_MultiVector& B, Epetra_MultiVector& X) const;

};
}//namespace HYMLS

#endif // USE_AMESOS is not defined - own interface to SuiteSparse or KLU

#endif // HYMLS_SPARSE_DIRECT_SOLVER_H_H
