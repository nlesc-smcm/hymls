#ifndef HYMLS_PRECONDITIONER_H
#define HYMLS_PRECONDITIONER_H

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Ifpack_Preconditioner.h"
#include "Ifpack_SparseContainer.h"
#include "Ifpack_Amesos.h"
#include "Epetra_Time.h"

#include "BelosLinearProblem.hpp"
#include "BelosSolverManager.hpp"
#include "BelosEpetraAdapter.hpp"
#include "BelosOutputManager.hpp"

#include "HYMLS_OverlappingPartitioner.H"
#include "HYMLS_EigenUtils.H"

// forward declarations
class Epetra_Comm;
class Epetra_Map;
class Epetra_RowMatrix;
class Epetra_Import;


namespace HYMLS 
  {
  class SchurComplement;
  class SchurPreconditioner;
  }

namespace Teuchos 
  {
  class ParameterList;
  }

namespace HYMLS {

//! main solver class in HYMLS

/*! This class 
  - sets parameters for the problem
  - creates an overlapping partitioning (HYMLS::OverlappingPartitioner)
  - computes a factorization of the subdomain matrices
  - creates a HYMLS::SchurComplement
  - creates a HYMLS::SchurPreconditioner for the SC
  - creates a Belos iterative solver
  - solves the problem (in ApplyInverse())
  
  The solver has the following features:
  
  - recursive application: the SchurPreconditioner object will check
  the parameter "Number of Levels" and see if it should function as 
  a direct or approximate solver. For instance, if you set "Number  
  of Levels" to 1, the solver does the domain decomposition and then
  solves the separator problem directly. Setting it to 2 gives the  
  standard two-level method where the preconditioner does orthogonal
  transformation and dropping and creates a direct solver for the   
  Vsum problem. For more levels, it will create another instance of 
  this "Preconditioner" class.                                      
  
  - monitoring and deflating the most unstable modes: to activate   
  this, set the "Deflated Subspace Dimension" to a positive integer.
  The solver computes the eigenvalues closest to 0 of the precondi- 
  tioner. If an eigenvalue has croosed the imaginary axis, a message
  is issued. That's all for now, the deflation still has to be      
  implemented.
  
*/
class Preconditioner : public Ifpack_Preconditioner
  {
  
public:

  typedef Ifpack_SparseContainer<Ifpack_Amesos> ifpackSolverType_;
  typedef Belos::EpetraPrecOp belosPrecType_;
  typedef Belos::LinearProblem<double,Epetra_MultiVector,Epetra_Operator> belosProblemType_;
  typedef Belos::SolverManager<double,Epetra_MultiVector,Epetra_Operator> belosSolverType_;
  typedef Belos::OutputManager<double> belosOutputType_;

  //! Our Schur-complement is our friend:
  friend class SchurComplement;


  //!                                                     
  //! Constructor                                         
  //!                                                     
  //! the caller should typically just use                
  //!                                                     
  //! Preconditioner(K,params);                           
  //!                                                     
  //! The constructor with pre-constructed partitioner is 
  //! used for recursive application of the method        
  //! (SchurPreconditioner will construct a Preconditioner
  //  object based on an OverlappingPartitioner created by
  //! SpawnNextLevel())                                   
  //!                                                     
  Preconditioner(Teuchos::RCP<const Epetra_RowMatrix> K, 
      Teuchos::RCP<Teuchos::ParameterList> params,
      Teuchos::RCP<const OverlappingPartitioner> hid=Teuchos::null,
      int myLevel=1);

  //! destructor
  ~Preconditioner();

  //! for eigenvalue computations - set mass matrix
  void SetMassMatrix(Teuchos::RCP<const Epetra_CrsMatrix> B);
  
  //! write solver data (like domain decomposition, separators ...) 
  //! to an m-file so that it can be imported to MATLAB.            
  void Visualize(std::string mfilename, bool no_recurse=false) const;

  //!\name Ifpack_Preconditioner interface
  
  //@{

  //! Sets all parameters for the preconditioner.
  int SetParameters(Teuchos::ParameterList& List);

  //! Computes all it is necessary to initialize the preconditioner.
  int Initialize();

  //! Returns true if the  preconditioner has been successfully initialized, false otherwise.
  bool IsInitialized() const;

  //! Computes all it is necessary to apply the preconditioner.
  int Compute();

  //! Returns true if the  preconditioner has been successfully computed, false otherwise.
  bool IsComputed() const;

  //! Computes the condition number estimate, returns its value.
  double Condest(const Ifpack_CondestType CT = Ifpack_Cheap,
                         const int MaxIters = 1550,
                         const double Tol = 1e-9,
                         Epetra_RowMatrix* Matrix = 0);

  //! Returns the computed condition number estimate, or -1.0 if not computed.
  double Condest() const;

  //! Applies the operator (not implemented)
  int Apply(const Epetra_MultiVector& X,
                           Epetra_MultiVector& Y) const {return -1;}

  //! Applies the preconditioner to vector X, returns the result in Y.
  int ApplyInverse(const Epetra_MultiVector& X,
                           Epetra_MultiVector& Y) const;

  //! Returns a pointer to the matrix to be preconditioned.
  const Epetra_RowMatrix& Matrix() const;

  //! Returns the number of calls to Initialize().
  int NumInitialize() const;

  //! Returns the number of calls to Compute().
  int NumCompute() const;

  //! Returns the number of calls to ApplyInverse().
  int NumApplyInverse() const;

  //! Returns the time spent in Initialize().
  double InitializeTime() const;

  //! Returns the time spent in Compute().
  double ComputeTime() const;

  //! Returns the time spent in ApplyInverse().
  double ApplyInverseTime() const;

  //! Returns the number of flops in the initialization phase.
  double InitializeFlops() const;

  //! Returns the number of flops in the computation phase.
  double ComputeFlops() const;

  //! Returns the number of flops in the application of the preconditioner.
  double ApplyInverseFlops() const;

  //! Prints basic information on iostream. This function is used by operator<<.
  ostream& Print(std::ostream& os) const;

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
    const Epetra_Map & OperatorDomainMap() const {return *rangeMap_;}

    //! Returns the Epetra_Map object associated with the range of this operator.
    const Epetra_Map & OperatorRangeMap() const {return *rangeMap_;}
  

  

  //@}

protected: 

  //! level of this solver (1: finest)
  int myLevel_;

  //! communicator
  Teuchos::RCP<const Epetra_Comm> comm_;

  //! fake communicator
  Teuchos::RCP<const Epetra_Comm> serialComm_;
  
  //! matrix based on range map
  Teuchos::RCP<const Epetra_RowMatrix> matrix_;
  
  //! internal reordered matrix (first interior and then separator vars)
  Teuchos::RCP<Epetra_CrsMatrix> reorderedMatrix_;

  //! range/domain map of matrix_
  Teuchos::RCP<const Epetra_Map> rangeMap_;

  //! row map of this operator (first all interior and then all separator variables, no overlap)
  Teuchos::RCP<const Epetra_Map> rowMap_;
  
  //! importer from range to row map
  Teuchos::RCP<Epetra_Import> importer_;

  //! Parameter List
  Teuchos::RCP<Teuchos::ParameterList> params_;

  //! our own minimally overlapped and reordered partitioning:
  Teuchos::RCP<const OverlappingPartitioner> hid_;

  //! we use Ifpack conainers for solving the subdomain problems
  Teuchos::Array<Teuchos::RCP<ifpackSolverType_> > subdomainSolver_;
  
  //! operator representation of our Schur-complement
  mutable Teuchos::RCP<SchurComplement> Schur_;
  
  //! left- and right scaling for the Schur complement
  bool scaleSchur_;
  Teuchos::RCP<Epetra_Vector> schurScaLeft_, schurScaRight_;

  //! A12, A21 and A22-part of matrix
  //! 1: associated with interior variables
  //! 2: associated with separator variables, but non-overlapping
  //! The range and domain of these operators are the rowMap_.
  Teuchos::RCP<Epetra_CrsMatrix> A12_, A21_, A22_;

  //! A12-parts of matrix in separate blocks.
  Teuchos::Array<Teuchos::RCP<Epetra_CrsMatrix> > localA12_;

  //! A12-part of matrix in separate blocks
  Teuchos::Array<Teuchos::RCP<Epetra_CrsMatrix> > localA21_;

  //! A22-part of matrix in separate blocks
  Teuchos::Array<Teuchos::RCP<Epetra_CrsMatrix> > localA22_;
  
  //! map with _all_ interior elements
  Teuchos::RCP<const Epetra_Map> map1_;

  //! map for all interior variables in subdomain i (i.e. those not in S)
  //! these maps are based on the serialComm, so localMap1_[i] on procs j and k
  //! have nothing to do with each other.
  Teuchos::Array<Teuchos::RCP<const Epetra_Map> > localMap1_;

  //! map defining the rows/columns of the Schur complement (separator nodes)
  Teuchos::RCP<const Epetra_Map> map2_;

  //! column maps
  Teuchos::RCP<Epetra_Map> colMap1_,colMap2_;

  //! map for all separator variables around subdomain i (i.e. those in S)
  //! these maps are based on the serialComm, so localMap2_[i] on procs j and k
  //! have nothing to do with each other.
  Teuchos::Array<Teuchos::RCP<const Epetra_Map> > localMap2_;

  //! importers for blocks
  Teuchos::Array<Teuchos::RCP<Epetra_Import> > localImport1_, localImport2_;

  //! importers for global parts
  Teuchos::RCP<Epetra_Import> import1_, import2_;

//!\name Belos data structures for solving the Schur-complement problem iteratively
//@{

  //! right-hand side vector
  Teuchos::RCP<Epetra_MultiVector> schurRhs_;

  //! solution vector
  Teuchos::RCP<Epetra_MultiVector> schurSol_;
  
  //! preconditioning operator
  Teuchos::RCP<SchurPreconditioner> schurPrec_;

  //! Belos preconditioner interface
  Teuchos::RCP<belosPrecType_> schurPrecPtr_;

  //! Belos linear problem interface
  Teuchos::RCP<belosProblemType_> schurProblemPtr_;

  //! Belos solver
  Teuchos::RCP<belosSolverType_> schurSolverPtr_;

  //! Belos output manager (not sure if this is actually needed)
  Teuchos::RCP<belosOutputType_> schurOutputPtr_;

//@}

//!\name data structures to monitor and deflate unstable modes
//@{

//! mass matrix - can be set using SetmassMatrix(). If not set, the 
//! standard eigenproblem is solved.
Teuchos::RCP<const Epetra_CrsMatrix> massMatrix_;

  //! mass matrix for eigenvalue computations
  Teuchos::RCP<Epetra_CrsMatrix> schurMass_;

//! number of eigenvalues computed initially
int numEigs_;

//! number of deflated eigenmodes
int numDeflated_;

//! tolerance to remove a mode. An eigenmode            
//! is deflated if |lambda(P^{-1}A)-1|>tol (set by      
//! "Deflation Threshold" in the "Solver" sublist).     
double deflThres_;

//! eigen-pairs of the preconditioner
Teuchos::RCP<EigenUtils::Solution> schurEigs_;

//! basis for the eigenspace to be deflated
Teuchos::RCP<Epetra_MultiVector> deflBase_;

//@}

  //! if false, no preconditioner is constructed for the Schur-complement
  bool usePreconditioner_;
  
  //! use transposed operator?
  bool useTranspose_;
  
  //! infinity norm
  double normInf_;
  
  //! label
  std::string label_;
  
  //! timer
  mutable Teuchos::RCP<Epetra_Time> time_;
  
  //! has Initialize() been called?
  bool initialized_;

  //! has Compute() been called?
  bool computed_;
  
  //! how often has Initialize() been called?
  int numInitialize_;

  //! how often has Compute() been called?
  int numCompute_;

  //! how often has ApplyInverse() been called?
  mutable int numApplyInverse_;
  
  //! flops during Initialize()
  double flopsInitialize_;

  //! flops during Compute()
  double flopsCompute_;

  //! flops during ApplyInverse()
  mutable double flopsApplyInverse_;

  //! time during Initialize()
  mutable double timeInitialize_;

  //! time during Compute()
  mutable double timeCompute_;

  //! time during ApplyInverse()
  mutable double timeApplyInverse_;
  
  //!@}

#ifdef TESTING
  mutable bool dumpVectors_;
#endif

#ifdef DEBUGGING
public:
#else
protected:
#endif
  //! apply X=A11\B. This only works if the solver is computed. The input vectors  
  //! should be based on RowMap(), but only interior variables (Map1(:)) are touched..
  int ApplyInverseA11(const Epetra_MultiVector& B, Epetra_MultiVector& X) const;

  //! apply Y=A12*X. This only works if the solver is computed. The input vector
  //! X can be based on RowMap() or Map2(), Y should be based on RowMap().
  int ApplyA12(const Epetra_MultiVector& X, Epetra_MultiVector& Y,
               double *flops_count=NULL) const;

  //! apply Y=A21*X. This only works if the solver is computed. The input vector
  //! Y can be based on RowMap() or Map2(), X should be based on RowMap().
  int ApplyA21(const Epetra_MultiVector& X, Epetra_MultiVector& Y,
               double* flops_cuont=NULL) const;

  //! apply Y=A22*X. This only works if the solver is computed. The input vectors  
  //! can be based on RowMap() or Map2().
  int ApplyA22(const Epetra_MultiVector& X, Epetra_MultiVector& Y,
               double* flops_count=NULL) const;
  
  //! map for all variables (same as OperatorRangeMap() and OperatorDomainMap())
  const Epetra_Map& RowMap() const {return *rowMap_;}

  //! get interior map for all subdomains
  const Epetra_Map& Map1() const {return *map1_;}

  //! get interior map for subdomain sd
  const Epetra_Map& Map1(int sd) const {return *(localMap1_[sd]);}

  //! get map for all separators
  const Epetra_Map& Map2() const {return *map2_;}

  //! get separator map for one particular subdomain
  const Epetra_Map& Map2(int sd) const {return *(localMap2_[sd]);}
  
  //! get the decomposition object
  const HYMLS::OverlappingPartitioner& Partitioner() const {return *hid_;}

  //! get solver for a diagonal block of A11
  ifpackSolverType_& SolverA11(int sd) const {return *subdomainSolver_[sd];}
  
  //! get a submatrix A12
  const Epetra_CrsMatrix& A12(int sd) const {return *localA12_[sd];}

  //! get a submatrix A21
  const Epetra_CrsMatrix& A21(int sd) const {return *localA21_[sd];}

  //! get a submatrix A22
  const Epetra_CrsMatrix& A22(int sd) const {return *localA22_[sd];}

  //! get entire matrix A21
  const Epetra_CrsMatrix& A21() const {return *A21_;}

  //! get entire matrix A12
  const Epetra_CrsMatrix& A12() const {return *A12_;}

  //! get entire matrix A22
  const Epetra_CrsMatrix& A22() const {return *A22_;}

 private:
 
   //! executed as part of Initialize() and Compute() (import matrices etc)
   int InitializeCompute();
   
   //! setup deflation of nasty eigenmodes after computing the standard
   //! preconditioner for the Schur-complement.
   int SetupDeflation();

   //! in this function we define a number of standard problems
   //! for which the solver can be set up by simply setting the
   //! "Equations" parameter in the "Problem" list. If it is
   //! not set, one has to set the "Problem Definition" sublist,
   //! which is a little tricky (see this function for examples)
   int SetProblemDefinition(std::string eqn, Teuchos::ParameterList& list);


  };


}

#endif
