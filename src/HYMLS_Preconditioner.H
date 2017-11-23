#ifndef HYMLS_PRECONDITIONER_H
#define HYMLS_PRECONDITIONER_H

#include "HYMLS_config.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Epetra_RowMatrix.h"
#include "Epetra_CrsMatrix.h"

#include "Ifpack_Preconditioner.h"
#include "Ifpack_Container.h"
#include "Epetra_Time.h"

#include "BelosLinearProblem.hpp"
#include "BelosSolverManager.hpp"
#include "BelosEpetraAdapter.hpp"
#include "BelosOutputManager.hpp"

#include "HYMLS_OverlappingPartitioner.H"
#include "HYMLS_PLA.H"

#include "HYMLS_BorderedOperator.H"
#include "HYMLS_MatrixBlock.H"
#include "HYMLS_BlockScaling.H"

// forward declarations
class Epetra_Comm;
class Epetra_Map;
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
class Preconditioner : public Ifpack_Preconditioner, 
                       public BorderedOperator,
                       public PLA
  {
  
public:

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
      int myLevel=1,
      Teuchos::RCP<Epetra_Vector> testVector=Teuchos::null);

  //! destructor
  virtual ~Preconditioner();

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
  
  //!\name Teuchos::ParameterListAcceptor
  //@{
  //!
  void setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& list);
  
  //! get a list of valid parameters for this object
  Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;
  

  

  //@}

  //!\name HYMLS::BorderedOperator interface
  //@{
  
  //! set a border on the preconditioner. So instead of solving 
  //! Px=y we then solve Px+Vs=y, W'x+Cs=0. If W is omitted, W=V
  //! is used. If C is omitted, C=0 is used.                    
  //! To remove an existing border, pass in V=W=C=Teuchos::null.
  //!
  //! Implementation details
  //! ----------------------
  //!
  //! Let the original preconditioner be
  //!
  //! |A11 A12|      |A11  0 ||I A11\A12|
  //! |A21 A22|   ~= |A21  S ||0   I    |, S=A22 - A21*A11\A12,
  //!
  //! then the bordered variant will be
  //!
  //! |A11 A12 V1|      |A11    0              0       | |I A11\A12 Q1|
  //! |A21 A22 V2|   ~= |A21    S            V2-A21*Q1 | |0    I     0|
  //! |W1' W2' C |      |W1' W2'-W1'A11\A12   C-W1'Q1  | |0    0     I|, Q1=A11\V1.
  //!
  //! The lower right 2x2 block is the new 'bordered Schur Complement' and is handled by class
  //! SchurPreconditioner.
  //!
  int setBorder(Teuchos::RCP<const Epetra_MultiVector> V,
                Teuchos::RCP<const Epetra_MultiVector> W=Teuchos::null,
                Teuchos::RCP<const Epetra_SerialDenseMatrix> C=Teuchos::null);

  //! returns true if a border has been added
  bool HaveBorder() const {return V_ != Teuchos::null;}

  //! if a border has been added, apply [Y T]' = [P V; W' C] [X S]'
  int Apply(const Epetra_MultiVector& X, const Epetra_SerialDenseMatrix& S,
                  Epetra_MultiVector& Y,       Epetra_SerialDenseMatrix& T) const;

  //! if a border has been added, apply [X S]' = [K V; W' C]\[Y T]'
  int ApplyInverse(const Epetra_MultiVector& Y, const Epetra_SerialDenseMatrix& T,
                         Epetra_MultiVector& X,       Epetra_SerialDenseMatrix& S) const;
  //x@}

  //! reset matrix pointer. This should only be used if you are positive that the 
  //! ordering can be reused for this matrix, otherwise the Preconditioner has to 
  //! be destroyed and rebuilt (the ordering can be reused if the pattern of the  
  //! matrix is essentially unchanged since the last Initialize() call). After    
  //! setting the matrix by this function you have to call Initialize() again, but
  //! it will skip rebuilding the ordering.
  void SetMatrix(Teuchos::RCP<const Epetra_CrsMatrix> matrix)
    {
    matrix_=matrix;
    initialized_=false;
    }

    //! specify that the matrix on which this preconditioner was created was actually  
    //! Sl*A*Sr and that the ApplyInverse() function should perform y <- Sr\(P\(Sl\x)).
    //! If scaling==Teuchos::null, the scaling (if any) is removed.
    //!
    //! *note* that you only need to set the scaling object here if you want to use the
    //! preconditioner to solve an *unscaled linear system*!
    void SetBlockScaling(Teuchos::RCP<const HYMLS::BlockScaling> scaling)
      {
      scaling_=scaling;
      }

protected:

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

  //! our own minimally overlapped and reordered partitioning:
  Teuchos::RCP<const OverlappingPartitioner> hid_;

  //! level of this solver (1: finest)
  int myLevel_;
  
  //! string to indicate wether we use a dense or sparse solver for the subdomains
  std::string sdSolverType_;

  //! operator representation of our Schur-complement
  mutable Teuchos::RCP<SchurComplement> Schur_;
  
  //! left- and right scaling for the Schur complement
  bool scaleSchur_;
  Teuchos::RCP<Epetra_Vector> schurScaLeft_, schurScaRight_;

  //! A11, A12, A21 and A22-part of matrix
  //! 1: associated with interior variables
  //! 2: associated with separator variables, but non-overlapping
  //! The range and domain of these operators are the rowMap_.
  Teuchos::RCP<MatrixBlock> A11_, A12_, A21_, A22_;

  //! right-hand side vector for Schur complement
  mutable Teuchos::RCP<Epetra_MultiVector> schurRhs_;

  //! solution vector for Schur complement
  mutable Teuchos::RCP<Epetra_MultiVector> schurSol_;
  
  //! a test vector for constructing good orthogonal transformations
  //! (all ones on the first level, passed to the approximate SC)
  Teuchos::RCP<Epetra_Vector> testVector_;
  
  //! scaling (if any).
  Teuchos::RCP<const HYMLS::BlockScaling> scaling_;
  
  //! preconditioning operator
  Teuchos::RCP<SchurPreconditioner> schurPrec_;

  //! \name data structures for solving a 'bordered' system
  //@{
 
  //! bordering: if addBorder has been called, we solve [K V; W' C] [x; s] = [b; O].
  Teuchos::RCP<Epetra_MultiVector> borderV1_, borderV2_, borderW1_, borderW2_;

  //! A11\V1
  Teuchos::RCP<Epetra_MultiVector> borderQ1_;

  //! border for the Schur complement (after eliminating A11)
  Teuchos::RCP<Epetra_MultiVector> borderSchurV_, borderSchurW_;
  
  //! lower right block of bordered Schur system
  Teuchos::RCP<Epetra_SerialDenseMatrix> borderSchurC_;
  
  //@}
  
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

  mutable bool dumpVectors_;

  //! max num threads to use for subdomain solve
  int numThreadsSD_;

#ifdef HYMLS_DEBUGGING
public:
#else
protected:
#endif

  //! map for all variables (same as OperatorRangeMap() and OperatorDomainMap())
  const Epetra_Map& RowMap() const {return *rowMap_;}
  
  //! get the decomposition object
  const HYMLS::OverlappingPartitioner& Partitioner() const {return *hid_;}
  
 private:
 
   //! executed as part of Initialize() and Compute() (import matrices etc)
   int InitializeCompute();
   
   //! in this function we define a number of standard problems
   //! for which the solver can be set up by simply setting the
   //! "Equations" parameter in the "Problem" list. If it is
   //! not set, one has to set the "Partitioner" sublist,
   //! which is a little tricky (see this function for examples)
   int SetProblemDefinition(std::string eqn, Teuchos::ParameterList& list);


  };


}

#endif
