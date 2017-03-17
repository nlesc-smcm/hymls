#ifndef HYMLS_SCHUR_PRECONDITIONER_H
#define HYMLS_SCHUR_PRECONDITIONER_H

#include "HYMLS_config.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Ifpack_Preconditioner.h"
#include "Ifpack_Amesos.h"
#include "Ifpack_SparseContainer.h"
#include "Epetra_Time.h"

#include "HYMLS_OverlappingPartitioner.H"
#include "HYMLS_OrthogonalTransform.H"
#include "HYMLS_BorderedOperator.H"
#include "HYMLS_PLA.H"

// forward declarations
class Epetra_Comm;
class Epetra_Map;
class Epetra_RowMatrix;
class Epetra_FECrsMatrix;
class Epetra_Import;
class Ifpack_Container;

namespace EpetraExt
  {
  class MultiVector_Reindex;
  class CrsMatrix_Reindex;
  class RestrictedCrsMatrixWrapper;
  class RestrictedMultiVectorWrapper;
  }
/*
namespace HYMLS 
  {
  class OverlappingPartitioner;
  class OrthogonalTransform;
  }
*/
namespace Teuchos 
  {
  class ParameterList;
  }

namespace HYMLS {

class SchurComplement;

//! Approximation of the Schur-Complement 

/*! this class is initially created by a HYMLS::Preconditioner object.
    It will perform the reduction to a next-level Schur-complement
    by means of orthogonal transformations and dropping. To solve 
    the reduced Schur-complement, either another HYMLS::Solver is 
    created (in a multi-level context) or another HYMLS::SchurPreconditioner.
    If the level parameter passed to the constructor is equal to the
    "Number of Levels" parameter in the input list, this class doesn't
    perform additional reductions but instead computes a scaling and a
    direct solver for the input matrix.
*/
class SchurPreconditioner : public Ifpack_Preconditioner,
                            public BorderedOperator,
                            public PLA
  {
  
public:

  //! The SC operator passed into this class can either be an   
  //! Epetra_CrsMatrix constructed by calling the               
  //! SchurComplement->Construct function, or the SchurComple-  
  //! ment object itself.. hid may be null on the coarsest      
  //! level. The testVector reflects scaling of the entries in  
  //! the B-part (for Stokes-C), typically it is something like 
  //! 1/dx or simply ones for scaled matrices, and is taken     
  //!from level to level by applying the orthogonal transforms  
  //! to it and extracting the Vsums.                           
  SchurPreconditioner(Teuchos::RCP<const Epetra_Operator> SC, 
                      Teuchos::RCP<const OverlappingPartitioner> hid,
                      Teuchos::RCP<Teuchos::ParameterList> params,
                      int level,
                      Teuchos::RCP<Epetra_Vector> testVector);

  //! destructor
  ~SchurPreconditioner();

  //! apply orthogonal transforms to a vector v
  
  /*! This class is actually a preconditioner for the system
      H'SH H'x = H'y, this function computes HV and H'V for some
      multivector V.
  */
  int ApplyOT(bool trans, Epetra_MultiVector& v, double* flops=NULL) const;
    
  //! write matlab data for visualization
  void Visualize(std::string filename, bool recurse=true) const;

  //!\name Ifpack_Preconditioner interface
  
  //@{

  //! Sets all parameters for the preconditioner.
  int SetParameters(Teuchos::ParameterList& List);
  
  //! from the Teuchos::ParameterListAcceptor base class
  void setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& list);
  
  //! get a list of valid parameters for this object
  Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;

  //! Computes all it is necessary to initialize the preconditioner.
  //! this function does not initialize anything, in fact, it de-ini
  //! tializes the preconditioner. Compute() does the whole initia- 
  //! lization, skipping setup of objects that already exist.
  int Initialize();

  //! Returns true if the  preconditioner has been successfully initialized, false otherwise.
  bool IsInitialized() const;

  //! Computes all that is necessary to apply the preconditioner. The first Compute() call
  //! after construction or Initialize() is more expensive than subsequent calls because  
  //! it does some more setup.
  int Compute();

  //! same as Compute() if build_schur==false. Otherwise the Schur-complement is 
  //! also assembled in the member Schur_, which is cheaper then assembling it there
  //! and in this class.
  int Compute(bool build_schur);

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
                  Epetra_MultiVector& Y) const;

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
  bool HasNormInf() const {return true;}

  //! infinity norm
  double NormInf() const;

  //! label
  const char* Label() const {return label_.c_str();}
  
  //! use transpose?
  bool UseTranspose() const {return useTranspose_;}
  
  //! communicator
  const Epetra_Comm & Comm() const {return *comm_;}
  
    //! Returns the Epetra_Map object associated with the domain of this operator.
    const Epetra_Map & OperatorDomainMap() const {return *map_;}

    //! Returns the Epetra_Map object associated with the range of this operator.
    const Epetra_Map & OperatorRangeMap() const {return *map_;}
  

  

  //@}

  //! \name HYMLS BorderedOperator interface
  //@{
  
  //!
  int setBorder(Teuchos::RCP<const Epetra_MultiVector> V,
                Teuchos::RCP<const Epetra_MultiVector> W,
                Teuchos::RCP<const Epetra_SerialDenseMatrix> C=Teuchos::null);
  
  //!
  bool HaveBorder() const {return haveBorder_;}
  
  //!
  int Apply(const Epetra_MultiVector & B, const Epetra_SerialDenseMatrix & C, 
                  Epetra_MultiVector& X, Epetra_SerialDenseMatrix & Y) const;

  //!
  int ApplyInverse(const Epetra_MultiVector & B, const Epetra_SerialDenseMatrix & C, 
                  Epetra_MultiVector& X, Epetra_SerialDenseMatrix & Y) const;
  
  //@}

protected: 
  
  //! communicator
  Teuchos::RCP<const Epetra_Comm> comm_;
  
  //! sparse matrix representation of the SC operator we want to precondition.
  //! May be null if a SchurComplement object is passed in.
  Teuchos::RCP<const Epetra_CrsMatrix> SchurMatrix_;

  //! for internal use in case of 1-level method
  Teuchos::RCP<Epetra_FECrsMatrix> tmpMatrix_;
  
  //! original SC object, may be null if a matrix is passed in.
  Teuchos::RCP<const SchurComplement> SchurComplement_;
    
  //! my level ID
  int myLevel_;
  
  //! if myLevel_==maxLevel_ we use a direct solver
  int maxLevel_;
  
  //! if the processor has no rows in the present SC, this is false.
  bool amActive_;
  
  //! we currently implement two variants of the approximate    
  //! Schur-Complement: one with a block diagonal approximation 
  //! of the non-Vsums, and one with a sparse direct solver for 
  //! the non-Vsums. The latter is more expensive but decreases 
  //! the number of iterations, so it is more suitable for      
  //! massively parallel runs.
  std::string variant_;
  
  //! obtained from user parameter "Dense Solvers on Level", used
  //! to switch to dense direct solvers on the subdomains on level
  //! denseSwitch_.
  int denseSwitch_;
  
  //! domain decomposition object
  Teuchos::RCP<const OverlappingPartitioner> hid_;
  
  //! row/range/domain map of Schur complement
  Teuchos::RCP<const Epetra_Map> map_;

  //! test vector to determine entries in orth. trans.
  Teuchos::RCP<Epetra_Vector> testVector_,localTestVector_;
    
  //! orthogonal transformaion for separators
  Teuchos::RCP<OrthogonalTransform> OT;
  
  //! sparse matrix representation of OT
  Teuchos::RCP<Epetra_CrsMatrix> sparseMatrixOT_;
  
  //! solvers for separator blocks (in principle they could be  
  //! either Sparse- or DenseContainers, but presently we       
  //! just make them Dense (which makes sense for our purposes) 
  Teuchos::Array<Teuchos::RCP<Ifpack_Container> > blockSolver_;
  
  //! sparse matrix representation of preconditioner
  Teuchos::RCP<Epetra_CrsMatrix> matrix_;
    
  //! map for the reduced problem (Vsum-nodes)
  Teuchos::RCP<const Epetra_Map> vsumMap_,vsumColMap_;
  
  //! importer for Vsum nodes
  Teuchos::RCP<Epetra_Import> vsumImporter_;

  //! linear map for the reduced SC
  Teuchos::RCP<Epetra_Map> linearMap_;
  
  //! pivot order (for MUMPS, experimental)
  Teuchos::Array<int> pivot_order_;

  //! this is to reindex the reduced SC, which is
  //! important when using a direct solver (I think)
  Teuchos::RCP<EpetraExt::CrsMatrix_Reindex> reindexA_;
    
  //! reindex corresponding vectors
  Teuchos::RCP<EpetraExt::MultiVector_Reindex> reindexX_, reindexB_;

  //! this is to restrict the reduced Schur problem on the
  //! coarsest level to only the active procs so that     
  //! independently of the Amesos solver, the number of   
  //! procs participating in the factorization is determi-
  //! ned by our own algorithm, e.g. max(np,nsd).
  Teuchos::RCP<EpetraExt::RestrictedCrsMatrixWrapper> restrictA_;
    
  //! restrict corresponding vectors
  Teuchos::RCP<EpetraExt::RestrictedMultiVectorWrapper> restrictX_, restrictB_;
  
  //! partitioner for the next level
  Teuchos::RCP<const OverlappingPartitioner> nextLevelHID_;
  
  //! parameters for the next level solver
  Teuchos::RCP<Teuchos::ParameterList> nextLevelParams_;
  
  //! sparse matrix representation of the reduced Schur-complement
  //! (associated with Vsum nodes)
  Teuchos::RCP<Epetra_CrsMatrix> reducedSchur_;

  //! right-hand side and solution for the reduced SC (based on linear map)
  mutable Teuchos::RCP<Epetra_MultiVector> vsumRhs_, vsumSol_;

  // View of SC2 with linear map
  Teuchos::RCP<Epetra_CrsMatrix> linearMatrix_;

  // View of SC2 with linear map and no empty partitions (restricted Comm)
  Teuchos::RCP<Epetra_CrsMatrix> restrictedMatrix_;

  //! Views and copies of vectors used in ApplyInverse(), mutable temporary data
  mutable Teuchos::RCP<Epetra_MultiVector> linearRhs_, linearSol_, restrictedRhs_, restrictedSol_;
  
  //! solver for the reduced Schur complement. Note that Ifpack_Preconditioner 
  //! is implemented by both Amesos (direct solver) and our HYMLS::Solver,     
  //! so we don't have to make a choice at this point.
  Teuchos::RCP<Ifpack_Preconditioner> reducedSchurSolver_;
  
  //! left and right scaling vectors for the reduced Schur-complement
  Teuchos::RCP<Epetra_Vector> reducedSchurScaLeft_,reducedSchurScaRight_;
  
  //! use transposed operator?
  bool useTranspose_;
  
  //! true if addBorder() has been called with non-null args
  bool haveBorder_;
  
  //! infinity norm
  double normInf_;
  
  //! label
  std::string label_;
  
  //! timer
  mutable Teuchos::RCP<Epetra_Time> time_;
 
 
  //! true if the Schur complement has 0 global rows
  bool isEmpty_;
  
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
  mutable double flopsInitialize_;

  //! flops during Compute()
  mutable double flopsCompute_;

  //! flops during ApplyInverse()
  mutable double flopsApplyInverse_;

  //! time during Initialize()
  mutable double timeInitialize_;

  //! time during Compute()
  mutable double timeCompute_;

  //! time during ApplyInverse()
  mutable double timeApplyInverse_;
  
  //! we can replace a number of rows and cols of the reduced SC
  //! by Dirichlet conditions. This is used to fix the pressure
  //! level
  Teuchos::Array<int> fix_gid_;
  
  //! subdivide separators created by the standard decomposition.
  //! This is necessary for i.e. THCM, where each subdomain retains
  //! two pressures. Velocities may couple to either of these, giving
  //! cross patterns in the reduced 'Grad' part. The subdivisino is 
  //! based on the Grad-part, so that the transform is applied to variables
  //! coupling to the same set of pressures.
  bool subdivideSeparators_;

  mutable bool dumpVectors_;
  
  //! \name data structures for bordering
  
  //! border split up and transformed by Householder
  Teuchos::RCP<Epetra_MultiVector> borderV_,borderV2_,borderW_,borderW2_;
  //! lower diagonal block of bordered system
  Teuchos::RCP<Epetra_SerialDenseMatrix> borderC_;
  
  //! augmented matrix for V-sums, [M22 V2; W2 C]
  Teuchos::RCP<Epetra_RowMatrix> augmentedMatrix_;
  
private:

  //! this function does the initialization things that have to be done
  //! before each Compute(), like rebuilding some solvers because the  
  //! matrix pointers have changed. We internally take care not to do  
  //! too much extra work by keeping some data structures if they exist
  int InitializeCompute();

  //! initialize grouping of separators for OT and factorization
  int InitializeSeparatorGroups();

  //! Initialize orthogonal transform
  int InitializeOT();
  
  //! apply orthogonal transform to sparse matrix,
  //! giving the matrix drop(T'*S*T) in matrix_.
  int TransformAndDrop();

  //! Assemble the Schur complement of the Preconditioner
  //! object creating this SchurPreconditioner, 
  //! apply orthogonal transformations and dropping on the
  //! fly. This variant is called if the SchurComplement is
  //! not yet assembled ('Constructed()'). 
  //!
  //! if paternOnly==true, a matrix with the right pattern
  //! but only 0 entries is created.
  int AssembleTransformAndDrop();

  //! Initialize dense solvers for diagonal blocks
  //! ("Block Diagonal" variant)
  int InitializeBlocks();

  //! Initialize single sparse solver for non-Vsums
  //! ("Domain Decomposition" variant)
  int InitializeSingleBlock();

  //! Initialize reduced Schur solver
  int InitializeNextLevel();

  //! apply block diagonal of non-Vsums inverse to vector
  //! (this is called if variant_=="Block Diagonal")
  int ApplyBlockDiagonal(const Epetra_MultiVector& B, Epetra_MultiVector& X) const;

  //! block triangular solve with non-Vsum blocks (does not touch Vsum-part of X)
  int ApplyBlockLowerTriangular(const Epetra_MultiVector& B, Epetra_MultiVector& X) const;

  //! block triangular solve with non-Vsum blocks (does not touch Vsum-part of X)
  int ApplyBlockUpperTriangular(const Epetra_MultiVector& B, Epetra_MultiVector& X) const;

  //! general block triangular solve with non-Vsum blocks (does not touch Vsum-part of X)
  int BlockTriangularSolve(const Epetra_MultiVector& B, Epetra_MultiVector& X,
        int start, int end, int incr) const;

  //! update Vsum part of the vector before solving reduced SC problem
  int UpdateVsumRhs(const Epetra_MultiVector& B, Epetra_MultiVector& X) const;

  //!                                                                   
  //! compute scaling for a sparse matrix.                              
  //!                                                                   
  //! the scaling we use is as follows:                                 
  //!                                                                   
  //! a = sqrt(max(|diag(A)|));                                         
  //! if A(i,i) != 0, sca_left = sca_right = 1/a                        
  //! else            sca_left = sca_right = a                          
  //!                                                                   
  //! If sca_left and/or sca_right are null, they are created.          
  //!                                                                   
  int ComputeScaling(const Epetra_CrsMatrix& A, 
        Teuchos::RCP<Epetra_Vector>& sca_left,
        Teuchos::RCP<Epetra_Vector>& sca_right);
  
  };
  
}

#endif
