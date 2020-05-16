#ifndef HYMLS_SCHUR_PRECONDITIONER_H
#define HYMLS_SCHUR_PRECONDITIONER_H

#include "HYMLS_config.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "Ifpack_CondestType.h"
#include "Ifpack_Preconditioner.h"

#include "HYMLS_BorderedOperator.hpp"
#include "HYMLS_PLA.hpp"

#include <iosfwd>
#include <string>

// forward declarations
class Epetra_Comm;
class Epetra_Map;
class Epetra_RowMatrix;
class Epetra_FECrsMatrix;
class Epetra_Import;
class Ifpack_Container;
class Epetra_CrsMatrix;
#ifdef HYMLS_LONG_LONG
class Epetra_LongLongSerialDenseVector;
#else
class Epetra_IntSerialDenseVector;
#endif
class Epetra_SerialDensematrix;
class Epetra_MultiVector;
class Epetra_Operator;
class Epetra_Vector;

namespace Teuchos
  {
class ParameterList;
  }

namespace HYMLS {

class Epetra_Time;
class HierarchicalMap;
class OrthogonalTransform;
class OverlappingPartitioner;
class SchurComplement;

//! Approximation of the Schur-complement

/*! this class is initially created by a HYMLS::Preconditioner object.
  It will perform the reduction to a next-level Schur-complement
  by means of orthogonal transformations and dropping. To solve
  the reduced Schur-complement, either another HYMLS::Preconditioner is
  created (in a multi-level context) or a in case the level parameter
  plus one is equal to the "Number of Levels" a HYMLS::CoarseSolver
  is created.
*/
class SchurPreconditioner : public Ifpack_Preconditioner,
                            public BorderedOperator,
                            public PLA
  {

public:

  //! The SC operator passed into this class describes the
  //! Schur-complement. The testVector reflects scaling of the entries
  //! in the B-part (for Stokes-C), typically it is something like
  //! 1/dx or simply ones for scaled matrices, and is taken from level
  //! to level by applying the orthogonal transforms to it and
  //! extracting the Vsums.
  SchurPreconditioner(Teuchos::RCP<const SchurComplement> SC,
    Teuchos::RCP<const OverlappingPartitioner> hid,
    Teuchos::RCP<Teuchos::ParameterList> params,
    int level,
    Teuchos::RCP<Epetra_Vector> testVector);

  //! destructor
  virtual ~SchurPreconditioner();

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
  std::ostream& Print(std::ostream& os) const;

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

  //! original SC object, may be null if a matrix is passed in.
  Teuchos::RCP<const SchurComplement> SchurComplement_;

  //! my level ID
  int myLevel_;

  //! if myLevel_==maxLevel_ we use a direct solver
  int maxLevel_;

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

  //! switch for applying dropping
  bool applyDropping_;

  //! switch for applying the OT
  bool applyOT_;

  //! domain decomposition object
  Teuchos::RCP<const OverlappingPartitioner> hid_;

  //! row/range/domain map of Schur complement
  Teuchos::RCP<const Epetra_Map> map_;

  //! test vector to determine entries in orth. trans.
  Teuchos::RCP<Epetra_Vector> testVector_,localTestVector_;

  //! orthogonal transformaion for separators
  Teuchos::RCP<OrthogonalTransform> OT_;

  //! sparse matrix representation of OT
  Teuchos::RCP<Epetra_CrsMatrix> sparseMatrixOT_;

  //! solvers for separator blocks (in principle they could be
  //! either Sparse- or DenseContainers, but presently we
  //! just make them Dense (which makes sense for our purposes)
  Teuchos::Array<Teuchos::RCP<Ifpack_Container> > blockSolver_;

  //! sparse matrix representation of preconditioner
  Teuchos::RCP<Epetra_CrsMatrix> matrix_;

  //! map for the reduced problem (Vsum-nodes)
  Teuchos::RCP<const Epetra_Map> vsumMap_, overlappingVsumMap_;

  //! importer for Vsum nodes
  Teuchos::RCP<Epetra_Import> vsumImporter_;

  //! partitioner for the next level
  Teuchos::RCP<const OverlappingPartitioner> nextLevelHID_;

  //! right-hand side and solution for the reduced SC (based on linear map)
  mutable Teuchos::RCP<Epetra_MultiVector> vsumRhs_, vsumSol_;

  //! solver for the reduced Schur complement. Note that Ifpack_Preconditioner
  //! is implemented by both Amesos (direct solver) and our HYMLS::Solver,
  //! so we don't have to make a choice at this point.
  Teuchos::RCP<Ifpack_Preconditioner> reducedSchurSolver_;

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

  mutable bool dumpVectors_;

  //! \name data structures for bordering

  //! border split up and transformed by Householder
  Teuchos::RCP<Epetra_MultiVector> borderV_,borderW_;
  //! lower diagonal block of bordered system
  Teuchos::RCP<Epetra_SerialDenseMatrix> borderC_;

private:

  //! this function does the initialization things that have to be done
  //! before each Compute(), like rebuilding some solvers because the
  //! matrix pointers have changed. We internally take care not to do
  //! too much extra work by keeping some data structures if they exist
  int InitializeCompute();

  //! Initialize orthogonal transform
  int InitializeOT();

  //! Assemble the Schur complement of the Preconditioner
  //! object creating this SchurPreconditioner.
  int Assemble();

  //! Assemble the Schur complement of the Preconditioner
  //! object creating this SchurPreconditioner,
  //! apply orthogonal transformations and dropping on the
  //! fly. This variant is called if the SchurComplement is
  //! not yet assembled ('Constructed()').
  //!
  //! if paternOnly==true, a matrix with the right pattern
  //! but only 0 entries is created.
  int AssembleTransformAndDrop();

  //! Helper function for AssembleTransformAndDrop
  int ConstructSCPart(int k, Epetra_Vector const &localTestVector,
    Epetra_SerialDenseMatrix & Sk,
#ifdef HYMLS_LONG_LONG
    Epetra_LongLongSerialDenseVector &indices,
#else
    Epetra_IntSerialDenseVector &indices,
#endif
    Teuchos::Array<Teuchos::RCP<Epetra_SerialDenseMatrix> > &SkArray,
#ifdef HYMLS_LONG_LONG
    Teuchos::Array<Teuchos::RCP<Epetra_LongLongSerialDenseVector> > &indicesArray
#else
    Teuchos::Array<Teuchos::RCP<Epetra_IntSerialDenseVector> > &indicesArray
#endif
    ) const;

  //! Initialize dense solvers for diagonal blocks
  //! ("Block Diagonal" variant)
  int InitializeBlocks();

  //! Initialize single sparse solver for non-Vsums
  //! ("Domain Decomposition" variant)
  int InitializeSingleBlock();

  //! Compute the reduced Schur solver
  int ComputeNextLevel();

  //! Create a VSum map for computing the next level hid
  Teuchos::RCP<const Epetra_Map> CreateVSumMap(
    Teuchos::RCP<const HierarchicalMap> &sepObject) const;

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
  //! Compute scaling for a sparse matrix. This is currently unused.
  //!
  //! The scaling we use is as follows:
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
