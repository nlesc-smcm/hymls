#ifndef HYMLS_SOLVERCONTAINER_H
#define HYMLS_SOLVERCONTAINER_H

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "Ifpack_Preconditioner.h"
#include "Ifpack_Container.h"

#include "HYMLS_OverlappingPartitioner.H"

// forward declarations
class Epetra_Map;
class Epetra_RowMatrix;
class Ifpack_DenseContainer;

template<typename T>
class Ifpack_SparseContainer;

namespace Teuchos 
  {
  class ParameterList;
  }

namespace HYMLS {

class SparseDirectSolver;

//! main solver class in HYMLS

/*! This class is a wrapper around the Ifpack_DenseContainer
 *  and Ifpack_SparseContainer classes that we use as subdomain solvers
*/
class SolverContainer
  {

public:
  typedef enum {SPARSE, DENSE} Type;

  SolverContainer(Type SolverType, const int NumRows, const int NumVectors = 1);

  //! destructor
  ~SolverContainer();

  //! Sets all parameters for the preconditioner.
  int SetParameters(Teuchos::ParameterList& List)
  {
    return Container_->SetParameters(List);
  }

  //@{ Get/Set methods.
  //! Returns the number of rows of the matrix and LHS/RHS.
  int NumRows() const
  {
    return Container_->NumRows();
  }

  //! Returns the number of vectors in LHS/RHS.
  int NumVectors() const
  {
    return Container_->NumVectors();
  }

  //! Sets the number of vectors for LHS/RHS.
  int SetNumVectors(const int NumVectors_in)
  {
    return Container_->SetNumVectors(NumVectors_in);
  }

  //! Returns the i-th component of the vector Vector of LHS.
  double& LHS(const int i, const int Vector = 0)
  {
    return Container_->LHS(i, Vector);
  }
  
  //! Returns the i-th component of the vector Vector of RHS.
  double& RHS(const int i, const int Vector = 0)
  {
    return Container_->RHS(i, Vector);
  }

  //! Returns the ID associated to local row i. 
  /*!
   * The set of (local) rows assigned to this container is defined
   * by calling ID(i) = j, where i (from 0 to NumRows()) indicates
   * the container-row, and j indicates the local row in the calling
   * process.
   *
   * This is usually used to recorder the local row ID (on calling process)
   * of the i-th row in the container.
   */
  int& ID(const int i)
  {
    return Container_->ID(i);
  }

  //! Set the matrix element (row,col) to \c value.
  int SetMatrixElement(const int row, const int col,
                      const double value)
  {
    return Container_->SetMatrixElement(row, col, value);
  }

  //! Returns \c true is the container has been successfully initialized.
  bool IsInitialized() const
  {
    return Container_->IsInitialized();
  }

  //! Returns \c true is the container has been successfully computed.
  bool IsComputed() const
  {
    return Container_->IsComputed();
  }

  //! Returns the label of \e this container.
  const char* Label() const
  {
    return Container_->Label();
  }
//~ 
  //~ //! Returns a pointer to the internally stored map.
  //~ Teuchos::RCP<const Epetra_Map> Map() const;

  //! Returns a pointer to the internally stored solution multi-vector.
  Teuchos::RCP<const Epetra_MultiVector> LHS() const;

  //! Returns a pointer to the internally stored rhs multi-vector.
  Teuchos::RCP<const Epetra_MultiVector> RHS() const;
//~ 
  //~ //! Returns a pointer to the internally stored matrix.
  //~ Teuchos::RCP<const Epetra_CrsMatrix> Matrix() const;

  //! Returns a pointer to the internally stored ID's.
  const Epetra_IntSerialDenseVector& ID() const;
  //@}

  //@{ Mathematical functions.
  /*! 
   * \brief Initializes the container, by completing all the operations based 
   * on matrix structure.
   *
   * \note Opposed to what is stated in the Ifpack Solver containers,
   * we allow the matrix to be changed after which InitializeCompute()
   * has to be called.
   */
  int Initialize()
  {
    return Container_->Initialize();
  }

  /*! 
   * Reinitializes the container before computation starts. This
   * should skip some work and makes sure the ID list is still filled
   */
  int InitializeCompute();

  //! Finalizes the linear system matrix and prepares for the application of the inverse.
  int Compute(const Epetra_RowMatrix& Matrix_in)
  {
    return Container_->Compute(Matrix_in);
  }

  //! Apply the matrix to RHS, result is stored in LHS.
  int Apply()
  {
    return Container_->Apply();
  }

  //! Apply the inverse of the matrix to RHS, result is stored in LHS.
  int ApplyInverse()
  {
    return Container_->ApplyInverse();
  }

  //@}

  //! Returns the flops in Compute().
  double InitializeFlops() const
  {
    return Container_->InitializeFlops();
  }

  //! Returns the flops in Compute().
  double ComputeFlops() const
  {
    return Container_->ComputeFlops();
  }

  //! Returns the flops in Apply().
  double ApplyFlops() const
  {
    return Container_->ApplyFlops();
  }

  //! Returns the flops in ApplyInverse().
  double ApplyInverseFlops() const
  {
    return Container_->ApplyInverseFlops();
  }
  
  //! Prints basic information on iostream. This function is used by operator<<.
  ostream& Print(std::ostream& os) const
  {
    return Container_->Print(os);
  }

  //! return number of nonzeros in original matrix and the factorization
  int NumGlobalNonzeros() const;

  //! return the SparseContainer if the solver is a sparse one, otherwise null
  Teuchos::RCP<Ifpack_DenseContainer> DenseContainer() const;

  //! return the DenseContainer if the solver is a dense one, otherwise null
  Teuchos::RCP<Ifpack_SparseContainer<SparseDirectSolver> > SparseContainer() const;

protected:

  //! the actual Ifpack container
  Teuchos::RCP<Ifpack_Container> Container_;

  //! Contains a copy of the input parameter list.
  Teuchos::ParameterList List_;

  //! Contains the number of subdomains
  int num_sd_;
  
  //! Contains the solver type used
  Type SolverType_;
  
  };

}

#endif
