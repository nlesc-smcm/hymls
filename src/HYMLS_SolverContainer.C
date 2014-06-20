//#define BLOCK_IMPLEMENTATION 1
//#include "HYMLS_no_debug.H"

#include "HYMLS_SolverContainer.H"

#include "HYMLS_Macros.H"
#include "HYMLS_Tools.H"
#include "HYMLS_MatrixUtils.H"
#include "HYMLS_OverlappingPartitioner.H"

#include "HYMLS_SparseDirectSolver.H"

#include "Epetra_Map.h"
#include "Epetra_RowMatrix.h"

#include "Ifpack_SparseContainer.h"
#include "Ifpack_DenseContainer.h"
#include "Ifpack_Amesos.h"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Ifpack_Partitioner.h"
#include "Ifpack_GreedyPartitioner.h"
#include "Ifpack_LinearPartitioner.h"
#include "Ifpack_METISPartitioner.h"
#include "Ifpack_Graph.h"
#include "Ifpack_Graph_Epetra_RowMatrix.h"

#include "Teuchos_StandardCatchMacros.hpp"
#include <omp.h>

namespace HYMLS {

  // constructor
  SolverContainer::SolverContainer(Type SolverType, const int NumRows, const int NumVectors)
  :
    SolverType_(SolverType)
    {
    switch (SolverType_)
      {
      case DENSE:
        Container_ = Teuchos::rcp(new Ifpack_DenseContainer(NumRows, NumVectors));
        break;
      case SPARSE:
        Container_ = Teuchos::rcp(new Ifpack_SparseContainer<SparseDirectSolver>(NumRows, NumVectors));
        break;
      default:
        Tools::Error("Solver type not supported", __FILE__, __LINE__);
        break;
      }
    }


  // destructor
  SolverContainer::~SolverContainer()
    {
    }
//~ 
  //~ //! Returns a pointer to the internally stored map.
  //~ Teuchos::RCP<const Epetra_Map> SolverContainer::Map() const;

  //! Returns a pointer to the internally stored solution multi-vector.
  Teuchos::RCP<const Epetra_MultiVector> SolverContainer::LHS() const
    {
    switch (SolverType_)
      {
      case DENSE:
        {
        Teuchos::RCP<Ifpack_DenseContainer> container = DenseContainer();
        Teuchos::RCP<Epetra_Comm> comm;
#ifdef HAVE_MPI
        comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_SELF));
#else
        comm = Teuchos::rcp(new Epetra_SerialComm);
#endif
        const Epetra_SerialDenseMatrix &denseLhs = container->LHS();
        Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(NumRows(), 0, *comm));
        return Teuchos::rcp(new Epetra_MultiVector(Copy, *map, denseLhs.A(), denseLhs.LDA(), denseLhs.N()));
        }
      default: // SPARSE
        return SparseContainer()->LHS();
      }
    return Teuchos::null;
    }

  //! Returns a pointer to the internally stored rhs multi-vector.
  Teuchos::RCP<const Epetra_MultiVector> SolverContainer::RHS() const
    {
    switch (SolverType_)
      {
      case DENSE:
        {
        Teuchos::RCP<Ifpack_DenseContainer> container = DenseContainer();
        Teuchos::RCP<Epetra_Comm> comm;
#ifdef HAVE_MPI
        comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_SELF));
#else
        comm = Teuchos::rcp(new Epetra_SerialComm);
#endif
        const Epetra_SerialDenseMatrix &denseRhs = container->RHS();
        Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(NumRows(), 0, *comm));
        return Teuchos::rcp(new Epetra_MultiVector(Copy, *map, denseRhs.A(), denseRhs.LDA(), denseRhs.N()));
        }
      default: // SPARSE
        return SparseContainer()->RHS();
      }
    return Teuchos::null;
    }
  //~ //! Returns a pointer to the internally stored matrix.
  //~ Teuchos::RCP<const Epetra_CrsMatrix> SolverContainer::Matrix() const;

  //! Returns a pointer to the internally stored ID's.
  const Epetra_IntSerialDenseVector& SolverContainer::ID() const
    {
    switch (SolverType_)
      {
      case DENSE:
        return DenseContainer()->ID();
      default: // SPARSE
        return SparseContainer()->ID();
      }
    }

  /*! 
   * Reinitializes the container before computation starts. This
   * should skip some work and makes sure the ID list is still filled
   */
  int SolverContainer::InitializeCompute()
    {
    switch (SolverType_)
      {
      case DENSE:
        {
        // Temporarily save the ID list and put it back. This doesn't change
        const int rows = NumRows();
        int ID_list[rows];
        for (int i = 0; i < rows; ++i)
          {
          ID_list[i] = ID(i);
          }
        Container_->Initialize();
        for (int i = 0; i < rows; ++i)
          {
          ID(i) = ID_list[i];
          }
        }
        break;
      default: // SPARSE
        Container_->Initialize();
        break;
      }
    return 0;
    }

  //! return number of nonzeros in original matrix and the factorization
  int SolverContainer::NumGlobalNonzeros() const
    {
    switch (SolverType_)
      {
      case DENSE:
        return NumRows() * NumRows() * 2;
      default: // SPARSE
        {
        int nnz = 0;
        Teuchos::RCP<Ifpack_SparseContainer<SparseDirectSolver> > container = SparseContainer();
        nnz += container->Inverse()->NumGlobalNonzerosA();
        nnz += container->Inverse()->NumGlobalNonzerosLU();
        return nnz;
        }
      }
    }

  //! return the SparseContainer if the solver is a sparse one, otherwise null
  Teuchos::RCP<Ifpack_DenseContainer> SolverContainer::DenseContainer() const
  {
    if (SolverType_ != DENSE)
      return Teuchos::null;

    return Teuchos::rcp_dynamic_cast<Ifpack_DenseContainer>(Container_);
  }

  //! return the DenseContainer if the solver is a dense one, otherwise null
  Teuchos::RCP<Ifpack_SparseContainer<SparseDirectSolver> > SolverContainer::SparseContainer() const
  {
    if (SolverType_ != SPARSE)
      return Teuchos::null;

    return Teuchos::rcp_dynamic_cast<Ifpack_SparseContainer<SparseDirectSolver> >(Container_);
  }

}//namespace
