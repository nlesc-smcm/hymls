#ifndef HYMLS_MATRIXBLOCK_H
#define HYMLS_MATRIXBLOCK_H

#include "HYMLS_config.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "HYMLS_HierarchicalMap.H"

namespace Teuchos {
class ParameterList;
  }

class Epetra_MultiVector;
class Epetra_Import;
class Epetra_BlockMap;
class Epetra_CrsMatrix;

class Ifpack_Container;

namespace HYMLS
  {

class OverlappingPartitioner;


//! This class implements the blocks that are used in a Schur complement.
//! It can compute both the actual blocks and a factorization based on
//! subdomains for the A11 block.
class MatrixBlock
  {
public:
  //! Constructor
  //! It requires the matrix from which the block should be extracted,
  //! the overlapping partitioner and a row and column strategy from
  //! which it can find out which block has to be extracted. For instance
  //! The A12 block has Interior rows and Separator columns.
  //! The myLevel_ argument is there just for timing and debugging
  //! purposes.
  MatrixBlock(Teuchos::RCP<const Epetra_CrsMatrix> matrix,
    Teuchos::RCP<const Epetra_CrsMatrix> extendedMatrix,
    Teuchos::RCP<const OverlappingPartitioner> hid,
    HierarchicalMap::SpawnStrategy rowStrategy,
    HierarchicalMap::SpawnStrategy colStrategy,
    int level);

  virtual ~MatrixBlock() {}

  //! Compute the actual block
  int Compute();

  //! Compute the blocks belonging to every subdomain
  int ComputeSubdomainBlocks();

  //! Initialize the subdomain solvers for the A11 block
  int InitializeSubdomainSolvers(std::string const &solverType,
  Teuchos::RCP<Teuchos::ParameterList>, int numThreads);

  //! Compute the subdomain solvers for the A11 block
  int ComputeSubdomainSolvers();

  //! Recompute the blocks and subdomain solver initialization
  //! for a new matrix (with the same maps)
  int Recompute(Teuchos::RCP<const Epetra_CrsMatrix> matrix,
    Teuchos::RCP<const Epetra_CrsMatrix> extendedMatrix);

  //! Apply a block
  int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y);

  //! Apply the inverse of a block (A11)
  int ApplyInverse(const Epetra_MultiVector& B, Epetra_MultiVector& X);

  //! Set whether we want to use transpose Apply and ApplyInverse
  int SetUseTranspose(bool useTranspose);

  //! Get the original matrix
  Teuchos::RCP<const Epetra_CrsMatrix> Matrix() const {return matrix_;}

  //! Get the extended matrix
  Teuchos::RCP<const Epetra_CrsMatrix> ExtendedMatrix() const {return extendedMatrix_;}

  //! Get the matrix block
  Teuchos::RCP<const Epetra_CrsMatrix> Block() const;

  //! Get the sd-th subdomain block
  Teuchos::RCP<const Epetra_CrsMatrix> SubBlock(int sd) const;

  //! Get the sd-th subdomain solver
  Teuchos::RCP<Ifpack_Container> SubdomainSolver(int sd) const;

  //! Get the row map of the block
  const Epetra_Map &RowMap() const {return *rangeMap_;};

  //! Get the column map of the block
  const Epetra_Map &ColMap() const {return *colMap_;};

  //! Get the range map of the block
  const Epetra_Map &RangeMap() const {return *rangeMap_;};

  //! Get the domain map of the block
  const Epetra_Map &DomainMap() const {return *domainMap_;};

  // Get the row map importer
  const Epetra_Import &Importer() const {return *import_;}

  //! Get the overlapping partitioner
  const OverlappingPartitioner &Partitioner() const {return *hid_; }

  //! Get the amount of flops in Initialization
  double InitializeFlops() const;

  //! Get the amount of flops from the compute methods
  double ComputeFlops() const;

  //! Get the amount of flops from the apply method
  double ApplyFlops() const;

  //! Get the amount of flops from the ApplyInverse method
  double ApplyInverseFlops() const;

protected:
  //! Original matrix (not the block)
  Teuchos::RCP<const Epetra_CrsMatrix> matrix_;

  //! Matrix extended with off-processor separator nodes
  Teuchos::RCP<const Epetra_CrsMatrix> extendedMatrix_;

  //! Overlapping partitioner on which the blocks are based
  Teuchos::RCP<const OverlappingPartitioner> hid_;

  //! Determines whether we are the 1st or 2nd block rowwise
  HierarchicalMap::SpawnStrategy rowStrategy_;

  //! Determines whether we are the 1st or 2nd block columnwise
  HierarchicalMap::SpawnStrategy colStrategy_;

  //! Internal string used for debugging
  std::string label_;

  //! Row map
  Teuchos::RCP<const Epetra_Map> rowMap_;

  //! Range map
  Teuchos::RCP<const Epetra_Map> rangeMap_;

  //! Domain map
  Teuchos::RCP<const Epetra_Map> domainMap_;

  //! Column map
  Teuchos::RCP<const Epetra_Map> colMap_;

  //! Importer for the block
  Teuchos::RCP<Epetra_Import> import_;

  //! The actual block
  Teuchos::RCP<Epetra_CrsMatrix> block_;

  //! Ifpack conainers for solving the subdomain problems
  Teuchos::Array<Teuchos::RCP<Ifpack_Container> > subdomainSolvers_;

  //! Subdomain blocks for this block
  Teuchos::Array<Teuchos::RCP<Epetra_CrsMatrix> > subBlocks_;

  //! Bool to set whether we want to perform transpose operations or not
  bool useTranspose_;

  //! The amount of flops in Initialization
  double initializeFlops_;

  //! The amount of flops from the compute methods
  double computeFlops_;

  //! The amount of flops from the apply method
  double applyFlops_;

  //! The amount of flops from the ApplyInverse method
  double applyInverseFlops_;

  //! Amount of threads used by the subdomain solvers
  int numThreads_;

  //! Level only used for debugging and timing
  int myLevel_;
  };
  }

#endif
