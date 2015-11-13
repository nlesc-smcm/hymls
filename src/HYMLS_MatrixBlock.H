#ifndef HYMLS_MATRIXBLOCK_H
#define HYMLS_MATRIXBLOCK_H

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "HYMLS_HierarchicalMap.H"

class Epetra_MultiVector;
class Epetra_Import;
class Epetra_BlockMap;
class Epetra_CrsMatrix;

class Ifpack_Container;

namespace HYMLS
  {

class OverlappingPartitioner;

class MatrixBlock
  {
public:
  MatrixBlock(Teuchos::RCP<const Epetra_CrsMatrix> matrix,
    Teuchos::RCP<const OverlappingPartitioner> hid,
    HierarchicalMap::SpawnStrategy rowStrategy,
    HierarchicalMap::SpawnStrategy colStrategy,
    int level);

  int Compute();

  int ComputeSubdomainBlocks();

  int InitializeSubdomainSolvers(std::string const &solverType,
  Teuchos::RCP<Teuchos::ParameterList>, int numThreads);

  int ComputeSubdomainSolvers();

  int Recompute(Teuchos::RCP<const Epetra_CrsMatrix> matrix);

  int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y);

  int ApplyInverse(const Epetra_MultiVector& B, Epetra_MultiVector& X);

  int SetUseTranspose(bool useTranspose);

  Teuchos::RCP<const Epetra_CrsMatrix> Matrix() const {return matrix_;}

  Teuchos::RCP<const Epetra_CrsMatrix> Block() const;

  Teuchos::RCP<const Epetra_CrsMatrix> SubBlock(int sd) const;

  Teuchos::RCP<Ifpack_Container> SubdomainSolver(int sd) const;

  const Epetra_Map &Map() const {return *rangeMap_;};

  const OverlappingPartitioner &Partitioner() const {return *hid_; }

  double InitializeFlops() const;

  double ComputeFlops() const;

  double ApplyFlops() const;

  double ApplyInverseFlops() const;

protected:
  Teuchos::RCP<const Epetra_CrsMatrix> matrix_;

  Teuchos::RCP<const OverlappingPartitioner> hid_;

  HierarchicalMap::SpawnStrategy rowStrategy_;

  HierarchicalMap::SpawnStrategy colStrategy_;

  std::string label_;

  Teuchos::RCP<const Epetra_Map> rangeMap_;

  Teuchos::RCP<const Epetra_Map> domainMap_;

  Teuchos::RCP<const Epetra_Map> colMap_;

  Teuchos::RCP<Epetra_Import> import_;

  Teuchos::RCP<Epetra_CrsMatrix> block_;

  //! Ifpack conainers for solving the subdomain problems
  Teuchos::Array<Teuchos::RCP<Ifpack_Container> > subdomainSolvers_;

  Teuchos::Array<Teuchos::RCP<Epetra_CrsMatrix> > subBlocks_;

  bool useTranspose_;

  double initializeFlops_;

  double computeFlops_;

  double applyFlops_;

  double applyInverseFlops_;

  int numThreads_;

  int myLevel_;
  };
  }

#endif
