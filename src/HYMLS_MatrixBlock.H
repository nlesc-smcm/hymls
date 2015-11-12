#ifndef HYMLS_MATRIXBLOCK_H
#define HYMLS_MATRIXBLOCK_H

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "HYMLS_HierarchicalMap.H"

class Epetra_MultiVector;
class Epetra_Import;
class Epetra_BlockMap;
class Epetra_CrsMatrix;

namespace HYMLS {

class OverlappingPartitioner;

class MatrixBlock
  {
public:
  MatrixBlock(Teuchos::RCP<const Epetra_CrsMatrix> matrix,
    Teuchos::RCP<const OverlappingPartitioner> hid,
    HierarchicalMap::SpawnStrategy rowStrategy,
    HierarchicalMap::SpawnStrategy colStrategy);

  int ComputeSubdomainBlocks();

  int Recompute(Teuchos::RCP<const Epetra_CrsMatrix> matrix);

  int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y);

  int SetUseTranspose(bool useTranspose);

  Teuchos::RCP<const Epetra_CrsMatrix> Matrix() const {return matrix_;}

  Teuchos::RCP<const Epetra_CrsMatrix> Block() const;

  Teuchos::RCP<const Epetra_CrsMatrix> SubBlock(int sd) const;

  const Epetra_Map &Map() const {return *rangeMap_;};

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

  Teuchos::Array<Teuchos::RCP<Epetra_CrsMatrix> > subBlocks_;

  bool useTranspose_;
  };
  }

#endif
