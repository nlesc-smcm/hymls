#include "HYMLS_MatrixBlock.H"

#include "Epetra_Import.h"
#include "Epetra_BlockMap.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_MultiVector.h"

#include "HYMLS_MatrixUtils.H"
#include "HYMLS_View_MultiVector.H"
#include "HYMLS_OverlappingPartitioner.H"
#include "HYMLS_HierarchicalMap.H"

namespace HYMLS {

MatrixBlock::MatrixBlock(Teuchos::RCP<const Epetra_CrsMatrix> matrix,
  Teuchos::RCP<const OverlappingPartitioner> hid,
  HierarchicalMap::SpawnStrategy rowStrategy,
  HierarchicalMap::SpawnStrategy colStrategy
  )
  :
  matrix_(matrix),
  hid_(hid),
  rowStrategy_(rowStrategy),
  colStrategy_(colStrategy),
  label_("MatrixBlock"),
  useTranspose_(false)
  {
  // First we get the maps belonging to the rows and columns of this
  // block. This will not cause any duplicate work because they are
  // cached in the hid.
  Teuchos::RCP<const HierarchicalMap> rowObject = hid_->Spawn(rowStrategy);
  rangeMap_ = rowObject->GetMap();

  Teuchos::RCP<const HierarchicalMap> colObject = hid_->Spawn(colStrategy);
  domainMap_ = colObject->GetMap();

  // This could be really expensive, but I want to try it anyway...
  colMap_ = MatrixUtils::CreateColMap(*matrix_, *domainMap_, *domainMap_);
  import_ = Teuchos::rcp(new Epetra_Import(*rangeMap_, matrix_->RowMap()));

  int MaxNumEntriesPerRow = matrix_->MaxNumEntries();
  block_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *rangeMap_,
      *colMap_, MaxNumEntriesPerRow));

  CHECK_ZERO(block_->Import(*matrix_, *import_, Insert));
  CHECK_ZERO(block_->FillComplete(*domainMap_, *rangeMap_));

  REPORT_SUM_MEM(label_, "copies of matrix parts", block_->NumMyNonzeros(),
    block_->NumMyNonzeros(), &matrix_->Comm());
  }

int MatrixBlock::ComputeSubdomainBlocks()
  {
  double nzCopy = 0;
  int num_sd = hid_->NumMySubdomains();
  subBlocks_.resize(num_sd);

// #pragma omp parallel for schedule(static)
  for (int sd = 0; sd < num_sd; sd++)
    {
    // subRangeMap_[sd] = hid_->SpawnMap(sd, rowStrategy);
    // DEBVAR(*subRangeMap_[sd]);
    // subDomainMap_[sd] = hid_->SpawnMap(sd, colStrategy);
    // DEBVAR(*subDomainMap_[sd]);

    Teuchos::RCP<const Epetra_Map> subRangeMap = hid_->SpawnMap(sd, rowStrategy_);
    DEBVAR(*subRangeMap);
    Teuchos::RCP<const Epetra_Map> subDomainMap = hid_->SpawnMap(sd, colStrategy_);
    DEBVAR(*subDomainMap);

    int MaxNumEntriesPerRow = matrix_->MaxNumEntries();
    subBlocks_[sd] = Teuchos::rcp(new
      Epetra_CrsMatrix(Copy, *subRangeMap, *subDomainMap, MaxNumEntriesPerRow));

    CHECK_ZERO(MatrixUtils::ExtractLocalBlock(*matrix_, *subBlocks_[sd]));

    CHECK_ZERO(subBlocks_[sd]->FillComplete(*subDomainMap,*subRangeMap));

    nzCopy += (double)(subBlocks_[sd]->NumMyNonzeros());
    }
  
  REPORT_SUM_MEM(label_, "copies of matrix parts", nzCopy, nzCopy, &matrix_->Comm());

  return 0;
  }

int MatrixBlock::Recompute(Teuchos::RCP<const Epetra_CrsMatrix> matrix)
  {
  matrix_ = matrix;

  if (subBlocks_.size())
    {
#pragma omp parallel for schedule(static)
    for (int sd = 0; sd < hid_->NumMySubdomains(); sd++)
      {
      CHECK_ZERO(subBlocks_[sd]->PutScalar(0.0));
      CHECK_ZERO(MatrixUtils::ExtractLocalBlock(*matrix_, *subBlocks_[sd]));
      }
    }

  if (block_ != Teuchos::null)
    {
    CHECK_ZERO(block_->PutScalar(0.0));
    CHECK_ZERO(block_->Import(*matrix_, *import_, Insert));
    }

  return 0;
  }

int MatrixBlock::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y)
  {
  // HYMLS_LPROF3(label_, "Apply"); // For this we need myLevel_ here...
  if (block_ == Teuchos::null)
    {
    Tools::Warning("Matrix block is not computed!", __FILE__, __LINE__);
    return -1;
    }

  if (useTranspose_)
    {
    HYMLS::MultiVector_View xView(X.Map(), *rangeMap_);
    HYMLS::MultiVector_View yView(Y.Map(), *domainMap_);
    CHECK_ZERO(block_->Apply(*xView(X),*yView(Y)));
    }
  else
    {
    HYMLS::MultiVector_View xView(X.Map(), *domainMap_);
    HYMLS::MultiVector_View yView(Y.Map(), *rangeMap_);
    CHECK_ZERO(block_->Apply(*xView(X),*yView(Y)));
    }

  return 0;
  }

int MatrixBlock::SetUseTranspose(bool useTranspose)
  {
  useTranspose_ = useTranspose;
  block_->SetUseTranspose(useTranspose);

  return 0;
  }

Teuchos::RCP<const Epetra_CrsMatrix> MatrixBlock::Block() const
  {
  if (block_ == Teuchos::null)
    {
    Tools::Warning("Matrix block is not computed!", __FILE__, __LINE__);
    }

  return block_;
  }

  
Teuchos::RCP<const Epetra_CrsMatrix> MatrixBlock::SubBlock(int sd) const
  {
  if (subBlocks_.size() < sd)
    {
      Tools::Warning("Matrix block for subdomain "+Teuchos::toString(sd)+
        " has not been computed!", __FILE__, __LINE__);
      return Teuchos::null;
    }
  return subBlocks_[sd];
  }
}
