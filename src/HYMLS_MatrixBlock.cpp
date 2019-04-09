#include "HYMLS_MatrixBlock.hpp"

#include "HYMLS_config.h"

#include "Teuchos_ParameterList.hpp"

#include "Epetra_Import.h"
#include "Epetra_BlockMap.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_MultiVector.h"

#include "HYMLS_Macros.hpp"
#include "HYMLS_MatrixUtils.hpp"
#include "HYMLS_OverlappingPartitioner.hpp"
#include "HYMLS_HierarchicalMap.hpp"
#include "HYMLS_SparseDirectSolver.hpp"

#include "Ifpack_DenseContainer.h"
#include "Ifpack_Amesos.h"

#undef HAVE_MPI
#include "Ifpack_SparseContainer.h"

#ifdef HYMLS_USE_MKL
#include <mkl.h>
#endif

namespace HYMLS {

MatrixBlock::MatrixBlock(
  Teuchos::RCP<const OverlappingPartitioner> hid,
  HierarchicalMap::SpawnStrategy rowStrategy,
  HierarchicalMap::SpawnStrategy colStrategy,
  int level)
  :
  hid_(hid),
  rowStrategy_(rowStrategy),
  colStrategy_(colStrategy),
  label_("MatrixBlock"),
  useTranspose_(false),
  myLevel_(level)
  {
  // First we get the maps belonging to the rows and columns of this
  // block. This will not cause any duplicate work because they are
  // cached in the hid.
  Teuchos::RCP<const HierarchicalMap> rowObject = hid_->Spawn(rowStrategy);
  rowMap_ = rowObject->GetMap();
  rangeMap_ = rowObject->GetMap();

  Teuchos::RCP<const HierarchicalMap> colObject = hid_->Spawn(colStrategy);
  domainMap_ = colObject->GetMap();

  /*
  int active_ranks=HYMLS::ProcTopo->getNumActive(myLevel_);
  if (active_ranks==comm_->NumProc())
    {
    Tools::out() << "ALL MPI RANKS ACTIVE"<<std::endl;
    }
  else
    {
    Tools::out() << "NUMBER OF ACTIVE RANKS: "<<active_ranks<<std::endl;
    }
  */
  /*
  int ranks_on_node=HYMLS::ProcTopo->numActiveProcsOnNode(myLevel_);
  int total_ranks_on_node=HYMLS::ProcTopo->numActiveProcsOnNode(0);
  // figure out how many threads we can use here
  numThreadsSD_=std::max(1,(int)(total_ranks_on_node/ranks_on_node));
  */

  }

int MatrixBlock::Compute(Teuchos::RCP<const Epetra_CrsMatrix> matrix,
  Teuchos::RCP<const Epetra_CrsMatrix> extendedMatrix)
  {
  HYMLS_LPROF(label_, "Compute");

  // This could be really expensive, but I want to try it anyway...
  if (colMap_ == Teuchos::null)
    colMap_ = MatrixUtils::CreateColMap(*extendedMatrix, *domainMap_, *domainMap_);
  if (import_ == Teuchos::null)
    import_ = Teuchos::rcp(new Epetra_Import(*rowMap_, matrix->RowMap()));

  if (block_ != Teuchos::null)
    {
    CHECK_ZERO(block_->PutScalar(0.0));
    CHECK_ZERO(block_->Import(*matrix, *import_, Insert));
    }
  else
    {
    int MaxNumEntriesPerRow = matrix->MaxNumEntries();
    block_ = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *rowMap_,
        *colMap_, MaxNumEntriesPerRow));

    CHECK_ZERO(block_->Import(*matrix, *import_, Insert));
    CHECK_ZERO(block_->FillComplete(*domainMap_, *rangeMap_));
    }

  if (subBlocks_.size())
    {
    for (int sd = 0; sd < hid_->NumMySubdomains(); sd++)
      {
      CHECK_ZERO(subBlocks_[sd]->PutScalar(0.0));
      CHECK_ZERO(MatrixUtils::ExtractLocalBlock(*extendedMatrix, *subBlocks_[sd]));
      }
    }
  else
    {
    double nzCopy = 0;
    int num_sd = hid_->NumMySubdomains();
    subBlocks_.resize(num_sd);

    for (int sd = 0; sd < num_sd; sd++)
      {
      Teuchos::RCP<const Epetra_Map> subRangeMap = hid_->SpawnMap(sd, rowStrategy_);
      HYMLS_DEBVAR(*subRangeMap);
      Teuchos::RCP<const Epetra_Map> subDomainMap = hid_->SpawnMap(sd, colStrategy_);
      HYMLS_DEBVAR(*subDomainMap);

      int MaxNumEntriesPerRow = extendedMatrix->MaxNumEntries();
      subBlocks_[sd] = Teuchos::rcp(new
        Epetra_CrsMatrix(Copy, *subRangeMap, *subDomainMap, MaxNumEntriesPerRow));

      CHECK_ZERO(MatrixUtils::ExtractLocalBlock(*extendedMatrix, *subBlocks_[sd]));

      CHECK_ZERO(subBlocks_[sd]->FillComplete(*subDomainMap,*subRangeMap));

      nzCopy += (double)(subBlocks_[sd]->NumMyNonzeros());
      }
    }

  return 0;
  }

int MatrixBlock::InitializeSubdomainSolvers(std::string const &solverType,
  Teuchos::RCP<Teuchos::ParameterList> sd_list, int numThreads)
  {
  HYMLS_LPROF2(label_, "InitializeSubdomainSolvers");

  HYMLS_DEBUG("initialize subdomain solvers...");

  numThreads_ = numThreads;

  subdomainSolvers_.resize(hid_->NumMySubdomains());

  for (int sd = 0; sd < hid_->NumMySubdomains(); sd++)
    {
    const int nrows = hid_->NumInteriorElements(sd);

    if (solverType == "Dense")
      {
      subdomainSolvers_[sd] =
        Teuchos::rcp(new Ifpack_DenseContainer(nrows));
      }
    else if (solverType == "Sparse")
      {
      subdomainSolvers_[sd] =
        Teuchos::rcp(new Ifpack_SparseContainer<SparseDirectSolver>(nrows));
      }
    else
      {
      Tools::Error("invalid 'Subdomain Solver Type' in 'Solver' sublist",
          __FILE__, __LINE__);
      }

      // copy parameter list
      Teuchos::ParameterList tmp_sd_list = *sd_list;

#if HYMLS_TIMING_LEVEL>2
      tmp_sd_list.set("Label", "direct solver (lev "+Teuchos::toString(myLevel_)+", sd "+Teuchos::toString(sd)+")");
#else
      tmp_sd_list.set("Label", "direct solver (lev "+Teuchos::toString(myLevel_)+")");
#endif
      IFPACK_CHK_ERR(subdomainSolvers_[sd]->SetParameters(tmp_sd_list));

#ifdef HYMLS_TESTING
    bool status = true;
    try {
#endif
      subdomainSolvers_[sd]->Initialize();
#ifdef HYMLS_TESTING
      } TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, status);
    if (!status)
      {
      Tools::Fatal("Caught an exception in subdomain solver init of sd="+
        Teuchos::toString(sd)+" on partition "+Teuchos::toString(Comm().MyPID()),
          __FILE__, __LINE__);
      }
#endif
    Epetra_Map const &rowMap = hid_->OverlappingMap();
    // set "global" ID of each partitioner row
    for (int j = 0 ; j < nrows ; j++)
      {
      const int LRID = rowMap.LID(hid_->GID(sd, 0, j));
      subdomainSolvers_[sd]->ID(j) = LRID;
      }
    }

  return 0;
  }

int MatrixBlock::ComputeSubdomainSolvers(Teuchos::RCP<const Epetra_CrsMatrix> extendedMatrix)
  {
  HYMLS_LPROF(label_, "ComputeSubdomainSolvers");

  HYMLS_DEBUG("compute subdomain solvers...");

  int nnz = 0;
  for (int sd = 0; sd < hid_->NumMySubdomains(); sd++)
    {
    if (subdomainSolvers_[sd]->NumRows() > 0)
      {
      // compute subdomain factorization
#ifdef HYMLS_TESTING
      bool status = true;
      try {
#endif
        Epetra_Map const &rowMap = extendedMatrix->RowMap();
        CHECK_ZERO(subdomainSolvers_[sd]->Initialize());
        if (Teuchos::rcp_dynamic_cast<Ifpack_DenseContainer>(
            subdomainSolvers_[sd]) != Teuchos::null)
          {
          // Initialize destroys the indices for the Ifpack_DenseContainer :(
          for (int j = 0; j < subdomainSolvers_[sd]->NumRows(); j++)
            {
            const int LRID = rowMap.LID(hid_->GID(sd, 0, j));
            subdomainSolvers_[sd]->ID(j) = LRID;
            }
          }
        
        CHECK_ZERO(subdomainSolvers_[sd]->Compute(*extendedMatrix));
        
#ifdef HYMLS_TESTING
        } TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, status);
      if (!status)
        {
        Tools::Fatal("caught an exception in subdomain factorization of sd="+
          Teuchos::toString(sd)+" on partition "+Teuchos::toString(Comm().MyPID()),
          __FILE__, __LINE__);
        }
#endif

      // Now some debugging code until the end of the function
#ifndef NO_MEMORY_TRACING
#ifndef USE_AMESOS
      Teuchos::RCP<Ifpack_SparseContainer<SparseDirectSolver> > container =
        Teuchos::rcp_dynamic_cast<Ifpack_SparseContainer<SparseDirectSolver> >(
          subdomainSolvers_[sd]);
      if (container != Teuchos::null)
        {
        nnz += container->Inverse()->NumGlobalNonzerosA();
        nnz += container->Inverse()->NumGlobalNonzerosLU();
        }
      else
        {
        int nr = subdomainSolvers_[sd]->NumRows();
        nnz += nr * nr;
        }
#endif
#endif

#ifdef STORE_SUBDOMAIN_MATRICES
      Teuchos::RCP<Ifpack_SparseContainer<SparseDirectSolver> > container =
        Teuchos::rcp_dynamic_cast<Ifpack_SparseContainer<SparseDirectSolver> >(
          subdomainSolvers_[sd]);
      if (container != Teuchos::null)
        {
        Tools::Warning("STORE_SUBDOMAIN_MATRICES is defined, this produces lots of output"
          " and makes the code VERY slow", __FILE__, __LINE__);
        const Epetra_RowMatrix& Asd = container->Inverse()->Matrix();
        std::string filename = "SubdomainMatrix_P"+Teuchos::toString(Comm().MyPID())+
          "_L"+Teuchos::toString(myLevel_)+
          "_SD"+Teuchos::toString(sd)+".txt";
        std::ofstream ofs(filename.c_str());
        MatrixUtils::PrintRowMatrix(Asd,ofs);
        ofs.close();
        }
#endif
      }
    }

#ifdef STORE_SD_LU
  if (hid_->NumMySubdomains() > 0)
    {
    if (subdomainSolvers_[0]->NumRows() > 0)
      {
      Teuchos::RCP<Ifpack_SparseContainer<SparseDirectSolver> > container =
        Teuchos::rcp_dynamic_cast<Ifpack_SparseContainer<SparseDirectSolver> >(
          subdomainSolvers_[0]);
      std::string label = "sdlu_L" + Teuchos::toString(myLevel_) +
        "_0_p" + Teuchos::toString(Comm().MyPID());
      container->Inverse()->DumpSolverStatus(label, false, Teuchos::null, Teuchos::null);
      }
    }
#endif

  return 0;
  }

int MatrixBlock::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y)
  {
  HYMLS_LPROF3(label_, "Apply");
  if (block_ == Teuchos::null)
    {
    Tools::Warning("Matrix block is not computed!", __FILE__, __LINE__);
    return -1;
    }

  CHECK_ZERO(block_->Apply(X, Y));

  applyFlops_ += 2 * block_->NumGlobalNonzeros64();

  return 0;
  }


int MatrixBlock::ApplyInverse(const Epetra_MultiVector& B, Epetra_MultiVector& X)
  {
  if (!subdomainSolvers_.size() && hid_->NumMySubdomains() > 0)
    {
    Tools::Warning("Subdomain Solvers have not been computed!", __FILE__, __LINE__);
    return -1;
    }

  HYMLS_LPROF3(label_, "ApplyInverse");

  // Force threading for the subdomain solvers when possible
  if (numThreads_ > 0)
    {
    //TODO - get #threads dynamically from processor topology
    //      (see ProcTopo sketch above)
#ifdef HYMLS_USE_MKL
    mkl_set_num_threads(numThreads_);
#endif
#ifdef HYMLS_USE_OPENMP
    omp_set_num_threads(numThreads_);
#endif
    }

  // assume that all block solvers have the same number of vectors...
  if (subdomainSolvers_.size() > 0)
    {
    if (subdomainSolvers_[0]->NumVectors() != X.NumVectors())
      {
      for (int sd = 0; sd < subdomainSolvers_.size() ; sd++)
        {
        CHECK_ZERO(subdomainSolvers_[sd]->SetNumVectors(X.NumVectors()));
        }
      }
    }
  // step 1: solve subdomain problems for temporary vector y
  for (int sd = 0 ; sd < subdomainSolvers_.size() ; sd++)
    {
    const int rows = subdomainSolvers_[sd]->NumRows();

    // copy IDs to be able to walk through the vectors columnwise
    int *IDlist = new int[rows];
    for (int j = 0 ; j < rows ; j++)
      IDlist[j] = B.Map().LID(hid_->OverlappingMap().GID64(subdomainSolvers_[sd]->ID(j)));

    // extract RHS from X
    for (int k = 0 ; k < B.NumVectors() ; k++)
      {
      const double *Bvec = B[k];
      for (int j = 0 ; j < rows ; j++)
        {
        subdomainSolvers_[sd]->RHS(j,k) = Bvec[IDlist[j]];
        }
      }

    // apply the inverse of each block. NOTE: flops occurred
    // in ApplyInverse() of each block are summed up in method
    // ApplyInverseFlops().
    if (subdomainSolvers_[sd]->NumRows()>0)
      {
      IFPACK_CHK_ERR(subdomainSolvers_[sd]->ApplyInverse());
      }
    // copy back into solution vector Y
    for (int k = 0 ; k < X.NumVectors() ; k++)
      {
      double *Xvec = X[k];
      for (int j = 0 ; j < rows ; j++)
        {
        Xvec[IDlist[j]] = subdomainSolvers_[sd]->LHS(j,k);
        }
      }
    delete[] IDlist;
    }

  return 0;
  }

int MatrixBlock::SetUseTranspose(bool useTranspose)
  {
  useTranspose_ = useTranspose;

  if (block_ != Teuchos::null)
    {
    block_->SetUseTranspose(useTranspose);
    }

  // Set transpose for the subdomain solvers
  Teuchos::RCP<const Ifpack_SparseContainer<SparseDirectSolver> > sparseLU = Teuchos::null;

  for (int sd = 0; sd < subdomainSolvers_.size(); sd++)
    {
    sparseLU = Teuchos::rcp_dynamic_cast
      <const Ifpack_SparseContainer<SparseDirectSolver> >(subdomainSolvers_[sd]);
    if (sparseLU != Teuchos::null)
      {
      CHECK_ZERO(Teuchos::rcp_const_cast<SparseDirectSolver>(
          sparseLU->Inverse())->SetUseTranspose(useTranspose));
      }
    else
      {
      Tools::Error("Transpose not implemented for dense subdomain solver!",
        __FILE__, __LINE__);
      }
    sparseLU = Teuchos::null;
    }

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

Teuchos::RCP<Ifpack_Container> MatrixBlock::SubdomainSolver(int sd) const
  {
  if (subdomainSolvers_.size() < sd)
    {
      Tools::Warning("Solver for subdomain "+Teuchos::toString(sd)+
        " has not been computed!", __FILE__, __LINE__);
      return Teuchos::null;
    }
  return subdomainSolvers_[sd];
  }

double MatrixBlock::InitializeFlops() const
  {
  double total = initializeFlops_;
  for (int i = 0 ; i < subdomainSolvers_.size(); i++)
    {
    if (subdomainSolvers_[i] != Teuchos::null)
      {
      total += subdomainSolvers_[i]->InitializeFlops();
      }
    }

  return total;
  }

double MatrixBlock::ComputeFlops() const
  {
  double total = computeFlops_;
  for (int i = 0 ; i < subdomainSolvers_.size(); i++)
    {
    if (subdomainSolvers_[i] != Teuchos::null)
      {
      total += subdomainSolvers_[i]->ComputeFlops();
      }
    }

  return total;
  }

double MatrixBlock::ApplyInverseFlops() const
  {
  double total = applyInverseFlops_;
  for (int i = 0 ; i < subdomainSolvers_.size(); i++)
    {
    if (subdomainSolvers_[i] != Teuchos::null)
      {
      total += subdomainSolvers_[i]->ApplyInverseFlops();
      }
    }

  return total;
  }

double MatrixBlock::ApplyFlops() const
  {
  return applyFlops_;
  }

Epetra_Comm const &MatrixBlock::Comm() const
  {
  return hid_->Comm();
  }

  }
