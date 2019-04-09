#include "phist_config.h"
#include <mpi.h>

// Include this before other phist headers
#include "HYMLS_PhistCustomCorrectionSolver.hpp"

#include "HYMLS_Solver.hpp"
#include "HYMLS_Macros.hpp"
#include "HYMLS_MatrixUtils.hpp"
#include "HYMLS_Macros.hpp"
#include "HYMLS_Tester.hpp"
#include "Epetra_MultiVector.h"
#include "Epetra_Import.h"

#include "phist_kernels.h"
#include "phist_jadaCorrectionSolver.h"
#include "phist_macros.h"
#include "phist_enums.h"
#include "phist_orthog.h"

#include <cstdlib>
#include <vector>

#include "phist_gen_d.h"

namespace HYMLS {
namespace phist {

void jadaCorrectionSolver_run1(void* vme,
  void const* vA_op, void const* vB_op, 
  TYPE(const_mvec_ptr) Qtil, TYPE(const_mvec_ptr) BQtil,
  double sigma_r, double sigma_i,
  TYPE(const_mvec_ptr) res,
  double tol, int maxIter,
  TYPE(mvec_ptr) t,
  int robust,
  int *iflag)
{
  PHIST_ENTER_FCN(__FUNCTION__);
  *iflag = 0;
  TYPE(const_linearOp_ptr) A_op=(TYPE(const_linearOp_ptr))vA_op;
  TYPE(const_linearOp_ptr) B_op=(TYPE(const_linearOp_ptr))vB_op;
  SolverWrapper* me=(SolverWrapper*)vme;

  PHIST_CHK_IERR(*iflag = (maxIter <= 0) ? -1 : 0, *iflag);
  
  // note: we should probably solve a system with two rhs but with a real preconditioner here,
  // as is done in phist
  if (sigma_i!=0.0) 
  {
    HYMLS::Tools::Warning("jadaCorrectionSolver with complex shifts not implemented",
                          __FILE__, __LINE__);
    *iflag=-99; // not implemented
    return;
  }

  Teuchos::RCP<HYMLS::Solver> solver = me->solver_;
  if (solver==Teuchos::null){ *iflag=PHIST_BAD_CAST; return;}
  bool status=true;
  try {
  solver->SetTolerance(tol);
  solver->setShift(1.0, -sigma_r);
  } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr,status);
  if (!status)
  {
    *iflag=PHIST_CAUGHT_EXCEPTION; 
    return;
  }
   
  const Epetra_MultiVector *Q_ptr = (const Epetra_MultiVector *)Qtil;
  const Epetra_MultiVector *BQ_ptr = (const Epetra_MultiVector *)BQtil;
  const Epetra_MultiVector *r_ptr = (const Epetra_MultiVector *)res;
  Epetra_MultiVector *t_ptr = (Epetra_MultiVector *)t;

  Teuchos::RCP<const Epetra_MultiVector> Q=Teuchos::rcp(Q_ptr,false);
  Teuchos::RCP<const Epetra_MultiVector> BQ=Teuchos::rcp(BQ_ptr,false);;

  if (BQtil == NULL && B_op!=NULL)
    {
    // the caller provided B!=I but not B*Q, so we have to build it.
    BQ = Teuchos::rcp(new Epetra_MultiVector(*Q_ptr));
    BQ_ptr = BQ.get();
    PHIST_CHK_IERR(B_op->apply(1.0, B_op->A, Qtil, 0.0,
          (TYPE(mvec_ptr))BQ_ptr, iflag), *iflag);
    }
  if (me->doBordering_)
    {
      solver->setBorder(Q,BQ);
      solver->SetupDeflation();
    }
    else
    {
      CHECK_ZERO(solver->setProjectionVectors(Q,BQ));
    }

  // This is allowed to not converge, so don't do CHECK_ZERO
  solver->ApplyInverse(*r_ptr, *t_ptr);

  HYMLS_TEST("jada",isDivFree(*(const Epetra_CrsMatrix *)A_op->A, *t_ptr), __FILE__, __LINE__);

  // normalize result vectors
  _MT_ tmp;
  PHIST_CHK_IERR(phist_Dmvec_normalize(t, &tmp, iflag), *iflag);
  
  // unset border (if any). TODO: we should also remove projection vectors from the solver because
  // it may be used elsewhere, but setProjectionVectors can't be called with Teuchos::null right now.
  if (me->doBordering_) solver->setBorder(Teuchos::null,Teuchos::null);
}

void jadaCorrectionSolver_run(void* vme,
  void const* vA_op, void const* vB_op, 
  TYPE(const_mvec_ptr) Qtil, TYPE(const_mvec_ptr) BQtil,
  const double *sigma_r, const double *sigma_i, 
  TYPE(const_mvec_ptr) res, const int resIndex[],
  const double *tol, int maxIter,
  TYPE(mvec_ptr) t,
  int robust, int abortAfterFirstConvergedInBlock,
  int *iflag)
{
  PHIST_ENTER_FCN(__FUNCTION__);

  int numSys;
  PHIST_CHK_IERR(SUBR(mvec_num_vectors)(t,&numSys,iflag),*iflag);
  if (numSys > 2)
  {
    HYMLS::Tools::Warning("jadaCorrectionSolver with more than RHS not implemented",
                          __FILE__, __LINE__);
    *iflag = -99; // not implemented
    return;
  }

  if (numSys > 1)
  {
    if (sigma_i[0] != 0.0 && sigma_i[1] != 0.0) 
    {
      HYMLS::Tools::Warning("jadaCorrectionSolver with complex shifts not implemented",
                            __FILE__, __LINE__);
      *iflag = -99; // not implemented
      return;
    }

    if (std::abs(sigma_r[0] - sigma_r[1]) > 1e-14) 
    {
      HYMLS::Tools::Warning("jadaCorrectionSolver with unequal shifts not implemented",
                            __FILE__, __LINE__);
      *iflag = -99; // not implemented
      return;
    }
  }

  PHIST_CHK_IERR(jadaCorrectionSolver_run1(
                   vme, vA_op, vB_op, Qtil, BQtil,
                   *sigma_r, *sigma_i, res, *tol,
                   maxIter, t, robust, iflag), *iflag);
}

}//namespace phist
}//namespace HYMLS
