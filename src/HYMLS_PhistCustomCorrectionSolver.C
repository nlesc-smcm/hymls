#include "phist_config.h"
#include <mpi.h>

// Include this before other phist headers
#include "HYMLS_PhistCustomCorrectionSolver.H"

#include "HYMLS_Solver.H"
#include "HYMLS_Macros.H"
#include "HYMLS_MatrixUtils.H"
#include "HYMLS_Macros.H"
#include "HYMLS_Tester.H"
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
   
  Teuchos::RCP<const Epetra_MultiVector> BQ;
  const Epetra_MultiVector *Q_ptr = (const Epetra_MultiVector *)Qtil;
  const Epetra_MultiVector *BQ_ptr = (const Epetra_MultiVector *)BQtil;
  const Epetra_MultiVector *r_ptr = (const Epetra_MultiVector *)res;
  Epetra_MultiVector *t_ptr = (Epetra_MultiVector *)t;

  if (BQtil == NULL)
    {
    if (B_op == NULL)
      {
      BQ_ptr = Q_ptr;
      }
    else
      {
      BQ = Teuchos::rcp(new Epetra_MultiVector(*Q_ptr));
      BQ_ptr = BQ.get();
      PHIST_CHK_IERR(B_op->apply(1.0, B_op->A, Qtil, 0.0,
          (TYPE(mvec_ptr))BQ_ptr, iflag), *iflag);
      }
    }

  const Epetra_BlockMap &map = Q_ptr->Map();
  const Epetra_BlockMap &map0 = solver->OperatorRangeMap();
  if (!(map.SameAs(map0)))
  {
    // System is only the v-part, not the full system, so import the vectors
    // on the v-part to the full system so we can use HYMLS on it.
    Epetra_Import import0(map0, Q_ptr->Map());

    Epetra_MultiVector vec0(map0, Q_ptr->NumVectors());
    vec0.PutScalar(0.0);
    CHECK_ZERO(vec0.Import(*Q_ptr, import0, Insert));

    Epetra_MultiVector vec1(map0, BQ_ptr->NumVectors());
    vec1.PutScalar(0.0);
    CHECK_ZERO(vec1.Import(*Q_ptr, import0, Insert));
    if (me->doBordering_)
    {
      solver->setBorder(Teuchos::rcp<const Epetra_MultiVector>(&vec1, false),
        Teuchos::rcp<const Epetra_MultiVector>(&vec0, false));
      solver->SetupDeflation();
    }
    else
    {
      CHECK_ZERO(solver->setProjectionVectors(Teuchos::rcp<const Epetra_MultiVector>(&vec0, false)));
    }
  }
  else
  {
    if (me->doBordering_)
    {
      solver->setBorder(Teuchos::rcp<const Epetra_MultiVector>(BQ_ptr, false),
        Teuchos::rcp<const Epetra_MultiVector>(Q_ptr, false));
      solver->SetupDeflation();
    }
    else
    {
      CHECK_ZERO(solver->setProjectionVectors(Teuchos::rcp<const Epetra_MultiVector>(Q_ptr, false)));
    }
  }

  if (!(map.SameAs(map0)))
  {
    // v-part only, so import back an forth between the full system like above
    const Epetra_Map &map1 = solver->OperatorRangeMap();
    Epetra_Import import1(map1, (r_ptr)->Map());
    Epetra_MultiVector vec1(map1, (r_ptr)->NumVectors());
    vec1.PutScalar(0.0);
    CHECK_ZERO(vec1.Import(*r_ptr, import1, Insert));

    const Epetra_Map &map2 = solver->OperatorDomainMap();
    Epetra_Import import2(map2, t_ptr->Map());
    Epetra_MultiVector vec2(map2, t_ptr->NumVectors());
    vec2.PutScalar(0.0);
    CHECK_ZERO(vec2.Import(*t_ptr, import2, Insert));

    solver->ApplyInverse(vec1, vec2);

    Epetra_Import invImport2(t_ptr->Map(), map2);
    CHECK_ZERO(t_ptr->Import(vec2, invImport2, Insert));
  }
  else
  {
    // This is allowed to not converge, so don't do CHECK_ZERO
    solver->ApplyInverse(*r_ptr, *t_ptr);
  }

  HYMLS_TEST("jada",isDivFree(*(const Epetra_CrsMatrix *)A_op->A, *t_ptr), __FILE__, __LINE__);

  // normalize result vectors
  _MT_ tmp;
  PHIST_CHK_IERR(phist_Dmvec_normalize(t, &tmp, iflag), *iflag);
  
  // unset border (if any)
  if (me->doBordering_) solver->setBorder(Teuchos::null,Teuchos::null);
  else             solver->setProjectionVectors(Teuchos::null);
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
