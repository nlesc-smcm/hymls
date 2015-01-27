#include "phist_config.h"
#include <mpi.h>

// Include this before other phist headers
#include "jadaCorrectionSolver_impl.H"

#include "HYMLS_Solver.H"

#include "phist_jadaCorrectionSolver.h"
#include "phist_macros.h"
#include "phist_enums.h"
#include "phist_orthog.h"

#include <cstdlib>
#include <vector>

#include "phist_gen_d.h"

//! create a jadaCorrectionSolver object
void SUBR(jadaCorrectionSolver_create)(TYPE(jadaCorrectionSolver_ptr) *me, int blockedGMRESBlockDim, const_map_ptr_t map, 
        linSolv_t method, int blockedGMRESMaxBase, bool useMINRES, int *iflag)
{
  PHIST_ENTER_FCN(__FUNCTION__);
  *iflag = 0;
  if (method==GMRES)
  {
    PHIST_CHK_IERR( *iflag = (blockedGMRESBlockDim <= 0) ? -1 : 0, *iflag);

    *me = new TYPE(jadaCorrectionSolver);
    (*me)->gmresBlockDim_ = blockedGMRESBlockDim;
    (*me)->blockedGMRESstates_  = NULL;
    (*me)->useMINRES_ = useMINRES;
  }
  else if (method==CARP_CG)
  {
    *iflag=-99;
  }
  else
  {
    PHIST_SOUT(PHIST_ERROR, "method %d (%s) not implemented",(int)method, linSolv2str(method));
    *iflag=-99;
  }
}

//! delete a jadaCorrectionSolver object
void SUBR(jadaCorrectionSolver_delete)(TYPE(jadaCorrectionSolver_ptr) me, int *iflag)
{
  PHIST_ENTER_FCN(__FUNCTION__);
  *iflag = 0;

  delete me;
}


//! calculate approximate solutions to given set of jacobi-davidson correction equations
//!
//! arguments:
//! jdCorrSolver    the jadaCorrectionSolver object
//! A_op            matrix A passed to jadaOp_create
//! B_op            matrix B passed to jadaOp_create
//! Qtil            projection vectors V passed to jadaOp_create
//! BQtil           projection vectors BV passed to jadaOp_create
//! sigma           (pos.!) shifts, -sigma[i], i in {1, ..., nvec} is passed to the jadaOp
//! res             JD residuals, e.g. rhs of the correction equations
//! resIndex        if not NULL, specifies permutation of the residual array to avoid unnecessary copying in the jada-algorithm
//! tol             desired accuracy (gmres residual tolerance) of the individual systems
//! maxIter         maximal number of iterations after which individial systems should be aborted
//! t               returns approximate solution vectors
//! iflag            a value > 0 indicates the number of systems that have not converged to the desired tolerance
void SUBR(jadaCorrectionSolver_run)(TYPE(jadaCorrectionSolver_ptr) me,
                                    TYPE(const_op_ptr)    A_op,     TYPE(const_op_ptr)    B_op, 
                                    TYPE(const_mvec_ptr)  Qtil,     TYPE(const_mvec_ptr)  BQtil,
                                    const _ST_            sigma[],  TYPE(const_mvec_ptr)  res,      const int resIndex[], 
                                    const _MT_            tol[],    int                   maxIter,
                                    TYPE(mvec_ptr)        t,
                                    bool useIMGS,                   bool abortAfterFirstConvergedInBlock,
                                    int *                 iflag)
{
  PHIST_ENTER_FCN(__FUNCTION__);
  *iflag = 0;

  PHIST_CHK_IERR(*iflag = (maxIter <= 0) ? -1 : 0, *iflag);

  Dmvec_ptr_t t_i = NULL;
  Dmvec_ptr_t r_i = NULL;

  // total number of systems to solve
  int totalNumSys;
  phist_Dmvec_num_vectors(t, &totalNumSys, iflag);
  TEUCHOS_TEST_FOR_EXCEPTION(*iflag != 0, std::runtime_error,
    "jadaCorrectionSolver_run: phist_Dmvec_num_vectors returned nonzero error code");

  // We can only solve one system at the moment
  for (int i = 0; i < 1; i++)
  {
    Teuchos::RCP<HYMLS::Solver> solver = ((extended_Dop_t const *)A_op)->solver;
    solver->SetTolerance(tol[i]);
    solver->setShift(1.0, -sigma[i]);
    //~ solver->setNullSpace(Teuchos::rcp<const Epetra_MultiVector>((const Epetra_MultiVector *)Qtil, false));
    //~ solver->SetupDeflation();
    solver->setProjectionVectors(Teuchos::rcp<const Epetra_MultiVector>((const Epetra_MultiVector *)Qtil, false));

    int ind = (resIndex == NULL ? i : resIndex[i]);
    phist_Dmvec_view_block(t, &t_i, i, i, iflag);
    TEUCHOS_TEST_FOR_EXCEPTION(*iflag != 0, std::runtime_error,
      "jadaCorrectionSolver_run: phist_Dmvec_view_block returned nonzero error code");

    phist_Dmvec_view_block((Dmvec_ptr_t)res, &r_i, ind, ind, iflag);
    TEUCHOS_TEST_FOR_EXCEPTION(*iflag != 0, std::runtime_error,
      "jadaCorrectionSolver_run: phist_Dmvec_view_block returned nonzero error code");

    solver->ApplyInverse(*(const Epetra_MultiVector *)r_i, *(Epetra_MultiVector *)t_i);
  }

  // normalize result vectors, TODO: should be done in updateSol/pgmres?
  _MT_ tmp[totalNumSys];
  phist_Dmvec_normalize(t, tmp, iflag);
  TEUCHOS_TEST_FOR_EXCEPTION(*iflag != 0, std::runtime_error,
    "jadaCorrectionSolver_run: phist_Dmvec_normalize returned nonzero error code");
}

