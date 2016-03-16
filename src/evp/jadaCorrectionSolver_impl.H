#ifndef PHIST_JADACORRECTIONSOLVERIMPL_H
#define PHIST_JADACORRECTIONSOLVERIMPL_H

// Get the HYMLS implementation of SCOREP_USER_REGION
#include "phist_macros.h"
#undef SCOREP_USER_REGION
#include "HYMLS_Macros.H"

#include "HYMLS_Solver.H"
#include "phist_operator.h"
#include "phist_enums.h"
#include "phist_gen_d.h"
#include "phist_void_aliases.h"

//! wrapper to pass the hymls solver object to phist
typedef struct hymls_wrapper_t 
  {

  Teuchos::RCP<HYMLS::Solver> solver;
  bool borderedSolver;
  
  } hymls_wrapper_t;

void HYMLS_jadaCorrectionSolver_run(void* vme,
  void const* vA_op, void const* vB_op, 
  TYPE(const_mvec_ptr) Qtil, TYPE(const_mvec_ptr) BQtil,
  const double *sigma_r, const double *sigma_i, 
  TYPE(const_mvec_ptr) res, const int resIndex[],
  const double *tol, int maxIter,
  TYPE(mvec_ptr) t,
  int robust, int abortAfterFirstConvergedInBlock,
  int *iflag);

//! simplified phist correction solver interface, only allows solving for a single shift at a time.
void HYMLS_jadaCorrectionSolver_run1(void* vme,
  void const* vA_op, void const* vB_op, 
  TYPE(const_mvec_ptr) Qtil, TYPE(const_mvec_ptr) BQtil,
  double sigma_r, double sigma_i,
  TYPE(const_mvec_ptr) res,
  const double tol, int maxIter,
  TYPE(mvec_ptr) t,
  int robust,
  int *iflag);

// we need to replace the residual computation in jdqr right now by re-implementing
// it and passing it via the jadaOpts. This is because we want to enforce the Div-
// constraint in Navier-Stokes here.
//
// Details: The B_op has a pointer to a hymls_wrapper
// as its A member, from which we can get both the mass matrix and the HYMLS solver.
// In the future we should probably have the operators involved (A,B,Precond) have
// the velocity map as range and domain map and always act in the div-free space.
void HYMLS_computeResidual(TYPE(const_linearOp_ptr) B_op, TYPE(mvec_ptr) r_ptr,
        TYPE(mvec_ptr) Au_ptr, TYPE(mvec_ptr) u_ptr, TYPE(mvec_ptr) rtil_ptr,
        TYPE(mvec_ptr) Qv, TYPE(mvec_ptr) tmp, TYPE(sdMat_ptr) Theta,
        TYPE(sdMat_ptr) atil, TYPE(sdMat_ptr) *atilv, _MT_ *resid,
        int nv, int nconv, int* iflag);

        

#endif
