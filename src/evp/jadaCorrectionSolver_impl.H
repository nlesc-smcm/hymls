#ifndef PHIST_JADACORRECTIONSOLVERIMPL_H
#define PHIST_JADACORRECTIONSOLVERIMPL_H

// Get the HYMLS implementation of SCOREP_USER_REGION
#include "phist_macros.h"
#undef SCOREP_USER_REGION
#include "HYMLS_Macros.H"

#include "HYMLS_Solver.H"
#include "phist_operator.h"
#include "phist_enums.h"

//! wrapper to pass the hymls solver object to phist
typedef struct hymls_wrapper_t {

  Teuchos::RCP<HYMLS::Solver> solver;
  bool borderedSolver;
  
  } hymls_wrapper_t;

//! simplified phist correction solver interface, only allows solving for a single shift at a time.
void HYMLS_jadaCorrectionSolver_run1)(hymls_wrapper_t* me,
                                    TYPE(const_op_ptr)    A_op,     TYPE(const_op_ptr)    B_op,
                                    TYPE(const_mvec_ptr)  Qtil,     TYPE(const_mvec_ptr)  BQtil,
                                    double                sigma_r,  double                sigma_i,
                                    TYPE(const_mvec_ptr)  res,
                                    const double          tol,      int                   maxIter,
                                    TYPE(mvec_ptr)        t,
                                    bool robust,                   bool abortAfterFirstConvergedInBlock,
                                    int *                 iflag);


#endif
