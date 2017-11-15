#ifndef HYMLS_PHIST_CUSTOM_CORRECTION_SOLVER_H
#define HYMLS_PHIST_CUSTOM_CORRECTION_SOLVER_H

// Get the HYMLS implementation of SCOREP_USER_REGION
#include "phist_macros.h"
#include "HYMLS_Macros.H"

#include "HYMLS_Solver.H"
#include "phist_operator.h"
#include "phist_enums.h"
#include "phist_gen_d.h"
#include "phist_void_aliases.h"

//! wrapper to pass the hymls solver object to phist
typedef struct phist_hymls_solver
{

    Teuchos::RCP<HYMLS::Solver> solver;
    bool borderedSolver;
  
} phist_hymls_solver;

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

#endif
