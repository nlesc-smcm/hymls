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

//! simplified phist correction solver interface, only allows solving for a single shift at a time.
void HYMLS_jadaCorrectionSolver_run1(void* me,
                                    void const* A_op,     void const*    B_op,
                                    TYPE(const_mvec_ptr)  Qtil,     TYPE(const_mvec_ptr)  BQtil,
                                    double                sigma_r,  double                sigma_i,
                                    TYPE(const_mvec_ptr)  res,
                                    const double          tol,      int                   maxIter,
                                    TYPE(mvec_ptr)        t,
                                    int robust,
                                    int *                 iflag);


#endif
