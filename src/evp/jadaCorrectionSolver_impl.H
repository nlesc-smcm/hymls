#ifndef PHIST_JADACORRECTIONSOLVERIMPL_H
#define PHIST_JADACORRECTIONSOLVERIMPL_H

// Get the HYMLS implementation of SCOREP_USER_REGION
#include "phist_macros.h"
#undef SCOREP_USER_REGION
#include "HYMLS_Macros.H"

#include "HYMLS_Solver.H"
#include "phist_operator.h"
#include "phist_enums.h"

typedef struct extended_Dop_t {
  Dop_t op;

//! Extended part for HYMLS only
  Teuchos::RCP<HYMLS::Solver> solver;
  bool borderedSolver;
  } extended_Dop_t;

#endif
