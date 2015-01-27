#ifndef PHIST_JADACORRECTIONSOLVERIMPL_H
#define PHIST_JADACORRECTIONSOLVERIMPL_H

// Get the HYMLS implementation of SCOREP_USER_REGION
#include "phist_macros.h"
#undef SCOREP_USER_REGION
#include "HYMLS_Macros.H"

#include "HYMLS_Solver.H"
#include "phist_operator.h"
#include "phist_enums.h"

typedef struct phist_extendedJadaOpts_t {

// what do you want to compute?
int numEigs; //! howmany eigenpairs are sought?
eigSort_t which; //! LM, SM, LR, SR, or TARGET
linSolv_t innerSolvType; //! GMRES or CARP_CG currently supported
double convTol; //! convergence tolerance for eigenvalues

// JaDa configuration
int maxIters; //! maximum iterations allowed
int blockSize; //! only for block methods (subspacejada and blockjada)
int minBas; //! number of vectors retained upon restart
int maxBas; //! maximum number of vectors allowed in the basis

// how should JaDa start up?
void* v0; //! can be used to pass in a start-up vector(-space) (can have any number of 
          //! columns). v0 is assumed to be orthonormal.
int arno; //! 0: no Arnoldi steps. 1: minBas Arnoldi steps to start up.
double initialShift; //! can be used to start with an initial shift
                     //! (ignored if arno!=0)

//! Extended part for HYMLS only
Teuchos::RCP<HYMLS::Solver> solver;
} phist_extendedJadaOpts_t;

typedef struct extended_Dop_t {
    Dop_t op;

//! Extended part for HYMLS only
    Teuchos::RCP<HYMLS::Solver> solver;
} extended_Dop_t;

#endif
