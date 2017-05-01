#ifndef PHIST_JADACORRECTIONSOLVERIMPL_H
#define PHIST_JADACORRECTIONSOLVERIMPL_H

// Get the HYMLS implementation of SCOREP_USER_REGION
#include "phist_macros.h"
#undef SCOREP_USER_REGION
#include "HYMLS_Macros.H"

#include "HYMLS_Preconditioner.H"
#include "phist_operator.h"
#include "phist_enums.h"
#include "phist_gen_d.h"
#include "phist_void_aliases.h"

namespace HYMLS {

//! \name functions used to wrap a HYMLS::Preconditioner as a phist_DlinearOp
//!@{

 //!
 void phist_precon_apply(_ST_ alpha, const void* P, 
        TYPE(const_mvec_ptr) X, _ST_ beta,  TYPE(mvec_ptr) Y, int* iflag);
 //!
 void phist_precon_apply_shifted(_ST_ alpha, const void* P, _ST_ const sigma[],
        TYPE(const_mvec_ptr) X, _ST_ beta,  TYPE(mvec_ptr) Y, int* iflag);
 //!
 void phist_precon_update(void const* P, void* aux, _ST_ sigma,
                    TYPE(const_mvec_ptr) Vkern,
                    TYPE(const_mvec_ptr) BVkern,
                    int* iflag);
//@}

}

#endif
