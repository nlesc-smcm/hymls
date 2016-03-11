// this file can be included before the other #includes
// in a .C file to disable debugging output for this  
// one file.

#include "HYMLS_config.h"

#ifdef HYMLS_DEBUGGING
#undef HYMLS_DEBUGGING
#endif

#ifdef HYMLS_FUNCTION_TRACING
#undef HYMLS_FUNCTION_TRACING
#endif

#ifdef HYMLS_DEBUG
#undef HYMLS_DEBUG
#endif
#define HYMLS_DEBUG(s) 

#ifdef HYMLS_DEBVAR
#undef HYMLS_DEBVAR
#endif
#define HYMLS_DEBVAR(s)

#ifdef HYMLS_TIMING_LEVEL
#undef HYMLS_TIMING_LEVEL
#endif
#define HYMLS_TIMING_LEVEL 2
