// this file can be included before the other #includes
// in a .C file to disable debugging output for this  
// one file.

#ifdef DEBUGGING
#undef DEBUGGING
#endif

#ifdef FUNCTION_TRACING
#undef FUNCTION_TRACING
#endif

#ifdef DEBUG
#undef DEBUG
#endif
#define DEBUG(s) 

#ifdef DEBVAR
#undef DEBVAR
#endif
#define DEBVAR(s)

#ifdef TIMING_LEVEL
#undef TIMING_LEVEL
#endif
#define TIMING_LEVEL 2
