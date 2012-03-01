#ifndef HYMLS_MACROS_H
#define HYMLS_MACROS_H

#include "Teuchos_StandardCatchMacros.hpp"

#ifdef DEBUGGING
#ifndef TESTING
#define TESTING
#endif
#endif

// the timing macro's are also used for function tracing, so enable them all:
#ifdef DEBUGGING
#define TIMING_LEVEL 10
#define PRINT_TIMING false
#else
#define TIMING_LEVEL 2
#define PRINT_TIMING false
#endif

#ifdef TESTING
#ifndef FUNCTION_TRACING
#define FUNCTION_TRACING
#endif
#endif

// the TIMING_LEVEL macro defines how many functions are included
// in the timing output at the end of the run.
//
// 0: no timing at all
// 1: timing of major functions like preconditioner setup
// 2: detailed timing of things like applying orthogonal transformations
// 3: time everything, this is mainly for the case where TESTING/DEBUGGING
//    is defined and the timing routines are abused for function tracing. 
//    setting this may infringe the performance substantially.


// the START_TIMER macros can be placed anywhere in the code,
// but only once in each scope ({}-block). The
// timing is stopped at the end of the current scope. So for 
// instance, to time a function foo and a part of it as well,
// do something like this:
// void bar::foo()
//   {
//   START_TIMER("bar","foo");
//   ... do something that is not timed separately ...
//   {
//   START_TIMER2("bar","foo-part")
//      ... do foo-part
//   }
//   ... do some more work that is not timed separately
//  }


#if TIMING_LEVEL>0
#define START_TIMER(s1,s2) HYMLS::TimerObject \
Error_You_are_trying_to_start_multiple_timers_in_one_scope \
(((std::string)(s1))+": "+(((std::string)(s2))),PRINT_TIMING)
#else
#define START_TIMER(s1,s2)
#endif

#if TIMING_LEVEL>1
#define START_TIMER2(s1,s2) START_TIMER(s1,s2)
#else
#define START_TIMER2(s1,s2)
#endif

#if TIMING_LEVEL>2
#define START_TIMER3(s1,s2) START_TIMER(s1,s2)
#else
#define START_TIMER3(s1,s2) 
#endif

#ifdef DEBUGGING
#ifndef DEBUG
#define DEBUG(s) if (HYMLS::Tools::InitializedIO()) HYMLS::Tools::deb() << s << std::endl;
#endif
#ifndef DEBVAR
#define DEBVAR(s) if (HYMLS::Tools::InitializedIO()) HYMLS::Tools::deb() << #s << " = "<<s << std::endl;
#endif
#else
#ifndef DEBUG
#define DEBUG(s) 
#endif
#ifndef DEBVAR
#define DEBVAR(s) 
#endif
#endif
#ifndef CHECK_ZERO
#define CHECK_ZERO(funcall) {int ierr=0; bool status=true; try{\
int ierr = funcall; } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr,status)\
if (!status) {\
std::string msg="Caught an exception in call "+std::string(#funcall);\
HYMLS::Tools::Fatal(msg,__FILE__,__LINE__);} \  
if (ierr) {\
std::string msg="error code "+Teuchos::toString(ierr)+" returned from call "+std::string(#funcall);\
HYMLS::Tools::Error(msg,__FILE__,__LINE__);}\
}
#endif

#ifndef CHECK_TRUE
#define CHECK_TRUE(funcall) {bool berr = funcall;\
if (!berr) {std::cerr<<"Trilinos call "<<#funcall<<" returned false"<<std::endl;}}
#endif
#ifndef CHECK_NONNEG
#define CHECK_NONNEG(funcall) {int ierr = funcall;\
if (ierr<0) {std::cerr<<"Trilinos Error "<<ierr<<" returned from call "<<#funcall<<std::endl;}}
#endif

//! our own 'modulo' function, which behaves like mod in matlab.
//! the C++ built-in '%' operator returns -1%n=-1 and is therefore
//! not very useful for periodic boundaries...
#ifndef MOD
#define MOD(x,y) (((double)(y)==0.0)? (double)(x): ((double)(x) - floor((double)(x)/((double)(y)))*((double)(y))))
#endif

#endif
