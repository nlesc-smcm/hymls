#include "mex.h"

#include "HYMLS_Preconditioner.hpp"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs != 1)
        mexErrMsgIdAndTxt("HYMLS_init:nrhs", "One input argument required.");
    if (nlhs != 0)
        mexErrMsgIdAndTxt("HYMLS_init:nlhs", "Zero output arguments required.");

    // Get back the HYMLS object
    if (mxGetNumberOfElements(prhs[0]) != 1 || mxGetClassID(prhs[0]) != mxUINT64_CLASS)
      mexErrMsgIdAndTxt("HYMLS_init:prhs", "Input must be a real uint64 scalar.");
    HYMLS::Preconditioner *prec = reinterpret_cast<HYMLS::Preconditioner *>(*((uint64_t *)mxGetData(prhs[0])));

    delete prec;
}
