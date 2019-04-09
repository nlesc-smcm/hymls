#include "mex.h"

#include "HYMLS_Preconditioner.hpp"

#include "Teuchos_RCP.hpp"

#include "Epetra_MultiVector.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2)
        mexErrMsgIdAndTxt("HYMLS_init:nrhs", "Two input arguments required.");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("HYMLS_init:nlhs", "One output argument required.");

    // Get back the HYMLS object
    if (mxGetNumberOfElements(prhs[0]) != 1 || mxGetClassID(prhs[0]) != mxUINT64_CLASS)
      mexErrMsgIdAndTxt("HYMLS_init:prhs", "Input must be a real uint64 scalar.");
    HYMLS::Preconditioner *prec = reinterpret_cast<HYMLS::Preconditioner *>(*((uint64_t *)mxGetData(prhs[0])));

    double *x = mxGetPr(prhs[1]);
    int m = mxGetM(prhs[1]);
    int n = mxGetN(prhs[1]);

    plhs[0] = mxCreateDoubleMatrix((mwSize)m, (mwSize)n, mxREAL);
    double *y = mxGetPr(plhs[0]);

    Epetra_MultiVector xmv(View, prec->OperatorDomainMap(), x, m ,n);
    Epetra_MultiVector ymv(View, prec->OperatorRangeMap(), y, m, n);

    prec->ApplyInverse(xmv, ymv);
}
