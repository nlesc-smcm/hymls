#include "mex.h"

#include "HYMLS_Preconditioner.H"

#include "Teuchos_RCP.hpp"

#include "Epetra_MultiVector.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2 || nrhs > 3)
        mexErrMsgIdAndTxt("HYMLS_init:nrhs", "Two or three input arguments required.");

    // Get back the HYMLS object
    if (mxGetNumberOfElements(prhs[0]) != 1 || mxGetClassID(prhs[0]) != mxUINT64_CLASS)
        mexErrMsgIdAndTxt("HYMLS_init:prhs", "Input must be a real uint64 scalar.");
    HYMLS::Preconditioner *prec = reinterpret_cast<HYMLS::Preconditioner *>(*((uint64_t *)mxGetData(prhs[0])));

    double *v = mxGetPr(prhs[1]);
    double *w = v;
    if (nrhs == 3)
        w = mxGetPr(prhs[2]);

    int m = mxGetM(prhs[1]);
    int n = mxGetN(prhs[1]);

    Teuchos::RCP<Epetra_MultiVector> vmv = Teuchos::rcp(
        new Epetra_MultiVector(View, prec->OperatorDomainMap(), v, m ,n));
    Teuchos::RCP<Epetra_MultiVector> wmv = Teuchos::rcp(
        new Epetra_MultiVector(View, prec->OperatorDomainMap(), w, m ,n));

    prec->setBorder(vmv, wmv);
}
