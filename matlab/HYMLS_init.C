#include "mex.h"

#include "HYMLS_Preconditioner.H"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"

#include "Epetra_SerialComm.h"
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2)
        mexErrMsgIdAndTxt("HYMLS_init:nrhs", "Two input arguments required.");
    if (nlhs != 1)
        mexErrMsgIdAndTxt("HYMLS_init:nlhs", "One output argument required.");

    if (!mxIsSparse(prhs[0]))
        mexErrMsgIdAndTxt("HYMLS_init:prhs", "Can't handle dense matrices.");

    // Get all the crs matrix arrays
    size_t *rows = mxGetIr(prhs[0]);
    size_t *cols = mxGetJc(prhs[0]);
    double *values = mxGetPr(prhs[0]);
    int n = mxGetM(prhs[0]);
    int nnz = mxGetNzmax(prhs[0]);

    // Get the parameterlist
    int buflen = mxGetNumberOfElements(prhs[1]) + 1;
    char *buf = (char *)mxCalloc(buflen, sizeof(char));
    if (mxGetString(prhs[1], buf, buflen) != 0)
      mexErrMsgIdAndTxt( "HYMLS_init::invalidStringArray",
        "Could not convert string data.");
    std::string params_file = buf;
    mxFree(buf);

    Teuchos::RCP<Teuchos::ParameterList> params = 
      Teuchos::getParametersFromXmlFile(params_file);

    // Copy the crs matrix array to an Epetra_CrsMatrix
    Epetra_SerialComm comm;
    Epetra_Map map(n, 0, comm);
    Teuchos::RCP<Epetra_CrsMatrix> mat = Teuchos::rcp(new Epetra_CrsMatrix(Copy, map, 2 * nnz / n));
    for (int j = 0; j < n; j++)
      {
      for (int i = cols[j]; i < cols[j+1]; i++)
        {
        mat->InsertGlobalValues(rows[i], 1, values + i, &j);
        }
      }

    mat->FillComplete();

    mexPrintf("Constructing preconditioner...\n");
    HYMLS::Preconditioner *prec = new HYMLS::Preconditioner(mat, params);

    bool status = true;

    mexPrintf("Initializing preconditioner...\n");
    try {
      prec->Initialize();
      } TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, status);

    if (!status)
      {
      mexErrMsgIdAndTxt("HYMLS_init::Initialize",
        "Caught an exception while initializing the preconditioner.");
      return;
      }

    mexPrintf("Computing preconditioner...\n");
    try {
      prec->Compute();
      } TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, status);

    if (!status)
      {
      mexErrMsgIdAndTxt("HYMLS_init:Compute",
        "Caught an exception while computing the preconditioner.");
      return;
      }

    mexPrintf("Preconditioner computed.\n");

    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t *)mxGetData(plhs[0])) = reinterpret_cast<uint64_t>(prec);
}
