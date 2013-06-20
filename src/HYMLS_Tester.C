#include "HYMLS_Tester.H"
#include "HYMLS_Tools.H"
#include "Teuchos_StandardCatchMacros.hpp"

#include "Epetra_CrsGraph.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_IntVector.h"

#include "EpetraExt_MatrixMatrix.h"

namespace HYMLS {

  std::stringstream Tester::msg_;


#define ASSERT_ZERO(FCN,STATUS) \
  try { \
  int ierr = FCN; \
  if (ierr!=0) {msg_ << "call "<<#FCN<<" returned non-zero value "<<ierr<<std::endl; STATUS=false;} \
  } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,msg_,STATUS); \
  if (!STATUS) return STATUS;

#define ASSERT_TRUE(FCN,STATUS) \
  try { \
  STATUS = FCN; \
  if (!STATUS) {msg_ << "call "<<#FCN<<" returned false" <<std::endl;} \
  } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,msg_,STATUS); \
  if (!STATUS) return STATUS; 

  // returns true if the input graph (i.e. the sparsity pattern of a matrix) is symmetric
  bool Tester::isSymmetric(const Epetra_CrsGraph& G)
    {
    START_TIMER(Label(),"isSymmetric(G)");
    bool status=true;
    ASSERT_TRUE(G.Filled(),status);
    Epetra_CrsMatrix A(Copy,G);
    ASSERT_ZERO(A.PutScalar(1.0),status);
    ASSERT_TRUE(isSymmetric(A),status);
    return status;
    }

  //! returns true if the input matrix is symmetric
  bool Tester::isSymmetric(const Epetra_CrsMatrix& A)
    {
    START_TIMER(Label(),"isSymmetric(A)");
    bool status=true;
    ASSERT_TRUE(A.Filled(),status);
    // we do assume here that the MatrixMatrix::Add function is correct
    Epetra_CrsMatrix C = A;
    ASSERT_ZERO(EpetraExt::MatrixMatrix::Add(A,true,-1.0,C,1.0),status);
    ASSERT_TRUE(C.HasNormInf(),status); 
    ASSERT_TRUE(C.NormInf()<=float_tol(),status);
    return status;
    }

  //! returns true if the input matrix is an F-matrix, where the 
  //! pressure is each dof'th unknown, starting from pvar
  bool Tester::isFmatrix(const Epetra_CrsMatrix& A, int dof, int pvar)
    {
    START_TIMER(Label(),"isFmatrix");
    DEBVAR(dof);
    DEBVAR(pvar);
    bool status = true;
    ASSERT_TRUE(isSymmetric(A.Graph()),status);
    if (pvar>0 && dof>1) 
      {
      int len;
      double * val;
      int *cols;
      for (int i=0; i<A.NumMyRows(); i++)
        {
        int grid = A.GRID(i);
        if (MOD(grid,dof)!=pvar)
          {
          ASSERT_ZERO(A.ExtractMyRowView(i,len,val,cols),status);
          int num_pcols=0; // should be at most 2
          double psum=0.0; // should be 0
          for (int j=0; j<len;j++)
            {
            int gcid = A.GCID(cols[j]);
            if (MOD(gcid,dof)==pvar)
              {
              num_pcols++;
              psum+=val[j];
              }
            }
          if (num_pcols>2) 
            {
            msg_ << "global row "<<grid<< " has "<< num_pcols << " entries in Grad-part"<<std::endl;
            status=false;
            }
          if (abs(psum)>float_tol())
            {
            msg_ << "global row "<<grid<< " has row sum(G)="<< psum << std::endl;
            status=false;
            }
          }
        }
      }
    return status;
    }

}//namespace
