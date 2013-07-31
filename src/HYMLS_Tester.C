#include "HYMLS_Tester.H"
#include "HYMLS_Tools.H"
#include "HYMLS_HierarchicalMap.H"
#include "Teuchos_StandardCatchMacros.hpp"

#include "Epetra_CrsGraph.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_IntVector.h"

#include "EpetraExt_MatrixMatrix.h"

namespace HYMLS {

  std::stringstream Tester::msg_;
  int Tester::dof_=1;
  int Tester::pvar_=-1;
  bool Tester::doFmatTests_=false;
  int Tester::numFailedTests_=0;

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
    msg_ << "||A-A'||="<<C.NormInf()<<std::endl;
    ASSERT_TRUE(C.NormInf()<=float_tol(),status);
    return status;
    }

  //! returns true if the input matrix is an F-matrix, where the 
  //! pressure is each dof'th unknown, starting from pvar
  bool Tester::isFmatrix(const Epetra_CrsMatrix& A, int dof_in, int pvar_in)
    {
    bool status=true;
    if (!doFmatTests_) return status; 
    START_TIMER(Label(),"isFmatrix");
    int dof = dof_in<0 ? dof_: dof_in;
    int pvar = pvar_in<0 ? pvar_: pvar_in;
    msg_<<"dof="<<dof<<std::endl;
    msg_<<"pvar="<<pvar<<std::endl;
    ASSERT_TRUE(dof>0,status)
    ASSERT_TRUE(pvar>0,status)
    ASSERT_TRUE(pvar<dof,status)
    ASSERT_TRUE(isSymmetric(A.Graph()),status);

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
    return status;
    }

  //! this test is a specialized test for the 3D Navier-Stokes equations on a C-grid.
  //! It checks that the interior velocities of full conservation tubes only connect 
  //! to interior pressures of the same tube.
  bool Tester::areTubesCorrect(const Epetra_CrsMatrix& K,
                              const Epetra_IntVector& p_nodeType,
                              int dof, int pvar)
    {
    START_TIMER(Label(),"areTubesCorrect");
    
    // note: we do not test wether A is an F-matrix here, that is, wether Div=-Grad' etc.
    // the isFmatrix() test can be used for that independently. We only look at the grad-
    // part here.
    
    // Tubes only occure in 3D Navier-Stokes and similar, so we assume dim=3 if this is called.

    bool status=true;
    ASSERT_TRUE(K.Filled(),status);
    int len;
    int *cols;
    double *val;
    for (int i=0;i<K.NumMyRows();i++)
      {
      int grid = K.GRID(i);
      int p_lrid = p_nodeType.Map().LID(grid);
      if (p_nodeType[p_lrid]<0 && MOD(grid,dof)!=pvar)
        {
        // V-node in a tube
        ASSERT_ZERO(K.ExtractMyRowView(i,len,val,cols),status);
        for (int j=0;j<len;j++)
          {
          int gcid = K.GCID(cols[j]);
          if (MOD(gcid,dof)==pvar)
            {
            // entry in the grad part of the matrix K
            int p_lcid = p_nodeType.Map().LID(gcid);
            ASSERT_TRUE(p_lcid>=0,status);
            // check if the P-node is eliminated together with the
            // V-node or retained in an FCC, otherwise print warning.
            if (p_nodeType[p_lcid]!=p_nodeType[p_lrid] &&
                p_nodeType[p_lcid]<4)
              {
              msg_ << "V-node "<<grid<<" (variable type "<<MOD(grid,dof)<<")\n";
              msg_ << "belongs to the interior of a full conservation tube,\n ";
              msg_ << "but couples to P-node "<<gcid<< " outside the tube.\n";
              status=false;
              }
            }
          }
        }
      }
    return status;
    }

  bool Tester::noPcouplingsDropped(const Epetra_CrsMatrix& transSC,
                                    const HierarchicalMap& sepObject)
  {
    bool status=true;
    if (!doFmatTests_) return status; 
    START_TIMER(Label(),"noPcouplingsDropped");

    msg_<<"dof="<<dof_<<", pvar="<<pvar_<<std::endl;
    
    int len;
    double* val;
    int* cols;
    
    int blk=0;

  // loop over all separators
  for (int sep=0;sep<sepObject.NumMySubdomains();sep++)
    {
    // loop over all local separator groups
    for (int grp=0;grp<sepObject.NumGroups(sep);grp++)
      {
      // loop over all elements in the group, skipping the first one (the V-sum node)
      for (int i=1;i<sepObject.NumElements(sep,grp);i++)
        {
        int grid=sepObject.GID(sep,grp,i);
        // if this element is a V-node, check that any P-node couplings are 0
        if (MOD(grid,dof_)!=pvar_)
          {
          int lrid = transSC.LRID(grid);
          ASSERT_ZERO(transSC.ExtractMyRowView(lrid,len,val,cols),status);
          for (int j=0;j<len;j++)
            {
            int gcid = cols[j];
            if (MOD(gcid,dof_)==pvar_ && std::abs(val[j])>float_tol())
              {
              msg_ << "Coupling between non-Vsum-node "<<grid<<" and P-node "<<gcid<<" found.\n";
              msg_ << "This coupling of size "<<std::abs(val[j])<<" will be dropped.\n";
              status=false;
              }
            }
          }
        }
      }
    }
  return status;
  }             

}//namespace
