#include "HYMLS_Tester.H"
#include "HYMLS_Tools.H"
#include "HYMLS_OverlappingPartitioner.H"
#include "HYMLS_HierarchicalMap.H"
#include "Teuchos_StandardCatchMacros.hpp"

#include "Epetra_CrsGraph.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Vector.h"
#include "Epetra_IntVector.h"

#include "EpetraExt_Transpose_RowMatrix.h"
#include "EpetraExt_Transpose_CrsGraph.h"

namespace HYMLS {

  std::stringstream Tester::msg_;
  int Tester::dim_=-1;
  int Tester::dof_=-1;
  int Tester::pvar_=-1;
  int Tester::nx_=-1;
  int Tester::ny_=-1;
  int Tester::nz_=-1;
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

#define ASSERT_EQUALITY(FCN1,FCN2,STATUS) \
  try { \
  STATUS = (FCN1 == FCN2); \
  if (!STATUS) {msg_ << "unexpected result " << #FCN1 << " = " << FCN1 << " != " << #FCN2 << " = " << FCN2 << std::endl;} \
  } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,msg_,STATUS); \
  if (!STATUS) return STATUS; 

#define ASSERT_FLOATING_EQUALITY(FCN1,FCN2,STATUS) \
  try { \
  STATUS = (std::abs(FCN1 - FCN2) < float_tol()); \
  if (!STATUS) {msg_ << "unexpected result " << #FCN1 << " = " << FCN1 << " != " << #FCN2 << " = " << FCN2 << std::endl;} \
  } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,msg_,STATUS); \
  if (!STATUS) return STATUS; 

#define EXPECT_ZERO(FCN,STATUS) \
  try { \
  int ierr = FCN; \
  if (ierr!=0) {msg_ << "call "<<#FCN<<" returned non-zero value "<<ierr<<std::endl; STATUS=false;} \
  } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,msg_,STATUS); \

#define EXPECT_TRUE(FCN,STATUS) \
  try { \
  STATUS = FCN; \
  if (!STATUS) {msg_ << "call "<<#FCN<<" returned false" <<std::endl;} \
  } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,msg_,STATUS); \

  //! returns true if the input graph (i.e. the sparsity pattern of a matrix) is symmetric
  bool Tester::isSymmetric(const Epetra_CrsGraph& G)
    {
    HYMLS_PROF(Label(),"isSymmetric(G)");
    bool status = true;
    ASSERT_TRUE(G.Filled(), status);

    Teuchos::RCP<EpetraExt::CrsGraph_Transpose> transposer = Teuchos::RCP<EpetraExt::CrsGraph_Transpose>(new EpetraExt::CrsGraph_Transpose());
    Epetra_CrsGraph *GTranspose = &(dynamic_cast<Epetra_CrsGraph&>(((*transposer)(const_cast<Epetra_CrsGraph&>(G)))));

    ASSERT_EQUALITY(G.NumMyRows(), GTranspose->NumMyRows(), status);

    int MaxNumEntries = std::max(G.GlobalMaxNumIndices(), GTranspose->GlobalMaxNumIndices());

    int GNumEntries, BNumEntries;
    int *GIndices = new int[MaxNumEntries];
    int *BIndices = new int[MaxNumEntries];

    for (int i = 0; i < G.NumMyRows(); ++i)
      {
      G.ExtractMyRowCopy(i, MaxNumEntries, GNumEntries, GIndices);
      for (int j = 0; j < GNumEntries; ++j)
        GIndices[j] = G.GCID(GIndices[j]);

      GTranspose->ExtractMyRowCopy(i, MaxNumEntries, BNumEntries, BIndices);
      for (int j = 0; j < BNumEntries; ++j)
        BIndices[j] = GTranspose->GCID(BIndices[j]);

      ASSERT_EQUALITY(GNumEntries, BNumEntries, status);

      std::sort(GIndices, GIndices+GNumEntries);
      std::sort(BIndices, BIndices+BNumEntries);

      for (int j = 0; j < GNumEntries; ++j)
        {
        ASSERT_EQUALITY(GIndices[j], BIndices[j], status);
        }
      }

    delete[] GIndices;
    delete[] BIndices;

    return status;
    }

  //! returns true if the input matrix is symmetric
  bool Tester::isSymmetric(const Epetra_CrsMatrix& A)
    {
    HYMLS_PROF(Label(),"isSymmetric(A)");
    bool status = true;
    ASSERT_TRUE(A.Filled(), status);

    Teuchos::RCP<EpetraExt::RowMatrix_Transpose> transposer = Teuchos::RCP<EpetraExt::RowMatrix_Transpose>(new EpetraExt::RowMatrix_Transpose());
    Epetra_CrsMatrix *ATranspose = &(dynamic_cast<Epetra_CrsMatrix&>(((*transposer)(const_cast<Epetra_CrsMatrix&>(A)))));

    ASSERT_EQUALITY(A.NumMyRows(), ATranspose->NumMyRows(), status);

    int MaxNumEntries = std::max(A.GlobalMaxNumEntries(), ATranspose->GlobalMaxNumEntries());

    int ANumEntries, BNumEntries;
    int *AIndices = new int[MaxNumEntries];
    int *BIndices = new int[MaxNumEntries];
    double *AValues = new double[MaxNumEntries];
    double *BValues = new double[MaxNumEntries];

    for (int i = 0; i < A.NumMyRows(); ++i)
      {
      A.ExtractMyRowCopy(i, MaxNumEntries, ANumEntries, AValues, AIndices);
      for (int j = 0; j < ANumEntries; ++j)
        AIndices[j] = A.GCID(AIndices[j]);

      ATranspose->ExtractMyRowCopy(i, MaxNumEntries, BNumEntries, BValues, BIndices);
      for (int j = 0; j < BNumEntries; ++j)
        BIndices[j] = ATranspose->GCID(BIndices[j]);

      ASSERT_EQUALITY(ANumEntries, BNumEntries, status);

      std::sort(AIndices, AIndices+ANumEntries);
      std::sort(BIndices, BIndices+BNumEntries);

      for (int j = 0; j < ANumEntries; ++j)
        {
        ASSERT_EQUALITY(AIndices[j], BIndices[j], status);
        ASSERT_FLOATING_EQUALITY(AValues[j], BValues[j], status);
        }
      }

    delete[] AIndices;
    delete[] BIndices;
    delete[] AValues;
    delete[] BValues;

    return status;
    }

  //! returns true if the input matrix is the identity matrix
  bool Tester::isIdentity(const Epetra_CrsMatrix& A)
    {
    HYMLS_PROF(Label(),"isIdentity(A)");
    bool status=true;
    ASSERT_TRUE(A.Filled(),status);

    double norm;
    Epetra_Vector diag(A.RangeMap());
    A.ExtractDiagonalCopy(diag);
    Epetra_Vector diag2(A.RangeMap());
    diag2.PutScalar(1.0);
    diag2.Update(1.0, diag, -1.0);

    diag2.Norm2(&norm);
    msg_ << "||diag(a)-e||="<<norm<<std::endl;
    ASSERT_TRUE(norm<=float_tol(),status);

    Epetra_CrsMatrix C = A;
    C.ReplaceDiagonalValues(diag2);
    ASSERT_TRUE(C.HasNormInf(),status);
    msg_ << "||A-I||="<<C.NormInf()<<std::endl;
    ASSERT_TRUE(C.NormInf()<=float_tol(),status);

    return status;
    }

  //! returns true if the input matrix is an F-matrix, where the 
  //! pressure is each dof'th unknown, starting from pvar
  bool Tester::isFmatrix(const Epetra_CrsMatrix& A, int dof_in, int pvar_in)
    {
    bool status=true;
    if (!doFmatTests_) return status; 
    HYMLS_PROF(Label(),"isFmatrix");
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
    HYMLS_PROF(Label(),"areTubesCorrect");
    
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
      ASSERT_TRUE(p_lrid>=0,status);
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

  // check that the domain decomposition worked, i.e. there are no couplings between 
  // interior nodes of two different subdomains.
  bool Tester::isDDcorrect(const Epetra_CrsMatrix& A, 
                                  const HYMLS::OverlappingPartitioner& hid)
    {
    HYMLS_PROF(Label(),"isDDcorrect");
    bool status=true;
    if (A.RowMap().Comm().NumProc()>1)
      {
      // the map spawning does not work if the nodes connect to subdomains
      // on other processes
      msg_ << "test skipped, it works only for serial runs"<<std::endl;
      return status;
      }
    const HYMLS::BasePartitioner& part = hid.Partitioner();
    int* cols;
    double* val;
    int len;
    // make sure everyting is correctly initialized for this test:
    ASSERT_TRUE(nx_>=0,status);
    ASSERT_TRUE(ny_>=0,status);
    ASSERT_TRUE(nz_>=0,status);
    ASSERT_TRUE(dof_>=0,status);
    ASSERT_TRUE(dim_>=0,status);
    msg_ << "nx="<<nx_<<std::endl;
    msg_ << "ny="<<ny_<<std::endl;
    msg_ << "nz="<<nz_<<std::endl;
    msg_ << "dim="<<dim_<<std::endl;
    msg_ << "dof="<<dof_<<std::endl;
    for (int i=0;i<A.NumMyRows();i++)
      {
      int gid_i= A.GRID(i);
      int sd_i = part(A.GRID(i));
      // map containing only interior nodes of subdomain i
      Teuchos::RCP<const Epetra_Map> imap_i = hid.SpawnMap(sd_i,HierarchicalMap::Interior);
      // map containing only separator nodes of subdomain i
      Teuchos::RCP<const Epetra_Map> smap_i = hid.SpawnMap(sd_i,HierarchicalMap::Separators);
      bool hasInteriorCoupling=false; // check if an interior node of sd_i
                                      // has any edge inside the subdomain,
                                      // if not the sd matrix has an empty 
                                      // row/column and is singular.
      ASSERT_ZERO(A.ExtractMyRowView(i,len,val,cols),status);
      for (int j=0;j<len;j++)
        {
        int gid_j= A.GCID(cols[j]);
        int sd_j = part(gid_j);
          
        if (sd_i!=sd_j)
          {
          if (A.RowMap().MyGID(gid_j))
            {
            // map containing only interior nodes of subdomain j
            Teuchos::RCP<const Epetra_Map> imap_j = hid.SpawnMap(sd_j,HierarchicalMap::Interior);
           // map containing only separator nodes of subdomain j
            Teuchos::RCP<const Epetra_Map> smap_j = hid.SpawnMap(sd_j,HierarchicalMap::Separators);
            // basic sanity check - no node can be both interior and separator or neither
            ASSERT_TRUE(imap_i->LID(gid_i)>=0 || smap_i->LID(gid_i)>=0, status);
            ASSERT_TRUE(!(imap_i->LID(gid_i)>=0 && smap_i->LID(gid_i)>=0), status);
            ASSERT_TRUE(imap_j->LID(gid_j)>=0 || smap_j->LID(gid_j)>=0, status);
            ASSERT_TRUE(!(imap_j->LID(gid_j)>=0 && smap_j->LID(gid_j)>=0), status);

            // check these options
            // a) i is a separator of sd_i and sd_j, j is interior of either sd_i or sd_j.
            //    the indices appear just on one of the interior maps and not
            //    as both a separator and interior node (ok1)
            // b) i is interior of sd_i, j is separator of sd_i and sd_j (ok2)
            // c) corner situation, both i and j are separators,
            // and at least one of them is in a group of it's own (a retained node) (ok3)
            bool ok1 = smap_i->LID(gid_i)>=0 && 
                     smap_j->LID(gid_i)>=0 &&
                     imap_j->LID(gid_j)>=0;
            bool ok2 = smap_i->LID(gid_j)>=0 && 
                     smap_j->LID(gid_j)>=0 &&
                     imap_i->LID(gid_i)>=0;
            bool ok3=false;
            if (!(ok1||ok2))
              {
              ok3 = smap_i->LID(gid_i)>=0 && 
                    smap_j->LID(gid_j)>=0;
              // we should do more tests if they are both
              // separator nodes, but covering all situations
              // is a bit difficult right now (TODO)
              if (ok3&&false) // both separators
                {
                msg_ << "edge ("<<gid_i<<", "<<gid_j<<") candidate for singleton coupling\n";
                // check if one of them is a retained node, that is, it is
                // in a singleton group. That one should be a separator of
                // both sd_i and sd_j, whereas the other is in only one of
                // the two.
                int grp_i_sd_i=-1;
                int grp_j_sd_i=-1;
                for (int grp=1;grp<hid.NumGroups(sd_i);grp++)
                  {
                  for (int jj=0; jj<hid.NumElements(sd_i,grp);jj++)
                    {
                    if (hid.GID(sd_i,grp,jj)==gid_i)
                      {
                      // check that the gid is only in one group
                      ASSERT_TRUE(grp_i_sd_i==-1,status);
                      grp_i_sd_i=grp;
                      }
                    if (hid.GID(sd_i,grp,jj)==gid_j)
                      {
                      // check that the gid is only in one group
                      ASSERT_TRUE(grp_j_sd_i==-1,status);
                      grp_j_sd_i=grp;
                      }
                    }
                  }
                int grp_i_sd_j=-1;
                int grp_j_sd_j=-1;
                for (int grp=1;grp<hid.NumGroups(sd_j);grp++)
                  {
                  for (int jj=0; jj<hid.NumElements(sd_j,grp);jj++)
                    {
                    if (hid.GID(sd_j,grp,jj)==gid_i)
                      {
                      // check that the gid is only in one group
                      ASSERT_TRUE(grp_i_sd_j==-1,status);
                      grp_i_sd_j=grp;
                      }
                    if (hid.GID(sd_j,grp,jj)==gid_j)
                      {
                      // check that the gid is only in one group
                      ASSERT_TRUE(grp_j_sd_j==-1,status);
                      grp_j_sd_j=grp;
                      }
                    }
                  }
                // gid_i or gid_j is in a singleton group
                ok3 = hid.NumElements(sd_i,grp_i_sd_i)==1 ||
                    hid.NumElements(sd_j,grp_j_sd_j)==1;
                if (!ok3)
                  {
                  msg_ << "gid "<<gid_i<<" sd "<<sd_i<<", group "<<grp_i_sd_i;
                  msg_ << " ("<<hid.NumElements(sd_i,grp_i_sd_i)<<" elements)"<<std::endl;
                  if (grp_i_sd_j>0)
                    {
                    msg_ << "gid "<<gid_i<<" sd "<<sd_j<<", group "<<grp_i_sd_j;
                    msg_ << " ("<<hid.NumElements(sd_j,grp_i_sd_j)<<" elements)"<<std::endl;
                    }
                  msg_ << "gid "<<gid_j<<" sd "<<sd_j<<", group "<<grp_j_sd_j;
                  msg_ << "("<<hid.NumElements(sd_j,grp_j_sd_j)<<" elements)"<<std::endl;
                  if (grp_j_sd_i>0)
                    {
                    msg_ << "gid "<<gid_j<<" sd "<<sd_i<<", group "<<grp_j_sd_i<<std::endl;
                    msg_ << " ("<<hid.NumElements(sd_i,grp_j_sd_i)<<" elements)"<<std::endl;
                    }
                  }
                }
              }
            if (!(ok1||ok2||ok3))
              {
              if (!(ok1||ok2))
                {
                msg_ <<" interior of sd_i: "<<*imap_i<<std::endl;
                msg_ <<" separators of sd_i: "<<*smap_i<<std::endl;
                msg_ <<" interior of sd_j: "<<*imap_j<<std::endl;
                msg_ <<" separators of sd_j: "<<*smap_j<<std::endl;
                }
              msg_<<" edge between gid "<<gid_i<<" [sd "<<sd_i<<", "<<gid2str(gid_i)<<"] and "
                                      <<gid_j<<" [sd "<<sd_j<<", "<<gid2str(gid_j)<<"] is incorrect\n";
              status=false;
              }
            }
          }
        else if (imap_i->LID(gid_i)>=0 && imap_i->LID(gid_j)>=0)
          {
          // both belong to same subdomain and are interior
          // edge between interior nodes. Is it nonzero?
          if (std::abs(val[j])>HYMLS_SMALL_ENTRY)
            {
            hasInteriorCoupling=true;
            }
          }
        }
      // singular subdomain matrix?
      if (imap_i->LID(gid_i)>=0 && (hasInteriorCoupling==false))
        {
        msg_<<"row "<<gid_i<<" [sd "<<sd_i<<", "<<gid2str(gid_i)<<"] is empty in sd "<<sd_i<<std::endl;
        msg_<<"matrix row: "<<std::endl;
        for (int j=0;j<len;j++)
          {
          int gid_j= A.GCID(cols[j]);
          int sd_j = part(gid_j);
          msg_ << gid_i << " " << gid_j << " [sd "<<sd_j<<", "<<gid2str(gid_j)<<"]\t" << val[j]<<std::endl;
          }
        msg_ <<" interior of sd_i: "<<*imap_i<<std::endl;
        msg_ <<" separators of sd_i: "<<*smap_i<<std::endl;
        status=false;
        }
      }
    return status;
    }

  bool Tester::noPcouplingsDropped(const Epetra_CrsMatrix& transSC,
                                    const HierarchicalMap& sepObject)
  {
    bool status=true;
    if (!doFmatTests_) return status; 
    HYMLS_PROF(Label(),"noPcouplingsDropped");

    msg_<<"dof="<<dof_<<", pvar="<<pvar_<<std::endl;

    int len;
    double* val;
    int* cols;

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
            int gcid = transSC.GCID(cols[j]);
            if (MOD(gcid,dof_)==pvar_ && std::abs(val[j])>float_tol())
              {
              msg_ << "Coupling between non-Vsum-node "<<grid<<" "<<gid2str(grid)<<
              " and P-node "<<gcid<<" "<<gid2str(gcid)<<" found.\n";
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

  bool Tester::noNumericalZeros(const Epetra_CrsMatrix& A)
    {
    bool status=true;
    HYMLS_PROF(Label(),"noNumericalZeros");
    int len;
    double* val;
    int* cols;
    for (int i=0;i<A.NumMyRows();i++)
      {
      int grid=A.GRID(i);
      ASSERT_ZERO(A.ExtractMyRowView(i,len,val,cols),status);
      for (int j=0;j<len;j++)
        {
        if (std::abs(val[j])<=std::numeric_limits<double>::epsilon())
          {
          msg_ << "small entry A("<<grid<<","<<A.GCID(cols[j])<<")="<<val[j]<<std::endl;
          msg_ << "row "<<gid2str(grid)<<std::endl;
          msg_ << "col "<<gid2str(A.GCID(cols[j]))<<std::endl;
          status=false;
          }
        }
     
      }
    
    return status;
    }

  bool Tester::isDivFree(const Epetra_CrsMatrix& A, const Epetra_MultiVector &V,  int dof, int pvar, double tol)
    {
    HYMLS_PROF(Label(),"isDivFree");
    Epetra_MultiVector out(A.OperatorRangeMap(), V.NumVectors());
    CHECK_ZERO(A.Apply(V, out));

    bool status = true;

    for (int j = 0; j < out.NumVectors(); j++)
      {
      for (int i = 0; i < out.MyLength(); i++)
        {
        if (std::abs(out[j][i]) > tol && (out.Map().GID(i) % dof == pvar))
          {
          msg_ << "Rowsum not zero but " << out[j][i]
               << " on row " << out.Map().GID(i) << std::endl;
          status = false;
          }
        }
      }
    return status;
    }

  // convert global index to string for output (gid => (i,j,k,v))
  std::string Tester::gid2str(int gid)
    {
    if (nx_<0) return "Tester not initialized correctly";
    int i,j,k,v;
    Tools::ind2sub(nx_,ny_,nz_,dof_,gid,i,j,k,v);
    std::stringstream ss;
    std::string var = (v==0 ? "U": 
                      (v==1 ? "V":
                      (v==2 && dim_==2 ? "P":
                      (v==2 && dim_==3 ? "W":
                      (v==3 ? "P":"X")))));
    ss<<"("<<i<<","<<j<<","<<k<<"):"<<var;
    return ss.str();
    }
    

}//namespace
