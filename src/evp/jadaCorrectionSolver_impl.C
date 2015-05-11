#include "phist_config.h"
#include <mpi.h>

// Include this before other phist headers
#include "jadaCorrectionSolver_impl.H"

#include "HYMLS_Solver.H"
#include "HYMLS_Macros.H"
#include "HYMLS_MatrixUtils.H"
#include "HYMLS_Macros.H"
#include "HYMLS_Tester.H"
#include "Epetra_MultiVector.h"
#include "Epetra_Import.h"

#include "phist_jadaCorrectionSolver.h"
#include "phist_macros.h"
#include "phist_enums.h"
#include "phist_orthog.h"

#include <cstdlib>
#include <vector>

#include "phist_gen_d.h"

//! create a jadaCorrectionSolver object
void SUBR(jadaCorrectionSolver_create)(TYPE(jadaCorrectionSolver_ptr) *me, int blockedGMRESBlockDim, const_map_ptr_t map, 
        linSolv_t method, int blockedGMRESMaxBase, bool useMINRES, int *iflag)
{
  PHIST_ENTER_FCN(__FUNCTION__);
  *iflag = 0;
  if (method==GMRES)
  {
    PHIST_CHK_IERR( *iflag = (blockedGMRESBlockDim <= 0) ? -1 : 0, *iflag);

    *me = new TYPE(jadaCorrectionSolver);
    (*me)->gmresBlockDim_ = blockedGMRESBlockDim;
    (*me)->blockedGMRESstates_  = NULL;
    (*me)->useMINRES_ = useMINRES;
  }
  else if (method==CARP_CG)
  {
    *iflag=-99;
  }
  else
  {
    PHIST_SOUT(PHIST_ERROR, "method %d (%s) not implemented",(int)method, linSolv2str(method));
    *iflag=-99;
  }
}

//! delete a jadaCorrectionSolver object
void SUBR(jadaCorrectionSolver_delete)(TYPE(jadaCorrectionSolver_ptr) me, int *iflag)
{
  PHIST_ENTER_FCN(__FUNCTION__);
  *iflag = 0;

  delete me;
}


//! calculate approximate solutions to given set of jacobi-davidson correction equations
//!
//! arguments:
//! jdCorrSolver    the jadaCorrectionSolver object
//! A_op            matrix A passed to jadaOp_create
//! B_op            matrix B passed to jadaOp_create
//! Qtil            projection vectors V passed to jadaOp_create
//! BQtil           projection vectors BV passed to jadaOp_create
//! sigma           (pos.!) shifts, -sigma[i], i in {1, ..., nvec} is passed to the jadaOp
//! res             JD residuals, e.g. rhs of the correction equations
//! resIndex        if not NULL, specifies permutation of the residual array to avoid unnecessary copying in the jada-algorithm
//! tol             desired accuracy (gmres residual tolerance) of the individual systems
//! maxIter         maximal number of iterations after which individial systems should be aborted
//! t               returns approximate solution vectors
//! iflag            a value > 0 indicates the number of systems that have not converged to the desired tolerance
void SUBR(jadaCorrectionSolver_run)(TYPE(jadaCorrectionSolver_ptr) me,
                                    TYPE(const_op_ptr)    A_op,     TYPE(const_op_ptr)    B_op, 
                                    TYPE(const_mvec_ptr)  Qtil,     TYPE(const_mvec_ptr)  BQtil,
                                    const _ST_            sigma[],  TYPE(const_mvec_ptr)  res,      const int resIndex[], 
                                    const _MT_            tol[],    int                   maxIter,
                                    TYPE(mvec_ptr)        t,
                                    bool useIMGS,                   bool abortAfterFirstConvergedInBlock,
                                    int *                 iflag)
{
  PHIST_ENTER_FCN(__FUNCTION__);
  *iflag = 0;

  PHIST_CHK_IERR(*iflag = (maxIter <= 0) ? -1 : 0, *iflag);

  Dmvec_ptr_t t_i = NULL;
  Dmvec_ptr_t r_i = NULL;

  // total number of systems to solve
  int totalNumSys;
  phist_Dmvec_num_vectors(t, &totalNumSys, iflag);
  TEUCHOS_TEST_FOR_EXCEPTION(*iflag != 0, std::runtime_error,
    "jadaCorrectionSolver_run: phist_Dmvec_num_vectors returned nonzero error code "+Teuchos::toString(*iflag));

  // We can only solve one system at the moment
  for (int i = 0; i < 1; i++)
  {
    Teuchos::RCP<HYMLS::Solver> solver = ((extended_Dop_t const *)A_op)->solver;
    solver->SetTolerance(tol[i]);
    solver->setShift(1.0, -sigma[i]);
    //~ solver->setNullSpace(Teuchos::rcp<const Epetra_MultiVector>((const Epetra_MultiVector *)Qtil, false));
    //~ solver->SetupDeflation();

    const Epetra_BlockMap &map = ((const Epetra_MultiVector *)Qtil)->Map();
    const Epetra_BlockMap &map0 = solver->OperatorRangeMap();
    if (!(map.SameAs(map0)))
    {
      // System is only the v-part, not the full system, so import the vectors
      // on the v-part to the full system so we can use HYMLS on it.
      const Epetra_Map &map0 = solver->OperatorRangeMap();
      Epetra_Import import0(map0, ((const Epetra_MultiVector *)Qtil)->Map());
      Epetra_MultiVector vec0(map0, ((const Epetra_MultiVector *)Qtil)->NumVectors());
      vec0.PutScalar(0.0);
      CHECK_ZERO(vec0.Import(*(const Epetra_MultiVector *)Qtil, import0, Insert));
      CHECK_ZERO(solver->setProjectionVectors(Teuchos::rcp<const Epetra_MultiVector>(&vec0, false)));
    }
    else
    {
      CHECK_ZERO(solver->setProjectionVectors(Teuchos::rcp<const Epetra_MultiVector>((const Epetra_MultiVector *)Qtil, false)));
    }

    int ind = (resIndex == NULL ? i : resIndex[i]);
    phist_Dmvec_view_block(t, &t_i, i, i+totalNumSys-1, iflag);
    TEUCHOS_TEST_FOR_EXCEPTION(*iflag != 0, std::runtime_error,
      "jadaCorrectionSolver_run: phist_Dmvec_view_block returned nonzero error code "+Teuchos::toString(*iflag));

    phist_Dmvec_view_block((Dmvec_ptr_t)res, &r_i, ind, ind+totalNumSys-1, iflag);
    TEUCHOS_TEST_FOR_EXCEPTION(*iflag != 0, std::runtime_error,
      "jadaCorrectionSolver_run: phist_Dmvec_view_block returned nonzero error code "+Teuchos::toString(*iflag));

    if (!(map.SameAs(map0)))
    {
      // v-part only, so import back an forth between the full system like above
      const Epetra_Map &map1 = solver->OperatorRangeMap();
      Epetra_Import import1(map1, ((const Epetra_MultiVector *)r_i)->Map());
      Epetra_MultiVector vec1(map1, ((const Epetra_MultiVector *)r_i)->NumVectors());
      vec1.PutScalar(0.0);
      CHECK_ZERO(vec1.Import(*(const Epetra_MultiVector *)r_i, import1, Insert));

      const Epetra_Map &map2 = solver->OperatorDomainMap();
      Epetra_Import import2(map2, ((Epetra_MultiVector *)t_i)->Map());
      Epetra_MultiVector vec2(map2, ((Epetra_MultiVector *)t_i)->NumVectors());
      vec2.PutScalar(0.0);
      CHECK_ZERO(vec2.Import(*(Epetra_MultiVector *)t_i, import2, Insert));

      solver->ApplyInverse(vec1, vec2);

      Epetra_Import invImport2(((Epetra_MultiVector *)t_i)->Map(), map2);
      CHECK_ZERO(((Epetra_MultiVector *)t_i)->Import(vec2, invImport2, Insert));
    }
    else
    {
      // This is allowed to not converge, so don't do CHECK_ZERO
      solver->ApplyInverse(*(const Epetra_MultiVector *)r_i, *(Epetra_MultiVector *)t_i);
    }

    if (solver->getNonconstParameterList()->sublist("Problem").get("Equations", "") == "Stokes-C")
      HYMLS_TEST("jada",isDivFree(*(const Epetra_CrsMatrix *)A_op->A, *(const Epetra_MultiVector *)t_i, 4, 3),__FILE__,__LINE__);

  }

  SUBR(mvec_delete)(r_i,iflag);
  SUBR(mvec_delete)(t_i,iflag);

  // normalize result vectors, TODO: should be done in updateSol/pgmres?
  _MT_ tmp[totalNumSys];
  phist_Dmvec_normalize(t, tmp, iflag);
  TEUCHOS_TEST_FOR_EXCEPTION(*iflag != 0, std::runtime_error,
    "jadaCorrectionSolver_run: phist_Dmvec_normalize returned nonzero error code "+Teuchos::toString(*iflag));
}

void SUBR(computeResidual)(TYPE(const_op_ptr) B_op, TYPE(mvec_ptr) r_ptr,
        TYPE(mvec_ptr) Au_ptr, TYPE(mvec_ptr) u_ptr, TYPE(mvec_ptr) rtil_ptr,
        TYPE(mvec_ptr) Qv, TYPE(mvec_ptr) tmp, TYPE(sdMat_ptr) Theta,
        TYPE(sdMat_ptr) atil, TYPE(sdMat_ptr) *atilv, _MT_ *resid,
        int nv, int nconv, int* iflag)
{
  // r=Au-theta*u; ([r_r, r_i] = [Au_r, Au_i] - [u_r, u_i]*Theta in the real case with 
  // complex theta)

  double nrm[2];

  // pointer to temporary storage for B*Q*atil if B is defined
  TYPE(mvec_ptr) tmp_ptr=NULL;

  // first: r=Au
  PHIST_CHK_IERR(SUBR(mvec_add_mvec)(1.0,Au_ptr,0.0,r_ptr,iflag),*iflag);

  if (B_op != NULL)
  {
    // rt = B * U
    PHIST_CHK_IERR(B_op->apply(1.0,B_op->A,u_ptr,0.0,rtil_ptr,iflag),*iflag);

    // update r = r - rt*Theta
    PHIST_CHK_IERR(SUBR(mvec_times_sdMat)(-1.0,rtil_ptr,Theta,1.0,r_ptr,iflag),*iflag);
  }
  else
  {
    // update r = r - u*Theta
    PHIST_CHK_IERR(SUBR(mvec_times_sdMat)(-1.0,u_ptr,Theta,1.0,r_ptr,iflag),*iflag);
  }

  Teuchos::RCP<HYMLS::Solver> solver = Teuchos::null;
  if (B_op != NULL)
  {
    solver = ((extended_Dop_t const *)B_op)->solver;

    // We only want to import vectors etc if we don't solve the full system but
    // only the v-part
    if (!(((Epetra_MultiVector *)rtil_ptr)->Map().SameAs(solver->OperatorRangeMap())))
    {
      ((Epetra_MultiVector *)rtil_ptr)->PutScalar(0.0);

      TEUCHOS_ASSERT(solver->OperatorRangeMap().SameAs(solver->OperatorDomainMap()));

      // make some space for the full vectors
      const Epetra_Map &map = solver->OperatorRangeMap();
      Epetra_Import import(map, ((const Epetra_MultiVector *)r_ptr)->Map());
      Epetra_MultiVector vec1(map, ((const Epetra_MultiVector *)r_ptr)->NumVectors());
      vec1.PutScalar(0.0);
      CHECK_ZERO(vec1.Import(*(const Epetra_MultiVector *)r_ptr, import, Insert));

      Epetra_MultiVector vec2(map, ((Epetra_MultiVector *)r_ptr)->NumVectors());

      // get the p-part
      solver->ApplyPrec(vec1, vec2);

      // put zeros in the v part
      vec2.Import(*(const Epetra_MultiVector *)rtil_ptr, import, Insert);

      // multiply by A, but since the v-part is zero this is D*p
      solver->ApplyMatrix(vec2, vec1);

      // now put back the result in rtil
      Epetra_Import invImport(((Epetra_MultiVector *)rtil_ptr)->Map(), map);
      CHECK_ZERO(((Epetra_MultiVector *)rtil_ptr)->Import(vec1, invImport, Insert));

      // recompute r by adding rtil
      PHIST_CHK_IERR(SUBR(mvec_add_mvec)(-1.0,rtil_ptr,1.0,r_ptr,iflag),*iflag);
    }
  }

  // set rtil=r
  PHIST_CHK_IERR(SUBR(mvec_add_mvec)(1.0,r_ptr,0.0,rtil_ptr,iflag),*iflag);

  // project out already converged eigenvectors
  // TODO - we could use our orthog routine here instead
  if (nconv>0)
  {
    // view next ~a, a temporary vector to compute ~a=Q'*r
    PHIST_CHK_IERR(SUBR(sdMat_view_block)(atil,atilv,0,nconv-1,0,nv-1,iflag),*iflag);

    //atil = Q'*r;
    PHIST_CHK_IERR(SUBR(mvecT_times_mvec)(1.0,Qv,r_ptr,0.0,*atilv,iflag),*iflag);

    if (B_op != NULL)
    {
      PHIST_CHK_IERR(SUBR(mvec_view_block)(tmp,&tmp_ptr,0,nv-1,iflag),*iflag);

      // tmp = Q*atil
      PHIST_CHK_IERR(SUBR(mvec_times_sdMat)(1.0,Qv,*atilv,0.0,tmp_ptr,iflag),*iflag);

      //rtil = r-B*Q*atil;
      PHIST_CHK_IERR(B_op->apply(-1.0,B_op->A,tmp_ptr,1.0,rtil_ptr,iflag),*iflag);

      PHIST_CHK_IERR(SUBR(mvec_delete)(tmp_ptr,iflag),*iflag);
      tmp_ptr = NULL;
    }
    else
    {
      //rtil = r-Q*atil;
      PHIST_CHK_IERR(SUBR(mvec_times_sdMat)(-1.0,Qv,*atilv,1.0,rtil_ptr,iflag),*iflag);
    }
  }

  //nrm=norm(rtil);
  // real case with complex r: ||v+iw||=sqrt((v+iw).'*(v-iw))=sqrt(v'v+w'w).
  // in the complex case we pass in a 're-interpret cast' of nrm as complex,
  // which should be fine (imaginary part will be 0).
  nrm[1]=0.0;
  PHIST_CHK_IERR(SUBR(mvec_dot_mvec)(rtil_ptr,rtil_ptr,nrm,iflag),*iflag);
  *resid=sqrt(nrm[0]+nrm[1]);

#ifdef TESTING
  // TODO: We can't do anything here if we don't have B
  if (solver == Teuchos::null)
    return;

  // Should work in both the v-part and full system solvers
  TEUCHOS_ASSERT(solver->OperatorRangeMap().SameAs(solver->OperatorDomainMap()));

  // make some space for the full vectors
  const Epetra_Map &map = solver->OperatorRangeMap();
  Epetra_Import import(map, ((const Epetra_MultiVector *)r_ptr)->Map());
  Epetra_MultiVector vec1(map, ((const Epetra_MultiVector *)r_ptr)->NumVectors());
  vec1.PutScalar(0.0);
  CHECK_ZERO(vec1.Import(*(const Epetra_MultiVector *)r_ptr, import, Insert));

  Epetra_MultiVector vec2(map, ((Epetra_MultiVector *)r_ptr)->NumVectors());

  // get the p-part
  solver->ApplyPrec(vec1, vec2);

  // put zeros in the v part
  PHIST_CHK_IERR(SUBR(mvec_view_block)(tmp,&tmp_ptr,0,nv-1,iflag),*iflag);
  ((Epetra_MultiVector *)tmp_ptr)->PutScalar(0.0);
  vec2.Import(*(const Epetra_MultiVector *)tmp_ptr, import, Insert);
  PHIST_CHK_IERR(SUBR(mvec_delete)(tmp_ptr,iflag),*iflag);

  // Compute the explicit residual
  Epetra_MultiVector explicit_resid(map, ((const Epetra_MultiVector *)u_ptr)->NumVectors());
  explicit_resid.PutScalar(0.0);
  CHECK_ZERO(explicit_resid.Import(*(const Epetra_MultiVector *)u_ptr, import, Insert));

  // Test if Au is really A*u
  solver->ApplyMatrix(explicit_resid, vec1);

#ifdef DEBUGGING
  HYMLS::MatrixUtils::Dump(*(const Epetra_MultiVector *)Au_ptr, "Au_ptr.txt");
  HYMLS::MatrixUtils::Dump(vec1, "Au.txt");
#endif

  double *norm1 = new double[nv];
  Epetra_MultiVector Au_full(map, ((const Epetra_MultiVector *)Au_ptr)->NumVectors());
  Au_full.PutScalar(0.0);
  CHECK_ZERO(Au_full.Import(*(const Epetra_MultiVector *)Au_ptr, import, Insert));
  vec1.Update(-1.0, Au_full, 1.0);
  vec1.Norm2(norm1);

#ifdef DEBUGGING
  HYMLS::MatrixUtils::Dump(vec1, "Au_update.txt");

  for (int i = 0; i < nv; i++)
    HYMLS::Tools::Out("||Au_ptr - A*u_"+Teuchos::toString(i)+"||="+Teuchos::toString(norm1[i]));
#endif

  norm1[0] = norm1[0]*norm1[0];
  for (int i = 1; i < nv; i++)
    norm1[0] += norm1[i]*norm1[i];

  TEUCHOS_ASSERT(norm1[0] < HYMLS_SMALL_ENTRY * vec1.MyLength());

  delete[] norm1;

  explicit_resid.Update(-1.0, vec2, 1.0);

  // A*u
  solver->ApplyMatrix(explicit_resid, vec1);

  // B*u
  solver->ApplyMass(explicit_resid, vec2);

  // r = A*u
  explicit_resid = vec1;

  // r = A*u - B*u*Theta
  PHIST_CHK_IERR(SUBR(mvec_times_sdMat)(-1.0,(Dconst_mvec_ptr_t)(&vec2),Theta,1.0,(Dmvec_ptr_t)(&explicit_resid),iflag),*iflag);

  double *res_norm = new double[nv];
  explicit_resid.Norm2(res_norm);

  res_norm[0] = res_norm[0]*res_norm[0];
  for (int i = 1; i < nv; i++)
    res_norm[0] += res_norm[i]*res_norm[i];
  HYMLS::Tools::Out("Explicit residual is "+Teuchos::toString(sqrt(res_norm[0])));

  delete[] res_norm;
#endif
}

