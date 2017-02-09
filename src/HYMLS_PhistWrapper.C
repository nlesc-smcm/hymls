#include "phist_config.h"
#include <mpi.h>

// Include this before other phist headers
#include "HYMLS_PhistWrapper.H"

#include "HYMLS_Solver.H"
#include "HYMLS_Macros.H"
#include "HYMLS_MatrixUtils.H"
#include "HYMLS_Macros.H"
#include "HYMLS_Tester.H"
#include "Epetra_MultiVector.h"
#include "Epetra_MultiVector.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_Import.h"

#include "phist_kernels.h"
#include "phist_jadaCorrectionSolver.h"
#include "phist_macros.h"
#include "phist_enums.h"
#include "phist_orthog.h"

#include <cstdlib>
#include <vector>

#include "phist_gen_d.h"


namespace HYMLS {

void phist_precon_apply(_ST_ alpha, const void* vP, 
        TYPE(const_mvec_ptr) vX, _ST_ beta,  TYPE(mvec_ptr) vY, int* iflag)
{
  PHIST_CAST_PTR_FROM_VOID(const HYMLS::Preconditioner,P,vP,*iflag);
  PHIST_CAST_PTR_FROM_VOID(const Epetra_MultiVector,X,vX,*iflag);
  PHIST_CAST_PTR_FROM_VOID(Epetra_MultiVector,Y,vY,*iflag);
  PHIST_CHK_IERR(*iflag=(alpha!=1.0 || beta!=0.0)?PHIST_NOT_IMPLEMENTED:0,*iflag);
  PHIST_CHK_IERR(*iflag=P->ApplyInverse(*X,*Y),*iflag); 
}
 //!
 void phist_precon_apply_shifted(_ST_ alpha, const void* P, _ST_ const sigma[],
        TYPE(const_mvec_ptr) X, _ST_ beta,  TYPE(mvec_ptr) Y, int* iflag)
{
  // our preconditioner does nothing special ith the shifts, so call the regular apply instead
  PHIST_CHK_IERR(phist_precon_apply(alpha,P,X,beta,Y,iflag),*iflag);
}

void phist_precon_update(void const* P, void* aux, _ST_ sigma,
                   TYPE(const_mvec_ptr) Vkern,
                   TYPE(const_mvec_ptr) BVkern,
                   int* iflag)
{
  PHIST_CAST_PTR_FROM_VOID(HYMLS::Preconditioner,Prec,aux,*iflag);
  // add the deflation space obtained from Jacobi-Davidson as a border to the
  // preconditioner
  if (Vkern==NULL) return;
  Teuchos::RCP<const Epetra_MultiVector> V = Teuchos::rcp((const Epetra_MultiVector*)Vkern,false);
  Teuchos::RCP<const Epetra_MultiVector> BV = V;
  if (BVkern!=NULL) BV=Teuchos::rcp((const Epetra_MultiVector*)BVkern,false);

  Teuchos::RCP<Epetra_MultiVector> PinvV = Teuchos::rcp(new Epetra_MultiVector(*V));
  Teuchos::RCP<Epetra_SerialDenseMatrix> C=Teuchos::rcp(new Epetra_SerialDenseMatrix(BV->NumVectors(),V->NumVectors()));
  
  Epetra_MultiVector* C_tmp=NULL;
  PHIST_CHK_IERR(SUBR(sdMat_create)((void**)(&C_tmp),BV->NumVectors(),V->NumVectors(),(phist_const_comm_ptr)(&(V->Comm())),iflag),*iflag);
  Teuchos::RCP<Epetra_MultiVector> _C=Teuchos::rcp(C_tmp,true);

  PHIST_CHK_IERR(*iflag=Prec->setBorder(Teuchos::null,Teuchos::null,Teuchos::null),*iflag);
  PHIST_CHK_IERR(*iflag=Prec->ApplyInverse(*V,*PinvV),*iflag);
  PHIST_CHK_IERR(SUBR(mvecT_times_mvec)(1.0,BV.get(),PinvV.get(),0.0,C_tmp,iflag),*iflag);
  for (int j=0; j<C->N(); j++)
    for (int i=0; i<C->M(); i++)
    {
      (*C)[j][i] = (*C_tmp)[j][i];
    }
  
  PHIST_CHK_IERR(*iflag=Prec->setBorder(PinvV,BV,C),*iflag);
  *iflag=0;
}

}

/*
void HYMLS_jadaCorrectionSolver_run1(void* vme,
  void const* vA_op, void const* vB_op, 
  TYPE(const_mvec_ptr) Qtil, TYPE(const_mvec_ptr) BQtil,
  double sigma_r, double sigma_i,
  TYPE(const_mvec_ptr) res,
  const double tol, int maxIter,
  TYPE(mvec_ptr) t,
  int robust,
  int *iflag)
{
  PHIST_ENTER_FCN(__FUNCTION__);
  *iflag = 0;
  TYPE(const_linearOp_ptr) A_op=(TYPE(const_linearOp_ptr))vA_op;
  TYPE(const_linearOp_ptr) B_op=(TYPE(const_linearOp_ptr))vB_op;
  phist_hymls_wrapper* me=(phist_hymls_wrapper*)vme;

  PHIST_CHK_IERR(*iflag = (maxIter <= 0) ? -1 : 0, *iflag);
  
  // note: we should probably solve a system with two rhs but with a real preconditioner here,
  // as is done in phist
  if (sigma_i!=0.0) 
  {
    HYMLS::Tools::Warning("jadaCorrectionSolver with complex shifts not implemented",
                          __FILE__, __LINE__);
    *iflag=-99; // not implemented
    return;
  }

  Teuchos::RCP<HYMLS::Solver> solver = me->solver;
  if (solver==Teuchos::null){ *iflag=PHIST_BAD_CAST; return;}
  bool status=true;
  try {
  solver->SetTolerance(tol);
  solver->setShift(1.0, -sigma_r);
  } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr,status);
  if (!status)
  {
    *iflag=PHIST_CAUGHT_EXCEPTION; 
    return;
  }
   
  Teuchos::RCP<const Epetra_MultiVector> BQ;
  const Epetra_MultiVector *Q_ptr = (const Epetra_MultiVector *)Qtil;
  const Epetra_MultiVector *BQ_ptr = (const Epetra_MultiVector *)BQtil;
  const Epetra_MultiVector *r_ptr = (const Epetra_MultiVector *)res;
  Epetra_MultiVector *t_ptr = (Epetra_MultiVector *)t;

  if (BQtil == NULL)
    {
    if (B_op == NULL)
      {
      BQ_ptr = Q_ptr;
      }
    else
      {
      BQ = Teuchos::rcp(new Epetra_MultiVector(*Q_ptr));
      BQ_ptr = BQ.get();
      PHIST_CHK_IERR(B_op->apply(1.0, B_op->A, Qtil, 0.0,
          (TYPE(mvec_ptr))BQ_ptr, iflag), *iflag);
      }
    }

  const Epetra_BlockMap &map = Q_ptr->Map();
  const Epetra_BlockMap &map0 = solver->OperatorRangeMap();
  if (!(map.SameAs(map0)))
  {
    // System is only the v-part, not the full system, so import the vectors
    // on the v-part to the full system so we can use HYMLS on it.
    Epetra_Import import0(map0, Q_ptr->Map());

    Epetra_MultiVector vec0(map0, Q_ptr->NumVectors());
    vec0.PutScalar(0.0);
    CHECK_ZERO(vec0.Import(*Q_ptr, import0, Insert));

    Epetra_MultiVector vec1(map0, BQ_ptr->NumVectors());
    vec1.PutScalar(0.0);
    CHECK_ZERO(vec1.Import(*Q_ptr, import0, Insert));
    if (me->borderedSolver)
    {
      solver->addBorder(Teuchos::rcp<const Epetra_MultiVector>(&vec1, false),
        Teuchos::rcp<const Epetra_MultiVector>(&vec0, false));
      solver->SetupDeflation();
    }
    else
    {
      CHECK_ZERO(solver->setProjectionVectors(Teuchos::rcp<const Epetra_MultiVector>(&vec0, false)));
    }
  }
  else
  {
    if (me->borderedSolver)
    {
      solver->addBorder(Teuchos::rcp<const Epetra_MultiVector>(BQ_ptr, false),
        Teuchos::rcp<const Epetra_MultiVector>(Q_ptr, false));
    }
    else
    {
      CHECK_ZERO(solver->setProjectionVectors(Teuchos::rcp<const Epetra_MultiVector>(Q_ptr, false)));
    }
  }

  if (!(map.SameAs(map0)))
  {
    // v-part only, so import back an forth between the full system like above
    const Epetra_Map &map1 = solver->OperatorRangeMap();
    Epetra_Import import1(map1, (r_ptr)->Map());
    Epetra_MultiVector vec1(map1, (r_ptr)->NumVectors());
    vec1.PutScalar(0.0);
    CHECK_ZERO(vec1.Import(*r_ptr, import1, Insert));

    const Epetra_Map &map2 = solver->OperatorDomainMap();
    Epetra_Import import2(map2, t_ptr->Map());
    Epetra_MultiVector vec2(map2, t_ptr->NumVectors());
    vec2.PutScalar(0.0);
    CHECK_ZERO(vec2.Import(*t_ptr, import2, Insert));

    solver->ApplyInverse(vec1, vec2);

    Epetra_Import invImport2(t_ptr->Map(), map2);
    CHECK_ZERO(t_ptr->Import(vec2, invImport2, Insert));
  }
  else
  {
    // This is allowed to not converge, so don't do CHECK_ZERO
    solver->ApplyInverse(*r_ptr, *t_ptr);
  }

  if (solver->getNonconstParameterList()->sublist("Problem").get("Equations", "") == "Stokes-C")
  HYMLS_TEST("jada",isDivFree(*(const Epetra_CrsMatrix *)A_op->A, *t_ptr, 4, 3),__FILE__,__LINE__);

  // normalize result vectors, TODO: should be done in updateSol/pgmres?
  _MT_ tmp;
  PHIST_CHK_IERR(phist_Dmvec_normalize(t, &tmp, iflag), *iflag);
}

void HYMLS_jadaCorrectionSolver_run(void* vme,
  void const* vA_op, void const* vB_op, 
  TYPE(const_mvec_ptr) Qtil, TYPE(const_mvec_ptr) BQtil,
  const double *sigma_r, const double *sigma_i, 
  TYPE(const_mvec_ptr) res, const int resIndex[],
  const double *tol, int maxIter,
  TYPE(mvec_ptr) t,
  int robust, int abortAfterFirstConvergedInBlock,
  int *iflag)
{
  PHIST_ENTER_FCN(__FUNCTION__);

  int numSys;
  PHIST_CHK_IERR(SUBR(mvec_num_vectors)(t,&numSys,iflag),*iflag);
  if (numSys > 2)
  {
    HYMLS::Tools::Warning("jadaCorrectionSolver with more than RHS not implemented",
                          __FILE__, __LINE__);
    *iflag = -99; // not implemented
    return;
  }

  if (numSys > 1)
  {
    if (sigma_i[0] != 0.0 && sigma_i[1] != 0.0) 
    {
      HYMLS::Tools::Warning("jadaCorrectionSolver with complex shifts not implemented",
                            __FILE__, __LINE__);
      *iflag = -99; // not implemented
      return;
    }

    if (std::abs(sigma_r[0] - sigma_r[1]) > 1e-14) 
    {
      HYMLS::Tools::Warning("jadaCorrectionSolver with unequal shifts not implemented",
                            __FILE__, __LINE__);
      *iflag = -99; // not implemented
      return;
    }
  }

  PHIST_CHK_IERR(HYMLS_jadaCorrectionSolver_run1(
                   vme, vA_op, vB_op, Qtil, BQtil,
                   *sigma_r, *sigma_i, res, *tol,
                   maxIter, t, robust, iflag), *iflag);
}

// we need to replace the residual computation in jdqr right now by re-implementing
// it and passing it via the jadaOpts. This is because we want to enforce the Div-
// constraint in Navier-Stokes here.
//
// Details: The B_op has a pointer to a phist_hymls_wrapper
// as its A member, from which we can get both the mass matrix and the HYMLS solver.
// In the future we should probably have the operators involved (A,B,Precond) have
// the velocity map as range and domain map and always act in the div-free space.
void HYMLS_computeResidual(void* customSolver, void const* vB_op, TYPE(mvec_ptr) r_ptr,
                           TYPE(const_mvec_ptr) Au_ptr, TYPE(const_mvec_ptr) u_ptr,
                           TYPE(mvec_ptr) rtil_ptr, TYPE(const_mvec_ptr) Qv,
                           TYPE(mvec_ptr) tmp, TYPE(sdMat_ptr) Theta,
                           TYPE(sdMat_ptr) atil, TYPE(sdMat_ptr) *atilv, _MT_ *resid,
                           int nv, int nconv, int* iflag)
{
  // r=Au-theta*u; ([r_r, r_i] = [Au_r, Au_i] - [u_r, u_i]*Theta in the real case with 
  // complex theta)

  double nrm[2];

  TYPE(const_linearOp_ptr) B_op=(TYPE(const_linearOp_ptr))vB_op;

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
    phist_hymls_wrapper* me= (phist_hymls_wrapper*)customSolver;
    if (me!=NULL)
    {
      solver = me->solver;
    }
    else
    {
      PHIST_SOUT(PHIST_ERROR,"in HYMLS overloaded computeResidual function, customSolver is "
                             "NULL (should point to the customSolver struct phist_hymls_wrapper)\n");
      *iflag=PHIST_BAD_CAST;
      return;
    }

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

#ifdef HYMLS_TESTING
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

#ifdef HYMLS_DEBUGGING
  HYMLS::MatrixUtils::Dump(*(const Epetra_MultiVector *)Au_ptr, "Au_ptr.txt");
  HYMLS::MatrixUtils::Dump(vec1, "Au.txt");
#endif

  double *norm1 = new double[nv];
  Epetra_MultiVector Au_full(map, ((const Epetra_MultiVector *)Au_ptr)->NumVectors());
  Au_full.PutScalar(0.0);
  CHECK_ZERO(Au_full.Import(*(const Epetra_MultiVector *)Au_ptr, import, Insert));
  vec1.Update(-1.0, Au_full, 1.0);
  vec1.Norm2(norm1);

#ifdef HYMLS_DEBUGGING
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
  PHIST_CHK_IERR(SUBR(mvec_times_sdMat)(-1.0,(phist_Dconst_mvec_ptr)(&vec2),Theta,1.0,(phist_Dmvec_ptr)(&explicit_resid),iflag),*iflag);

  double *res_norm = new double[nv];
  explicit_resid.Norm2(res_norm);

  res_norm[0] = res_norm[0]*res_norm[0];
  for (int i = 1; i < nv; i++)
    res_norm[0] += res_norm[i]*res_norm[i];
  HYMLS::Tools::Out("Explicit residual is "+Teuchos::toString(sqrt(res_norm[0])));

  delete[] res_norm;
#endif
}

*/
