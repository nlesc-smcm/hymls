#ifndef HYMLS_PHIST_WRAPPER_H_H
#define HYMLS_PHIST_WRAPPER_H_H

// Get the HYMLS implementation of SCOREP_USER_REGION
#include "phist_macros.h"
#include "phist_kernels.h"
#include "phist_operator.h"
#include "HYMLS_Macros.H"

#include "Epetra_Operator.h"
#include "Epetra_MultiVector.h"
#include "Epetra_SerialDenseMatrix.h"
#include "HYMLS_Preconditioner.H"
#include "phist_operator.h"
#include "phist_enums.h"
#include "phist_gen_d.h"
#include "phist_void_aliases.h"

#include "phist_gen_d.h"

// alternative preconditioning options
#include "Ifpack_Preconditioner.h"
#include "ml_MultiLevelPreconditioner.h"

namespace HYMLS {

//! \name functions used to wrap a HYMLS::Preconditioner as a phist_DlinearOp
//!@{

template<typename PREC>
class PhistPreconTraits
{
public:
 //!
 static void apply(_ST_ alpha, const void* P, 
        TYPE(const_mvec_ptr) X, _ST_ beta,  TYPE(mvec_ptr) Y, int* iflag)
  {
    PHIST_CHK_IERR(*iflag=PHIST_NOT_IMPLEMENTED,*iflag);
  }
 //!
 static void apply_shifted(_ST_ alpha, const void* P, _ST_ const sigma[],
        TYPE(const_mvec_ptr) X, _ST_ beta,  TYPE(mvec_ptr) Y, int* iflag)
  {
    PHIST_CHK_IERR(*iflag=PHIST_NOT_IMPLEMENTED,*iflag);
  }
 //!
 static void update(void const* P, void* aux, _ST_ sigma,
                    TYPE(const_mvec_ptr) Vkern,
                    TYPE(const_mvec_ptr) BVkern,
                    int* iflag)
  {
    PHIST_CHK_IERR(*iflag=PHIST_NOT_IMPLEMENTED,*iflag);
  }
};

//@}



template<>
void PhistPreconTraits<HYMLS::Preconditioner>::apply(_ST_ alpha, const void* vP, 
        TYPE(const_mvec_ptr) vX, _ST_ beta,  TYPE(mvec_ptr) vY, int* iflag)
{
  PHIST_CAST_PTR_FROM_VOID(const HYMLS::Preconditioner,P,vP,*iflag);
  PHIST_CAST_PTR_FROM_VOID(const Epetra_MultiVector,X,vX,*iflag);
  PHIST_CAST_PTR_FROM_VOID(Epetra_MultiVector,Y,vY,*iflag);
  PHIST_CHK_IERR(*iflag=(alpha!=1.0 || beta!=0.0)?PHIST_NOT_IMPLEMENTED:0,*iflag);
  PHIST_CHK_IERR(*iflag=P->ApplyInverse(*X,*Y),*iflag); 
}
 //!
template<>   
void PhistPreconTraits<HYMLS::Preconditioner>::apply_shifted(_ST_ alpha, const void* P, _ST_ const sigma[],
        TYPE(const_mvec_ptr) X, _ST_ beta,  TYPE(mvec_ptr) Y, int* iflag)
{
  // our preconditioner does nothing special ith the shifts, so call the regular apply instead
  PHIST_CHK_IERR(apply(alpha,P,X,beta,Y,iflag),*iflag);
}

template<>   
void PhistPreconTraits<HYMLS::Preconditioner>::update(void const* P, void* aux, _ST_ sigma,
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


///////////////////////////////
// ML interface              //
///////////////////////////////

template<>
void PhistPreconTraits<ML_Epetra::MultiLevelPreconditioner>::apply(_ST_ alpha, const void* vP, 
        TYPE(const_mvec_ptr) vX, _ST_ beta,  TYPE(mvec_ptr) vY, int* iflag)
{
  PHIST_CAST_PTR_FROM_VOID(const ML_Epetra::MultiLevelPreconditioner,P,vP,*iflag);
  PHIST_CAST_PTR_FROM_VOID(const Epetra_MultiVector,X,vX,*iflag);
  PHIST_CAST_PTR_FROM_VOID(Epetra_MultiVector,Y,vY,*iflag);
  PHIST_CHK_IERR(*iflag=(alpha!=1.0 || beta!=0.0)?PHIST_NOT_IMPLEMENTED:0,*iflag);
  PHIST_CHK_IERR(*iflag=P->ApplyInverse(*X,*Y),*iflag); 
}
 //!
template<>   
void PhistPreconTraits<ML_Epetra::MultiLevelPreconditioner>::apply_shifted(_ST_ alpha, const void* P, _ST_ const sigma[],
        TYPE(const_mvec_ptr) X, _ST_ beta,  TYPE(mvec_ptr) Y, int* iflag)
{
  // our preconditioner does nothing special ith the shifts, so call the regular apply instead
  PHIST_CHK_IERR(apply(alpha,P,X,beta,Y,iflag),*iflag);
}

template<>   
void PhistPreconTraits<ML_Epetra::MultiLevelPreconditioner>::update(void const* P, void* aux, _ST_ sigma,
                   TYPE(const_mvec_ptr) Vkern,
                   TYPE(const_mvec_ptr) BVkern,
                   int* iflag)
{
  *iflag=0;
}

///////////////////////////////////////////
// general Ifpack interface              //
///////////////////////////////////////////

template<>
void PhistPreconTraits<Ifpack_Preconditioner>::apply(_ST_ alpha, const void* vP, 
        TYPE(const_mvec_ptr) vX, _ST_ beta,  TYPE(mvec_ptr) vY, int* iflag)
{
  PHIST_CAST_PTR_FROM_VOID(const Ifpack_Preconditioner,P,vP,*iflag);
  PHIST_CAST_PTR_FROM_VOID(const Epetra_MultiVector,X,vX,*iflag);
  PHIST_CAST_PTR_FROM_VOID(Epetra_MultiVector,Y,vY,*iflag);
  PHIST_CHK_IERR(*iflag=(alpha!=1.0 || beta!=0.0)?PHIST_NOT_IMPLEMENTED:0,*iflag);
  PHIST_CHK_IERR(*iflag=P->ApplyInverse(*X,*Y),*iflag); 
}
 //!
template<>   
void PhistPreconTraits<Ifpack_Preconditioner>::apply_shifted(_ST_ alpha, const void* P, _ST_ const sigma[],
        TYPE(const_mvec_ptr) X, _ST_ beta,  TYPE(mvec_ptr) Y, int* iflag)
{
  // our preconditioner does nothing special ith the shifts, so call the regular apply instead
  PHIST_CHK_IERR(apply(alpha,P,X,beta,Y,iflag),*iflag);
}

template<>   
void PhistPreconTraits<Ifpack_Preconditioner>::update(void const* P, void* aux, _ST_ sigma,
                   TYPE(const_mvec_ptr) Vkern,
                   TYPE(const_mvec_ptr) BVkern,
                   int* iflag)
{
  *iflag=0;
}

}


#endif
