#include "HYMLS_ProjectedOperator.H"
#include "Epetra_MultiVector.h"
#include "Epetra_Map.h"

#include "HYMLS_Tools.H"
#include "HYMLS_DenseUtils.H"

namespace HYMLS 
  {
    
    //!constructor
    ProjectedOperator::ProjectedOperator(Teuchos::RCP<const Epetra_Operator> A, 
                      Teuchos::RCP<const Epetra_MultiVector> V,
                      bool useVorth) :
  A_(A), V_(V),
  useVorth_(useVorth),
  leftPrecond_(Teuchos::null),
  useTranspose_(false)
  {
  HYMLS_PROF3("ProjectedOperator", "Constructor");
  if (A_->OperatorRangeMap().SameAs(A_->OperatorDomainMap())==false)
    {
    Tools::Error("operator must be 'square'",__FILE__,__LINE__);
    }  
  if (A_->OperatorRangeMap().SameAs(V_->Map())==false)
    {
    Tools::Error("operator and vector space must have compatible maps",
        __FILE__,__LINE__);
    }  
  // re-allocated if Apply(Inverse)() is called with more vectors:
  tmpVector_ = Teuchos::rcp(new Epetra_MultiVector(V_->Map(),1));
  labelV_="V";
if (useVorth_) labelV_=labelV_+"_orth";
labelA_="("+std::string(A_->Label())+")";
labelT_="";
  }

    int ProjectedOperator::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
      {
      HYMLS_PROF3("ProjectedOperator", "Apply");
      
      if (useTranspose_) Tools::Error("not implemented!",__FILE__,__LINE__);
      
      if (X.NumVectors()!=tmpVector_->NumVectors())
        {
        tmpVector_ = Teuchos::rcp(new Epetra_MultiVector(V_->Map(),X.NumVectors()));
        }
      if (useVorth_)
        {
        //TODO: this may be optimized in several ways, e.g. keep temporary vectors
        // used inside ApplyOrth etc.
        CHECK_ZERO(DenseUtils::ApplyOrth(*V_,X,Y));
        CHECK_ZERO(A_->Apply(Y,*tmpVector_));
        CHECK_ZERO(DenseUtils::ApplyOrth(*V_,*tmpVector_,Y));
        }
      else
        {
        Tools::Error("not implemented!",__FILE__,__LINE__);
        }
      if (leftPrecond_!=Teuchos::null)
        {
        *tmpVector_=Y;
        CHECK_ZERO(leftPrecond_->ApplyInverse(*tmpVector_,Y));
        }
      return 0;
      }

    int ProjectedOperator::ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
      {
      HYMLS_PROF3("ProjectedOperator", "ApplyInverse");
      if (useTranspose_) Tools::Error("not implemented!",__FILE__,__LINE__);
      Tools::Error("not implemented",__FILE__,__LINE__);
      return -99;
      }

    
  }//namespace HYMLS
