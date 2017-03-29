#include "HYMLS_ProjectedOperator.H"
#include "HYMLS_Macros.H"
#include "Epetra_MultiVector.h"
#include "Epetra_Map.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"

#include "HYMLS_Tools.H"
#include "HYMLS_DenseUtils.H"

namespace HYMLS 
  {
    
    //!constructor
    ProjectedOperator::ProjectedOperator(Teuchos::RCP<const Epetra_Operator> A,
                      Teuchos::RCP<const Epetra_MultiVector> V,
                      Teuchos::RCP<const Epetra_MultiVector> BV,
                      bool useVorth) :
  A_(A), V_(V), BV_(BV),
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

  if (BV_ == Teuchos::null)
    {
    BV_ = V_;
    }
  DenseUtils::CheckOrthogonal(*V_, *BV_, __FILE__, __LINE__, true);

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
        tmpVector_ = Teuchos::rcp(new Epetra_MultiVector(V_->Map(), X.NumVectors()));
        }
      if (useVorth_)
        {
        //TODO: this may be optimized in several ways, e.g. keep temporary vectors
        // used inside ApplyOrth etc.
        CHECK_ZERO(DenseUtils::ApplyOrth(*V_,X,Y, BV_, true));
        CHECK_ZERO(A_->Apply(Y,*tmpVector_));
        // CHECK_ZERO(A_->Apply(X,*tmpVector_));
        CHECK_ZERO(DenseUtils::ApplyOrth(*V_,*tmpVector_,Y, BV_));
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

      // Old implementation for preconditioner for JDQZ (see nonl.ps.gz)
      // Y=K^-1(I-ZH^-1Q'K^-1)X, H=Q'K^-1Z
      // With B it is
      //~ // Y=K^-1(I-BZH^-1Q'K^-1)X, H=Q'BK^-1BZ
      // Y=K^-1(I-BZH^-1Q'BK^-1)X, H=Q'K^-1BZ

      int m = V_->NumVectors();
      int k = X.NumVectors();

      if (k != tmpVector_->NumVectors())
        {
        tmpVector_ = Teuchos::rcp(new Epetra_MultiVector(V_->Map(), k));
        }

      if (HSolver_ == Teuchos::null)
        {
        Epetra_MultiVector tmpVectorm(V_->Map(), m);

        // H=Q'K^-1Z
        H_ = Teuchos::rcp(new Epetra_SerialDenseMatrix(m, m));
        CHECK_ZERO(A_->ApplyInverse(*BV_, tmpVectorm));
        CHECK_ZERO(DenseUtils::MatMul(*BV_, tmpVectorm, *H_));

        HSolver_ = Teuchos::rcp(new Epetra_SerialDenseSolver());
        CHECK_ZERO(HSolver_->SetMatrix(*H_));
        CHECK_ZERO(HSolver_->Factor());
        }

      //Y=K^-1X
      CHECK_ZERO(A_->ApplyInverse(X, Y));

      Epetra_SerialDenseMatrix C(m, k);
      Epetra_SerialDenseMatrix D(m, k);

      //C=Q'K^-1X
      CHECK_ZERO(DenseUtils::MatMul(*BV_, Y, C));

      //D=H^-1Q'K^-1X
      CHECK_ZERO(HSolver_->SetVectors(D, C));
      HSolver_->Solve();

      Teuchos::RCP<Epetra_MultiVector> HVY = DenseUtils::CreateView(D);

      //T=X-ZH^-1Q'K^-1X
      *tmpVector_ = X;
      CHECK_ZERO(tmpVector_->Multiply('N', 'N', -1.0, *BV_, *HVY, 1.0));

      //Y=K^-1(X-ZH^-1Q'K^-1X)
      CHECK_ZERO(A_->ApplyInverse(*tmpVector_, Y));
      return 0;
      }

  }//namespace HYMLS
