#include "HYMLS_BorderedDeflatedSolver.hpp"

#include "HYMLS_BaseSolver.hpp"
#include "HYMLS_Macros.hpp"
#include "HYMLS_Tools.hpp"
#include "HYMLS_DenseUtils.hpp"

#include "Epetra_LocalMap.h"
#include "Epetra_Map.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Operator.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_SerialComm.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"

#include "Ifpack_Preconditioner.h"

#include "BelosSolverManager.hpp"

#include "AnasaziTypes.hpp"
#include "AnasaziEpetraAdapter.hpp"
#include "AnasaziSVQBOrthoManager.hpp"

namespace Teuchos { class ParameterList; }

namespace HYMLS {

// constructor
BorderedDeflatedSolver::BorderedDeflatedSolver(Teuchos::RCP<const Epetra_Operator> K,
  Teuchos::RCP<Epetra_Operator> P,
  Teuchos::RCP<Teuchos::ParameterList> params,
  int numRhs, bool validate)
  :
  BaseSolver(K, P, params, numRhs, validate),
  DeflatedSolver(K, P, params, numRhs, validate),
  BorderedSolver(K, P, params, numRhs, validate),
  label_("BorderedDeflatedSolver")
  {
  HYMLS_PROF3(label_,"Constructor");
  }

// destructor
BorderedDeflatedSolver::~BorderedDeflatedSolver()
  {
  HYMLS_PROF3(label_,"Destructor");
  }

// Sets all parameters for the solver
void BorderedDeflatedSolver::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params)
  {
  setParameterList(params, validateParameters_);
  }

// Sets all parameters for the solver
void BorderedDeflatedSolver::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params,
  bool validateParameters)
  {
  HYMLS_PROF3(label_,"SetParameterList");

  setMyParamList(params);

  DeflatedSolver::setParameterList(params, validateParameters);
  BorderedSolver::setParameterList(params, validateParameters);
  }

// Sets all parameters for the solver
Teuchos::RCP<const Teuchos::ParameterList> BorderedDeflatedSolver::getValidParameters() const
  {
  HYMLS_PROF3(label_, "getValidParameterList");

  DeflatedSolver::getValidParameters();
  BorderedSolver::getValidParameters();
  return validParams_;
  }

int BorderedDeflatedSolver::SetupDeflation()
  {
  if (numEigs_ <= 0)
    return -1;

  // TODO: Add functionality to disable the border, rather than having to recompute everything.
  Teuchos::RCP<Ifpack_Preconditioner> ifpack_precond =
    Teuchos::rcp_dynamic_cast<Ifpack_Preconditioner>(precond_);
  if (ifpack_precond == Teuchos::null)
    {
    Tools::Error("Only Ifpack_Preconditioners are supported since Compute() has to be called.", __FILE__, __LINE__);
    }

  CHECK_ZERO(BorderedSolver::SetBorder(Teuchos::null, Teuchos::null));
  CHECK_ZERO(ifpack_precond->Compute());

  precEigs_ = EigsPrec(numEigs_);
  numEigs_ = precEigs_->numVecs;

  if (numEigs_ == 0)
    {
    return 1;
    }

  if (precEigs_->Evecs == Teuchos::null)
    {
    Tools::Error("no eigenvectors have been returned.", __FILE__, __LINE__);
    }

  if (precEigs_->Espace == Teuchos::null)
    {
    Tools::Error("no eigenvector basis has been returned.", __FILE__, __LINE__);
    }

  // FIXME: This should always be orthogonal according to Anasazi documentation
  // but it is not.
  deflationVectors_ = precEigs_->Espace;
  Teuchos::RCP<Anasazi::SVQBOrthoManager<double, Epetra_MultiVector, Epetra_Operator> > ortho = Teuchos::rcp(new Anasazi::SVQBOrthoManager<double, Epetra_MultiVector, Epetra_Operator>(massMatrix_));
  ortho->normalize(*deflationVectors_);

  int n = deflationVectors_->NumVectors();
  deflationMatrix_ = Teuchos::rcp(new Epetra_SerialDenseMatrix(n, n));

  //TODO: Do we need this?
  massDeflationVectors_ = Teuchos::rcp(new Epetra_MultiVector(*deflationVectors_));
  if (massMatrix_ != Teuchos::null)
    {
    massMatrix_->Apply(*deflationVectors_, *massDeflationVectors_);
    }

  Epetra_MultiVector AV(*deflationVectors_);
  // TODO: Filter A for zero vectors (vectors that the prec captures)
  CHECK_ZERO(BaseSolver::ApplyMatrix(*deflationVectors_, AV));
  CHECK_ZERO(BorderedSolver::SetBorder(deflationVectors_, massDeflationVectors_));
  CHECK_ZERO(ifpack_precond->Compute());

  deflationRhs_ = Teuchos::rcp(new Epetra_MultiVector(*deflationVectors_));
  Epetra_MultiVector tmp(*deflationVectors_);
  CHECK_ZERO(DenseUtils::ApplyOrth(*deflationVectors_, AV, tmp, massDeflationVectors_));
  int ret = BorderedSolver::ApplyInverse(tmp, *deflationRhs_);

  CHECK_ZERO(DenseUtils::MatMul(*deflationVectors_, AV, *deflationMatrix_));

  ATV_ = Teuchos::rcp(new Epetra_MultiVector(*deflationVectors_));
  CHECK_ZERO(ApplyMatrixTranspose(*deflationVectors_, *ATV_));

  Epetra_SerialDenseMatrix tmpMat(n, n);
  CHECK_ZERO(DenseUtils::MatMul(*ATV_, *deflationRhs_, tmpMat));
  CHECK_ZERO(tmpMat.Scale(-1.0));

  *deflationMatrix_ += tmpMat;
  deflationMatrixFactors_ = Teuchos::rcp(new Epetra_SerialDenseMatrix(*deflationMatrix_));

  // TODO: Use SVD
  deflationMatrixSolver_ = Teuchos::rcp(new Epetra_SerialDenseSolver());
  CHECK_ZERO(deflationMatrixSolver_->SetMatrix(*deflationMatrixFactors_));
  deflationMatrixSolver_->FactorWithEquilibration(true);
  CHECK_ZERO(deflationMatrixSolver_->Factor());

  deflationComputed_ = true;

  return ret;
  }

// Applies the solver to vector X, returns the result in Y.
int BorderedDeflatedSolver::ApplyInverse(const Epetra_MultiVector& X,
  Epetra_MultiVector& Y) const
  {
  if (numEigs_ == 0)
    {
    return BaseSolver::ApplyInverse(X, Y);
    }

  if (!deflationComputed_)
    {
    Tools::Error("You need to compute the deflated vectors first", __FILE__, __LINE__);
    }

  if (massMatrix_ != Teuchos::null && !deflationWithMassMatrix_)
    {
    // for e.g. Navier-Stokes it is important to set the mass matrix before
    // calling EigsPrec, by calling SetMassMatrix() in both the solver
    // and the preconditioner.
    Tools::Error("EigsPrec() called without mass matrix", __FILE__, __LINE__);
    }

  int ret = 0;

  int dim0 = deflationVectors_->NumVectors();

  Epetra_MultiVector Wb(OperatorRangeMap(), X.NumVectors());
  Epetra_MultiVector tmp(OperatorRangeMap(), X.NumVectors());

  CHECK_ZERO(DenseUtils::ApplyOrth(*deflationVectors_, X, tmp, massDeflationVectors_));

  ret = BorderedSolver::ApplyInverse(tmp, Wb);

  DenseUtils::CheckOrthogonal(*deflationVectors_, Wb , __FILE__, __LINE__, false,
    belosSolverPtr_->achievedTol() * 100);

  Epetra_SerialDenseMatrix v(deflationMatrix_->N(), X.NumVectors());
  Epetra_SerialDenseMatrix w(deflationMatrix_->N(), X.NumVectors());

  Epetra_SerialDenseMatrix tmpMat;
  Epetra_SerialDenseMatrix w1(View, &w(0, 0), w.LDA(), dim0, X.NumVectors());
  CHECK_ZERO(DenseUtils::MatMul(*ATV_, Wb, tmpMat));
  w1 += tmpMat;
  
  Epetra_SerialDenseMatrix Vb(deflationVectors_->NumVectors(), X.NumVectors());
  CHECK_ZERO(DenseUtils::MatMul(*deflationVectors_, X, Vb));
  Vb.Scale(-1.0);
  w1 += Vb;

  CHECK_ZERO(deflationMatrixSolver_->SetVectors(v, w));
  CHECK_ZERO(deflationMatrixSolver_->Solve());

  Epetra_SerialComm comm;
  Epetra_LocalMap map1(dim0, 0, comm);
  Epetra_MultiVector v1(View, map1, v.A(), v.LDA(), X.NumVectors());

  // Vperp = Wb - WA*v
  CHECK_ZERO(Wb.Multiply('N', 'N', 1.0, *deflationRhs_, v1, 1.0));
  CHECK_ZERO(Y.Multiply('N', 'N', -1.0, *deflationVectors_, v1, 0.0));

  // y = Vperp + Vv
  CHECK_ZERO(Y.Update(1.0, Wb, 1.0));

  return ret;
  }

  }//namespace HYMLS
