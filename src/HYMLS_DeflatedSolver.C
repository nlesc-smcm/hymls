#include "HYMLS_DeflatedSolver.H"

#include "Epetra_MultiVector.h"
#include "Epetra_Map.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"

#include "Epetra_InvOperator.h"
#include "HYMLS_EpetraExt_ProductOperator.H"

#include "HYMLS_DenseUtils.H"
#include "HYMLS_MatrixUtils.H"

#include "AnasaziEpetraAdapter.hpp"
#include "AnasaziSVQBOrthoManager.hpp"

namespace HYMLS {

// constructor
DeflatedSolver::DeflatedSolver(Teuchos::RCP<const Epetra_RowMatrix> K,
  Teuchos::RCP<Epetra_Operator> P,
  Teuchos::RCP<Teuchos::ParameterList> params,
  int numRhs, bool validate)
  :
  BaseSolver(K, P, params, numRhs, validate),
  label_("HYMLS::DeflatedSolver"),
  deflationComputed_(false), numEigs_(0)
  {
  HYMLS_PROF3(label_, "Constructor");
  }

// destructor
DeflatedSolver::~DeflatedSolver()
  {
  HYMLS_PROF3(label_, "Destructor");
  }

// Sets all parameters for the solver
void DeflatedSolver::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& List)
  {
  setParameterList(List, validateParameters_);
  }

//! set solver parameters (the list is the "HYMLS"->"Solver" sublist)
void DeflatedSolver::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& params,
  bool validateParameters)
  {
  HYMLS_PROF3(label_, "SetParameterList");

  setMyParamList(params);

  numEigs_ = PL().get("Deflated Subspace Dimension", numEigs_);
  deflThres_ = PL().get("Deflation Threshold", 0.0);

  BaseSolver::setParameterList(params, validateParameters);
  }

//! get a list of valid parameters for this object
Teuchos::RCP<const Teuchos::ParameterList> DeflatedSolver::getValidParameters() const
  {
  HYMLS_PROF3(label_, "getValidParameterList");

  BaseSolver::getValidParameters();

  VPL().set("Deflated Subspace Dimension", 0,
    "Maximum number of eigenmodes to deflate");

  VPL().set("Deflation Threshold", 1.0e-3,
    "An eigenmode is deflated if the eigenvalue is within [-eps 0]");

  return validParams_;
  }

int DeflatedSolver::SetupDeflation(int numEigs)
  {

  if (numEigs>=0) numEigs_=numEigs;

  if (numEigs_ <= 0)
    return -1;
    
    
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
  CHECK_ZERO(BaseSolver::setProjectionVectors(deflationVectors_, massDeflationVectors_));

  deflationRhs_ = Teuchos::rcp(new Epetra_MultiVector(*deflationVectors_));
  Epetra_MultiVector tmp(*deflationVectors_);
  CHECK_ZERO(DenseUtils::ApplyOrth(*deflationVectors_, AV, tmp, massDeflationVectors_));
  int ret = BaseSolver::ApplyInverse(tmp, *deflationRhs_);

  CHECK_ZERO(DenseUtils::MatMul(*deflationVectors_, AV, *deflationMatrix_));

  ATV_ = Teuchos::rcp(new Epetra_MultiVector(*deflationVectors_));
  CHECK_ZERO(matrix_->Multiply(true, *deflationVectors_, *ATV_));

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

int DeflatedSolver::ApplyInverse(const Epetra_MultiVector& X,
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
  int dim1 = 0;

  Epetra_MultiVector Wb(OperatorRangeMap(), X.NumVectors());
  Epetra_MultiVector tmp(OperatorRangeMap(), X.NumVectors());

  if (deflationV_ != Teuchos::null)
    {
    CHECK_ZERO(DenseUtils::ApplyOrth(*deflationV_, X, Wb));
    CHECK_ZERO(DenseUtils::ApplyOrth(*deflationVectors_, Wb, tmp, massDeflationVectors_));
    }
  else
    {
    CHECK_ZERO(DenseUtils::ApplyOrth(*deflationVectors_, X, tmp, massDeflationVectors_));
    }

  ret = BaseSolver::ApplyInverse(tmp, Wb);

  DenseUtils::CheckOrthogonal(*deflationVectors_, Wb , __FILE__, __LINE__);

  Epetra_SerialDenseMatrix v(deflationMatrix_->N(), X.NumVectors());
  Epetra_SerialDenseMatrix w(deflationMatrix_->N(), X.NumVectors());

  Epetra_SerialDenseMatrix tmpMat;
  Epetra_SerialDenseMatrix w1(View, &w(0, 0), w.LDA(), dim0, X.NumVectors());
  CHECK_ZERO(DenseUtils::MatMul(*ATV_, Wb, tmpMat));
  w1 += tmpMat;

  if (deflationV_ != Teuchos::null)
    {
    dim1 = deflationV_->NumVectors();
    Epetra_SerialDenseMatrix w2(View, &w(dim0, 0), w.LDA(), dim1, X.NumVectors());
    CHECK_ZERO(DenseUtils::MatMul(*deflationV_, Wb, tmpMat));
    w2 += tmpMat;
    }
  
  Epetra_SerialDenseMatrix Vb(deflationVectors_->NumVectors(), X.NumVectors());
  CHECK_ZERO(DenseUtils::MatMul(*deflationVectors_, X, Vb));
  Vb.Scale(-1.0);
  w1 += Vb;

  CHECK_ZERO(deflationMatrixSolver_->SetVectors(v, w));
  CHECK_ZERO(deflationMatrixSolver_->Solve());

  Epetra_SerialComm comm;
  Epetra_LocalMap map1(dim0, 0, comm);
  Epetra_MultiVector v1(View, map1, v.A(), v.LDA(), X.NumVectors());

  if (deflationV_ != Teuchos::null)
    {
    Epetra_LocalMap map2(dim1, 0, comm);
    Epetra_MultiVector v2(View, map2, v.A()+dim0, v.LDA(), X.NumVectors());
    CHECK_ZERO(Wb.Multiply('N', 'N', 1.0, *AinvDeflationV_, v2, 1.0));
    }

  // Vperp = Wb - WA*v
  CHECK_ZERO(Wb.Multiply('N', 'N', 1.0, *deflationRhs_, v1, 1.0));
  CHECK_ZERO(Y.Multiply('N', 'N', -1.0, *deflationVectors_, v1, 0.0));

  // y = Vperp + Vv
  CHECK_ZERO(Y.Update(1.0, Wb, 1.0));

  return ret;
  }

Teuchos::RCP<Anasazi::Eigensolution<double, Epetra_MultiVector> > DeflatedSolver::EigsPrec(int numEigs) const
  {
  // If there is a null-space, deflate it.
  // If no NS and no additional vectors asked for -
  // nothing to be done.
  HYMLS_PROF(label_, "EigsPrec");

  Teuchos::RCP<Epetra_Operator> op, iop;
  Teuchos::RCP<Anasazi::Eigensolution<double, Epetra_MultiVector> > precEigs = Teuchos::null;
  Teuchos::RCP<const Epetra_Operator> op_array[2];

  op = precond_;

  ////////////////////////////////////////////////////////////////////////
  // Start by constructing the operator iop, which will be
  // [P N; N' 0] * [M 0; 0 I], with P the preconditioner, M the mass
  // matrix, N the null space and I the identity matrix.
  ////////////////////////////////////////////////////////////////////////
  if (massMatrix_ != Teuchos::null)
    {
    // construct the operator P\M
    EpetraExt::ProductOperator::EApplyMode mode[2];
    Teuchos::ETransp trans[2];

    op_array[1] = massMatrix_;
    mode[1] = EpetraExt::ProductOperator::APPLY_MODE_APPLY;
    trans[1] = Teuchos::NO_TRANS;
    op_array[0] = precond_;
    mode[0] = EpetraExt::ProductOperator::APPLY_MODE_APPLY_INVERSE;
    trans[0] = Teuchos::NO_TRANS;

    iop = Teuchos::rcp(new EpetraExt::ProductOperator(2, &op_array[0], trans, mode));
    deflationWithMassMatrix_ = true;
    }
  else
    {
    iop = Teuchos::rcp(new Epetra_InvOperator(op.get()));
    deflationWithMassMatrix_ = false;
    }

  ////////////////////////////////////////////////////
  // compute dominant eigenvalues of (P^{-1}, M).
  ////////////////////////////////////////////////////
  bool status = true;
  try {
    Tools::Out("Compute max eigs of inv(P)");
    precEigs = MatrixUtils::Eigs(iop, Teuchos::null, numEigs, 1.0e-8);
    } TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, status);
  if (!status) Tools::Fatal("caught an exception", __FILE__, __LINE__);

  // I think this should never occur:
  if (precEigs == Teuchos::null)
    {
    Tools::Error("null returned from Eigs routine?", __FILE__, __LINE__);
    }

  if (precEigs->numVecs < numEigs)
    {
    Tools::Warning("found "+Teuchos::toString(precEigs->numVecs)
      +" eigenpairs in EigsPrec(), while you requested "+Teuchos::toString(numEigs),
      __FILE__, __LINE__);
    }
  return precEigs;
  }

int DeflatedSolver::setProjectionVectors(Teuchos::RCP<const Epetra_MultiVector> V,
  Teuchos::RCP<const Epetra_MultiVector> W)
  {
  if (!deflationComputed_)
    {
    Tools::Error("You need to compute the deflated vectors first", __FILE__, __LINE__);
    }

  if (massMatrix_ != Teuchos::null || (W != Teuchos::null && V.get() != W.get()))
    {
    Tools::Error("DeflatedSolver with extra borders not yet implemented with "
      "mass matrix", __FILE__, __LINE__);
    }

  int n = deflationVectors_->NumVectors() + V->NumVectors();
  deflationMatrix_->Reshape(n, n);

  // Expand the deflation space with the border that was added here
  int dim0 = deflationVectors_->NumVectors();
  int dim1 = V->NumVectors();

  AinvDeflationV_ = Teuchos::rcp(new Epetra_MultiVector(*V));
  Epetra_MultiVector tmp(*V);
  CHECK_ZERO(DenseUtils::ApplyOrth(*deflationVectors_, *V, tmp, massDeflationVectors_));

  BaseSolver::setProjectionVectors(V_, W_);
  int ret = BaseSolver::ApplyInverse(tmp, *AinvDeflationV_);

  Epetra_SerialDenseMatrix tmpMat;
  Epetra_SerialDenseMatrix A12(View, &(*deflationMatrix_)(0, dim0),
    deflationMatrix_->LDA(), dim0, dim1);
  CHECK_ZERO(DenseUtils::MatMul(*ATV_, *AinvDeflationV_, tmpMat));
  A12.Scale(0.0);
  A12 += tmpMat;
  CHECK_ZERO(A12.Scale(-1.0));
  CHECK_ZERO(DenseUtils::MatMul(*deflationVectors_, *V, tmpMat));
  A12 += tmpMat;

  Epetra_SerialDenseMatrix A21(View, &(*deflationMatrix_)(dim0, 0),
    deflationMatrix_->LDA(), dim1, dim0);
  CHECK_ZERO(DenseUtils::MatMul(*V, *deflationRhs_, tmpMat));
  A21.Scale(0.0);
  A21 += tmpMat;
  CHECK_ZERO(A21.Scale(-1.0));
  CHECK_ZERO(DenseUtils::MatMul(*V, *deflationVectors_, tmpMat));
  A21 += tmpMat;

  Epetra_SerialDenseMatrix A22(View, &(*deflationMatrix_)(dim0, dim0),
    deflationMatrix_->LDA(), dim1, dim1);
  CHECK_ZERO(DenseUtils::MatMul(*V, *AinvDeflationV_, tmpMat));
  A22.Scale(0.0);
  A22 += tmpMat;
  CHECK_ZERO(A22.Scale(-1.0));

  deflationMatrixFactors_ = Teuchos::rcp(new Epetra_SerialDenseMatrix(*deflationMatrix_));
  CHECK_ZERO(deflationMatrixSolver_->SetMatrix(*deflationMatrixFactors_));
  deflationMatrixSolver_->FactorWithEquilibration(true);
  CHECK_ZERO(deflationMatrixSolver_->Factor());

  deflationV_ = V;
  return ret;
  }

void DeflatedSolver::setShift(double shiftA, double shiftB)
  {
  Tools::Warning("Shifted DeflatedSolver not yet implemented", __FILE__, __LINE__);
  }

  }//namespace HYMLS
