#include "HYMLS_DeflatedSolver.H"

#include "Epetra_MultiVector.h"
#include "Epetra_Map.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"

#include "HYMLS_DenseUtils.H"

namespace HYMLS {

// constructor
DeflatedSolver::DeflatedSolver(Teuchos::RCP<const Epetra_RowMatrix> K,
  Teuchos::RCP<Epetra_Operator> P,
  Teuchos::RCP<Teuchos::ParameterList> params,
  int numRhs, bool validate)
  :
  BaseSolver(K, P, params, numRhs, validate),
  label_("HYMLS::DeflatedSolver"),
  deflationComputed_(false)
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
    Tools::Error("no eigenvectors have been returned.",__FILE__,__LINE__);
    }

  if (precEigs_->Espace == Teuchos::null)
    {
    Tools::Error("no eigenvector basis has been returned.",__FILE__,__LINE__);
    }

  deflationVectors_ = precEigs_->Espace;

  int n = deflationVectors_->NumVectors();
  deflationMatrix_ = Teuchos::rcp(new Epetra_SerialDenseMatrix(n, n));

  Epetra_MultiVector AV(*deflationVectors_);
  // TODO: Filter A for zero vectors (vectors that the prec captures)
  CHECK_ZERO(ApplyMatrix(*deflationVectors_, AV));
  CHECK_ZERO(setProjectionVectors(deflationVectors_));

  deflationRhs_ = Teuchos::rcp(new Epetra_MultiVector(*deflationVectors_));
  Epetra_MultiVector tmp(*deflationVectors_);
  CHECK_ZERO(DenseUtils::ApplyOrth(*deflationVectors_, AV, tmp));
  CHECK_ZERO(DenseUtils::MatMul(*deflationVectors_, tmp, *deflationMatrix_));
  std::cout << *deflationMatrix_;
  BaseSolver::ApplyInverse(tmp, *deflationRhs_);

  CHECK_ZERO(DenseUtils::MatMul(*deflationVectors_, AV, *deflationMatrix_));

  Epetra_MultiVector AWA(*deflationVectors_);
  CHECK_ZERO(ApplyMatrix(*deflationRhs_, AWA));

  Epetra_SerialDenseMatrix tmpMat(n, n);
  CHECK_ZERO(DenseUtils::MatMul(*deflationVectors_, AWA, tmpMat));
  tmpMat.Scale(-1.0);

  *deflationMatrix_ += tmpMat;

  // TODO: Use SVD
  deflationMatrixSolver_ = Teuchos::rcp(new Epetra_SerialDenseSolver());
  CHECK_ZERO(deflationMatrixSolver_->SetMatrix(*deflationMatrix_));
  CHECK_ZERO(deflationMatrixSolver_->Factor());

  deflationComputed_ = true;

  return 0;
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

  // TODO: Need mass matrix handling

  Epetra_MultiVector Wb(OperatorRangeMap(), X.NumVectors());
  Epetra_MultiVector tmp(OperatorRangeMap(), X.NumVectors());
  CHECK_ZERO(DenseUtils::ApplyOrth(*deflationVectors_, X, tmp));
  CHECK_ZERO(BaseSolver::ApplyInverse(tmp, Wb));

  Epetra_MultiVector &AWb = tmp;
  CHECK_ZERO(ApplyMatrix(Wb, AWb));

  Epetra_SerialDenseMatrix tmpMat(deflationVectors_->NumVectors(), X.NumVectors());
  CHECK_ZERO(DenseUtils::MatMul(*deflationVectors_, AWb, tmpMat));
  tmpMat.Scale(-1.0);

  Epetra_SerialDenseMatrix Vb(deflationVectors_->NumVectors(), X.NumVectors());
  CHECK_ZERO(DenseUtils::MatMul(*deflationVectors_, X, Vb));

  tmpMat += Vb;

  Epetra_SerialDenseMatrix v(deflationVectors_->NumVectors(), X.NumVectors());
  CHECK_ZERO(deflationMatrixSolver_->SetVectors(v, tmpMat));

  deflationMatrixSolver_->Solve();

  Teuchos::RCP<Epetra_MultiVector> MVv = DenseUtils::CreateView(v);

  // Vperp = Wb - WA*v
  CHECK_ZERO(Wb.Multiply('N', 'N', -1.0, *deflationRhs_, *MVv, 1.0));
  CHECK_ZERO(Y.Multiply('N', 'N', 1.0, *deflationVectors_, *MVv, 0.0));
  
  // y = Vperp + Vv
  CHECK_ZERO(Y.Update(1.0, Wb, 1.0));
  return 0;
  }

  }//namespace HYMLS
