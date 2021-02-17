//#define BLOCK_IMPLEMENTATION 1
#include "HYMLS_BaseSolver.hpp"

#include "HYMLS_config.h"

#include "HYMLS_Tools.hpp"
#include "HYMLS_MatrixUtils.hpp"
#include "HYMLS_DenseUtils.hpp"
#include "HYMLS_ProjectedOperator.hpp"
#include "HYMLS_ShiftedOperator.hpp"

#include "Epetra_Comm.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_MultiVector.h"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Utils.hpp"

#include "BelosTypes.hpp"

#include "BelosLinearProblem.hpp"
#include "BelosSolverManager.hpp"
#include "BelosEpetraAdapter.hpp"

#include "BelosBlockCGSolMgr.hpp"
#include "BelosBlockGmresSolMgr.hpp"
//#include "BelosPCPGSolMgr.hpp"

#include "Teuchos_StandardParameterEntryValidators.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

using BelosGmresType = Belos::BlockGmresSolMgr<
  double, Epetra_MultiVector, Epetra_Operator>;
using BelosCGType = Belos::BlockCGSolMgr<
  double, Epetra_MultiVector, Epetra_Operator>;

namespace HYMLS {

// constructor
BaseSolver::BaseSolver(Teuchos::RCP<const Epetra_Operator> K,
  Teuchos::RCP<Epetra_Operator> P,
  Teuchos::RCP<Teuchos::ParameterList> params,
  int numRhs, bool validate)
  :
  PLA("Solver"), comm_(Teuchos::rcp(K->Comm().Clone())),
  matrix_(K), operator_(K), precond_(P),
  shiftA_(1.0), shiftB_(0.0),
  massMatrix_(Teuchos::null),
  V_(Teuchos::null), W_(Teuchos::null),
  useTranspose_(false), normInf_(-1.0), numIter_(0),
  label_("HYMLS::BaseSolver"),
  lor_default_("Right")
  {
  HYMLS_PROF3(label_,"Constructor");
  setParameterList(params, validate && validateParameters_);

  belosProblemPtr_ = Teuchos::rcp(new BelosProblemType());
  belosProblemPtr_->setOperator(operator_);
  
  this->SetPrecond(precond_);

  Teuchos::ParameterList& belosList = PL().sublist("Iterative Solver");

//  belosList.set("Output Style",::Belos::Brief);
  belosList.set("Output Style",1);
  belosList.set("Verbosity",::Belos::Errors+::Belos::Warnings
    +::Belos::IterationDetails
    +::Belos::StatusTestDetails
    +::Belos::FinalSummary
    +::Belos::TimingDetails);

  belosList.set("Output Stream",Tools::out().getOStream());

  // create the solver
  Teuchos::RCP<Teuchos::ParameterList> belosListPtr=rcp(&belosList,false);
  if (solverType_=="CG")
    {
    belosSolverPtr_ = rcp(new BelosCGType(belosProblemPtr_,belosListPtr));
    }
  else if (solverType_=="PCG")
    {
    Tools::Error("NOT IMPLEMENTED!",__FILE__,__LINE__);
/*
  belosSolverPtr_ = Teuchos::rcp(new 
  ::Belos::PCPGSolMgr<ST,MV,OP>
  (belosProblemPtr_,belosListPtr));
*/
    }
  else if (solverType_=="GMRES")
    {
    belosSolverPtr_ = Teuchos::rcp(new BelosGmresType(
        belosProblemPtr_,belosListPtr));
    }
  else
    {
    Tools::Error("Currently only 'GMRES' is supported as 'Belos Solver'",__FILE__,__LINE__);
    }
  }


// destructor
BaseSolver::~BaseSolver()
  {
  HYMLS_PROF3(label_,"Destructor");
  }

void BaseSolver::SetOperator(Teuchos::RCP<const Epetra_Operator> A)
  {
  HYMLS_PROF3(label_, "SetOperator");
  matrix_ = A;
  if (shiftB_ != 0.0 || shiftA_ != 1.0)
    {
    Tools::Warning("SetOperator called while operator used is shifted."
      "Discarding shifts.", __FILE__, __LINE__);
    shiftB_ = 0.0;
    shiftA_ = 1.0;
    }
  operator_ = matrix_;
  belosProblemPtr_->setOperator(operator_);
  }

void BaseSolver::SetTolerance(double tol)
  {
  Teuchos::ParameterList& belosList = PL().sublist("Iterative Solver");
  belosList.set("Convergence Tolerance", tol);
  Teuchos::RCP<Teuchos::ParameterList> belosListPtr = rcp(&belosList, false);
  belosSolverPtr_->setParameters(belosListPtr);
  }

void BaseSolver::SetPrecond(Teuchos::RCP<Epetra_Operator> P)
  {
  HYMLS_PROF3(label_,"SetPrecond");
  precond_=P;
  if (precond_==Teuchos::null) return;

  belosPrecPtr_=Teuchos::rcp(new BelosPrecType(precond_));
  std::string lor = PL().get("Left or Right Preconditioning",lor_default_);
  if (lor=="Left")
    {
    belosProblemPtr_->setLeftPrec(belosPrecPtr_);
    }
  else if (lor=="Right")
    {
    belosProblemPtr_->setRightPrec(belosPrecPtr_);
    }
  else if (lor=="None")
    {
    // no preconditioning
    }
  }

void BaseSolver::SetMassMatrix(Teuchos::RCP<const Epetra_RowMatrix> mass)
  {
  HYMLS_PROF3(label_, "SetMassMatrix");
  if (mass == Teuchos::null)
    return;

  if (mass->OperatorRangeMap().SameAs(matrix_->OperatorRangeMap()))
    {
    massMatrix_ = mass;
    }
  else
    {
    Tools::Error("Mass matrix must have same row map as solver",
      __FILE__, __LINE__);
    }

  if (shiftB_ != 0.0 || shiftA_ != 1.0)
    {
    Tools::Warning("SetMassMatrix called while solving shifted system. "
      "Discarding shifts.", __FILE__, __LINE__);
    shiftB_ = 0.0;
    shiftA_ = 1.0;
    }

  operator_ = matrix_;
  }

int BaseSolver::Apply(const Epetra_MultiVector& X,
  Epetra_MultiVector& Y) const
  {
  return operator_->Apply(X,Y);
  }

int BaseSolver::ApplyMatrix(const Epetra_MultiVector& X,
  Epetra_MultiVector& Y) const
  {
  return matrix_->Apply(X,Y);
  }

int BaseSolver::ApplyMatrixTranspose(const Epetra_MultiVector& X,
  Epetra_MultiVector& Y) const
  {
  Teuchos::RCP<Epetra_Operator> op = Teuchos::rcp_const_cast<Epetra_Operator>(matrix_);
  CHECK_ZERO(op->SetUseTranspose(!op->UseTranspose()));
  int ierr = matrix_->Apply(X,Y);
  CHECK_ZERO(op->SetUseTranspose(!op->UseTranspose()));
  return ierr;
  }

int BaseSolver::ApplyPrec(const Epetra_MultiVector& X,
  Epetra_MultiVector& Y) const
  {
  return precond_->ApplyInverse(X,Y);
  }

int BaseSolver::ApplyMass(const Epetra_MultiVector& X,
  Epetra_MultiVector& Y) const
  {
  return massMatrix_->Apply(X,Y);
  }

int BaseSolver::SetUseTranspose(bool UseTranspose)
  {
  useTranspose_=false; // not implemented.
  return -1;
  }

const Epetra_Map & BaseSolver::OperatorDomainMap() const
  {
  return matrix_->OperatorDomainMap();
  }

const Epetra_Map & BaseSolver::OperatorRangeMap() const
  {
  return matrix_->OperatorRangeMap();
  }

void BaseSolver::SetShift(double shiftA, double shiftB)
  {
  shiftA_ = shiftA;
  shiftB_ = shiftB;
  operator_ = Teuchos::rcp(new ShiftedOperator(matrix_, massMatrix_, shiftA_, shiftB_));
  belosProblemPtr_->setOperator(operator_);
  }

// Sets all parameters for the solver
void BaseSolver::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& List)
  {
  setParameterList(List, validateParameters_);
  }

// Sets all parameters for the solver
void BaseSolver::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& List,
  bool validateParameters)
  {
  HYMLS_PROF3(label_,"SetParameterList");

  setMyParamList(List);

  //TODO: use validators everywhere

  solverType_= PL().get("Krylov Method","GMRES");
  startVec_=PL().get("Initial Vector","Random");
  PL().get("Left or Right Preconditioning",lor_default_);

  if (belosSolverPtr_!=Teuchos::null)
    {
    Teuchos::RCP<Teuchos::ParameterList> belosListPtr = Teuchos::rcp(&(PL().sublist("Iterative Solver")),false);
    belosSolverPtr_->setParameters(belosListPtr);
    }

  // this is the place where we check for
  // valid parameters for the iterative solver
  if (validateParameters)
    {
    getValidParameters();
    PL().validateParameters(VPL());
    }
  HYMLS_DEBUG(PL());
  }

// Sets all parameters for the solver
Teuchos::RCP<const Teuchos::ParameterList> BaseSolver::getValidParameters() const
  {
  if (validParams_!=Teuchos::null) return validParams_;
  HYMLS_PROF3(label_,"getValidParameterList");
    
  //TODO: use validators everywhere

  Teuchos::RCP<Teuchos::StringToIntegralParameterEntryValidator<int> >
    solverValidator = Teuchos::rcp(
      new Teuchos::StringToIntegralParameterEntryValidator<int>(
        Teuchos::tuple<std::string>( "GMRES", "CG" ),"Krylov Method"));
  VPL().set("Krylov Method", "GMRES",
    "Type of Krylov method to be used", solverValidator);

  Teuchos::RCP<Teuchos::StringToIntegralParameterEntryValidator<int> >
    x0Validator = Teuchos::rcp(
      new Teuchos::StringToIntegralParameterEntryValidator<int>(
        Teuchos::tuple<std::string>( "Zero", "Random", "Previous" ),"Initial Vector"));                                        

  VPL().set("Initial Vector","Zero",
    "How to construct the starting vector for the Krylov series", x0Validator);

  Teuchos::RCP<Teuchos::StringToIntegralParameterEntryValidator<int> >
    lorValidator = Teuchos::rcp(
      new Teuchos::StringToIntegralParameterEntryValidator<int>(
        Teuchos::tuple<std::string>( "Left", "Right"),"Left or Right Preconditioning"));
        
  VPL().set("Left or Right Preconditioning", lor_default_,
    "wether to do left (P\\Ax=P\\b) or right (AP\\(Px)=b) preconditioning",
    lorValidator);

  // Belos parameters should be specified in this list:
  VPL().sublist("Iterative Solver",false,
    "Parameter list for the Krylov method (passed to Belos)").disableRecursiveValidation();
  return validParams_;
  }

int BaseSolver::setProjectionVectors(Teuchos::RCP<const Epetra_MultiVector> V,
  Teuchos::RCP<const Epetra_MultiVector> W)
  {
  V_ = V;
  W_ = W;

  Aorth_=Teuchos::rcp(new ProjectedOperator(operator_, V, W, true));
  if (precond_ != Teuchos::null)
    {
    Teuchos::RCP<ProjectedOperator> newPrec = Teuchos::rcp(new ProjectedOperator(precond_, V, W, true));

    belosPrecPtr_=Teuchos::rcp(new BelosPrecType(newPrec));
    std::string lor = PL().get("Left or Right Preconditioning",lor_default_);
    if (lor=="Left")
      {
      belosProblemPtr_->setLeftPrec(belosPrecPtr_);
      }
    else if (lor=="Right")
      {
      belosProblemPtr_->setRightPrec(belosPrecPtr_);
      }
    }
  belosProblemPtr_->setOperator(Aorth_);
  return 0;
  }

// Applies the preconditioner to vector X, returns the result in Y.
int BaseSolver::ApplyInverse(const Epetra_MultiVector& B,
  Epetra_MultiVector& X) const
  {
  HYMLS_PROF(label_,"ApplyInverse");
  int ierr = 0;
#ifdef HYMLS_TESTING
  if (X.NumVectors()!=B.NumVectors())
    {
    Tools::Error("different number of input and output vectors",__FILE__,__LINE__);
    }
#endif

  Teuchos::RCP<const Epetra_MultiVector> belosRhs = Teuchos::rcp(&B, false);
  Teuchos::RCP<Epetra_MultiVector> belosSol = Teuchos::rcp(&X, false);

  if (startVec_=="Random")
    {
#ifdef HYMLS_DEBUGGING
    int seed=42;
    MatrixUtils::Random(*belosSol, seed);
#else
    MatrixUtils::Random(*belosSol);
#endif
    }
  else if (startVec_=="Zero")
    {
    // set initial vector to 0
    CHECK_ZERO(belosSol->PutScalar(0.0));
    }

  // Make the initial guess orthogonal to the V_ space
  if (V_ != Teuchos::null)
    {
    Teuchos::RCP<Epetra_MultiVector> tmp = Teuchos::rcp(new Epetra_MultiVector(X));
    CHECK_ZERO(DenseUtils::ApplyOrth(*V_, *belosSol, *tmp, W_));
    *belosSol = *tmp;
    }

  CHECK_TRUE(belosProblemPtr_->setProblem(belosSol, belosRhs));

  ::Belos::ReturnType ret = ::Belos::Unconverged;
  bool status = true;
  try {
    ret = belosSolverPtr_->solve();
    } TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, status);
  if (!status) Tools::Warning("caught an exception", __FILE__, __LINE__);

  numIter_ = belosSolverPtr_->getNumIters();

  if (ret != ::Belos::Converged)
    {
    HYMLS::Tools::Warning("Belos returned "+::Belos::convertReturnTypeToString(ret)+"'!",__FILE__,__LINE__);    
#ifdef HYMLS_TESTING
    Teuchos::RCP<const Epetra_CrsMatrix> Acrs =
      Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);
    MatrixUtils::Dump(*Acrs,"FailedMatrix.txt");
    MatrixUtils::Dump(B,"FailedRhs.txt");
#endif
    ierr = -1;
    }
  if (comm_->MyPID()==0)
    {
    Tools::Out("++++++++++++++++++++++++++++++++++++++++++++++++");
    Tools::Out("+ Number of iterations: "+Teuchos::toString(numIter_));
    Tools::Out("++++++++++++++++++++++++++++++++++++++++++++++++");
    Tools::Out("");
    }

#ifdef HYMLS_TESTING

  Tools::Out("explicit residual test");
  Tools::out() << "we were solving (a*A*x+b*B)*x=rhs\n"
               << "   with " << X.NumVectors() << " rhs\n"
               << "        a = " << shiftA_ << "\n"
               << "        b = " << shiftB_ << "\n";
  if (massMatrix_ == Teuchos::null)
    Tools::out() << "        B = I\n";

  // compute explicit residual
  int dim = PL("Problem").get<int>("Dimension");
  int dof = PL("Problem").get<int>("Degrees of Freedom");

  Epetra_BlockMap const &map = X.Map();
  Epetra_MultiVector resid(map, X.NumVectors());
  CHECK_ZERO(matrix_->Apply(X,resid));
  if (shiftB_ != 0.0)
    {
    Epetra_MultiVector Bx = X;
    if (massMatrix_ != Teuchos::null)
      {
      CHECK_ZERO(massMatrix_->Apply(X, Bx));
      }
    CHECK_ZERO(resid.Update(shiftB_, Bx, shiftA_));
    }
  else if (shiftA_ != 1.0)
    {
    CHECK_ZERO(resid.Scale(shiftA_));
    }
  CHECK_ZERO(resid.Update(1.0, B, -1.0));

  double *resNorm  = new double[resid.NumVectors()];
  double *resNormV = new double[resid.NumVectors()];
  double *resNormP = new double[resid.NumVectors()];
  double *rhsNorm  = new double[resid.NumVectors()];
  B.Norm2(rhsNorm);
  resid.Norm2(resNorm);

  if (dof >= dim)
    {
    Epetra_MultiVector residV = resid;
    Epetra_MultiVector residP = resid;
    for (int j = 0; j < resid.NumVectors(); j++)
      for (int i = 0; i < resid.MyLength(); i++)
        {
        hymls_gidx gid = map.GID64(i);
        if (gid % dof  ==  dim)
          residV[j][i] = 0.0;
        else
          residP[j][i] = 0.0;
        }
    residV.Norm2(resNormV);
    residP.Norm2(resNormP);
    }

  if (comm_->MyPID() == 0)
    {
    Tools::out() << "Exp. res. norm(s): ";
    for (int i = 0; i < resid.NumVectors(); i++)
      {
      Tools::out() << resNorm[i] << " ";
      }
    Tools::out() << std::endl;
    Tools::out() << "Rhs norm(s): ";
    for (int i = 0; i < resid.NumVectors(); i++)
      {
      Tools::out() << rhsNorm[i] << " ";
      }
    Tools::out() << std::endl;
    if (dof >= dim)
      {
      Tools::out() << "Exp. res. norm(s) of V-part: ";
      for (int i = 0; i < resid.NumVectors(); i++)
        {
        Tools::out() << resNormV[i] << " ";
        }
      Tools::out() << std::endl;
      Tools::out() << "Exp. res. norm(s) of P-part: ";
      for (int i = 0; i < resid.NumVectors(); i++)
        {
        Tools::out() << resNormP[i] << " ";
        }
      Tools::out() << std::endl;
      }
    }
  delete [] resNorm;
  delete [] rhsNorm;
  delete [] resNormV;
  delete [] resNormP;
#endif

  return ierr;
  }

  }//namespace HYMLS
