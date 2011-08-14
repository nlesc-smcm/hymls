//#define BLOCK_IMPLEMENTATION 1
#include "HYMLS_Solver.H"

#include "HYMLS_Tools.H"
#include "HYMLS_MatrixUtils.H"
#include "HYMLS_EigenUtils.H"

#include <Epetra_Time.h> 
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Import.h"
#include "Epetra_InvOperator.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Utils.hpp"

#include "BelosBlockGmresSolMgr.hpp"
#include "BelosBlockCGSolMgr.hpp"
#include "BelosPCPGSolMgr.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

namespace HYMLS {

  // constructor
  Solver::Solver(Teuchos::RCP<const Epetra_RowMatrix> K, 
      Teuchos::RCP<Epetra_Operator> P,
      Teuchos::RCP<Teuchos::ParameterList> params,
      int numRhs)
      : matrix_(K), precond_(P), comm_(Teuchos::rcp(&(K->Comm()),false)), 
        params_(Teuchos::null),
        massMatrix_(Teuchos::null),
        normInf_(-1.0), useTranspose_(false),
        numEigs_(0),
        label_("HYMLS::Solver")
  {
  START_TIMER2(label_,"Constructor");
  SetParameters(*params);
  
  belosRhs_=Teuchos::rcp(new Epetra_MultiVector(matrix_->OperatorRangeMap(),numRhs));
  belosSol_=Teuchos::rcp(new Epetra_MultiVector(matrix_->OperatorDomainMap(),numRhs));

  belosProblemPtr_=Teuchos::rcp(new belosProblemType_(matrix_,belosSol_,belosRhs_));

  this->SetPrecond(precond_);

  Teuchos::ParameterList& belosList = params_->sublist("Solver").sublist("Iterative Solver");
  string linearSolver = params_->sublist("Solver").get("Krylov Method","GMRES");

  belosList.set("Output Style",Belos::Brief);
  belosList.set("Verbosity",Belos::Errors+Belos::Warnings
                         +Belos::IterationDetails
                         +Belos::StatusTestDetails
                         +Belos::FinalSummary
                         +Belos::TimingDetails);

  belosList.set("Output Stream",Tools::out().getOStream());

  // create the solver
  RCP<Teuchos::ParameterList> belosListPtr=rcp(&belosList,false);
  if (linearSolver=="CG")
    {
    belosSolverPtr_ = rcp(new 
      Belos::BlockCGSolMgr<ST,MV,OP>
      (belosProblemPtr_,belosListPtr));
    }
  else if (linearSolver=="PCG")
    {
    Tools::Warning("NOT IMPLEMENTED!",__FILE__,__LINE__);
    belosSolverPtr_ = rcp(new 
        Belos::PCPGSolMgr<ST,MV,OP>
        (belosProblemPtr_,belosListPtr));
    }
  else if (linearSolver=="GMRES")
    {
    belosSolverPtr_ = rcp(new 
        Belos::BlockGmresSolMgr<ST,MV,OP>
        (belosProblemPtr_,belosListPtr));
    }
  else
    {
    Tools::Error("Currently only 'GMRES' is supported as 'Belos Solver'",__FILE__,__LINE__);
    }
  
  STOP_TIMER2(label_,"Constructor");
  }


  // destructor
  Solver::~Solver()
    {
    DEBUG("Solver::~Solver()");
    }

  void Solver::SetMatrix(Teuchos::RCP<const Epetra_RowMatrix> A)
    {
    matrix_ = A;
    belosProblemPtr_->setOperator(matrix_);
    }

void Solver::SetPrecond(Teuchos::RCP<Epetra_Operator> P)
  {
  precond_=P;
  if (precond_==Teuchos::null) return;
  belosPrecPtr_=Teuchos::rcp(new belosPrecType_(precond_));
  string lor = params_->sublist("Solver").get("Left or Right Preconditioning","Right");
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
  else
    {
    Tools::Error("Parameter 'Left or Right Preconditioning' has an invalid value",
              __FILE__, __LINE__);
    }
  }

  void Solver::SetMassMatrix(Teuchos::RCP<const Epetra_RowMatrix> mass)
    {
    START_TIMER3(label_,"SetMassMatrix");
    if (mass->RowMatrixRowMap().SameAs(matrix_->RowMatrixRowMap()))
      {
      massMatrix_ = mass;
      }
    else
      {
      Tools::Error("Mass matrix must have same row map as solver",
                __FILE__,__LINE__);
      }
    STOP_TIMER3(label_,"SetMassMatrix");
    }



  // Sets all parameters for the solver
  int Solver::SetParameters(Teuchos::ParameterList& List)
    {
    DEBUG("Enter Solver::SetParameters()");
    if (Teuchos::is_null(params_))
      {
      params_=Teuchos::rcp(new Teuchos::ParameterList);
      }

    // this is the place where we check for
    // valid parameters for the whole
    // method

    Teuchos::ParameterList& solverList_ = params_->sublist("Solver");
    Teuchos::ParameterList& solverList = List.sublist("Solver");

    solverList_.set("Krylov Method",solverList.get("Krylov Method","GMRES"));
    solverList_.set("Initial Vector",solverList.get("Initial Vector","Random"));
    solverList_.set("Left or Right Preconditioning",solverList.get("Left or Right Preconditioning","Right"));

    numEigs_=solverList.get("Deflated Subspace Dimension",numEigs_);
    solverList_.set("Deflated Subspace Dimension",numEigs_);
    //TODO: put a reasonable default value here
    deflThres_=solverList.get("Deflation Threshold",1.0);
    solverList_.set("Deflation Threshold",deflThres_);

    // Belos parameters should be specified in this list:
    solverList_.sublist("Iterative Solver")=solverList.sublist("Iterative Solver");

    DEBUG("Leave Solver::SetParameters()");
    return 0;
    }



int Solver::SetupDeflation()
  {
  START_TIMER(label_,"SetupDeflation");
#if 0

//TODO: this is not all tested, some of it works (the first few steps)
//  if (massMatrix_!=Teuchos::null)
    Teuchos::RCP<Epetra_Operator> op = 
    Teuchos::rcp(new Epetra_InvOperator(precond_.get()));
    // compute dominant eigenvalues of P^{-1}.
    // mass matrix may still be null, the routine works for both cases.
    eigenInfo_ = EigenUtils::Eigs
            (op, belosMass_, numEigs_,1e-4);

#ifdef STORE_MATRICES
  if (massMatrix_!=Teuchos::null) MatrixUtils::Dump(*massMatrix_,"MassMat.txt",false);
  if (belosMass_!=Teuchos::null) MatrixUtils::Dump(*belosMass_,"SchurMassMatReindexed.txt",true);
#endif

  // now project the original Schur-complement onto the space spanned by these most
  // unstable modes of the preconditioner, e.g. compute V'SV
  const Epetra_MultiVector& V = *(eigenInfo_->Espace);
  Epetra_MultiVector SV = V;
  CHECK_ZERO(matrix_->Apply(V,SV));

  int n = V.NumVectors();
  
  Epetra_SerialDenseMatrix C(n,n);
  CHECK_ZERO(EigenUtils::MatMul(V,SV,C));

  Epetra_SerialDenseMatrix EVL(n,n);
  Epetra_SerialDenseMatrix EVR(n,n);
  Epetra_SerialDenseVector lambda_r(n);
  Epetra_SerialDenseVector lambda_i(n);
  EigenUtils::Eig(C,lambda_r,lambda_i,EVR,EVL);
  double lambda_max=-1.0e100;
#ifdef TESTING
  Tools::out()<<std::scientific;
  Tools::out() << "eigenvalue estimates near zero:"<<std::endl;
  Tools::out() << "==============================="<<std::endl;
#endif

  for (int i=0;i<n;i++)
    {
#ifdef TESTING
    Tools::out() << lambda_r[i]<<"\t+ i\t"<<lambda_i[i]<<std::endl;
#endif
    if (lambda_r[i]<0.0)
      {
      lambda_max=std::max(lambda_max,lambda_r[i]);
      }
    else
      {
      Tools::out() << "an eigenvalue is in the right half plane."<<std::endl;
      Tools::out() << "lambda(V'SV)="<<lambda_r[i]<<" + i "<<lambda_i[i]<<std::endl;
      }
    }
  Tools::out() << "======================================="<<std::endl;
  Tools::out() << "next most delicate eigenvalue: "<< lambda_max<<std::endl;
  Tools::out() << "======================================="<<std::endl;


  // now compute V'P\SV and see which modes are not well handled
  // by the preconditioner.
  Epetra_MultiVector PSV = V;
  CHECK_ZERO(belosPrec_->ApplyInverse(SV,PSV));
  
  CHECK_ZERO(EigenUtils::MatMul(V,PSV,C));
  // compute all eigenvalues of this operator
  EigenUtils::Eig(C,lambda_r,lambda_i,EVR,EVL);

  // keep those that are further away from 1 then the "Deflation Threshold"
  std::vector<bool> keep(n);
  numDeflated_=n;
  double re,im,tol=deflThres_*deflThres_;
  for (int i=0;i<n;i++)
    {
    re=1.0-lambda_r[i]; im = lambda_i[i];
#ifdef TESTING
      Tools::out() << "lambda(V'P\\SV)["<<i<<"]=";
      Tools::out() << lambda_r[i] << " + i (" << lambda_i[i] << ")"<<std::endl;
#endif      
    if ((re*re+im*im)>tol)
      {
      Tools::out() << "deflating eigenmode "<<i<<": lambda(V'P\\SV)=";
      Tools::out() << lambda_r[i] << " + i (" << lambda_i[i] << ")"<<std::endl;
      keep[i]=true;
      }
    else
      {
      numDeflated_--;
      keep[i]=false;
      }
    }
  Tools::out() << "number of eigenmodes to be deflated: "<<numDeflated_<<std::endl;

  // throw out the according vectors from W and V
  Epetra_SerialDenseMatrix W(n,numDeflated_);
  Epetra_MultiVector V_hat(*map2_,numDeflated_));
  
  int pos=0;
  
  
  for (int j=0;j<n;j++)
    {
    if (keep[j])
      {
      *V_hat(pos) = *V(j);
      for (int i=0;i<n;i++)
        {
        W[pos][i]=EVR[j][i];
        }
      pos++;
      }
    }

  // orthogonalize W
  CHECK_ZERO(EigenUtils::Orthogonalize(W_hat));

  // interpret W as a MultiVector
  Teuchos::RCP<Epetra_MultiVector> W_hat = EigenUtils::View(W);

  
  // compute the new V as V*W_hat  
  deflBase_=Teuchos::rcp(new Epetra_MultiVector(*map2_,numDeflated_));
  CHECK_ZERO(deflBase_->Multiply('N','N',1.0,V_hat,*W_hat,0.0));
// .. so far, so good.
#endif
  STOP_TIMER(label_,"SetupDeflation");
  return 0;
  }

// Applies the preconditioner to vector X, returns the result in Y.
int Solver::ApplyInverse(const Epetra_MultiVector& B,
                           Epetra_MultiVector& X) const
  {
  START_TIMER(label_,"ApplyInverse");

#ifdef TESTING
  if (X.NumVectors()!=B.NumVectors())
    {
    Tools::Error("different number of input and output vectors",__FILE__,__LINE__);
    } 
  if (X.NumVectors()!=belosRhs_->NumVectors())
    {
    // not implemented, number of vectors has to be passed to constructor
    Tools::Error("cannot change number of vectors",__FILE__,__LINE__);
    }
  if (!(X.Map().SameAs(belosSol_->Map()) && 
        B.Map().SameAs(belosRhs_->Map()) ))
    {
    Tools::Error("incompatible maps",__FILE__,__LINE__);
    }
#endif

   //TODO: avoid this copy operation
   *belosRhs_ = B;

    string startVec = params_->sublist("Solver").get("Initial Vector","Previous");

    if (startVec=="Random")
      {      
#ifdef DEBUGGING
      int seed=42;
      MatrixUtils::Random(*belosSol_, seed);
#else
      MatrixUtils::Random(*belosSol_);
#endif      
      }
    else if (startVec=="Zero")
      {
      // set initial vector to 0
      EPETRA_CHK_ERR(belosSol_->PutScalar(0.0));
      }
    else
      {
      *belosSol_=X;
      }

    CHECK_TRUE(belosProblemPtr_->setProblem());

    Belos::ReturnType ret;
    int status;
    try {
    ret=belosSolverPtr_->solve();
    } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr, status);

  int numIters = belosSolverPtr_->getNumIters();
  if (comm_->MyPID()==0)
     {
     Tools::Out("++++++++++++++++++++++++++++++++++++++++++++++++");
     Tools::Out("+ Number of iterations: "+Teuchos::toString(numIters));
     Tools::Out("++++++++++++++++++++++++++++++++++++++++++++++++");
     Tools::Out("");
     }

    //TODO: avoid this copy operation
    X=*belosSol_;
    
    for (int j=0;j<X.NumVectors();j++)
      {
      for (int i=0;i<X.MyLength();i++)
        if (X[j][i]!=(*belosSol_)[j][i])
          {
          Tools::out() << "belos: "<<*belosSol_<<std::endl;
          Tools::out() << "output: "<<X <<std::endl;
          Tools::Error("Copy operation failed!",__FILE__,__LINE__);
          }
      }

#ifdef TESTING
// compute explicit residual of Schur problem
Epetra_MultiVector resid(belosRhs_->Map(),belosRhs_->NumVectors());
CHECK_ZERO(matrix_->Apply(*belosSol_,resid));
CHECK_ZERO(resid.Update(1.0,B,-1.0));
double resNorm[resid.NumVectors()];
resid.Norm2(resNorm);
if (comm_->MyPID()==0)
  {
  Tools::out() << "Residual norm: ";
  for (int ii=0;ii<resid.NumVectors();ii++)
    {
    Tools::out() << resNorm[ii] << " ";
    }
  Tools::out() << std::endl;
  }
#endif

    if (ret!=Belos::Converged)
      {
      // the nature of the problem is kind of hard to determine...
      Tools::Warning("Belos returned "+Teuchos::toString((int)ret)+"!",__FILE__,__LINE__);    
#ifdef TESTING
      Teuchos::RCP<const Epetra_CrsMatrix> Acrs =
        Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);
      MatrixUtils::Dump(*Acrs,"FailedMatrix.txt");
      MatrixUtils::Dump(B,"FailedRhs.txt");
#endif      
      }
      
    STOP_TIMER(label_,"ApplyInverse");
    return 0;
    }


}//namespace HYMLS
