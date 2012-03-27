//#define BLOCK_IMPLEMENTATION 1
#include "HYMLS_Solver.H"

#include "HYMLS_Tools.H"
#include "HYMLS_MatrixUtils.H"
#include "HYMLS_DenseUtils.H"
#include "HYMLS_BorderedSolver.H"
#include "HYMLS_BorderedLU.H"
#include "HYMLS_ProjectedOperator.H"

#include <Epetra_Time.h> 
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"
#include "Epetra_Import.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"
#include "Epetra_InvOperator.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Utils.hpp"

#include "BelosBlockGmresSolMgr.hpp"
#include "BelosBlockCGSolMgr.hpp"
//#include "BelosPCPGSolMgr.hpp"
#include "BelosEpetraOperator.h"

#include "Teuchos_StandardParameterEntryValidators.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

namespace HYMLS {

  // constructor
  Solver::Solver(Teuchos::RCP<const Epetra_RowMatrix> K, 
      Teuchos::RCP<Epetra_Operator> P,
      Teuchos::RCP<Teuchos::ParameterList> params,
      int numRhs)
      : matrix_(K), precond_(P), comm_(Teuchos::rcp(&(K->Comm()),false)), 
        massMatrix_(Teuchos::null), nullSpace_(Teuchos::null),
        normInf_(-1.0), useTranspose_(false),
        numEigs_(0),
        label_("HYMLS::Solver"), PLA("Solver")
  {
  START_TIMER2(label_,"Constructor");
  setParameterList(params);
  
  belosRhs_=Teuchos::rcp(new Epetra_MultiVector(matrix_->OperatorRangeMap(),numRhs));
  belosSol_=Teuchos::rcp(new Epetra_MultiVector(matrix_->OperatorDomainMap(),numRhs));

  // try to construct the nullspace for the operator, right now we only implement
  // this for the Stokes-C case (constant pressure as single vector), otherwise we
  // assume the matrix is nonsingular (i.e. leave nullSpace==null).
  string nullSpaceType=PL().get("Null Space","None");
  if (nullSpaceType=="Constant")
    {
    nullSpace_ = Teuchos::rcp(new Epetra_Vector(matrix_->OperatorDomainMap()));
    CHECK_ZERO(nullSpace_->PutScalar(1.0));
    }
  else if (nullSpaceType=="Constant P")
    {
    nullSpace_ = Teuchos::rcp(new Epetra_Vector(matrix_->OperatorDomainMap()));
    int pvar = PL().get("Pressure Variable",-1);
    // TODO: this is all a bit ad-hoc
    if (pvar==-1)
      {
      pvar=2;
      Tools::Warning("'Pressure Variable' not specified in 'Solver' sublist", 
        __FILE__, __LINE__);
      }
    int dof = PL().get("Degrees of Freedom",-1);
    if (dof==-1)
      {
      dof=3;
      Tools::Warning("'Pressure Variable' not specified in 'Solver' sublist", 
        __FILE__, __LINE__);
      }
    
    CHECK_ZERO(nullSpace_->PutScalar(0.0))
    for (int i=dof-1;i<nullSpace_->MyLength();i+=dof)
      {
      (*nullSpace_)[0][i]=1.0;
      }
    }    
  else if (nullSpaceType!="None")
    {
    Tools::Error("'Null Space'='"+nullSpaceType+"' not implemented",__FILE__,__LINE__);
    }

  belosProblemPtr_=Teuchos::rcp(new belosProblemType_(matrix_,belosSol_,belosRhs_));

  this->SetPrecond(precond_);

  Teuchos::ParameterList& belosList = PL().sublist("Iterative Solver");

//  belosList.set("Output Style",Belos::Brief);
  belosList.set("Output Style",1);
  belosList.set("Verbosity",Belos::Errors+Belos::Warnings
                         +Belos::IterationDetails
                         +Belos::StatusTestDetails
                         +Belos::FinalSummary
                         +Belos::TimingDetails);

  belosList.set("Output Stream",Tools::out().getOStream());

  // create the solver
  Teuchos::RCP<Teuchos::ParameterList> belosListPtr=rcp(&belosList,false);
  if (solverType_=="CG")
    {
    belosSolverPtr_ = rcp(new 
      Belos::BlockCGSolMgr<ST,MV,OP>
      (belosProblemPtr_,belosListPtr));
    }
  else if (solverType_=="PCG")
    {
    Tools::Error("NOT IMPLEMENTED!",__FILE__,__LINE__);
/*
    belosSolverPtr_ = Teuchos::rcp(new 
        Belos::PCPGSolMgr<ST,MV,OP>
        (belosProblemPtr_,belosListPtr));
*/
    }
  else if (solverType_=="GMRES")
    {
    belosSolverPtr_ = Teuchos::rcp(new 
        Belos::BlockGmresSolMgr<ST,MV,OP>
        (belosProblemPtr_,belosListPtr));
    }
  else
    {
    Tools::Error("Currently only 'GMRES' is supported as 'Belos Solver'",__FILE__,__LINE__);
    }  
  }


  // destructor
  Solver::~Solver()
    {
    START_TIMER3(label_,"Destructor");
    }

  void Solver::SetMatrix(Teuchos::RCP<const Epetra_RowMatrix> A)
    {
    START_TIMER3(label_,"SetMatrix");
    matrix_ = A;
    belosProblemPtr_->setOperator(matrix_);
    }

void Solver::SetPrecond(Teuchos::RCP<Epetra_Operator> P)
  {
  START_TIMER3(label_,"SetPrecond");
  precond_=P;
  if (precond_==Teuchos::null) return;
  belosPrecPtr_=Teuchos::rcp(new belosPrecType_(precond_));
  string lor = PL().get("Left or Right Preconditioning","Right");
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

  void Solver::SetMassMatrix(Teuchos::RCP<const Epetra_RowMatrix> mass)
    {
    if (mass==Teuchos::null) return;
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
    }



  // Sets all parameters for the solver
  void Solver::setParameterList(const Teuchos::RCP<Teuchos::ParameterList>& List)
    {
    START_TIMER3(label_,"SetParameterList");

    setMyParamList(List);

    //TODO: use validators everywhere

    solverType_= PL().get("Krylov Method","GMRES");
    startVec_=PL().get("Initial Vector","Random");
    PL().get("Left or Right Preconditioning","Right");

    numEigs_=PL().get("Deflated Subspace Dimension",numEigs_);
    deflThres_=PL().get("Deflation Threshold",0.0);

    // this is the place where we check for
    // valid parameters for the iterative solver
    if (validateParameters_)
      {
      this->getValidParameters();
      PL().validateParameters(VPL());
      }
    DEBUG(PL());
    }

  // Sets all parameters for the solver
  Teuchos::RCP<const Teuchos::ParameterList> Solver::getValidParameters() const
    {
    if (validParams_!=Teuchos::null) return validParams_;
    START_TIMER3(label_,"getValidParameterList");
    
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
        
    VPL().set("Left or Right Preconditioning", "Left",
                            "wether to do left (P\\Ax=P\\b) or right (AP\\(Px)=b) preconditioning",
                            lorValidator);

    VPL().set("Deflated Subspace Dimension",0,"maximum number of eigenmodes to deflate");

    VPL().set("Deflation Threshold",1.0e-3,"An eigenmode is deflated if the eigenvalue is within [-eps 0]");

    // these are temporarily added to the parameter list for developing the
    // projection method and should be handled differently in the end.
    VPL().set("Null Space","None","type of null vector, only for development in this list");
    VPL().set("Pressure Variable","which is the pressure variable, only for development in this list");
    VPL().set("Degrees of Freedom","dof/cell, only for development in this list");

    // Belos parameters should be specified in this list:
    VPL().sublist("Iterative Solver",false,
        "Parameter list for the Krylov method (passed to Belos)").disableRecursiveValidation();
    return validParams_;
    }


int Solver::SetupDeflation(int maxEigs)
  {
  // by default leave numEigs_ at its present value:
  if (maxEigs!=-2) numEigs_=maxEigs;
  
  // nothing to be done:
  if (numEigs_==0) return 0;

  START_TIMER(label_,"SetupDeflation");

  Teuchos::RCP<Epetra_Operator> op, iop;
  
  op=precond_;
  
  if (nullSpace_!=Teuchos::null)
    {
#ifdef STORE_MATRICES
    MatrixUtils::Dump(*nullSpace_,"DefaultNullSpace.txt");
#endif
#ifdef DEBUGGING
    Epetra_Vector test_y(nullSpace_->Map());
    Epetra_Vector test_x(nullSpace_->Map());
    CHECK_ZERO(test_x.Random());
    CHECK_ZERO(test_y.PutScalar(0.0));
    CHECK_ZERO(precond_->ApplyInverse(test_x,test_y));
    MatrixUtils::Dump(test_y,"PROJ_prec_sol.txt");
    MatrixUtils::Dump(test_y,"PROJ_rhs.txt");
#endif

  // we compute eigs of the operator [K e; e' 0] to make the operator nonsingular
  Teuchos::RCP<BorderedSolver> bprec
      = Teuchos::rcp_dynamic_cast<BorderedSolver>(precond_);
    if (bprec!=Teuchos::null)
      {
      Tools::out()<<"using preconditioner's bordering option"<<std::endl;
      CHECK_ZERO(bprec->SetBorder(nullSpace_,nullSpace_));      
      }
    else
      {
      Tools::out()<<"using LU bordering"<<std::endl;
      op = Teuchos::rcp(new BorderedLU(precond_,nullSpace_,nullSpace_));
      }
#ifdef DEBUGGING
    CHECK_ZERO(test_y.PutScalar(0.0));
    CHECK_ZERO(op->ApplyInverse(test_x,test_y));
    MatrixUtils::Dump(test_y,"PROJ_bordered_prec_sol.txt");
#endif
    }    

  iop = Teuchos::rcp(new Epetra_InvOperator(op.get()));
#ifdef STORE_MATRICES
  if (massMatrix_!=Teuchos::null) 
    {
    Teuchos::RCP<const Epetra_CrsMatrix> massCrs
        = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(massMatrix_);
    if (!Teuchos::is_null(massCrs))
      {
      MatrixUtils::Dump(*massCrs,"MassMatrix.txt");
      }
    else
      {
      Tools::Warning("mass matrix not in CRS format??",__FILE__,__LINE__);
      }
    }
  else
    {
    // for e.g. Navier-Stokes it is important to set the mass matrix before
    // calling this function, by calling SetMassMatrix() in both the solver
    // and the preconditioner.
    Tools::Warning("SetupDeflation() called without mass matrix, is that what you want?",
        __FILE__,__LINE__);
    }
#endif

    // compute dominant eigenvalues of P^{-1}.
    // mass matrix may still be null, the routine works for both cases.
    precEigs_ = MatrixUtils::Eigs
            (iop, massMatrix_, numEigs_,1.0e-8);

// I think this should never occur:
if (precEigs_==Teuchos::null)
  {
  Tools::Error("null returned from Eigs routine?",__FILE__,__LINE__);
  }

if (precEigs_->numVecs<numEigs_)
  {
  Tools::Warning("found "+Teuchos::toString(precEigs_->numVecs)
  +" eigenpairs in SetupDeflation(), while you requested "+Teuchos::toString(numEigs_),
  __FILE__,__LINE__);
  numEigs_=precEigs_->numVecs;
  }

if (numEigs_==0) 
  {
  return 1;
  }

// neither should this happen:
if (precEigs_->Evecs==Teuchos::null)
  {
  Tools::Error("no eigenvectors have been returned.",__FILE__,__LINE__);
  }
// ... or this:
if (precEigs_->Espace==Teuchos::null)
  {
  Tools::Error("no eigenvector basis has been returned.",__FILE__,__LINE__);
  }

  Epetra_MultiVector V = *(precEigs_->Espace);
  
  // now project the original Schur-complement onto the space spanned by these most
  // unstable modes of the preconditioner, e.g. compute V'SV

  Epetra_MultiVector KV = V;
  CHECK_ZERO(matrix_->Apply(V,KV));

  int n = V.NumVectors();
  
  Epetra_SerialDenseMatrix C(n,n);
  CHECK_ZERO(DenseUtils::MatMul(V,KV,C));

  Epetra_SerialDenseMatrix EVL(n,n);
  Epetra_SerialDenseMatrix EVR(n,n);
  Epetra_SerialDenseVector lambda_r(n);
  Epetra_SerialDenseVector lambda_i(n);
  DenseUtils::Eig(C,lambda_r,lambda_i,EVR,EVL);
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
      Tools::out() << "lambda(V'KV)="<<lambda_r[i]<<" + i "<<lambda_i[i]<<std::endl;
      }
    }
  Tools::out() << "======================================="<<std::endl;
  Tools::out() << "next most delicate eigenvalue: "<< lambda_max<<std::endl;
  Tools::out() << "======================================="<<std::endl;

  // now compute V'P\SV and see which modes are not well handled
  // by the preconditioner.
  Epetra_MultiVector PKV = V;

  CHECK_ZERO(precond_->ApplyInverse(KV,PKV));
  
  CHECK_ZERO(DenseUtils::MatMul(V,PKV,C));
  // compute all eigenvalues of this operator
  DenseUtils::Eig(C,lambda_r,lambda_i,EVR,EVL);

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
  
  if (numDeflated_>0)
    {
    // throw out the according vectors from W and V
    Epetra_SerialDenseMatrix W(n,numDeflated_);
    Epetra_MultiVector V_hat(V.Map(),numDeflated_);
  
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
    CHECK_ZERO(DenseUtils::Orthogonalize(W));

    // interpret W as a MultiVector
    Teuchos::RCP<Epetra_MultiVector> W_hat = DenseUtils::CreateView(W);
  
    // compute the new V as V*W_hat  
    V_=Teuchos::rcp(new Epetra_MultiVector(V.Map(),numDeflated_));
    CHECK_ZERO(V_->Multiply('N','N',1.0,V_hat,*W_hat,0.0));
    
#ifdef DEBUGGING
MatrixUtils::Dump(*(precEigs_->Evecs),"PROJ_evecs.txt");
MatrixUtils::Dump(*V_,"PROJ_V.txt");
Epetra_MultiVector test_x(V_->Map(),5);
Epetra_MultiVector test_y(V_->Map(),5);
test_x.Random();
CHECK_ZERO(Aorth_->Apply(test_x,test_y));
MatrixUtils::Dump(test_x,"PROJ_x.txt");
MatrixUtils::Dump(test_y,"PROJ_Aorth_x.txt");
#endif
    

    borderV_=Teuchos::rcp(new Epetra_MultiVector(V_->Map(),numDeflated_));    
    // compute V_orth'KV
    DenseUtils::ApplyOrth(*V_,KV,*borderV_);
    // compute V'KV_orth (K is not symmetric, so we
    // actually compute borderW_' = V_orthK'V. V_orth=I-VV' is symmetric)
    borderW_=Teuchos::rcp(new Epetra_MultiVector(V_->Map(),numDeflated_));
    CHECK_ZERO(matrix_->Multiply(true,*V_,V_hat));
    CHECK_ZERO(DenseUtils::ApplyOrth(*V_,V_hat,*borderW_));
    // compute V'KV
    borderC_=Teuchos::rcp(new Epetra_SerialDenseMatrix(numDeflated_,numDeflated_));
  CHECK_ZERO(DenseUtils::MatMul(*V_,KV,*borderC_));

    // check if the preconditioner can handle a bordered system. It has to
    // solve a system of the form                               
    // |A  V| |x|   |b|                                         
    // |V' O| |s| = |0| now to maintain orthogonality wrt V.    
    //                                                          
    Teuchos::RCP<HYMLS::BorderedSolver> borderedPrec =
        Teuchos::rcp_dynamic_cast<HYMLS::BorderedSolver>(precond_);
    
    if (Teuchos::is_null(borderedPrec))
      {
      Tools::Error("preconditioner cannot handle bordering",__FILE__,__LINE__);
      }
    else
      {
      Teuchos::RCP<Epetra_SerialDenseMatrix> C = Teuchos::rcp(new
        Epetra_SerialDenseMatrix(numDeflated_,numDeflated_));
      CHECK_ZERO(borderedPrec->SetBorder(V_,V_,C));
      }

    Aorth_=Teuchos::rcp(new ProjectedOperator(matrix_,V_,true));

    belosProblemPtr_->setOperator(Aorth_);
    belosProblemPtr_->setProblem();

  Teuchos::ParameterList& belosList = PL().sublist("Iterative Solver");
    
    Teuchos::RCP<Teuchos::ParameterList> belosParamPtr = Teuchos::rcp(
        new Teuchos::ParameterList(belosList));
        
    if (solverType_=="GMRES")
      {
      belosParamPtr->set("Solver","BlockGmres");
      belosParamPtr->set("Block Size",numDeflated_);
      }
    else
      {
      Tools::Error("not implemented",__FILE__,__LINE__);
      }

    AorthSolver_ = Teuchos::rcp(new Belos::EpetraOperator
        (belosProblemPtr_,belosParamPtr));

    // setup the LU decomposition
    LU_=Teuchos::rcp(new BorderedLU(AorthSolver_,borderV_,borderW_,borderC_));
#ifdef DEBUGGING
// dump more PROJ_ stuff
#endif
    }//numDeflated_>0

  return 0;
  }

// Applies the preconditioner to vector X, returns the result in Y.
int Solver::ApplyInverse(const Epetra_MultiVector& B,
                           Epetra_MultiVector& X) const
  {
  START_TIMER(label_,"ApplyInverse");
  int ierr=0;
  if (LU_!=Teuchos::null)
    {
    // bordered solve
    ierr=LU_->ApplyInverse(B,X);
    }
  else
    {
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

    if (startVec_=="Random")
      {      
#ifdef DEBUGGING
      int seed=42;
      MatrixUtils::Random(*belosSol_, seed);
#else
      MatrixUtils::Random(*belosSol_);
#endif      
      }
    else if (startVec_=="Zero")
      {
      // set initial vector to 0
      CHECK_ZERO(belosSol_->PutScalar(0.0));
      }
    else
      {
      *belosSol_=X;
      }

    CHECK_TRUE(belosProblemPtr_->setProblem());

    Belos::ReturnType ret;
    bool status=true;
    try {
    ret=belosSolverPtr_->solve();
    } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr, status);
    if (!status) Tools::Warning("caught an exception",__FILE__,__LINE__);

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
/*    
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
*/
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
      ierr = -1;
      }
    }
  return ierr;
  }


}//namespace HYMLS
