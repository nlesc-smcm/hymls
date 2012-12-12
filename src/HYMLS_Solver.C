//#define BLOCK_IMPLEMENTATION 1
#include "HYMLS_no_debug.H"
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
#include "HYMLS_EpetraExt_ProductOperator.H"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Utils.hpp"

#include "BelosBlockGmresSolMgr.hpp"
#include "BelosTypes.hpp"
#include "BelosBlockCGSolMgr.hpp"
//#include "BelosPCPGSolMgr.hpp"
#include "HYMLS_BelosEpetraOperator.H"

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
        numEigs_(0), numIter_(0), doBordering_(false),
        label_("HYMLS::Solver"), PLA("Solver")
  {
  START_TIMER3(label_,"Constructor");
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
    // NOTE: we assume u/v/w/p[/T] ordering here, it works for 2D and 3D as long
    // as var[dim]=P
    nullSpace_ = Teuchos::rcp(new Epetra_Vector(matrix_->OperatorDomainMap()));
    int pvar = PL("Problem").get("Dimension",-1);
    int dof = PL("Problem").get("Degrees of Freedom",-1);
    // TODO: this is all a bit ad-hoc
    if (pvar==-1||dof==-1)
      {
      Tools::Error("'Dimension' or 'Degrees of Freedom' not set in 'Problem' sublist",
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
  if (nullSpace_!=Teuchos::null)
    {
    int k = nullSpace_->NumVectors();
    double *nrm2 = new double[k];
    CHECK_ZERO(nullSpace_->Norm2(nrm2));
    for (int i=0;i<k;i++)
      {
      CHECK_ZERO((*nullSpace_)(i)->Scale(1.0/nrm2[i]));
      }
    delete [] nrm2;
    }

  belosProblemPtr_=Teuchos::rcp(new belosProblemType_(matrix_,belosSol_,belosRhs_));
  
  doBordering_ = ((nullSpaceType!="None") ||
                  (PL().get("Deflated Subspace Dimension",0)>0));

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
    belosSolverPtr_ = rcp(new 
      ::Belos::BlockCGSolMgr<ST,MV,OP>
      (belosProblemPtr_,belosListPtr));
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
    belosSolverPtr_ = Teuchos::rcp(new 
        ::Belos::BlockGmresSolMgr<ST,MV,OP>
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
  if (precond_==Teuchos::null || doBordering_) return;
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
    VPL().set("Pressure Variable",-1,"which is the pressure variable, only for development in this list");
    VPL().set("Degrees of Freedom",-1,"dof/cell, only for development in this list");

    // Belos parameters should be specified in this list:
    VPL().sublist("Iterative Solver",false,
        "Parameter list for the Krylov method (passed to Belos)").disableRecursiveValidation();
    return validParams_;
    }


Teuchos::RCP<MatrixUtils::Eigensolution> Solver::EigsPrec(int numEigs) const
  {
  // If there is a null-space, deflate it.
  // If no NS and no additional vectors asked for -
  // nothing to be done.
  START_TIMER(label_,"EigsPrec");

  Teuchos::RCP<Epetra_Operator> op, iop;
  Teuchos::RCP<MatrixUtils::Eigensolution> precEigs=Teuchos::null;
  
  op=precond_;

  ////////////////////////////////////////////////////////////////////////
  // Start by constructing the operator iop, which will be                
  // [P N; N' 0] * [M 0; 0 I], with P the preconditioner, M the mass      
  // matrix, N the null space and I the identity matrix.                  
  ////////////////////////////////////////////////////////////////////////
  if (massMatrix_!=Teuchos::null) 
    {
    // construct the operator P\M
    EpetraExt::ProductOperator::EApplyMode mode[2];
    Teuchos::ETransp trans[2];
    
    op_array_[1]=massMatrix_;
    mode[1]=EpetraExt::ProductOperator::APPLY_MODE_APPLY;
    trans[1]=Teuchos::NO_TRANS;
    op_array_[0]=precond_;
    mode[0]=EpetraExt::ProductOperator::APPLY_MODE_APPLY_INVERSE;
    trans[0]=Teuchos::NO_TRANS;
    
    iop=Teuchos::rcp(new EpetraExt::ProductOperator(2,&op_array_[0],trans,mode));
    }
  else
    {
    // for e.g. Navier-Stokes it is important to set the mass matrix before
    // calling this function, by calling SetMassMatrix() in both the solver
    // and the preconditioner.
    Tools::Warning("EigsPrec() called without mass matrix, is that what you want?",
          __FILE__,__LINE__);
    iop = Teuchos::rcp(new Epetra_InvOperator(op.get()));
    }

    ////////////////////////////////////////////////////
    // compute dominant eigenvalues of (P^{-1}, M).     
    ////////////////////////////////////////////////////
    bool status=true;
    try {
    Tools::Out("Compute max eigs of inv(P)");
    precEigs = MatrixUtils::Eigs
            (iop, Teuchos::null, numEigs_,1.0e-8);
    } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr,status);
    if (!status) Tools::Fatal("caught an exception",__FILE__,__LINE__);
    
    // I think this should never occur:
    if (precEigs==Teuchos::null)
      {
      Tools::Error("null returned from Eigs routine?",__FILE__,__LINE__);
      }

    if (precEigs->numVecs<numEigs)
      {
      Tools::Warning("found "+Teuchos::toString(precEigs->numVecs)
      +" eigenpairs in EigsPrec(), while you requested "+Teuchos::toString(numEigs),
        __FILE__,__LINE__);
      }
    return precEigs;
    }


int Solver::SetupDeflation(int maxEigs)
  {
  // by default leave numEigs_ at its present value:
  if (maxEigs!=-2) numEigs_=maxEigs;
  
  // If there is a null-space, deflate it.
  // If no NS and no additional vectors asked for -
  // nothing to be done.
  if (numEigs_==0 && nullSpace_==Teuchos::null) return 0;
  START_TIMER(label_,"SetupDeflation");

  Teuchos::RCP<Epetra_MultiVector> KV;
        
  // add null space as border to preconditioner
  if (nullSpace_!=Teuchos::null)
    {
    // we compute eigs of the operator [K N; N' 0] to make the operator nonsingular
    Teuchos::RCP<BorderedSolver> bprec
        = Teuchos::rcp_dynamic_cast<BorderedSolver>(precond_);
    if (bprec!=Teuchos::null)
      {
      Tools::Out("set null space as border for preconditioner");
      CHECK_ZERO(bprec->SetBorder(nullSpace_,nullSpace_));      
      }
    else if (precond_!=Teuchos::null)
      {
      Tools::Error("feature not implemented",__FILE__,__LINE__);
      // should work in principle, but we can handle borders smarter
      /*
      Tools::Out("add null space to preconditioner by LU");
      op = Teuchos::rcp(new BorderedLU(precond_,nullSpace_,nullSpace_));
      */
      }
    }

////////////////////////////////////////////////////////////////////////
// if we want to deflate more eigenmodes, compute the dominant eigen-   
// values and -vectors of the preconditioner (augmented by the null-    
// space), and select which are worth deflating.                        
////////////////////////////////////////////////////////////////////////
  if (numEigs_!=0)    
    {
    Tools::Warning("Deflation is experimental functionality...",
        __FILE__,__LINE__);
    precEigs_ = this->EigsPrec(numEigs_);
    
    numEigs_=precEigs_->numVecs;

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

#ifdef DEBUGGING
  DEBUG("precEigs_->Evals = ...");
  for (unsigned int i=0;i<precEigs_->Evals.size();i++)
    {
    DEBUG(precEigs_->Evals[i].realpart << "\t"<<precEigs_->Evals[i].imagpart);
    }
  HYMLS::MatrixUtils::Dump(*(precEigs_->Evecs),"EigenvectorsOfBorderedPrec.txt");
  HYMLS::MatrixUtils::Dump(*(precEigs_->Espace),"EigenBasisOfBorderedPrec.txt");

#endif 

    Epetra_MultiVector V = *(precEigs_->Espace);
  
    // now project the original matrix onto the space spanned by these most
    // unstable modes of the preconditioner, e.g. compute C=V'KV
    KV=Teuchos::rcp(new Epetra_MultiVector(V));
    CHECK_ZERO(matrix_->Apply(V,*KV));

    int n = V.NumVectors();
  
    Epetra_SerialDenseMatrix C(n,n);
    CHECK_ZERO(DenseUtils::MatMul(V,*KV,C));

    // compute all eigenpairs of V'KV
    Epetra_SerialDenseMatrix EVL(n,n);
    Epetra_SerialDenseMatrix EVR(n,n);
    Epetra_SerialDenseVector lambda_r(n);
    Epetra_SerialDenseVector lambda_i(n);
    DenseUtils::Eig(C,lambda_r,lambda_i,EVR,EVL);
    double lambda_max=-1.0e100;
#ifdef TESTING
    Tools::out()<<std::scientific;
    Tools::Out("eigenvalue estimates near zero:");
    Tools::Out("===============================");
#endif

    // these eigenvalues give a good indication of e.g. a bifurcation:
    for (int i=0;i<n;i++)
      {
      Tools::Out(Teuchos::toString(lambda_r[i])+"\t+ i\t"+Teuchos::toString(lambda_i[i]));
      if (lambda_r[i]<0.0)
        {
        lambda_max=std::max(lambda_max,lambda_r[i]);
        }
      else
        {
        Tools::Out("an eigenvalue is in the right half plane.");
        Tools::Out("lambda(V'KV)="+Teuchos::toString(lambda_r[i])+" + i "+Teuchos::toString(lambda_i[i]));
        }
      }
    Tools::Out("=======================================");
    Tools::Out("next most delicate eigenvalue: "+Teuchos::toString(lambda_max));
    Tools::Out("=======================================");

    ////////////////////////////////////////////////////////////////////
    // now compute V'P\KV and see which modes are not well handled      
    // by the preconditioner.                                           
    ////////////////////////////////////////////////////////////////////
    Epetra_MultiVector PKV = V;


    CHECK_ZERO(precond_->ApplyInverse(*KV,PKV));

    CHECK_ZERO(DenseUtils::MatMul(V,PKV,C));
    // compute all eigenvalues of this operator
    DenseUtils::Eig(C,lambda_r,lambda_i,EVR,EVL);

    DEBUG("V'inv(P)KV, should be close to identity if preconditioner is good:");
    DEBVAR(C);

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
        DEBUG("lambda(V'P\\SV)["<<i<<"]="<<lambda_r[i] << " + i (" << lambda_i[i] << ")");
#endif
      if ((re*re+im*im)>tol)
        {
        Tools::out() << "deflating eigenmode "<<i<<": lambda(V'P\\SV)=";
        Tools::out() << lambda_r[i] << " + i (" << lambda_i[i] << ")"<<std::endl;
        keep[i]=true;
        DEBUG("deflated");
        }
      else
        {
        DEBUG("not deflated");
        numDeflated_--;
        keep[i]=false;
        }
      }
    Tools::Out("number of eigenmodes to be deflated: "+Teuchos::toString(numDeflated_));
    if (numDeflated_==0)
      {
      numDeflated_=nullSpace_->NumVectors();
      V_=nullSpace_;
      }
    else
      {
      Epetra_SerialDenseMatrix W(n,numDeflated_);
 
      // throw out the according vectors from W
      Epetra_MultiVector V_hat(V.Map(),numDeflated_);

      int pos=0;
 
      for (int j=0;j<n;j++)
        {
        if (keep[j])
          {
          for (int i=0;i<n;i++)
            {
            W[pos][i]=EVR[j][i];
            }
          pos++;
          }
        }
    
      // interpret W as a MultiVector
      Teuchos::RCP<Epetra_MultiVector> W_hat = DenseUtils::CreateView(W);

      // orthogonalize W (probably it already is orthogonal?)
      CHECK_ZERO(DenseUtils::Orthogonalize(W));

      // compute the new V as V*W_hat  
      CHECK_ZERO(V_hat.Multiply('N','N',1.0,V,*W_hat,0.0));

      // also deflate the default null space.
      int dimNull = nullSpace_->NumVectors();
      V_=Teuchos::rcp(new Epetra_MultiVector(V.Map(),numDeflated_+dimNull));
      for (int k=0;k<dimNull;k++) *((*V_)(k)) = *((*nullSpace_)(k));
      for (int k=0;k<numDeflated_;k++) *((*V_)(dimNull+k)) = *(V_hat(k));
      numDeflated_+=dimNull;
      }
    }
  else
    {
    numDeflated_=nullSpace_->NumVectors();
    V_=nullSpace_;
    }
    
#ifdef STORE_MATRICES
  MatrixUtils::Dump(*V_,"DeflationVectors.txt");
#endif

////////////////////////////////////////////////////////////////////////
// construct the operator we use to solve for v_orth:                   
// (V_orth'P V_orth) \ V_orth' K V_orth, where the preconditioner hand- 
// les the projection as a border.                                      
////////////////////////////////////////////////////////////////////////

  // check if the preconditioner can handle a bordered system. It has to
  // solve a system of the form                               
  // |A  V| |x|   |b|                                         
  // |V' O| |s| = |0| now to maintain orthogonality wrt V.    
  //                                                          
  Teuchos::RCP<HYMLS::BorderedSolver> borderedPrec =
        Teuchos::rcp_dynamic_cast<HYMLS::BorderedSolver>(precond_);
    
  if (Teuchos::is_null(borderedPrec) && (precond_!=Teuchos::null))
    {
    Tools::Error("preconditioner cannot handle bordering",__FILE__,__LINE__);
    }
  else if (borderedPrec!=Teuchos::null)
    {
    int nulDim = nullSpace_==Teuchos::null? 0: nullSpace_->NumVectors();
    if (numDeflated_>nulDim)
      {
      Teuchos::RCP<Epetra_SerialDenseMatrix> C = Teuchos::rcp(new
        Epetra_SerialDenseMatrix(numDeflated_,numDeflated_));
      CHECK_ZERO(borderedPrec->SetBorder(V_,V_,C));
      }
    }
    
  Aorth_=Teuchos::rcp(new ProjectedOperator(matrix_,V_,true));

  CHECK_ZERO(Teuchos::rcp_dynamic_cast<HYMLS::ProjectedOperator>(Aorth_)->SetLeftPrecond(precond_));

  /////////////////////////////////////////////////////////////////////////////////
  // construct the global LU factorization of the bordered system
  // |A_orth      V_orth'AV |
  // |V'AV_orth   V'AV      |
  /////////////////////////////////////////////////////////////////////////////////
  KV = Teuchos::rcp(new Epetra_MultiVector(*V_));

  // compute V'KV_orth (K is not symmetric, so we
  // actually compute borderW_' = V_orthK'V. V_orth=I-VV' is symmetric)
  CHECK_ZERO(matrix_->Multiply(true,*V_,*KV));
  borderW_=Teuchos::rcp(new Epetra_MultiVector(V_->Map(),numDeflated_));
  CHECK_ZERO(DenseUtils::ApplyOrth(*V_,*KV,*borderW_));

  // compute V_orth'KV
  CHECK_ZERO(matrix_->Multiply(false,*V_,*KV));
  borderV_=Teuchos::rcp(new Epetra_MultiVector(V_->Map(),numDeflated_));    
  CHECK_ZERO(DenseUtils::ApplyOrth(*V_,*KV,*borderV_));
  // compute C=V'KV
  borderC_=Teuchos::rcp(new Epetra_SerialDenseMatrix(numDeflated_,numDeflated_));
  CHECK_ZERO(DenseUtils::MatMul(*V_,*KV,*borderC_));

  // the actual LU-factorizatoin of the system is handled by a separate class,
  // but we need to provide an operator that can do A_orth.ApplyInverse by GMRES:
  bool status=true;
  try {
  belosProblemPtr_->setOperator(Aorth_);
  belosProblemPtr_->setLeftPrec(Teuchos::null);
  belosProblemPtr_->setRightPrec(Teuchos::null);
  CHECK_TRUE(belosProblemPtr_->setProblem(belosSol_,belosRhs_));
  } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr,status);
  if (status==false) Tools::Fatal("caught an exception updating Belos",__FILE__,__LINE__);
  Teuchos::ParameterList& belosList = PL().sublist("Iterative Solver");
    
  Teuchos::RCP<Teuchos::ParameterList> belosParamPtr = Teuchos::rcp(
        new Teuchos::ParameterList(belosList));
        
  if (solverType_=="GMRES")
    {
    belosParamPtr->set("Solver","BlockGmres");
    //belosParamPtr->set("Block Size",V_->NumVectors());
    // for some reason the block variant fails if there is
    // a null vector among the RHS's so that one of the solution
    // vectors converges in 1 iteration (TODO)
    belosParamPtr->set("Block Size",1);
    }
  else if (solverType_=="CG")
    {
    belosParamPtr->set("Solver","BlockCG");
    belosParamPtr->set("Block Size",V_->NumVectors());
    }
  else
    {
    Tools::Error("not implemented",__FILE__,__LINE__);
    }
try {
    AorthSolver_ = Teuchos::rcp(new HYMLS::Belos::EpetraOperator
        (belosProblemPtr_,belosParamPtr,
        precond_,V_));
    } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr,status);
  if (status==false) Tools::Fatal("failed to create Belos operator",__FILE__,__LINE__);
    
  // setup the LU decomposition
  status = true;
  try {
    LU_=Teuchos::rcp(new BorderedLU(AorthSolver_,borderV_,borderW_,borderC_));
    } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr,status);
  if (!status) Tools::Error("failed to create bordered outer solver",__FILE__,__LINE__);
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
    DEBUG("Solver: Bordered Solve");
    // bordered solve
    ierr=LU_->ApplyInverse(B,X);
    numIter_ = AorthSolver_->getNumIters();
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

    ::Belos::ReturnType ret;
    bool status=true;
    try {
    ret=belosSolverPtr_->solve();
    } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr, status);
    if (!status) Tools::Warning("caught an exception",__FILE__,__LINE__);

  numIter_ = belosSolverPtr_->getNumIters();

  //TODO: avoid this copy operation
  X=*belosSol_;

#ifdef TESTING
// compute explicit residual
int dim = PL("Problem").get<int>("Dimension");
int dof = PL("Problem").get<int>("Degrees of Freedom");

Epetra_MultiVector resid(belosRhs_->Map(),belosRhs_->NumVectors());
CHECK_ZERO(matrix_->Apply(*belosSol_,resid));
CHECK_ZERO(resid.Update(1.0,B,-1.0));
double *resNorm,*rhsNorm,*resNormV,*resNormP;
resNorm=new double[resid.NumVectors()];
resNormV=new double[resid.NumVectors()];
resNormP=new double[resid.NumVectors()];
rhsNorm =new double[resid.NumVectors()];
B.Norm2(rhsNorm);
resid.Norm2(resNorm);

if (dof>=dim)
  {
  Epetra_MultiVector residV=resid;
  Epetra_MultiVector residP=resid;
  for (int i=0;i<resid.MyLength();i+=dof)
    {
    for (int j=0;j<resid.NumVectors();j++)
      {
      for (int k=0;k<dim;k++)
        {
        residP[j][i+k]=0.0;
        }
      for (int k=dim+1;k<dof;k++)
        {
        residP[j][i+k]=0.0;
        }
      residV[j][i+dim]=0.0;
      }
    }  
  residV.Norm2(resNormV);
  residP.Norm2(resNormP);
  }

if (comm_->MyPID()==0)
  {
  Tools::out() << "Exp. res. norm(s): ";
  for (int ii=0;ii<resid.NumVectors();ii++)
    {
    Tools::out() << resNorm[ii] << " ";
    }
  Tools::out() << std::endl;
  Tools::out() << "Rhs norm(s): ";
  for (int ii=0;ii<resid.NumVectors();ii++)
    {
    Tools::out() << rhsNorm[ii] << " ";
    }
  Tools::out() << std::endl;
  if (dof>=dim)
    {
    Tools::out() << "Exp. res. norm(s) of V-part: ";
    for (int ii=0;ii<resid.NumVectors();ii++)
      {
      Tools::out() << resNormV[ii] << " ";
      }
    Tools::out() << std::endl;
    Tools::out() << "Exp. res. norm(s) of P-part: ";
    for (int ii=0;ii<resid.NumVectors();ii++)
      {
      Tools::out() << resNormP[ii] << " ";
      }
    Tools::out() << std::endl;
    }
  }
delete [] resNorm;
delete [] rhsNorm;
delete [] resNormV;
delete [] resNormP;
#endif

    if (ret!=::Belos::Converged)
      {
      HYMLS::Tools::Warning("Belos returned "+::Belos::convertReturnTypeToString(ret)+"'!",__FILE__,__LINE__);    
#ifdef TESTING
      Teuchos::RCP<const Epetra_CrsMatrix> Acrs =
        Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(matrix_);
      MatrixUtils::Dump(*Acrs,"FailedMatrix.txt");
      MatrixUtils::Dump(B,"FailedRhs.txt");
#endif
      ierr = -1;
      }
    }
  if (comm_->MyPID()==0)
    {
    Tools::Out("++++++++++++++++++++++++++++++++++++++++++++++++");
    Tools::Out("+ Number of iterations: "+Teuchos::toString(numIter_));
    Tools::Out("++++++++++++++++++++++++++++++++++++++++++++++++");
    Tools::Out("");
    }
  return ierr;
  }


}//namespace HYMLS
