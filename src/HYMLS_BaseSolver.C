//#define BLOCK_IMPLEMENTATION 1
#include "HYMLS_BaseSolver.H"

#include "HYMLS_Tools.H"
#include "HYMLS_MatrixUtils.H"
#include "HYMLS_DenseUtils.H"
#include "HYMLS_BorderedPreconditioner.H"
#include "HYMLS_BorderedLU.H"
#include "HYMLS_ProjectedOperator.H"
#include "HYMLS_ShiftedOperator.H"

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

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Utils.hpp"

#include "BelosBlockGmresSolMgr.hpp"
#include "BelosTypes.hpp"
#include "BelosBlockCGSolMgr.hpp"
//#include "BelosPCPGSolMgr.hpp"
#include "HYMLS_BelosEpetraOperator.H"

#include "Teuchos_StandardParameterEntryValidators.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include "HYMLS_no_debug.H"


namespace HYMLS {

// constructor
BaseSolver::BaseSolver(Teuchos::RCP<const Epetra_RowMatrix> K,
  Teuchos::RCP<Epetra_Operator> P,
  Teuchos::RCP<Teuchos::ParameterList> params,
  int numRhs, bool validate)
  :
  PLA("Solver"), comm_(Teuchos::rcp(&(K->Comm()),false)),
  matrix_(K), operator_(K), precond_(P),
  shiftA_(1.0), shiftB_(0.0),
  massMatrix_(Teuchos::null), nullSpace_(Teuchos::null),
  V_(Teuchos::null), W_(Teuchos::null),
  useTranspose_(false), normInf_(-1.0), numIter_(0),
  label_("HYMLS::BaseSolver"),
  lor_default_("Right")
  {
  HYMLS_PROF3(label_,"Constructor");
  setParameterList(params, validate && validateParameters_);
  
  belosRhs_=Teuchos::rcp(new Epetra_MultiVector(matrix_->OperatorRangeMap(),numRhs));
  belosSol_=Teuchos::rcp(new Epetra_MultiVector(matrix_->OperatorDomainMap(),numRhs));

  belosProblemPtr_=Teuchos::rcp(new belosProblemType_(operator_,belosSol_,belosRhs_));
  
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
    int blockSize=belosList.get("Block Size",1);
    int numBlocks=belosList.get("Num Blocks",300);
    REPORT_MEM(label_,"GMRES (estimate)",numBlocks*blockSize*matrix_->NumMyRows(),0);
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

void BaseSolver::SetMatrix(Teuchos::RCP<const Epetra_RowMatrix> A)
  {
  HYMLS_PROF3(label_,"SetMatrix");
  matrix_ = A;
  if (shiftB_!=0.0 || shiftA_!=1.0)
    {
    Tools::Warning("SetMatrix called while operator used is shifted."
      "Discarding shifts.",__FILE__,__LINE__);
    shiftB_=0.0;
    shiftA_=1.0;
    }
  operator_=matrix_;
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

  belosPrecPtr_=Teuchos::rcp(new belosPrecType_(precond_));
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
  if (mass==Teuchos::null) return;
  HYMLS_PROF3(label_,"SetMassMatrix");
  if (mass->RowMatrixRowMap().SameAs(matrix_->RowMatrixRowMap()))
    {
    massMatrix_ = mass;
    }
  else
    {
    Tools::Error("Mass matrix must have same row map as solver",
      __FILE__,__LINE__);
    }
  if (shiftB_!=0.0 || shiftA_!=1.0)
    {
    Tools::Warning("SetMassMatrix called while solving shifted system."
      "Discarding shifts.",__FILE__,__LINE__);
    shiftB_=0.0;
    shiftA_=1.0;
    }
  operator_=matrix_;
  }

void BaseSolver::setShift(double shiftA, double shiftB)
  {
  shiftA_=shiftA;
  shiftB_=shiftB;
  operator_=Teuchos::rcp(new ShiftedOperator(matrix_,massMatrix_,shiftA_,shiftB_));
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

int BaseSolver::setNullSpace(Teuchos::RCP<const Epetra_MultiVector> const &V)
  {
  if (!nullSpace_.is_null() && !V.is_null())
    {
      Tools::Warning("The null space was already set before calling setNullSpace",
        __FILE__, __LINE__);
    }
  else if (!V_.is_null())
    {
      Tools::Warning("The null space was set after a border was already present."
      " Please set the border after setting the null space",
        __FILE__, __LINE__);
    }

  // Reset the null space when the preconditioner is changed for instance
  if (V.is_null())
    {
    Teuchos::RCP<const Epetra_MultiVector> nullSpace = nullSpace_;
    nullSpace_ = Teuchos::null;
    CHECK_ZERO(addBorder(nullSpace, nullSpace));
    nullSpace_ = nullSpace;

    return 0;
    }

  nullSpace_ = Teuchos::null;
  CHECK_ZERO(addBorder(V, V));
  nullSpace_ = V;

  return 0;
  }

int BaseSolver::addBorder(Teuchos::RCP<const Epetra_MultiVector> const &V,
  Teuchos::RCP<const Epetra_MultiVector> const &W)
  {
  if (V.is_null())
    {
    return 0;
    }

  if (nullSpace_.is_null())
    {
    V_ = V;
    W_ = W;
    if (W.is_null())
      {
      W_ = V;
      }
    }
  else
    {
    // Expand the nullspace with the border that was added here
    int dim0 = nullSpace_->NumVectors();
    int dim1 = V->NumVectors();

    V_ = Teuchos::rcp(new Epetra_MultiVector(OperatorRangeMap(), dim0 + dim1));

    Epetra_MultiVector V0(View, *V_, 0, dim0);
    Epetra_MultiVector V1(View, *V_, dim0, dim1);
    V0 = *nullSpace_;
    V1 = *V;

    if (!W.is_null())
      {
      if (W->NumVectors() != dim1)
        {
        Tools::Error("Borders have unequal dimensions",
          __FILE__, __LINE__);
        }

      W_ = Teuchos::rcp(new Epetra_MultiVector(OperatorRangeMap(), dim0 + dim1));

      Epetra_MultiVector W0(View, *W_, 0, dim0);
      Epetra_MultiVector W1(View, *W_, dim0, dim1);
      W0 = *nullSpace_;
      W1 = *W;
      }
    else
      {
      W_ = V_;
      }
    }

  // Set the border for the matrix and the preconditioner
  Teuchos::RCP<HYMLS::BorderedPreconditioner> bprec
    = Teuchos::rcp_dynamic_cast<BorderedPreconditioner>(precond_);
  if (bprec!=Teuchos::null)
    {
    CHECK_ZERO(bprec->setBorder(V_, W_));
    }
  Aorth_ = Teuchos::rcp(new ProjectedOperator(operator_, V_, W_, true));
  belosProblemPtr_->setOperator(Aorth_);

  return 0;
  }

int BaseSolver::SetupDeflation(int maxEigs)
  {
  Tools::Error("Functionality moved to DeflatedSolver",  __FILE__, __LINE__);
#if 0
  // by default leave numEigs_ at its present value:
  if (maxEigs!=-2) numEigs_=maxEigs;
  
  // If there is a null-space, deflate it.
  // If no NS and no additional vectors asked for -
  // nothing to be done.
  if (numEigs_==0 && nullSpace_ == Teuchos::null) return 0;
  HYMLS_PROF(label_,"SetupDeflation");

  setNullSpace();

  Teuchos::RCP<Epetra_MultiVector> KV;
        
  // add null space as border to preconditioner
  if (V_!=Teuchos::null)
    {
    // we compute eigs of the operator [K N; N' 0] to make the operator nonsingular
    Teuchos::RCP<HYMLS::BorderedPreconditioner> bprec
      = Teuchos::rcp_dynamic_cast<BorderedPreconditioner>(precond_);
    if (bprec!=Teuchos::null)
      {
      CHECK_ZERO(bprec->setBorder(V_,W_));
      }
    else if (precond_!=Teuchos::null)
      {
      Tools::Error("feature not implemented",__FILE__,__LINE__);
      // should work in principle, but we can handle borders smarter
      /*
        Tools::Out("add null space to preconditioner by LU");
        op = Teuchos::rcp(new BorderedLU(precond_,V_,W_));
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

#ifdef HYMLS_DEBUGGING
    HYMLS_DEBUG("precEigs_->Evals = ...");
    for (unsigned int i=0;i<precEigs_->Evals.size();i++)
      {
      HYMLS_DEBUG(precEigs_->Evals[i].realpart << "\t"<<precEigs_->Evals[i].imagpart);
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
#ifdef HYMLS_TESTING
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

    HYMLS_DEBUG("V'inv(P)KV, should be close to identity if preconditioner is good:");
    HYMLS_DEBVAR(C);

    // keep those that are further away from 1 then the "Deflation Threshold"
    std::vector<bool> keep(n);
    numDeflated_=n;
    double re,im,tol=deflThres_*deflThres_;
    for (int i=0;i<n;i++)
      {
      re=1.0-lambda_r[i]; im = lambda_i[i];
#ifdef HYMLS_TESTING
      Tools::out() << "lambda(V'P\\SV)["<<i<<"]=";
      Tools::out() << lambda_r[i] << " + i (" << lambda_i[i] << ")"<<std::endl;
      HYMLS_DEBUG("lambda(V'P\\SV)["<<i<<"]="<<lambda_r[i] << " + i (" << lambda_i[i] << ")");
#endif
      if ((re*re+im*im)>tol)
        {
        Tools::out() << "deflating eigenmode "<<i<<": lambda(V'P\\SV)=";
        Tools::out() << lambda_r[i] << " + i (" << lambda_i[i] << ")"<<std::endl;
        keep[i]=true;
        HYMLS_DEBUG("deflated");
        }
      else
        {
        HYMLS_DEBUG("not deflated");
        numDeflated_--;
        keep[i]=false;
        }
      }
    Tools::Out("number of eigenmodes to be deflated: "+Teuchos::toString(numDeflated_));
    if (numDeflated_==0)
      {
      numDeflated_=V_->NumVectors();
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
      int dimNull = V_->NumVectors();
      Teuchos::RCP<Epetra_MultiVector> Vnew = Teuchos::rcp(
        new Epetra_MultiVector(V.Map(), numDeflated_ + dimNull));
      for (int k = 0; k < dimNull; k++)
        *((*Vnew)(k)) = *((*V_)(k));
      for (int k = 0; k < numDeflated_; k++)
        *((*Vnew)(dimNull+k)) = *(V_hat(k));
      numDeflated_ += dimNull;
      V_ = Vnew;
      }
    }
  else
    {
    numDeflated_=V_->NumVectors();
    }

#ifdef HYMLS_STORE_MATRICES
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
  Teuchos::RCP<HYMLS::BorderedPreconditioner> borderedPrec =
    Teuchos::rcp_dynamic_cast<HYMLS::BorderedPreconditioner>(precond_);
    
  if (Teuchos::is_null(borderedPrec) && (precond_!=Teuchos::null))
    {
    Tools::Error("preconditioner cannot handle bordering",__FILE__,__LINE__);
    }
  else if (borderedPrec!=Teuchos::null)
    {
    int nulDim = V_ == Teuchos::null ? 0 : V_->NumVectors();
    if (numDeflated_ > nulDim)
      {
      Teuchos::RCP<Epetra_SerialDenseMatrix> C = Teuchos::rcp(new
        Epetra_SerialDenseMatrix(numDeflated_, numDeflated_));
      CHECK_ZERO(borderedPrec->setBorder(V_, W_, C));
      } // otherwise this was done above (only deflate a given null space)
    }

  Aorth_=Teuchos::rcp(new ProjectedOperator(operator_,V_,Teuchos::null,true));

  // TODO: Implement right preconditioning because at the moment only left preconditioning works
  CHECK_ZERO(Teuchos::rcp_dynamic_cast<HYMLS::ProjectedOperator>(Aorth_)->SetLeftPrecond(precond_));

  /////////////////////////////////////////////////////////////////////////////////
  // construct the global LU factorization of the bordered system
  // |A_orth      V_orth'AV |
  // |V'AV_orth   V'AV      |
  /////////////////////////////////////////////////////////////////////////////////
  KV = Teuchos::rcp(new Epetra_MultiVector(*V_));

  // compute V'KV_orth (K is not symmetric, so we
  // actually compute borderW_' = V_orthK'V. V_orth=I-VV' is symmetric)
  Teuchos::RCP<Epetra_Operator> op =
    Teuchos::rcp_const_cast<Epetra_Operator>(operator_);
  bool trans = op->UseTranspose();
  op->SetUseTranspose(!trans);
  CHECK_ZERO(op->Apply(*V_,*KV));
  op->SetUseTranspose(trans);
  borderW_=Teuchos::rcp(new Epetra_MultiVector(V_->Map(),numDeflated_));
  CHECK_ZERO(DenseUtils::ApplyOrth(*V_,*KV,*borderW_));

  // compute V_orth'KV
  CHECK_ZERO(operator_->Apply(*V_,*KV));
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
#endif
  return 0;
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

    belosPrecPtr_=Teuchos::rcp(new belosPrecType_(newPrec));
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
  CHECK_TRUE(belosProblemPtr_->setProblem(belosSol_,belosRhs_));
  return 0;
  }

// Applies the preconditioner to vector X, returns the result in Y.
int BaseSolver::ApplyInverse(const Epetra_MultiVector& B,
  Epetra_MultiVector& X) const
  {
  HYMLS_PROF(label_,"ApplyInverse");
  int ierr = 0;
  if (!nullSpace_.is_null() && V_.is_null())
    {
      Tools::Error("You need to call setNullSpace on the solver after "
        "the preconditioner has been computed", __FILE__, __LINE__);
    }

  if (X.NumVectors()!=belosRhs_->NumVectors())
    {
    int numRhs=X.NumVectors();
    belosRhs_=Teuchos::rcp(new Epetra_MultiVector(matrix_->OperatorRangeMap(),numRhs));
    belosSol_=Teuchos::rcp(new Epetra_MultiVector(matrix_->OperatorDomainMap(),numRhs));
    }
#ifdef HYMLS_TESTING
  if (X.NumVectors()!=B.NumVectors())
    {
    Tools::Error("different number of input and output vectors",__FILE__,__LINE__);
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
#ifdef HYMLS_DEBUGGING
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

  // Make the initial guess orthogonal to the V_ space
  if (V_ != Teuchos::null)
    {
    CHECK_ZERO(DenseUtils::ApplyOrth(*V_, *belosSol_, X, W_));
    *belosSol_ = X;
    }

  CHECK_TRUE(belosProblemPtr_->setProblem(belosSol_, belosRhs_));

  if (LU_ != Teuchos::null)
    {
    HYMLS_DEBUG("BaseSolver: Bordered Solve");
    // bordered solve. We use belosSol_ here because it contains
    // the initial guess. Not used at the moment

    // ierr=LU_->ApplyInverse(B, X);
    ierr=LU_->ApplyInverse(*belosRhs_, *belosSol_);

    //TODO: avoid this copy operation
    X = *belosSol_;

    numIter_ = AorthSolver_->getNumIters();
    }
  else
    {
    ::Belos::ReturnType ret = ::Belos::Unconverged;
    bool status = true;
    try {
      ret = belosSolverPtr_->solve();
      } TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, status);
    if (!status) Tools::Warning("caught an exception", __FILE__, __LINE__);

    numIter_ = belosSolverPtr_->getNumIters();

    //TODO: avoid this copy operation
    X = *belosSol_;

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
  Tools::out()<<"we were solving (a*A*x+b*B)*x=rhs\n" <<
    "   with "<<X.NumVectors()<<" rhs\n" <<
    "        a = "<<shiftA_<<"\n" <<
    "        b = "<<shiftB_<<"\n";
  if (massMatrix_==Teuchos::null)
    Tools::out()<<
      "        B = I\n"; 
// compute explicit residual
  int dim = PL("Problem").get<int>("Dimension");
  int dof = PL("Problem").get<int>("Degrees of Freedom");

  Epetra_MultiVector resid(X.Map(),X.NumVectors());
  CHECK_ZERO(matrix_->Apply(X,resid));
  if (shiftB_!=0.0)
    {
    Epetra_MultiVector Bx=X;
  
    if (massMatrix_!=Teuchos::null)
      {
      CHECK_ZERO(massMatrix_->Apply(X,Bx));
      }
    CHECK_ZERO(resid.Update(shiftB_,Bx,shiftA_));
    }
  else if (shiftA_!=1.0)
    {
    CHECK_ZERO(resid.Scale(shiftA_));
    }
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


  return ierr;
  }

  }//namespace HYMLS
