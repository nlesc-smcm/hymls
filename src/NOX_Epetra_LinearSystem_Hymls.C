#include "NOX_Config.h"

#include "NOX_Epetra_LinearSystem_Hymls.H"	// class definition

#include "HYMLS_Solver.H"
#include "HYMLS_Preconditioner.H"

#include "BelosTypes.hpp"

// NOX includes
#include "NOX_Epetra_Interface_Required.H"
#include "NOX_Epetra_Interface_Jacobian.H"
#include "NOX_Epetra_Interface_Preconditioner.H"
#include "NOX_Epetra_MatrixFree.H"
#include "NOX_Epetra_FiniteDifference.H"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "NOX_Epetra_Scaling.H"
#include "NOX_Utils.H"

// External include files for Epetra, Belos, and Ifpack
#include "Epetra_Map.h"
#include "Epetra_Vector.h" 
#include "Epetra_Operator.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_LinearProblem.h"
#include "Teuchos_ParameterList.hpp"

// EpetraExt includes for dumping a matrix
//#ifdef HAVE_NOX_DEBUG
#ifdef HAVE_NOX_EPETRAEXT
#include "EpetraExt_BlockMapOut.h"
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_RowMatrixOut.h"
#endif
//#endif

#include <typeinfo>



// ***********************************************************************
NOX::Epetra::LinearSystemHymls::
LinearSystemHymls(
 Teuchos::ParameterList& printParams, 
 Teuchos::ParameterList& linearSolverParams, 
 const Teuchos::RCP<NOX::Epetra::Interface::Required>& iReq, 
 const NOX::Epetra::Vector& cloneVector,
 const Teuchos::RCP<NOX::Epetra::Scaling> s):
 LinearSystemAztecOO(printParams, linearSolverParams,
                     iReq, cloneVector, s)
{
  // this is not tested and should not be used.
  std::cerr << "this constructor should not be used"<<std::endl;
  throw "NOX Error";
  reset(linearSolverParams);
}

// ***********************************************************************
NOX::Epetra::LinearSystemHymls::
LinearSystemHymls(
 Teuchos::ParameterList& printParams, 
 Teuchos::ParameterList& linearSolverParams,  
 const Teuchos::RCP<NOX::Epetra::Interface::Required>& iReq, 
 const Teuchos::RCP<NOX::Epetra::Interface::Jacobian>& iJac, 
 const Teuchos::RCP<Epetra_Operator>& jacobian,
 const NOX::Epetra::Vector& cloneVector,
 const Teuchos::RCP<NOX::Epetra::Scaling> s):
  LinearSystemAztecOO(printParams, linearSolverParams,
                      iReq, iJac, jacobian, cloneVector, s)
{  
  // this is not tested and should not be used.
  std::cerr << "this constructor should not be used"<<std::endl;
  throw "NOX Error";
  reset(linearSolverParams);
}

// ***********************************************************************
NOX::Epetra::LinearSystemHymls::
LinearSystemHymls(
 Teuchos::ParameterList& printParams, 
 Teuchos::ParameterList& linearSolverParams, 
 const Teuchos::RCP<NOX::Epetra::Interface::Required>& iReq, 
 const Teuchos::RCP<NOX::Epetra::Interface::Preconditioner>& iPrec, 
 const Teuchos::RCP<Epetra_Operator>& preconditioner,
 const NOX::Epetra::Vector& cloneVector,
 const Teuchos::RCP<NOX::Epetra::Scaling> s,
 Teuchos::RCP<Epetra_CrsMatrix> massMatrix):
   LinearSystemAztecOO(printParams, linearSolverParams,
            iReq, iPrec, preconditioner, cloneVector, s),
            massMatrix_(massMatrix)
{  
  reset(linearSolverParams);
}

// ***********************************************************************
NOX::Epetra::LinearSystemHymls::
LinearSystemHymls(
 Teuchos::ParameterList& printParams, 
 Teuchos::ParameterList& linearSolverParams,
 const Teuchos::RCP<NOX::Epetra::Interface::Jacobian>& iJac, 
 const Teuchos::RCP<Epetra_Operator>& jacobian,
 const Teuchos::RCP<NOX::Epetra::Interface::Preconditioner>& iPrec, 
 const Teuchos::RCP<Epetra_Operator>& preconditioner,
 const NOX::Epetra::Vector& cloneVector,
 const Teuchos::RCP<NOX::Epetra::Scaling> s,
 Teuchos::RCP<Epetra_CrsMatrix> massMatrix):
   LinearSystemAztecOO(printParams, linearSolverParams, 
            iJac, jacobian, iPrec, preconditioner, cloneVector, s),
            massMatrix_(massMatrix)
{  
  reset(linearSolverParams);
}

// ***********************************************************************
NOX::Epetra::LinearSystemHymls::~LinearSystemHymls() 
{
// handled by base class destructor and RCP's
}

// ***********************************************************************

//TODO: reset discards everything, is that what we want?
void NOX::Epetra::LinearSystemHymls::
reset(Teuchos::ParameterList& p)
{  
  // do everything the base class wants to do
  LinearSystemAztecOO::reset(p);

  // retrieve User's Belos list, add more things
  Teuchos::ParameterList& hymlsList=p.sublist("HYMLS");
  Teuchos::ParameterList& belosList=hymlsList.sublist("Solver").sublist("Iterative Solver");

  bool verbose = utils.isPrintType(Utils::LinearSolverDetails);
  bool debug = utils.isPrintType(Utils::Debug);
  
  int verbosity = Belos::Errors + Belos::Warnings;
  if (verbose)
    { //TODO: where to put which option? how do we get readable output?
    verbosity+=Belos::TimingDetails+Belos::IterationDetails;
    verbosity+=Belos::StatusTestDetails+Belos::OrthoDetails+Belos::FinalSummary;
    }
  if (debug) verbosity+=Belos::Debug;

  // User is allowed to override these settings
  if (belosList.isParameter("Verbosity")==false)
    belosList.set("Verbosity",verbosity);

   Teuchos::RCP<std::ostream> out = 
        Teuchos::rcp(&(utils.pout()),false);
  
  belosList.set("Output Stream",out);

//  belosList.set("Output Style",Belos::Brief);
  belosList.set("Output Style",1);

  // NOX puts its adaptive choice of tolerance into this place:
  double tol = p.get("Tolerance",1.0e-6);
  // so we use it to override the settings in the Belos list.
  belosList.set("Convergence Tolerance",tol);

  Teuchos::RCP<const Epetra_RowMatrix> mat = 
        Teuchos::rcp_dynamic_cast<const Epetra_RowMatrix>(jacPtr);
  
  if (mat==Teuchos::null)
    {
    std::cerr << "NOX::Epetra::LinearSystemHymls requires an Epetra_RowMatrix!"<<std::endl;
    std::cerr << "("<<__FILE__<<", line "<< __LINE__<<")"<<std::endl;
    throw "NOX Error";
    }

  if (precPtr==Teuchos::null)
    {
    std::cerr << "NOX::Epetra::LinearSystemHymls requires a preconditioner!"<<std::endl;
    std::cerr << "("<<__FILE__<<", line "<< __LINE__<<")"<<std::endl;
    throw "NOX Error";
    }
  
  hymls_ = Teuchos::rcp(new 
        HYMLS::Solver(mat,precPtr,Teuchos::rcp(&hymlsList,false),1));
    
}

bool NOX::Epetra::LinearSystemHymls::
createPreconditioner(const NOX::Epetra::Vector& x, Teuchos::ParameterList& p, 
                     bool recomputeGraph) const
  {
  LinearSystemAztecOO::createPreconditioner(x,p,recomputeGraph);
  // setup deflation in the solver
  if (massMatrix_!=Teuchos::null)
    {
    hymls_->SetMassMatrix(massMatrix_);
    }
  hymls_->SetupDeflation();
  return true;
  }

bool NOX::Epetra::LinearSystemHymls::
recomputePreconditioner(const NOX::Epetra::Vector& x, Teuchos::ParameterList& p) const
  {
  LinearSystemAztecOO::recomputePreconditioner(x,p);
  // setup deflation in the solver
  if (massMatrix_!=Teuchos::null)
    {
    hymls_->SetMassMatrix(massMatrix_);
    }
  hymls_->SetupDeflation();
  return true;
  }

// ***********************************************************************

bool NOX::Epetra::LinearSystemHymls::
applyJacobianInverse(Teuchos::ParameterList &p,
		     const NOX::Epetra::Vector& input, 
		     NOX::Epetra::Vector& result)
{
  
  int ierr = 0;

  // AGS: Rare option, similar to Max Iters=1 but twice as fast.
    if ( p.get("Use Preconditioner as Solver", false) ) 
      return applyRightPreconditioning(false, p, input, result);

  double startTime = timer.WallTime();
  
  // Zero out the delta X of the linear problem if requested by user.
  if (zeroInitialGuess)
    result.init(0.0);

  // Create Epetra linear problem object (only used for scaling)

  // Need non-const version of the input vector
  // Epetra_LinearProblem requires non-const versions so we can perform
  // scaling of the linear problem.
  NOX::Epetra::Vector& nonConstInput = const_cast<NOX::Epetra::Vector&>(input);
        
  Epetra_LinearProblem Problem(jacPtr.get(),
                               &(result.getEpetraVector()),
                               &(nonConstInput.getEpetraVector()));

  // ************* Begin linear system scaling *******************
  if ( !Teuchos::is_null(scaling) ) {

    if ( !manualScaling )
      scaling->computeScaling(Problem);
    
    scaling->scaleLinearSystem(Problem);

    if (utils.isPrintType(Utils::Details)) {
      utils.out() << *scaling << endl;
    }  
  }
  // ************* End linear system scaling *******************

  // Use EpetraExt to dump linear system if debuggging
#ifdef HAVE_NOX_DEBUG
#ifdef HAVE_NOX_EPETRAEXT

  ++linearSolveCount;
  std::std::ostringstream iterationNumber;
  iterationNumber << linearSolveCount;
    
  std::string prefixName = p.get("Write Linear System File Prefix", 
				 "NOX_LinSys");
  std::string postfixName = iterationNumber.str();
  postfixName += ".mm";

  if (p.get("Write Linear System", false)) {

    std::string mapFileName = prefixName + "_Map_" + postfixName;
    std::string jacFileName = prefixName + "_Jacobian_" + postfixName;    
    std::string rhsFileName = prefixName + "_RHS_" + postfixName;
    
    Epetra_RowMatrix* printMatrix = NULL;
    printMatrix = dynamic_cast<Epetra_RowMatrix*>(jacPtr.get()); 

    if (printMatrix == NULL) {
      cout << "Error: NOX::Epetra::LinearSystemAztecOO::applyJacobianInverse() - "
	   << "Could not cast the Jacobian operator to an Epetra_RowMatrix!"
	   << "Please set the \"Write Linear System\" parameter to false."
	   << endl;
      throw "NOX Error";
    }

    EpetraExt::BlockMapToMatrixMarketFile(mapFileName.c_str(), 
					  printMatrix->RowMatrixRowMap()); 
    EpetraExt::RowMatrixToMatrixMarketFile(jacFileName.c_str(), *printMatrix, 
					   "test matrix", "Jacobian XXX");
    EpetraExt::MultiVectorToMatrixMarketFile(rhsFileName.c_str(), 
					     nonConstInput.getEpetraVector());

  }
#endif
#endif

  // Make sure preconditioner was constructed if requested
  if (!isPrecConstructed && (precAlgorithm != None_)) {
    throwError("applyJacobianInverse", 
       "Preconditioner is not constructed!  Call createPreconditioner() first.");
  }
  

  // do Belos solve
  if (utils.isPrintType(Utils::Debug)) {  
    utils.out() << "**************************************"<<std::endl;
    utils.out() << "* HYMLS Parameter List               *"<<std::endl;
    utils.out() << "**************************************"<<std::endl;
    utils.out() << p.sublist("HYMLS");
    utils.out() << "**************************************"<<std::endl;
  }
  //
  // Perform solve
  //
  
  
  // this may change from solve to solve and is not in the Belos list.
  // it is set by NOX if "Forcing Term Method" is not "Constant, for instance.
  double tol = p.get("Tolerance",1.0e-6);
  Teuchos::ParameterList& hymlsList = p.sublist("HYMLS");
  Teuchos::ParameterList& belosList = hymlsList.sublist("Iterative Solver");
  belosList.set("Convergence Tolerance",tol);
  
  hymls_->setParameterList(Teuchos::rcp(&hymlsList, false));
  
  Epetra_Vector& sol = result.getEpetraVector();
  const Epetra_Vector& rhs = input.getEpetraVector();
  
  ierr=hymls_->ApplyInverse(rhs,sol);  
  
  if (ierr!=0) {
    utils.out() << std::endl << "WARNING:  HYMLS returned "<<ierr << std::endl;
  }

//TODO: this is not implemented yet

  // Unscale the linear system
  if ( !Teuchos::is_null(scaling) )
    scaling->unscaleLinearSystem(Problem);

  // Set the output parameters in the "Output" sublist
  if (outputSolveDetails) {
    Teuchos::ParameterList& outputList = p.sublist("Output");
    int prevLinIters = 
      outputList.get("Total Number of Linear Iterations", 0);
    int curLinIters = 0;
    double achievedTol = -1.0;
 //   curLinIters = hymls_->getNumIters();
 //   for ( int i=0; i<numrhs; i++) {
 //     double actRes = actual_resids[i]/rhs_norm[i];
 //     utils.out()<<"Problem "<<i<<" : \t"<< actRes <<std::endl;
 //     if (actRes > achievedTol) achievedTol = actRes;
 //   }

    outputList.set("Number of Linear Iterations", curLinIters);
    outputList.set("Total Number of Linear Iterations", 
			    (prevLinIters + curLinIters));
    outputList.set("Achieved Tolerance", achievedTol);
  }

  // Dump solution of linear system
#ifdef HAVE_NOX_DEBUG
#ifdef HAVE_NOX_EPETRAEXT
  if (p.get("Write Linear System", false)) {
    std::string lhsFileName = prefixName + "_LHS_" + postfixName;
    EpetraExt::MultiVectorToMatrixMarketFile(lhsFileName.c_str(), 
					   result.getEpetraVector());
  }
#endif
#endif

  double endTime = timer.WallTime();
  timeApplyJacbianInverse += (endTime - startTime);

  return (ierr==0);
}

// ***********************************************************************

void
NOX::Epetra::LinearSystemHymls::setAztecOOJacobian() const
{
Teuchos::RCP<const Epetra_RowMatrix> matPtr = 
        Teuchos::rcp_dynamic_cast<const Epetra_RowMatrix>(jacPtr);
if (matPtr!=Teuchos::null)
  {
  hymls_->SetMatrix(matPtr);
  }
else
  {
  std::cerr << "NOX/HYMLS interface requires an Epetra_RowMatrix"<<std::endl;
  std::cerr << "("<<__FILE__<<", line "<<__LINE__<<")"<<std::endl;
  throw "NOX Error";
  }
}

// ***********************************************************************
void
NOX::Epetra::LinearSystemHymls::setAztecOOPreconditioner() const
{  
if (!Teuchos::is_null(solvePrecOpPtr))
  {
  hymls_->SetPrecond(solvePrecOpPtr);
  }
}

// ***********************************************************************

