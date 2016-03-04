/**********************************************************************
 * Copyright by Jonas Thies, Univ. of Groningen 2006/7/8.             *
 * Permission to use, copy, modify, redistribute is granted           *
 * as long as this header remains intact.                             *
 * contact: jonas@math.rug.nl                                         *
 **********************************************************************/
 
/************************************************************************
//OCEAN MODEL 
************************************************************************/

#include <iostream>

#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_MultiVector.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_LinearProblem.h"
#include "AztecOO.h"
#include "LOCA_Epetra.H"
#include "LOCA.H"
#include "LOCA_Parameter_SublistParser.H"

#include "HYMLS_Tools.H"
#include "HYMLS_HyperCube.H"
#include "NOX_Epetra_LinearSystem_Hymls.H"
//#include "NOX_Epetra_LinearSystem_Belos.H"

#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#include "HYMLS_MatrixUtils.H"
#include <sstream>

#include "HYMLS_ExampleLocaInterface.H"

#include "Epetra_LinearProblem.h"

int main(int argc, char *argv[])
  {

#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
  
  int ierr = 0;   
  
////////////////////////////////////////
// Create the Parallel  Communicator  //
////////////////////////////////////////

#ifdef HAVE_MPI
//Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  HYMLS::HyperCube Topology;
  Teuchos::RCP<Epetra_MpiComm> Comm = Topology.GetComm();
#else
  Teuchos::RCP<Epetra_SerialComm> Comm=rcp(new Epetra_SerialComm());
#endif

  //Get process ID and total number of processes
  int MyPID = Comm->MyPID();
  int NumProc = Comm->NumProc();

// write to stdout, better because we can run several jobs in one directory
// on machines with queueing systems.
#ifndef OUTPUT_TO_FILE
  HYMLS::Tools::InitializeIO(Comm);
#else
  Teuchos::RCP<std::ostream> contstream;
  if (Comm->MyPID()==0)
    {

    contstream = Teuchos::rcp(new std::ofstream("continuation.out"));
    }
  else
    {
    contstream = Teuchos::rcp(new Teuchos::oblackholestream());
    }

  // this allows us to use the DEBUG macro, the Tools::Out function etc.
  HYMLS::Tools::InitializeIO_std(Comm, contstream);
#endif
  HYMLS::Tools::out() << "HYMLS revision: "<<HYMLS::Tools::Revision()<<std::endl;

  std::string param_file="params.xml";
  if (argc>1)
    {
    param_file=std::string(argv[1]);
    }

  HYMLS::Tools::out() << "reading parameters from "<<param_file<<std::endl;

#ifdef HAVE_MPI
  //HYMLS::Tools::out() << Topology << std::endl;
#endif  

  DEBUG("*********************************************")
    DEBUG("* Debugging output for process "<<MyPID)
    DEBUG("* To prevent this file from being written,  ")
    DEBUG("* omit the -DDEBUGGING flag when compiling. ")
    DEBUG("*********************************************")
  
    bool stat=true;
  try {
  
    ////////////////////////////////////////////////////////
    // Setup Parameter Lists                              //
    ////////////////////////////////////////////////////////
  
    // read parameters from files 

    // we create one big parameter list and further down we will set some things
    // that are not straight-forwars in XML (verbosity etc.)
    Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new 
      Teuchos::ParameterList);

    // override default settings with parameters from user input file
    Teuchos::updateParametersFromXmlFile(param_file,paramList.ptr());

    // extract the final sublists:
            
    // Get the model sublist
    Teuchos::ParameterList& modelList = paramList->sublist("Model");

    // Get the LOCA sublist
    Teuchos::ParameterList& locaParamsList = paramList->sublist("LOCA");
    
    // get the Bifurcation sublist
    Teuchos::ParameterList& bifurcationList = locaParamsList.sublist("Bifurcation");
    
    // get the Stepper sublist
    Teuchos::ParameterList& stepperList = locaParamsList.sublist("Stepper");
    string cont_param = stepperList.get("Continuation Parameter","Undefined");
    double start_value = stepperList.get("Initial Value",0.0);

    // Get Anasazi Eigensolver sublist (needs --with-loca-anasazi)
    Teuchos::ParameterList& aList = stepperList.sublist("Eigensolver");
    aList.set("Verbosity", Anasazi::Errors+Anasazi::IterationDetails+Anasazi::Warnings+Anasazi::FinalSummary);

    // Get the "Solver" parameters sublist to be used with NOX Solvers
    Teuchos::ParameterList& nlParams = paramList->sublist("NOX");

    Teuchos::ParameterList& printParams = nlParams.sublist("Printing");
    printParams.set("MyPID", MyPID);
#ifdef OUTPUT_TO_FILE
    printParams.set("Output Stream",contstream);
    printParams.set("Error Stream",contstream);
#endif    
    printParams.set("Output Process",0);
    printParams.set("Output Information",
      NOX::Utils::Details + 
      NOX::Utils::OuterIteration + 
      NOX::Utils::InnerIteration +
      NOX::Utils::OuterIterationStatusTest + 
      NOX::Utils::LinearSolverDetails + 
#ifdef DEBUGGING
      NOX::Utils::Debug +
      NOX::Utils::StepperParameters +
#endif                          
      NOX::Utils::Warning +
      NOX::Utils::StepperDetails +
      NOX::Utils::StepperIteration); 


    //Create the "Direction" sublist for the "Line Search Based" solver
    Teuchos::ParameterList& dirParams = nlParams.sublist("Direction");

    //Create the "Line Search" sublist for the "Line Search Based" solver
    Teuchos::ParameterList& searchParams = nlParams.sublist("Line Search");

    //Create the "Direction" sublist for the "Line Search Based" solver
    Teuchos::ParameterList& newtParams = dirParams.sublist("Newton");

    //Create the "Linear Solver" sublist for the "Direction' sublist
    Teuchos::ParameterList& lsParams = newtParams.sublist("Linear Solver");

///////////////////////////////////////////////////////////
// Setup the Problem                                     //
///////////////////////////////////////////////////////////
        
    // put the correct starting value for the continuation parameter
    // in the thcm-list
    modelList.sublist("Starting Parameters").set(cont_param,start_value);
        
    // these LOCA data structures aren't really used by the OceanModel,
    // but are required for the ModelEvaluatorInterface

    // Create Epetra factory
    Teuchos::RCP<LOCA::Abstract::Factory> epetraFactory =
      Teuchos::rcp(new LOCA::Epetra::Factory);

    // Create global data object
    Teuchos::RCP<LOCA::GlobalData> globalData =
      LOCA::createGlobalData(paramList, epetraFactory);

    
    // the model serves as Preconditioner factory if "User Defined" is selected
     std::string PrecType = lsParams.get("Preconditioner","HYMLS");
     
      {
      HYMLS::Tools::Warning("overriding parameter value for 'Preconditioner' by 'HYMLS'\n",__FILE__,__LINE__);
      }
     // lsParams.set("Preconditioner","HYMLS");
//      lsParams.set("Preconditioner","New Ifpack");

    
  // note: we inherit globalData from LOCA's ModelEvaluatorInterface
  Teuchos::RCP<Teuchos::ParameterList> sharedParams = Teuchos::rcp(new Teuchos::ParameterList(*paramList));
  LOCA::Parameter::SublistParser parsedParams(globalData);
  parsedParams.parseSublists(Teuchos::rcp(sharedParams.get(),false));              
    // (i.e. backup in regular intervals)
    modelList.set("Parameter Name",cont_param);                                     
    modelList.set("Parameter Initial Value",start_value);

    //Teuchos::ParameterList& hymlsList=lsParams.sublist("HYMLS");

     Teuchos::RCP<Teuchos::ParameterList> hymlsList=Teuchos::null;

       hymlsList = Teuchos::rcp(&lsParams,false);
     
    Teuchos::RCP<HYMLS::exampleLocaInterface> model =
      Teuchos::rcp(new HYMLS::exampleLocaInterface(Comm, modelList, hymlsList, sharedParams));
 
    //Get the vector from the problem
    Teuchos::RCP<Epetra_Vector> soln = model->getSolution();

    //Create the Epetra_RowMatrix for the Jacobian/Preconditioner
    Teuchos::RCP<Epetra_CrsMatrix> A = model->getJacobian();
    Teuchos::RCP<Epetra_CrsMatrix> B = model->getMassMatrix();

    Teuchos::RCP<LOCA::ParameterVector> pVector=model->getParameterVector();

// NOX/LOCA interface setup

    //Teuchos::RCP<NOX::Abstract::PrePostOperator> prepost = model;
          
    // register pre- and post operations
//    nlParams.sublist("Solver Options").set("User Defined Pre/Post Operator",prepost);

    // register our own eigenvalue output routine
//    Teuchos::RCP<LOCA::SaveEigenData::AbstractStrategy> eigOut = model;
    
//    aList.set("Save Eigen Data Method","User-Defined");
//    aList.set("User-Defined Save Eigen Data Name","my eig-out object");
///    aList.set("my eig-out object",eigOut);
      
    Teuchos::RCP<LOCA::Epetra::Interface::TimeDependent> iReq = model;
    Teuchos::RCP<NOX::Epetra::Interface::Jacobian> iJac = model;
  
    
    Teuchos::RCP<NOX::Epetra::LinearSystem> linsys;

    if (PrecType == "User Defined")
      {
      DEBUG("user defined preconditioning");
      Teuchos::RCP<Epetra_Operator> myPrecOperator = model->getPreconditioner();
      Teuchos::RCP<NOX::Epetra::Interface::Preconditioner> iPrec = model;

      if (Teuchos::is_null(myPrecOperator))
        {
        HYMLS::Tools::Error("preconditioner is null!",__FILE__,__LINE__);
        }

      Teuchos::RCP<NOX::Epetra::Scaling> scaling=Teuchos::null;

//      linsys = Teuchos::rcp(new NOX::Epetra::LinearSystemBelos(printParams,
//                        lsParams, iJac, A, iPrec, myPrecOperator, soln, scaling));
      linsys = Teuchos::rcp(new NOX::Epetra::LinearSystemHymls(printParams,
          lsParams, iJac, A, iPrec, myPrecOperator, soln, scaling,
          B));
      }
     else
      {
      DEBUG("Trilinos preconditioning not implemented");
      return -1;
/*
  Teuchos::RCP<NOX::Epetra::Scaling> scaling = Teuchos::null;
  linsys = Teuchos::rcp(new NOX::Epetra::LinearSystemBelos(
  printParams, lsParams, iReq, iJac, A, soln,scaling));
*/
      }

    // we use the same linear system for the shift-inverted operator
    Teuchos::RCP<NOX::Epetra::LinearSystem> shiftedLinSys = linsys;
    
    //Create the loca vector
    NOX::Epetra::Vector locaSoln(soln);
    
    // Create the Group
    Teuchos::RCP<LOCA::Epetra::Group> grp =
      Teuchos::rcp(new LOCA::Epetra::Group(globalData, printParams,
          iReq, locaSoln, linsys, shiftedLinSys,
          *pVector));

    grp->computeF();

    double TolNewton = nlParams.get("Convergence Tolerance",1.0e-9);

    // Set up the Solver Convergence tests
    Teuchos::RCP<NOX::StatusTest::NormF> wrms =
      Teuchos::rcp(new NOX::StatusTest::NormF(TolNewton));
    Teuchos::RCP<NOX::StatusTest::MaxIters> maxiters
      = Teuchos::rcp(new NOX::StatusTest::MaxIters(searchParams.get("Max Iters", 10)));
    Teuchos::RCP<NOX::StatusTest::Combo> comboOR = 
      Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR)); 
    comboOR->addStatusTest(wrms);
    comboOR->addStatusTest(maxiters);


    // Create the stepper  
    LOCA::Stepper stepper(globalData, grp, comboOR, paramList);



//Teuchos::writeParameterListToXmlFile(*paramList,"parameters.xml");

    HYMLS::Tools::Out("\n*****************************");
    HYMLS::Tools::Out("Start Continuation process...");
    HYMLS::Tools::Out("*****************************\n\n");

    // Perform continuation run
    LOCA::Abstract::Iterator::IteratorStatus status;
      {
      HYMLS_PROF("main","continuation run");
      status = stepper.run();
      }   
    if (status != LOCA::Abstract::Iterator::Finished) {
      if (globalData->locaUtils->isPrintType(NOX::Utils::Error))
        globalData->locaUtils->out()
          << "Stepper failed to converge!" << std::endl;
      }
    globalData->locaUtils->out() << "Continuation status -> \t"<< status << std::endl;

    // Output the parameter list
    if (globalData->locaUtils->isPrintType(NOX::Utils::StepperParameters)) {
      globalData->locaUtils->out()
        << std::endl << "Final Parameters" << std::endl
        << "****************" << std::endl;
      stepper.getList()->print(globalData->locaUtils->out());
      globalData->locaUtils->out() << std::endl;
      }

    // Get the final solution from the stepper
    Teuchos::RCP<const LOCA::Epetra::Group> finalGroup = 
      Teuchos::rcp_dynamic_cast<const LOCA::Epetra::Group>(stepper.getSolutionGroup());
    const NOX::Epetra::Vector& finalSolutionNOX = 
      dynamic_cast<const NOX::Epetra::Vector&>(finalGroup->getX());
    const Epetra_Vector& finalSolution = finalSolutionNOX.getEpetraVector(); 

    double finalParamValue = finalGroup->getParam(cont_param);

    model->printSolution(finalSolution, finalParamValue);
    Comm->Barrier(); // make sure the root process has written the files
    LOCA::destroyGlobalData(globalData);

    // print timing details of routines/sections timed by HYMLS_PROF()
    if (Comm->MyPID()==0)
      {
      HYMLS::Tools::out() << "======================"<<std::endl;
      HYMLS::Tools::out() << " final parameter list "<<std::endl;
      HYMLS::Tools::out() << "======================"<<std::endl;
      HYMLS::Tools::out() << *paramList<<std::endl;
      HYMLS::Tools::out() << std::endl << std::endl;
      HYMLS::Tools::PrintTiming(HYMLS::Tools::out());
      HYMLS::Tools::PrintMemUsage(HYMLS::Tools::out());
      }
    } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr,stat);

#ifdef HAVE_MPI
  MPI_Finalize();
#endif


    
//end main
  return ierr;

  }


