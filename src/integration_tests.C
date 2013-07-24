#include <cstdlib>

#include <iostream>

#include <mpi.h>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_MultiVector.h"
#include "Epetra_CrsMatrix.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_ParameterListAcceptorHelpers.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_StrUtils.hpp"

#include "main_utils.H"

/*
#include "EpetraExt_HDF5.h"
#include "EpetraExt_Exception.h"
*/
#include "HYMLS_HyperCube.H"
#include "HYMLS_Tools.H"
#include "HYMLS_Tester.H"
#include "HYMLS_Preconditioner.H"
#include "HYMLS_Solver.H"
#include "HYMLS_MatrixUtils.H"

#include "Teuchos_FancyOStream.hpp"

typedef enum
{
PASSED=0,
MAX_ITER_EXCEEDED=1,
RES_TOO_LARGE =2,
ERR_TOO_LARGE =8,
CAUGHT_EXCEPTION=16,
INTERNAL_TESTS_FAILED=32
} ReturnCode;


int runTest(Teuchos::RCP<const Epetra_Comm> comm,
                   Teuchos::RCP<Teuchos::ParameterList> params);
void printError(int ierr);

int main(int argc, char* argv[])
  {
  MPI_Init(&argc, &argv);

  std::srand(std::time(NULL));

  int counter = 0;
  int failed = 0;
  int skipped= 0;
  bool status=true;
  try {
//  Teuchos::RCP<Epetra_MpiComm> comm=Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  HYMLS::HyperCube Topology;
  Teuchos::RCP<const Epetra_MpiComm> comm = Teuchos::rcp
        (&Topology.Comm(), false);

  // construct file streams, otherwise the output won't work correctly
  HYMLS::Tools::InitializeIO(comm);
  
  HYMLS::Tools::out() << "this is HYMLS, rev "<<HYMLS::Tools::Revision()<<std::endl;
  
  HYMLS::Tools::StartTiming("Integration Tests");

  Teuchos::RCP<Teuchos::ParameterList> tests = 
        Teuchos::getParametersFromXmlFile("all_tests.xml");
  bool skip3D = tests->get("skip 3D tests",false);
  tests->remove("skip 3D tests");
  int ierr;
  for (Teuchos::ParameterList::ConstIterator i=tests->begin(); i!=tests->end(); i++)
    {
    ierr=PASSED;
    counter++;
    std::string test_name = i->first;
    std::string test_file="undefined";
    test_file = tests->get(test_name,test_file);

    if (skip3D && (test_name.length()>2))
      {
      if (Teuchos::StrUtils::subString(test_name,0,2)=="3D")
        {
        HYMLS::Tools::out() << "SKIPPED TEST "+Teuchos::toString(counter)+": "+test_file+"\n";
        skipped++;
        continue;
        }
      }

    Teuchos::RCP<Teuchos::ParameterList> params = 
        Teuchos::getParametersFromXmlFile("default.xml");
        
    Teuchos::updateParametersFromXmlFile(test_file,params.ptr());

    std::string description = params->name();
    if (params->isParameter("Description"))
      {
      description = params->get("Description","no description");
      }
        
    Teuchos::ParameterList& driverList=params->sublist("Driver");
    Teuchos::ParameterList& problemList=params->sublist("Problem");
    
    int num_refines = driverList.get("Number of refinements",0);
    bool read_linsys=driverList.get("Read Linear System",false);
    int dim= problemList.get("Dimension",2);
    int nx = problemList.get("nx",32);
    int ny = problemList.get("ny",nx);
    int nz = problemList.get("nz",dim>2? nx: 1);

    HYMLS::Tools::Out(test_name+": "+description+" ["+test_file+"]");

    for (int ref=0;ref<=num_refines;ref++)
      {
      std::string nxs = Teuchos::toString(nx);
      std::string nys = Teuchos::toString(ny);
      std::string nzs = Teuchos::toString(nz);
      std::string var = nxs+"x"+nys;
      if (dim>2) var = var+"x"+nzs;
      HYMLS::Tools::out() << "\t\t- grid size "<<var<<std::endl;
      ierr = runTest(comm,params);
      if (ierr!=PASSED)
        {
        std::string msg=params->get("runTest output","no output available");
        HYMLS::Tools::Out("------------------------------------------------------------");
        HYMLS::Tools::out() << "Test "+Teuchos::toString(counter)+" ('"+test_file+"') failed.\n";
        HYMLS::Tools::out() << "at resolution: "<<nx<<"x"<<ny<<"x"<<nz<<"\n";
        printError(ierr);
        HYMLS::Tools::out() << msg << std::endl;
#ifdef TESTING
        HYMLS::Tools::out() << *params << std::endl;
#endif
        HYMLS::Tools::Out("------------------------------------------------------------");
        failed++;
        break; // stop grid refinement
        }
      nx*=2;
      ny*=2;
      if (dim>2) nz*=2;
      params->sublist("Problem").set("nx",nx);
      params->sublist("Problem").set("ny",ny);
      params->sublist("Problem").set("nz",nz);
      
      if (read_linsys)
        {
        nxs = Teuchos::toString(nx);
        nys = Teuchos::toString(ny);
        nzs = Teuchos::toString(nz);
        std::string new_var = nxs+"x"+nys;
        if (dim>2) new_var = new_var+"x"+nzs;
        std::string data_dir=params->sublist("Driver").get("Data Directory","undefined");
        std::string new_data_dir=Teuchos::StrUtils::varSubstitute(data_dir,var,new_var);
        params->sublist("Driver").set("Data Directory",new_data_dir);
        }

      }// refinement
    if (ierr==PASSED)
      {
      HYMLS::Tools::out() << "PASSED TEST "+Teuchos::toString(counter)+": "+test_file+"\n";
      }
    }// test files
  if (skipped>0)
    {
    HYMLS::Tools::out() << skipped<<" TESTS SKIPPED"<<std::endl;
    }
  if (failed==0)
    {
    HYMLS::Tools::out() << "ALL TESTS PASSED"<<std::endl;
    }
  else
    {
    HYMLS::Tools::out() << "WARNING: "<<failed<<" TESTS OUT OF "<<counter<<" FAILED"<<std::endl;
    }
  } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr, status);
  if (!status) HYMLS::Tools::Warning("Caught an exception",__FILE__,__LINE__);
  
  HYMLS::Tools::StopTiming("Integration Tests",true);

  MPI_Finalize();
  return 0;
  }
  

void printError(int ierr)
  {

  if (ierr&MAX_ITER_EXCEEDED)
    {
    HYMLS::Tools::out() << "linear solver required more iterations than expected."<<std::endl;
    }
  if (ierr&RES_TOO_LARGE)
    {
    HYMLS::Tools::out() << "residual norm too large."<<std::endl;
    }
  if (ierr&ERR_TOO_LARGE)
    {
    HYMLS::Tools::out() << "error norm too large."<<std::endl;
    }
  if (ierr&CAUGHT_EXCEPTION)
    {
    HYMLS::Tools::out() << "caught an exception."<<std::endl;
    }
  }

int runTest(Teuchos::RCP<const Epetra_Comm> comm,
        Teuchos::RCP<Teuchos::ParameterList> params)
  {
  int ierr=PASSED;
  HYMLS::Tester::numFailedTests_=0; // check if internal tests fail on the way
  std::string message="";
  int no_exception=true;

#ifndef TESTING  
  // suppress all HYMLS output during the test
  Teuchos::RCP<std::ostream> no_output
        = Teuchos::rcp(new Teuchos::oblackholestream());
  HYMLS::Tools::InitializeIO_std(comm,no_output,no_output);
#endif

  try {
  Teuchos::RCP<Epetra_Map> map;
  Teuchos::RCP<Epetra_CrsMatrix> K;
  Teuchos::RCP<Epetra_Vector> u_ex;
  Teuchos::RCP<Epetra_Vector> f;

    Teuchos::ParameterList& driverList = params->sublist("Driver");

    int numComputes=driverList.get("Number of factorizations",1);
    int numSolves=driverList.get("Number of solves",1);
    int numRhs   =driverList.get("Number of rhs",1);
    
    bool read_problem=driverList.get("Read Linear System",false);
    string datadir,file_format;
    bool have_exact_sol=false;

    if (read_problem)
      {
      datadir = driverList.get("Data Directory","not specified");
      if (datadir=="not specified")
        {
        HYMLS::Tools::Error("'Data Directory' not specified although 'Read Linear System' is true",
                __FILE__,__LINE__);
        }                
      file_format = driverList.get("File Format","MatrixMarket");
      have_exact_sol = driverList.get("Exact Solution Available",false);
      }

    driverList.unused(std::cerr);
    
    Teuchos::ParameterList& targetList = params->sublist("Targets");
    int target_num_iter = targetList.get("Number of Iterations",9999);
    double target_rel_res_norm2 = targetList.get("Relative Residual 2-Norm",1.0);
    double target_rel_err_norm2 = targetList.get("Relative Error 2-Norm",1.0);
        
    Teuchos::ParameterList& probl_params = params->sublist("Problem");
            
    int dim=probl_params.get("Dimension",2);
    std::string eqn=probl_params.get("Equations","Laplace");

  // create a copy of the parameter list
  Teuchos::RCP<Teuchos::ParameterList> params_copy
        = Teuchos::rcp(new Teuchos::ParameterList(*params));

    params_copy->remove("Targets");
    params_copy->remove("Driver");
    
    map = HYMLS::MainUtils::create_map(*comm,probl_params);
  
  // create a random exact solution
  Teuchos::RCP<Epetra_MultiVector> x_ex = Teuchos::rcp(new Epetra_MultiVector(*map,numRhs));

  // construct right-hand side
  Teuchos::RCP<Epetra_MultiVector> b = Teuchos::rcp(new Epetra_MultiVector(*map,numRhs));

  // approximate solution
  Teuchos::RCP<Epetra_MultiVector> x = Teuchos::rcp(new Epetra_MultiVector(*map,numRhs));
  

  if (read_problem)
    {
    K=HYMLS::MainUtils::read_matrix(datadir,file_format,map);
    b=HYMLS::MainUtils::read_vector("rhs",datadir,file_format,map);
    if (have_exact_sol)
      {
      x_ex=HYMLS::MainUtils::read_vector("sol",datadir,file_format,map);
      }
    }
  else
    {
    HYMLS::Tools::Out("Create matrix");
    std::string galeriLabel = driverList.get("Galeri Label","");
    Teuchos::ParameterList galeriList;
    if (driverList.isSublist("Galeri"))
      { 
      galeriList = driverList.sublist("Galeri");
      }
    // we pass in params_copy here because in case of Laplace-Neumann
    // the "Problem" list is adjusted for the solver, which we do not
    // want in the original list.
    K=HYMLS::MainUtils::create_matrix(*map,params_copy->sublist("Problem"),
        galeriLabel, galeriList);
    
    }

  HYMLS::Tools::Out("Create Preconditioner");

  Teuchos::RCP<HYMLS::Preconditioner> precond = Teuchos::rcp(new HYMLS::Preconditioner(K, 
        params_copy));

  HYMLS::Tools::Out("Initialize Preconditioner...");
  CHECK_ZERO(precond->Initialize());

  HYMLS::Tools::Out("Create Solver");
  Teuchos::RCP<HYMLS::Solver> solver = Teuchos::rcp
        (new HYMLS::Solver(K, precond, params_copy, numRhs));
  
  bool doDeflation = false;
  Teuchos::RCP<Epetra_MultiVector> Nul=Teuchos::null;
  if (params_copy->sublist("Solver").get("Null Space","None")!="None")
    {
    doDeflation=true;
    Nul=solver->getNullSpace();
    }
  
  if (params_copy->sublist("Solver").get("Deflated Subspace Dimension",0)>0)
    {
    doDeflation=true;
    }

message = "";

for (int f=0;f<numComputes;f++)
  {

  CHECK_ZERO(precond->Compute());
  if (doDeflation)
    {
    CHECK_ZERO(solver->SetupDeflation());
    }
    
 // std::cout << *solver << std::endl;
 int xseed=-1;
  
  for (int s=0;s<numSolves;s++)
    {
    if (read_problem==false)
      {
      xseed = x_ex->Seed();
      CHECK_ZERO(HYMLS::MatrixUtils::Random(*x_ex));
       if (Nul!=Teuchos::null)
        {
        double alpha;
        double vnrm2;
        CHECK_ZERO(x_ex->Dot(*Nul,&alpha));
        CHECK_ZERO(Nul->Norm2(&vnrm2));
        alpha/=(vnrm2*vnrm2);
        CHECK_ZERO(x_ex->Update(-alpha,*Nul,1.0));
        }

      CHECK_ZERO(K->Multiply(false,*x_ex,*b));
      }

    HYMLS::Tools::Out("Solve ("+Teuchos::toString(s+1)+")");
    CHECK_ZERO(solver->ApplyInverse(*b,*x));

    // subtract constant from pressure if solving Stokes-C
    if (eqn=="Stokes-C")
      {
      int dof=dim+1;
      for (int k=0;k<numRhs;k++)
        {
        double pref=(*x)[k][dim];
        if (have_exact_sol)
          {
          pref -= (*x_ex)[k][dim];
          }
        for (int i=dim; i<x->MyLength();i+=dof)
          {
          (*x)[k][i]-=pref;
          }
        }
      }

//    HYMLS::Tools::Out("Compute residual.");
  
    // compute residual and error vectors

    Teuchos::RCP<Epetra_MultiVector> res = Teuchos::rcp(new 
        Epetra_MultiVector(*map,numRhs));
    Teuchos::RCP<Epetra_MultiVector> err = Teuchos::rcp(new 
        Epetra_MultiVector(*map,numRhs));

    CHECK_ZERO(K->Multiply(false,*x,*res));
    CHECK_ZERO(res->Update(1.0,*b,-1));
  
    CHECK_ZERO(err->Update(1.0,*x,-1.0,*x_ex,0.0));
  
    double *errNorm,*resNorm,*rhsNorm;
    errNorm = new double[numRhs];
    resNorm = new double[numRhs];
    rhsNorm = new double[numRhs];
  
    err->Norm2(errNorm);
    res->Norm2(resNorm);
    b->Norm2(rhsNorm);
  
    double maxRes = 0.0;
    double maxErr = 0.0;
  
    HYMLS::Tools::out()<< "||Ax-b||_2/||b||_2 \t (||x||_2-||x_ex||_2)/||b||_2\n";
    for (int k=0;k<numRhs;k++)
      {
      HYMLS::Tools::out()<<std::setw(8)<<std::setprecision(8)<<std::scientific;
      HYMLS::Tools::out()<<Teuchos::toString(resNorm[k]/rhsNorm[k])<<" \t "<<Teuchos::toString(errNorm[k]/rhsNorm[k])<<" \n";
      maxRes = std::max(maxRes,resNorm[k]/rhsNorm[k]);
      maxErr = std::max(maxRes,errNorm[k]/rhsNorm[k]);
      }
      
    delete rhsNorm;
    delete resNorm;
    delete errNorm;

    if (maxRes>target_rel_res_norm2) ierr = ierr | RES_TOO_LARGE;
    if (maxErr>target_rel_err_norm2) ierr = ierr | ERR_TOO_LARGE;
    int num_iter = solver->getNumIter();
    HYMLS::Tools::out() << std::endl;
#ifdef DEBUGGING
if (ierr!=PASSED)
  {
  HYMLS::MatrixUtils::Dump(*K,"BadMatrix.txt");
  HYMLS::MatrixUtils::Dump(*x,"BadSolution.txt");
  HYMLS::MatrixUtils::Dump(*x_ex,"BadExactSolution.txt");
  HYMLS::MatrixUtils::Dump(*b,"BadRhs.txt");
  HYMLS::MatrixUtils::Dump(*res,"BadRes.txt");
  HYMLS::MatrixUtils::Dump(*err,"BadErr.txt");
  }
#endif    
    if (num_iter>target_num_iter) ierr = ierr | MAX_ITER_EXCEEDED;

    message = message 
         +"setup "+Teuchos::toString(f)+", solve "+Teuchos::toString(s)+"\n"
         +"max error: "+Teuchos::toString(maxErr)+"\n"
         +"max res:   "+Teuchos::toString(maxRes)+"\n"
         +"num iter:  "+Teuchos::toString(num_iter)+"\n";
    if (xseed>0)
      {
      message+="rand seed: "+Teuchos::toString(xseed)+"\n";
      }
    message+="------------------------------------------\n";
    }
  }

  comm->Barrier();
  } catch(HYMLS::Exception hym)
    {
    no_exception = false;
    message = message + hym.what();
    }
    catch (std::exception e)
      {
      message = message + e.what();
      no_exception=false;
      }
    catch (...)
      {
      message = message + "unknown exception";
      no_exception=false;
      }
  if (no_exception==false) ierr = ierr | CAUGHT_EXCEPTION;  

  if (HYMLS::Tester::numFailedTests_>0) ierr = ierr | INTERNAL_TESTS_FAILED;  

  params->set("runTest output",message);

#ifndef TESTING
  // reset to HYMLS output
  HYMLS::Tools::InitializeIO(comm);
#endif
  return ierr;
  }


