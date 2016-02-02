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

#include "Teuchos_FancyOStream.hpp"

#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziEpetraAdapter.hpp"
#include "AnasaziBlockKrylovSchurSolMgr.hpp"

#ifdef HAVE_PHIST
#include "evp/AnasaziPhistSolMgr.hpp"
#else
#include "evp/AnasaziJacobiDavidsonSolMgr.hpp"
#include "evp/AnasaziHymlsAdapter.hpp"
#endif

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

int testSolver(std::string &message, Teuchos::RCP<const Epetra_Comm> comm,
    Teuchos::RCP<Teuchos::ParameterList> params, Teuchos::RCP<Epetra_CrsMatrix> &K,
    Teuchos::RCP<Epetra_MultiVector> &b, Teuchos::RCP<Epetra_MultiVector> &x_ex,
    Teuchos::RCP<HYMLS::Solver> &solver, Teuchos::RCP<HYMLS::Preconditioner> &precond);

int testEigenSolver(std::string &message, Teuchos::RCP<const Epetra_Comm> comm,
    Teuchos::RCP<Teuchos::ParameterList> params, Teuchos::RCP<Epetra_CrsMatrix> &K,
    Teuchos::RCP<Epetra_CrsMatrix> &M, Teuchos::RCP<HYMLS::Solver> &solver,
    Teuchos::RCP<HYMLS::Preconditioner> &precond);

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

      ierr = runTest(comm, params);
      if (ierr != PASSED)
        {
        std::string msg=params->get("runTest output","no output available");
        HYMLS::Tools::Out("------------------------------------------------------------");
        HYMLS::Tools::out() << "Test "+Teuchos::toString(counter)+" ('"+test_file+"') failed.\n";
        HYMLS::Tools::out() << "at resolution: "<<nx<<"x"<<ny<<"x"<<nz<<"\n";
        HYMLS::Tools::out() << "Reason: ";
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
  int mask = 1;
  while (ierr && mask)
    {
    switch (ierr & mask)
      {
      case MAX_ITER_EXCEEDED:
        HYMLS::Tools::out() << "Linear solver required more iterations than expected." << std::endl;
        break;
      case RES_TOO_LARGE:
        HYMLS::Tools::out() << "Residual norm too large." << std::endl;
        break;
      case ERR_TOO_LARGE:
        HYMLS::Tools::out() << "Error norm too large." << std::endl;
        break;
      case CAUGHT_EXCEPTION:
        HYMLS::Tools::out() << "Caught an exception." << std::endl;
        break;
      case INTERNAL_TESTS_FAILED:
        HYMLS::Tools::out() << "Internal tests failed." << std::endl;
        break;
      default:
        break;
      }
    ierr &= ~mask;
    mask <<= 1;
    }
  }

void getLinearSystem(Teuchos::RCP<const Epetra_Comm> comm,
    Teuchos::RCP<Teuchos::ParameterList> params,
    Teuchos::RCP<Epetra_CrsMatrix> &K, Teuchos::RCP<Epetra_CrsMatrix> &M,
    Teuchos::RCP<Epetra_MultiVector> &b, Teuchos::RCP<Epetra_MultiVector> &x_ex)
  {
  Teuchos::ParameterList& driverList = params->sublist("Driver");

  bool read_problem = driverList.get("Read Linear System", false);
  int numRhs = driverList.get("Number of rhs", 1);

  Teuchos::RCP<Epetra_Map> map = HYMLS::MainUtils::create_map(*comm, params->sublist("Problem"));

  // exact solution
  x_ex = Teuchos::rcp(new Epetra_MultiVector(*map, numRhs));

  // right-hand side
  b = Teuchos::rcp(new Epetra_MultiVector(*map, numRhs));

  // mass matrix
  M = Teuchos::null;

  if (read_problem)
    {
    std::string datadir = driverList.get("Data Directory", "not specified");
    if (datadir == "not specified")
      {
      HYMLS::Tools::Error("'Data Directory' not specified although 'Read Linear System' is true",
              __FILE__, __LINE__);
      }
    std::string file_format = driverList.get("File Format", "MatrixMarket");

    K = HYMLS::MainUtils::read_matrix(datadir, file_format, map);
    b = HYMLS::MainUtils::read_vector("rhs", datadir, file_format, map);
    if (driverList.get("Exact Solution Available", false))
      {
      x_ex = HYMLS::MainUtils::read_vector("sol", datadir, file_format, map);
      }
    if (driverList.get("Mass Matrix Available", false))
      {
      M = HYMLS::MainUtils::read_matrix(datadir, file_format, map, "mass");
      }
    }
  else
    {
    HYMLS::Tools::Out("Create matrix");
    std::string galeriLabel = driverList.get("Galeri Label", "");
    Teuchos::ParameterList galeriList;
    if (driverList.isSublist("Galeri"))
      {
      galeriList = driverList.sublist("Galeri");
      }
    K = HYMLS::MainUtils::create_matrix(*map, params->sublist("Problem"),
        galeriLabel, galeriList);
    }
  }

int runTest(Teuchos::RCP<const Epetra_Comm> comm,
        Teuchos::RCP<Teuchos::ParameterList> params)
  {
  int ierr = PASSED;
  HYMLS::Tester::numFailedTests_ = 0; // check if internal tests fail on the way
  std::string message = "";
  int no_exception = true;

  Teuchos::ParameterList& driverList = params->sublist("Driver");

#ifndef TESTING
  // suppress all HYMLS output during the test
  Teuchos::RCP<std::ostream> no_output
        = Teuchos::rcp(new Teuchos::oblackholestream());
  HYMLS::Tools::InitializeIO_std(comm,no_output,no_output);
#endif

  try {
    Teuchos::RCP<Epetra_CrsMatrix> K;
    Teuchos::RCP<Epetra_CrsMatrix> M;
    Teuchos::RCP<Epetra_Vector> u_ex;
    Teuchos::RCP<Epetra_Vector> f;

    Teuchos::RCP<Epetra_MultiVector> b;
    Teuchos::RCP<Epetra_MultiVector> x_ex;

    // create a copy of the parameter list
    Teuchos::RCP<Teuchos::ParameterList> params_copy
          = Teuchos::rcp(new Teuchos::ParameterList(*params));

    // we pass in params_copy here because in case of Laplace-Neumann
    // the "Problem" list is adjusted for the solver, which we do not
    // want in the original list.
    getLinearSystem(comm, params_copy, K, M, b, x_ex);

    HYMLS::Tools::Out("Create Preconditioner");

    Teuchos::RCP<HYMLS::Preconditioner> precond = Teuchos::rcp(new HYMLS::Preconditioner(K,
          params_copy));

    HYMLS::Tools::Out("Initialize Preconditioner...");
    CHECK_ZERO(precond->Initialize());

    HYMLS::Tools::Out("Create Solver");
    Teuchos::RCP<HYMLS::Solver> solver = Teuchos::rcp
          (new HYMLS::Solver(K, precond, params_copy));

    message = "";

    ierr |= testSolver(message, comm, params, K, b, x_ex, solver, precond);

#ifdef HAVE_PHIST
    if (driverList.isSublist("Eigenvalues"))
      {
      ierr |= testEigenSolver(message, comm, params, K, M, solver, precond);
      }
#endif

    comm->Barrier();
    }
  catch (HYMLS::Exception &e)
    {
    message += e.what();
    no_exception = false;
    }
  catch (std::exception &e)
    {
    message += e.what();
    no_exception = false;
    }
  catch (...)
    {
    message += "unknown exception";
    no_exception = false;
    }
  if (!no_exception) ierr = ierr | CAUGHT_EXCEPTION;

  if (HYMLS::Tester::numFailedTests_ > 0) ierr = ierr | INTERNAL_TESTS_FAILED;

  int global_ierr;
  comm->MaxAll(&ierr, &global_ierr, 1);

  params->set("runTest output", message);

#ifndef TESTING
  // reset to HYMLS output
  HYMLS::Tools::InitializeIO(comm);
#endif
  return global_ierr;
  }

int testSolver(std::string &message, Teuchos::RCP<const Epetra_Comm> comm,
    Teuchos::RCP<Teuchos::ParameterList> params, Teuchos::RCP<Epetra_CrsMatrix> &K,
    Teuchos::RCP<Epetra_MultiVector> &b, Teuchos::RCP<Epetra_MultiVector> &x_ex,
    Teuchos::RCP<HYMLS::Solver> &solver, Teuchos::RCP<HYMLS::Preconditioner> &precond)
  {
  int ierr = PASSED;

  Teuchos::ParameterList& driverList = params->sublist("Driver");
  int numComputes = driverList.get("Number of factorizations", 1);
  int numSolves = driverList.get("Number of solves", 1);
  int numRhs = driverList.get("Number of rhs", 1);
  bool read_problem = driverList.get("Read Linear System", false);

  Teuchos::ParameterList& targetList = params->sublist("Targets");
  int target_num_iter = targetList.get("Number of Iterations",9999);
  double target_rel_res_norm2 = targetList.get("Relative Residual 2-Norm",1.0);
  double target_rel_err_norm2 = targetList.get("Relative Error 2-Norm",1.0);

  Teuchos::ParameterList& probl_params = params->sublist("Problem");
  int dim = probl_params.get("Dimension", 2);
  std::string eqn = probl_params.get("Equations", "Laplace");

  //~ driverList.unused(std::cerr);

  // Use the map from the vectors from getLinearSystem here
  Epetra_BlockMap const &map = x_ex->Map();

  // approximate solution
  Teuchos::RCP<Epetra_MultiVector> x = Teuchos::rcp(new Epetra_MultiVector(map, numRhs));

  bool doDeflation = false;
  Teuchos::RCP<const Epetra_MultiVector> Nul = Teuchos::null;
  if (params->sublist("Solver").get("Null Space", "None") != "None")
    {
    Nul = solver->getNullSpace();
    }

  if (params->sublist("Solver").get("Deflated Subspace Dimension", 0) > 0)
    {
    doDeflation = true;
    }

  for (int f=0;f<numComputes;f++)
    {
    CHECK_ZERO(precond->Compute());
    if (doDeflation)
      {
      CHECK_ZERO(solver->SetupDeflation());
      }
    else
      {
      CHECK_ZERO(solver->setNullSpace());
      }

    int xseed=-1;

    for (int s=0;s<numSolves;s++)
      {
      if (!read_problem)
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
          double pref = (*x)[k][dim] - (*x_ex)[k][dim];
          for (int i=dim; i<x->MyLength();i+=dof)
            {
            (*x)[k][i] -= pref;
            }
          }
        }

  //    HYMLS::Tools::Out("Compute residual.");

      // compute residual and error vectors

      Teuchos::RCP<Epetra_MultiVector> res = Teuchos::rcp(new
          Epetra_MultiVector(map, numRhs));
      Teuchos::RCP<Epetra_MultiVector> err = Teuchos::rcp(new
          Epetra_MultiVector(map, numRhs));

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
        maxErr = std::max(maxErr,errNorm[k]/rhsNorm[k]);
        }

      delete[] rhsNorm;
      delete[] resNorm;
      delete[] errNorm;

      if (maxRes > target_rel_res_norm2) ierr = ierr | RES_TOO_LARGE;
      if (maxErr > target_rel_err_norm2) ierr = ierr | ERR_TOO_LARGE;
      int num_iter = solver->getNumIter();
      HYMLS::Tools::out() << std::endl;
#ifdef DEBUGGING
      if (ierr != PASSED)
        {
        HYMLS::MatrixUtils::Dump(*K,"BadMatrix.txt");
        HYMLS::MatrixUtils::Dump(*x,"BadSolution.txt");
        HYMLS::MatrixUtils::Dump(*x_ex,"BadExactSolution.txt");
        HYMLS::MatrixUtils::Dump(*b,"BadRhs.txt");
        HYMLS::MatrixUtils::Dump(*res,"BadRes.txt");
        HYMLS::MatrixUtils::Dump(*err,"BadErr.txt");
        }
#endif
      if (num_iter > target_num_iter) ierr = ierr | MAX_ITER_EXCEEDED;

      message += "setup " + Teuchos::toString(f)
           + ", solve " + Teuchos::toString(s) + "\n"
           + "max error: " + Teuchos::toString(maxErr)
           + ", expected: " + Teuchos::toString(target_rel_res_norm2) + "\n"
           + "max res:   " + Teuchos::toString(maxRes)
           + ", expected: " + Teuchos::toString(target_rel_err_norm2) + "\n"
           + "num iter:  " + Teuchos::toString(num_iter)
           + ", expected: " + Teuchos::toString(target_num_iter) + "\n";

      if (xseed>0)
        {
        message += "rand seed: " + Teuchos::toString(xseed) + "\n";
        }

      message += "------------------------------------------\n";
      }
    }

  return ierr;
  }

int testEigenSolver(std::string &message, Teuchos::RCP<const Epetra_Comm> comm,
    Teuchos::RCP<Teuchos::ParameterList> params, Teuchos::RCP<Epetra_CrsMatrix> &K,
    Teuchos::RCP<Epetra_CrsMatrix> &M, Teuchos::RCP<HYMLS::Solver> &solver,
    Teuchos::RCP<HYMLS::Preconditioner> &precond)
  {
  int ierr = PASSED;

  // Use the map from the vectors from getLinearSystem here
  Epetra_BlockMap const &map = K->RowMap();

  // approximate solution
  Teuchos::RCP<Epetra_MultiVector> x = Teuchos::rcp(new Epetra_MultiVector(map, 1));

  Teuchos::ParameterList& driverList = params->sublist("Driver");

  Teuchos::ParameterList& targetList = params->sublist("Targets");
  double target_err = targetList.get("Error Eigenvalues", 1.0);

  Teuchos::ParameterList& probl_params = params->sublist("Problem");
  int dim = probl_params.get("Dimension", 2);
  std::string eqn = probl_params.get("Equations", "Laplace");
  int dof = 1;

  Teuchos::RCP<Epetra_MultiVector> v0 = Teuchos::rcp(new Epetra_Vector(x->Map()));
  HYMLS::MatrixUtils::Random(*v0);

  for (int i = 0; i < v0->MyLength(); i++)
    {
    if (v0->Map().GID(i) % dof == dim)
      {
      (*v0)[0][i] = 0.0;
      }
    }

  precond->ApplyInverse(*v0, *x);

  // Make x B-orthogonal
  if (M != Teuchos::null)
    M->Multiply(false, *x, *v0);
  else
    *v0 = *x;

  double result;
  x->Dot(*v0, &result);
  x->Scale(1.0/sqrt(result));

  // Create the eigenproblem
  HYMLS::Tools::Out("Create Eigenproblem");
  Teuchos::ParameterList eigList = driverList.sublist("Eigenvalues");

  typedef double ST;
  typedef Epetra_MultiVector MV;
  typedef Epetra_Operator OP;
  typedef HYMLS::Solver PREC;

  Teuchos::RCP<Anasazi::BasicEigenproblem<ST, MV, OP> > eigProblem;
  eigProblem = Teuchos::rcp(new Anasazi::BasicEigenproblem<ST,MV,OP>(K, M, x));
  eigProblem->setHermitian(false);
  eigProblem->setNEV(eigList.get("How Many", 10));

  if (!eigProblem->setProblem())
    {
    HYMLS::Tools::Error("eigProblem->setPoroblem returned 'false'",__FILE__,__LINE__);
    }

#ifdef HAVE_PHIST
  Anasazi::PhistSolMgr<ST,MV,OP,PREC> jada(eigProblem, solver, eigList);
#else
  Anasazi::JacobiDavidsonSolMgr<ST,MV,OP,PREC> jada(eigProblem, solver, eigList);
#endif

  // Solve the problem to the specified tolerances or length
  Anasazi::ReturnType returnCode;
  returnCode = jada.solve();
  if (returnCode != Anasazi::Converged)
    {
    HYMLS::Tools::Warning("Anasazi::EigensolverMgr::solve() returned unconverged.",
        __FILE__,__LINE__);
    }

  const Anasazi::Eigensolution<ST,MV>& eigSol = eigProblem->getSolution();

  const std::vector<Anasazi::Value<ST> >& evals = eigSol.Evals;
  int numEigs = evals.size();

  // We can compute the exact eigenvaleus for Laplace
  if (eqn == "Laplace")
    {
    Teuchos::ParameterList& problemList = params->sublist("Problem");

    int nx = problemList.get("nx", 32);
    int ny = problemList.get("ny", nx);
    int nz = problemList.get("nz", dim > 2 ? nx: 1);

    double hx = 1.0 / (double)(nx+1);
    double hy = 1.0 / (double)(ny+1);
    double hz = 1.0 / (double)(nz+1);

    // Generate all the exact eigenvalues of -K
    std::vector<double> ev_list;
    for (int i = 1; i < nx + 1; i++)
    for (int j = 1; j < ny + 1; j++)
    for (int k = 1; k < nz + 1; k++)
      {
      double ev = 4.0 * pow(sin(M_PI*i*hx / 2.0), 2)
                + 4.0 * pow(sin(M_PI*j*hy / 2.0), 2);
      if (dim > 2)
        {
        ev += 4.0 * pow(sin(M_PI*k*hz / 2.0), 2);
        }
      ev_list.push_back(ev);
      }

    // Sort them so the smallest ones are first
    std::sort(ev_list.begin(), ev_list.end());

    // Now compare with the computed eigenvalues. We do numEigs-1, because
    // depending on the random starting vector it may sometimes happen
    // that we find one larger eigenvalue.
    for (int i = 0; i < numEigs-1; i++)
      {
      if (std::abs(evals[i].imagpart) > target_err)
        ierr = ierr | ERR_TOO_LARGE;

      if (std::abs(evals[i].realpart + ev_list[i]) > target_err)
        ierr = ierr | ERR_TOO_LARGE;

      message += "found " + Teuchos::toString(evals[i].realpart)
           + ", expected: " + Teuchos::toString(-ev_list[i]) + "\n";
      }
    }

  message += "------------------------------------------\n";

  return ierr;
  }
