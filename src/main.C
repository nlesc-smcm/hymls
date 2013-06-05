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
#ifdef DEBUGGING
#include <signal.h>
#endif

#include "main_utils.H"

/*
#include "EpetraExt_HDF5.h"
#include "EpetraExt_Exception.h"
*/
#include "HYMLS_HyperCube.H"
#include "HYMLS_Tools.H"
#include "HYMLS_Preconditioner.H"
#include "HYMLS_Solver.H"
#include "HYMLS_MatrixUtils.H"



int main(int argc, char* argv[])
  {
  MPI_Init(&argc, &argv);

#ifdef DEBUGGING
  signal(SIGINT,HYMLS::Tools::SignalHandler);
  signal(SIGSEGV,HYMLS::Tools::SignalHandler);
#endif

// random number initialization (if TESTING is defined
// we provide our own seed when creating vectors below)
std::srand ( std::time(NULL) );

bool status=true;

//  Teuchos::RCP<Epetra_MpiComm> comm=Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  HYMLS::HyperCube Topology;
  Teuchos::RCP<const Epetra_MpiComm> comm = Teuchos::rcp
        (&Topology.Comm(), false);
    
  // construct file streams, otherwise the output won't work correctly
  HYMLS::Tools::InitializeIO(comm);
  
  HYMLS::Tools::out() << "this is HYMLS, rev "<<HYMLS::Tools::Revision()<<std::endl;
  

  try {

  START_TIMER("main","entire run");

  std::string param_file;
  Teuchos::Array<std::string> extra_files;

  if (argc<2)
    {
    HYMLS::Tools::Out("USAGE: main <parameter_filename>");
    MPI_Finalize();
    return 0;
    }
  else
    {
    param_file = argv[1];
    HYMLS::Tools::Out("Reading parameters from "+param_file);
    extra_files.resize(argc-2);
    for (int i=0;i<argc-2;i++)
      {
      extra_files[i]=argv[2+i];
    HYMLS::Tools::Out("... overloading parameters from "+extra_files[i]);
      }
    }


  Teuchos::RCP<Epetra_Map> map;
  Teuchos::RCP<Epetra_CrsMatrix> K;
  Teuchos::RCP<Epetra_Vector> u_ex;
  Teuchos::RCP<Epetra_Vector> f;

  Teuchos::RCP<Teuchos::ParameterList> params = 
        Teuchos::getParametersFromXmlFile(param_file);
        
  for (int i=0;i<extra_files.size();i++)
    {
    Teuchos::updateParametersFromXmlFile(extra_files[i],params.ptr());
    }
        
    Teuchos::ParameterList& driverList = params->sublist("Driver");

    int seed = driverList.get("Random Seed",-1);
    if (seed!=-1) std::srand(seed);
    bool print_final_list = driverList.get("Store Final Parameter List",false);        
    bool store_solution = driverList.get("Store Solution",true);
    bool store_matrix = driverList.get("Store Matrix",false);
    int numComputes=driverList.get("Number of factorizations",1);
    int numSolves=driverList.get("Number of solves",1);
    int numRhs   =driverList.get("Number of rhs",1);
    double perturbation = driverList.get("Diagonal Perturbation",0.0);
    
    std::string galeriLabel=driverList.get("Galeri Label","");
    Teuchos::ParameterList galeriList;
    if (driverList.isSublist("Galeri")) galeriList = driverList.sublist("Galeri");
 
    // copy here rather than reference because the driver list will be removed 
    // alltogether...   
    bool read_problem=driverList.get("Read Linear System",false);
    string datadir,file_format;
    bool have_rhs=true;
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
      have_rhs = driverList.get("RHS Available",true);
      have_exact_sol = driverList.get("Exact Solution Available",false);
      }

    driverList.unused(std::cerr);
    params->remove("Driver");

        
    Teuchos::ParameterList& probl_params = params->sublist("Problem");
            
    int dim=probl_params.get("Dimension",2);
    int nx=probl_params.get("nx",32);
    int ny=probl_params.get("ny",nx);
    int nz=probl_params.get("nz",dim>2?nx:1);
    
    std::string eqn=probl_params.get("Equations","Laplace");

    map = HYMLS::MainUtils::create_map(*comm,probl_params); 
#ifdef STORE_MATRICES
HYMLS::MatrixUtils::Dump(*map,"MainMatrixMap.txt");
#endif
  if (read_problem)
    {
    K=HYMLS::MainUtils::read_matrix(datadir,file_format,map);
    }
  else
    {
    HYMLS::Tools::Out("Create matrix");

    K=HYMLS::MainUtils::create_matrix(*map,probl_params,
        galeriLabel, galeriList);
    }
  // create a random exact solution
  Teuchos::RCP<Epetra_MultiVector> x_ex = Teuchos::rcp(new Epetra_MultiVector(*map,numRhs));

  // construct right-hand side
  Teuchos::RCP<Epetra_MultiVector> b = Teuchos::rcp(new Epetra_MultiVector(*map,numRhs));

  // approximate solution
  Teuchos::RCP<Epetra_MultiVector> x = Teuchos::rcp(new Epetra_MultiVector(*map,numRhs));
  
  if (read_problem)
    {
    if (have_exact_sol)
      {
      x_ex=HYMLS::MainUtils::read_vector("sol",datadir,file_format,map);
      }
    if (have_rhs)
      {
      b=HYMLS::MainUtils::read_vector("rhs",datadir,file_format,map);
      }
    else
      {
      b=Teuchos::rcp(new Epetra_Vector(*map));
      }
    }

  Teuchos::ParameterList& solver_params = params->sublist("Solver");
  bool do_deflation = (solver_params.get("Deflated Subspace Dimension",0)>0);
  if (solver_params.get("Null Space","None")!="None")
    {
    do_deflation=true;
    }
  Teuchos::RCP<Epetra_CrsMatrix> M = Teuchos::null;
  if (do_deflation) // need a mass matrix
    {
    HYMLS::Tools::Out("Create dummy mass matrix");
    M=Teuchos::rcp(new Epetra_CrsMatrix(Copy,*map,1,true));
    int gid;
    double val1=1.0/(nx*ny*nz);
    double val0=0.0;
    if (eqn=="Stokes-C")
      {
      int dof=dim+1;
      for (int i=0;i<M->NumMyRows();i+=dof)
        {
        for (int j=i;j<i+dof-1;j++)
          {
          gid = map->GID(j);
          CHECK_ZERO(M->InsertGlobalValues(gid,1,&val1,&gid));
          }
        gid = map->GID(i+dof-1);
        CHECK_ZERO(M->InsertGlobalValues(gid,1,&val0,&gid));
        }
      }
    else
      {
      for (int i=0;i<M->NumMyRows();i++)
        {
        gid = map->GID(i);
        CHECK_ZERO(M->InsertGlobalValues(gid,1,&val1,&gid));
        }
      }
    CHECK_ZERO(M->FillComplete());
    }

  if (eqn=="Stokes-C")
    {
    // scale equations by -1 to make operator negative indefinite
    // (for testing the deflation capabilities)
    K->Scale(-1.0);
    b->Scale(-1.0);
    // put a zero in the mass matrix for singletons
    if (M!=Teuchos::null)
      {
     int lenA, lenM;
      int *indA, *indM;
      double *valA, *valM;
      for (int i=0;i<K->NumMyRows();i++)
        {
        CHECK_ZERO(K->ExtractMyRowView(i,lenA,valA,indA));
        CHECK_ZERO(M->ExtractMyRowView(i,lenM,valM,indM));
        if (lenA==1)
          {
          if (K->GCID(indA[0])==K->GRID(i))
            {
            for (int j=0;j<lenM;j++)
              {
              valM[j]=0.0;
              }
            }
          }
        }
      }
    }
  HYMLS::Tools::Out("Create Preconditioner");

  Teuchos::RCP<HYMLS::Preconditioner> precond = Teuchos::rcp(new HYMLS::Preconditioner(K, params));

  HYMLS::Tools::Out("Initialize Preconditioner...");
  HYMLS::Tools::StartTiming("main: Initialize Preconditioner");
  CHECK_ZERO(precond->Initialize());
  HYMLS::Tools::StopTiming("main: Initialize Preconditioner",true);

  HYMLS::Tools::Out("Create Solver");
  Teuchos::RCP<HYMLS::Solver> solver = Teuchos::rcp(new HYMLS::Solver(K, precond, params,numRhs));

  // get the null space (if any), as specified in the xml-file
  Teuchos::RCP<Epetra_MultiVector> Nul = solver->getNullSpace();
  
for (int f=0;f<numComputes;f++)
  {
  if (perturbation!=0)
    {
    // change the matrix values just to see if that works
    Epetra_Vector diag(*map);
    CHECK_ZERO(K->ExtractDiagonalCopy(diag));
    Epetra_Vector diag_pert(*map);
    HYMLS::MatrixUtils::Random(diag_pert);
    for (int i=0;i<diag_pert.MyLength();i++)
      {
//      diag[i]=diag_pert[i]*perturbation;
      diag[i]=diag[i] + diag_pert[i]*perturbation;
      }
    CHECK_ZERO(K->ReplaceDiagonalValues(diag));
    }
  HYMLS::Tools::Out("Compute Preconditioner ("+Teuchos::toString(f+1)+")");

  if (precond!=Teuchos::null) 
    {
    HYMLS::Tools::StartTiming("main: Compute Preconditioner");
    CHECK_ZERO(precond->Compute());
    HYMLS::Tools::StopTiming("main: Compute Preconditioner",true);
    }

  if (do_deflation)
    {
    solver->SetMassMatrix(M);
    CHECK_ZERO(solver->SetupDeflation());
    }

 // std::cout << *solver << std::endl;
  
  for (int s=0;s<numSolves;s++)
    {
    if (read_problem==false || have_rhs==false)
      {
      if (seed!=-1)
        {
        CHECK_ZERO(HYMLS::MatrixUtils::Random(*x_ex, seed));
        seed++;
        }
      else
        {
        CHECK_ZERO(HYMLS::MatrixUtils::Random(*x_ex));
        }
      if (eqn=="Darcy" || eqn=="Stokes-C")
        {
        // make sure the div equation is Div U = 0 and the RHS is consistent
        CHECK_ZERO(HYMLS::MainUtils::MakeSystemConsistent(*K,*x_ex,*b,driverList));
        }
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
  HYMLS::Tools::StartTiming("main: Solve");
    CHECK_ZERO(solver->ApplyInverse(*b,*x));
  HYMLS::Tools::StopTiming("main: Solve",true);

    // subtract constant from pressure if solving Stokes-C
    if (eqn=="Stokes-C"&&false)
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
DEBVAR(*x);
DEBVAR(*b);
    HYMLS::Tools::Out("Compute residual.");
  
    // compute residual and error vectors

    Teuchos::RCP<Epetra_MultiVector> res = Teuchos::rcp(new 
        Epetra_MultiVector(*map,numRhs));
    Teuchos::RCP<Epetra_MultiVector> err = Teuchos::rcp(new 
        Epetra_MultiVector(*map,numRhs));
    CHECK_ZERO(K->Multiply(false,*x,*res));
    CHECK_ZERO(res->Update(1.0,*b,-1));
  
    CHECK_ZERO(err->Update(1.0,*x,-1.0,*x_ex,0.0));
  
    double *errNorm,*resNorm,*rhsNorm;
    resNorm=new double[numRhs];
    errNorm=new double[numRhs];
    rhsNorm=new double[numRhs];
  
    err->Norm2(errNorm);
    res->Norm2(resNorm);
    b->Norm2(rhsNorm);
  
    HYMLS::Tools::out()<< "Residual Norm ||Ax-b||/||b||: ";
    for (int k=0;k<numRhs;k++)
      {
      HYMLS::Tools::out()<<std::setw(8)<<std::setprecision(8)<<std::scientific<<Teuchos::toString(resNorm[k]/rhsNorm[k])<<"  ";
      }
    HYMLS::Tools::out()<<std::endl;
    HYMLS::Tools::out()<<"Error Norm ||x-x_ex||/||b||: ";
    for (int k=0;k<numRhs;k++)
      {
      HYMLS::Tools::out()<<std::setw(8)<<std::setprecision(8)<<std::scientific<<Teuchos::toString(errNorm[k]/rhsNorm[k])<<"  ";
      }
    delete [] resNorm;
    delete [] rhsNorm;
    delete [] errNorm;
    HYMLS::Tools::out() << std::endl;
    }
  }

  if (store_matrix)
    {
    HYMLS::Tools::Out("store matrix...");
    HYMLS::MatrixUtils::Dump(*K, "Matrix.txt",false);
    }
    
  if (store_solution)
    {
    HYMLS::Tools::Out("store solution...");
    HYMLS::MatrixUtils::Dump(*x_ex, "ExactSolution.txt",false);
    HYMLS::MatrixUtils::Dump(*x, "Solution.txt",false);
    HYMLS::MatrixUtils::Dump(*b, "RHS.txt",false);
    }
    
  if (print_final_list)
    {
    if (comm->MyPID()==0)
      {
      Teuchos::RCP<const Teuchos::ParameterList> finalList
        = solver->getParameterList();
      std::string filename1 = param_file+".final";        
      HYMLS::Tools::out() << "final parameter list is written to '" << filename1<<"'"<<std::endl;
      writeParameterListToXmlFile(*finalList,filename1);

      HYMLS::Tools::out() << "parameter documentation is written to file param_doc.txt" << std::endl;
      std::ofstream ofs("paramDoc.txt");
      ofs << "valid parameters for HYMLS::Solver "<<std::endl;
      printValidParameters(*solver,ofs);
      ofs << "valid parameters for HYMLS::Preconditioner "<<std::endl;
      printValidParameters(*precond,ofs);
      }
    }
  
  
    } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr, status);
  if (!status) HYMLS::Tools::Fatal("Caught an exception",__FILE__,__LINE__);

  HYMLS::Tools::PrintTiming(HYMLS::Tools::out());
  HYMLS::Tools::PrintMemUsage(HYMLS::Tools::out());

comm->Barrier();
  HYMLS::Tools::Out("leaving main program");  

  MPI_Finalize();
  return 0;
  }


