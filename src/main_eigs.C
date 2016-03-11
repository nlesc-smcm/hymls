#include <cstdlib>
#include <iostream>

#include <mpi.h>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_MultiVector.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Import.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_ParameterListAcceptorHelpers.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#ifdef HYMLS_DEBUGGING
#include <signal.h>
#endif

#include "main_utils.H"

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
#include "HYMLS_Preconditioner.H"
#include "HYMLS_Solver.H"
#include "HYMLS_MatrixUtils.H"

typedef double ST;
typedef Epetra_MultiVector MV;
typedef Epetra_Operator OP;
typedef HYMLS::Solver PREC;

Teuchos::RCP<Epetra_Map> GetVelocityMap(Epetra_Map const &map, int dim = 3, int dof = 4)
{
  int num = map.NumMyElements();
  int *elementList = new int[num];
  int *newElementList = new int[num];
  map.MyGlobalElements(elementList);
  int j = 0;
  for (int i = 0; i < num; ++i)
    {
    if (elementList[i] % dof != dim)
      {
      newElementList[j] = elementList[i];
      j++;
      }
    }
  Teuchos::RCP<Epetra_Map> newMap = Teuchos::rcp(new Epetra_Map(-1, j, newElementList, map.IndexBase(), map.Comm()));

  delete[] elementList;
  delete[] newElementList;

  return newMap;
}

Teuchos::RCP<Epetra_Import> GetVelocityImporter(Epetra_Map const &map, Teuchos::RCP<Epetra_Map> velocityMap=Teuchos::null, int dim = 3, int dof = 4)
{
  if (velocityMap == Teuchos::null)
    {
    velocityMap = GetVelocityMap(map, dim, dof);
    }
  return Teuchos::rcp(new Epetra_Import(*velocityMap, map));
}

int main(int argc, char* argv[])
  {
  MPI_Init(&argc, &argv);

#ifdef HYMLS_DEBUGGING
  signal(SIGINT,HYMLS::Tools::SignalHandler);
  signal(SIGSEGV,HYMLS::Tools::SignalHandler);
#endif

// random number initialization (if HYMLS_TESTING is defined
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

  HYMLS_PROF("main","entire run");
  REPORT_MEM("main","base line",0,0);

  std::string param_file;
  Teuchos::Array<std::string> extra_files;

  if (argc<2)
    {
    HYMLS::Tools::Out("USAGE: main_eigs <parameter_filename>");
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

  Teuchos::RCP<Teuchos::ParameterList> params = 
        Teuchos::getParametersFromXmlFile(param_file);
        
  for (int i=0;i<extra_files.size();i++)
    {
    Teuchos::updateParametersFromXmlFile(extra_files[i],params.ptr());
    }
        
    Teuchos::ParameterList& driverList = params->sublist("Driver");
    // copy out this list because we need to pass it on later
    Teuchos::ParameterList eigList = driverList.sublist("Eigenvalues");
    
    int numEigs=eigList.get("How Many",8);

    int seed = driverList.get("Random Seed",-1);
    if (seed!=-1) std::srand(seed);
    bool print_final_list = driverList.get("Store Final Parameter List",false);        
    bool store_solution = driverList.get("Store Solution",true);
    
    std::string galeriLabel=driverList.get("Galeri Label","");
    Teuchos::ParameterList galeriList;
    if (driverList.isSublist("Galeri")) galeriList = driverList.sublist("Galeri");
 
    // copy here rather than reference because the driver list will be removed 
    // alltogether...   
    bool read_problem=driverList.get("Read Linear System",false);
    string datadir,file_format;
    bool have_rhs=true;
    bool have_exact_sol=false;
    bool have_massmatrix=false;

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
      have_massmatrix = driverList.get("Mass Matrix Available",false);
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
#ifdef HYMLS_STORE_MATRICES
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

// create start vector
Teuchos::RCP<Epetra_Vector> x=Teuchos::rcp(new Epetra_Vector(*map));
HYMLS::MatrixUtils::Random(*x);

  Teuchos::ParameterList& solver_params = params->sublist("Solver");
  bool do_deflation = (solver_params.get("Deflated Subspace Dimension",0)>0);

  int dof = 1;
  if (eqn=="Stokes-C")
    {
    dof=dim+1;
    }

  Teuchos::RCP<Epetra_CrsMatrix> M = Teuchos::null;
  if (do_deflation||true) // need a mass matrix
    {
    if (have_massmatrix)
      {
      M = HYMLS::MainUtils::read_matrix(datadir, file_format, map, "mass");
      }
    else
      {
      HYMLS::Tools::Out("Create dummy mass matrix");
      M=Teuchos::rcp(new Epetra_CrsMatrix(Copy,*map,1,true));
      int gid;
      double val1=1.0/(nx*ny*nz);
      double val0=0.0;
      if (eqn=="Stokes-C")
        {
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
    }
//~ 
  //~ if (eqn=="Stokes-C")
    //~ {
    //~ // scale equations by -1 to make operator 'negative indefinite'
    //~ // (for testing the deflation capabilities)
//~ //    K->Scale(-1.0);
    //~ // put a zero in the mass matrix for singletons
    //~ if (M!=Teuchos::null)
      //~ {
     //~ int lenA, lenM;
      //~ int *indA, *indM;
      //~ double *valA, *valM;
      //~ for (int i=0;i<K->NumMyRows();i++)
        //~ {
        //~ CHECK_ZERO(K->ExtractMyRowView(i,lenA,valA,indA));
        //~ CHECK_ZERO(M->ExtractMyRowView(i,lenM,valM,indM));
        //~ if (lenA==1)
          //~ {
          //~ if (K->GCID(indA[0])==K->GRID(i))
            //~ {
            //~ for (int j=0;j<lenM;j++)
              //~ {
              //~ valM[j]=0.0;
              //~ }
            //~ }
          //~ }
        //~ }
      //~ }
    //~ }
  HYMLS::MatrixUtils::Dump(*M,"massMatrix.txt");
#ifdef HYMLS_STORE_MATRICES
  HYMLS::MatrixUtils::Dump(*M,"massMatrix.txt");
#endif
  HYMLS::Tools::Out("Create Preconditioner");

{
  Teuchos::RCP<Epetra_Import> velocityImporter = GetVelocityImporter(K->RangeMap());
  Teuchos::RCP<Epetra_Map> velocityMap = GetVelocityMap(K->DomainMap());
  Teuchos::RCP<Epetra_CrsMatrix> velocityK = Teuchos::rcp(new Epetra_CrsMatrix(*K, *velocityImporter));

  HYMLS::MatrixUtils::Dump(*velocityK, "origVelocityK.txt");
  HYMLS::MatrixUtils::Dump(*K, "origK.txt");
}

  int num = K->MaxNumEntries();
  Teuchos::RCP<Epetra_Map> velocityMap = GetVelocityMap(K->RangeMap(), dim, dof);
  Teuchos::RCP<Epetra_Import> velocityImporter = GetVelocityImporter(K->RangeMap(), velocityMap, dim, dof);
  Teuchos::RCP<Epetra_Map> velocityColMap = HYMLS::MatrixUtils::CreateColMap(*K, *velocityMap, *velocityMap);

  Teuchos::RCP<Epetra_CrsMatrix> velocityK = Teuchos::rcp(new
          Epetra_CrsMatrix(Copy, *velocityMap, *velocityColMap, num));
  CHECK_ZERO(velocityK->Import(*K, *velocityImporter, Insert));
  CHECK_ZERO(velocityK->FillComplete(*velocityMap, *velocityColMap));

  Teuchos::RCP<Epetra_CrsMatrix> velocityM = Teuchos::rcp(new
          Epetra_CrsMatrix(Copy, *velocityMap, *velocityColMap, num));
  CHECK_ZERO(velocityM->Import(*M, *velocityImporter, Insert));
  CHECK_ZERO(velocityM->FillComplete(*velocityMap, *velocityColMap));

  HYMLS::MatrixUtils::Dump(*velocityK, "velocityK.txt");
  HYMLS::MatrixUtils::Dump(*K, "K.txt");

    Teuchos::RCP<HYMLS::Preconditioner> precond = Teuchos::rcp(new HYMLS::Preconditioner(K, params));
    //~ Teuchos::RCP<HYMLS::Preconditioner> precond = Teuchos::null;

  if (precond!=Teuchos::null)
    {
    HYMLS::Tools::Out("Initialize Preconditioner...");
    HYMLS::Tools::StartTiming ("main: Initialize Preconditioner");
    REPORT_MEM("main","before Initialize",0,0);
    CHECK_ZERO(precond->Initialize());
    REPORT_MEM("main","after Initialize",0,0);
    HYMLS::Tools::StopTiming("main: Initialize Preconditioner",true);
    }

  HYMLS::Tools::Out("Create Solver");
  Teuchos::RCP<HYMLS::Solver> solver = Teuchos::rcp(new HYMLS::Solver(K, precond, params,1));

  // get the null space (if any), as specified in the xml-file
  Teuchos::RCP<const Epetra_MultiVector> Nul = solver->getNullSpace();

  REPORT_MEM("main","before HYMLS",0,0);
  
  if (precond!=Teuchos::null) 
    {
    HYMLS::Tools::StartTiming("main: Compute Preconditioner");
    CHECK_ZERO(precond->Compute());
    HYMLS::Tools::StopTiming("main: Compute Preconditioner",true);
    }

  if (M!=Teuchos::null)
    {
    solver->SetMassMatrix(M);
    }
  if (do_deflation)
    {
    //~ solver->SetMassMatrix(M);
    CHECK_ZERO(solver->SetupDeflation());
    }
  else
    {
    CHECK_ZERO(solver->setNullSpace());
    }
    

  // Set verbosity level
  int verbosity = Anasazi::Errors + Anasazi::Warnings;
  verbosity += Anasazi::IterationDetails;
  verbosity += Anasazi::OrthoDetails;
  verbosity += Anasazi::FinalSummary + Anasazi::TimingDetails;
#ifdef HYMLS_DEBUGGING
  verbosity += Anasazi::Debug;
#endif
  eigList.set("Verbosity",verbosity);
  eigList.set("Output Stream",HYMLS::Tools::out().getOStream());

  Teuchos::RCP<MV> v0 = Teuchos::rcp(new Epetra_Vector(x->Map()));
  HYMLS::MatrixUtils::Random(*v0);

  for (int i = 0; i < v0->MyLength(); i++)
    {
    if (v0->Map().GID(i) % dof == dim)
      {
      (*v0)[0][i] = 0.0;
      }
    }

  precond->ApplyInverse(*v0, *x);

  HYMLS_TEST("main_eigs",isDivFree(*Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(K), *x, dof, dim),__FILE__,__LINE__);

  Teuchos::RCP<Epetra_Vector> velocityx = Teuchos::rcp(new
          Epetra_Vector(*velocityMap));
  Teuchos::RCP<Epetra_Vector> velocityTmp = Teuchos::rcp(new
          Epetra_Vector(*velocityMap));
  CHECK_ZERO(velocityx->Import(*x, *velocityImporter, Insert));

  // Make x B-orthogonal
  double result;
  velocityM->Multiply(false, *velocityx, *velocityTmp);
  velocityx->Dot(*velocityTmp, &result);
  velocityx->Scale(1.0/sqrt(result));

  HYMLS::MatrixUtils::Dump(*velocityM, "velocityM.txt");
  HYMLS::MatrixUtils::Dump(*M, "M.txt");

  HYMLS::MatrixUtils::Dump(*velocityx, "velocityx.txt");
  HYMLS::MatrixUtils::Dump(*x, "x.txt");

  // Create the eigenproblem.
  HYMLS_DEBUG("create eigen-problem");
  Teuchos::RCP<Anasazi::BasicEigenproblem<ST, MV, OP> > eigProblem;
  eigProblem = Teuchos::rcp( new Anasazi::BasicEigenproblem<ST,MV,OP>(velocityK, velocityM, velocityx) );
  eigProblem->setHermitian(false);
  eigProblem->setNEV(numEigs);
  if (eigProblem->setProblem()==false)
    {
    HYMLS::Tools::Error("eigProblem->setPoroblem returned 'false'",__FILE__,__LINE__);
    }

#ifdef HAVE_PHIST
  Anasazi::PhistSolMgr<ST,MV,OP,PREC> jada(eigProblem,solver,eigList);
#else
  Anasazi::JacobiDavidsonSolMgr<ST,MV,OP,PREC> jada(eigProblem,solver,eigList);
#endif

  // Solve the problem to the specified tolerances or length
  Anasazi::ReturnType returnCode;
  HYMLS_DEBUG("solve eigenproblem");
  returnCode = jada.solve();
  if (returnCode != Anasazi::Converged)
    {
    HYMLS::Tools::Warning("Anasazi::EigensolverMgr::solve() returned unconverged.",
        __FILE__,__LINE__);
    }

  HYMLS_DEBUG("post-process returned solution");

  const Anasazi::Eigensolution<ST,MV>& eigSol =
        eigProblem->getSolution();
  
  const std::vector<Anasazi::Value<ST> >& evals = eigSol.Evals;
  numEigs = (int)evals.size();


  if (1)
    {
    // Output computed eigenvalues and their direct residuals
    HYMLS::Tools::out()<<std::endl<< "Computed Ritz Values"<< std::endl;
    HYMLS::Tools::out()<< std::setw(20) << "Real Part"
            << std::setw(20) << "Imag Part"
            << std::endl;
    HYMLS::Tools::out()<<"-----------------------------------------------------------"<<std::endl;
    for (int i=0; i<numEigs; i++)
      {
      HYMLS::Tools::out()<< std::setw(20) << evals[i].realpart
              << std::setw(20) << evals[i].imagpart
              << std::endl;
      }
    HYMLS::Tools::out()<<"-----------------------------------------------------------"<<std::endl;
    }

  REPORT_MEM("main","after HYMLS",0,0);

  if (store_solution)
    {
//    HYMLS::Tools::Out("store solution...");
//    HYMLS::MatrixUtils::Dump(*x, "EigenSolution.txt",false);
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


