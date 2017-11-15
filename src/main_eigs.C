#include <cstdlib>
#include <iostream>

#include <mpi.h>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_MultiVector.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Import.h"
#include "EpetraExt_MultiVectorIn.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_ParameterListAcceptorHelpers.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include <BelosIMGSOrthoManager.hpp>

#include "HYMLS_config.h"

#ifdef HYMLS_DEBUGGING
#include <signal.h>
#endif

#include "main_utils.H"

#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziEpetraAdapter.hpp"

#include "AnasaziBlockKrylovSchurSolMgr.hpp"

#ifdef HYMLS_USE_PHIST
#include "AnasaziPhistSolMgr.hpp"
#endif

#ifdef EPETRA_HAVE_OMP
#include <omp.h>
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
#include "HYMLS_Tester.H"

typedef double ST;
typedef Epetra_MultiVector MV;
typedef Epetra_Operator OP;
#ifdef HYMLS_USE_PHIST_CORRECTION_SOLVER
typedef HYMLS::Preconditioner PHIST_PREC;
#else
typedef HYMLS::Solver PHIST_PREC;
#endif

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

#ifdef EPETRA_HAVE_OMP
#warning "Epetra is installed with OpenMP support, make sure to set OMP_NUM_THREADS=1"
  // If Epetra tries to parallelize local ops this causes
  // massive problems because many of our data tructures 
  // are so small.
  omp_set_num_threads(1);
#endif
    
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

    std::string startingBasisFile = driverList.get("Starting Basis", "None");
    
    std::string nullSpaceType = driverList.get("Null Space Type","None");
    int dim0=0;

    // copy here rather than reference because the driver list will be removed 
    // alltogether...   
    bool read_problem=driverList.get("Read Linear System",false);
    
    std::string datadir,file_format;
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
      have_massmatrix = driverList.get("Mass Matrix Available",false);
      if (nullSpaceType=="File")
        {
        dim0=driverList.get("Null Space Dimension",1);
        }
      }

    driverList.unused(std::cerr);
    params->remove("Driver");

    Teuchos::ParameterList& probl_params = params->sublist("Problem");

    int dim=probl_params.get("Dimension",2);
    int dof=probl_params.get("Degrees of Freedom", 1);
/*
    int nx=probl_params.get("nx",32);
    int ny=probl_params.get("ny",nx);
    int nz=probl_params.get("nz",dim>2?nx:1);
*/
     
    // copy problem sublist so that the main utils don't modify the original
    Teuchos::ParameterList probl_params_cpy = probl_params;
    std::string eqn=probl_params_cpy.get("Equations", "not-set");

    map = HYMLS::MainUtils::create_map(*comm, params); 
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

    K=HYMLS::MainUtils::create_matrix(*map,probl_params_cpy,
        galeriLabel, galeriList);
    }

  // read or create the null space
  Teuchos::RCP<Epetra_MultiVector> nullSpace=Teuchos::null;
  if (nullSpaceType=="File")
    {
    nullSpace=Teuchos::rcp(new Epetra_MultiVector(*map,dim0));
    std::string nullSpace_file=datadir+"/nullSpace.mtx";
    HYMLS::Tools::Out("Try to read null space from file '"+nullSpace_file+"'");
    HYMLS::MatrixUtils::mmread(nullSpace_file,*nullSpace);
    }
  else if (nullSpaceType!="None")
    {
    nullSpace=HYMLS::MainUtils::create_nullspace(*K, nullSpaceType, probl_params);
    dim0=nullSpace->NumVectors();
    }

  // create start vector
  Teuchos::RCP<Epetra_MultiVector> x=Teuchos::rcp(new Epetra_Vector(*map));
  HYMLS::MatrixUtils::Random(*x);

  if (eqn=="Stokes-C")
    {
    dof=dim+1;
    }

  Teuchos::RCP<Epetra_CrsMatrix> M = Teuchos::null;
  if (false) // need a mass matrix
    {
    if (have_massmatrix)
      {
      M = HYMLS::MainUtils::read_matrix(datadir, file_format, map, "mass");
      }
    else
      {
      HYMLS::Tools::Out("Create dummy mass matrix");
      M=Teuchos::rcp(new Epetra_CrsMatrix(Copy,*map,1,true));
      hymls_gidx gid;
      // double val1=1.0/(nx*ny*nz);
      double val1=1.0; //for turing problem Weiyan 

      double val0=0.0;
      if (eqn=="Stokes-C")
        {
        for (int i=0;i<M->NumMyRows();i+=dof)
          {
          for (int j=i;j<i+dof-1;j++)
            {
            gid = map->GID64(j);
            CHECK_ZERO(M->InsertGlobalValues(gid,1,&val1,&gid));
            }
          gid = map->GID64(i+dof-1);
          CHECK_ZERO(M->InsertGlobalValues(gid,1,&val0,&gid));
          }
        }
      else
        {
        for (int i=0;i<M->NumMyRows();i++)
          {
          gid = map->GID64(i);
          CHECK_ZERO(M->InsertGlobalValues(gid,1,&val1,&gid));
          }
        }
      CHECK_ZERO(M->FillComplete());
      }
    }

    Teuchos::RCP<HYMLS::Preconditioner> precond = Teuchos::rcp(new HYMLS::Preconditioner(K, params));

  if (precond!=Teuchos::null)
    {

    HYMLS::Tools::Out("Initialize Preconditioner...");
    HYMLS::Tools::StartTiming ("main: Initialize Preconditioner");
    REPORT_MEM("main","before Initialize",0,0);

    CHECK_ZERO(precond->Initialize());
    REPORT_MEM("main","after Initialize",0,0);
    HYMLS::Tools::StopTiming("main: Initialize Preconditioner",true);
    }

  REPORT_MEM("main","before HYMLS",0,0);
  
  if (precond!=Teuchos::null)
    {
    HYMLS::Tools::StartTiming("main: Compute Preconditioner");
    CHECK_ZERO(precond->Compute());
    HYMLS::Tools::StopTiming("main: Compute Preconditioner",true);
    }

#if defined(HYMLS_USE_PHIST) && !defined(HYMLS_USE_PHIST_CORRECTION_SOLVER)
  Teuchos::ParameterList& solver_params = params->sublist("Solver");
  Teuchos::RCP<HYMLS::Solver> solver=Teuchos::rcp(new HYMLS::Solver(K,precond,params,1));
#endif

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

  if (startingBasisFile != "None")
    {
    // Use a provided starting basis
    Epetra_MultiVector *vecout;
    EpetraExt::MatrixMarketFileToMultiVector(startingBasisFile.c_str(), x->Map(), vecout);
    x = Teuchos::rcp(vecout);

    // Reorthogonalize because of round-off errors
    typedef Belos::IMGSOrthoManager<ST, MV, Epetra_Operator> orthoMan_t;
    Teuchos::RCP<orthoMan_t> ortho = Teuchos::rcp(new orthoMan_t("hist/orthog/imgs", M));

    Teuchos::RCP<const Teuchos::ParameterList> default_params = ortho->getValidParameters();
    params->setParametersNotAlreadySet(*default_params);

    ortho->setParameterList(params);
    Teuchos::ArrayView<Teuchos::RCP<const Epetra_MultiVector > > V_array;
    Teuchos::Array<Teuchos::RCP<Teuchos::SerialDenseMatrix<int, double> > > C_array;
    Teuchos::RCP< Teuchos::SerialDenseMatrix<int, double> > mat = Teuchos::null;
    ortho->projectAndNormalize(*x, Teuchos::null, C_array, mat, V_array);
    }
  else
    {
    // Use a random B-orthogonal starting vector
    Teuchos::RCP<MV> v0 = Teuchos::rcp(new Epetra_Vector(x->Map()));
    HYMLS::MatrixUtils::Random(*v0);

    if (eqn=="Stokes-C")
      {
      for (int i = 0; i < v0->MyLength(); i++)
        {
        if (v0->Map().GID64(i) % dof == dim-1)
          {
          (*v0)[0][i] = 0.0;
          }
        }
      

        precond->ApplyInverse(*v0, *x);

        // Make x B-orthogonal
        double result;
        M->Multiply(false, *x, *v0);
        x->Dot(*v0, &result);
        x->Scale(1.0/sqrt(result));
      }
    }
  
  HYMLS_TEST("main_eigs", isDivFree(*Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(K), *x), __FILE__, __LINE__);
  
  // Create the eigenproblem.
  HYMLS_DEBUG("create eigen-problem");
  Teuchos::RCP<Anasazi::BasicEigenproblem<ST, MV, OP> > eigProblem;
  // note: use the default constructor because otherwise only Op and not AOp gets set
  eigProblem = Teuchos::rcp( new Anasazi::BasicEigenproblem<ST,MV,OP>() );
    eigProblem->setA(K);
    if (M!=Teuchos::null) eigProblem->setM(M);
    eigProblem->setInitVec(x);
    eigProblem->setHermitian(false);
    eigProblem->setNEV(numEigs);

#ifndef HYMLS_USE_PHIST
  eigProblem->setPrec(precond);
#endif

  if (eigProblem->setProblem()==false)
    {
    HYMLS::Tools::Error("eigProblem->setProblem returned 'false'",__FILE__,__LINE__);
    }

#ifdef HYMLS_USE_PHIST
  Teuchos::RCP<PHIST_PREC> phist_prec=Teuchos::null;
# ifdef HYMLS_USE_PHIST_CORRECTION_SOLVER
  phist_prec=precond;
#else
  phist_prec=solver;
#endif
  Anasazi::PhistSolMgr<ST,MV,OP,PHIST_PREC> esolver(eigProblem,phist_prec,eigList);
#else
  Anasazi::BlockKrylovSchurSolMgr<ST,MV,OP> esolver(eigProblem,eigList);
#endif

  // Solve the problem to the specified tolerances or length
  Anasazi::ReturnType returnCode;
  HYMLS_DEBUG("solve eigenproblem");
  returnCode = esolver.solve();
  if (returnCode != Anasazi::Converged)

    HYMLS::Tools::Warning("Anasazi::EigensolverMgr::solve() returned unconverged.",
        __FILE__,__LINE__);
    

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
    HYMLS::Tools::Out("store solution...");
    HYMLS::MatrixUtils::mmwrite("Eigenvectors.txt", *eigSol.Evecs);
    }
    
  if (print_final_list)
    {
    if (comm->MyPID()==0)
      {
      HYMLS::Tools::out() << "parameter documentation is written to file param_doc.txt" << std::endl;
      std::ofstream ofs("param_doc.txt");
      ofs << "valid parameters for HYMLS::Preconditioner "<<std::endl;
      printValidParameters(*precond,ofs);
      }
    }
  
  
    }TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr, status);
  if (!status) HYMLS::Tools::Fatal("Caught an exception",__FILE__,__LINE__);

  HYMLS::Tools::PrintTiming(HYMLS::Tools::out());
  HYMLS::Tools::PrintMemUsage(HYMLS::Tools::out());

comm->Barrier();
  HYMLS::Tools::Out("leaving main program");  

  MPI_Finalize();
  return 0;
  }


