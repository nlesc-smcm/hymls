#include <cstdlib>
#include <iostream>

#include <mpi.h>

#include "HYMLS_config.h"

#include "Epetra_config.h"
#include "Epetra_MpiComm.h"
#include "Epetra_SerialComm.h"
#include "Epetra_Map.h"
#include "Epetra_LocalMap.h"
#include "Epetra_Vector.h"
#include "Epetra_MultiVector.h"
#include "Epetra_CrsMatrix.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_ParameterListAcceptorHelpers.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#ifdef HYMLS_DEBUGGING
#include <signal.h>
#endif

#include "HYMLS_MainUtils.hpp"

#ifdef EPETRA_HAVE_OMP
#include <omp.h>
#endif

#include <fstream>

/*
#include "EpetraExt_HDF5.h"
#include "EpetraExt_Exception.h"
*/
#include "HYMLS_Macros.hpp"
#include "HYMLS_HyperCube.hpp"
#include "HYMLS_Tools.hpp"
#include "HYMLS_Preconditioner.hpp"
#include "HYMLS_Solver.hpp"
#include "HYMLS_MatrixUtils.hpp"



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

  Teuchos::RCP<Epetra_Map> map = Teuchos::null;
  Teuchos::RCP<Epetra_CrsMatrix> K = Teuchos::null;
  Teuchos::RCP<Epetra_Vector> u_ex = Teuchos::null;
  Teuchos::RCP<Epetra_Vector> f = Teuchos::null;
  Teuchos::RCP<Epetra_Vector> testvector = Teuchos::null;
  Teuchos::RCP<HYMLS::Preconditioner> precond = Teuchos::null;
  Teuchos::RCP<HYMLS::Solver> solver = Teuchos::null;
  Teuchos::RCP<Epetra_CrsMatrix> M = Teuchos::null;

  try {

  HYMLS_PROF("main","entire run");

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
    double diag_shift = driverList.get("Diagonal Shift",0.0);
    double diag_shift_i = driverList.get("Diagonal Shift (imag)",0.0);
    
    std::string galeriLabel=driverList.get("Galeri Label","");
    Teuchos::ParameterList galeriList;
    if (driverList.isSublist("Galeri")) galeriList = driverList.sublist("Galeri");
 
    // copy here rather than reference because the driver list will be removed 
    // alltogether...   
    bool read_problem=driverList.get("Read Linear System",false);
    std::string datadir,file_format;
    bool have_rhs=false;
    bool have_exact_sol=false;
    std::string nullSpaceType=driverList.get("Null Space Type","None");
    int dim0=0; // if the problem is read from a file, a null space can be read, too, with dim0 columns.

    Teuchos::ParameterList& solver_params = params->sublist("Solver");

    if (read_problem)
      {
      datadir = driverList.get("Data Directory","not specified");
      if (datadir=="not specified")
        {
        HYMLS::Tools::Error("'Data Directory' not specified although 'Read Linear System' is true",
                __FILE__,__LINE__);
        }                
      file_format = driverList.get("File Format","MatrixMarket");
      have_rhs = driverList.get("RHS Available",false);
      have_exact_sol = driverList.get("Exact Solution Available",false);
      if (nullSpaceType=="File") dim0=driverList.get("Null Space Dimension",0);
      }


    driverList.unused(std::cerr);
    params->remove("Driver");

        
    Teuchos::ParameterList& probl_params = params->sublist("Problem");
            
    int dim=probl_params.get("Dimension",2);
    int nx=probl_params.get("nx",32);
    int ny=probl_params.get("ny",nx);
    int nz=probl_params.get("nz",dim>2?nx:1);
    
    // copy problem sublist so that the main utils don't modify the original
    Teuchos::ParameterList probl_params_cpy = probl_params;
    std::string eqn=probl_params_cpy.get("Equations", "not-set");

    map = HYMLS::MainUtils::create_map(*comm, params); 
#ifdef HYMLS_STORE_MATRICES
HYMLS::MatrixUtils::Dump(*map,"MainMatrixMap.txt");
#endif

  HYMLS::Tools::StartMemory("main: Matrix");
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
  HYMLS::Tools::StopMemory("main: Matrix",true);

  if (store_matrix)
    {
    HYMLS::Tools::Out("store matrix...");
    HYMLS::MatrixUtils::Dump(*K, "Matrix.txt",false);
    }

  // create a random exact solution
  Teuchos::RCP<Epetra_MultiVector> x_ex = Teuchos::rcp(new Epetra_MultiVector(*map,numRhs));

  // construct right-hand side
  Teuchos::RCP<Epetra_MultiVector> b = Teuchos::rcp(new Epetra_MultiVector(*map,numRhs));

  // approximate solution
  Teuchos::RCP<Epetra_MultiVector> x = Teuchos::rcp(new Epetra_MultiVector(*map,numRhs));

  // read nullspace from a file if requiested
  Teuchos::RCP<Epetra_MultiVector> nullSpace=Teuchos::null;
  if (read_problem)
    {
    if (nullSpaceType=="File")
      {
      nullSpace=Teuchos::rcp(new Epetra_MultiVector(*map,dim0));
      std::string nullSpace_file=datadir+"/nullSpace.mtx";
      HYMLS::Tools::Out("Try to read null space from file '"+nullSpace_file+"'");
      HYMLS::MatrixUtils::mmread(nullSpace_file,*nullSpace);
      }
    }

  if (nullSpaceType!="None" && nullSpace==Teuchos::null)
    {
    nullSpace=HYMLS::MainUtils::create_nullspace(*map, nullSpaceType, probl_params);
    dim0=nullSpace->NumVectors();
    }
#ifdef HYMLS_STORE_MATRICES
  if (nullSpace!=Teuchos::null)
  {
    HYMLS::MatrixUtils::Dump(*nullSpace,"nullSpace.mtx");
  }
#endif  
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
      //b=Teuchos::rcp(new Epetra_Vector(*map));
      HYMLS::MatrixUtils::Random(*b);
      }
    }

  //bool do_deflation = (solver_params.get("Deflated Subspace Dimension",0)>0);
  bool do_deflation = solver_params.get("Use Deflation", false);
    
  HYMLS::Tools::Out("Create dummy mass matrix");
  M= Teuchos::rcp(new Epetra_CrsMatrix(Copy,*map,1,true));

  hymls_gidx gid;
  double val1=1.0/(nx*ny*nz);
  double val0=0.0;
  if (eqn=="Stokes-C")
    {
    int dof=dim+1;
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
          if (K->GCID64(indA[0])==K->GRID64(i))
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

  Epetra_Vector diagK(*map);
  CHECK_ZERO(K->ExtractDiagonalCopy(diagK));

  testvector = HYMLS::MainUtils::create_testvector(probl_params_cpy, *K);

  HYMLS::Tools::Out("Create Preconditioner");

  HYMLS::Tools::StartMemory("main: Initialize Preconditioner");
  precond = Teuchos::rcp(new HYMLS::Preconditioner(K, params, testvector));

  HYMLS::Tools::Out("Initialize Preconditioner...");
  HYMLS::Tools::StartTiming("main: Initialize Preconditioner");
  CHECK_ZERO(precond->Initialize());
  HYMLS::Tools::StopTiming("main: Initialize Preconditioner",true);
  HYMLS::Tools::StopMemory("main: Initialize Preconditioner",true);

  HYMLS::Tools::Out("Create Solver");
  solver = Teuchos::rcp(new HYMLS::Solver(K, precond, params,numRhs));

  solver->SetMassMatrix(M);

for (int f=0;f<numComputes;f++)
  {
  if (diag_shift_i!=0.0)
    {
    HYMLS::Tools::Warning("complex shifts not implemented",__FILE__,__LINE__);
    }
  if (perturbation!=0 || diag_shift!=0)
    {
    // change the matrix values just to see if that works
    Epetra_Vector diag=diagK;
    Epetra_Vector diag_pert(*map);
    HYMLS::MatrixUtils::Random(diag_pert);
    for (int i=0;i<diag_pert.MyLength();i++)
      {
      diag[i]=diag[i] + diag_shift + diag_pert[i]*perturbation;
      }
    CHECK_ZERO(K->ReplaceDiagonalValues(diag));
    }
  HYMLS::Tools::Out("Compute Preconditioner ("+Teuchos::toString(f+1)+")");

  if (nullSpace!=Teuchos::null)
    {
    CHECK_ZERO(solver->setBorder(nullSpace));
    }

  if (precond!=Teuchos::null) 
    {
    HYMLS::Tools::StartMemory("main: Compute Preconditioner");
    HYMLS::Tools::StartTiming("main: Compute Preconditioner");
    CHECK_ZERO(precond->Compute());
    HYMLS::Tools::StopTiming("main: Compute Preconditioner",true);
    HYMLS::Tools::StopMemory("main: Compute Preconditioner",true);
    }

  if (do_deflation)
    {
    CHECK_ZERO(solver->SetupDeflation());
    }
  
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
      // project out the null space from x_ex, x_ex <- x_ex - V0*(V0'x_ex)
      if (nullSpace!=Teuchos::null)
        {
        Epetra_SerialComm serialComm;
        Epetra_LocalMap localMap(dim0,0,*comm);
        Teuchos::RCP<Epetra_MultiVector> M = Teuchos::rcp(new Epetra_MultiVector(localMap,1),true);
                            
        CHECK_ZERO(M->Multiply('T','N',1.0,*nullSpace,*x_ex,0.0));
        CHECK_ZERO(x_ex->Multiply('N','N',-1.0,*nullSpace,*M,1.0));
        }
      CHECK_ZERO(K->Multiply(false,*x_ex,*b));
      }

    HYMLS::Tools::Out("Solve ("+Teuchos::toString(s+1)+")");
    HYMLS::Tools::StartMemory("main: Solve");
    HYMLS::Tools::StartTiming("main: Solve");
    //CHECK_ZERO(
    solver->ApplyInverse(*b,*x);
    //);
    HYMLS::Tools::StopTiming("main: Solve",true);
    HYMLS::Tools::StopMemory("main: Solve",true);

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
HYMLS_DEBVAR(*x);
HYMLS_DEBVAR(*b);
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
   if (perturbation!=0 || diag_shift!=0)
    {
    CHECK_ZERO(K->ReplaceDiagonalValues(diagK));
    }
 }//f - number of factorizations

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

  map = Teuchos::null;
  K = Teuchos::null;
  u_ex = Teuchos::null;
  f = Teuchos::null;
  testvector = Teuchos::null;
  precond = Teuchos::null;
  solver = Teuchos::null;
  M = Teuchos::null;

  comm->Barrier();

  HYMLS::Tools::Out("leaving main program");

  MPI_Finalize();
  return 0;
  }


