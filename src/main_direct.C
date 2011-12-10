#include <cstdlib>

#include <iostream>

#include <mpi.h>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include "Galeri_Maps.h"
#include "Galeri_CrsMatrices.h"

/*
#include "EpetraExt_HDF5.h"
#include "EpetraExt_Exception.h"
*/

#include "HYMLS_Tools.H"
#include "HYMLS_Solver.H"
#include "HYMLS_MatrixUtils.H"

#include "Ifpack_Amesos.h"

#include "main_utils.H"

using namespace Teuchos;


/*
void ReadTestCase(std::string, int, int, 
                  RCP<Epetra_Comm> comm,
                  RCP<Epetra_Map>&,
                  RCP<Epetra_CrsMatrix>&,
                  RCP<Epetra_Vector>&,
                  RCP<Epetra_Vector>&);
*/

int main(int argc, char* argv[])
  {
  MPI_Init(&argc, &argv);

int status=0;

  RCP<Epetra_MpiComm> comm=rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  // construct file streams, otherwise the output won't work correctly
  HYMLS::Tools::InitializeIO(comm);
  
  START_TIMER(std::string("main"),"entire run");

  try {

  std::string param_file;

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
    }


  RCP<Epetra_Map> map;
  RCP<Epetra_CrsMatrix> K;
  RCP<Epetra_Vector> u_ex;
  RCP<Epetra_Vector> f;

  RCP<ParameterList> params = getParametersFromXmlFile(param_file);
        
    ParameterList& driverList = params->sublist("Driver");
        
    bool store_solution = driverList.get("Store Solution",true);
    bool store_matrix = driverList.get("Store Matrix",false);
    int numComputes=driverList.get("Number of factorizations",1);
    int numSolves=driverList.get("Number of solves",1);
    double perturbation = driverList.get("Diagonal Perturbation",0.0);

    bool read_problem=driverList.get("Read Linear System",false);
    string datadir,file_format;
    bool have_exact_sol;

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
    params->remove("Driver");

        
    ParameterList& probl_params = params->sublist("Problem");
            
    int dim=probl_params.get("Dimension",2);
    std::string eqn=probl_params.get("Equations","Laplace");

  int nx=probl_params.get("nx",32);
  int ny=probl_params.get("ny",nx);
  int nz=probl_params.get("nz",dim>2? nx:1);

  ParameterList galeriList;
  
  galeriList.set("nx",nx);
  galeriList.set("ny",ny);
  galeriList.set("nz",nz);

  if (eqn=="Laplace")
    {    
    std::string mapType="Cartesian"+toString(dim)+"D";

    HYMLS::Tools::Out("Create map");
    try {
    map=rcp(Galeri::CreateMap(mapType, *comm, galeriList));
    } catch (Galeri::Exception G) {G.Print();}
    }
  else if (eqn=="Stokes-C")
    {
    int dof=dim+1;
    map=HYMLS::MatrixUtils::CreateMap(nx,ny,nz,dof,0,*comm);
    }
  else
    {
    HYMLS::Tools::Error("cannot determine problem type from 'Equation' parameter "+eqn,
        __FILE__, __LINE__);
    }

  // right-hand side
  Teuchos::RCP<Epetra_Vector> b;

  // approximate solution
  Teuchos::RCP<Epetra_Vector> x = Teuchos::rcp(new Epetra_Vector(*map));

  // exact solution
  Teuchos::RCP<Epetra_Vector> x_ex;

  if (read_problem)
    {
    K=read_matrix(datadir,file_format,map);
    b=read_vector("rhs",datadir,file_format,map);
    if (have_exact_sol)
      {
      x_ex=read_vector("sol",datadir,file_format,map);
      }
    }
  else
    {
    HYMLS::Tools::Out("Create matrix");

    std::string matrixType=eqn+toString(dim)+"D";
    try {
    K=rcp(Galeri::CreateCrsMatrix(matrixType, map.get(), galeriList));
    } catch (Galeri::Exception G) {G.Print();}

    // create a random exact solution
    x_ex = Teuchos::rcp(new Epetra_Vector(*map));
    b = Teuchos::rcp(new Epetra_Vector(*map));
    have_exact_sol=true;
    }
#ifdef TESTING  
  HYMLS::MatrixUtils::Dump(*K, "Matrix.txt");
#endif  
  
  
  HYMLS::Tools::Out("Create Solver");

  //RCP<HYMLS::Solver> solver = rcp(new HYMLS::Solver(K, params));
  RCP<Ifpack_Amesos> solver = rcp(new Ifpack_Amesos(K.get()));
  Teuchos::ParameterList& directList = params->sublist("Solver").sublist("Coarse Solver");
  solver->SetParameters(directList);

  HYMLS::Tools::Out("Initialize Solver...");
  START_TIMER(std::string("main"),"INITIALIZE");
  CHECK_ZERO(solver->Initialize());
  STOP_TIMER(std::string("main"),"INITIALIZE");

for (int f=0;f<numComputes;f++)
  {
  /*
  // change the matrix values just to see if that works
  Epetra_Vector diag(*x_ex);
  CHECK_ZERO(K->ExtractDiagonalCopy(diag));
  Epetra_Vector diag_pert(*x_ex);
  HYMLS::MatrixUtils::Random(diag_pert);
  for (int i=0;i<diag_pert.MyLength();i++)
    {
    diag[i]=diag[i] + diag_pert[i]*perturbation;
    }
  CHECK_ZERO(K->ReplaceDiagonalValues(diag));
  */
  HYMLS::Tools::Out("Compute Solver ("+Teuchos::toString(f+1)+")");
  START_TIMER(std::string("main"),"COMPUTE");
  CHECK_ZERO(solver->Compute());
  STOP_TIMER(std::string("main"),"COMPUTE");
  
 // std::cout << *solver << std::endl;
  
  for (int s=0;s<numSolves;s++)
    {
    if (!read_problem)
      {
      CHECK_ZERO(HYMLS::MatrixUtils::Random(*x_ex));
      CHECK_ZERO(K->Multiply(false,*x_ex,*b));
      }

    HYMLS::Tools::Out("Solve ("+Teuchos::toString(s+1)+")");
    START_TIMER(std::string("main"),"SOLVE");
    CHECK_ZERO(solver->ApplyInverse(*b,*x));
    STOP_TIMER(std::string("main"),"SOLVE");
    
    // subtract constant from pressure if solving Stokes-C
    if (eqn=="Stokes-C")
      {
      int dof=dim+1;
      double pref=(*x)[dim];
      if (have_exact_sol)
        {
        pref -= (*x_ex)[dim];
        }
      for (int i=dim; i<x->MyLength();i+=dof)
        {
        (*x)[i]-=pref;
        }
      }
  
    HYMLS::Tools::Out("Compute residual.");
  
    // compute residual and error vectors

    Teuchos::RCP<Epetra_Vector> res = Teuchos::rcp(new Epetra_Vector(*map));
    Teuchos::RCP<Epetra_Vector> err = Teuchos::rcp(new Epetra_Vector(*map));

    CHECK_ZERO(K->Multiply(false,*x,*res));
    CHECK_ZERO(res->Update(1.0,*b,-1));
    if (have_exact_sol)
      {
      CHECK_ZERO(err->Update(1.0,*x,-1.0,*x_ex,0.0));
      }
  
    double errNorm,resNorm;
  
    err->Norm2(&errNorm);
    res->Norm2(&resNorm);
  
    HYMLS::Tools::Out("Residual Norm: "+toString(resNorm));
    if (have_exact_sol)
      {
      HYMLS::Tools::Out("Error Norm: "+toString(errNorm));
      }
    }
  }
  
  if (store_matrix)
    {
    HYMLS::Tools::Out("store matrix...");
    HYMLS::MatrixUtils::Dump(*K, "Matrix.txt");
    }
    
  if (store_solution)
    {
    HYMLS::Tools::Out("store solution...");
    HYMLS::MatrixUtils::Dump(*x_ex, "ExactSolution.txt");
    HYMLS::MatrixUtils::Dump(*x, "Solution.txt");
    HYMLS::MatrixUtils::Dump(*b, "RHS.txt");
    }

if (comm->MyPID()==0)
  {
  double tInit = solver->InitializeTime();
  double fInit = solver->InitializeFlops();
  double tCompute = solver->ComputeTime();
  double fCompute = solver->ComputeFlops();
  double tSolve = solver->ApplyInverseTime();
  double fSolve = solver->ApplyInverseFlops();
  std::cout << std::scientific;
  std::cout << "======= TIMING & PERFORMANCE ========"<<std::endl;
  std::cout << " Init: "<<tInit<<"s ("<<fInit/tInit << " flop/s"<<std::endl;
  std::cout << "Setup: "<<tCompute<<"s ("<<fCompute/tCompute << " flop/s"<<std::endl;
  std::cout << "Solve: "<<tSolve<<"s ("<<fSolve/tSolve << " flop/s"<<std::endl;
  std::cout << "====================================="<<std::endl;
  }

  
    } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr, status);

  STOP_TIMER(std::string("main"),"entire run");  
  HYMLS::Tools::PrintTiming(HYMLS::Tools::out());

comm->Barrier();
  HYMLS::Tools::Out("leaving main program");  
  

  MPI_Finalize();
  return 0;
  }


/*
void ReadTestCase(std::string problem, int nx, int sx, 
                  RCP<Epetra_Comm> comm,
                  RCP<Epetra_Map>& map,
                  RCP<Epetra_CrsMatrix>& K,
                  RCP<Epetra_Vector>& u_ex,
                  RCP<Epetra_Vector>& f)

  {
  std::stringstream ss;
  ss<<"data/"<<problem<<nx<<".h5";
  std::string filename=ss.str();
  
  std::cout << "***************************************"<<std::endl;
  std::cout << "* READING TEST PROBLEM FROM '"<<filename<<"'"<<std::endl;
  std::cout << "***************************************"<<std::endl;
  
  EpetraExt::HDF5 file(*comm);
  file.Open(filename);
  
  if (!file.IsOpen())
    {
    std::cerr << "Error opening testcase file."<<std::endl;
    std::cerr << "Make sure an appropriate data file exists."<<std::endl;
    std::cerr << "For your input it should be '"<<filename<<"'"<<std::endl;
    PrintUsage(std::cerr);
    MPI_Finalize();
    exit(0);
    }
  int nx_,ny_;
  std::cout << "Read grid size..."<<std::endl;
  try{
  file.Read("grid","nx",nx_);
  } catch (EpetraExt::Exception e){e.Print();}
  std::cout << nx_<<std::endl;
  return;
  file.Read("grid","ny",ny_);
  std::cout << "grid-size: "<<nx_<< "x"<<ny_<<std::endl;
  
  if (nx!=nx_||nx!=ny_)
    {
    Error("Dimension mismatch between filename and contents!",-1);
    }
  
  }
*/
