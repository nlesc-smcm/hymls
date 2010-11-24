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

#include "EpetraExt_CrsMatrixIn.h"
#include "EpetraExt_VectorIn.h"

#include "Galeri_Maps.h"
#include "Galeri_CrsMatrices.h"

/*
#include "EpetraExt_HDF5.h"
#include "EpetraExt_Exception.h"
*/

#include "HYMLS_Tools.H"
#include "HYMLS_Solver.H"
#include "HYMLS_MatrixUtils.H"


Teuchos::RCP<Epetra_CrsMatrix> read_matrix(string datadir,
        string file_format, Teuchos::RCP<Epetra_Map> map);

Teuchos::RCP<Epetra_Vector> read_vector(string name,string datadir, string file_format,Teuchos::RCP<Epetra_Map> map);

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

  RCP<ParameterList> params = Teuchos::getParametersFromXmlFile(param_file);
        
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


  ParameterList galeriList;
  
  int nx=probl_params.get("nx",32);
  int ny=probl_params.get("ny",nx);
  int nz=probl_params.get("nz",(dim>2)?nx:1);

  galeriList.set("nx",nx);
  galeriList.set("ny",ny);
  galeriList.set("nz",nz);
    
  std::string mapType="Cartesian"+Teuchos::toString(dim)+"D";

  HYMLS::Tools::Out("Create map");
  
  if (eqn=="Laplace")
    {
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
  

  if (read_problem)
    {
    K=read_matrix(datadir,file_format,map);
    }
  else
    {
    HYMLS::Tools::Out("Create matrix");

    std::string matrixType=eqn+Teuchos::toString(dim)+"D";
    try {
      K=rcp(Galeri::CreateCrsMatrix(matrixType, map.get(), galeriList));
      } catch (Galeri::Exception G) {G.Print();}
    }
#ifdef TESTING  
  HYMLS::MatrixUtils::Dump(*K, "Matrix.txt");
#endif  

  // create a random exact solution
  Teuchos::RCP<Epetra_Vector> x_ex = Teuchos::rcp(new Epetra_Vector(*map));

#ifdef DEBUGGING
  int seed=42;
  CHECK_ZERO(HYMLS::MatrixUtils::Random(*x_ex, seed));
#else
  CHECK_ZERO(HYMLS::MatrixUtils::Random(*x_ex));
#endif

  // construct right-hand side
  Teuchos::RCP<Epetra_Vector> b = Teuchos::rcp(new Epetra_Vector(*map));

  // approximate solution
  Teuchos::RCP<Epetra_Vector> x = Teuchos::rcp(new Epetra_Vector(*map));
  
  if (read_problem)
    {
    b=read_vector("rhs",datadir,file_format,map);
    if (have_exact_sol)
      {
      x_ex=read_vector("sol",datadir,file_format,map);
      }
    }
  
  HYMLS::Tools::Out("Create Solver");

  RCP<HYMLS::Solver> solver = rcp(new HYMLS::Solver(K, params));

  HYMLS::Tools::Out("Initialize Solver...");
  CHECK_ZERO(solver->Initialize());

for (int f=0;f<numComputes;f++)
  {
  if (perturbation!=0)
    {
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
    }
  HYMLS::Tools::Out("Compute Solver ("+Teuchos::toString(f+1)+")");
  CHECK_ZERO(solver->Compute());
  
 // std::cout << *solver << std::endl;
  
  for (int s=0;s<numSolves;s++)
    {
    if (read_problem==false)
      {
      CHECK_ZERO(HYMLS::MatrixUtils::Random(*x_ex));
      CHECK_ZERO(K->Multiply(false,*x_ex,*b));
      }

    HYMLS::Tools::Out("Solve ("+Teuchos::toString(s+1)+")");
    CHECK_ZERO(solver->ApplyInverse(*b,*x));
  
    HYMLS::Tools::Out("Compute residual.");
  
    // compute residual and error vectors

    Teuchos::RCP<Epetra_Vector> res = Teuchos::rcp(new Epetra_Vector(*map));
    Teuchos::RCP<Epetra_Vector> err = Teuchos::rcp(new Epetra_Vector(*map));

    CHECK_ZERO(K->Multiply(false,*x,*res));
    CHECK_ZERO(res->Update(1.0,*b,-1));
  
    CHECK_ZERO(err->Update(1.0,*x,-1.0,*x_ex,0.0));
  
    double errNorm,resNorm;
  
    err->Norm2(&errNorm);
    res->Norm2(&resNorm);
  
    HYMLS::Tools::Out("Residual Norm: "+Teuchos::toString(resNorm));
    HYMLS::Tools::Out("Error Norm: "+Teuchos::toString(errNorm));
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


Teuchos::RCP<Epetra_CrsMatrix> read_matrix(string datadir,string file_format, Teuchos::RCP<Epetra_Map> map)
  {
  
  if (map==Teuchos::null)
    {
    HYMLS::Tools::Error("map must have been allocated before this function",
        __FILE__,__LINE__);
    }
  
  string suffix;
  if (file_format=="MatrixMarket")
    {
    suffix=".mtx";
    }
  else
    {
    HYMLS::Tools::Error("File format '"+file_format+"' not supported",__FILE__,__LINE__);
    }

  string filename = datadir+"/jac"+suffix;

  HYMLS::Tools::Out("... read matrix from file '"+filename+"'");
  HYMLS::Tools::Out("    file format: "+file_format);

  Teuchos::RCP<Epetra_CrsMatrix> K=Teuchos::null;

  if (file_format=="MatrixMarket")
    {
    Epetra_CrsMatrix* Kptr;
    CHECK_ZERO(EpetraExt::MatrixMarketFileToCrsMatrix(filename.c_str(),*map,Kptr));
    K=Teuchos::rcp(Kptr,true);
    }
  else
    {
    HYMLS::Tools::Error("File format '"+file_format+"' not supported",__FILE__,__LINE__);
    }
  return K;
  }

Teuchos::RCP<Epetra_Vector>  read_vector(string name,string datadir,
                string file_format,Teuchos::RCP<Epetra_Map> map)
  {
  if (map==Teuchos::null)
    {
    HYMLS::Tools::Error("map must have been allocated before this function",
        __FILE__,__LINE__);
    }
  
  string suffix;
  if (file_format=="MatrixMarket")
    {
    suffix=".mtx";
    }
  else
    {
    HYMLS::Tools::Error("File format '"+file_format+"' not supported",__FILE__,__LINE__);
    }

  string filename = datadir+"/"+name+suffix;
  
  HYMLS::Tools::Out("... read vector from file '"+filename+"'");
  HYMLS::Tools::Out("    file format: "+file_format);
  
  Teuchos::RCP<Epetra_Vector> v;

  if (file_format=="MatrixMarket")
    {
    Epetra_Vector* vptr;
    CHECK_ZERO(EpetraExt::MatrixMarketFileToVector(filename.c_str(),*map,vptr));
    v=Teuchos::rcp(vptr,true);
    }
  else
    {
    HYMLS::Tools::Error("File format '"+file_format+"' not supported",__FILE__,__LINE__);
    }
  return v;
  }

/////////////////////////////////////////////////////////////////////////////////////////

#if 0
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
#endif
