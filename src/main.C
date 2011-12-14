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
#include "Teuchos_StandardCatchMacros.hpp"

#include "main_utils.H"

#include "Galeri_Maps.h"
#include "Galeri_CrsMatrices.h"

/*
#include "EpetraExt_HDF5.h"
#include "EpetraExt_Exception.h"
*/

#include "HYMLS_Tools.H"
#include "HYMLS_Preconditioner.H"
#include "HYMLS_Solver.H"
#include "HYMLS_MatrixUtils.H"



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
    int numRhs   =driverList.get("Number of rhs",1);
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
  int dof=1;
  if (eqn=="Laplace")
    {
    try {
      map=rcp(Galeri::CreateMap(mapType, *comm, galeriList));
      } catch (Galeri::Exception G) {G.Print();}
    }
  else if (eqn=="Stokes-C")
    {
    dof=dim+1;
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
    K->Scale(-1.0); // we like our matrix negative definite
                     // (just to conform with the diffusion operator in the NSE,
                     // the solver works anyway, of course).
    }

  // create a random exact solution
  Teuchos::RCP<Epetra_MultiVector> x_ex = Teuchos::rcp(new Epetra_MultiVector(*map,numRhs));

#ifdef TESTING
  int seed=42;
#endif

  // construct right-hand side
  Teuchos::RCP<Epetra_MultiVector> b = Teuchos::rcp(new Epetra_MultiVector(*map,numRhs));

  // approximate solution
  Teuchos::RCP<Epetra_MultiVector> x = Teuchos::rcp(new Epetra_MultiVector(*map,numRhs));
  
  if (read_problem)
    {
    b=read_vector("rhs",datadir,file_format,map);
    if (have_exact_sol)
      {
      x_ex=read_vector("sol",datadir,file_format,map);
      }
    }

  ParameterList& solver_params = params->sublist("Solver");
  bool do_deflation = (solver_params.get("Deflated Subspace Dimension",0)>0);
  Teuchos::RCP<Epetra_CrsMatrix> M = Teuchos::null;
  if (do_deflation) // need a mass matrix
    {
    HYMLS::Tools::Out("Create dummy mass matrix");
    M=Teuchos::rcp(new Epetra_CrsMatrix(Copy,*map,1,true));
    int gid;
    double val1=1.0/(nx*ny*nz);
    double val0=0.0;
    if (dof>1)
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
  
  HYMLS::Tools::Out("Create Preconditioner");

  RCP<HYMLS::Preconditioner> precond = rcp(new HYMLS::Preconditioner(K, params));

  HYMLS::Tools::Out("Initialize Preconditioner...");
  CHECK_ZERO(precond->Initialize());

  HYMLS::Tools::Out("Create Solver");
  RCP<HYMLS::Solver> solver = rcp(new HYMLS::Solver(K, precond, params,numRhs));

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
      diag[i]=diag[i] + diag_pert[i]*perturbation;
      }
    CHECK_ZERO(K->ReplaceDiagonalValues(diag));
    }
  HYMLS::Tools::Out("Compute Solver ("+Teuchos::toString(f+1)+")");
  CHECK_ZERO(precond->Compute());
  if (do_deflation)
    {
    solver->SetMassMatrix(M);
    CHECK_ZERO(solver->SetupDeflation());
    }
    
 // std::cout << *solver << std::endl;
  
  for (int s=0;s<numSolves;s++)
    {
    if (read_problem==false)
      {
#ifdef TESTING
      seed++;
      CHECK_ZERO(HYMLS::MatrixUtils::Random(*x_ex, seed));
#else
      CHECK_ZERO(HYMLS::MatrixUtils::Random(*x_ex));
#endif
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
  
    double errNorm[numRhs],resNorm[numRhs],rhsNorm[numRhs];
  
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
  
    } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr, status);

  STOP_TIMER(std::string("main"),"entire run");  
  
  HYMLS::Tools::PrintTiming(HYMLS::Tools::out());

comm->Barrier();
  HYMLS::Tools::Out("leaving main program");  
  

  MPI_Finalize();
  return 0;
  }


