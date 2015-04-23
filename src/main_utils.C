
#include "main_utils.H"

#include <iostream>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Import.h"

#include "Teuchos_RCP.hpp"

#include "EpetraExt_CrsMatrixIn.h"
#include "EpetraExt_VectorIn.h"

#include "HYMLS_Tools.H"
#include "HYMLS_MatrixUtils.H"

#include "GaleriExt_Cross2DN.h"
#include "Galeri_CrsMatrices.h"

#include "GaleriExt_Darcy2D.h"
#include "GaleriExt_Darcy3D.h"

namespace HYMLS {

namespace MainUtils {

Teuchos::RCP<Epetra_CrsMatrix> read_matrix(string datadir,string file_format, Teuchos::RCP<Epetra_Map> map, std::string name)
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
  else if (file_format=="MatrixMarket (2)")
    {
    suffix="2.mtx";
    }
  else
    {
    HYMLS::Tools::Error("File format '"+file_format+"' not supported",__FILE__,__LINE__);
    }

  string filename = datadir+"/"+name+suffix;

  HYMLS::Tools::Out("... read matrix from file '"+filename+"'");
  HYMLS::Tools::Out("    file format: "+file_format);

  Teuchos::RCP<Epetra_CrsMatrix> K=Teuchos::null;

  if (file_format=="MatrixMarket")
    {
    Epetra_CrsMatrix* Kptr;
    CHECK_ZERO(EpetraExt::MatrixMarketFileToCrsMatrix(filename.c_str(),*map,Kptr));
    K=Teuchos::rcp(Kptr,true);
    }
  else if (file_format=="MatrixMarket (2)")
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
  else if (file_format=="MatrixMarket (2)")
    {
    suffix="2.mtx";
    }
  else
    {
    HYMLS::Tools::Error("File format '"+file_format+"' not supported",__FILE__,__LINE__);
    }

  string filename = datadir+"/"+name+suffix;
  
  HYMLS::Tools::Out("... read vector from file '"+filename+"'");
  HYMLS::Tools::Out("    file format: "+file_format);
  
  Teuchos::RCP<Epetra_Vector> v;

  if (file_format=="MatrixMarket" || file_format=="MatrixMarket (2)")
    {
    // the EpetraExt function only works for a linear map in parallel,
    // so we need to reindex ourselves:
    Epetra_Map linearMap(map->NumGlobalElements(),
                         map->NumMyElements(),
                         map->IndexBase(),
                         map->Comm());

    Epetra_Vector* vptr;
    CHECK_ZERO(EpetraExt::MatrixMarketFileToVector(filename.c_str(),linearMap,vptr));
    
    v=Teuchos::rcp(new Epetra_Vector(*map));
    Epetra_Import import(*map,linearMap);
    CHECK_ZERO(v->Import(*vptr,import,Insert));
    delete vptr;
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

                
Teuchos::RCP<Epetra_Map> create_map(const Epetra_Comm& comm,
                                    Teuchos::ParameterList& probl_params)
  {

  int dim = probl_params.get("Dimension",2);
  int nx=probl_params.get("nx",32);
  int ny=probl_params.get("ny",nx);
  int nz=probl_params.get("nz",(dim>2)?nx:1);
  
  std::string eqn = probl_params.get("Equations","Laplace");

  Teuchos::RCP<Epetra_Map> map=Teuchos::null;


  Teuchos::ParameterList galeriList;
  galeriList.set("nx",nx);
  galeriList.set("ny",ny);
  galeriList.set("nz",nz);
    
  std::string mapType="Cartesian"+Teuchos::toString(dim)+"D";

  HYMLS::Tools::Out("Create map");
  int dof=probl_params.get("Degrees of Freedom",1);
  bool is_complex=probl_params.get("Complex Arithmetic",false);
  if (is_complex) dof*=2;

  if (eqn=="Laplace" || eqn=="Laplace Neumann")
    {
    map=HYMLS::MatrixUtils::CreateMap(nx,ny,nz,dof,0,comm);
//    try {
//      map= Teuchos::rcp(Galeri::CreateMap(mapType, *comm, galeriList));
//      } catch (Galeri::Exception G) {G.Print();}
    }
  else if (eqn=="Stokes-C")
    {
    dof=dim+1;
    map=HYMLS::MatrixUtils::CreateMap(nx,ny,nz,dof,0,comm);
    }
  else
    {
    HYMLS::Tools::Error("cannot determine problem type from 'Equation' parameter "+eqn,
        __FILE__, __LINE__);
    }
  return map;
  }

Teuchos::RCP<Epetra_CrsMatrix> create_matrix(const Epetra_Map& map,
                                Teuchos::ParameterList& probl_params,
                                std::string galeriLabel,
                                Teuchos::ParameterList& galeriList
                                )
  {
  Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::null;
  std::string eqn = probl_params.get("Equations","Laplace");
  int dim = probl_params.get("Dimension",2);
  int nx=probl_params.get("nx",32);
  int ny=probl_params.get("ny",nx);
  int nz=probl_params.get("nz",(dim>2)?nx:1);

  galeriList.set("nx",nx);
  galeriList.set("ny",ny);
  galeriList.set("nz",nz);

    if (galeriLabel=="Laplace Neumann")
      {
      if (dim==2)
        {
        matrix = Teuchos::rcp(GaleriExt::Matrices::Cross2DN(&map,
                nx, ny, 4, -1, -1, -1, -1), true);
        }
      }
    else if (galeriLabel=="Darcy")
      {
      if (dim==2)
        {
        matrix = Teuchos::rcp(GaleriExt::Matrices::Darcy2D(&map,
                nx, ny, 1, -1), true);
        }
      else if (dim==3)
        {
        matrix = Teuchos::rcp(GaleriExt::Matrices::Darcy3D(&map,
                nx, ny, nz, 1, -1), true);
        }
      else
        {
        HYMLS::Tools::Error("not implemented!",__FILE__,__LINE__);
        }
      }
    else
      {
      std::string matrixType=galeriLabel;
      if (galeriLabel=="")
        {
        matrixType=eqn+Teuchos::toString(dim)+"D";
        }
      try {
        matrix= Teuchos::rcp(Galeri::CreateCrsMatrix(matrixType, &map, galeriList));
        } catch (Galeri::Exception G) {G.Print();}
      }
    if (probl_params.get("Equations","Laplace")=="Laplace")
      {
      matrix->Scale(-1.0); // we like our matrix negative definite
             // (just to conform with the diffusion operator in the NSE,
             // the solver works anyway, of course).
      }
  return matrix;
  }

int MakeSystemConsistent(const Epetra_CrsMatrix& A,
                               Epetra_MultiVector& x_ex,
                               Epetra_MultiVector& b,
                               Teuchos::ParameterList& driverList)
  {
  return 0;
  }
                                                                                             

}//MainUtils

}//HYMLS
