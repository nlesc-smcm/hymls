
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
#include "HYMLS_CartesianPartitioner.H"
#include "HYMLS_SkewCartesianPartitioner.H"

#include "GaleriExt_Cross2DN.h"
#include "Galeri_CrsMatrices.h"

#include "GaleriExt_Darcy2D.h"
#include "GaleriExt_Darcy3D.h"
#include "GaleriExt_Stokes2D.h"
#include "GaleriExt_Stokes3D.h"

namespace HYMLS {

namespace MainUtils {

Teuchos::RCP<Epetra_CrsMatrix> read_matrix(std::string datadir,
  std::string file_format, Teuchos::RCP<Epetra_Map> map, std::string name)
  {
  if (map==Teuchos::null)
    {
    HYMLS::Tools::Error("map must have been allocated before this function",
        __FILE__,__LINE__);
    }
  
  std::string suffix;
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

  std::string filename = datadir+"/"+name+suffix;

  HYMLS::Tools::Out("... read matrix from file '"+filename+"'");
  HYMLS::Tools::Out("    file format: "+file_format);

  Teuchos::RCP<Epetra_CrsMatrix> K=Teuchos::null;

  if (file_format=="MatrixMarket" || file_format=="MatrixMarket (2)")
    {
    Epetra_CrsMatrix* Kptr;
#ifdef HYMLS_LONG_LONG
    CHECK_ZERO(EpetraExt::MatrixMarketFileToCrsMatrix64(filename.c_str(), *map, Kptr));
#else
    CHECK_ZERO(EpetraExt::MatrixMarketFileToCrsMatrix(filename.c_str(), *map, Kptr));
#endif
    K=Teuchos::rcp(Kptr, true);
    }
  else
    {
    HYMLS::Tools::Error("File format '"+file_format+"' not supported",__FILE__,__LINE__);
    }
  return K;
  }

Teuchos::RCP<Epetra_Vector> read_vector(std::string name,std::string datadir,
  std::string file_format,Teuchos::RCP<Epetra_Map> map)
  {
  if (map==Teuchos::null)
    {
    HYMLS::Tools::Error("map must have been allocated before this function",
        __FILE__,__LINE__);
    }
  
  std::string suffix;
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

  std::string filename = datadir+"/"+name+suffix;
  
  HYMLS::Tools::Out("... read vector from file '"+filename+"'");
  HYMLS::Tools::Out("    file format: "+file_format);
  
  Teuchos::RCP<Epetra_Vector> v;

  if (file_format=="MatrixMarket" || file_format=="MatrixMarket (2)")
    {
    // the EpetraExt function only works for a linear map in parallel,
    // so we need to reindex ourselves:
    Epetra_Map linearMap((hymls_gidx)map->NumGlobalElements64(),
                         map->NumMyElements(),
                         (hymls_gidx)map->IndexBase64(),
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
  Teuchos::ParameterList& probl_params,
  Teuchos::ParameterList& prec_params)
  {
  int pvar = -1;
  int dim = probl_params.get("Dimension",2);
  int nx=probl_params.get("nx",32);
  int ny=probl_params.get("ny",nx);
  int nz=probl_params.get("nz",(dim>2)?nx:1);
  std::string eqn=probl_params.get("Equations","not-set");

  int sx,sy,sz;
  if (prec_params.isParameter("Separator Length (x)"))
    {
    sx = prec_params.get("Separator Length (x)", -1);
    sy = prec_params.get("Separator Length (y)", sx);
    sz = prec_params.get("Separator Length (z)", dim < 3 ? 1 : sx);
    }
  else
    {
    sx = prec_params.get("Separator Length", -1);
    sy = sx;
    sz = dim < 3 ? 1 : sx;
    }

  Teuchos::RCP<Epetra_Map> map=Teuchos::null;

  int dof=probl_params.get("Degrees of Freedom",1);
  bool is_complex=probl_params.get("Complex Arithmetic",false);

  if (eqn == "Stokes-C")
    {
    dof = dim + 1;
    pvar = dim;
    }
  else if (eqn!="Laplace" && eqn=="Laplace Neumann")
    {
    HYMLS::Tools::Warning("cannot determine problem type from 'Equation' parameter "+eqn+"\n"
      "Assuming dof="+Teuchos::toString(dof)+" cartesian grid in "+Teuchos::toString(dim)+"D",
      __FILE__, __LINE__);
    }
  if (is_complex) dof*=2;

  hymls_gidx n = nx*ny*nz*dof;
  HYMLS::Tools::out()<<"Create map with dof="<<dof<<" in "<<dim<<"D"<<std::endl;
  map = Teuchos::rcp(new Epetra_Map(n, 0, comm));

  std::string partMethod = prec_params.get("Partitioner", "Cartesian");
  Teuchos::RCP<HYMLS::BasePartitioner> part = Teuchos::null;
  if (partMethod == "Cartesian")
    {
    Teuchos::RCP<HYMLS::CartesianPartitioner> cartPart =
      Teuchos::rcp(new HYMLS::CartesianPartitioner(map, nx, ny, nz, dof, pvar));
    cartPart->Partition(sx, sy, sz, true);
    part = cartPart;
    }
  else if (partMethod == "Skew Cartesian")
    {
    Teuchos::RCP<HYMLS::SkewCartesianPartitioner> cartPart =
      Teuchos::rcp(new HYMLS::SkewCartesianPartitioner(map, nx, ny, nz, dof, pvar));
    cartPart->Partition(sx, sy, sz, true);
    part = cartPart;
    }
  else
    HYMLS::Tools::Error("Partitioner not recognised", __FILE__, __LINE__);

  return Teuchos::rcp(new Epetra_Map(part->Map()));
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

  bool xperio = probl_params.get("x-periodic", false);
  bool yperio = probl_params.get("y-periodic", false);
  bool zperio = probl_params.get("z-periodic", false);

  GaleriExt::PERIO_Flag perio = GaleriExt::NO_PERIO;

  if (xperio) perio = (GaleriExt::PERIO_Flag)(perio | GaleriExt::X_PERIO);
  if (yperio) perio = (GaleriExt::PERIO_Flag)(perio | GaleriExt::Y_PERIO);
  if (zperio) perio = (GaleriExt::PERIO_Flag)(perio | GaleriExt::Z_PERIO);

  if (galeriLabel == "Laplace Neumann")
    {
    if (dim==2)
      {
      matrix = Teuchos::rcp(GaleriExt::Matrices::Cross2DN(&map,
          nx, ny, 4, -1, -1, -1, -1), true);
      }
    }
  else if (galeriLabel == "Darcy")
    {
    if (dim == 2)
      {
      matrix = Teuchos::rcp(GaleriExt::Matrices::Darcy2D(&map,
          nx, ny, 1, -1, perio), true);
      }
    else if (dim == 3)
      {
      matrix = Teuchos::rcp(GaleriExt::Matrices::Darcy3D(&map,
          nx, ny, nz, 1, -1, perio), true);
      }
    }
  else if (galeriLabel == "Stokes-C")
    {
    if (dim == 2)
      {
      if (nx!=ny) HYMLS::Tools::Warning("GaleriExt::Stokes2D only gives correct matrix entries if nx=ny, but the graph is corret\n",__FILE__,__LINE__);
      matrix = Teuchos::rcp(GaleriExt::Matrices::Stokes2D(&map,
          nx, ny, nx*nx, 1, perio), true);
      }
    else if (dim == 3)
      {
      if (nx!=ny||nx!=nz) HYMLS::Tools::Warning("GaleriExt::Stokes3D only gives correct matrix entries if nx=ny, but the graph is corret\n",__FILE__,__LINE__);
      matrix = Teuchos::rcp(GaleriExt::Matrices::Stokes3D(&map,
          nx, ny, nz, nx*nx, 1, perio), true);
      }
    else
      {
      HYMLS::Tools::Error("not implemented!",__FILE__,__LINE__);
      }
    }
  else
    {
    std::string matrixType = galeriLabel;
    if (galeriLabel == "")
      {
      matrixType = eqn + Teuchos::toString(dim)+"D";
      }
    try {
      matrix = Teuchos::rcp(Galeri::CreateCrsMatrix(matrixType, &map, galeriList));
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

  // try to construct the nullspace for the operator, right now we only implement
  Teuchos::RCP<Epetra_MultiVector> create_nullspace(const Epetra_CrsMatrix& A,
                                                    const std::string& nullSpaceType,
                                                          Teuchos::ParameterList& probl_params)
  {
  int dim  = probl_params.get("Dimension", -1);
  int dof = probl_params.get("Degrees of Freedom", -1);

  Teuchos::RCP<Epetra_MultiVector> nullSpace = Teuchos::null;
  if (nullSpaceType == "Constant")
    {
    if (dof == -1)
      {
      Tools::Error("'Degrees of Freedom' not set in 'Problem' sublist",
        __FILE__, __LINE__);
      }

    nullSpace = Teuchos::rcp(new Epetra_MultiVector(A.OperatorDomainMap(), dof));
    CHECK_ZERO(nullSpace->PutScalar(0.0));

    for (int lid = 0; lid < nullSpace->MyLength(); lid++)
      {
      hymls_gidx gid = nullSpace->Map().GID64(lid);
      (*nullSpace)[gid % dof][lid] = 1.0 / sqrt(nullSpace->GlobalLength64() / dof);
      }
    }
  else if (nullSpaceType == "Constant P")
    {
    int pvar = probl_params.get("Pressure Variable", dim);
    // NOTE: we assume u/v/w/p[/T] ordering here, it works for 2D and 3D as long
    // as var[dim]=P
    nullSpace = Teuchos::rcp(new Epetra_Vector(A.OperatorDomainMap()));
    // TODO: this is all a bit ad-hoc
    if (pvar == -1 || dof == -1)
      {
      Tools::Error("'Dimension' or 'Degrees of Freedom' not set in 'Problem' sublist",
        __FILE__, __LINE__);
      }
    CHECK_ZERO(nullSpace->PutScalar(0.0));
    for (int i = dof - 1; i < nullSpace->MyLength(); i+= dof)
      {
      (*nullSpace)[0][i] = 1.0;
      }
    }
  else if (nullSpaceType == "Checkerboard")
    {
    nullSpace = Teuchos::rcp(new Epetra_MultiVector(A.OperatorDomainMap(), 3));
    int nx = probl_params.get("nx", 1);
    int ny = probl_params.get("ny", nx);
    int nz = probl_params.get("nz", dim > 2 ? nx : 1);
    for (int lid = 0; lid < nullSpace->MyLength(); lid++)
      {
      hymls_gidx gid=nullSpace->Map().GID64(lid);
      int i,j,k,v;
      HYMLS::Tools::ind2sub(nx, ny, nz, dof, gid, i, j, k, v);
      double val1 =  (double)(MOD(i+j+k,2));
      double val2 =  1.0-val1;
      (*nullSpace)[0][lid]=val1;
      (*nullSpace)[1][lid]=val2;
      (*nullSpace)[2][lid]=1.0;
      }
    }    
  else if (nullSpaceType != "None")
    {
    Tools::Error("'Null Space'='"+nullSpaceType+"' not implemented",
      __FILE__, __LINE__);
    }

  // normalize each column
  int k = nullSpace->NumVectors();
  double *nrm2 = new double[k];
  CHECK_ZERO(nullSpace->Norm2(nrm2));
  for (int i=0;i<k;i++)
    {
    CHECK_ZERO((*nullSpace)(i)->Scale(1.0/nrm2[i]));
    }
  delete [] nrm2;
  return nullSpace;
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
