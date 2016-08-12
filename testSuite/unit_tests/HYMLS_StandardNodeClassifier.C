#include "HYMLS_StandardNodeClassifier.H"
#include "HYMLS_CartesianPartitioner.H"
#include "HYMLS_Tools.H"
#include "Galeri_CrsMatrices.h"
#include <Epetra_IntVector.h>
#include <Epetra_LocalMap.h>
#include <Epetra_Import.h>

#include <Teuchos_RCP.hpp>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"

#include "HYMLS_UnitTests.H"

using namespace HYMLS::UnitTests;

class BuildNodeTypeVectorTest
{
  public:
  
  BuildNodeTypeVectorTest(int nx,int ny, int nz, int dof, int nparts,int sx, int sy, int sz)
  : nx_(nx),ny_(ny),nz_(nz),dof_(dof),nparts_(nparts),sx_(sx),sy_(sy),sz_(sz)
  {
    comm_ = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
    HYMLS::Tools::InitializeIO(comm_);
    dim_=(nz_>0)?3: 2;
    stokes_=(dof_==dim_+1);
    int n = nx * ny * nz * dof;
    map_ = Teuchos::rcp(new Epetra_Map(n, 0, *comm_));
    localMap_ = Teuchos::rcp(new Epetra_LocalMap(n,0,*comm_));
    cartPart_ = Teuchos::rcp(new HYMLS::CartesianPartitioner(map_, nx_, ny_, nz_, dof_));
    // This will cause an exception when compiled with TESTING if it fails
    cartPart_->Partition(nparts_, true);

  // create a matrix to get the CrsGraph that the node classifier needs right now
  Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::null;
  
  Teuchos::ParameterList galeriList;
  galeriList.set("nx",nx);
  galeriList.set("ny",ny);
  galeriList.set("nz",nz);
  
  std::string matrixType;
  if (!stokes_)
  {
    matrixType=(dim_==2)? "Laplace2D": "Laplace3D";
  }
  else
  {
    matrixType=(dim_==2)? "Stokes2D": "Stokes3D";
  }

  try {
    if (!stokes_)
    {
      matrix= Teuchos::rcp(Galeri::CreateCrsMatrix(matrixType, map_.get(), galeriList));
    }
    else
    {
      throw "test not implemented";
    }

    } catch (Galeri::Exception G) {G.Print();}

  // get the graph of the matrix, and import it into the localMap so that the classifier
  // can access all nodes it needs
  Teuchos::RCP<const Epetra_CrsGraph> graph = Teuchos::rcp(&matrix->Graph(),false);
  localGraph_ = Teuchos::rcp(new Epetra_CrsGraph(Copy,*localMap_,5,false));

  Teuchos::RCP<Epetra_Import> import =
        Teuchos::rcp(new Epetra_Import(*localMap_,*map_));
  localGraph_->Import(*graph,*import,Insert);
  
    localGraph_->FillComplete();

    // there is a function 'flow(i,j)' which returns !=0 if there is an edge between
    // two nodes i,j and they are on different subdomains. In the current implementation
    // this requires that the CartesianPartitioner knows about the graph.
    cartPart_->SetGraph(localGraph_);

  }
  
  ~BuildNodeTypeVectorTest()
  {
  }
  
  // construct the reference vector. Note that the partitioner defines its own map
  Teuchos::RCP<Epetra_IntVector> ConstructExpectedNodeTypeVector()
  {
    Teuchos::RCP<Epetra_IntVector> refNodeTypes = Teuchos::rcp(new Epetra_IntVector(cartPart_->Map()));

  refNodeTypes->PutValue(0);
  for (int k=0; k<nz_; k++)
  {
    for (int j=0; j<ny_; j++)
    {
      for (int i=0; i<nx_; i++) 
      {
        int gid = HYMLS::Tools::sub2ind(nx_, ny_, nz_, dof_, i, j, 0, 0);
        int lid = cartPart_->Map().LID(gid);
        if (lid>=0)
        {
          if ((i+1)%sx_==0 && i+1<nx_) (*refNodeTypes)[lid]++;
          if ((j+1)%sy_==0 && j+1<ny_) (*refNodeTypes)[lid]++;
          if ((k+1)%sz_==0 && j+1<nz_) (*refNodeTypes)[lid]++;
        }
      }
    }
  }
  
    return refNodeTypes;
  }

  Teuchos::RCP<const Epetra_IntVector> ConstructNodeTypeVector()  
  {
  Teuchos::Array<std::string> variableType(dof_);
  Teuchos::Array<bool> retainIsolated(dof_);
  
  for (int i=0; i<dof_-1; i++) 
  {
    variableType[i]="Laplace";
    retainIsolated[i]=false;
  }
  if (stokes_)
  { 
    variableType[dof_-1]="Retain 1";
    retainIsolated[dof_-1]=true;
  }
  
  // create the classifier object
  // The constructor takes nx/ny/nz/dof so it can use geometric info for printing the node type vector,
  // not sure if it works if this info is not given to it
  int level=0; // HYMLS Preconditioner level, 0: original matrix
  Teuchos::RCP<HYMLS::BaseNodeClassifier> classi = Teuchos::rcp(new HYMLS::StandardNodeClassifier(localGraph_, 
        cartPart_, variableType, retainIsolated,level,nx_,ny_,nz_));

  // get the "node type vector"
  classi->BuildNodeTypeVector();
  return classi->GetVector();
  }

  double RunTest()
  {
    Teuchos::RCP<const Epetra_IntVector> refNodeTypes=this->ConstructExpectedNodeTypeVector();
    Teuchos::RCP<const Epetra_IntVector> myNodeTypes=this->ConstructNodeTypeVector();
  
    //  std::cout << "refNodeTypes: \n"<<refNodeTypes << std::endl;
    //  std::cout << "nodeTypes: \n"<<nodeTypes << std::endl;
    return NormInfAminusB(*refNodeTypes,*myNodeTypes);
  }
  
  Teuchos::RCP<const Epetra_MpiComm> comm_; 
  int nx_,ny_,nz_,dim_,dof_,stokes_;
  int sx_,sy_,sz_,nparts_;
  Teuchos::RCP<Epetra_Map> map_;
  Teuchos::RCP<Epetra_LocalMap> localMap_;
  Teuchos::RCP<HYMLS::CartesianPartitioner> cartPart_;
  Teuchos::RCP<Epetra_CrsGraph> localGraph_;
};



// actual tests start here, they just construct the object above with different input args
// and compare the reference node type vector with the actual one.
TEUCHOS_UNIT_TEST(StandardNodeClassifier, BuildNodeTypeVectorForLaplace2D)
{

  // this should give 3x2 subdomains of 4x4 each
  int dim = 2;
  int dof = 1;
  int nx=12;
  int ny=8;
  int nz=1;
  int sx=4;
  int sy=4;
  int sz=1;
  int nparts=6;

  BuildNodeTypeVectorTest test(nx,ny,nz,dof,nparts,sx,sy,sz);
  
  // compare the two vectors
  TEST_EQUALITY(0,test.RunTest());
}

