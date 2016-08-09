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

// helper function for comparing Epetra_IntVectors
int NormInfAminusB(const Epetra_IntVector& A, const Epetra_IntVector& B)
{
  if (A.Map().SameAs(B.Map())==false) return -1;
  int value=0;
  for (int i=0; i<A.MyLength(); i++)
  {
    value=std::max(value,std::abs(A[i]-B[i]));
  }
  int global_value;
  A.Map().Comm().MaxAll(&value,&global_value,1);
  return global_value;
}

TEUCHOS_UNIT_TEST(StandardNodeClassifier, BuildNodeTypeVectorForLaplace2D)
{
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  Teuchos::RCP<const Epetra_MpiComm> comm = Teuchos::rcp(&Comm, false);
  HYMLS::Tools::InitializeIO(comm);
  int dim = 2;
  int dof = 1;
  int nx=12;
  int ny=8;
  int nz=1;

  int n = nx * ny * nz * dof;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(n, 0, Comm));
  Teuchos::RCP<Epetra_LocalMap> localMap = Teuchos::rcp(new Epetra_LocalMap(n,0,Comm));

  Teuchos::RCP<HYMLS::CartesianPartitioner> part
        = Teuchos::rcp(new HYMLS::CartesianPartitioner(map, nx, ny, nz, dof));


  // this should give 3x2 subdomains of 4x4 each
  int sx=4;
  int sy=4;


  // This will cause an exception when compiled with TESTING if it fails
  part->Partition(6, true);

  // create a matrix to get the CrsGraph that the node classifier needs right now
  Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::null;
  
  Teuchos::ParameterList galeriList;
  galeriList.set("nx",nx);
  galeriList.set("ny",ny);
  galeriList.set("nz",nz);
  
  std::string matrixType="Laplace2D";
  try {
    matrix= Teuchos::rcp(Galeri::CreateCrsMatrix(matrixType, map.get(), galeriList));

    } catch (Galeri::Exception G) {G.Print();}

  // get the graph of the matrix, and import it into the localMap so that the classifier
  // can access all nodes it needs
  Teuchos::RCP<const Epetra_CrsGraph> graph = Teuchos::rcp(&matrix->Graph(),false);
  Teuchos::RCP<Epetra_CrsGraph> localGraph = Teuchos::rcp(new Epetra_CrsGraph(Copy,*localMap,5,false));

  Teuchos::RCP<Epetra_Import> import =
        Teuchos::rcp(new Epetra_Import(*localMap,*map));
  localGraph->Import(*graph,*import,Insert);
  
  localGraph->FillComplete();

  // there is a function 'flow(i,j)' which returns !=0 if there is an edge between
  // two nodes i,j and they are on different subdomains. In the current implementation
  // this requires that the CartesianPartitioner knows about the graph.
  part->SetGraph(localGraph);

  // until here, everything was test setup

  Teuchos::Array<std::string> variableType(1);
  variableType[0]="Laplace";
  
  Teuchos::Array<bool> retainIsolated(1);
  retainIsolated[0]=false;

  // create the classifier object
  // The constructor takes nx/ny/nz/dof so it can use geometric info for printing the node type vector,
  // not sure if it works if this info is not given to it
  int level=0; // HYMLS Preconditioner level, 0: original matrix
  Teuchos::RCP<HYMLS::BaseNodeClassifier> classi = Teuchos::rcp(new HYMLS::StandardNodeClassifier(localGraph, 
        part, variableType, retainIsolated,level,nx,ny,nz));

  // get the "node type vector"
  const Epetra_IntVector& nodeTypes = classi->Vector();
  classi->BuildNodeTypeVector();
  
  // construct the reference vector. Note that the partitioner defines its own map
  Epetra_IntVector refNodeTypes(part->Map());
  refNodeTypes.PutValue(0);
  for (int j=0; j<ny; j++)
  {
    for (int i=0; i<nx; i++) 
    {
      int gid = HYMLS::Tools::sub2ind(nx, ny, nz, dof, i, j, 0, 0);
      int lid = part->Map().LID(gid);
      if (lid>=0)
      {
        if ((i+1)%sx==0 && i+1<nx) refNodeTypes[lid]++;
        if ((j+1)%sy==0 && j+1<ny) refNodeTypes[lid]++;
      }
    }
  }
  
//  std::cout << "refNodeTypes: \n"<<refNodeTypes << std::endl;
//  std::cout << "nodeTypes: \n"<<nodeTypes << std::endl;
  
  // compare the two vectors
  TEST_EQUALITY(0,NormInfAminusB(refNodeTypes,nodeTypes));
}

