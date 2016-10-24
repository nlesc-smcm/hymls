#include "HYMLS_StandardNodeClassifier.H"
#include "HYMLS_CartesianPartitioner.H"
#include "HYMLS_MatrixUtils.H"
#include "HYMLS_Tools.H"
#include "Galeri_CrsMatrices.h"
#include "GaleriExt_CrsMatrices.h"
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
  : nx_(nx),ny_(ny),nz_(nz),dof_(dof),sx_(sx),sy_(sy),sz_(sz),nparts_(nparts)
  {
    comm_ = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
    HYMLS::Tools::InitializeIO(comm_);
    dim_=(nz_>1)?3: 2;
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
      matrix= Teuchos::rcp(GaleriExt::CreateCrsMatrix(matrixType,map_.get(),galeriList));
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
          int vmax=1;
          if (stokes_) vmax=dof_-1;
          for (int var=0; var<vmax;var++)
          {
            int gid = HYMLS::Tools::sub2ind(nx_, ny_, nz_, dof_, i, j, k, var);
            int lid = cartPart_->Map().LID(gid);
            if (lid>=0)
            {
              if ((i+1)%sx_==0 && i+1<nx_) (*refNodeTypes)[lid]++;
              if ((j+1)%sy_==0 && j+1<ny_) (*refNodeTypes)[lid]++;
              if ((k+1)%sz_==0 && k+1<nz_) (*refNodeTypes)[lid]++;
            }
          }
        }
      }
    }
    if (stokes_)
    {
      int var=dof_-1;
      // P-variable: retain one P node per subdomain interior
      for (int k=0;k<nz_;k+=sz_)
        for (int j=0;j<ny_;j+=sy_)
          for (int i=0;i<nx_;i+=sx_)
          {
            int gid = HYMLS::Tools::sub2ind(nx_, ny_, nz_, dof_, i, j, k, var);
            int lid = cartPart_->Map().LID(gid);
            if (lid>=0) (*refNodeTypes)[lid]=5;
          }
          
      // mark "full conservation cells" with P=5, V=4
      if (dim_==2)
      {
        for (int j=0;j<ny_;j+=sy_)
        {
          for (int i=0;i<nx_;i+=sx_)
          {
            if ((i+sx_)<nx_ && (j+sy_)<ny_)
            {
              // mark the P-node with 5
              int gid = HYMLS::Tools::sub2ind(nx_, ny_, nz_, dof_, i+sx_-1, j+sy_-1, 0, var);
              int lid = cartPart_->Map().LID(gid);
              if (lid>=0) (*refNodeTypes)[lid]=5;
              
              // mark the velocities surrounding this P as 4
              gid = HYMLS::Tools::sub2ind(nx_, ny_, nz_, dof_, i+sx_-1, j+sy_-1, 0, 0);
              lid = cartPart_->Map().LID(gid);
              if (lid>=0) (*refNodeTypes)[lid]=4;
              gid = HYMLS::Tools::sub2ind(nx_, ny_, nz_, dof_, i+sx_-1, j+sy_-1, 0, 1);
              lid = cartPart_->Map().LID(gid);
              if (lid>=0) (*refNodeTypes)[lid]=4;
              gid = HYMLS::Tools::sub2ind(nx_, ny_, nz_, dof_, i+sx_-2, j+sy_-1, 0, 0);
              lid = cartPart_->Map().LID(gid);
              if (lid>=0) (*refNodeTypes)[lid]=4;
              gid = HYMLS::Tools::sub2ind(nx_, ny_, nz_, dof_, i+sx_-1, j+sy_-2, 0, 1);
              lid = cartPart_->Map().LID(gid);
              if (lid>=0) (*refNodeTypes)[lid]=4;
            }
          }
        }
      }
      else
      {
        for (int k=0;k<nz_;k++)
        {
          for (int j=0;j<ny_;j++)
          {
            for (int i=0;i<nx_;i++)
            {
              int b1=(i+1<nx_ && (i+1)%sx_==0)?1:0;
              int b2=(j+1<ny_ && (j+1)%sy_==0)?1:0;
              int b3=(k+1<nz_ && (k+1)%sz_==0)?1:0;
              int bsum=b1+b2+b3;
              int uval=bsum+2;
              if (bsum>=2)
              {
                int gid_p = HYMLS::Tools::sub2ind(nx_, ny_, nz_, dof_, i, j, k, var);
                int lid_p = cartPart_->Map().LID(gid_p);
                if (lid_p>=0) (*refNodeTypes)[lid_p]=5;

                for (int xx=-1;xx<=0;xx++)
                {
                  int gid=-1,lid=-1;
                  try {
                  gid = HYMLS::Tools::sub2ind(nx_, ny_, nz_, dof_, i+xx, j, k, 0);
                  } catch(...) {gid=-1;}
                  lid = cartPart_->Map().LID(gid);
                  if (lid>=0) (*refNodeTypes)[lid]=std::max(uval,(*refNodeTypes)[lid]);
                  try {
                  gid = HYMLS::Tools::sub2ind(nx_, ny_, nz_, dof_, i, j+xx, k, 1);
                  } catch(...) {gid=-1;}
                  lid = cartPart_->Map().LID(gid);
                  if (lid>=0) (*refNodeTypes)[lid]=std::max(uval,(*refNodeTypes)[lid]);
                  try {
                  gid = HYMLS::Tools::sub2ind(nx_, ny_, nz_, dof_, i, j, k+xx, 2);
                  } catch(...) {gid=-1;}
                  lid = cartPart_->Map().LID(gid);
                  if (lid>=0) (*refNodeTypes)[lid]=std::max(uval,(*refNodeTypes)[lid]);
                }
              }
            }
          }
        }
      }
      // in the Stokes matrices there are some singletons due to Dirichlet BC e.g. for u at i=nx_-1,
      // these do not get a node type from our StandardNodeClassifier because they are not coupled 
      // to anyone, so they are eliminated (treated as interior, 0)
      var=0;
      int i=nx_-1;
      for (int k=0; k<nz_; k++)
      {
        for (int j=0; j<ny_; j++)
        {
          int gid = HYMLS::Tools::sub2ind(nx_, ny_, nz_, dof_, i, j, k, var);
          int lid = cartPart_->Map().LID(gid);
          if (lid>=0) (*refNodeTypes)[lid]=0;
        }
      }
      var=1;
      int j=ny_-1;
      for (int k=0; k<nz_; k++)
      {
        for (int i=0; i<nx_; i++)
        {
          int gid = HYMLS::Tools::sub2ind(nx_, ny_, nz_, dof_, i, j, k, var);
          int lid = cartPart_->Map().LID(gid);
          if (lid>=0) (*refNodeTypes)[lid]=0;
        }
      }
      if (dim_>2) {
      var=2;
      int k=nz_-1;
      for (int j=0; j<ny_; j++)
      {
        for (int i=0; i<nx_; i++)
        {
          int gid = HYMLS::Tools::sub2ind(nx_, ny_, nz_, dof_, i, j, k, var);
          int lid = cartPart_->Map().LID(gid);
          if (lid>=0) (*refNodeTypes)[lid]=0;
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

  void PrintGrid(const Epetra_IntVector& v, std::ostream& os)
  {
    Teuchos::RCP<Epetra_IntVector> out = HYMLS::MatrixUtils::Gather(v, 0);
    if (comm_->MyPID() != 0)
      return;

    for (int var=0; var<dof_;var++)
    {
      for (int k=0; k<nz_;k++)
      {
        os << "var="<<var<<", k="<<k<<std::endl;

        for (int j=0; j<ny_; j++)
        {
          for (int i=0; i<nx_; i++)
          {
            int gid = (((k*ny_)+j)*nx_+i)*dof_+var;
            int lid = out->Map().LID(gid);
            if (lid >= 0)
            {
              if ((*out)[lid])
              {
                os << " " << std::setw(5) << (*out)[lid];
              }
              else
              {
                os << "     ~";
              }
            }
          }
          os << std::endl;
        }
        os << std::endl;
      }
      os << "#######################################################"<<std::endl;
    }
  }

  int RunTest(bool output_on_failure=true)
  {
    Teuchos::RCP<const Epetra_IntVector> refNodeTypes=this->ConstructExpectedNodeTypeVector();
    Teuchos::RCP<const Epetra_IntVector> myNodeTypes=this->ConstructNodeTypeVector();
  
    //  std::cout << "refNodeTypes: \n"<<refNodeTypes << std::endl;
    //  std::cout << "nodeTypes: \n"<<nodeTypes << std::endl;
    int diff = NormInfAminusB(*refNodeTypes,*myNodeTypes);
    if (output_on_failure && diff!=0)
    {
      if (diff == -1)
      {
        if (comm_->MyPID() == 0)
          std::cout << "Maps are not the same:\n";
        if (comm_->MyPID() == 0)
          std::cout << "Reference node type:\n";
        std::cout << refNodeTypes->Map();
        if (comm_->MyPID() == 0)
          std::cout << "Constructed node type:\n";
        std::cout << myNodeTypes->Map();
      }
      else
      {
        Epetra_IntVector diff_vec = *refNodeTypes;
        for (int i=0;i<myNodeTypes->MyLength();i++)
          diff_vec[i]-=(*myNodeTypes)[i];

        if (comm_->MyPID() == 0)
          std::cout << "Reference node type vector:\n";
        this->PrintGrid(*refNodeTypes,std::cout);
        if (comm_->MyPID() == 0)
          std::cout << "Constructed node type vector:\n";
        this->PrintGrid(*myNodeTypes,std::cout);
        if (comm_->MyPID() == 0)
          std::cout << "Diff: (ref <-> computed)\n";
        this->PrintGrid(diff_vec,std::cout);
      }
    }
    return diff;
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
TEUCHOS_UNIT_TEST(StandardNodeClassifier, Laplace2D)
{
  // Laplace 2D, 3x2 subdomains of 4x4 each
  int dof = 1;
  int nx=12, ny=8, nz=1;
  int sx=4, sy=4, sz=1;
  int nparts=6;

  BuildNodeTypeVectorTest test(nx,ny,nz,dof,nparts,sx,sy,sz);

  int ret = test.RunTest();
  TEST_EQUALITY(0, ret);
}

TEUCHOS_UNIT_TEST(StandardNodeClassifier, Laplace3D)
{
  // Laplace 3D, 3x4x2 subdomains of 3x3x3 each
  int dof = 1;
  int nx=9, ny=12, nz=6;
  int sx=3, sy=3, sz=3;
  int nparts=24;

  BuildNodeTypeVectorTest test(nx,ny,nz,dof,nparts,sx,sy,sz);

  int ret = test.RunTest();
  TEST_EQUALITY(0, ret);
}

TEUCHOS_UNIT_TEST(StandardNodeClassifier, Stokes2D)
{
  // Laplace 2D, 3x2 subdomains of 4x4 each
  int dof = 3;
  int nx=12, ny=8, nz=1;
  int sx=4, sy=4, sz=1;
  int nparts=6;

  BuildNodeTypeVectorTest test(nx,ny,nz,dof,nparts,sx,sy,sz);

  int ret = test.RunTest();
  TEST_EQUALITY(0, ret);
}

TEUCHOS_UNIT_TEST(StandardNodeClassifier, Stokes3D)
{
  // Laplace 3D, 3x4x2 subdomains of 3x3x3 each
  int dof = 4;
  int nx=9, ny=12, nz=6;
  int sx=3, sy=3, sz=3;
  int nparts=24;

  BuildNodeTypeVectorTest test(nx,ny,nz,dof,nparts,sx,sy,sz);

  int ret = test.RunTest();
  TEST_EQUALITY(0, ret);
}

