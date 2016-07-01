#include "HYMLS_CartesianPartitioner.H"
#include "HYMLS_Tools.H"
#include "Teuchos_Utils.hpp"
#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "HYMLS_MatrixUtils.H"

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#include "Epetra_MpiDistributor.h"
#endif


using Teuchos::toString;

namespace HYMLS {

  // constructor 
  CartesianPartitioner::CartesianPartitioner
        (Teuchos::RCP<const Epetra_Map> map, int nx, int ny, int nz, int dof, 
        GaleriExt::PERIO_Flag perio)
        : BaseCartesianPartitioner(map,nx,ny,nz,dof,perio)
        {
        label_="CartesianPartitioner";
        HYMLS_PROF3(label_,"Constructor");
        }
        
        
  // destructor
  CartesianPartitioner::~CartesianPartitioner()
    {
    HYMLS_PROF3(label_,"Destructor");
    }

  Teuchos::RCP<Epetra_Map> CartesianPartitioner::CreateSubdomainMap(
    int num_active) const
    {
    int NumMyElements = 0;
    int NumGlobalElements = npx_ * npy_ * npz_;
    int *MyGlobalElements = new int[NumGlobalElements];

    for (int i = 0; i < npx_; i++)
      for (int j = 0; j < npy_; j++)
        for (int k = 0; k < npz_; k++)
          {
          int pid = PID(sx_ * i, sy_ * j, sz_ * k);
          if (pid == comm_->MyPID())
            MyGlobalElements[NumMyElements++] = i + j * npx_ + k * npx_ * npy_;
          }

    Teuchos::RCP<Epetra_Map> result = Teuchos::rcp(new Epetra_Map(NumGlobalElements,
        NumMyElements, MyGlobalElements, 0, *comm_));

    delete [] MyGlobalElements;
    return result;
    }

}
