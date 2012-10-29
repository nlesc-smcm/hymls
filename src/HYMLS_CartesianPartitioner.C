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
        START_TIMER3(label_,"Constructor");
        }
        
        
  // destructor
  CartesianPartitioner::~CartesianPartitioner()
    {
    START_TIMER3(label_,"Destructor");
    }

  Teuchos::RCP<Epetra_Map> CartesianPartitioner::CreateSubdomainMap() const
    {
    return MatrixUtils::CreateMap(npx_,npy_,npz_,1,0,*comm_);
    }


  
}
