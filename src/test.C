#include <mpi.h>
#include <iostream>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StrUtils.hpp"

#include "Galeri_CrsMatrices.h"
#include "Galeri_Maps.h"

int main(int argc, char** argv)
  {
  MPI_Init(&argc,&argv);
  Epetra_MpiComm comm(MPI_COMM_WORLD);

  char* procname;
  int procname_len=0;
  MPI_Get_processor_name(procname,&procname_len);
  std::string proc(procname);
  for (int i=0; i<proc.length();i++)
    {
    if (proc[i]<'0' || proc[i]>'9') proc[i]='0';
    }

  int rank = comm.MyPID();
    
  std::cout << rank << ": '"<<proc<<"' => "<<Teuchos::StrUtils::atoi(proc)<<std::endl<<std::flush;  

  // create a subcommunicator
  int color=MPI_UNDEFINED;
  if (rank%2==0) color=1;
  MPI_Comm MPI_SubComm_;
  MPI_Comm_split(comm.Comm(),color,rank,&MPI_SubComm_);

  bool active = (MPI_SubComm_!=MPI_COMM_NULL);
  
  Teuchos::RCP<Epetra_MpiComm> subComm = Teuchos::null;
  if (active) subComm = Teuchos::rcp(new Epetra_MpiComm(MPI_SubComm_));
  
  int n=64;
  Epetra_Map map(n,1,comm);
  Teuchos::RCP<Epetra_Map> restrictedMap = Teuchos::null;
  
  if (active)
    {
    restrictedMap = Teuchos::rcp(new Epetra_Map(n,1,*subComm));
    }
  
  Epetra_Vector vec(map);
  for (int i=0;i<vec.MyLength();i++) vec[i] = (double)map.GID(i);
  
  Teuchos::RCP<Epetra_Import> import=Teuchos::null;
  Teuchos::RCP<Epetra_Vector> restrictedVector;
  
  if (active) 
    {
    import=Teuchos::rcp(new Epetra_Import(*map,*restrictedMap));
    restrictedVec = Teuchos::rcp(new Epetra_Vector(*restrictedMap));
    }


  MPI_Finalize();
  }
