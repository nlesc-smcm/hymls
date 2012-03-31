#include <mpi.h>
#include <iostream>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Galeri_CrsMatrices.h"
#include "Galeri_Maps.h"

int main(int argc, char** argv)
  {
  MPI_Init(&argc,&argv);
  Epetra_MpiComm comm(MPI_COMM_WORLD);

  Teuchos::ParameterList galeriList;
  
  MPI_Finalize();
  }
