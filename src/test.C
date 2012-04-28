#include <mpi.h>
#include <iostream>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"

#include "Epetra_Import.h"
#include "Epetra_Export.h"

#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StrUtils.hpp"

#include "Galeri_CrsMatrices.h"
#include "Galeri_Maps.h"

#include "HYMLS_Tools.H"

int main(int argc, char** argv)
  {
  MPI_Init(&argc,&argv);
  Epetra_MpiComm comm(MPI_COMM_WORLD);

  int nx=8;
  int np = comm.NumProc();
  int pid= comm.MyPID();
  int base=0;
  Epetra_Map map(nx,nx/np,base,comm);
  
  int i0=pid*nx/np;
  int i1=(pid+1)*nx/np;
  
  // create overlap
  if (i0>0) i0--;
  if (i1<nx) i1++;
  
  int len = i1-i0;
  int *my_gids = new int[len];
  for (int i=0;i<len;i++) my_gids[i]=i0+i;  
  Epetra_Map map2(-1,len,my_gids,base,comm);
  
  Epetra_Export import(map2,map);
  
  Epetra_Vector v1(map);
  Epetra_Vector v2a(map2);
  Epetra_Vector v2b(map2);
  
  v2a.PutScalar(1.0);
  v2b.PutScalar(0.0);
  CHECK_ZERO(v1.Export(v2a,import,Add));
  CHECK_ZERO(v2b.Import(v1,import,Insert));
  
  for (int i=0;i<np;i++)
    {
    if (i==pid)
      {
      std::cout << "PID "<<pid<<": ";
      for (int j=0;j<v2b.MyLength();j++) 
          std::cout << v2b[j] <<" ";
      std::cout << std::endl;
      }
    comm.Barrier();
    comm.Barrier();
    comm.Barrier();
    comm.Barrier();
    comm.Barrier();
    }
  
  delete [] my_gids;
  MPI_Finalize();
  }
