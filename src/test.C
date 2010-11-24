#include <mpi.h>
#include <iostream>

#include "HYMLS_OrthogonalTransform.H"
#include "HYMLS_Householder.H"
#include "Epetra_SerialDenseVector.h"
#include "Epetra_SerialDenseMatrix.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"

#include "Epetra_CrsMatrix.h"
#include "Epetra_SerialComm.h"
#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"

#include "HYMLS_Tools.H"

int main(int argc, char** argv)
  {
  MPI_Init(&argc,&argv);

  Epetra_MpiComm comm(MPI_COMM_WORLD);
  Epetra_SerialComm serialComm;
  
  HYMLS::Tools::InitializeIO(Teuchos::rcp(&comm,false));
  std::cout << std::setw(15)<<std::setprecision(15);
//{
#if 0

  if (comm.NumProc()!=4)
    {
    HYMLS::Tools::Error("intended for four procs!",__FILE__,__LINE__);
    }

int nx=16;
int ny=16;

int nx_loc=8;
int ny_loc=8;

int xoff, yoff;

if (comm.MyPID()==1)
  {
  xoff=8;
  }

if (comm.MyPID()==2)
  {
  yoff=8;
  }

if (comm.MyPID()==3)
  {
  xoff=8;
  yoff=8;
  }

int NumMyElements = nx_loc*ny_loc;
int *MyElements = new int[NumMyElements];

int pos=0;
for (int j=0;j<ny_loc;j++)
  for (int i=0;i<nx_loc;i++)
    {
    MyElements[pos++]=(yoff+j)*nx+xoff+i;
    }
 
Teuchos::RCP<Epetra_Map> baseMap = Teuchos::rcp(new Epetra_Map
        (-1, NumMyElements, MyElements, 0, comm));

delete [] MyElements;

DEBVAR(*baseMap);

int numOverlapElements=0;
// create an overlapping map
if (comm.MyPID()==0)
  {
  numOverlapElements=17;
  }
if (comm.MyPID()==1)
  {
  numOverlapElements=8;
  }
if (comm.MyPID()==2)
  {
  numOverlapElements=8;
  }

NumMyElements=NumMyElements+numOverlapElements;
MyElements = new int[NumMyElements];

delete [] MyElements;

#endif
//}
// local maps {
#if 0  
  
  if (comm.NumProc()!=2)
    {
    HYMLS::Tools::Error("intended for two procs!",__FILE__,__LINE__);
    }

  int nrows=10;
  
  int NumMyElements;
  int *MyElements;
  int offset;
  
  if (comm.MyPID()==0)
    {
    NumMyElements=4;
    offset=0;
    }
  else
    {
    NumMyElements=6;
    offset=4;
    }
  
  MyElements = new int[NumMyElements];
  for (int i=0;i<NumMyElements;i++) 
    {
    MyElements[i]=offset+i;
    }

  Epetra_Map map(-1, NumMyElements, MyElements, 0, comm);
  int groupSize=2;
  int numLocalGroups=NumMyElements/groupSize;
  Teuchos::Array<Teuchos::RCP<Epetra_Map> > localMaps;
  localMaps.resize(numLocalGroups);
  
  DEBVAR(map);
  
  for (int grp=0;grp<numLocalGroups;grp++)
    {
    int elements[2];
    elements[0]=offset+grp*groupSize;
    elements[1]=offset+grp*groupSize+1;
    localMaps[grp]=Teuchos::rcp(new Epetra_Map(2,2,elements,0,serialComm));
    DEBVAR(*localMaps[grp]);
    }

#endif
//}
// Householder transform {
#if 1
  Teuchos::RCP<HYMLS::OrthogonalTransform> T;
  T=Teuchos::rcp(new HYMLS::Householder());
  Epetra_SerialDenseVector v,Tv;
  std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
  /*
  for (int n=1;n<=8;n++)
    {
    std::cout << "n="<<n<<std::endl;
    v.Size(n);
    Tv.Size(n);
    for (int i=0;i<n;i++) v[i]=1.0;
    std::cout << v;
    T->Apply(v,Tv);
    std::cout << Tv;
    T->ApplyInverse(Tv,v);
    std::cout << v;
    std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    }
    */
  int m=5, n=8;
  v.Size(n);
  Tv.Size(n);
  v.Random();
  std::cout << v;
  T->Apply(v,Tv);
  T->ApplyInverse(Tv,v);
  std::cout << Tv;
  std::cout << v;

  Epetra_SerialDenseMatrix A(m,n);
  Epetra_SerialDenseMatrix TAT(m,n);
  A.Random();
  
  std::cout << "n="<<n<<", m="<<m<<std::endl;
  std::cout << A;
  T->Apply(A,TAT);
  std::cout << TAT;
  T->ApplyInverse(TAT,A);
  std::cout << A;
#endif
//}

  
  MPI_Finalize();
  }
