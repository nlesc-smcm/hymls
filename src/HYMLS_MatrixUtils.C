/**********************************************************************
 * Copyright by Jonas Thies, Univ. of Groningen 2006/7/8.             *
 * Permission to use, copy, modify, redistribute is granted           *
 * as long as this header remains intact.                             *
 * contact: jonas@math.rug.nl                                         *
 **********************************************************************/
#include "Teuchos_RCP.hpp"
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Vector.h"
#include "Epetra_IntVector.h"
#include "Epetra_Import.h"
#include "HYMLS_MatrixUtils.H"
#include "Epetra_Comm.h"
#include "EpetraExt_MatrixMatrix.h"

#include "Teuchos_FancyOStream.hpp"

// for sorting indices
#include <algorithm>

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#include <mpi.h>
#endif

#include "HYMLS_Tools.H"

#include "Teuchos_StandardCatchMacros.hpp"

#include "EpetraExt_HDF5.h"

#include "EpetraExt_Reindex_CrsMatrix.h"
#include "EpetraExt_Reindex_MultiVector.h"
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_VectorOut.h"
#include "EpetraExt_BlockMapOut.h"
//#include "EpetraExt_SubCopy_CrsMatrix.h"

#include "Isorropia_EpetraOrderer.hpp"

#include "AnasaziBlockKrylovSchurSolMgr.hpp"
#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziEpetraAdapter.hpp"

using Teuchos::null;
using Teuchos::rcp;

namespace HYMLS 
  {

    Teuchos::RCP<Epetra_Map> 
    MatrixUtils::CreateMap(int nx, int ny, int nz,
                           int dof, int indexbase,
                           const Epetra_Comm& comm)
      {
  // create a parallel map. We first figure out where in the domain we are                                                                                                               
  int np = comm.NumProc();
  int pid=comm.MyPID();

  int npX,npY,npZ;
  int pidX,pidY,pidZ;
  int offX,offY,offZ;
  int nXloc,nYloc,nZloc;

  HYMLS::Tools::SplitBox(nx,ny,nz,np,npX,npY,npZ);

    HYMLS::Tools::ind2sub(npX,npY,npZ,pid,pidX,pidY,pidZ);


  // dimension of subdomain

  nXloc = (int)(nx/npX);
  nYloc = (int)(ny/npY);
  nZloc = (int)(nz/npZ);

  // offsets for local->global index conversion
  offX = pidX*(int)(nx/npX);
  offY = pidY*(int)(ny/npY);
  offZ = pidZ*(int)(nz/npZ);

  // distribute remaining points among first few cpu's
  int remX=nx%npX;
  int remY=ny%npY;
  int remZ=nz%npZ;

  if (pidX<remX) nXloc++;
  if (pidY<remY) nYloc++;
  if (pidZ<remZ) nZloc++;

  for (int i=0;i<std::min(remX,pidX);i++) offX++;
  for (int i=0;i<std::min(remY,pidY);i++) offY++;
  for (int i=0;i<std::min(remZ,pidZ);i++) offZ++;

  if (indexbase!=0)
    {
    // this could easily be implemented but I didn't need it up to now.
    Tools::Error("only index base 0 is implemented right now",
        __FILE__, __LINE__);
    }

return CreateMap(offX,offX+nXloc-1,                                                                                                                                  
                 offY,offY+nYloc-1,
                 offZ,offZ+nZloc-1,
                 0, nx-1, 0, ny-1, 0, nz-1,
                 dof,comm);
      
      }

    Teuchos::RCP<Epetra_Map> 
    MatrixUtils::CreateMap(int i0, int i1, int j0, int j1, int k0, int k1,        
                           int I0, int I1, int J0, int J1, int K0, int K1,
                           int dof,
                           const Epetra_Comm& comm)
      {
      Teuchos::RCP<Epetra_Map> result = null;
      
      DEBUG("MatrixUtils::CreateMap ");
      DEBUG("["<<i0<<".."<<i1<<"]");
      DEBUG("["<<j0<<".."<<j1<<"]");
      DEBUG("["<<k0<<".."<<k1<<"]");
      
      int n = i1-i0+1; int N=I1-I0+1;
      int m = j1-j0+1; int M=J1-J0+1;
      int l = k1-k0+1; int L=K1-K0+1;
      
      DEBVAR(N);
      DEBVAR(M);
      DEBVAR(L);
      
      int NumMyElements = n*m*l*dof;
      int NumGlobalElements = -1; // note that there may be overlap
      int *MyGlobalElements = new int[NumMyElements];
      
      int pos = 0;
      for (int k=k0; k<=k1; k++)
        for (int j=j0; j<=j1; j++)
          for (int i=i0; i<=i1; i++)
            for (int var=0;var<dof;var++)
              {
              MyGlobalElements[pos++] = 
                Tools::sub2ind(N,M,L,dof,i,j,k,var);
              }
      result = rcp(new Epetra_Map(NumGlobalElements,
                NumMyElements,MyGlobalElements,0,comm));
      delete [] MyGlobalElements;
      return result;
      }

  // extract indices in a given global range [i1,i2]
  Teuchos::RCP<Epetra_Map> MatrixUtils::ExtractRange(const Epetra_Map& M, int i1, int i2)
    {
    
    int n = M.MaxAllGID();

#ifdef TESTING
 if (i1<0||i1>n) Tools::Error("CreateSubMap: lower bound out of range!",__FILE__,__LINE__);
 if (i2<0||i2>n) Tools::Error("CreateSubMap: upper bound out of range!",__FILE__,__LINE__);
 if (i2<i1) Tools::Error("CreateSubMap: invalid interval bounds!",__FILE__,__LINE__);
#endif    

    int *MyGlobalElements = new int[M.NumMyElements()];
    int p=0;
    int gid;
    for (int i=0;i<M.NumMyElements();i++)
      {
      gid = M.GID(i);
      if (gid>=i1 && gid<=i2) MyGlobalElements[p++]=gid;
      }
    
    
    // build the two new maps. Set global num el. to -1 so Epetra recomputes it
    Teuchos::RCP<Epetra_Map> M1 = Teuchos::rcp(new 
Epetra_Map(-1,p,MyGlobalElements,M.IndexBase(),M.Comm()) );
    delete [] MyGlobalElements;
    return M1;
    }
    

    //! extract a map with nun=nvars from a map with nun=6. 'var'
    //! is the array of variables to be extracted.
    Teuchos::RCP<Epetra_Map> MatrixUtils::CreateSubMap
        (const Epetra_Map& map, int dof, int var)
      {
      return CreateSubMap(map,dof,&var,1);
      }

    //! extract a map with nun=2 from a map with nun=6. 'var'
    //! are the variables to be extracted, i.e. {UU,VV}, {TT,SS} etc.
    Teuchos::RCP<Epetra_Map> 
    MatrixUtils::CreateSubMap(const Epetra_Map& map, int dof, const int var[2])
      {
      return CreateSubMap(map,dof,var,2);
      }
    
    //! extract a map with nun=nvars from a map with nun=6. 'var'
    //! is the array of variables to be extracted.
    Teuchos::RCP<Epetra_Map> 
    MatrixUtils::CreateSubMap(const Epetra_Map& map, int dof, const int *var, int nvars)
      {
      int dim = map.NumMyElements(); // number of entries in original map
      int numel = dim/dof; // number of blocks
      int subdim = numel*nvars; // number of entries in new map (<=dim)
      if (numel*dof!=dim)
        {
        Tools::Error("unexpected number of elements in map!",__FILE__,__LINE__);
        }
        
      int *MyGlobalElements = new int[subdim];
      
      // take the entries from the old map that correspond
      // to those in 'vars' and put them in the input array
      // for the new map.
      int k=0;
      for (int i=0; i<numel; i++)
        {
        for (int j=0; j<nvars; j++)
          {
          MyGlobalElements[k] = map.GID(i*dof+(var[j]-1));
          k++;
          }
        }
              
      Teuchos::RCP<Epetra_Map> submap = 
        Teuchos::rcp(new Epetra_Map(-1, subdim, MyGlobalElements, 0, map.Comm()));
      delete [] MyGlobalElements;
      return submap;
      }

    //! given a map and an array indicating wether each node of the map is to be 
    //! discarded (true) or not (false), this function creates a new map with the
    //! discarded entries removed.
    Teuchos::RCP<Epetra_Map> MatrixUtils::CreateSubMap
                 (const Epetra_Map& map, const bool* discard)
      {
      int numel = map.NumMyElements(); 
      int *MyGlobalElements = new int[numel]; // 'worst' case: no discarded nodes
      int numel_new = 0;
             
      for (int k=0;k<numel;k++)
        {
        if (!discard[k])
          {
          MyGlobalElements[numel_new] = map.GID(k); 
          numel_new++;
          }
        }
      Teuchos::RCP<Epetra_Map> submap = Teuchos::rcp(new Epetra_Map(-1, numel_new, 
MyGlobalElements, 
                  map.IndexBase(), map.Comm()));
      delete [] MyGlobalElements;
      return submap;
      }


  // compress a matrix' column map so that the resulting map contains 
  // only points actually appearing as column indices of the matrix   
  Teuchos::RCP<Epetra_Map> MatrixUtils::CompressColMap(const Epetra_CrsMatrix& A)
    {
    DEBUG("Compress column map of "<<A.Label());
    
    if (!A.HaveColMap()) Tools::Error("Matrix has no column map!",__FILE__,__LINE__);
    
    const Epetra_Map& old_map = A.ColMap();
    int n_old = old_map.NumMyElements();
    bool *is_col_entry = new bool[n_old];
    
    for (int i=0;i<n_old;i++) is_col_entry[i]=false;
    
    for (int i=0;i<A.NumMyRows();i++)
      {
      int *ind;
      int len;
      CHECK_ZERO(A.Graph().ExtractMyRowView(i,len,ind));
      for (int j=0;j<len;j++) is_col_entry[ind[j]]=true;
      }
      
    int n_new = 0;
    int *new_elements = new int[n_old];
    
    for (int i=0;i<n_old;i++) 
      {
      if (is_col_entry[i]) 
        {
        new_elements[n_new++] = old_map.GID(i);
        }
      }
    
    Teuchos::RCP<Epetra_Map> new_map = rcp(new 
           Epetra_Map(-1,n_new,new_elements,old_map.IndexBase(),old_map.Comm()));
    
    delete [] new_elements;
    delete [] is_col_entry;
        
    return new_map;
    }


  // create "Gather" map from "Solve" map
  Teuchos::RCP<Epetra_Map> MatrixUtils::Gather(const Epetra_BlockMap& map, int root)
    {

    int NumMyElements = map.NumMyElements();
    int NumGlobalElements = map.NumGlobalElements();
    const Epetra_Comm& Comm = map.Comm();
    
    int *MyGlobalElements = new int[NumMyElements];
    int *AllGlobalElements = NULL;

    for (int i=0; i<NumMyElements;i++)
      {
      MyGlobalElements[i] = map.GID(i);
      }
    
    if (Comm.MyPID()==root)
      {
      AllGlobalElements = new int[NumGlobalElements];
      }

if (Comm.NumProc()>1)
  {    
#ifdef HAVE_MPI    

    const Epetra_MpiComm MpiComm = dynamic_cast<const Epetra_MpiComm&>(Comm);
    int *counts, *disps;
    counts = new int[Comm.NumProc()];
    disps = new int[Comm.NumProc()+1];
    MPI_Gather(&NumMyElements,1,MPI_INTEGER,
               counts,1,MPI_INTEGER,root,MpiComm.GetMpiComm());
    
    if (Comm.MyPID()==root)
      {
      disps[0]=0;
      for (int p=0;p<Comm.NumProc();p++)
        {
        disps[p+1] = disps[p]+counts[p];
        }
      }

    MPI_Gatherv(MyGlobalElements, NumMyElements,MPI_INTEGER, 
                AllGlobalElements, counts,disps, MPI_INTEGER, root, MpiComm.GetMpiComm());
  delete [] counts;
  delete [] disps;                
#else
  Tools::Error("No MPI but still parallel??? We don't do that.",__FILE__,__LINE__);
#endif
  }
else
  {
  for (int i=0;i<NumMyElements;i++) AllGlobalElements[i]=MyGlobalElements[i];
  }  
    if (Comm.MyPID()!=root) 
      {
      NumMyElements=0;
      }
    else
      {
      NumMyElements=NumGlobalElements;
      std::sort(AllGlobalElements,AllGlobalElements+NumGlobalElements);
      }

  // build the new (gathered) map
  Teuchos::RCP<Epetra_Map> gmap = rcp(new Epetra_Map (NumGlobalElements, NumMyElements, 
                       AllGlobalElements, map.IndexBase(), Comm) );
    
    if (Comm.MyPID()==root)
      {      
      delete [] AllGlobalElements;
      }
    
    
    delete [] MyGlobalElements;
    
    return gmap;
    
    }


  // create "col" map from "Solve" map
  Teuchos::RCP<Epetra_Map> MatrixUtils::AllGather(const Epetra_BlockMap& map, bool reorder)
    {

    int NumMyElements = map.NumMyElements();
    int NumGlobalElements = map.NumGlobalElements();
    const Epetra_Comm& Comm = map.Comm();
    
    int *MyGlobalElements = new int[NumMyElements];
    int *AllGlobalElements = new int[NumGlobalElements];
    
    for (int i=0; i<NumMyElements;i++)
      {
      MyGlobalElements[i] = map.GID(i);
      }
    
  if (Comm.NumProc()>1)
    {
#ifdef HAVE_MPI
    const Epetra_MpiComm MpiComm = dynamic_cast<const Epetra_MpiComm&>(Comm);
    int *counts, *disps;
    counts = new int[Comm.NumProc()];
    disps = new int[Comm.NumProc()+1];
    MPI_Allgather(&NumMyElements,1,MPI_INTEGER,
               counts,1,MPI_INTEGER,MpiComm.GetMpiComm());
    
    disps[0]=0;
    for (int p=0;p<Comm.NumProc();p++)
      {
      disps[p+1] = disps[p]+counts[p];
      }

    MPI_Allgatherv(MyGlobalElements, NumMyElements,MPI_INTEGER, 
                AllGlobalElements, counts,disps, MPI_INTEGER, MpiComm.GetMpiComm());
    delete [] counts;
    delete [] disps;
#else
    Tools::Error("No MPI but still parallel? We don't do tthat.",__FILE__,__LINE__);                
#endif
    }
  else
    {
    for (int i=0;i<NumMyElements;i++) AllGlobalElements[i]=MyGlobalElements[i];
    }
    
  NumMyElements=NumGlobalElements;
  NumGlobalElements = -1;
  
  if (reorder)
    {
    std::sort(AllGlobalElements,AllGlobalElements+NumMyElements);
    }

  // build the new (gathered) map
  Teuchos::RCP<Epetra_Map> gmap = rcp(new Epetra_Map (NumGlobalElements, NumMyElements, 
                       AllGlobalElements, map.IndexBase(), Comm) );
    
    
    
    delete [] MyGlobalElements;
    delete [] AllGlobalElements;
    
    return gmap;
    
    }//AllGather
    
    Teuchos::RCP<Epetra_Vector> MatrixUtils::Gather(const Epetra_Vector& vec, int root)
      {
      DEBUG("Gather vector "<<vec.Label());
      const Epetra_BlockMap& map_dist = vec.Map();
      Teuchos::RCP<Epetra_Map> map = Gather(map_dist,root);
      
      Teuchos::RCP<Epetra_Vector> gvec = rcp(new Epetra_Vector(*map));
      
      Teuchos::RCP<Epetra_Import> import = rcp(new Epetra_Import(*map,map_dist) );
      
      CHECK_ZERO(gvec->Import(vec,*import,Insert));
      
      gvec->SetLabel(vec.Label());
      
      return gvec;
      
      }

    Teuchos::RCP<Epetra_Vector> MatrixUtils::AllGather(const Epetra_Vector& vec)
      {
      DEBUG("AllGather vector "<<vec.Label());
      const Epetra_BlockMap& map_dist = vec.Map();
      Teuchos::RCP<Epetra_Map> map = AllGather(map_dist);
      Teuchos::RCP<Epetra_Vector> gvec = rcp(new Epetra_Vector(*map));
      
      Teuchos::RCP<Epetra_Import> import = rcp(new Epetra_Import(*map,map_dist) );
      
      CHECK_ZERO(gvec->Import(vec,*import,Insert));
      
      gvec->SetLabel(vec.Label());
      DEBUG("done!");      
      return gvec;
      
      }

    Teuchos::RCP<Epetra_IntVector> MatrixUtils::Gather
        (const Epetra_IntVector& vec, int root)
      {
      DEBUG("Gather vector "<<vec.Label());
      const Epetra_BlockMap& map_dist = vec.Map();
      Teuchos::RCP<Epetra_Map> map = Gather(map_dist,root);
      
      Teuchos::RCP<Epetra_IntVector> gvec = rcp(new Epetra_IntVector(*map));
      
      Teuchos::RCP<Epetra_Import> import = rcp(new Epetra_Import(*map,map_dist) );
      
      CHECK_ZERO(gvec->Import(vec,*import,Insert));
      
      gvec->SetLabel(vec.Label());
      
      return gvec;
      
      }

    Teuchos::RCP<Epetra_IntVector> MatrixUtils::AllGather(const Epetra_IntVector& vec)
      {
      DEBUG("AllGather vector "<<vec.Label());
      const Epetra_BlockMap& map_dist = vec.Map();
      Teuchos::RCP<Epetra_Map> map = AllGather(map_dist);
      Teuchos::RCP<Epetra_IntVector> gvec = rcp(new Epetra_IntVector(*map));
      
      Teuchos::RCP<Epetra_Import> import = rcp(new Epetra_Import(*map,map_dist) );
      
      CHECK_ZERO(gvec->Import(vec,*import,Insert));
      
      gvec->SetLabel(vec.Label());
      DEBUG("done!");      
      return gvec;
      
      }

    Teuchos::RCP<Epetra_CrsMatrix> MatrixUtils::Gather(const Epetra_CrsMatrix& mat, int root)
      {
      DEBUG("Gather matrix "<<mat.Label());
      const Epetra_Map& rowmap_dist = mat.RowMap();
      // we take the domain map as the colmap is potentially overlapping
      const Epetra_Map& colmap_dist = mat.DomainMap();
      // gather the row map
      Teuchos::RCP<Epetra_Map> rowmap = Gather(rowmap_dist,root);
      // gather the col map
      Teuchos::RCP<Epetra_Map> colmap = Gather(colmap_dist,root);
      
      //we only guess the number of row entries, this routine is not performance critical
      // as it should only be used for debugging anyway
      int num_entries = mat.NumGlobalNonzeros()/mat.NumGlobalRows();
      Teuchos::RCP<Epetra_CrsMatrix> gmat = rcp(new Epetra_CrsMatrix(Copy,*rowmap, *colmap, num_entries) );
      
      Teuchos::RCP<Epetra_Import> import = rcp(new Epetra_Import(*rowmap,rowmap_dist) );
      
      CHECK_ZERO(gmat->Import(mat,*import,Insert));
      
      CHECK_ZERO(gmat->FillComplete());
      gmat->SetLabel(mat.Label());
      
      return gmat;
      
      }

  // distribute a gathered vector among processors
  Teuchos::RCP<Epetra_Vector> MatrixUtils::Scatter(const Epetra_Vector& vec, const Epetra_BlockMap& distmap)
    {
    Teuchos::RCP<Epetra_Vector> dist_vec =  rcp(new Epetra_Vector(distmap));
    Teuchos::RCP<Epetra_Import> import = rcp(new Epetra_Import(vec.Map(),distmap));
    CHECK_ZERO(dist_vec->Export(vec,*import,Insert));
    return dist_vec;
    }

// workaround for the buggy Trilinos routine with the same name
Teuchos::RCP<Epetra_CrsMatrix> MatrixUtils::ReplaceRowMap(Teuchos::RCP<Epetra_CrsMatrix> A,const Epetra_Map& newmap)
   {
   int maxlen = A->MaxNumEntries();
   int len;
   int *ind = new int[maxlen];
   double *val = new double[maxlen];
   int nloc = A->NumMyRows();
   int *row_lengths = new int[nloc];
   for (int i=0;i<nloc;i++) row_lengths[i]=A->NumMyEntries(i);
   Teuchos::RCP<Epetra_CrsMatrix> tmpmat;
   if (A->HaveColMap())
     {
     tmpmat = rcp(new Epetra_CrsMatrix(Copy,newmap,A->ColMap(), row_lengths) );
     }
   else
     {
     tmpmat = rcp(new Epetra_CrsMatrix(Copy,newmap, row_lengths) );
     }
   
   int rowA,rowNew;
   for (int i=0;i<A->NumMyRows();i++)
      {
      rowA = A->GRID(i);
      rowNew = newmap.GID(i);
      CHECK_ZERO(A->ExtractGlobalRowCopy(rowA,maxlen,len,val,ind));
      CHECK_ZERO(tmpmat->InsertGlobalValues(rowNew, len, val, ind));
      }
   tmpmat->SetLabel(A->Label());
   delete [] ind;
   delete [] val;
   delete [] row_lengths;   
   return tmpmat;
   }    

// create an exact copy of a matrix replacing the column map.
// The column maps have to be 'compatible' 
// in the sense that the new ColMap is a subset of the old one.
Teuchos::RCP<Epetra_CrsMatrix> MatrixUtils::ReplaceColMap(Teuchos::RCP<Epetra_CrsMatrix> A, const Epetra_Map& newcolmap)
   {
   int maxlen = A->MaxNumEntries();
   int len;
   int *ind = new int[maxlen];
   double *val = new double[maxlen];
   int nloc = A->NumMyRows();
   int *row_lengths = new int[nloc];
   for (int i=0;i<nloc;i++) row_lengths[i]=A->NumMyEntries(i);
   Teuchos::RCP<Epetra_CrsMatrix> tmpmat;
   tmpmat = rcp(new Epetra_CrsMatrix(Copy,A->RowMap(),
        newcolmap, row_lengths) );
   
   int grid;
   for (int i=0;i<nloc;i++)
      {
      grid = A->GRID(i);
      CHECK_ZERO(A->ExtractGlobalRowCopy(grid,maxlen,len,val,ind));
#ifdef DEBUGGING
//      (*debug) << "row " << grid << ": ";
//      for (int j=0;j<len;j++) (*debug) << ind[j] << " ";
//      (*debug) << std::endl;
#endif      
      CHECK_ZERO(tmpmat->InsertGlobalValues(grid, len, val, ind));
      }
   tmpmat->SetLabel(A->Label());
   delete [] ind;
   delete [] val;
   delete [] row_lengths;
   return tmpmat;
   }    
   
   
// create an exact copy of a matrix removing the column map.
// This means that row- and column map have to be 'compatible' 
// in the sense that the ColMap is a subset of the RowMap.
// It seems to be required in order to use Ifpack in some cases.
Teuchos::RCP<Epetra_CrsMatrix> MatrixUtils::RemoveColMap(Teuchos::RCP<Epetra_CrsMatrix> A)
   {
   int maxlen = A->MaxNumEntries();
   int len;
   int *ind = new int[maxlen];
   double *val = new double[maxlen];
   int nloc = A->NumMyRows();
   int *row_lengths = new int[nloc];
   for (int i=0;i<nloc;i++) row_lengths[i]=A->NumMyEntries(i);
   Teuchos::RCP<Epetra_CrsMatrix> tmpmat;
   tmpmat = rcp(new Epetra_CrsMatrix(Copy,A->RowMap(), row_lengths) );
   
   int grid;
   for (int i=0;i<A->NumMyRows();i++)
      {
      grid = A->GRID(i);
      CHECK_ZERO(A->ExtractGlobalRowCopy(grid,maxlen,len,val,ind));
      CHECK_ZERO(tmpmat->InsertGlobalValues(grid, len, val, ind));
      }
   tmpmat->SetLabel(A->Label());
   delete [] ind;
   delete [] val;
   delete [] row_lengths;
   return tmpmat;
   }    
   
   
   
// simultaneously replace row and column map
Teuchos::RCP<Epetra_CrsMatrix> MatrixUtils::ReplaceBothMaps(Teuchos::RCP<Epetra_CrsMatrix> A,const Epetra_Map& newmap, 
                   const Epetra_Map& newcolmap)
   {
   DEBUG("Replace Row and Col Map...");
   DEBVAR(A->RowMap());
   DEBVAR(newmap);
   DEBVAR(A->ColMap());
   DEBVAR(newcolmap);
   int maxlen = A->MaxNumEntries();
   int len;
   int *ind = new int[maxlen];
   double *val = new double[maxlen];
   int nloc = A->NumMyRows();
   int *row_lengths = new int[nloc];
   for (int i=0;i<nloc;i++) row_lengths[i]=A->NumMyEntries(i);
   Teuchos::RCP<Epetra_CrsMatrix> tmpmat;
   tmpmat = rcp(new Epetra_CrsMatrix(Copy,newmap,newcolmap,row_lengths) );
   
   int rowA,rowNew;
   
   for (int i=0;i<A->NumMyRows();i++)
      {
      rowA = A->GRID(i);
      rowNew = newmap.GID(i);
      CHECK_ZERO(A->ExtractGlobalRowCopy(rowA,maxlen,len,val,ind));
      for (int j=0;j<len;j++) 
        {
        int newind=newcolmap.GID(A->LCID(ind[j]));
//        DEBUG(i<<" ("<<rowA<<"->"<<rowNew<<"), "<<A->LCID(ind[j])<<"("<<ind[j]<<"->"<<newind<<")");
        ind[j] = newind;
        }
      CHECK_ZERO(tmpmat->InsertGlobalValues(rowNew, len, val, ind));
      }

   tmpmat->SetLabel(A->Label());
   delete [] ind;
   delete [] val;
   delete [] row_lengths;
   return tmpmat;
   }

//! work-around for 'Solve' bug (not sure it is one, yet)
void MatrixUtils::TriSolve(const Epetra_CrsMatrix& A, const Epetra_Vector& b, Epetra_Vector& x)
  {
#ifdef TESTING
  if (!(A.UpperTriangular()||A.LowerTriangular()))
    Tools::Error("Matrix doesn't look (block-)triangular enough for TriSolve...",__FILE__,__LINE__);
  if (!A.StorageOptimized())
    Tools::Error("Matrix has to be StorageOptimized() for TriSolve!",__FILE__,__LINE__);
  if (!b.Map().SameAs(A.RangeMap()))
    Tools::Error("Rhs vector out of range for TriSolve!",__FILE__,__LINE__);
  if (!x.Map().SameAs(A.DomainMap()))
    Tools::Error("Sol vector not in domain!",__FILE__,__LINE__);
#endif  
    
  if (A.UpperTriangular())
    {
    DEBUG("Upper Tri Solve with "<<A.Label()<<"...");
    int *begA,*jcoA;
    double *coA;
    CHECK_ZERO(A.ExtractCrsDataPointers(begA,jcoA,coA));
    double sum;
    int diag;
    for (int i=A.NumMyRows()-1;i>=0;i--)
      {
      diag = begA[i];
      sum = 0.0;
      for (int j=diag+1;j<begA[i+1];j++)
        {
//        DEBUG(i<<" "<<jcoA[j]<<" "<<coA[j]);
        sum+=coA[j]*x[jcoA[j]];
        }
//      DEBUG("diag: "<<i<<" "<<jcoA[diag]<<" "<<coA[diag]);
      x[i] = (b[i] - sum)/coA[diag];
      }
    }
  else
    {
    DEBUG("Lower Tri Solve with"<<A.Label()<<"...");
    int *begA,*jcoA;
    double *coA;
    CHECK_ZERO(A.ExtractCrsDataPointers(begA,jcoA,coA));
    double sum;
    int diag;
    for (int i=0;i<A.NumMyRows();i++)
      {
      diag = begA[i+1]-1;
      sum = 0.0;
      for (int j=0;j<diag;j++)
        {
//        DEBUG(i<<" "<<jcoA[j]<<" "<<coA[j]);
        sum+=coA[j]*x[jcoA[j]];
        }
//      DEBUG("diag: "<<i<<" "<<jcoA[diag]<<" "<<coA[diag]);
      x[i] = (b[i] - sum)/coA[diag];
      }
    }
  }//TriSolve


// make A identity matrix
void MatrixUtils::Identity(Teuchos::RCP<Epetra_CrsMatrix> A)
  {
  double val =1.0;
  int ind;
  A->PutScalar(0.0);
  for (int i=0;i<A->NumMyRows();i++)
    {
    ind = A->GRID(i);
    CHECK_ZERO(A->ReplaceGlobalValues(ind,1,&val,&ind));
    }
  A->SetLabel("Identity");
  CHECK_ZERO(A->FillComplete());
  }
  
    


// write CRS matrix to file
void MatrixUtils::Dump(const Epetra_CrsMatrix& A, const string& filename,bool reindex)
  {
  DEBUG("Matrix with label "<<A.Label()<<" is written to file "<<filename);
  
  if (reindex)
    {
    Teuchos::RCP<Epetra_Map> newMap;
    int myLength = A.NumMyRows();
    newMap=Teuchos::rcp(new Epetra_Map(-1,myLength,0,A.Comm()));
    EpetraExt::CrsMatrix_Reindex renumber(*newMap);
    Dump(renumber(const_cast<Epetra_CrsMatrix&>(A)),filename,false);
    }
  else
    {
#if 0
    EpetraExt::RowMatrixToMatrixMarketFile(filename.c_str(),A);
#elif 0
    Teuchos::RCP<std::ostream> ofs = rcp(new Teuchos::oblackholestream());
    int my_rank = A.Comm().MyPID();
    if (my_rank==0)
      {
      ofs = rcp(new std::ofstream(filename.c_str()));
      }
    *ofs << std::scientific << std::setw(15) << std::setprecision(15);
    *ofs << *(MatrixUtils::Gather(A,0));
#else
    int my_rank = A.Comm().MyPID();
    for (int p=0; p<A.Comm().NumProc();p++)
      {
      if (my_rank==p)
        {
        Teuchos::RCP<std::ofstream> ofs;
        if (p==0)
          {
          ofs = rcp(new std::ofstream(filename.c_str(),std::ios::trunc));
          }
        else
          {
          ofs = rcp(new std::ofstream(filename.c_str(),std::ios::app));
          }          
        *ofs << std::scientific << std::setw(15) << std::setprecision(15);
        *ofs << A;
        ofs->close();
        }
      A.Comm().Barrier();
      }    
#endif
    }
  }


void MatrixUtils::DumpHDF(const Epetra_CrsMatrix& A, 
                                const string& filename, 
                                const string& groupname,
                                bool new_file)
  {
#ifndef HAVE_XDMF
  Tools::Error("HDF format can't be stored, recompile with -DHAVE_XDMF",__FILE__,__LINE__);
#else
  bool verbose=true;
  bool success;
  DEBUG("Matrix with label "<<A.Label()<<" is written to HDF5 file "<<filename<<", group "<<groupname);
  RCP<EpetraExt::HDF5> hdf5 = rcp(new EpetraExt::HDF5(A.Comm()));
try {
  if (new_file)
    {       
    hdf5->Create(filename.c_str());
    }
  else
    {
    hdf5->Open(filename.c_str());
    }
  hdf5->Write(groupname,A);
  hdf5->Close();
} TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose,std::cerr,success);
#endif  
  }                                

// write CRS matrix to file
void MatrixUtils::Dump(const Epetra_MultiVector& x, const string& filename,bool reindex)
  {  
  if (reindex)
    {
    Teuchos::RCP<Epetra_Map> newMap;
    int myLength = x.MyLength();
    newMap=Teuchos::rcp(new Epetra_Map(-1,myLength,0,x.Comm()));
    EpetraExt::MultiVector_Reindex renumber(*newMap);
    Dump(renumber(const_cast<Epetra_MultiVector&>(x)),filename,false);
    }
  else
    {
    DEBUG("Vector with label "<<x.Label()<<" is written to file "<<filename);

    //EpetraExt::VectorToMatrixMarketFile(filename.c_str(),x);
    
    if (x.NumVectors()!=1)
      {
      Tools::Warning("Only dumping vector 1!",__FILE__,__LINE__);
      }
    
    Teuchos::RCP<std::ostream> ofs = rcp(new Teuchos::oblackholestream());
    int my_rank = x.Comm().MyPID();
    if (my_rank==0)
      {
      ofs = rcp(new std::ofstream(filename.c_str()));
      }
    *ofs << std::scientific << std::setw(15) << std::setprecision(15);
    *ofs << *(MatrixUtils::Gather(*(x(0)),0));
    }
  }

// write CRS IntVector to file
void MatrixUtils::Dump(const Epetra_IntVector& x, const string& filename)
  {  
  DEBUG("Vector with label "<<x.Label()<<" is written to file "<<filename);

  //EpetraExt::VectorToMatrixMarketFile(filename.c_str(),x);
    
  Teuchos::RCP<std::ostream> ofs = rcp(new Teuchos::oblackholestream());
  int my_rank = x.Comm().MyPID();
  if (my_rank==0)
    {
    ofs = rcp(new std::ofstream(filename.c_str()));
    }
  *ofs << std::setw(15) << std::setprecision(15);
  *ofs << *(MatrixUtils::Gather(x,0));
  }


void MatrixUtils::DumpHDF(const Epetra_MultiVector& x, 
                                const string& filename, 
                                const string& groupname,
                                bool new_file)
  {
#ifndef HAVE_XDMF
  Tools::Error("HDF format can't be stored, recompile with -DHAVE_XDMF",__FILE__,__LINE__);
#else
  bool verbose=true;
  bool success;
  DEBUG("Vector with label "<<x.Label()<<" is written to HDF5 file "<<filename<<", group   "<<groupname);
  RCP<EpetraExt::HDF5> hdf5 = rcp(new EpetraExt::HDF5(x.Comm()));
try {
  if (new_file)
    {       
    hdf5->Create(filename.c_str());
    }
  else
    {
    hdf5->Open(filename.c_str());
    }
  hdf5->Write(groupname,x);
  hdf5->Close();
} TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose,std::cerr,success);
#endif  
  x.Comm().Barrier();
  }

// write map to file
void MatrixUtils::Dump(const Epetra_Map& M, const string& filename)
  {
  DEBUG("Map with label "<<M.Label()<<" is written to file "<<filename);
  EpetraExt::BlockMapToMatrixMarketFile(filename.c_str(),M);
  /*
  Teuchos::RCP<std::ostream> ofs = rcp(new Teuchos::oblackholestream());
  int my_rank = M.Comm().MyPID();
  if (my_rank==0)
    {
    ofs = rcp(new std::ofstream(filename.c_str()));
    }
  *ofs << std::setw(15) << std::setprecision(15);
  *ofs << *(MatrixUtils::Gather(M,0));  
  */
  }

// print row matrix
void MatrixUtils::PrintRowMatrix(const Epetra_RowMatrix& A, std::ostream& os)
  {
  DEBUG("Print Row Matrix: "<<A.Label());
  int nrows = A.NumMyRows();
  int ncols = A.NumMyCols();
  int nnz = A.NumMyNonzeros();
  int nrows_g = A.NumGlobalRows();
  int ncols_g = A.NumGlobalCols();
  int nnz_g = A.NumGlobalNonzeros();
  int maxlen = ncols;  
  int len;
  int *indices = new int[maxlen];
  double *values = new double[maxlen];
  int grid,gcid;
  
  os << "Number of Rows: " << nrows;
  
  if (nrows!=nrows_g) os << " [g"<<nrows_g<<"]";
    
  os << std::endl;

  os << "Number of Columns: " << ncols;

  if (ncols!=ncols_g) os << " [g"<<ncols_g<<"]";

  os << std::endl;

  os << "Number of Nonzero Entries: " << nnz;

  if (nnz!=nnz_g) os << " [g"<<nnz_g<<"]";

  
  os << std::endl;
  
  for (int i=0;i<nrows;i++)
    {
    grid = A.RowMatrixRowMap().GID(i);
    CHECK_ZERO(A.ExtractMyRowCopy(i,maxlen,len,values,indices));
    for (int j=0;j<len;j++)
      {
      gcid = A.RowMatrixColMap().GID(indices[j]);
//      os << A.Comm().MyPID() << "\t";
      os << i;
      if (grid!=i) os << " [g"<< grid <<"]";
      os << "\t";
      os << indices[j];
      if (gcid!=indices[j]) os << " [g"<< gcid <<"]";
      os << "\t";
      os << values[j] << std::endl;
      }
    }
  delete [] indices;
  delete [] values;
  }

Teuchos::RCP<Epetra_CrsMatrix> MatrixUtils::TripleProduct(bool transA, const Epetra_CrsMatrix& A,
                                          bool transB, const Epetra_CrsMatrix& B,
                                          bool transC, const Epetra_CrsMatrix& C)
  {
  
    // trans(A) is not available as we prescribe the row-map of A*B, but if it is needed
    // at some point it can be readily implemented
    if(transA) Tools::Error("This case is not implemented: trans(A)*op(B)*op(C)\n",__FILE__,__LINE__);
  
    // temp matrix
    Teuchos::RCP<Epetra_CrsMatrix> AB = rcp(new Epetra_CrsMatrix(Copy,A.RowMap(),A.MaxNumEntries()) );

    DEBUG("compute A*B...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(A,transA,B,transB,*AB));

    // result matrix
    Teuchos::RCP<Epetra_CrsMatrix> ABC = rcp(new Epetra_CrsMatrix(Copy,AB->RowMap(),AB->MaxNumEntries()) );

    DEBUG("compute ABC...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(*AB,false,C,transC,*ABC));

    DEBUG("done!");
    return ABC;
    }

void MatrixUtils::TripleProduct(Teuchos::RCP<Epetra_CrsMatrix> ABC, bool transA, const Epetra_CrsMatrix& A,
                                          bool transB, const Epetra_CrsMatrix& B,
                                          bool transC, const Epetra_CrsMatrix& C)
  {
  
    // temp matrix
    Teuchos::RCP<Epetra_CrsMatrix> AB = rcp(new Epetra_CrsMatrix(Copy,ABC->Graph()) );

    DEBUG("compute A*B...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(A,transA,B,transB,*AB));


    DEBUG("compute ABC...");
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(*AB,false,C,transC,*ABC));

    DEBUG("done!");
    }

Teuchos::RCP<Epetra_CrsMatrix> MatrixUtils::MatrixProduct(bool transA, const Epetra_CrsMatrix& A,
                                          bool transB, const Epetra_CrsMatrix& B)
  {
  
    Teuchos::RCP<Epetra_CrsMatrix> AB = rcp(new Epetra_CrsMatrix(Copy,A.RowMap(),A.MaxNumEntries()) );

    DEBUG("compute A*B...");
    DEBVAR(transA);
    DEBVAR(A.NumGlobalRows());
    DEBVAR(A.NumGlobalCols());
    DEBVAR(transB);
    DEBVAR(B.NumGlobalRows());
    DEBVAR(B.NumGlobalCols());

    
#ifdef TESTING
  if (!A.Filled()) Tools::Error("Matrix A not filled!",__FILE__,__LINE__);
  if (!B.Filled()) Tools::Error("Matrix B not filled!",__FILE__,__LINE__);
#endif    
    
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(A,transA,B,transB,*AB));

    DEBUG("done!");
    return AB;
    }

void MatrixUtils::MatrixProduct(Teuchos::RCP<Epetra_CrsMatrix> AB,bool transA, const Epetra_CrsMatrix& A,
                                          bool transB, const Epetra_CrsMatrix& B)
  {
  
    DEBUG("compute A*B...");
    DEBVAR(transA);
    DEBVAR(A.NumGlobalRows());
    DEBVAR(A.NumGlobalCols());
    DEBVAR(transB);
    DEBVAR(B.NumGlobalRows());
    DEBVAR(B.NumGlobalCols());
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(A,transA,B,transB,*AB));

    DEBUG("done!");
    }


Teuchos::RCP<MatrixUtils::Eigensolution> MatrixUtils::Eigs(
                Teuchos::RCP<const Epetra_Operator> A,
                Teuchos::RCP<const Epetra_Operator> B,
                int howMany,
                double tol)
  {
  START_TIMER(Label(),"Eigs");

  typedef double ST;
  typedef Epetra_MultiVector MV;
  typedef Epetra_Operator OP;
  typedef Teuchos::ScalarTraits<ST>        SCT;
  typedef SCT::magnitudeType               MagnitudeType;
  typedef Anasazi::MultiVecTraits<ST,MV>     MVT;
  typedef Anasazi::OperatorTraits<ST,MV,OP>  OPT;

  int ierr;

  // ************************************
  // Start the block Arnoldi iteration
  // ***********************************
  //
  //  Variables used for the Block Krylov Schur Method
  //
  bool boolret;
  int MyPID = A->Comm().MyPID();

  bool verbose = true;
  bool debug = false;
#ifdef TESTING
  verbose=true;
#endif

  std::string which("LR");

  int blockSize = 1;
  int numBlocks = 250;
  int maxRestarts = 10;

  // Create a sort manager to pass into the block Krylov-Schur solver manager
  // -->  Make sure the reference-counted pointer is of type Anasazi::SortManager<>
  // -->  The block Krylov-Schur solver manager uses Anasazi::BasicSort<> by default,
  //      so you can also pass in the parameter "Which", instead of a sort manager.
//  Teuchos::RCP<Anasazi::SortManager<ST> > MySort =
//    Teuchos::rcp( new Anasazi::BasicSort<ST>( which ) );

  // Set verbosity level
  int verbosity = Anasazi::Errors + Anasazi::Warnings;
  if (verbose) {
    verbosity += Anasazi::FinalSummary + Anasazi::TimingDetails;
  }
  if (debug) {
    verbosity += Anasazi::Debug;
  }

  //
  // Create parameter list to pass into solver manager
  //
  Teuchos::ParameterList MyPL;
  MyPL.set( "Verbosity", verbosity );
  //TODO: in the Trilinos version on Huygens, one can't 
  //      set the output stream, is that fixed in the   
  //      more recent versions?
  MyPL.set( "Output Stream",Tools::out().getOStream());
  

//  MyPL.set( "Sort Manager", MySort );
  MyPL.set( "Which", which );
  MyPL.set( "Block Size", blockSize );
  MyPL.set( "Num Blocks", numBlocks );
  MyPL.set( "Maximum Restarts", maxRestarts );
  //MyPL.set( "Step Size", stepSize );
  MyPL.set( "Convergence Tolerance", tol );

  // Create an Epetra_MultiVector for an initial vector to start the solver.
  // Note:  This needs to have the same number of columns as the blocksize.
  Teuchos::RCP<Epetra_MultiVector> ivec =
    Teuchos::rcp( new Epetra_MultiVector(A->OperatorRangeMap(), blockSize) );
  MatrixUtils::Random(*ivec);
  
  Epetra_MultiVector tmp = *ivec;
  if (!Teuchos::is_null(B))
    {
    CHECK_ZERO(B->Apply(tmp,*ivec));
    }
  // Create the eigenproblem.
  Teuchos::RCP<Anasazi::BasicEigenproblem<ST, MV, OP> > MyProblem;

  if (Teuchos::is_null(B))
    {
    MyProblem =  Teuchos::rcp( new Anasazi::BasicEigenproblem<ST, MV, OP>(A, ivec) );
    }
  else
    { 
    MyProblem =  Teuchos::rcp( new Anasazi::BasicEigenproblem<ST, MV, OP>(A,B, ivec) );
    }

  // Inform the eigenproblem that the operator A is symmetric
  MyProblem->setHermitian(false);

  // Set the number of eigenvalues requested
  MyProblem->setNEV( howMany );

  // Inform the eigenproblem that you are finishing passing it information
  boolret = MyProblem->setProblem();
  if (boolret != true)
    {
    Tools::Error("Anasazi::BasicEigenproblem::setProblem() returned with error.",
        __FILE__,__LINE__);
    }

  // Initialize the Block Arnoldi solver
  Anasazi::BlockKrylovSchurSolMgr<ST, MV, OP> MySolverMgr(MyProblem, MyPL);

  // Solve the problem to the specified tolerances or length
  Anasazi::ReturnType returnCode;
  returnCode = MySolverMgr.solve();
  if (returnCode != Anasazi::Converged)
    {
    Tools::Warning("Anasazi::EigensolverMgr::solve() returned unconverged.",
        __FILE__,__LINE__);
    }

  // Get the Ritz values from the eigensolver
  std::vector<Anasazi::Value<double> > ritzValues = MySolverMgr.getRitzValues();

  if (verbose)
    {
    // Output computed eigenvalues and their direct residuals
    int numritz = (int)ritzValues.size();
    Tools::out()<<"operator: "<<A->Label()<<std::endl;
    Tools::out()<<std::endl<< "Computed Ritz Values"<< std::endl;
    Tools::out()<< std::setw(16) << "Real Part"
            << std::setw(16) << "Imag Part"
            << std::endl;
    Tools::out()<<"-----------------------------------------------------------"<<std::endl;
    for (int i=0; i<numritz; i++)
      {
      Tools::out()<< std::setw(16) << ritzValues[i].realpart
              << std::setw(16) << ritzValues[i].imagpart
              << std::endl;
      }
    Tools::out()<<"-----------------------------------------------------------"<<std::endl;
    }

  STOP_TIMER(Label(),"Eigs");
  return Teuchos::rcp(new Eigensolution(MyProblem->getSolution()));
  }


Teuchos::RCP<Epetra_CrsMatrix> MatrixUtils::ReadThcmMatrix(string prefix, const Epetra_Comm& comm,
              const Epetra_Map& rowmap,
              const Epetra_Map& colmap,
              const Epetra_Map* rangemap,
              const Epetra_Map* domainmap)
  {
  
  if (comm.NumProc()>1)
    {
    // this routine is only intended for sequential debugging, up to now...
    Tools::Error("Fortran Matrix input is not possible in parallel case!",__FILE__,__LINE__);
    }
  
  DEBUG("Read THCM Matrix with label "<<prefix);
  string infofilename = prefix+".info";
  std::ifstream infofile(infofilename.c_str());
  int nnz,nrows;
  infofile >> nrows >> nnz;
  infofile.close();

  int *begA = new int[nrows+1];
  int *jcoA = new int[nnz];
  double *coA = new double[nnz];
  int *indices = new int[nrows];
  double *values = new double[nrows];

  read_fortran_array(nrows+1,begA,prefix+".beg");
  read_fortran_array(nnz,jcoA,prefix+".jco");
  read_fortran_array(nnz,coA,prefix+".co");
  int *len = new int[nrows];
  for (int i=0;i<nrows;i++) 
    {
    len[i] = begA[i+1]-begA[i];
    }

Teuchos::RCP<Epetra_CrsMatrix> A=rcp(new Epetra_CrsMatrix(Copy, rowmap, colmap, len, true));

  // put CSR arrays in Trilinos Jacobian
  for (int i = 0; i<nrows; i++)
    {
    int row = rowmap.GID(i);
    int index = begA[i]; // note that these arrays use 1-based indexing
    int numentries = begA[i+1] - index;
    for (int j = 0; j <  numentries ; j++)
      {
      indices[j] = colmap.GID(jcoA[index-1+j] - 1);
      values[j] = coA[index - 1 + j];
      }
    CHECK_ZERO(A->InsertGlobalValues(row, numentries, values, indices));
    }
  A->SetLabel(prefix.c_str());
  if (rangemap==NULL || domainmap==NULL)
    {
    CHECK_ZERO(A->FillComplete());
    }
  else
    {
    CHECK_ZERO(A->FillComplete(*domainmap,*rangemap));
    }
  return A;
  }

//! private helper function for THCM I/O
void MatrixUtils::read_fortran_array(int n, int* array, string filename)
  {
  std::ifstream ifs(filename.c_str());
  for (int i=0;i<n;i++)
    {
    ifs >> array[i];
    }
  ifs.close();
  }

//! private helper function for THCM I/O
void MatrixUtils::read_fortran_array(int n, double* array, string filename)
  {
  std::ifstream ifs(filename.c_str());
  for (int i=0;i<n;i++)
    {
    ifs >> array[i];
    }
  ifs.close();
  }


int MatrixUtils::Random(Epetra_MultiVector& v, int seed)
  {
  Teuchos::RCP<Epetra_Vector> gVec;
  for (int k=0;k<v.NumVectors();k++)
    {
    gVec=Gather(*v(k),0);
    if (seed>0)
      {
      gVec->SetSeed(seed+k);
      }
    gVec->Random();
    *v(k) = *Scatter(*gVec,v.Map());
    }
  return 0;
  }

// drop small matrix entries (relative to diagonal element)
Teuchos::RCP<Epetra_CrsMatrix> MatrixUtils::DropByValue
        (Teuchos::RCP<const Epetra_CrsMatrix> A, double droptol,DropType type)
  {
  START_TIMER2(std::string("MatrixUtils"),"DropByValue");

  Teuchos::RCP<Epetra_CrsMatrix> mat
        = Teuchos::rcp(new Epetra_CrsMatrix(Copy, A->RowMap(), A->MaxNumEntries()));
        
  Teuchos::RCP<Epetra_Vector> diagA;
  
  if (type==Relative)
    {
    diagA = Teuchos::rcp(new Epetra_Vector(A->RowMap()));
    CHECK_ZERO(A->ExtractDiagonalCopy(*diagA));
    }

  int len;
  int *indices; 
  double *values; 
  
  int new_len;
  int *new_indices = new int[A->MaxNumEntries()];
  double *new_values = new double[A->MaxNumEntries()];

  double thres=droptol;

  for (int i=0;i<A->NumMyRows();i++)
    {
    CHECK_ZERO(A->ExtractMyRowView(i,len,values,indices));
    
    if (type==Relative)
      {
      thres = droptol*abs((*diagA)[i]);
      }
    
    new_len=0;
    for (int j=0;j<len;j++)
      {
      if ( (std::abs(values[j]) > thres)||(A->GCID(indices[j])==A->GRID(i)) )
        {
        new_values[new_len]=values[j];
        new_indices[new_len]=A->GCID(indices[j]);
        new_len++;
        }
      }
#ifdef DEBUGGING
DEBUG("insert row "<<A->GRID(i)<<" (length "<<new_len<<")");
for (int j=0;j<new_len;j++)
  {
  Tools::deb() << "("<<new_indices[j]<<", "<<new_values[j]<<") ";
  }
Tools::deb() << std::endl;
#endif    
    CHECK_ZERO(mat->InsertGlobalValues(A->GRID(i),new_len,new_values,new_indices));
    }

  delete [] new_indices;
  delete [] new_values;

  DEBUG("calling FillComplete()");
  CHECK_ZERO(mat->FillComplete());
  
#ifdef TESTING
int old_nnz = A->NumGlobalNonzeros();
int new_nnz = mat->NumGlobalNonzeros();
int nnz_dropped = old_nnz-new_nnz;
double percent_dropped=100.0*(((double)nnz_dropped)/((double)old_nnz));

Tools::Out("DropByValue ("+Teuchos::toString(droptol)+"):");
Tools::Out(" => dropped "+Teuchos::toString(percent_dropped)+"% of nonzeros");

#endif

  STOP_TIMER2(std::string("MatrixUtils"),"DropByValue");
  return mat;
  }


int MatrixUtils::PutDirichlet(Epetra_CrsMatrix& A, int gid)
  {

  // find out which proc owns this row

  int lid, pid;
  
  EPETRA_CHK_ERR(A.RowMap().RemoteIDList(1,&gid,&pid,&lid));

  // find out how long that row is (how many nonzeros)
  int len;

  if (pid==A.Comm().MyPID())
    {
    EPETRA_CHK_ERR(A.NumMyRowEntries(lid,len));
    }
  
  EPETRA_CHK_ERR(A.Comm().Broadcast(&len,1,pid));

  int* indices=new int[len];
  double* values=new double[len];
      
  if (pid==A.Comm().MyPID())
    {
    int dummy_len;
    EPETRA_CHK_ERR(A.ExtractGlobalRowCopy(gid,len,dummy_len,values,indices));
    // set row to 0 and diagonal to 1
    for (int i=0;i<len;i++)
      {
      if (indices[i]==gid)
        {
        values[i]=1.0;
        }
      else
        {
        values[i]=0.0;
        }
      // put it back in
      EPETRA_CHK_ERR(A.ReplaceGlobalValues(gid,len,values,indices));
      }
    }

  // broadcast indices to everyone
  EPETRA_CHK_ERR(A.Comm().Broadcast(indices,len,pid));
  
  // we assume that the pattern of the matrix is symmetric and process all the rows in 
  // indices, setting any coupling to gid to 0
  int *indices_i;
  double *values_i;
  int len_i;
  for (int i=0;i<len;i++)
    {
    int grid = indices[i];
    if (A.RowMap().MyGID(grid))
      {
      if (grid!=gid)
        {
        int lrid = A.LRID(grid);
        EPETRA_CHK_ERR(A.ExtractMyRowView(lrid,len_i,values_i,indices_i));
        for (int j=0;j<len_i;j++)
          {
          if (A.GCID(indices_i[j])==gid)
            {
            values_i[j]=0.0;
            }
          }
        }
      }
    }
  
    
  return 0;
  }


int MatrixUtils::FillReducingOrdering(const Epetra_CrsMatrix& Matrix,
                                             Teuchos::Array<int>& global_ordering,
                                             Teuchos::ParameterList& probDescription)
  {
  int dim=probDescription.get("Dimension",-1);
  int dof=probDescription.get("Degrees of Freedom",-1);
  string presType="none";
  int pres=dim;
  if (dof>1)
    {    
    presType=probDescription.get
        ("Variable Type ("+Teuchos::toString(pres)+")","undefined");
    }
  if (dim==-1 || dof==-1 || presType=="undefined")
    {
    Tools::Error("'Problem Definition' list incorrect",__FILE__,__LINE__);
    }
  Teuchos::RCP<Epetra_CrsMatrix> tmpMatrix;
  Teuchos::RCP<const Epetra_CrsGraph> graph;
  Teuchos::RCP<Epetra_Map> map1, map2;
  Teuchos::RCP<Epetra_Map> colmap1, colmap2;
  
  if (presType!="Retain 1")
    {
    Tools::Error("only intended for a special class of saddle-point problems",
        __FILE__,__LINE__);
    }

  // we assume that the matrix A 
  std::cout<<"Reordering based on graph of A+BB'"<<std::endl;

  // create the graph of A+BB', where B is the Div operator
    
  // 1) create row/colmaps of A and B
    
  // a) row maps
  int numel1=0;
  int numel2=0;
  for (int i=0;i<Matrix.NumMyRows();i++)
    {
    if (MOD(Matrix.GRID(i),dof)==pres)
      {
      numel2++;
      }
    else
      {
      numel1++;
      }
    }
  int *myElements1=new int[numel1];
  int *myElements2=new int[numel2];
    
  int pos1=0;
  int pos2=0;
    
  for (int i=0;i<Matrix.NumMyRows();i++)
    {
    if (MOD(Matrix.GRID(i),dof)==pres)
      {
      myElements2[pos2++]=Matrix.GRID(i);
      }
    else
      {
      myElements1[pos1++]=Matrix.GRID(i);
      }
    }

  int base=Matrix.RowMap().IndexBase();
  const Epetra_Comm& comm=Matrix.Comm();
    
  map1=Teuchos::rcp(new Epetra_Map(-1,numel1,myElements1, base,comm));
  map2=Teuchos::rcp(new Epetra_Map(-1,numel2,myElements2, base,comm));
    
  delete [] myElements1;
  delete [] myElements2;

  // b) create column maps
  const Epetra_Map& colmap = Matrix.ColMap();

  numel1=0;
  numel2=0;
  for (int i=0; i<colmap.NumMyElements();i++)
    {
    if (MOD(colmap.GID(i),dof)==pres)
      {
      numel2++;
      }
    else
      {
      numel1++;
      }
    }
  myElements1=new int[numel1];
  myElements2=new int[numel2];
  
  pos1=0;
  pos2=0;

  for (int i=0; i<colmap.NumMyElements();i++)
    {
    if (MOD(colmap.GID(i),dof)==pres)
      {
      myElements2[pos2++]=colmap.GID(i);
      }
    else
      {
      myElements1[pos1++]=colmap.GID(i);
      }
    }
    
  colmap1=Teuchos::rcp(new Epetra_Map(-1,numel1,myElements1, base,comm));
  colmap2=Teuchos::rcp(new Epetra_Map(-1,numel2,myElements2, base,comm));
    
  delete [] myElements1;
  delete [] myElements2;

  // c) create a copy of the matrices A, B (=Div), and B' (=Grad)
    
  Epetra_CrsMatrix A(Copy,*map1,*colmap1,Matrix.MaxNumEntries());
  Epetra_CrsMatrix B(Copy,*map2,*colmap1,Matrix.MaxNumEntries());
  Epetra_CrsMatrix Bt(Copy,*map1,*colmap2,Matrix.MaxNumEntries());

  Epetra_Import import1(Matrix.RowMap(),*map1);
  Epetra_Import import2(Matrix.RowMap(),*map2);
  
  CHECK_ZERO(A.Export(Matrix,import1,Insert));
  CHECK_ZERO(B.Export(Matrix,import2,Insert));
  CHECK_ZERO(Bt.Export(Matrix,import1,Insert));
  
  CHECK_ZERO(A.FillComplete(*map1,*map1));
  CHECK_ZERO(B.FillComplete(*map1,*map2));
  CHECK_ZERO(Bt.FillComplete(*map2,*map1));

MatrixUtils::Dump(*map1,"map1.txt");
MatrixUtils::Dump(*map2,"map2.txt");
MatrixUtils::Dump(A,"A.txt");
MatrixUtils::Dump(B,"B.txt");
MatrixUtils::Dump(Bt,"Bt.txt");
  
  tmpMatrix=Teuchos::rcp(new Epetra_CrsMatrix(Copy,*map1,A.MaxNumEntries()));
  CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(Bt,false,B,false,*tmpMatrix,false));
  CHECK_ZERO(EpetraExt::MatrixMatrix::Add(A, false, 1.0, *tmpMatrix, 1.0));
  CHECK_ZERO(tmpMatrix->FillComplete());
  tmpMatrix=DropByValue(tmpMatrix,1e-12,Relative);

//MatrixUtils::Dump(*tmpMatrix,"tmpMatrix.txt");
std::cout << "call Isorropia to reorder"<<std::endl;
  Teuchos::ParameterList params;
  //Teuchos::ParameterList zList=params.sublist("Zoltan");
  // set Zoltan/ParMETIS   parameters
  //zList.set(...);
  
  // reindex the matrix
  Teuchos::RCP<Epetra_Map> linearMap = Teuchos::rcp(new 
        Epetra_Map(tmpMatrix->NumGlobalRows(),
                   tmpMatrix->NumMyRows(),
                   0, tmpMatrix->Comm()) );

  Teuchos::RCP<EpetraExt::CrsMatrix_Reindex> reindex = Teuchos::rcp(new 
        EpetraExt::CrsMatrix_Reindex(*linearMap));

  Teuchos::RCP<Epetra_CrsMatrix> linearMatrix = 
        Teuchos::rcp(&((*reindex)(*tmpMatrix)),false);

//  graph = Teuchos::rcp(&(tmpMatrix->Graph()), false);
  graph = Teuchos::rcp(&(linearMatrix->Graph()), false);

  //TODO: this causes an exception!!!
  Isorropia::Epetra::Orderer reorder(graph,params);

  // now we have an ordering for A+BB', add in the pressures at the
  // right positions
  const int* velocity_ordering;
  int len;
  EPETRA_CHK_ERR(reorder.extractPermutationView(len,velocity_ordering));
  if (len!=map1->NumMyElements())
    {
    Tools::Error("in- and output array size inconsistent?",
      __FILE__,__LINE__);
    }


  // if ordering[i]=j, MUMPS will use the variable with LID i as j'th pivot.

  // go through the Div-matrix, and add each pressure in the ordering
  // just before the first velocity it couples to. That way we should
  // force the direct solver to form a 2x2 pivot with the first connected
  // velocity when encountering the pressure (which has a zero on the diagonal)


  // gather the Div matrix
  Teuchos::RCP<Epetra_CrsMatrix> Grad = Gather(Bt,0);

  // collect the current ordering
  Epetra_IntVector ivec1(A.RowMap());
  for (int i=0;i<ivec1.MyLength();i++)
    {
    ivec1[i]=velocity_ordering[i];
    }
  Teuchos::RCP<Epetra_IntVector> ord1 =
      MatrixUtils::Gather(ivec1,0);

#ifdef TESTING
MatrixUtils::Dump(*linearMatrix,"InputMatrixReorderer.txt");
MatrixUtils::Dump(*ord1, "ordering1.txt");
#endif

  if (Matrix.Comm().MyPID()==0)
    {
    Teuchos::Array<bool> in_ordering(Matrix.NumGlobalRows());

    Teuchos::Array<Teuchos::Array<int> > iord(Grad->NumMyRows());
    
    for (int i=0;i<Matrix.NumGlobalRows();i++)
      {
      in_ordering[i]=false;
      }

    for (int i=0;i<Grad->NumMyRows();i++)
      {
      in_ordering[Grad->GRID(i)]=true;
      }
    
    for (int i=0;i<Grad->NumMyRows();i++)
      {
      iord[i].resize(3);
      }

    for (int i=0;i<Grad->NumMyRows();i++)
      {
      int grid = Grad->GRID(i);
      int idx=(*ord1)[i];
      iord[idx][0]=grid;
      
      iord[idx][1]=-1;
      iord[idx][2]=-1;


      int *indices;
      double* values;
      int len2;
      CHECK_ZERO(Grad->ExtractMyRowView(grid,len2,values,indices));
      int pos=0;
      for (int j=0;j<len2;j++)
        {
        int gcid = Grad->GCID(j);
        if (!(in_ordering[gcid]))
          {
          iord[idx][pos++]=gcid;
          if (pos>=2) break; //safety
          }
        }
      }
    // construct the inverse ordering
    Teuchos::Array<int> iord_glob(Matrix.NumGlobalRows());
    int pos=0;
    for (int i=0;i<iord.size();i++)
      {
      for (int j=0;j<iord[i].size();j++)
        {
        if (iord[i][j]>=0)
          {
          iord_glob[pos++]=iord[i][j];
          }
        }
      }
    global_ordering.resize(Matrix.NumGlobalRows());
    for (int i=0;i<Matrix.NumGlobalRows();i++)
      {
      global_ordering[iord_glob[i]]=i;
      }
    }
  return 0;  
  }

    Teuchos::RCP<Epetra_Map> 
    MatrixUtils::CreateMap(int i0, int i1, int j0, int j1, int k0, int k1,        
                           int I0, int I1, int J0, int J1, int K0, int K1,
                           const Epetra_Comm& comm)
      {
      Teuchos::RCP<Epetra_Map> result = Teuchos::null;
      
      DEBUG("MatrixUtils::CreateMap ");
      DEBUG("["<<i0<<".."<<i1<<"]");
      DEBUG("["<<j0<<".."<<j1<<"]");
      DEBUG("["<<k0<<".."<<k1<<"]");
      
      int n = i1-i0+1; int N=I1-I0+1;
      int m = j1-j0+1; int M=J1-J0+1;
      int l = k1-k0+1; int L=K1-K0+1;
      
      DEBVAR(M);
      DEBVAR(N);
      DEBVAR(L);
      
      int NumMyElements = n*m*l;
      int NumGlobalElements = -1; // note that there may be overlap
      int *MyGlobalElements = new int[NumMyElements];
      
      int pos = 0;
      for (int k=k0; k<=k1; k++)
        for (int j=j0; j<=j1; j++)
          for (int i=i0; i<=i1; i++)
            {
            MyGlobalElements[pos++] = k*N*M + j*N + MOD((double)i,(double)N);
            }
      result = Teuchos::rcp(new Epetra_Map(NumGlobalElements,
                NumMyElements,MyGlobalElements,0,comm));
      delete [] MyGlobalElements;
      return result;
      }
  

}

