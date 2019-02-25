/**********************************************************************
 * Copyright by Jonas Thies, Univ. of Groningen 2006/7/8.             *
 * Permission to use, copy, modify, redistribute is granted           *
 * as long as this header remains intact.                             *
 * contact: jonas@math.rug.nl                                         *
 **********************************************************************/

#include "Trilinos_version.h"

#include "Teuchos_ParameterList.hpp"
#include "Epetra_Map.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_Vector.h"
#include "Epetra_IntVector.h"
#include "Epetra_Import.h"
#include "Epetra_Util.h"
#include "HYMLS_MatrixUtils.H"
#include "Epetra_Comm.h"
#include "EpetraExt_MatrixMatrix.h"

#include "Teuchos_FancyOStream.hpp"

#ifdef HYMLS_USE_PHIST
// reproducible random number generator from phist, very useful for tests etc.
extern "C" void phist_Dmvec_random(void* v, int* iflag);
#endif

// for sorting indices
#include <algorithm>

#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#include <mpi.h>
#endif

#include "HYMLS_Tools.H"

#include "Teuchos_StandardCatchMacros.hpp"
#ifdef HAVE_HDF5
#include "EpetraExt_HDF5.h"
#endif
#include "EpetraExt_Reindex_CrsMatrix.h"
#include "EpetraExt_Reindex_MultiVector.h"
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_VectorOut.h"
#include "EpetraExt_MultiVectorIn.h"
#include "EpetraExt_MultiVectorOut.h"
#include "EpetraExt_BlockMapOut.h"
#ifdef HAVE_METIS
#include "Zoltan_config.h"
#include "Isorropia_EpetraOrderer.hpp"
#endif
#include "AnasaziBlockKrylovSchurSolMgr.hpp"
#include "AnasaziBasicEigenproblem.hpp"
#include "AnasaziEpetraAdapter.hpp"

#include <fstream>

#if TRILINOS_MAJOR_MINOR_VERSION>=121200
#include "trilinos_amd.h"
#else
#include "amesos_amd.h"
#define TRILINOS_AMD_INFO AMD_INFO
#define trilinos_amd_order amesos_amd_order
#endif
// ASCII output formatting

// unfortunately Epetra sets the width to 20 so we can't really change that here.
#define OUTPUT_WIDTH 20
// to keep the output human readable we only use 12 digits behind the comma, so there
// still is white space between multivector entries.
#define OUTPUT_PREC 12

namespace HYMLS
  {
// create an optimal column map for extracting A(rowMap, colMap), given a distributed
// column map which has entries owned by other procs that we need for the column map.
Teuchos::RCP<Epetra_Map> MatrixUtils::CreateColMap(const Epetra_CrsMatrix& A,
  const Epetra_Map& newRows, const Epetra_Map& newCols)
  {
  HYMLS_PROF3(Label(), "CreateColMap");

  if (!A.HaveColMap()) Tools::Error("Matrix has no column map!", __FILE__, __LINE__);

  const Epetra_Map& old_map = A.ColMap();

  // build a test vector based on the old colmap and fill it with 0s
  Epetra_IntVector test1(old_map);
  test1.PutValue(0);

  // build a test vector based on the new columns and fill it with ones
  Epetra_IntVector test2(newCols);
  test2.PutValue(1);

  // import / add to see which of the previously owned cols we still need
  Epetra_Import import(old_map, newCols);

  CHECK_ZERO(test1.Import(test2, import, Add));
  int numel = 0;
  for (int i = 0; i < test1.MyLength(); i++) if (test1[i]) numel++;

  hymls_gidx *my_gids = new hymls_gidx[numel];
  int pos = 0;
  for (int i = 0; i < test1.MyLength(); i++)
    {
    if (test1[i])
      {
      my_gids[pos++] = old_map.GID64(i);
      }
    }
  Teuchos::RCP<Epetra_Map> new_map =
    Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1),
        numel, my_gids, (hymls_gidx)old_map.IndexBase64(),
        old_map.Comm()));
  delete [] my_gids;
  return new_map;
  }


// create "Gather" map from "Solve" map
Teuchos::RCP<Epetra_BlockMap> MatrixUtils::Gather(const Epetra_BlockMap& map, int root)
  {
  HYMLS_PROF3(Label(), "Gather (1)");
  int ElementSize = map.ElementSize();
#ifdef HYMLS_TESTING
  if (ElementSize != 1)
    {
    HYMLS_DEBVAR(ElementSize);
    Tools::Warning("this is possibly not implemented correctly!",
      __FILE__, __LINE__);
    ElementSize = 1;
    }
#endif
  int NumMyElements = map.NumMyElements();
  hymls_gidx NumGlobalElements = map.NumGlobalElements64();
  const Epetra_Comm& Comm = map.Comm();
  hymls_gidx *MyGlobalElements = new hymls_gidx[NumMyElements];
  hymls_gidx *AllGlobalElements = NULL;

  for (int i = 0; i < NumMyElements; i++)
    {
    MyGlobalElements[i] = map.GID64(i);
    }

  if (Comm.MyPID() == root)
    {
    AllGlobalElements = new hymls_gidx[NumGlobalElements];
    }

  if (Comm.NumProc() > 1)
    {
#ifdef HAVE_MPI
    const Epetra_MpiComm MpiComm = dynamic_cast < const Epetra_MpiComm& > (Comm);
    int *counts, *disps;
    counts = new int[Comm.NumProc()];
    disps = new int[Comm.NumProc()+1];

    MPI_Gather(&NumMyElements, 1, MPI_INTEGER,
      counts, 1, MPI_INTEGER, root, MpiComm.GetMpiComm());

    if (Comm.MyPID() == root)
      {
      disps[0] = 0;
      for (int p = 0; p < Comm.NumProc(); p++)
        {
        disps[p+1] = disps[p] + counts[p];
        }
      }

#ifdef HYMLS_LONG_LONG
    MPI_Gatherv(MyGlobalElements, NumMyElements, MPI_LONG_LONG_INT,
      AllGlobalElements, counts, disps, MPI_LONG_LONG_INT, root, MpiComm.GetMpiComm());
#else
    MPI_Gatherv(MyGlobalElements, NumMyElements, MPI_INTEGER,
      AllGlobalElements, counts, disps, MPI_INTEGER, root, MpiComm.GetMpiComm());
#endif
    delete [] counts;
    delete [] disps;
#else
    Tools::Error("No MPI but still parallel??? We don't do that.", __FILE__, __LINE__);
#endif
    }
  else
    {
    for (int i = 0; i < NumMyElements; i++) AllGlobalElements[i] = MyGlobalElements[i];
    }
  if (Comm.MyPID() != root)
    {
    NumMyElements = 0;
    }
  else
    {
    NumMyElements = NumGlobalElements;
    Teuchos::ArrayView<hymls_gidx> view(AllGlobalElements, NumGlobalElements);
    std::sort(view.begin(), view.end());
    Teuchos::ArrayView<hymls_gidx>::iterator new_end = std::unique(view.begin(), view.end());
    NumMyElements = std::distance(view.begin(), new_end);
    NumGlobalElements = NumMyElements;
    }
  CHECK_ZERO(Comm.Broadcast(&NumGlobalElements, 1, root));
  // build the new (gathered) map
  Teuchos::RCP<Epetra_BlockMap> gmap = Teuchos::rcp(new Epetra_BlockMap
      (NumGlobalElements, NumMyElements, AllGlobalElements,
      ElementSize, (hymls_gidx)map.IndexBase64(), Comm) );


  if (Comm.MyPID() == root)
    {
    delete [] AllGlobalElements;
    }


  delete [] MyGlobalElements;

  return gmap;
  }


// create "col" map from "Solve" map
Teuchos::RCP<Epetra_BlockMap> MatrixUtils::AllGather(const Epetra_BlockMap& map, bool reorder)
  {
  HYMLS_PROF3(Label(), "AllGather (1)");
  int ElementSize = map.ElementSize();
  int NumMyElements = map.NumMyElements();
  hymls_gidx NumGlobalElements = map.NumGlobalElements64();
  const Epetra_Comm& Comm = map.Comm();

#ifdef HYMLS_TESTING
  if (ElementSize > 1)
    {
    Tools::Warning("this is possibly not implemented correctly!",
      __FILE__, __LINE__);
    }
#endif

  hymls_gidx *MyGlobalElements = new hymls_gidx[NumMyElements];
  hymls_gidx *AllGlobalElements = new hymls_gidx[NumGlobalElements];

  for (int i = 0; i < NumMyElements; i++)
    {
    MyGlobalElements[i] = map.GID64(i);
    }

  if (Comm.NumProc() > 1)
    {
#ifdef HAVE_MPI
    const Epetra_MpiComm MpiComm = dynamic_cast < const Epetra_MpiComm& > (Comm);
    int *counts, *disps;
    counts = new int[Comm.NumProc()];
    disps = new int[Comm.NumProc()+1];

    MPI_Allgather(&NumMyElements, 1, MPI_INTEGER,
      counts, 1, MPI_INTEGER, MpiComm.GetMpiComm());

    disps[0] = 0;
    for (int p = 0; p < Comm.NumProc(); p++)
      {
      disps[p+1] = disps[p] + counts[p];
      }

#ifdef HYMLS_LONG_LONG
    MPI_Allgatherv(MyGlobalElements, NumMyElements, MPI_LONG_LONG_INT,
      AllGlobalElements, counts, disps, MPI_LONG_LONG_INT, MpiComm.GetMpiComm());
#else
    MPI_Allgatherv(MyGlobalElements, NumMyElements, MPI_INTEGER,
      AllGlobalElements, counts, disps, MPI_INTEGER, MpiComm.GetMpiComm());
#endif
    delete [] counts;
    delete [] disps;
#else
    Tools::Error("No MPI but still parallel? We don't do tthat.", __FILE__, __LINE__);
#endif
    }
  else
    {
    for (int i = 0; i < NumMyElements; i++) AllGlobalElements[i] = MyGlobalElements[i];
    }

  NumMyElements = NumGlobalElements;
  NumGlobalElements = -1;

  if (reorder)
    {
    std::sort(AllGlobalElements, AllGlobalElements + NumMyElements);
    }

  // build the new (gathered) map
  Teuchos::RCP<Epetra_BlockMap> gmap = Teuchos::rcp(new Epetra_BlockMap (NumGlobalElements, NumMyElements,
      AllGlobalElements, ElementSize, (hymls_gidx)map.IndexBase64(), Comm) );

  delete [] MyGlobalElements;
  delete [] AllGlobalElements;

  return gmap;
  }// AllGather

// create "Gather" map from "Solve" map
Teuchos::RCP<Epetra_Map> MatrixUtils::Gather(const Epetra_Map& map, int root)
  {
  HYMLS_PROF3(Label(), "Gather (2)");
  int NumMyElements = map.NumMyElements();
  hymls_gidx NumGlobalElements = map.NumGlobalElements64();
  const Epetra_Comm& Comm = map.Comm();

  hymls_gidx *MyGlobalElements = new hymls_gidx[NumMyElements];
  hymls_gidx *AllGlobalElements = NULL;

  for (int i = 0; i < NumMyElements; i++)
    {
    MyGlobalElements[i] = map.GID64(i);
    }

  if (Comm.MyPID() == root)
    {
    AllGlobalElements = new hymls_gidx[NumGlobalElements];
    }

  if (Comm.NumProc() > 1)
    {
#ifdef HAVE_MPI

    const Epetra_MpiComm MpiComm = dynamic_cast < const Epetra_MpiComm& > (Comm);
    int *counts, *disps;
    counts = new int[Comm.NumProc()];
    disps = new int[Comm.NumProc()+1];

    MPI_Gather(&NumMyElements, 1, MPI_INTEGER,
      counts, 1, MPI_INTEGER, root, MpiComm.GetMpiComm());

    if (Comm.MyPID() == root)
      {
      disps[0] = 0;
      for (int p = 0; p < Comm.NumProc(); p++)
        {
        disps[p+1] = disps[p] + counts[p];
        }
      }

#ifdef HYMLS_LONG_LONG
    MPI_Gatherv(MyGlobalElements, NumMyElements, MPI_LONG_LONG_INT,
      AllGlobalElements, counts, disps, MPI_LONG_LONG_INT, root, MpiComm.GetMpiComm());
#else
    MPI_Gatherv(MyGlobalElements, NumMyElements, MPI_INTEGER,
      AllGlobalElements, counts, disps, MPI_INTEGER, root, MpiComm.GetMpiComm());
#endif
    delete [] counts;
    delete [] disps;
#else
    Tools::Error("No MPI but still parallel??? We don't do that.", __FILE__, __LINE__);
#endif
    }
  else
    {
    for (int i = 0; i < NumMyElements; i++) AllGlobalElements[i] = MyGlobalElements[i];
    }
  if (Comm.MyPID() != root)
    {
    NumMyElements = 0;
    }
  else
    {
    NumMyElements = NumGlobalElements;
    std::sort(AllGlobalElements, AllGlobalElements + NumMyElements);
    auto end = std::unique(AllGlobalElements, AllGlobalElements + NumMyElements);
    NumMyElements = std::distance(AllGlobalElements, end);
    NumGlobalElements = NumMyElements;
    }
  CHECK_ZERO(Comm.Broadcast(&NumGlobalElements, 1, root));

  // build the new (gathered) map
  Teuchos::RCP<Epetra_Map> gmap = Teuchos::rcp(new Epetra_Map
      (NumGlobalElements, NumMyElements, AllGlobalElements,
      (hymls_gidx)map.IndexBase64(), Comm) );

  if (Comm.MyPID() == root)
    {
    delete [] AllGlobalElements;
    }

  delete [] MyGlobalElements;

  return gmap;
  }


// create "col" map from "Solve" map
Teuchos::RCP<Epetra_Map> MatrixUtils::AllGather(const Epetra_Map& map, bool reorder)
  {
  HYMLS_PROF3(Label(), "AllGather (2)");
  int NumMyElements = map.NumMyElements();
  hymls_gidx NumGlobalElements = map.NumGlobalElements64();
  const Epetra_Comm& Comm = map.Comm();

  hymls_gidx *MyGlobalElements = new hymls_gidx[NumMyElements];
  hymls_gidx *AllGlobalElements = new hymls_gidx[NumGlobalElements];

  for (hymls_gidx i = 0; i < NumMyElements; i++)
    {
    MyGlobalElements[i] = map.GID64(i);
    }

  if (Comm.NumProc() > 1)
    {
#ifdef HAVE_MPI
    const Epetra_MpiComm MpiComm = dynamic_cast < const Epetra_MpiComm& > (Comm);
    int *counts, *disps;
    counts = new int[Comm.NumProc()];
    disps = new int[Comm.NumProc()+1];

    MPI_Allgather(&NumMyElements, 1, MPI_INTEGER,
      counts, 1, MPI_INTEGER, MpiComm.GetMpiComm());

    disps[0] = 0;
    for (int p = 0; p < Comm.NumProc(); p++)
      {
      disps[p+1] = disps[p] + counts[p];
      }

#ifdef HYMLS_LONG_LONG
    MPI_Allgatherv(MyGlobalElements, NumMyElements, MPI_LONG_LONG_INT,
      AllGlobalElements, counts, disps, MPI_LONG_LONG_INT, MpiComm.GetMpiComm());
#else
    MPI_Allgatherv(MyGlobalElements, NumMyElements, MPI_INTEGER,
      AllGlobalElements, counts, disps, MPI_INTEGER, MpiComm.GetMpiComm());
#endif
    delete [] counts;
    delete [] disps;
#else
    Tools::Error("No MPI but still parallel? We don't do tthat.", __FILE__, __LINE__);
#endif
    }
  else
    {
    for (int i = 0; i < NumMyElements; i++) AllGlobalElements[i] = MyGlobalElements[i];
    }

  NumMyElements = NumGlobalElements;
  NumGlobalElements = -1;

  if (reorder)
    {
    std::sort(AllGlobalElements, AllGlobalElements + NumMyElements);
    }

  // build the new (gathered) map
  Teuchos::RCP<Epetra_Map> gmap = Teuchos::rcp(new Epetra_Map (NumGlobalElements, NumMyElements,
      AllGlobalElements, (hymls_gidx)map.IndexBase64(), Comm) );

  delete [] MyGlobalElements;
  delete [] AllGlobalElements;

  return gmap;
  }// AllGather

Teuchos::RCP<Epetra_MultiVector> MatrixUtils::Gather(const Epetra_MultiVector& vec, int root)
  {
  HYMLS_PROF3(Label(), "Gather (3)");
  const Epetra_BlockMap& map_dist = vec.Map();
  Teuchos::RCP<Epetra_BlockMap> map = Gather(map_dist, root);
  Teuchos::RCP<Epetra_MultiVector> gvec =
    Teuchos::rcp(new Epetra_MultiVector(*map, vec.NumVectors()));
  Teuchos::RCP<Epetra_Import> import = Teuchos::rcp(new Epetra_Import(*map, map_dist) );
  CHECK_ZERO(gvec->Import(vec, *import, Insert));
  gvec->SetLabel(vec.Label());
  return gvec;
  }

Teuchos::RCP<Epetra_MultiVector> MatrixUtils::AllGather(const Epetra_MultiVector& vec)
  {
  HYMLS_PROF3(Label(), "AllGather (3)");
  const Epetra_BlockMap& map_dist = vec.Map();
  Teuchos::RCP<Epetra_BlockMap> map = AllGather(map_dist);
  Teuchos::RCP<Epetra_MultiVector> gvec = Teuchos::rcp(new Epetra_Vector(*map, vec.NumVectors()));

  Teuchos::RCP<Epetra_Import> import = Teuchos::rcp(new Epetra_Import(*map, map_dist) );

  CHECK_ZERO(gvec->Import(vec, *import, Insert));

  gvec->SetLabel(vec.Label());
  return gvec;

  }

Teuchos::RCP<Epetra_IntVector> MatrixUtils::Gather
(const Epetra_IntVector& vec, int root)
  {
  HYMLS_PROF3(Label(), "Gather (4)");
  const Epetra_BlockMap& map_dist = vec.Map();
  Teuchos::RCP<Epetra_BlockMap> map = Gather(map_dist, root);

  Teuchos::RCP<Epetra_IntVector> gvec = Teuchos::rcp(new Epetra_IntVector(*map));

  Teuchos::RCP<Epetra_Import> import = Teuchos::rcp(new Epetra_Import(*map, map_dist) );

  CHECK_ZERO(gvec->Import(vec, *import, Insert));

  gvec->SetLabel(vec.Label());

  return gvec;

  }

Teuchos::RCP<Epetra_IntVector> MatrixUtils::AllGather(const Epetra_IntVector& vec)
  {
  HYMLS_PROF3(Label(), "AllGather (4)");
  const Epetra_BlockMap& map_dist = vec.Map();
  Teuchos::RCP<Epetra_BlockMap> map = AllGather(map_dist);
  Teuchos::RCP<Epetra_IntVector> gvec = Teuchos::rcp(new Epetra_IntVector(*map));

  Teuchos::RCP<Epetra_Import> import = Teuchos::rcp(new Epetra_Import(*map, map_dist) );

  CHECK_ZERO(gvec->Import(vec, *import, Insert));

  gvec->SetLabel(vec.Label());
  return gvec;
  }

Teuchos::RCP<Epetra_CrsMatrix> MatrixUtils::Gather(const Epetra_CrsMatrix& mat, int root)
  {
  HYMLS_PROF3(Label(), "Gather (5)");
  const Epetra_Map& rowmap_dist = mat.RowMap();
  // we take the domain map as the colmap is potentially overlapping
  const Epetra_Map& colmap_dist = mat.DomainMap();
  // gather the row map
  Teuchos::RCP<Epetra_Map> rowmap = Gather(rowmap_dist, root);
  // gather the col map
  Teuchos::RCP<Epetra_Map> colmap = Gather(colmap_dist, root);

  // we only guess the number of row entries, this routine is not performance critical
  // as it should only be used for debugging anyway
  int num_entries = mat.NumGlobalNonzeros64() / mat.NumGlobalRows64();
  Teuchos::RCP<Epetra_CrsMatrix> gmat = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *rowmap, *colmap, num_entries) );

  Teuchos::RCP<Epetra_Import> import = Teuchos::rcp(new Epetra_Import(*rowmap, rowmap_dist) );

  CHECK_ZERO(gmat->Import(mat, *import, Insert));

  CHECK_ZERO(gmat->FillComplete());
  gmat->SetLabel(mat.Label());

  return gmat;

  }

// distribute a gathered vector among processors
Teuchos::RCP<Epetra_MultiVector> MatrixUtils::Scatter
(const Epetra_MultiVector& vec, const Epetra_BlockMap& distmap)
  {
  HYMLS_PROF3(Label(), "Scatter (1)");
  Teuchos::RCP<Epetra_MultiVector> dist_vec = Teuchos::rcp(new Epetra_MultiVector(distmap, vec.NumVectors()));
  Teuchos::RCP<Epetra_Import> import = Teuchos::rcp(new Epetra_Import(vec.Map(), distmap));
  CHECK_ZERO(dist_vec->Export(vec, *import, Insert));
  return dist_vec;
  }

// write CRS matrix to file
void MatrixUtils::Dump(const Epetra_CrsMatrix& A, const std::string& filename,
  bool reindex, PrintMethod how)
  {
  HYMLS_PROF3(Label(), "Dump (1)");
  HYMLS_DEBUG("Matrix with label " << A.Label() << " is written to file " << filename);

  if (reindex)
    {
    Teuchos::RCP<Epetra_Map> newMap;
    int myLength = A.NumMyRows();
    newMap = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1), myLength, 0, A.Comm()));
    EpetraExt::CrsMatrix_Reindex renumber(*newMap);
    Dump(renumber(const_cast < Epetra_CrsMatrix& > (A)), filename, false, how);
    }
  else
    {
    if (how == MATRIXMARKET)
      {
      CHECK_ZERO(EpetraExt::RowMatrixToMatrixMarketFile(filename.c_str(), A));
      }
    else if (how == GATHER)
      {
      Teuchos::RCP<std::ostream> ofs = Teuchos::rcp(new Teuchos::oblackholestream());
      int my_rank = A.Comm().MyPID();
      if (my_rank == 0)
        {
        ofs = Teuchos::rcp(new std::ofstream(filename.c_str(), std::ios::trunc));
        }
      *ofs << std::scientific << std::setw(OUTPUT_WIDTH) << std::setprecision(OUTPUT_PREC);
      *ofs << *(MatrixUtils::Gather(A, 0));
      }
    else
      {
      Tools::Error("not implemented", __FILE__, __LINE__);
      }
    }
  return;
  }

void MatrixUtils::DumpHDF(const Epetra_CrsMatrix& A,
  const std::string& filename,
  const std::string& groupname,
  bool new_file)
  {
  HYMLS_PROF3(Label(), "DumpHDF (1)");
#ifndef HAVE_HDF5
  Tools::Error("HDF format can't be stored, recompile with -DHAVE_HDF5", __FILE__, __LINE__);
#else
  bool verbose = true;
  bool success;
  HYMLS_DEBUG("Matrix with label " << A.Label() << " is written to HDF5 file " << filename << ", group " << groupname);
  RCP<EpetraExt::HDF5> hdf5 = Teuchos::rcp(new EpetraExt::HDF5(A.Comm()));
  try {
    if (new_file)
      {
      hdf5->Create(filename.c_str());
      }
    else
      {
      hdf5->Open(filename.c_str());
      }
    hdf5->Write(groupname, A);
    hdf5->Close();
    } TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);
// give a warning and ignore the exception
  if (!success) Tools::Warning("caught an exception", __FILE__, __LINE__);
#endif
  }

// write vector / dense matrix to file
void MatrixUtils::Dump(const Epetra_MultiVector& x, const std::string& filename,
  bool reindex, PrintMethod how)
  {
  HYMLS_PROF3(Label(), "Dump (2)");
  if (reindex)
    {
    Teuchos::RCP<Epetra_Map> newMap;
    int myLength = x.MyLength();
    newMap = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1), myLength, 0, x.Comm()));
    EpetraExt::MultiVector_Reindex renumber(*newMap);
    Dump(renumber(const_cast < Epetra_MultiVector& > (x)), filename, false, how);
    }
  else
    {
    HYMLS_DEBUG("Vector with label " << x.Label() << " is written to file " << filename);
    if (how == MATRIXMARKET)
      {
      EpetraExt::MultiVectorToMatrixMarketFile(filename.c_str(), x);
      }
    else if (how == GATHER)
      {
      Teuchos::RCP<std::ostream> ofs = Teuchos::rcp(new Teuchos::oblackholestream());
      int my_rank = x.Comm().MyPID();
      if (my_rank == 0)
        {
        ofs = Teuchos::rcp(new std::ofstream(filename.c_str(), std::ios::trunc));
        }
      *ofs << std::scientific << std::setw(OUTPUT_WIDTH) << std::setprecision(OUTPUT_PREC);
      *ofs << *(MatrixUtils::Gather(x, 0));
      }
    else
      {
      Tools::Error("not implemented", __FILE__, __LINE__);
      }
    }
  }

// write CRS IntVector to file
void MatrixUtils::Dump(const Epetra_IntVector& x, const std::string& filename)
  {
  HYMLS_PROF3(Label(), "Dump (3)");
  HYMLS_DEBUG("Vector with label " << x.Label() << " is written to file " << filename);

  // EpetraExt::VectorToMatrixMarketFile(filename.c_str(), x);

  Teuchos::RCP<std::ostream> ofs = Teuchos::rcp(new Teuchos::oblackholestream());
  int my_rank = x.Comm().MyPID();
  if (my_rank == 0)
    {
    ofs = Teuchos::rcp(new std::ofstream(filename.c_str()));
    }
  *ofs << std::setw(OUTPUT_WIDTH) << std::setprecision(OUTPUT_PREC);
  *ofs << *(MatrixUtils::Gather(x, 0));
  }

int MatrixUtils::mmwrite(std::string filename, const Epetra_MultiVector& vec)
  {
  HYMLS_PROF2(Label(), "mmwrite");
  const Epetra_BlockMap& map = vec.Map();
  int myLength = vec.MyLength();
  hymls_gidx base = map.IndexBase64();
  Epetra_Map linearMap((hymls_gidx)(-1), myLength, base, vec.Comm());
  // the EpetraExt function here just creates a view of the vector with the linear map,
  // which means that first all entries on partition 0 are written, then partition 1 etc.
  // This destroys the ordering of the vector, however, so we do an import instead which
  // physically moves the GIDs 0:(nloc - 1) to proc 0, etc.
  Epetra_Import import(linearMap, map);
  Epetra_MultiVector linearVec(linearMap, vec.NumVectors());
  CHECK_ZERO(linearVec.Import(vec, import, Insert));
  CHECK_ZERO(EpetraExt::MultiVectorToMatrixMarketFile(filename.c_str(), linearVec));
  return 0;
  }

// ! MatrixMarket input of MultiVector.
int MatrixUtils::mmread(std::string filename, Epetra_MultiVector& vec)
  {
  HYMLS_PROF2(Label(), "mmread");
  const Epetra_BlockMap& map = vec.Map();
  int myLength = vec.MyLength();
  hymls_gidx base = map.IndexBase64();
  Epetra_Map linearMap((hymls_gidx)(-1), myLength, base, vec.Comm());
  Epetra_Import import(map, linearMap);

  Epetra_MultiVector *ptr;
  CHECK_ZERO(EpetraExt::MatrixMarketFileToMultiVector(filename.c_str(), linearMap, ptr));
  CHECK_ZERO(vec.Import(*ptr, import, Insert));
  delete ptr;

  return 0;
  }

void MatrixUtils::DumpHDF(const Epetra_MultiVector& x,
  const std::string& filename,
  const std::string& groupname,
  bool new_file)
  {
  HYMLS_PROF3(Label(), "DumpHDF (2)");
#ifndef HAVE_HDF5
  Tools::Error("HDF format can't be stored, recompile with -DHAVE_HDF5", __FILE__, __LINE__);
#else
  bool verbose = true;
  bool success;
  HYMLS_DEBUG("Vector with label " << x.Label() << " is written to HDF5 file " << filename << ", group   " << groupname);
  RCP<EpetraExt::HDF5> hdf5 = Teuchos::rcp(new EpetraExt::HDF5(x.Comm()));
  try {
    if (new_file)
      {
      hdf5->Create(filename.c_str());
      }
    else
      {
      hdf5->Open(filename.c_str());
      }
    hdf5->Write(groupname, x);
    hdf5->Close();
    } TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);
  if (!success) Tools::Warning("caught an exception", __FILE__, __LINE__);
#endif
  x.Comm().Barrier();
  }

// write map to file
void MatrixUtils::Dump(const Epetra_Map& M, const std::string& filename, PrintMethod how)
  {
  HYMLS_PROF3(Label(), "Dump (4)");
  HYMLS_DEBUG("Map with label " << M.Label() << " is written to file " << filename);
  if (how == MATRIXMARKET)
    {
    EpetraExt::BlockMapToMatrixMarketFile(filename.c_str(), M);
    }
  else if (how == GATHER)
    {
    Teuchos::RCP<std::ostream> ofs = Teuchos::rcp(new Teuchos::oblackholestream());
    int my_rank = M.Comm().MyPID();
    if (my_rank == 0)
      {
      ofs = Teuchos::rcp(new std::ofstream(filename.c_str()));
      }
    *ofs << std::setw(OUTPUT_WIDTH) << std::setprecision(OUTPUT_PREC);
    *ofs << *(MatrixUtils::Gather(M, 0));
    }
  else
    {
    Tools::Error("not implemented", __FILE__, __LINE__);
    }
  }

// print row matrix
void MatrixUtils::PrintRowMatrix(const Epetra_RowMatrix& A, std::ostream& os)
  {
  HYMLS_PROF3(Label(), "PrintRowMatrix");
  HYMLS_DEBUG("Print Row Matrix: " << A.Label());
  int nrows = A.NumMyRows();
  int ncols = A.NumMyCols();
  int nnz = A.NumMyNonzeros();
  hymls_gidx nrows_g = A.NumGlobalRows64();
  hymls_gidx nnz_g = A.NumGlobalNonzeros64();
  int maxlen = ncols;
  int len;
  int *indices = new int[maxlen];
  double *values = new double[maxlen];
  hymls_gidx grid, gcid;

  os << "% nloc nglob nnz_loc nnz_glob\n";
  os << nrows << " " << nrows_g << " " << nnz << " " << nnz_g << std::endl;
  os << std::scientific << std::setw(16) << std::setprecision(16);
  for (int i = 0; i < nrows; i++)
    {
    grid = A.RowMatrixRowMap().GID64(i);
    CHECK_ZERO(A.ExtractMyRowCopy(i, maxlen, len, values, indices));
    for (int j = 0; j < len; j++)
      {
      gcid = A.RowMatrixColMap().GID64(indices[j]);
      os << grid << "\t" << gcid << "\t" << values[j] << std::endl;
      }
    }
  delete [] indices;
  delete [] values;
  }

Teuchos::RCP<Anasazi::Eigensolution<double, Epetra_MultiVector> > MatrixUtils::Eigs(
  Teuchos::RCP<const Epetra_Operator> A,
  Teuchos::RCP<const Epetra_Operator> B,
  int howMany,
  double tol)
  {
  HYMLS_PROF2(Label(), "Eigs");

  typedef double ST;
  typedef Epetra_MultiVector MV;
  typedef Epetra_Operator OP;

  // ************************************
  // Start the block Arnoldi iteration
  // ***********************************
  //
  //  Variables used for the Block Krylov Schur Method
  //
  bool boolret;

  bool verbose = true;
  bool debug = false;
#ifdef HYMLS_TESTING
  verbose = true;
#endif
#ifdef HYMLS_DEBUGGING
  debug = true;
#endif

  std::string which("LM");

  int blockSize = 1;
  int numBlocks = 120;
  int stepSize = 10;
  int maxRestarts = 10;

  // Create a sort manager to pass into the block Krylov - Schur solver manager
  // -->Make sure the reference - counted pointer is of type Anasazi::SortManager <>
  // -->The block Krylov - Schur solver manager uses Anasazi::BasicSort <> by default,
  //      so you can also pass in the parameter "Which", instead of a sort manager.
//  Teuchos::RCP<Anasazi::SortManager < ST> > MySort =
//    Teuchos::rcp( new Anasazi::BasicSort<ST> ( which ) );

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
  MyPL.set( "Output Stream", Tools::getOutputStream());

//  MyPL.set( "Sort Manager", MySort );
  MyPL.set( "Which", which );
  MyPL.set( "Block Size", blockSize );
  MyPL.set( "Num Blocks", numBlocks );
  MyPL.set( "Maximum Restarts", maxRestarts );
  MyPL.set( "Step Size", stepSize );
  MyPL.set( "Convergence Tolerance", tol );

  // Create an Epetra_MultiVector for an initial vector to start the solver.
  // Note:  This needs to have the same number of columns as the blocksize.
  HYMLS_DEBUG("create random starting vector for Anasazi");
  Teuchos::RCP<Epetra_MultiVector> ivec =
    Teuchos::rcp( new Epetra_MultiVector(A->OperatorRangeMap(), blockSize) );
  MatrixUtils::Random(*ivec);

  Epetra_MultiVector tmp = *ivec;
  if (!Teuchos::is_null(B))
    {
    HYMLS_DEBUG("multiply it by the mass-matrix...");
    CHECK_ZERO(B->Apply(tmp, *ivec));
    }
  // Create the eigenproblem.
  HYMLS_DEBUG("create eigen-problem");
  Teuchos::RCP < Anasazi::BasicEigenproblem < ST, MV, OP > > MyProblem;

  // as of Trilinos 10.12 it doesn't seem to make any difference if we
  // pass in the mass matrix, so in our Solver class where we call this
  // function we provide M\B as operator and solve a standard EVP.
  if (Teuchos::is_null(B) || true)
    {
    MyProblem = Teuchos::rcp( new Anasazi::BasicEigenproblem < ST, MV, OP > (A, ivec) );
    }
  else
    {
    MyProblem = Teuchos::rcp( new Anasazi::BasicEigenproblem < ST, MV, OP > (A, B, ivec) );
    }

  // Inform the eigenproblem that the operator A is symmetric
  MyProblem->setHermitian(false);

  // Set the number of eigenvalues requested
  HYMLS_DEBVAR(howMany);
  MyProblem->setNEV( howMany );

  // Inform the eigenproblem that you are finishing passing it information
  HYMLS_DEBUG("call setProblem");
  boolret = MyProblem->setProblem();
  if (boolret != true)
    {
    Tools::Error("Anasazi::BasicEigenproblem::setProblem() returned with error.",
      __FILE__, __LINE__);
    }

  // Initialize the Block Arnoldi solver
  HYMLS_DEBUG("create BKS eigensolver");
  Anasazi::BlockKrylovSchurSolMgr < ST, MV, OP > MySolverMgr(MyProblem, MyPL);

  // Solve the problem to the specified tolerances or length
  Anasazi::ReturnType returnCode;
  HYMLS_DEBUG("solve eigenproblem");
  returnCode = MySolverMgr.solve();
  if (returnCode != Anasazi::Converged)
    {
    Tools::Warning("Anasazi::EigensolverMgr::solve() returned unconverged.",
      __FILE__, __LINE__);
    }

  HYMLS_DEBUG("post-process returned solution");
  // Get the Ritz values from the eigensolver
  std::vector<Anasazi::Value < double> > ritzValues = MySolverMgr.getRitzValues();

  if (verbose)
    {
    // Output computed eigenvalues and their direct residuals
    int numritz = (int)ritzValues.size();
//    Tools::out() << "operator: " << A->Label() << std::endl;
    Tools::out() << std::endl << "Computed Ritz Values" << std::endl;
    Tools::out() << std::setw(OUTPUT_WIDTH) << "Real Part"
                 << std::setw(OUTPUT_WIDTH) << "Imag Part"
                 << std::endl;
    Tools::out() << "-----------------------------------------------------------" << std::endl;
    for (int i = 0; i < numritz; i++)
      {
      Tools::out() << std::setw(OUTPUT_WIDTH) << ritzValues[i].realpart
                   << std::setw(OUTPUT_WIDTH) << ritzValues[i].imagpart
                   << std::endl;
      }
    Tools::out() << "-----------------------------------------------------------" << std::endl;
    }

  return Teuchos::rcp(new Anasazi::Eigensolution<double, Epetra_MultiVector>(MyProblem->getSolution()));
  }

int MatrixUtils::Random(Epetra_MultiVector& v, int seed)
  {
  HYMLS_PROF3(Label(), "Random");
#ifdef HYMLS_USE_PHIST
  if (seed == -1)
    {
    int iflag=0;
    // note: we can't set the seed in phist
    phist_Dmvec_random(&v,&iflag);
    return iflag;
    }
  else
    {
    Tools::Error("PHIST random does not accept a seed", __FILE__, __LINE__);
    }
#else
  const hymls_gidx len = v.GlobalLength64();
  Epetra_BlockMap const &map = v.Map();
  Epetra_Util util;
  // communicate the seed to be able to generate the same vector on all processors
  if (!(seed > 0))
    {
    seed = util.Seed();
    v.Comm().Broadcast(&seed, 1, 0);
    }
  util.SetSeed(seed);
  Tools::out() << "SEED: " << util.Seed() << std::endl;
  // generate a consistent random vector that is independent of the number of processors
  for (int j = 0; j < v.NumVectors(); j++)
    {
    for (hymls_gidx i = 0; i < len; i++)
      {
      // check if the value is on the current processor
      if (map.MyGID(i))
        {
        CHECK_ZERO(v.ReplaceGlobalValue(i, j, util.RandomDouble()));
        }
      else
        {
        // generate the next seed value. RandomInt is faster than RandomDouble
        util.RandomInt();
        }
      }
    }
#endif
  return 0;
  }

// drop small matrix entries (relative to diagonal element)
Teuchos::RCP<Epetra_CrsMatrix> MatrixUtils::DropByValue
(Teuchos::RCP<const Epetra_CrsMatrix> A, double droptol, DropType type)
  {
  HYMLS_PROF2(Label(), "DropByValue");

  // shortcut
  if (droptol == 0.0) return Teuchos::rcp_const_cast<Epetra_CrsMatrix> (A);

  int NumRows = A->NumMyRows();
  int *NumMyEntries = new int[NumRows];
  for (int i = 0; i < NumRows; i++)
    NumMyEntries[i] = A->NumMyEntries(i) + 1;

  Teuchos::RCP<Epetra_CrsMatrix> mat = Teuchos::rcp(
    new Epetra_CrsMatrix(Copy, A->RowMap(), A->ColMap(), NumMyEntries));

  delete[] NumMyEntries;

  // diagonal of A in column map
  Teuchos::RCP<Epetra_Vector> diagA;

  // should the drop tol be scaled with anything?
  bool rel = (type == Relative || type == RelDropDiag || type == RelZeroDiag ||  type == RelFullDiag);
  // if so, should an absolute dropping strategy be used on the diagonal?
  bool absDiag = (type == RelDropDiag || type == RelZeroDiag || type == RelFullDiag ||
    type == AbsZeroDiag || type == AbsFullDiag || type == Absolute);
  // should physical zeros be put on the diagonal where dropping occurs?
  bool zeroDiag = (type == RelZeroDiag || type == AbsZeroDiag);
  // should physical zeros be put on the diagonal where dropping occurs?
  bool fullDiag = (type == RelFullDiag || type == AbsFullDiag);

  diagA = Teuchos::rcp(new Epetra_Vector(A->RowMap()));
  CHECK_ZERO(A->ExtractDiagonalCopy(*diagA));
  // import diagA into the column map of A, we need this
  // in case we have to look for ajj when considering dropping
  // aij in a row with aii = 0.
  if (!A->HaveColMap())
    {
    Tools::Error("matrix has no col map, you may have to call FillComplete() first.", __FILE__, __LINE__);
    }
  if (A->Importer() != NULL)
    {
    Teuchos::RCP<Epetra_Vector> diagA_tmp = diagA;
    diagA = Teuchos::rcp(new Epetra_Vector(A->ColMap()));
    CHECK_ZERO(diagA->Import(*diagA_tmp, *A->Importer(), Insert));
    }
  else
    {
    if (A->RowMap().SameAs(A->ColMap()) == false)
      {
      Tools::Error("your matrix is suspicious, row map != col map, but no importer...",
        __FILE__, __LINE__);
      }
    }

  int len;
  int *indices;
  double *values;

  int new_len;
  hymls_gidx *new_indices = new hymls_gidx[A->MaxNumEntries()+1];
  double *new_values = new double[A->MaxNumEntries()+1];

  double scal = 1.0;
  double scal_i = 1.0;
  for (int i = 0; i < NumRows; i++)
    {
    CHECK_ZERO(A->ExtractMyRowView(i, len, values, indices));

    // this index trafo is required because diagA is based on the col map of A
    int lcid_i = diagA->Map().LID(A->GRID64(i));
    if (rel)
      {
#ifdef HYMLS_TESTING
      if (lcid_i < 0) Tools::Error("matrix is missing a column", __FILE__, __LINE__);
#endif
      scal_i = std::abs((*diagA)[lcid_i]);
      }

    new_len = 0;

    // If fullDiag we always want the diagonal entries
    if (fullDiag)
      {
      double diagValue = (*diagA)[lcid_i];
      new_values[new_len] = (std::abs(diagValue) > droptol) ? diagValue : 0.0;
      new_indices[new_len] = A->GRID64(i);
      new_len++;
      }

    for (int j = 0; j < len; j++)
      {
      bool isDiag = (A->GCID64(indices[j]) == A->GRID64(i));

      // We already did this
      if (isDiag && fullDiag)
        continue;

      scal = scal_i;

      if (isDiag && absDiag)
        {
        // use absolute tol on diagonal
        scal = 1.0;
        }
      else if (rel)
        {
        // for F - matrices with zeros on the diagonal, use tol*|ajj| instead
        // of tol*|aii|, this prevents loss of structural symmetry.
        int lcid_j = diagA->Map().LID(A->GCID64(indices[j]));
#ifdef HYMLS_TESTING
        if (lcid_j < 0) Tools::Error("diagonal entry not imported?", __FILE__, __LINE__);
#endif
        scal = std::max(scal_i, std::abs((*diagA)[lcid_j]));
        }

      if (std::abs(values[j]) > scal*droptol && std::abs(values[j]) > droptol)
        {
        // retain the entry
        new_values[new_len] = values[j];
        new_indices[new_len] = A->GCID64(indices[j]);
        new_len++;
        }
      else if (isDiag && zeroDiag)
        {
        // put physical 0.0 in
        new_values[new_len] = 0.0;
        new_indices[new_len] = A->GCID64(indices[j]);
        new_len++;
        }
      }// j
#ifdef HYMLS_TESTING
    bool testFailed = false;
    for (int jj = 0; jj < new_len; jj++)
      {
      if (std::abs(new_values[jj])<std::numeric_limits < double>::epsilon()
        && new_indices[jj] != A->GRID64(i))
        {
        testFailed = true;
        Tools::out() << "row " << A->GRID64(i) << " col " << new_indices[jj] << std::endl;
        }
      }
    if (testFailed)
      {
      Tools::out() << "original matrix row (" << len << " entries): " << std::endl;
      for (int jj = 0; jj < len; jj++)
        {
        Tools::out() << A->GRID64(i) << " " << A->GCID64(indices[jj]) << " " << values[jj] << std::endl;
        }
      Tools::out() << "diagonal entry i: " << (*diagA)[diagA->Map().LID(A->GRID64(i))] << std::endl;
      Tools::out() << "diagonal entries j: " << std::endl;
      for (int jj = 0; jj < len; jj++)
        {
        Tools::out() << A->GCID64(indices[jj]) << " " << A->GCID64(indices[jj]) << " " <<
          (*diagA)[diagA->Map().LID(A->GCID64(indices[jj]))] << std::endl;
        }
      Tools::out() << "matrix row after dropping (" << new_len << " entries): " << std::endl;
      for (int jj = 0; jj < new_len; jj++)
        {
        Tools::out() << A->GRID64(i) << " " << new_indices[jj] << " ";
        Tools::out() << new_values[jj] << std::endl;
        }
      Tools::Warning("matrix contains tiny entries after dropping", __FILE__, __LINE__);
      }
#endif
    CHECK_ZERO(mat->InsertGlobalValues((hymls_gidx)A->GRID64(i),
        new_len, new_values, new_indices));
    }

  delete[] new_indices;
  delete[] new_values;

  HYMLS_DEBUG("calling FillComplete()");
  CHECK_ZERO(mat->FillComplete());

#ifdef HYMLS_TESTING
  hymls_gidx old_nnz = A->NumGlobalNonzeros64();
  hymls_gidx new_nnz = mat->NumGlobalNonzeros64();
  hymls_gidx nnz_dropped = old_nnz - new_nnz;
  double percent_dropped = 100.0*(((double)nnz_dropped) / ((double)old_nnz));

#define STR(var) (var? #var : "")

  Tools::Out("DropByValue (" + Teuchos::toString((float)droptol) + "):");
  Tools::out() << "DropType: " << (int)type << std::endl;
  Tools::out() << "condition: " << STR(rel) << " " << STR(absDiag) << " "
               << STR(zeroDiag) << std::endl;
  Tools::Out(" => dropped " + Teuchos::toString((float)percent_dropped) + "% of nonzeros");

#endif

  return mat;
  }


int MatrixUtils::PutDirichlet(Epetra_CrsMatrix& A, hymls_gidx gid)
  {
  HYMLS_PROF3(Label(), "PutDirichlet");

  // find out which proc owns this row

  int lid, pid;

  CHECK_ZERO(A.RowMap().RemoteIDList(1, &gid, &pid, &lid));

  if (lid < 0)
    {
    Dump(A, "BadMatrix.txt");
    Tools::Error("fix GID: " + Teuchos::toString(gid) + " not in matrix row map", __FILE__, __LINE__);
    }

  // find out how long that row is (how many nonzeros)
  int len;

  if (pid == A.Comm().MyPID())
    {
    CHECK_ZERO(A.NumMyRowEntries(lid, len));
    }

  CHECK_ZERO(A.Comm().Broadcast(&len, 1, pid));

  hymls_gidx* indices = new hymls_gidx[len];
  double* values = new double[len];

  if (pid == A.Comm().MyPID())
    {
    int dummy_len;
    CHECK_ZERO(A.ExtractGlobalRowCopy(gid, len, dummy_len, values, indices));
    // set row to 0 and diagonal to 1
    for (int i = 0; i < len; i++)
      {
      if (indices[i] == gid)
        {
        values[i] = 1.0;
        }
      else
        {
        values[i] = 0.0;
        }
      }
    // put it back in
    CHECK_ZERO(A.ReplaceGlobalValues(gid, len, values, indices));
    }

  // broadcast indices to everyone
  CHECK_ZERO(A.Comm().Broadcast(indices, len, pid));

  // we assume that the pattern of the matrix is symmetric and process all the rows in
  // indices, setting any coupling to gid to 0
  int *indices_i;
  double *values_i;
  int len_i;
  for (int i = 0; i < len; i++)
    {
    hymls_gidx grid = indices[i];
    if (A.RowMap().MyGID(grid))
      {
      if (grid != gid)
        {
        int lrid = A.LRID(grid);
        CHECK_ZERO(A.ExtractMyRowView(lrid, len_i, values_i, indices_i));
        for (int j = 0; j < len_i; j++)
          {
          if (A.GCID64(indices_i[j]) == gid)
            {
            values_i[j] = 0.0;
            }
          }
        }
      }
    }

  delete [] indices;
  delete [] values;
  return 0;
  }

int MatrixUtils::FillReducingOrdering(const Epetra_CrsMatrix& Matrix,
  Teuchos::Array<int> & rowperm,
  Teuchos::Array<int> & colperm,
  bool dummy)
  {
  HYMLS_PROF3(Label(), "FillReducingOrdering");

  if (!Matrix.Filled()) Tools::Error("matrix not filled", __FILE__, __LINE__);

  bool parallel = (Matrix.Comm().NumProc() > 1);
  HYMLS_DEBVAR(parallel);

  int N = Matrix.NumMyRows();
  int n = 0; // number of V - nodes
  int m = 0; // number of P - nodes

  Teuchos::RCP<Epetra_CrsMatrix> tmpMatrix;

  Teuchos::RCP<Epetra_Map> map1, map2;

  int* col;
  double* val;
  int len;

  int* elts1 = new int[N];
  int* elts2 = new int[N];

  for (int i = 0; i < Matrix.NumMyRows(); i++)
    {
    int row = Matrix.GRID64(i);
    HYMLS_DEBVAR(row);
    CHECK_ZERO(Matrix.ExtractMyRowView(i, len, val, col));
    int j;
    bool no_diag = true;
    for (j = 0; j < len; j++)
      {
      HYMLS_DEBVAR(Matrix.GCID64(col[j]));
      if (Matrix.GCID64(col[j]) == row)
        {
        no_diag = (std::abs(val[j]) == 0.0);
        break;
        }
      }
    HYMLS_DEBVAR(no_diag);
    if (no_diag)
      {
      elts2[m++] = row;
      }
    else
      {
      elts1[n++] = row;
      }
    }

  HYMLS_DEBVAR(N);
  HYMLS_DEBVAR(n);
  HYMLS_DEBVAR(m);

  bool indefinite = m > 0;
  HYMLS_DEBVAR(indefinite);

  bool fmatrix = false;

  Teuchos::RCP<Epetra_CrsMatrix> Bmat = Teuchos::null;
  Teuchos::RCP<Epetra_CrsMatrix> BTmat = Teuchos::null;

  if (!indefinite)
    {
    tmpMatrix = Teuchos::rcp(const_cast < Epetra_CrsMatrix*> (&Matrix), false);
    }
  else
    {
    // 1) create maps of A and B
    int base = Matrix.RowMap().IndexBase64();
    const Epetra_Comm& comm = Matrix.Comm();

    map1 = Teuchos::rcp(new Epetra_Map(-1, n, elts1, base, comm));
    map2 = Teuchos::rcp(new Epetra_Map(-1, m, elts2, base, comm));

    Teuchos::RCP<Epetra_Map> colmap1 = map1;
    Teuchos::RCP<Epetra_Map> colmap2 = map2;
    // in parallel we would need actual column maps, and hymls_gidx indices
    if (parallel) Tools::Error("not implemented!", __FILE__, __LINE__);

    // c) create a copy of the matrices A, B ( = grad) and B' ( = div)
    Epetra_CrsMatrix A(Copy, *map1, *colmap1, Matrix.MaxNumEntries());
    // we need B outside this if statement later on to add the P - nodes
    Bmat = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map1, *colmap2, Matrix.MaxNumEntries()));
    BTmat = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map2, *colmap1, Matrix.MaxNumEntries()));
    Epetra_CrsMatrix& B = *Bmat;
    Epetra_CrsMatrix& Bt = *BTmat;

    Epetra_Import import1(Matrix.RowMap(), *map1);
    Epetra_Import import2(Matrix.RowMap(), *map2);

    CHECK_ZERO(A.Export(Matrix, import1, Insert));
    CHECK_ZERO(B.Export(Matrix, import1, Insert));
    CHECK_ZERO(Bt.Export(Matrix, import2, Insert));

    CHECK_ZERO(A.FillComplete(*map1, *map1));
    CHECK_ZERO(B.FillComplete(*map2, *map1));
    CHECK_ZERO(Bt.FillComplete(*map1, *map2));

    fmatrix = B.MaxNumEntries() == 2 || B.MaxNumEntries() == 1;
    HYMLS_DEBVAR(fmatrix);

    if (!fmatrix)
      {
      std::cerr << "B-part has MaxNumEntries!=2\n"
                << "writing it to BadMatrixB.txt for you.\n";
      Dump(B, "BadMatrixB.txt");
      }

    CHECK_ZERO(A.PutScalar(1.0));
    CHECK_ZERO(B.PutScalar(1.0));
    CHECK_ZERO(Bt.PutScalar(1.0));

    // create the graph of A + BB'
    tmpMatrix = Teuchos::rcp(new Epetra_CrsMatrix(Copy, *map1, A.MaxNumEntries()));
    CHECK_ZERO(EpetraExt::MatrixMatrix::Multiply(B, false, Bt, false, *tmpMatrix, false));
    CHECK_ZERO(EpetraExt::MatrixMatrix::Add(A, false, 1.0, *tmpMatrix, 1.0));
    CHECK_ZERO(tmpMatrix->FillComplete());
    } // indefinite matrix->tmpMat = A + BB'

  delete [] elts1;
  delete [] elts2;

  if (indefinite == true && ((fmatrix == false) || (parallel == true)))
    {
    Dump(*map1, "BadMatrixMap1.txt");
    Dump(*map2, "BadMatrixMap2.txt");

    Dump(Matrix, "BadMatrix.txt");
    std::cerr << "parallel=" << parallel << ", indefinite=" << indefinite << ", fmatrix=" << fmatrix << std::endl;
    Tools::Error("this subroutine is intended for serial F-matrices \n"
      " or matrices with nonzero diagonal right now.\n"
      " the invalid matrix is written to BadMatrix.txt",
      __FILE__, __LINE__);
    }

#ifdef HYMLS_DEBUGGING
  std::ofstream deb;
  if (dump)
    {
    deb.open("fill_reducing_ordering.m");

    deb << "N=" << N << "; n=" << n << "; m=" << m << ";\n";
      {
      int len;
      int *inds;
      double* values;
      deb << std::setw(16) << std::setprecision(16) << std::scientific;
      deb << "tmp=[...\n";
      for (int i = 0; i < Matrix.NumMyRows(); i++)
        {
        CHECK_ZERO(Matrix.ExtractMyRowView(i, len, values, inds));
        for (int j = 0; j < len; j++)
          {
          deb << Matrix.GRID64(i) + 1 << " " << Matrix.GCID64(inds[j]) + 1 << " " << values[j] << std::endl;
          }
        }
      deb << "];" << std::endl;
      deb << "K=sparse(tmp(:,1),tmp(:,2),tmp(:,3));\n";
      }
    deb << std::endl;
    if (map1 != Teuchos::null)
      {
      deb << "map1=[";
      for (int i = 0; i < map1->NumMyElements(); i++) deb << map1->GID64(i) + 1 << " ";
      deb << "];\n";
      }
    if (map2 != Teuchos::null)
      {
      deb << "map2=[";
      for (int i = 0; i < map2->NumMyElements(); i++) deb << map2->GID64(i) + 1 << " ";
      deb << "];\n";
      }
    if (tmpMatrix.get() != &Matrix)
      {
      int len;
      int *inds;
      double* values;
      deb << "tmp=[...\n";
      for (int i = 0; i < tmpMatrix->NumMyRows(); i++)
        {
        CHECK_ZERO(tmpMatrix->ExtractMyRowView(i, len, values, inds));
        for (int j = 0; j < len; j++)
          {
          deb << tmpMatrix->GRID64(i) + 1 << " " << tmpMatrix->GCID64(inds[j]) + 1 << " " << values[j] << std::endl;
          }
        }
      deb << "];" << std::endl;
      deb << "Atilde=sparse(tmp(:,1),tmp(:,2),tmp(:,3));\n";
      }
    deb << std::flush;
    }
#endif

  Teuchos::Array<int> q(n);

  if (!dummy) {
#ifndef HAVE_METIS
    // use AMD
    CHECK_ZERO(AMD(tmpMatrix->Graph(), q));
#else
    // use Zoltan (also works in parallel but requires (Par)METIS
    HYMLS_DEBUG("zoltan ordering step");

    // compute fill - reducing ordering of A + BB' using Isorropia->Zoltan->Metis
    // (complicated somehow, but right now that's the only method directly available for
    // Epetra graphs)
    Teuchos::ParameterList params;
    Teuchos::ParameterList& zList = params.sublist("Zoltan");
    zList.set("ORDER_METHOD", "ParMETIS");
    // reindex the matrix to have a linear map
    Teuchos::RCP<Epetra_Map> linearMap = Teuchos::rcp(new
      Epetra_Map(tmpMatrix->NumGlobalRows(),
        tmpMatrix->NumMyRows(),
        0, tmpMatrix->Comm()) );

    Teuchos::RCP<EpetraExt::CrsMatrix_Reindex> reindex = Teuchos::rcp(new
      EpetraExt::CrsMatrix_Reindex(*linearMap));

    Teuchos::RCP<Epetra_CrsMatrix> linearMatrix =
      Teuchos::rcp(&((*reindex)(*tmpMatrix)), false);

    //  graph = Teuchos::rcp(&(tmpMatrix->Graph()), false);
    Teuchos::RCP<const Epetra_CrsGraph> graph = Teuchos::rcp(&(linearMatrix->Graph()), false);

    // TODO: this causes an exception!!!
    Isorropia::Epetra::Orderer reorder(graph, params);

    // now we have an ordering for A + BB', add in the pressures at the
    // right positions
    const int* q0;
    int len_q;
    CHECK_ZERO(reorder.extractPermutationView(len_q, q0));
    if (len_q != n) Tools::Error("inconsistency discovered", __FILE__, __LINE__);

    // CAUTION - the ordering q0 is the inverse of the one we're used to from MATLAB,
    //           so Anew(q0, q0) = Aold instead of Anew = Aold(q, q)! We invert the ordering
    //           before returning it to the caller.
    if (parallel) Tools::Error("not implemented", __FILE__, __LINE__);

    for (int i = 0; i < len_q; i++) q[q0[i]] = i;
#endif
    }
  else
    {
    // for testing - disable fill - reducing ordering of V - nodes
    for (int i = 0; i < n; i++) q[i] = i;
    }// dummy == true

  if (rowperm.size() != N)
    {
    rowperm.resize(N);
    }
  if (colperm.size() != N)
    {
    colperm.resize(N);
    }
  if (!indefinite)
    {
    for (int i = 0; i < N; i++)
      {
      rowperm[i] = q[i];
      colperm[i] = q[i];
      }
    }
  else
    {
    HYMLS_DEBUG("adjust ordering to include P-nodes");
    Teuchos::Array<int> symperm(N);
    Teuchos::Array<int> perm(N);

    // implementation taken from Fred's matlab variant addindefnodes3.m
    Teuchos::Array<Teuchos::Array < int> > Gr(n);

    for (int i = 0; i < n; i++)
      {
      Gr[i].resize(2);
      for (int j = 0; j < 2; j++) Gr[i][j] = m;
      }

    // test vector to see if there is a coupling to a p - node
    Teuchos::Array<int> cont(m);


    // cont = sum(spones(B)); We assume B( = Grad) == B'( == Div) so we
    // sum over the rows of Bt rather tnan the cols of B. Further -
    // more we assume that there are no zeros in B so we simply
    // count the row lengths of B'.
    for (int i = 0; i < m; i++)
      {
      cont[i] = BTmat->NumMyEntries(i);
      }
    // Gr(Ip(i), 1 or 2) = Jp(i);
    for (int i = 0; i < n; i++)
      {
      CHECK_ZERO(Bmat->ExtractMyRowView(i, len, val, col));
      for (int j = 0; j < len; j++)
        {
        if (Gr[i][0] == m)
          {
          Gr[i][0] = col[j];
          }
        else
          {
          Gr[i][1] = col[j];
          }
        }
      }
    // pressure id's
    Teuchos::Array<int> pid(m + 1);
    for (int i = 0; i < m + 1; i++) pid[i] = i;

    // row perm to get all diagonal entries nonzero
    for (int i = 0; i < N; i++) perm[i] = i;

#ifdef HYMLS_DEBUGGING
    if (dump)
      {
      deb << "N=" << N << std::endl;
      deb << "n=" << n << std::endl;
      deb << "m=" << m << std::endl;
      deb << "Gr=[..." << std::endl;
      for (int i = 0; i < n; i++)
        deb << Gr[i][0] << " " << Gr[i][1] << ";\n";
      deb << "];\n\n";
      deb << std::flush;
      }
#endif
    int jj = 0;
    int gr1, gr2;
    bool status = true;
    try {


      // for all V - nodes
      for (int i = 0; i < n; i++)
        {
        // where is the original position in the system?
        int qi = q[i];
        // put the V - node in the ordering first
        symperm[jj] = map1->GID64(qi);
        // to which P - nodes does this V - node couple?
        gr1 = Gr[qi][0]; // first pressure node
        gr2 = Gr[qi][1]; // second pressure node
//      HYMLS_DEBUG(i << ": " << qi << "[" << gr1 << " " << gr2 << "]");
        // go through all the P - nodes that 'have been eliminated'
        // (e.g. put in the ordering) and find the first not yet
        // eliminated
        while (pid[gr1] != gr1){gr1 = pid[gr1]; }
        while (pid[gr2] != gr2) {gr2 = pid[gr2]; }
        if (gr1 != gr2)
          {
          if (gr1 == m) // formally eliminate V - node coupled to gr2
            {
            pid[gr2] = pid[gr1];
            symperm[jj+1] = map2->GID64(gr2);
            }
          else if (gr2 == m) // formally eliminate V - node coupled to gr1
            {
            pid[gr1] = pid[gr2];
            symperm[jj+1] = map2->GID64(gr1);
            }
          else if (cont[gr2] > cont[gr1])
            {
            pid[gr1] = pid[gr2];
            symperm[jj+1] = map2->GID64(gr1);
            cont[gr2] = cont[gr1] + cont[gr2] - 2;
            }
          else
            {
            pid[gr2] = pid[gr1];
            symperm[jj+1] = map2->GID64(gr2);
            cont[gr1] = cont[gr1] + cont[gr2] - 2;
            }
          // interchange the V - and P - rows to get a pivot
          // Of the form b 0 rather than a b
          //             a b             b 0.
          perm[jj] = jj + 1; perm[jj+1] = jj;
          jj = jj + 2;
          }
        else // V - node has no P - couplings (anymore)
          {
          jj = jj + 1;
          }
        }
      } TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, status);
    if (!status) Tools::Fatal("caught exception while computing fill-reducing ordering",
      __FILE__, __LINE__);


#ifdef HYMLS_DEBUGGING
    HYMLS_DEBUG("temporary symperm:");
    for (int i = 0; i < jj; i++) Tools::deb() << symperm[i] << " ";
#endif
    // test = ones(1, m + n); test(p) = 0; p(j:n + m) = find(test);
    Teuchos::Array<int> test(N);
    for (int i = 0; i < N; i++) test[i] = 1;
    for (int i = 0; i < jj; i++) test[symperm[i]] = 0;
    int kk = 0;
    for (int i = 0; i < N; i++)
      {
      if (test[i])
        {
        symperm[jj+kk] = Matrix.GRID64(i);
        kk++;
        }
      }

#ifdef HYMLS_DEBUGGING
    HYMLS_DEBUG("final symperm:");
    for (int i = 0; i < N; i++) Tools::deb() << symperm[i] << " ";
#endif

    for (int i = 0; i < N; i++)
      {
      colperm[i] = symperm[i];
      rowperm[i] = symperm[perm[i]];
      }
    }
#ifdef HYMLS_DEBUGGING
  if (dump)
    {
    deb << "% initial ordering of v-nodes\n";
    deb << "q0=[";
    for (int i = 0; i < n; i++) deb << q[i] + 1 << " ";
    deb << "];\n\n";
    deb << "% row permutation\n";
    deb << "p=[";
    for (int i = 0; i < N; i++) deb << rowperm[i] + 1 << " ";
    deb << "];\n";
    deb << "% column permutation\n";
    deb << "q=[";
    for (int i = 0; i < N; i++) deb << colperm[i] + 1 << " ";
    deb << "];\n";
    deb.close();
    }
#endif

  return 0;
  }

int MatrixUtils::AMD(const Epetra_CrsGraph& A, Teuchos::Array<int> & p)
  {
  HYMLS_PROF3(Label(), "AMD");

  int n = A.NumMyRows();
  if (p.size() < n)
    {
    p.resize(n);
    }
  for (int i = 0; i < n; i++) p[i] = i;

  if (n == 0) return 0;

  if (A.Comm().NumProc() > 1)
    {
    Tools::Warning("AMD not available for parallel matrices", __FILE__, __LINE__);
    return - 2;
    }
  if (A.StorageOptimized() == false)
    {
    Tools::Warning("graph must be StorageOptimized() for AMD interface", __FILE__, __LINE__);
    return - 3;
    }
  int len;
  int* Ap = new int[n+1];
  int* Ai;
  CHECK_ZERO(A.ExtractMyRowView(0, len, Ai ));
  Ap[0] = 0;
  int *tmp;
  for ( int i = 0; i < n; i++)
    {
    CHECK_ZERO(A.ExtractMyRowView( i, len, tmp ));
    Ap[i+1] = Ap[i] + len ;
    }

  double *control = NULL;
  double *info = new double[TRILINOS_AMD_INFO];

  /* returns AMD_OK, AMD_OK_BUT_JUMBLED,
     AMD_INVALID, or AMD_OUT_OF_MEMORY */
  int ierr = trilinos_amd_order(n, Ap, Ai, &p[0], control, info);
  // TODO: check for errors
  delete [] info;
  delete [] Ap;
  return ierr;
  }

// this piece of code is borrowed from Epetra_CrsMatrix.cpp
int MatrixUtils::SortMatrixRow(int* indices, double* values, int len)
  {
  int n = len;
  int m = n / 2;
  while(m > 0) {
    int max = n - m;
    for (int j = 0; j < max; j++) {
      for (int k = j; k >= 0; k -= m) {
        if (indices[k+m] >= indices[k])
          break;
        double dtemp = values[k+m];
        values[k+m] = values[k];
        values[k] = dtemp;
        int itemp = indices[k+m];
        indices[k+m] = indices[k];
        indices[k] = itemp;
        }
      }
    m = m / 2;
    }
  return 0;
  }

// extract a local part of a matrix. This can not easily be done by Import
// objects because they tend to do collective communication. A_loc should
// be !DistributedGlobal() and A should be IndicesAreLocal() (Filled()?).
int MatrixUtils::ExtractLocalBlock(const Epetra_RowMatrix& A, Epetra_CrsMatrix& A_loc)
  {
  HYMLS_PROF3(Label(), "ExtractLocalBlock");

//  if (A.IndicesAreLocal() == false) Tools::Error("A must be Filled()", __FILE__, __LINE__);
  if (A_loc.DistributedGlobal() == true) Tools::Error("A_loc must be serial", __FILE__, __LINE__);
  int ierr = 0;
  int maxLen = A.MaxNumEntries();
  int *inds = new int[maxLen];
  double *vals = new double[maxLen];
  int len;
  for (int i = 0; i < A_loc.NumMyRows(); i++)
    {
    int iA = A.RowMatrixRowMap().LID(A_loc.GRID64(i));
//    HYMLS_DEBUG("")
//    HYMLS_DEBUG("row " << i << " " << iA << " " << A_loc.GRID(i));
    if (iA < 0) return - 1;
    CHECK_ZERO(A.ExtractMyRowCopy(iA, maxLen, len, vals, inds));
//    HYMLS_DEBVAR(len);
    int new_len = 0;
    for (int j = 0; j < len; j++)
      {
      hymls_gidx gcid = A.RowMatrixColMap().GID64(inds[j]);
      int lcid = A_loc.LCID(gcid);
      //    HYMLS_DEBUG("\t" << lcid << " " << gcid);
      if (lcid >= 0)
        {
        inds[new_len] = lcid;
        vals[new_len] = vals[j];
        new_len++;
        }
      }
    //  HYMLS_DEBVAR(new_len);
    if (A_loc.Filled())
      {
      CHECK_NONNEG(A_loc.ReplaceMyValues(i, new_len, vals, inds));
      }
    else
      {
      CHECK_NONNEG(A_loc.InsertMyValues(i, new_len, vals, inds));
      }
    }
  delete [] inds;
  delete [] vals;
  return ierr;
  }
  }
