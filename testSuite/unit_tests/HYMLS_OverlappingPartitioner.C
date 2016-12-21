#include "HYMLS_OverlappingPartitioner.H"

#include "HYMLS_CartesianPartitioner.H"
#include "HYMLS_SkewCartesianPartitioner.H"

#include "Galeri_CrsMatrices.h"
#include "GaleriExt_CrsMatrices.h"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"

#include "HYMLS_UnitTests.H"

#define FOR_EACH_1(FUN, X) FUN(X) 
#define FOR_EACH_2(FUN, X, ...) FUN(X) FOR_EACH_1(FUN, __VA_ARGS__)
#define FOR_EACH_3(FUN, X, ...) FUN(X) FOR_EACH_2(FUN, __VA_ARGS__)
#define FOR_EACH_4(FUN, X, ...) FUN(X) FOR_EACH_3(FUN, __VA_ARGS__)
#define FOR_EACH_5(FUN, X, ...) FUN(X) FOR_EACH_4(FUN, __VA_ARGS__)
#define FOR_EACH_6(FUN, X, ...) FUN(X) FOR_EACH_5(FUN, __VA_ARGS__)

#define GET_MACRO(ARG1, ARG2, ARG3, ARG4, ARG5, ARG6, NAME, ...) NAME
#define FOR_EACH(FUN, ...) \
  GET_MACRO(__VA_ARGS__, FOR_EACH_6, FOR_EACH_5, FOR_EACH_4, FOR_EACH_3,\
    FOR_EACH_2, FOR_EACH_1)(FUN, __VA_ARGS__)

#define ASSIGN_DATA_MEMBER(X) this->X = X;
#define DECLARE_DATA_MEMBER(X) int X;
#define FUNCTION_ARGUMENT(X) ,int X

#define TEUCHOS_UNIT_TEST_DECL(TEST_GROUP, TEST_NAME, FIRST, ...)        \
  class TEST_GROUP##_##TEST_NAME##_UnitTest : public Teuchos::UnitTestBase \
    {                                                                   \
    FOR_EACH(DECLARE_DATA_MEMBER, FIRST, __VA_ARGS__);                   \
    public:                                                             \
    TEST_GROUP##_##TEST_NAME##_UnitTest(int FIRST FOR_EACH(FUNCTION_ARGUMENT, __VA_ARGS__)) \
      : Teuchos::UnitTestBase( #TEST_GROUP, #TEST_NAME )                \
      {                                                                 \
      FOR_EACH(ASSIGN_DATA_MEMBER, FIRST, __VA_ARGS__);                  \
      }                                                                 \
    virtual void runUnitTestImpl( Teuchos::FancyOStream &out, bool &success ) const; \
    virtual std::string unitTestFile() const { return __FILE__; }       \
    virtual long int unitTestFileLineNumber() const { return __LINE__; } \
    };                                                                  \
                                                                        \
  void TEST_GROUP##_##TEST_NAME##_UnitTest::runUnitTestImpl(            \
    Teuchos::FancyOStream &out, bool &success ) const                   \


#define TEUCHOS_UNIT_TEST_INST(TEST_GROUP, TEST_NAME, NUM, ...)          \
  TEST_GROUP##_##TEST_NAME##_UnitTest                                   \
  instance_##TEST_GROUP##_##TEST_NAME##_##NUM##_UnitTest(__VA_ARGS__);

class TestableOverlappingPartitioner : public HYMLS::OverlappingPartitioner
  {
public:
  TestableOverlappingPartitioner(Teuchos::RCP<const Epetra_RowMatrix> K,
    Teuchos::RCP<Teuchos::ParameterList> params, int level)
    :
    HYMLS::OverlappingPartitioner(K, params, level)
    {}

  Teuchos::RCP<TestableOverlappingPartitioner> RemoveCornerSeparators()
    {
    Teuchos::RCP<TestableOverlappingPartitioner> newPart = Teuchos::rcp(new TestableOverlappingPartitioner(matrix_, getMyNonconstParamList(), Level()));
    newPart->Reset(partitioner_->NumLocalParts());

    // Walk over subdomains
    for (int sd = 0; sd < partitioner_->NumLocalParts(); sd++)
      {
      Teuchos::Array<int> interior = GetGroup(sd, 0);
      newPart->AddGroup(sd, interior);
      int gsd = (*partitioner_)(interior[0]);
      // And groups
      for (int grp = 1; grp < NumGroups(sd); grp++)
        {
        int found = 0;
        Teuchos::Array<int> group = GetGroup(sd, grp);
        // Check if the node is in the domain
        if ((*partitioner_)(group[0]) == gsd)
          found = 4;

        // See if moving a separator 1 step puts it inside the interior
        int search[3] = {group[0] + dof_, group[0] + dof_ * nx_, group[0] + dof_ * nx_ * ny_};
        for (int i = 0; i < 3; i++)
          {
          Teuchos::Array<int>::iterator it = std::find(interior.begin(), interior.end(), search[i]);
          if (it != interior.end())
            {
            // It is a direct neighbour in 1 direction
            found++;
            break;
            }
          }

        if (found > 0)
          newPart->AddGroup(sd, group);
        }
      }

    newPart->FillComplete();
    return newPart;
    }
  };

TEUCHOS_UNIT_TEST_DECL(OverlappingPartitioner, Laplace2D, nx, ny, sx, sy)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  HYMLS::Tools::InitializeIO(Comm);

  int nsx = nx / sx;
  int nsy = ny / sy;

  int dof = 1;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(nx*ny*dof, 0, *Comm));

  Teuchos::RCP<HYMLS::CartesianPartitioner> part = Teuchos::rcp(new HYMLS::CartesianPartitioner(map, nx, ny, 1, dof));
  part->Partition(nsx * nsy, true);
  *map = *part->GetMap();

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::ParameterList &problemList = paramList->sublist("Problem");
  problemList.set("nx", nx);
  problemList.set("ny", ny);
  problemList.set("nz", 1);

  Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::rcp(
    Galeri::CreateCrsMatrix("Laplace2D", map.get(), problemList));

  problemList.set("Dimension", 2);
  problemList.set("Degrees of Freedom", dof);

  Teuchos::ParameterList &solverList = paramList->sublist("Preconditioner");
  solverList.set("Separator Length", sx);
  solverList.set("Coarsening Factor", 2);

  TestableOverlappingPartitioner opart(matrix, paramList, 0);
  Teuchos::RCP<TestableOverlappingPartitioner> opart2 = opart.RemoveCornerSeparators();

  for (int sd = 0; sd < opart2->NumMySubdomains(); sd++)
    {
    int gsd = opart2->Partitioner().SubdomainMap().GID(sd);
    int substart = gsd % nsx * nx / nsx * dof +
      gsd / nsx * ny / nsy * dof * nx;

    // Compute the number of groups we expect
    int numGrps = 6;
    // Right
    numGrps -= ((gsd + 1) % nsx == 0) * 2;
    // Bottom
    numGrps -= (gsd / nsx == nsy - 1) * 2;
    // Left
    numGrps -= gsd % nsx == 0;
    // Top
    numGrps -= gsd / nsx == 0;
    if (numGrps < 3)
      numGrps = 3;

    TEST_EQUALITY(opart2->NumGroups(sd), numGrps);

    for (int grp = 0; grp < opart2->NumGroups(sd); grp++)
      {
      if (grp == 0)
        {
        // Interior
        if ((gsd + 1) % nsx == 0 && gsd / nsx == nsy - 1)
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sx * sy);
          for (int i = 0; i < opart2->NumElements(sd, grp); i++)
            {
            TEST_EQUALITY(opart2->GID(sd, grp, i), substart + i % sx + i / sx * nx);
            }
          }
        else if ((gsd + 1) % nsx == 0)
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sx * (sy-1));
          for (int i = 0; i < opart2->NumElements(sd, grp); i++)
            {
            TEST_EQUALITY(opart2->GID(sd, grp, i), substart + i % sx + i / sx * nx);
            }
          }
        else if (gsd / nsx == nsy - 1)
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sy * (sx-1));
          for (int i = 0; i < opart2->NumElements(sd, grp); i++)
            {
            TEST_EQUALITY(opart2->GID(sd, grp, i), substart + i % (sx-1) + i / (sx-1) * nx);
            }
          }
        else
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), (sx-1) * (sy-1));
          for (int i = 0; i < opart2->NumElements(sd, grp); i++)
            {
            TEST_EQUALITY(opart2->GID(sd, grp, i), substart + i % (sx-1) + i / (sx-1) * nx);
            }
          }
        }
      else if (opart2->GID(sd, grp, 0) == substart - nx || opart2->GID(sd, grp, 0) == substart + nx * (sx - 1))
        {
        // Top or bottom border
        if ((gsd + 1) % nsx == 0)
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sx);
          }
        else
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sx - 1);
          }
        for (int i = 0; i < opart2->NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart2->GID(sd, grp, i), opart2->GID(sd, grp, 0) + i);
          }
        }
      else if (opart2->GID(sd, grp, 0) == substart + sy - 1 || opart2->GID(sd, grp, 0) == substart - 1)
        {
        // Left or right border
        if (gsd / nsx == nsy - 1)
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sy);
          }
        else
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sy - 1);
          }
        for (int i = 0; i < opart2->NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart2->GID(sd, grp, i), opart2->GID(sd, grp, 0) + i * nx);
          }
        }
      else
        {
        // Corner
        TEST_EQUALITY(opart2->NumElements(sd, grp), 1);
        TEST_EQUALITY(opart2->GID(sd, grp, 0), substart + nx * (sy - 1) + sx - 1);
        }
      }
    }
  }

TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace2D, 1, 8, 8, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace2D, 2, 16, 16, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace2D, 3, 16, 8, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace2D, 4, 4, 4, 2, 2);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace2D, 5, 64, 64, 16, 16);

TEUCHOS_UNIT_TEST_DECL(OverlappingPartitioner, Laplace3D, nx, ny, nz, sx, sy, sz)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  HYMLS::Tools::InitializeIO(Comm);

  int nsx = nx / sx;
  int nsy = ny / sy;
  int nsz = nz / sz;

  int dof = 1;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(nx*ny*nz*dof, 0, *Comm));

  Teuchos::RCP<HYMLS::CartesianPartitioner> part = Teuchos::rcp(new HYMLS::CartesianPartitioner(map, nx, ny, nz, dof));
  part->Partition(nsx * nsy * nsz, true);
  *map = *part->GetMap();

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::ParameterList &problemList = paramList->sublist("Problem");
  problemList.set("nx", nx);
  problemList.set("ny", ny);
  problemList.set("nz", nz);

  Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::rcp(
    Galeri::CreateCrsMatrix("Laplace3D", map.get(), problemList));

  problemList.set("Dimension", 3);
  problemList.set("Degrees of Freedom", dof);

  Teuchos::ParameterList &solverList = paramList->sublist("Preconditioner");
  solverList.set("Separator Length", sx);
  solverList.set("Coarsening Factor", 2);

  TestableOverlappingPartitioner opart(matrix, paramList, 0);
  Teuchos::RCP<TestableOverlappingPartitioner> opart2 = opart.RemoveCornerSeparators();

  for (int sd = 0; sd < opart2->NumMySubdomains(); sd++)
    {
    int gsd = opart2->Partitioner().SubdomainMap().GID(sd);
    int substart = gsd % nsx * nx / nsx * dof +
      (gsd % (nsx * nsy)) / nsx * ny / nsy * dof * nx +
      gsd / (nsx * nsy) * nz / nsz * dof * nx * ny;

    // Compute the number of groups we expect
    int numGrps = 11;
    int pos = 0;
    // Right
    if ((gsd + 1) % nsx == 0)
      {
      numGrps -= 4;
      pos += 1;
      }
    // Bottom
    if ((gsd % (nsx * nsy)) / nsx == nsy - 1)
      {
      numGrps -= 4 - 2 * pos;
      pos += 2;
      }
    // Back
    if (gsd / (nsx * nsy) == nsz - 1)
      {
      numGrps -= 4 - (pos == 1 ? 2 : pos);
      pos += 4;
      }
    // Left
    if (gsd % nsx == 0)
      {
      pos += 8;
      numGrps -= 1;
      }
    // Top
    if ((gsd % (nsx * nsy)) / nsx == 0)
      {
      pos += 16;
      numGrps -= 1;
      }
    // Front
    if (gsd / (nsx * nsy) == 0)
      {
      pos += 32;
      numGrps -= 1;
      }

    TEST_EQUALITY(opart2->NumGroups(sd), numGrps);

    int totalNodes = 0;
    for (int grp = 0; grp < opart2->NumGroups(sd); grp++)
      {
      totalNodes += opart2->NumElements(sd, grp);
      if (grp == 0)
        {
        // Interior
        if (pos == 7)
          {
          // Corner
          TEST_EQUALITY(opart2->NumElements(sd, grp), sx * sy * sz);
          for (int i = 0; i < opart2->NumElements(sd, grp); i++)
            {
            TEST_EQUALITY(opart2->GID(sd, grp, i), substart + i % sx + ((i / sx) % sy) * nx + i / (sx * sy) * nx * ny);
            }
          }
        else if (pos == 0)
          {
          // Center
          TEST_EQUALITY(opart2->NumElements(sd, grp), (sx-1) * (sy-1) * (sz-1));
          }
        }
      }
    if (pos == 0)
      {
      TEST_EQUALITY(totalNodes, sx * sy * sz + (sx-1) * (sy-1) + (sx-1) * (sz-1) + (sy-1) * (sz-1));
      }
    }
  }

TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace3D, 1, 8, 8, 8, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace3D, 2, 16, 16, 16, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace3D, 3, 16, 8, 8, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace3D, 4, 4, 4, 4, 2, 2, 2);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace3D, 5, 8, 4, 4, 4, 4, 4);


TEUCHOS_UNIT_TEST_DECL(OverlappingPartitioner, Stokes2D, nx, ny, sx, sy)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  HYMLS::Tools::InitializeIO(Comm);

  int nsx = nx / sx;
  int nsy = ny / sy;

  int dof = 3;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(nx*ny*dof, 0, *Comm));

  Teuchos::RCP<HYMLS::CartesianPartitioner> part = Teuchos::rcp(new HYMLS::CartesianPartitioner(map, nx, ny, 1, dof));
  part->Partition(nsx * nsy, true);
  *map = *part->GetMap();

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::ParameterList &problemList = paramList->sublist("Problem");
  problemList.set("nx", nx);
  problemList.set("ny", ny);
  problemList.set("nz", 1);
  
  Teuchos::ParameterList problemListCopy = problemList;
  Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::rcp(
    GaleriExt::CreateCrsMatrix("Stokes2D", map.get(), problemListCopy));

  for (int i = 0; i < 2; i++)
      {
      Teuchos::ParameterList& velList =
        problemList.sublist("Variable " + Teuchos::toString(i));
      velList.set("Variable Type", "Laplace");
      }

  Teuchos::ParameterList& presList =
    problemList.sublist("Variable "+Teuchos::toString(2));
  presList.set("Variable Type", "Retain 1");
  presList.set("Retain Isolated", true);

  problemList.set("Dimension", 2);
  problemList.set("Degrees of Freedom", dof);

  Teuchos::ParameterList &solverList = paramList->sublist("Preconditioner");
  solverList.set("Separator Length", sx);
  solverList.set("Coarsening Factor", 2);

  TestableOverlappingPartitioner opart(matrix, paramList, 0);
  Teuchos::RCP<TestableOverlappingPartitioner> opart2 = opart.RemoveCornerSeparators();

  for (int sd = 0; sd < opart2->NumMySubdomains(); sd++)
    {
    int gsd = opart2->Partitioner().SubdomainMap().GID(sd);
    int substart = gsd % nsx * nx / nsx * dof +
      gsd / nsx * ny / nsy * dof * nx;

    // Compute the number of groups we expect
    int numGrps = 13;
    // Right
    numGrps -= ((gsd + 1) % nsx == 0) * 5;
    // Bottom
    numGrps -= (gsd / nsx == nsy - 1) * 5;
    // Left
    numGrps -= (gsd % nsx == 0) * 2;
    // Top
    numGrps -= (gsd / nsx == 0) * 2;
    // Bottom right
    if (numGrps < 6)
      numGrps = 6;

    TEST_EQUALITY(opart2->NumGroups(sd), numGrps);

    for (int grp = 0; grp < opart2->NumGroups(sd); grp++)
      {
      if (grp == 0)
        {
        // Interior
        if ((gsd + 1) % nsx == 0 && gsd / nsx == nsy - 1)
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sx * sy * dof - 1);
          int pos = 0;
          for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++)
              for (int d = 0; d < dof; d++)
                if (!(d == 2 && pos == 2) && !(d == 2 && x == sx-1 && y == sy-1))
                  {
                  int gid = opart2->GID(sd, grp, pos++);
                  TEST_EQUALITY(gid, substart + x * dof + y * nx * dof + d);
                  }
          }
        else if ((gsd + 1) % nsx == 0)
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sx * (sy-1) * 2 + sx * sy - 1);
          int pos = 0;
          for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++)
              for (int d = 0; d < dof; d++)
                if (((x < sx && y < sy - 1) || d == 2)
                  && !(d == 2 && pos == 2) && !(d == 2 && x == sx-1 && y == sy-1))
                  {
                  int gid = opart2->GID(sd, grp, pos++);
                  TEST_EQUALITY(gid, substart + x * dof + y * nx * dof + d);
                  }
          }
        else if (gsd / nsx == nsy - 1)
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sy * (sx-1) * 2 + sx * sy - 1);
          int pos = 0;
          for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++)
              for (int d = 0; d < dof; d++)
                if (((x < sx - 1 && y < sy) || d == 2)
                  && !(d == 2 && pos == 2) && !(d == 2 && x == sx-1 && y == sy-1))
                  {
                  int gid = opart2->GID(sd, grp, pos++);
                  TEST_EQUALITY(gid, substart + x * dof + y * nx * dof + d);
                  }
          }
        else
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), (sx-1) * (sy-1) * 2 + sx * sy - 2);
          int pos = 0;
          for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++)
              for (int d = 0; d < dof; d++)
                if (((x < sx - 1 && y < sy - 1) || d == 2)
                  && !(d == 2 && pos == 2) && !(d == 2 && x == sx-1 && y == sy-1))
                  {
                  int gid = opart2->GID(sd, grp, pos++);
                  TEST_EQUALITY(gid, substart + x * dof + y * nx * dof + d);
                  }
          }
        }
      else if (opart2->GID(sd, grp, 0) / dof == substart / dof - nx || opart2->GID(sd, grp, 0) / dof == substart / dof + nx * (sx - 1))
        {
        // Right border
        if ((gsd + 1) % nsx == 0)
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sx);
          }
        else
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sx - 1);
          }
        for (int i = 0; i < opart2->NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart2->GID(sd, grp, i), opart2->GID(sd, grp, 0) + i * dof);
          }
        }
      else if (opart2->GID(sd, grp, 0) / dof == substart / dof + sy - 1 || opart2->GID(sd, grp, 0) / dof == substart / dof - 1)
        {
        // Bottom border
        if (gsd / nsx == nsy - 1)
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sy);
          }
        else
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sy - 1);
          }
        for (int i = 0; i < opart2->NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart2->GID(sd, grp, i), opart2->GID(sd, grp, 0) + i * nx * dof);
          }
        }
      else
        {
        // Corner
        TEST_EQUALITY(opart2->NumElements(sd, grp), 1);
        }
      }
    }
  }

TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Stokes2D, 1, 8, 8, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Stokes2D, 2, 16, 16, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Stokes2D, 3, 16, 8, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Stokes2D, 5, 64, 64, 16, 16);

TEUCHOS_UNIT_TEST_DECL(OverlappingPartitioner, Stokes3D, nx, ny, nz, sx, sy, sz)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  HYMLS::Tools::InitializeIO(Comm);

  int nsx = nx / sx;
  int nsy = ny / sy;
  int nsz = nz / sz;

  int dof = 4;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(nx*ny*nz*dof, 0, *Comm));

  Teuchos::RCP<HYMLS::CartesianPartitioner> part = Teuchos::rcp(new HYMLS::CartesianPartitioner(map, nx, ny, nz, dof));
  part->Partition(nsx * nsy * nsz, true);
  *map = *part->GetMap();

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::ParameterList &problemList = paramList->sublist("Problem");
  problemList.set("nx", nx);
  problemList.set("ny", ny);
  problemList.set("nz", nz);
  
  Teuchos::ParameterList problemListCopy = problemList;
  Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::rcp(
    GaleriExt::CreateCrsMatrix("Stokes3D", map.get(), problemListCopy));

  for (int i = 0; i < 3; i++)
      {
      Teuchos::ParameterList& velList =
        problemList.sublist("Variable " + Teuchos::toString(i));
      velList.set("Variable Type", "Laplace");
      }

  Teuchos::ParameterList& presList =
    problemList.sublist("Variable "+Teuchos::toString(3));
  presList.set("Variable Type", "Retain 1");
  presList.set("Retain Isolated", true);

  problemList.set("Dimension", 3);
  problemList.set("Degrees of Freedom", dof);

  Teuchos::ParameterList &solverList = paramList->sublist("Preconditioner");
  solverList.set("Separator Length", sx);
  solverList.set("Coarsening Factor", 2);

  TestableOverlappingPartitioner opart(matrix, paramList, 0);
  Teuchos::RCP<TestableOverlappingPartitioner> opart2 = opart.RemoveCornerSeparators();

  for (int sd = 0; sd < opart2->NumMySubdomains(); sd++)
    {
    int gsd = opart2->Partitioner().SubdomainMap().GID(sd);
    int substart = gsd % nsx * nx / nsx * dof +
      (gsd % (nsx * nsy)) / nsx * ny / nsy * dof * nx +
      gsd / (nsx * nsy) * nz / nsz * dof * nx * ny;

    // Compute the number of groups we expect
    int numGrps = 33;
    int pos = 0;
    // Right
    if ((gsd + 1) % nsx == 0)
      {
      numGrps -= 13;
      pos += 1;
      }
    // Bottom
    if ((gsd % (nsx * nsy)) / nsx == nsy - 1)
      {
      numGrps -= 13 - 7 * pos;
      pos += 2;
      }
    // Back
    if (gsd / (nsx * nsy) == nsz - 1)
      {
      if (pos == 2 || pos == 1)
        numGrps -= 13 - 7;
      else if (pos == 3)
        numGrps -= 13 - 10;
      else
        numGrps -= 13;
      pos += 4;
      }
    // Left
    if (gsd % nsx == 0)
      {
      pos += 8;
      numGrps -= 3;
      }
    // Top
    if ((gsd % (nsx * nsy)) / nsx == 0)
      {
      pos += 16;
      numGrps -= 3;
      }
    // Front
    if (gsd / (nsx * nsy) == 0)
      {
      pos += 32;
      numGrps -= 3;
      }

    TEST_EQUALITY(opart2->NumGroups(sd), numGrps);

    int totalNodes = 0;
    for (int grp = 0; grp < opart2->NumGroups(sd); grp++)
      {
      totalNodes += opart2->NumElements(sd, grp);
      if (grp == 0)
        {
        // Interior
        if (pos == 7)
          {
          // Corner
          TEST_EQUALITY(opart2->NumElements(sd, grp), sx * sy * sz * dof - 1);
          int pos = 0;
          for (int i = 0; i < opart2->NumElements(sd, grp) / dof; i++)
            {
            for (int d = 0; d < dof; d++)
              {
              if (d == 3 && pos == 3)
                continue;
              TEST_EQUALITY(opart2->GID(sd, grp, pos), substart + (i % sx) * dof + ((i / sx) % sy) * nx * dof + i / (sx * sy) * nx * ny * dof + d);
              pos++;
              }
            }
          }
        else if (pos == 0)
          {
          // Center
          TEST_EQUALITY(opart2->NumElements(sd, grp), (sx-1) * (sy-1) * (sz-1) * (dof - 1) + sx * sy * sz - 2);
          }
        }
      }
    if (pos == 0)
      {
      TEST_EQUALITY(totalNodes, sx * sy * sz * dof + (sx-1) * (sy-1) * (dof-1) + (sx-1) * (sz-1) * (dof-1) + (sy-1) * (sz-1) * (dof-1));
      }
    }
  }

TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Stokes3D, 1, 8, 8, 8, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Stokes3D, 2, 16, 16, 16, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Stokes3D, 3, 16, 8, 8, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Stokes3D, 4, 4, 4, 4, 2, 2, 2);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Stokes3D, 5, 8, 4, 4, 4, 4, 4);

TEUCHOS_UNIT_TEST_DECL(OverlappingPartitioner, SkewLaplace2D, nx, ny, sx, sy)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  HYMLS::Tools::InitializeIO(Comm);

  int nsx = nx / sx + 1;
  int nsy = ny / sy / 2;
  int nsl = nsx * nsy + nsx / 2;

  int dof = 1;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(nx*ny*dof, 0, *Comm));

  Teuchos::RCP<HYMLS::SkewCartesianPartitioner> part = Teuchos::rcp(new HYMLS::SkewCartesianPartitioner(map, nx, ny, 1, dof));
  part->Partition(sx, sy, 1, true);
  *map = *part->GetMap();

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::ParameterList &problemList = paramList->sublist("Problem");
  problemList.set("nx", nx);
  problemList.set("ny", ny);
  problemList.set("nz", 1);

  Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::rcp(
    Galeri::CreateCrsMatrix("Laplace2D", map.get(), problemList));

  problemList.set("Dimension", 2);
  problemList.set("Degrees of Freedom", dof);

  Teuchos::ParameterList &solverList = paramList->sublist("Preconditioner");
  solverList.set("Separator Length", sx);
  solverList.set("Coarsening Factor", 2);
  solverList.set("Partitioner", "Skew Cartesian");

  Teuchos::RCP<HYMLS::OverlappingPartitioner> opart2 = Teuchos::rcp(
    new HYMLS::OverlappingPartitioner(matrix, paramList, 0));

  for (int sd = 0; sd < opart2->NumMySubdomains(); sd++)
    {
    int gsd = opart2->Partitioner().SubdomainMap().GID(sd);
    int substart = part->First(sd);

    // Compute the number of groups we expect
    int numGrps = 9;
    // Right
    numGrps -= (gsd % nsx == nsx / 2 * 2) * 3;
    // Bottom
    numGrps -= (gsd > (nsl - nsx / 2 - 1)) * 3;
    // Left
    numGrps -= (gsd % nsx == nsx / 2) * 5;
    numGrps -= (gsd % nsx == 0);
    // Top
    numGrps -= (gsd < nsx / 2) * 5;
    numGrps -= (gsd >= nsx / 2 and gsd < nsx);

    if (numGrps < 4)
      numGrps = 4;

    TEST_EQUALITY(opart2->NumGroups(sd), numGrps);

    for (int grp = 0; grp < opart2->NumGroups(sd); grp++)
      {
      if (grp == 0)
        {
        // Interior
        if (gsd % nsx == nsx / 2 * 2)
          {
          // Right
          TEST_EQUALITY(opart2->NumElements(sd, grp), sx * sy);
          int m = 0;
          int pos = 0;
          for (int j = 0; j < sx * 2 - 1; j++)
            {
            for (int i = -m; i <= 0; i++)
              {
              TEST_EQUALITY(opart2->GID(sd, grp, pos), substart + i + j * nx);
              pos++;
              }
            if (j < sx - 1)
              m++;
            else
              m--;
            }
          }
        else if (gsd > (nsl - nsx / 2 - 1))
          {
          // Bottom
          TEST_EQUALITY(opart2->NumElements(sd, grp), sy * sx);
          int m = 0;
          int pos = 0;
          for (int j = 0; j < sx; j++)
            {
            for (int i = -m; i <= m; i++)
              {
              TEST_EQUALITY(opart2->GID(sd, grp, pos), substart + i + j * nx);
              pos++;
              }
            m++;
            }
          }
        else if (gsd % nsx == nsx / 2)
          {
          // Left
          TEST_EQUALITY(opart2->NumElements(sd, grp), sy * sx - sx - (sx - 1));
          int m = 1;
          int pos = 0;
          for (int j = 0; j < sx * 2 - 1; j++)
            {
            for (int i = 1; i < m; i++)
              {
              TEST_EQUALITY(opart2->GID(sd, grp, pos), substart + i + j * nx);
              pos++;
              }
            if (j < sx - 1)
              m++;
            else
              m--;
            }
          }
        else if (gsd < nsx / 2)
          {
          // Top
          TEST_EQUALITY(opart2->NumElements(sd, grp), sy * sx - sx - (sx - 1));
          int m = sx - 2;
          int pos = 0;
          for (int j = 0; j < sx - 1; j++)
            {
            for (int i = -m; i <= m; i++)
              {
              TEST_EQUALITY(opart2->GID(sd, grp, pos), substart + nx * sy + i + j * nx);
              pos++;
              }
            m--;
            }
          }
        else
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), 2 * sx * sy - sx - (sx - 1));
          int m = 0;
          int pos = 0;
          for (int j = 0; j < sx * 2 - 1; j++)
            {
            for (int i = -m; i <= m; i++)
              {
              TEST_EQUALITY(opart2->GID(sd, grp, pos), substart + i + j * nx);
              pos++;
              }
            if (j < sx - 1)
              m++;
            else
              m--;
            }
          }
        }
      else if (opart2->GID(sd, grp, 0) == substart + dof || opart2->GID(sd, grp, 0) == substart + nx * sy - sy + 1)
        {
        // Top left to bottom right
        TEST_EQUALITY(opart2->NumElements(sd, grp), sy - 1);
        for (int i = 0; i < opart2->NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart2->GID(sd, grp, i), opart2->GID(sd, grp, 0) + i * (nx + 1));
          }
        }
      else if (opart2->GID(sd, grp, 0) == substart - dof || opart2->GID(sd, grp, 0) == substart + nx * sy + sy - 1)
        {
        // Top right to bottom left
        TEST_EQUALITY(opart2->NumElements(sd, grp), sy - 1);
        for (int i = 0; i < opart2->NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart2->GID(sd, grp, i), opart2->GID(sd, grp, 0) + i * (nx - 1));
          }
        }
      else
        {
        // Corner
        TEST_EQUALITY(opart2->NumElements(sd, grp), 1);
        }
      }
    }
  }

TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewLaplace2D, 1, 8, 8, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewLaplace2D, 2, 16, 16, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewLaplace2D, 3, 16, 8, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewLaplace2D, 4, 4, 4, 2, 2);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewLaplace2D, 5, 64, 64, 16, 16);


TEUCHOS_UNIT_TEST_DECL(OverlappingPartitioner, SkewLaplace3D, nx, ny, nz, sx, sy, sz)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  HYMLS::Tools::InitializeIO(Comm);


  int nsx = nx / sx + 1;
  int nsy = ny / sy / 2;
  int nsl = nsx * nsy + nsx / 2;

  int dof = 1;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(nx*ny*nz*dof, 0, *Comm));

  Teuchos::RCP<HYMLS::SkewCartesianPartitioner> part = Teuchos::rcp(
    new HYMLS::SkewCartesianPartitioner(map, nx, ny, nz, dof));
  part->Partition(sx, sy, sz, true);
  *map = *part->GetMap();

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::ParameterList &problemList = paramList->sublist("Problem");
  problemList.set("nx", nx);
  problemList.set("ny", ny);
  problemList.set("nz", nz);

  Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::rcp(
    Galeri::CreateCrsMatrix("Laplace3D", map.get(), problemList));

  problemList.set("Dimension", 3);
  problemList.set("Degrees of Freedom", dof);

  Teuchos::ParameterList &solverList = paramList->sublist("Preconditioner");
  solverList.set("Separator Length", sx);
  solverList.set("Coarsening Factor", 2);
  solverList.set("Partitioner", "Skew Cartesian");

  Teuchos::RCP<HYMLS::OverlappingPartitioner> opart2 = Teuchos::rcp(
    new HYMLS::OverlappingPartitioner(matrix, paramList, 0));

  for (int sd = 0; sd < opart2->NumMySubdomains(); sd++)
    {
    int gsd = opart2->Partitioner().SubdomainMap().GID(sd);
    int substart = part->First(sd);

    // Compute the number of groups we expect
    int numGrps = 9 + 9 + 9;
    int leftOver = 9;
    // Right
    if ((gsd % nsl) % nsx == nsx / 2 * 2)
      {
      numGrps -= 9;
      leftOver -= 3;
      }
    // Bottom
    if ((gsd % nsl) > (nsl - nsx / 2 - 1))
      {
      numGrps -= 9;
      leftOver -= 3;
      }
    // Left
    if ((gsd % nsl) % nsx == nsx / 2)
      {
      numGrps -= 15;
      leftOver -= 5;
      }
    if ((gsd % nsl) % nsx == 0)
      {
      numGrps -= 3;
      leftOver -= 1;
      }
    // Top
    if ((gsd % nsl) < nsx / 2)
      {
      numGrps -= 15;
      leftOver -= 5;
      }
    if ((gsd % nsl) >= nsx / 2 and (gsd % nsl) < nsx)
      {
      numGrps -= 3;
      leftOver -= 1;
      }

    if (numGrps < 12)
      {
      numGrps = 12;
      leftOver = 4;
      }

    // Front
    if (gsd / nsl == 0)
      {
      numGrps -= leftOver;
      }

    TEST_EQUALITY(opart2->NumGroups(sd), numGrps);

    int totalNodes = 0;
    for (int grp = 0; grp < opart2->NumGroups(sd); grp++)
      {
      totalNodes += opart2->NumElements(sd, grp);
      if (grp == 0)
        {
        // Interior
        if (numGrps == 27)
          {
          // Center
          TEST_EQUALITY(opart2->NumElements(sd, grp), (2 * sy * sy - sy - sy + 1) * (sz - 1));
          int pos = 0;
          for (int k = 0; k < sz - 1; k++)
            {
            int m = 0;
            for (int j = 0; j < sx * 2 - 1; j++)
              {
              for (int i = -m; i <= m; i++)
                {
                TEST_EQUALITY(opart2->GID(sd, grp, pos), substart + i + j * nx + k * nx * ny);
                pos++;
                }
              if (j < sx - 1)
                m++;
              else
                m--;
              }
            }
          }
        }
      }
    if (numGrps == 27)
      {
      TEST_EQUALITY(totalNodes, sy * sy * 2 * (sz + 1) + (sz + 1) * (sy + sy + 1));
      }
    }
  }

TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewLaplace3D, 1, 8, 8, 8, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewLaplace3D, 2, 16, 16, 16, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewLaplace3D, 3, 16, 8, 8, 4, 4, 4);

TEUCHOS_UNIT_TEST_DECL(OverlappingPartitioner, SkewStokes2D, nx, ny, sx, sy)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  HYMLS::Tools::InitializeIO(Comm);

  int nsx = nx / sx + 1;
  int nsy = ny / sy / 2;
  int nsl = nsx * nsy + nsx / 2;

  int dof = 3;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(nx*ny*dof, 0, *Comm));

  Teuchos::RCP<HYMLS::SkewCartesianPartitioner> part = Teuchos::rcp(new HYMLS::SkewCartesianPartitioner(map, nx, ny, 1, dof));
  part->Partition(sx, sy, 1, true);
  *map = *part->GetMap();

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::ParameterList &problemList = paramList->sublist("Problem");
  problemList.set("nx", nx);
  problemList.set("ny", ny);
  problemList.set("nz", 1);

  Teuchos::ParameterList problemListCopy = problemList;
  Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::rcp(
    GaleriExt::CreateCrsMatrix("Stokes2D", map.get(), problemListCopy));

  for (int i = 0; i < 2; i++)
      {
      Teuchos::ParameterList& velList =
        problemList.sublist("Variable " + Teuchos::toString(i));
      velList.set("Variable Type", "Laplace");
      }

  Teuchos::ParameterList& presList =
    problemList.sublist("Variable "+Teuchos::toString(2));
  presList.set("Variable Type", "Retain 1");
  presList.set("Retain Isolated", true);

  problemList.set("Dimension", 2);
  problemList.set("Degrees of Freedom", dof);

  Teuchos::ParameterList &solverList = paramList->sublist("Preconditioner");
  solverList.set("Separator Length", sx);
  solverList.set("Coarsening Factor", 2);
  solverList.set("Partitioner", "Skew Cartesian");

  Teuchos::RCP<HYMLS::OverlappingPartitioner> opart2 = Teuchos::rcp(
    new HYMLS::OverlappingPartitioner(matrix, paramList, 0));

  for (int sd = 0; sd < opart2->NumMySubdomains(); sd++)
    {
    int gsd = opart2->Partitioner().SubdomainMap().GID(sd);
    int substart = part->First(sd);

    // Compute the number of groups we expect
    int numGrps = 8 + 4 + 1 + 1;
    // Right
    numGrps -= (gsd % nsx == nsx / 2 * 2) * 3;
    // Bottom
    numGrps -= (gsd > (nsl - nsx / 2 - 1)) * 5;
    // Left
    numGrps -= (gsd % nsx == nsx / 2) * 7;
    numGrps -= (gsd % nsx == 0);
    // Top
    numGrps -= (gsd < nsx / 2) * 7;
    numGrps -= (gsd >= nsx / 2 and gsd < nsx);

    if (numGrps < 7)
      numGrps = 7;

    TEST_EQUALITY(opart2->NumGroups(sd), numGrps);

    int totalNodes = 0;
    for (int grp = 0; grp < opart2->NumGroups(sd); grp++)
      {
      totalNodes += opart2->NumElements(sd, grp);
      if (grp == 0)
        {
        // Interior
        if (gsd % nsx == nsx / 2 * 2)
          {
          // Right
          TEST_EQUALITY(opart2->NumElements(sd, grp), sx * sy * 3 + sy - 1 - 1 + sy  - 1);
          int m = 0;
          int pos = 0;
          for (int j = 0; j < sx * 2 - 1; j++)
            {
            for (int i = -m; i <= 0; i++)
              for (int d = 0; d < dof; d++)
                {
                // First p-node in the interior
                if (d == 2 && j == 0 && i == -m)
                  continue;
                if (d == 1 && i == -m && j > sx - 1)
                  continue;
                if (d == 0 && i == -m && j == 0)
                  continue;
                TEST_EQUALITY(opart2->GID(sd, grp, pos), substart + i * dof + j * nx * dof + d);
                pos++;
                }
            if (j < sx - 1)
              m++;
            else if (j > sx - 1)
              m--;
            }
          }
        else if (gsd > (nsl - nsx / 2 - 1))
          {
          // Bottom
          TEST_EQUALITY(opart2->NumElements(sd, grp), sy * sx * 3 - 1 - sx);
          int m = 0;
          int pos = 0;
          for (int j = 0; j < sx; j++)
            {
            for (int i = -m; i <= m; i++)
              for (int d = 0; d < dof; d++)
                {
                // First p-node in the interior
                if (d == 2 && j == 0 && i == -m)
                  continue;
                if (d == 0 && i == m)
                  continue;
                TEST_EQUALITY(opart2->GID(sd, grp, pos), substart + i * dof + j * nx * dof + d);
                pos++;
                }
            m++;
            }
          }
        else if (gsd % nsx == nsx / 2)
          {
          // Left
          TEST_EQUALITY(opart2->NumElements(sd, grp), (sy * sx - sx - (sx - 1)) * 3 - 1);
          int m = 1;
          int pos = 0;
          for (int j = 0; j < sx * 2 - 1; j++)
            {
            for (int i = 1; i < m; i++)
              for (int d = 0; d < dof; d++)
                {
                // First p-node in the interior
                if (d == 2 && j == 1 && i == 1)
                  continue;
                if (d == 0 && i == m-1 && j <= sx - 1)
                  continue;
                if (d != 2 && i == m-1 && j > sx - 1)
                  continue;
                TEST_EQUALITY(opart2->GID(sd, grp, pos), substart + i * dof + j * nx * dof + d);
                pos++;
                }
            if (j < sx - 1)
              m++;
            else if (j > sx - 1)
              m--;
            }
          }
        else if (gsd < nsx / 2)
          {
          // Top
          TEST_EQUALITY(opart2->NumElements(sd, grp), (sy * sx - sx - (sx - 1)) * 3 + 2 * sx - 2 + sx - 1);
          int m = sx - 1;
          int pos = 0;
          for (int j = 0; j < sx - 1; j++)
            {
            for (int i = -m; i <= m; i++)
              for (int d = 0; d < dof; d++)
                {
                // First p-node in the interior
                if (d == 2 && j == 0 && i == -m)
                  continue;
                if ((d == 1 && (i == -m || i == m)) || (d == 0 && (i == m)))
                  continue;
                TEST_EQUALITY(opart2->GID(sd, grp, pos), substart + nx * sy * dof + i * dof + j * nx * dof + d);
                pos++;
                }
            m--;
            }
          }
        else
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sy * sy * 2 * 3 - (sx + sx - 1) - 1 - sx * 2);
          int m = 0;
          int pos = 0;
          for (int j = 0; j < sx * 2 - 1; j++)
            {
            for (int i = -m; i <= m; i++)
              for (int d = 0; d < dof; d++)
                {
                // First p-node in the interior
                if (d == 2 && j == 0 && i == -m)
                  continue;
                if (d == 1 && (i == -m || i == m) && j > sx - 1)
                  continue;
                if (d == 0 && ((i == m && j <= sx - 1) || (i == m && j > sx - 1)))
                  continue;
                TEST_EQUALITY(opart2->GID(sd, grp, pos), substart + i * dof + j * nx * dof + d);
                pos++;
                }
            if (j < sx - 1)
              m++;
            else if (j > sx - 1)
              m--;
            }
          }
        }
      else if (opart2->GID(sd, grp, 0) % dof != 0 &&
          (std::abs(opart2->GID(sd, grp, 0) - (substart + dof) - 0.5) < 1 ||
          std::abs(opart2->GID(sd, grp, 0) - (substart + nx * sy * dof - sy * dof + dof) - 0.5) < 1))
        {
        // Top left to bottom right
        TEST_EQUALITY(opart2->NumElements(sd, grp), sy - 1);
        for (int i = 0; i < opart2->NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart2->GID(sd, grp, i), opart2->GID(sd, grp, 0) + dof * i * (nx + 1));
          }
        }
      else if (opart2->GID(sd, grp, 0) % dof != 0 &&
          (std::abs(opart2->GID(sd, grp, 0) - (substart - dof) - 0.5) < 1 ||
          std::abs(opart2->GID(sd, grp, 0) - (substart + nx * sy * dof + sy * dof - dof) - 0.5) < 1))
        {
        // Top right to bottom left
        TEST_EQUALITY(opart2->NumElements(sd, grp), sy - 1);
        for (int i = 0; i < opart2->NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart2->GID(sd, grp, i), opart2->GID(sd, grp, 0) + dof * i * (nx - 1));
          }
        }
      else if (opart2->GID(sd, grp, 0) % dof == 0 &&
          (opart2->GID(sd, grp, 0) == substart ||
          opart2->GID(sd, grp, 0) == substart + dof * (nx+1) ||
          opart2->GID(sd, grp, 0) == substart + nx * sy * dof - sy * dof ||
          opart2->GID(sd, grp, 0) == substart + nx * sy * dof - sy * dof + dof * (nx+1)))
        {
        // Top left to bottom right
        if (gsd % nsx == nsx / 2 * 2 && opart2->GID(sd, grp, 0) == substart)
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), 1);
          }
        else if (opart2->GID(sd, grp, 0) == substart + dof * (nx+1) ||
          opart2->GID(sd, grp, 0) == substart + nx * sy * dof - sy * dof + dof * (nx+1))
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sy-1);
          }
        else
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sy);
          }
        for (int i = 0; i < opart2->NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart2->GID(sd, grp, i), opart2->GID(sd, grp, 0) + dof * i * (nx + 1));
          }
        }
      else if (opart2->GID(sd, grp, 0) % dof == 0 &&
          (opart2->GID(sd, grp, 0) == substart - dof ||
          opart2->GID(sd, grp, 0) == substart + nx * sy * dof + sy * dof - dof))
        {
        // Top right to bottom left
        if (gsd % nsx == nsx / 2 || (gsd % nsx == 0 && opart2->GID(sd, grp, 0) == substart - dof))
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sy-1);
          }
        else
          {
          TEST_EQUALITY(opart2->NumElements(sd, grp), sy);
          }
        for (int i = 0; i < opart2->NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart2->GID(sd, grp, i), opart2->GID(sd, grp, 0) + dof * i * (nx - 1));
          }
        }
      else
        {
        // Corner
        TEST_EQUALITY(opart2->NumElements(sd, grp), 1);
        }
      }
    if (numGrps == 14)
      {
      TEST_EQUALITY(totalNodes, sx * sy * 2 * 3 + (sx + sx + 1) + (sx + sx));
      }
    }
  }

TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes2D, 1, 8, 8, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes2D, 2, 16, 16, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes2D, 3, 16, 8, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes2D, 4, 4, 4, 2, 2);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes2D, 5, 64, 64, 16, 16);

TEUCHOS_UNIT_TEST_DECL(OverlappingPartitioner, SkewStokes3D, nx, ny, nz, sx, sy, sz)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  HYMLS::Tools::InitializeIO(Comm);

  int nsx = nx / sx + 1;
  int nsy = ny / sy / 2;
  int nsl = nsx * nsy + nsx / 2;

  int dof = 4;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(nx*ny*nz*dof, 0, *Comm));

  Teuchos::RCP<HYMLS::SkewCartesianPartitioner> part = Teuchos::rcp(new HYMLS::SkewCartesianPartitioner(map, nx, ny, nz, dof));
  part->Partition(sx, sy, sz, true);
  *map = *part->GetMap();

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::ParameterList &problemList = paramList->sublist("Problem");
  problemList.set("nx", nx);
  problemList.set("ny", ny);
  problemList.set("nz", nz);
  
  Teuchos::ParameterList problemListCopy = problemList;
  Teuchos::RCP<Epetra_CrsMatrix> matrix = Teuchos::rcp(
    GaleriExt::CreateCrsMatrix("Stokes3D", map.get(), problemListCopy));

  for (int i = 0; i < 3; i++)
      {
      Teuchos::ParameterList& velList =
        problemList.sublist("Variable " + Teuchos::toString(i));
      velList.set("Variable Type", "Laplace");
      }

  Teuchos::ParameterList& presList =
    problemList.sublist("Variable "+Teuchos::toString(3));
  presList.set("Variable Type", "Retain 1");
  presList.set("Retain Isolated", true);

  problemList.set("Dimension", 3);
  problemList.set("Degrees of Freedom", dof);

  Teuchos::ParameterList &solverList = paramList->sublist("Preconditioner");
  solverList.set("Separator Length", sx);
  solverList.set("Coarsening Factor", 2);
  solverList.set("Partitioner", "Skew Cartesian");

  Teuchos::RCP<HYMLS::OverlappingPartitioner> opart2 = Teuchos::rcp(
    new HYMLS::OverlappingPartitioner(matrix, paramList, 0));

  for (int sd = 0; sd < opart2->NumMySubdomains(); sd++)
    {
    int gsd = opart2->Partitioner().SubdomainMap().GID(sd);

    // Compute the number of groups we expect
    int numGrps = 23 + 21 + 23 + 1;
    int leftOver = 23;
    // Right
    if ((gsd % nsl) % nsx == nsx / 2 * 2)
      {
      numGrps -= 18;
      leftOver -= 6;
      }
    // Bottom
    if ((gsd % nsl) > (nsl - nsx / 2 - 1))
      {
      numGrps -= 24;
      leftOver -= 8;
      }
    // Left
    if ((gsd % nsl) % nsx == nsx / 2)
      {
      numGrps -= 36;
      leftOver -= 12;
      }
    if ((gsd % nsl) % nsx == 0)
      {
      numGrps -= 6;
      leftOver -= 2;
      }
    // Top
    if ((gsd % nsl) < nsx / 2)
      {
      numGrps -= 36;
      leftOver -= 12;
      }
    if ((gsd % nsl) >= nsx / 2 and (gsd % nsl) < nsx)
      {
      numGrps -= 6;
      leftOver -= 2;
      }

    if (numGrps < 32)
      {
      numGrps = 32;
      leftOver = 11;
      }

    // Front
    if (gsd / nsl == 0)
      {
      numGrps -= leftOver;
      }

    TEST_EQUALITY(opart2->NumGroups(sd), numGrps);

    int totalNodes = 0;
    for (int grp = 0; grp < opart2->NumGroups(sd); grp++)
      {
      totalNodes += opart2->NumElements(sd, grp);
      }
    if (numGrps == 23 + 21 + 23 + 1)
      {
      TEST_EQUALITY(totalNodes,
        (sx * sy * 2 * 3 + 2 * (sx + sx + 1) + (sx + sx)) * (sz + 1) +
        sx * sy * 2 * sz);
      }
    }
  }

TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes3D, 1, 8, 8, 8, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes3D, 2, 16, 16, 16, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes3D, 3, 16, 8, 8, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes3D, 4, 16, 16, 16, 8, 8, 8);
