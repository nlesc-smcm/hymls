#include "HYMLS_OverlappingPartitioner.H"

#include "HYMLS_CartesianPartitioner.H"

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
      // And groups
      for (int grp = 1; grp < NumGroups(sd); grp++)
        {
        int found = 0;
        Teuchos::Array<int> group = GetGroup(sd, grp);
        // Check if the node is in the domain
        for (int lid = partitioner_->First(sd); lid < partitioner_->First(sd+1); lid++)
          {
          int gid = partitioner_->Map().GID(lid);
          if (group[0] == gid)
            {
            found = 4;
            break;
            }
          }
        
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
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(nx*ny, 0, *Comm));

  Teuchos::RCP<HYMLS::CartesianPartitioner> part = Teuchos::rcp(new HYMLS::CartesianPartitioner(map, nx, ny, 1, dof));
  part->Partition(nsx * nsy, true);
  Teuchos::RCP<Epetra_Map> subdomainMap = part->CreateSubdomainMap(0);
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
    int gsd = subdomainMap->GID(sd);
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
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(nx*ny*nz, 0, *Comm));

  Teuchos::RCP<HYMLS::CartesianPartitioner> part = Teuchos::rcp(new HYMLS::CartesianPartitioner(map, nx, ny, nz, dof));
  part->Partition(nsx * nsy * nsz, true);
  Teuchos::RCP<Epetra_Map> subdomainMap = part->CreateSubdomainMap(0);
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
    int gsd = subdomainMap->GID(sd);
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
