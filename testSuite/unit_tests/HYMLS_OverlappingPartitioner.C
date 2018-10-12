#include "HYMLS_OverlappingPartitioner.H"

#include "HYMLS_CartesianPartitioner.H"
#include "HYMLS_SkewCartesianPartitioner.H"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"

#include <numeric>

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

TEUCHOS_UNIT_TEST_DECL(OverlappingPartitioner, Laplace2D, nx, ny, sx, sy)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  DISABLE_OUTPUT;

  int nsx = nx / sx;
  int nsy = ny / sy;

  int dof = 1;

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::ParameterList &problemList = paramList->sublist("Problem");
  problemList.set("nx", nx);
  problemList.set("ny", ny);
  problemList.set("nz", 1);

  problemList.set("Dimension", 2);
  problemList.set("Degrees of Freedom", dof);

  Teuchos::ParameterList &solverList = paramList->sublist("Preconditioner");
  solverList.set("Separator Length (x)", sx);
  solverList.set("Separator Length (y)", sy);
  solverList.set("Coarsening Factor", 2);

  Teuchos::RCP<HYMLS::CartesianPartitioner> part = Teuchos::rcp(
    new HYMLS::CartesianPartitioner(Teuchos::null, paramList, *Comm));
  part->Partition(true);
  Teuchos::RCP<const Epetra_Map> map = part->GetMap();
  HYMLS::OverlappingPartitioner opart(map, paramList, 0);

  ENABLE_OUTPUT;
  for (int sd = 0; sd < opart.NumMySubdomains(); sd++)
    {
    int gsd = part->SubdomainMap().GID(sd);
    hymls_gidx substart = gsd % nsx * nx / nsx * dof +
      gsd / nsx * ny / nsy * dof * nx;

    // Compute the number of groups we expect
    std::vector<int> isGroup(9, 1);

    // Right
    if ((gsd + 1) % nsx == 0)
      {
      isGroup[2] = 0;
      isGroup[5] = 0;
      isGroup[8] = 0;
      }
    // Bottom
    if (gsd / nsx == nsy - 1)
      {
      isGroup[6] = 0;
      isGroup[7] = 0;
      isGroup[8] = 0;
      }
    // Left
    if (gsd % nsx == 0)
      {
      isGroup[0] = 0;
      isGroup[3] = 0;
      isGroup[6] = 0;
      }
    // Top
    if (gsd / nsx == 0)
      {
      isGroup[0] = 0;
      isGroup[1] = 0;
      isGroup[2] = 0;
      }
    int numGroups = std::accumulate(isGroup.begin(), isGroup.end(), 0);

    TEST_EQUALITY(opart.NumGroups(sd), numGroups);
    TEST_EQUALITY(opart.NumGroups(sd)-1, opart.NumLinks(sd));

    for (int grp = 0; grp < opart.NumGroups(sd); grp++)
      {
      if (grp == 0)
        {
        // Interior
        if ((gsd + 1) % nsx == 0 && gsd / nsx == nsy - 1)
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), sx * sy);
          for (int i = 0; i < opart.NumElements(sd, grp); i++)
            {
            TEST_EQUALITY(opart.GID(sd, grp, i), substart + i % sx + i / sx * nx);
            }
          }
        else if ((gsd + 1) % nsx == 0)
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), sx * (sy-1));
          for (int i = 0; i < opart.NumElements(sd, grp); i++)
            {
            TEST_EQUALITY(opart.GID(sd, grp, i), substart + i % sx + i / sx * nx);
            }
          }
        else if (gsd / nsx == nsy - 1)
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), sy * (sx-1));
          for (int i = 0; i < opart.NumElements(sd, grp); i++)
            {
            TEST_EQUALITY(opart.GID(sd, grp, i), substart + i % (sx-1) + i / (sx-1) * nx);
            }
          }
        else
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), (sx-1) * (sy-1));
          for (int i = 0; i < opart.NumElements(sd, grp); i++)
            {
            TEST_EQUALITY(opart.GID(sd, grp, i), substart + i % (sx-1) + i / (sx-1) * nx);
            }
          }
        }
      else if (opart.GID(sd, grp, 0) == substart - nx || opart.GID(sd, grp, 0) == substart + nx * (sy - 1))
        {
        // Top or bottom border
        if ((gsd + 1) % nsx == 0)
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), sx);
          }
        else
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), sx - 1);
          }
        for (int i = 0; i < opart.NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart.GID(sd, grp, i), opart.GID(sd, grp, 0) + i);
          }
        }
      else if (opart.GID(sd, grp, 0) == substart + sx - 1 || opart.GID(sd, grp, 0) == substart - 1)
        {
        // Left or right border
        if (gsd / nsx == nsy - 1)
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), sy);
          }
        else
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), sy - 1);
          }
        for (int i = 0; i < opart.NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart.GID(sd, grp, i), opart.GID(sd, grp, 0) + i * nx);
          }
        }
      else
        {
        // Corner
        TEST_EQUALITY(opart.NumElements(sd, grp), 1);
        }
      }
    }
  }

TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace2D, 1, 8, 8, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace2D, 2, 16, 16, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace2D, 3, 16, 8, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace2D, 4, 4, 4, 2, 2);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace2D, 5, 64, 64, 16, 16);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace2D, 6, 64, 60, 16, 10);

TEUCHOS_UNIT_TEST_DECL(OverlappingPartitioner, Laplace3D, nx, ny, nz, sx, sy, sz)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  DISABLE_OUTPUT;

  int nsx = nx / sx;
  int nsy = ny / sy;
  int nsz = nz / sz;

  int dof = 1;

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::ParameterList &problemList = paramList->sublist("Problem");
  problemList.set("nx", nx);
  problemList.set("ny", ny);
  problemList.set("nz", nz);

  problemList.set("Dimension", 3);
  problemList.set("Degrees of Freedom", dof);

  Teuchos::ParameterList &solverList = paramList->sublist("Preconditioner");
  solverList.set("Separator Length (x)", sx);
  solverList.set("Separator Length (y)", sy);
  solverList.set("Separator Length (z)", sz);
  solverList.set("Coarsening Factor", 2);

  Teuchos::RCP<HYMLS::CartesianPartitioner> part = Teuchos::rcp(
    new HYMLS::CartesianPartitioner(Teuchos::null, paramList, *Comm));
  part->Partition(true);
  Teuchos::RCP<const Epetra_Map> map = part->GetMap();
  HYMLS::OverlappingPartitioner opart(map, paramList, 0);

  ENABLE_OUTPUT;
  for (int sd = 0; sd < opart.NumMySubdomains(); sd++)
    {
    int gsd = part->SubdomainMap().GID(sd);
    hymls_gidx substart = gsd % nsx * nx / nsx * dof +
      (gsd % (nsx * nsy)) / nsx * ny / nsy * dof * nx +
      gsd / (nsx * nsy) * nz / nsz * dof * nx * ny;

    // Compute the number of groups we expect
    std::vector<int> isGroup(27, 1);
    // Right
    if ((gsd + 1) % nsx == 0)
      {
      for (int i = 2; i < 27; i += 3)
        isGroup[i] = 0;
      }
    // Bottom
    if ((gsd % (nsx * nsy)) / nsx == nsy - 1)
      {
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          isGroup[6+i+j*9] = 0;
      }
    // Back
    if (gsd / (nsx * nsy) == nsz - 1)
      {
      for (int i = 18; i < 27; i++)
        isGroup[i] = 0;
      }
    // Left
    if (gsd % nsx == 0)
      {
      for (int i = 0; i < 27; i += 3)
        isGroup[i] = 0;
      }
    // Top
    if ((gsd % (nsx * nsy)) / nsx == 0)
      {
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          isGroup[i+j*9] = 0;
      }
    // Front
    if (gsd / (nsx * nsy) == 0)
      {
      for (int i = 0; i < 9; i++)
        isGroup[i] = 0;
      }
    int numGroups = std::accumulate(isGroup.begin(), isGroup.end(), 0);

    TEST_EQUALITY(opart.NumGroups(sd), numGroups);
    TEST_EQUALITY(opart.NumGroups(sd)-1, opart.NumLinks(sd));

    int totalNodes = 0;
    for (int grp = 0; grp < opart.NumGroups(sd); grp++)
      {
      totalNodes += opart.NumElements(sd, grp);
      if (grp == 0)
        {
        // Interior
        if (isGroup[14] == 0 && isGroup[16] == 0 && isGroup[22] == 0)
          {
          // Right back bottom
          TEST_EQUALITY(opart.NumElements(sd, grp), sx * sy * sz);
          for (int i = 0; i < opart.NumElements(sd, grp); i++)
            {
            TEST_EQUALITY(opart.GID(sd, grp, i), substart + i % sx + ((i / sx) % sy) * nx + i / (sx * sy) * nx * ny);
            }
          }
        else if (numGroups == 27)
          {
          // Center
          TEST_EQUALITY(opart.NumElements(sd, grp), (sx-1) * (sy-1) * (sz-1));
          }
        }
      }
    if (numGroups == 27)
      {
      TEST_EQUALITY(totalNodes, sx * sy * sz + (sx + 1) * (sy + 1) + (sx + 1) * sz + sy * sz);
      }
    }
  }

TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace3D, 1, 8, 8, 8, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace3D, 2, 16, 16, 16, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace3D, 3, 16, 8, 8, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace3D, 4, 4, 4, 4, 2, 2, 2);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace3D, 5, 8, 4, 4, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace3D, 6, 16, 15, 12, 4, 5, 3);


TEUCHOS_UNIT_TEST_DECL(OverlappingPartitioner, Stokes2D, nx, ny, sx, sy)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  DISABLE_OUTPUT;

  int nsx = nx / sx;
  int nsy = ny / sy;

  int dof = 3;

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::ParameterList &problemList = paramList->sublist("Problem");
  problemList.set("nx", nx);
  problemList.set("ny", ny);
  problemList.set("nz", 1);

  for (int i = 0; i < 2; i++)
    {
    Teuchos::ParameterList& velList =
      problemList.sublist("Variable " + Teuchos::toString(i));
    velList.set("Variable Type", "Velocity");
    }

  Teuchos::ParameterList& presList =
    problemList.sublist("Variable "+Teuchos::toString(2));
  presList.set("Variable Type", "Pressure");

  problemList.set("Dimension", 2);
  problemList.set("Degrees of Freedom", dof);

  Teuchos::ParameterList &solverList = paramList->sublist("Preconditioner");
  solverList.set("Separator Length", sx);
  solverList.set("Coarsening Factor", 2);

  Teuchos::RCP<HYMLS::CartesianPartitioner> part = Teuchos::rcp(
    new HYMLS::CartesianPartitioner(Teuchos::null, paramList, *Comm));
  part->Partition(true);
  Teuchos::RCP<const Epetra_Map> map = part->GetMap();
  HYMLS::OverlappingPartitioner opart(map, paramList, 0);

  ENABLE_OUTPUT;
  for (int sd = 0; sd < opart.NumMySubdomains(); sd++)
    {
    int gsd = part->SubdomainMap().GID(sd);
    hymls_gidx substart = gsd % nsx * nx / nsx * dof +
      gsd / nsx * ny / nsy * dof * nx;

    // Compute the number of groups we expect
    std::vector<int> isGroup(9, 1);

    // Right
    if ((gsd + 1) % nsx == 0)
      {
      isGroup[2] = 0;
      isGroup[5] = 0;
      isGroup[8] = 0;
      }
    // Bottom
    if (gsd / nsx == nsy - 1)
      {
      isGroup[6] = 0;
      isGroup[7] = 0;
      isGroup[8] = 0;
      }
    // Left
    if (gsd % nsx == 0)
      {
      isGroup[0] = 0;
      isGroup[3] = 0;
      isGroup[6] = 0;
      }
    // Top
    if (gsd / nsx == 0)
      {
      isGroup[0] = 0;
      isGroup[1] = 0;
      isGroup[2] = 0;
      }
    int numGroups = std::accumulate(isGroup.begin(), isGroup.end(), 0) * 2 // velocities
      - 1 // interiors
      + 1 // interior pressure
      + isGroup[8]; // corner pressures

    TEST_EQUALITY(opart.NumGroups(sd), numGroups);

    for (int grp = 0; grp < opart.NumGroups(sd); grp++)
      {
      if (grp == 0)
        {
        // Interior
        if ((gsd + 1) % nsx == 0 && gsd / nsx == nsy - 1)
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), sx * sy * dof - 1);
          int pos = 0;
          for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++)
              for (int d = 0; d < dof; d++)
                if (!(d == 2 && pos == 2) && !(d == 2 && x == sx-1 && y == sy-1))
                  {
                  int gid = opart.GID(sd, grp, pos++);
                  TEST_EQUALITY(gid, substart + x * dof + y * nx * dof + d);
                  }
          }
        else if ((gsd + 1) % nsx == 0)
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), sx * (sy-1) * 2 + sx * sy - 1);
          int pos = 0;
          for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++)
              for (int d = 0; d < dof; d++)
                if (((x < sx && y < sy - 1) || d == 2)
                  && !(d == 2 && pos == 2) && !(d == 2 && x == sx-1 && y == sy-1))
                  {
                  int gid = opart.GID(sd, grp, pos++);
                  TEST_EQUALITY(gid, substart + x * dof + y * nx * dof + d);
                  }
          }
        else if (gsd / nsx == nsy - 1)
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), sy * (sx-1) * 2 + sx * sy - 1);
          int pos = 0;
          for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++)
              for (int d = 0; d < dof; d++)
                if (((x < sx - 1 && y < sy) || d == 2)
                  && !(d == 2 && pos == 2) && !(d == 2 && x == sx-1 && y == sy-1))
                  {
                  int gid = opart.GID(sd, grp, pos++);
                  TEST_EQUALITY(gid, substart + x * dof + y * nx * dof + d);
                  }
          }
        else
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), (sx-1) * (sy-1) * 2 + sx * sy - 2);
          int pos = 0;
          for (int y = 0; y < sy; y++)
            for (int x = 0; x < sx; x++)
              for (int d = 0; d < dof; d++)
                if (((x < sx - 1 && y < sy - 1) || d == 2)
                  && !(d == 2 && pos == 2) && !(d == 2 && x == sx-1 && y == sy-1))
                  {
                  int gid = opart.GID(sd, grp, pos++);
                  TEST_EQUALITY(gid, substart + x * dof + y * nx * dof + d);
                  }
          }
        }
      else if (opart.GID(sd, grp, 0) / dof == substart / dof - nx || opart.GID(sd, grp, 0) / dof == substart / dof + nx * (sx - 1))
        {
        // Right border
        if ((gsd + 1) % nsx == 0)
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), sx);
          }
        else
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), sx - 1);
          }
        for (int i = 0; i < opart.NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart.GID(sd, grp, i), opart.GID(sd, grp, 0) + i * dof);
          }
        }
      else if (opart.GID(sd, grp, 0) / dof == substart / dof + sy - 1 || opart.GID(sd, grp, 0) / dof == substart / dof - 1)
        {
        // Bottom border
        if (gsd / nsx == nsy - 1)
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), sy);
          }
        else
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), sy - 1);
          }
        for (int i = 0; i < opart.NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart.GID(sd, grp, i), opart.GID(sd, grp, 0) + i * nx * dof);
          }
        }
      else
        {
        // Corner
        TEST_EQUALITY(opart.NumElements(sd, grp), 1);
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

  DISABLE_OUTPUT;

  int nsx = nx / sx;
  int nsy = ny / sy;
  int nsz = nz / sz;

  int dof = 4;

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::ParameterList &problemList = paramList->sublist("Problem");
  problemList.set("nx", nx);
  problemList.set("ny", ny);
  problemList.set("nz", nz);

  for (int i = 0; i < 3; i++)
    {
    Teuchos::ParameterList& velList =
      problemList.sublist("Variable " + Teuchos::toString(i));
    velList.set("Variable Type", "Velocity");
    }

  Teuchos::ParameterList& presList =
    problemList.sublist("Variable "+Teuchos::toString(3));
  presList.set("Variable Type", "Pressure");

  problemList.set("Dimension", 3);
  problemList.set("Degrees of Freedom", dof);

  Teuchos::ParameterList &solverList = paramList->sublist("Preconditioner");
  solverList.set("Separator Length", sx);
  solverList.set("Coarsening Factor", 2);

  Teuchos::RCP<HYMLS::CartesianPartitioner> part = Teuchos::rcp(
    new HYMLS::CartesianPartitioner(Teuchos::null, paramList, *Comm));
  part->Partition(true);
  Teuchos::RCP<const Epetra_Map> map = part->GetMap();
  HYMLS::OverlappingPartitioner opart(map, paramList, 0);

  ENABLE_OUTPUT;
  for (int sd = 0; sd < opart.NumMySubdomains(); sd++)
    {
    int gsd = part->SubdomainMap().GID(sd);
    hymls_gidx substart = gsd % nsx * nx / nsx * dof +
      (gsd % (nsx * nsy)) / nsx * ny / nsy * dof * nx +
      gsd / (nsx * nsy) * nz / nsz * dof * nx * ny;

    // Compute the number of groups we expect
    std::vector<int> isGroup(27, 1);
    // Right
    if ((gsd + 1) % nsx == 0)
      {
      for (int i = 2; i < 27; i += 3)
        isGroup[i] = 0;
      }
    // Bottom
    if ((gsd % (nsx * nsy)) / nsx == nsy - 1)
      {
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          isGroup[6+i+j*9] = 0;
      }
    // Back
    if (gsd / (nsx * nsy) == nsz - 1)
      {
      for (int i = 18; i < 27; i++)
        isGroup[i] = 0;
      }
    // Left
    if (gsd % nsx == 0)
      {
      for (int i = 0; i < 27; i += 3)
        isGroup[i] = 0;
      }
    // Top
    if ((gsd % (nsx * nsy)) / nsx == 0)
      {
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          isGroup[i+j*9] = 0;
      }
    // Front
    if (gsd / (nsx * nsy) == 0)
      {
      for (int i = 0; i < 9; i++)
        isGroup[i] = 0;
      }
    int numGroups = std::accumulate(isGroup.begin(), isGroup.end(), 0) * 3 // velocities
      - 2 // interiors
      + 1 // interior pressure
      + isGroup[17] + isGroup[23] + isGroup[25] + isGroup[26]; // corner pressures

    TEST_EQUALITY(opart.NumGroups(sd), numGroups);

    int totalNodes = 0;
    for (int grp = 0; grp < opart.NumGroups(sd); grp++)
      {
      totalNodes += opart.NumElements(sd, grp);
      if (grp == 0)
        {
        // Interior
        if (isGroup[14] == 0 && isGroup[16] == 0 && isGroup[22] == 0)
          {
          // Right back bottom
          TEST_EQUALITY(opart.NumElements(sd, grp), sx * sy * sz * dof - 1);
          int pos = 0;
          for (int i = 0; i < opart.NumElements(sd, grp) / dof; i++)
            {
            for (int d = 0; d < dof; d++)
              {
              if (d == 3 && pos == 3)
                continue;
              TEST_EQUALITY(opart.GID(sd, grp, pos), substart + (i % sx) * dof + ((i / sx) % sy) * nx * dof + i / (sx * sy) * nx * ny * dof + d);
              pos++;
              }
            }
          }
        else if (numGroups == 27 * 3 - 2 + 1 + 4)
          {
          // Center
          TEST_EQUALITY(opart.NumElements(sd, grp), (sx-1) * (sy-1) * (sz-1) * dof - 1 + (sx-1) * (sy-1) + (sx-1) * (sz-1) + (sy-1) * (sz-1));
          }
        }
      }
    if (numGroups == 27 * 3 - 2 + 1 + 4)
      {
      TEST_EQUALITY(totalNodes, sx * sy * sz * dof + ((sx + 1) * (sy + 1) + (sx + 1) * sz + sy * sz) * (dof-1));
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

  DISABLE_OUTPUT;

  // Old sx size (the version of before July 2017). Should give the same subdomains
  // as the new version with sx = old sx * 2
  int osx = sx / 2;
  int osy = sy / 2;

  int nsx = nx / osx + 1;
  int nsy = ny / osy / 2;
  int nsl = nsx * nsy + nsx / 2;

  int npx = nx / sx;
  int npy = ny / sy;

  int totNum2DCubes = npx * npy; // number of cubes for fixed z
  int numPerLayer = 2 * totNum2DCubes + npx + npy; // domains for fixed z
  int numPerRow = 2*npx + 1; // domains in a row (both lattices); fixed y

  int dof = 1;

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::ParameterList &problemList = paramList->sublist("Problem");
  problemList.set("nx", nx);
  problemList.set("ny", ny);
  problemList.set("nz", 1);

  problemList.set("Dimension", 2);
  problemList.set("Degrees of Freedom", dof);

  Teuchos::ParameterList &solverList = paramList->sublist("Preconditioner");
  solverList.set("Separator Length", sx);
  solverList.set("Coarsening Factor", 2);
  solverList.set("Partitioner", "Skew Cartesian");

  Teuchos::RCP<HYMLS::SkewCartesianPartitioner> part = Teuchos::rcp(
    new HYMLS::SkewCartesianPartitioner(Teuchos::null, paramList, *Comm));
  part->Partition(true);
  Teuchos::RCP<const Epetra_Map> map = part->GetMap();
  HYMLS::OverlappingPartitioner opart(map, paramList, 0);

  ENABLE_OUTPUT;
  for (int sd = 0; sd < opart.NumMySubdomains(); sd++)
    {
    int gsd = part->SubdomainMap().GID(sd);

    // Get domain coordinates and its first node
    // Considers superposed lattices
    int Z = gsd / numPerLayer;
    double Y = ((gsd - Z * numPerLayer) / numPerRow) - 0.5;
    double X = (gsd - Z * numPerLayer) % numPerRow;
    if (X >= npx)
      {
      X -= npx + 0.5;
      Y += 0.5;
      }

    hymls_gidx substart = dof * sx * (X + Y * nx) + dof * (sx / 2 - 1);

    // Compute the number of groups we expect
    int numGroups = 9;
    // Right
    numGroups -= (gsd % nsx == nsx / 2 * 2) * 3;
    // Bottom
    numGroups -= (gsd > (nsl - nsx / 2 - 1)) * 3;
    // Left
    numGroups -= (gsd % nsx == nsx / 2) * 5;
    numGroups -= (gsd % nsx == 0);
    // Top
    numGroups -= (gsd < nsx / 2) * 5;
    numGroups -= (gsd >= nsx / 2 and gsd < nsx);

    if (numGroups < 4)
      numGroups = 4;

    TEST_EQUALITY(opart.NumGroups(sd), numGroups);
    TEST_EQUALITY(opart.NumGroups(sd)-1, opart.NumLinks(sd));

    for (int grp = 0; grp < opart.NumGroups(sd); grp++)
      {
      if (grp == 0)
        {
        // Interior
        if (gsd % nsx == nsx / 2 * 2)
          {
          // Right
          TEST_EQUALITY(opart.NumElements(sd, grp), osx * osy);
          int m = 0;
          int pos = 0;
          for (int j = 0; j < osx * 2 - 1; j++)
            {
            for (int i = -m; i <= 0; i++)
              {
              TEST_EQUALITY(opart.GID(sd, grp, pos), substart + i + j * nx);
              pos++;
              }
            if (j < osx - 1)
              m++;
            else
              m--;
            }
          }
        else if (gsd > (nsl - nsx / 2 - 1))
          {
          // Bottom
          TEST_EQUALITY(opart.NumElements(sd, grp), osy * osx);
          int m = 0;
          int pos = 0;
          for (int j = 0; j < osx; j++)
            {
            for (int i = -m; i <= m; i++)
              {
              TEST_EQUALITY(opart.GID(sd, grp, pos), substart + i + j * nx);
              pos++;
              }
            m++;
            }
          }
        else if (gsd % nsx == nsx / 2)
          {
          // Left
          TEST_EQUALITY(opart.NumElements(sd, grp), osy * osx - osx - (osx - 1));
          int m = 1;
          int pos = 0;
          for (int j = 0; j < osx * 2 - 1; j++)
            {
            for (int i = 1; i < m; i++)
              {
              TEST_EQUALITY(opart.GID(sd, grp, pos), substart + i + j * nx);
              pos++;
              }
            if (j < osx - 1)
              m++;
            else
              m--;
            }
          }
        else if (gsd < nsx / 2)
          {
          // Top
          TEST_EQUALITY(opart.NumElements(sd, grp), osy * osx - osx - (osx - 1));
          int m = osx - 2;
          int pos = 0;
          for (int j = 0; j < osx - 1; j++)
            {
            for (int i = -m; i <= m; i++)
              {
              TEST_EQUALITY(opart.GID(sd, grp, pos), substart + nx * osy + i + j * nx);
              pos++;
              }
            m--;
            }
          }
        else
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), 2 * osx * osy - osx - (osx - 1));
          int m = 0;
          int pos = 0;
          for (int j = 0; j < osx * 2 - 1; j++)
            {
            for (int i = -m; i <= m; i++)
              {
              TEST_EQUALITY(opart.GID(sd, grp, pos), substart + i + j * nx);
              pos++;
              }
            if (j < osx - 1)
              m++;
            else
              m--;
            }
          }
        }
      else if (opart.GID(sd, grp, 0) == substart + dof || opart.GID(sd, grp, 0) == substart + nx * osy - osy + 1)
        {
        // Top left to bottom right
        TEST_EQUALITY(opart.NumElements(sd, grp), osy - 1);
        for (int i = 0; i < opart.NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart.GID(sd, grp, i), opart.GID(sd, grp, 0) + i * (nx + 1));
          }
        }
      else if (opart.GID(sd, grp, 0) == substart - dof || opart.GID(sd, grp, 0) == substart + nx * osy + osy - 1)
        {
        // Top right to bottom left
        TEST_EQUALITY(opart.NumElements(sd, grp), osy - 1);
        for (int i = 0; i < opart.NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart.GID(sd, grp, i), opart.GID(sd, grp, 0) + i * (nx - 1));
          }
        }
      else
        {
        // Corner
        TEST_EQUALITY(opart.NumElements(sd, grp), 1);
        }
      }
    }
  }

TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewLaplace2D, 1, 8, 8, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewLaplace2D, 2, 16, 16, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewLaplace2D, 3, 16, 8, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewLaplace2D, 5, 16, 16, 8, 8);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewLaplace2D, 6, 64, 64, 16, 16);


TEUCHOS_UNIT_TEST_DECL(OverlappingPartitioner, SkewStokes2D, nx, ny, sx, sy)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  DISABLE_OUTPUT;

  // Old sx size (the version of before July 2017). Should give the same subdomains
  // as the new version with sx = old sx * 2
  int osx = sx / 2;
  int osy = sy / 2;

  int nsx = nx / osx + 1;
  int nsy = ny / osy / 2;
  int nsl = nsx * nsy + nsx / 2;

  int npx = nx / sx;
  int npy = ny / sy;

  int totNum2DCubes = npx * npy; // number of cubes for fixed z
  int numPerLayer = 2 * totNum2DCubes + npx + npy; // domains for fixed z
  int numPerRow = 2*npx + 1; // domains in a row (both lattices); fixed y

  int dof = 3;

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::ParameterList &problemList = paramList->sublist("Problem");
  problemList.set("nx", nx);
  problemList.set("ny", ny);
  problemList.set("nz", 1);

  for (int i = 0; i < 2; i++)
    {
    Teuchos::ParameterList& velList =
      problemList.sublist("Variable " + Teuchos::toString(i));
    velList.set("Variable Type", "Velocity");
    }

  Teuchos::ParameterList& presList =
    problemList.sublist("Variable "+Teuchos::toString(2));
  presList.set("Variable Type", "Pressure");

  problemList.set("Dimension", 2);
  problemList.set("Degrees of Freedom", dof);

  Teuchos::ParameterList &solverList = paramList->sublist("Preconditioner");
  solverList.set("Separator Length", sx);
  solverList.set("Coarsening Factor", 2);
  solverList.set("Partitioner", "Skew Cartesian");

  Teuchos::RCP<HYMLS::SkewCartesianPartitioner> part = Teuchos::rcp(
    new HYMLS::SkewCartesianPartitioner(Teuchos::null, paramList, *Comm));
  part->Partition(true);
  Teuchos::RCP<const Epetra_Map> map = part->GetMap();
  HYMLS::OverlappingPartitioner opart(map, paramList, 0);

  ENABLE_OUTPUT;
  for (int sd = 0; sd < opart.NumMySubdomains(); sd++)
    {
    int gsd = part->SubdomainMap().GID(sd);

    // Get subdomain number for test output
    TEST_EQUALITY(gsd, gsd);

    // Get domain coordinates and its first node
    // Considers superposed lattices
    int Z = gsd / numPerLayer;
    double Y = ((gsd - Z * numPerLayer) / numPerRow) - 0.5;
    double X = (gsd - Z * numPerLayer) % numPerRow;
    if (X >= npx)
      {
      X -= npx + 0.5;
      Y += 0.5;
      }
    hymls_gidx substart = dof * sx * (X + Y * nx) + dof * (sx / 2 - 1);
    bool somewhatBottom = gsd <= (nsl - nsx / 2 - 1) and gsd > nsl - nsx;

    // Compute the number of groups we expect
    int numGroups = 8 + 4 + 1 + 1;
    // Right
    numGroups -= (gsd % nsx == nsx / 2 * 2) * 5;
    // Bottom
    numGroups -= (gsd > (nsl - nsx / 2 - 1)) * 7;
    numGroups -= somewhatBottom;
    // Left
    numGroups -= (gsd % nsx == nsx / 2) * 7;
    numGroups -= (gsd % nsx == 0);
    // Top
    numGroups -= (gsd < nsx / 2) * 7;
    numGroups -= (gsd >= nsx / 2 and gsd < nsx);

    if (numGroups < 7)
      numGroups = 7;
 
    TEST_EQUALITY(opart.NumGroups(sd), numGroups);

    // Compute the number of links we expect
    int numLinks = 4 + 4 + 1;
    // Right
    numLinks -= (gsd % nsx == nsx / 2 * 2) * 3;
    // Bottom
    numLinks -= (gsd > (nsl - nsx / 2 - 1)) * 5;
    numLinks -= somewhatBottom;
    // Left
    numLinks -= (gsd % nsx == nsx / 2) * 5;
    numLinks -= (gsd % nsx == 0);
    // Top
    numLinks -= (gsd < nsx / 2) * 5;
    numLinks -= (gsd >= nsx / 2 and gsd < nsx);

    if (numLinks < 4)
      numLinks = 4;

    TEST_EQUALITY(opart.NumLinks(sd), numLinks);

    int totalNodes = 0;
    for (int grp = 0; grp < opart.NumGroups(sd); grp++)
      {
      totalNodes += opart.NumElements(sd, grp);
      if (grp == 0)
        {
        // Interior
        if (gsd % nsx == nsx / 2 * 2)
          {
          // Right
          TEST_EQUALITY(opart.NumElements(sd, grp), osx * osy * 3 + osy + osy  - 1 + somewhatBottom);
          int m = 0;
          int pos = 0;
          for (int j = 0; j < osx * 2 - 1; j++)
            {
            for (int i = -m; i <= 0; i++)
              for (int d = 0; d < dof; d++)
                {
                // First p-node in the interior
                if (d == 2 && j == 0 && i == -m)
                  continue;
                if (d == 1 && i == -m && j > osx - 1 && !(j == 0 && somewhatBottom))
                  continue;
                TEST_EQUALITY(opart.GID(sd, grp, pos), substart + i * dof + j * nx * dof + d);
                pos++;
                }
            if (j < osx - 1)
              m++;
            else if (j > osx - 1)
              m--;
            }
          }
        else if (gsd > (nsl - nsx / 2 - 1))
          {
          // Bottom
          TEST_EQUALITY(opart.NumElements(sd, grp), osy * osx * 3 - 1 - osx);
          int m = 0;
          int pos = 0;
          for (int j = 0; j < osx; j++)
            {
            for (int i = -m; i <= m; i++)
              for (int d = 0; d < dof; d++)
                {
                // First p-node in the interior
                if (d == 2 && j == 0 && i == -m)
                  continue;
                if (d == 0 && i == m)
                  continue;
                TEST_EQUALITY(opart.GID(sd, grp, pos), substart + i * dof + j * nx * dof + d);
                pos++;
                }
            m++;
            }
          }
        else if (gsd % nsx == nsx / 2)
          {
          // Left
          TEST_EQUALITY(opart.NumElements(sd, grp), (osy * osx - osx - (osx - 1)) * 3 - 1);
          int m = 1;
          int pos = 0;
          for (int j = 0; j < osx * 2 - 1; j++)
            {
            for (int i = 1; i < m; i++)
              for (int d = 0; d < dof; d++)
                {
                // First p-node in the interior
                if (d == 2 && j == 1 && i == 1)
                  continue;
                if (d == 0 && i == m-1 && j <= osx - 1)
                  continue;
                if (d != 2 && i == m-1 && j > osx - 1)
                  continue;
                TEST_EQUALITY(opart.GID(sd, grp, pos), substart + i * dof + j * nx * dof + d);
                pos++;
                }
            if (j < osx - 1)
              m++;
            else if (j > osx - 1)
              m--;
            }
          }
        else if (gsd < nsx / 2)
          {
          // Top
          TEST_EQUALITY(opart.NumElements(sd, grp), (osy * osx - osx - (osx - 1)) * 3 + 2 * osx - 2 + osx - 1);
          int m = osx - 1;
          int pos = 0;
          for (int j = 0; j < osx - 1; j++)
            {
            for (int i = -m; i <= m; i++)
              for (int d = 0; d < dof; d++)
                {
                // First p-node in the interior
                if (d == 2 && j == 0 && i == -m)
                  continue;
                if ((d == 1 && (i == -m || i == m)) || (d == 0 && (i == m)))
                  continue;
                TEST_EQUALITY(opart.GID(sd, grp, pos), substart + nx * osy * dof + i * dof + j * nx * dof + d);
                pos++;
                }
            m--;
            }
          }
        else
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), osy * osy * 2 * 3 - (osx + osx - 1) - 1 - osx * 2 + somewhatBottom);
          int m = 0;
          int pos = 0;
          for (int j = 0; j < osx * 2 - 1; j++)
            {
            for (int i = -m; i <= m; i++)
              for (int d = 0; d < dof; d++)
                {
                // First p-node in the interior
                if (d == 2 && j == 0 && i == -m)
                  continue;
                if (d == 1 && (i == -m || i == m) && j > osx - 1 && !(j == 0 && somewhatBottom))
                  continue;
                if (d == 0 && ((i == m && j <= osx - 1) || (i == m && j > osx - 1)))
                  continue;
                TEST_EQUALITY(opart.GID(sd, grp, pos), substart + i * dof + j * nx * dof + d);
                pos++;
                }
            if (j < osx - 1)
              m++;
            else if (j > osx - 1)
              m--;
            }
          }
        }
      else if (opart.GID(sd, grp, 0) % dof != 0 &&
          (std::abs(opart.GID(sd, grp, 0) - (substart + dof) - 0.5) < 1 ||
          std::abs(opart.GID(sd, grp, 0) - (substart + nx * osy * dof - osy * dof + dof) - 0.5) < 1))
        {
        // Top left to bottom right
        TEST_EQUALITY(opart.NumElements(sd, grp), osy - 1);
        for (int i = 0; i < opart.NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart.GID(sd, grp, i), opart.GID(sd, grp, 0) + dof * i * (nx + 1));
          }
        }
      else if (opart.GID(sd, grp, 0) % dof != 0 &&
          (std::abs(opart.GID(sd, grp, 0) - (substart - dof) - 0.5) < 1 ||
          std::abs(opart.GID(sd, grp, 0) - (substart + nx * osy * dof + osy * dof - dof) - 0.5) < 1))
        {
        // Top right to bottom left
        TEST_EQUALITY(opart.NumElements(sd, grp), osy - 1);
        for (int i = 0; i < opart.NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart.GID(sd, grp, i), opart.GID(sd, grp, 0) + dof * i * (nx - 1));
          }
        }
      else if (opart.GID(sd, grp, 0) % dof == 0 &&
          (opart.GID(sd, grp, 0) == substart ||
          opart.GID(sd, grp, 0) == substart + dof * (nx+1) ||
          opart.GID(sd, grp, 0) == substart + nx * osy * dof - osy * dof ||
          opart.GID(sd, grp, 0) == substart + nx * osy * dof - osy * dof + dof * (nx+1)))
        {
        // Top left to bottom right
        if (gsd % nsx == nsx / 2 * 2 && opart.GID(sd, grp, 0) == substart)
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), 1);
          }
        else if (opart.GID(sd, grp, 0) == substart + dof * (nx+1) ||
          opart.GID(sd, grp, 0) == substart + nx * osy * dof - osy * dof + dof * (nx+1))
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), osy-1);
          }
        else
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), osy);
          }
        for (int i = 0; i < opart.NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart.GID(sd, grp, i), opart.GID(sd, grp, 0) + dof * i * (nx + 1));
          }
        }
      else if (opart.GID(sd, grp, 0) % dof == 0 &&
          (opart.GID(sd, grp, 0) == substart - dof ||
          opart.GID(sd, grp, 0) == substart + nx * osy * dof + osy * dof - dof))
        {
        // Top right to bottom left
        if (gsd % nsx == nsx / 2 || (gsd % nsx == 0 && opart.GID(sd, grp, 0) == substart - dof))
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), osy-1);
          }
        else
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), osy);
          }
        for (int i = 0; i < opart.NumElements(sd, grp); i++)
          {
          TEST_EQUALITY(opart.GID(sd, grp, i), opart.GID(sd, grp, 0) + dof * i * (nx - 1));
          }
        }
      else
        {
        // Corner
        TEST_EQUALITY(opart.NumElements(sd, grp), 1);
        }
      }
    if (numGroups == 14)
      {
      TEST_EQUALITY(totalNodes, osx * osy * 2 * 3 + (osx + osx + 1) + (osx + osx));
      }
    }
  }

TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes2D, 1, 8, 8, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes2D, 2, 16, 16, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes2D, 3, 16, 8, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes2D, 5, 16, 16, 8, 8);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes2D, 6, 64, 64, 16, 16);

TEUCHOS_UNIT_TEST_DECL(OverlappingPartitioner, SkewStokes3D, nx, ny, nz, sx, sy, sz)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));

  DISABLE_OUTPUT;

  int dof = 4;

  Teuchos::RCP<Teuchos::ParameterList> paramList = Teuchos::rcp(new Teuchos::ParameterList);
  Teuchos::ParameterList &problemList = paramList->sublist("Problem");
  problemList.set("nx", nx);
  problemList.set("ny", ny);
  problemList.set("nz", nz);

  for (int i = 0; i < 3; i++)
    {
    Teuchos::ParameterList& velList =
      problemList.sublist("Variable " + Teuchos::toString(i));
    velList.set("Variable Type", "Velocity");
    }

  Teuchos::ParameterList& presList =
    problemList.sublist("Variable "+Teuchos::toString(3));
  presList.set("Variable Type", "Pressure");

  problemList.set("Dimension", 3);
  problemList.set("Degrees of Freedom", dof);

  Teuchos::ParameterList &solverList = paramList->sublist("Preconditioner");
  solverList.set("Separator Length", sx);
  solverList.set("Coarsening Factor", 2);
  solverList.set("Partitioner", "Skew Cartesian");

  Teuchos::RCP<HYMLS::SkewCartesianPartitioner> part = Teuchos::rcp(
    new HYMLS::SkewCartesianPartitioner(Teuchos::null, paramList, *Comm));
  part->Partition(true);
  Teuchos::RCP<const Epetra_Map> map = part->GetMap();
  HYMLS::OverlappingPartitioner opart(map, paramList, 0);

  ENABLE_OUTPUT;
  for (int sd = 0; sd < opart.NumMySubdomains(); sd++)
    {
    int totalNodes[4] = {0, 0, 0, 0};
    for (int grp = 0; grp < opart.NumGroups(sd); grp++)
      {
      for (int pos = 0; pos < opart.NumElements(sd, grp); pos++)
        totalNodes[opart.GID(sd, grp, pos) % dof]++;
      }
    if (opart.NumGroups(sd) == 84)
      {
      TEST_EQUALITY(totalNodes[0],
        sx / 2 * ((sx - 1) * sx + sx ) +
        2 + (sx / 2 + 2 + sx ) /2 * (sx / 2-1) * 4 + sx / 2 * 2);
      TEST_EQUALITY(totalNodes[1],
        sx / 2 * ((sx - 1) * sx + sx ) +
        sx + 1 + (sx / 2 + 2 + sx ) /2 * (sx / 2-1) * 4 + sx / 2 * 2);
      TEST_EQUALITY(totalNodes[2],
        sx / 2 * ((sx - 1) * sx + sx ) + (sx / 2 + 1) * (sx * 3-2));
      TEST_EQUALITY(totalNodes[3],
        sx / 2 * ((sx - 1) * sx + sx ));
      }
    }
  }

TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes3D, 1, 8, 8, 8, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes3D, 2, 16, 16, 16, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes3D, 3, 16, 8, 8, 4, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, SkewStokes3D, 4, 16, 16, 16, 8, 8, 8);
