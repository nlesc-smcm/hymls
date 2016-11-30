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

  HYMLS::OverlappingPartitioner opart(matrix, paramList, 0);

  for (int sd = 0; sd < opart.NumMySubdomains(); sd++)
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

    TEST_EQUALITY(opart.NumGroups(sd), numGrps);

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
      else if (opart.GID(sd, grp, 0) == substart - nx || opart.GID(sd, grp, 0) == substart + nx * (sx - 1))
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
      else if (opart.GID(sd, grp, 0) == substart + sy - 1 || opart.GID(sd, grp, 0) == substart - 1)
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
        TEST_EQUALITY(opart.GID(sd, grp, 0), substart + nx * (sy - 1) + sx - 1);
        }
      }
    }
  }

TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace2D, 1, 8, 8, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace2D, 2, 16, 16, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace2D, 3, 16, 8, 4, 4);
TEUCHOS_UNIT_TEST_INST(OverlappingPartitioner, Laplace2D, 4, 4, 4, 2, 2);
