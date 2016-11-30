#include "HYMLS_OverlappingPartitioner.H"

#include "HYMLS_CartesianPartitioner.H"

#include "Galeri_CrsMatrices.h"
#include "GaleriExt_CrsMatrices.h"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

#include "Epetra_MpiComm.h"
#include "Epetra_Map.h"

#include "HYMLS_UnitTests.H"

TEUCHOS_UNIT_TEST(OverlappingPartitioner, Laplace2D)
  {
  Teuchos::RCP<Epetra_MpiComm> Comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
  HYMLS::Tools::InitializeIO(Comm);

  int nx = 16;
  int ny = 16;

  int sx = 4;
  int sy = 4;

  int nsx = nx / sx;
  int nsy = ny / sy;

  int dof = 1;
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(nx*ny, 0, *Comm));

  Teuchos::RCP<HYMLS::CartesianPartitioner> part = Teuchos::rcp(new HYMLS::CartesianPartitioner(map, nx, ny, 1, dof));
  part->Partition(sx * sy, true);
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
          TEST_EQUALITY(opart.NumElements(sd, grp), 16);
          for (int i = 0; i < opart.NumElements(sd, grp); i++)
            {
            TEST_EQUALITY(opart.GID(sd, grp, i), substart + i % sx + i / sx * nx);
            }
          }
        else if ((gsd + 1) % nsx == 0)
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), 12);
          for (int i = 0; i < opart.NumElements(sd, grp); i++)
            {
            TEST_EQUALITY(opart.GID(sd, grp, i), substart + i % sx + i / sx * nx);
            }
          }
        else if (gsd / nsx == nsy - 1)
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), 12);
          for (int i = 0; i < opart.NumElements(sd, grp); i++)
            {
            TEST_EQUALITY(opart.GID(sd, grp, i), substart + i % (sx-1) + i / (sx-1) * nx);
            }
          }
        else
          {
          TEST_EQUALITY(opart.NumElements(sd, grp), 9);
          for (int i = 0; i < opart.NumElements(sd, grp); i++)
            {
            TEST_EQUALITY(opart.GID(sd, grp, i), substart + i % (sx-1) + i / (sx-1) * nx);
            }
          }
        }
      else if (opart.NumElements(sd, grp) > 1)
        {
        // Borders
        if (opart.GID(sd, grp, 0) == substart - nx || opart.GID(sd, grp, 0) == substart + nx * (sx - 1))
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
          // Should not happen
          TEST_EQUALITY(true, false);
          }
        }
      else
        {
        // Corner
        TEST_EQUALITY(opart.NumElements(sd, grp), 1);
        TEST_EQUALITY(opart.GID(sd, grp, 0), substart + nx * (sx - 1) - 1 + nx / nsx);
        }
      }
    }
  }
