#include "HYMLS_Tools.H"

#include "HYMLS_UnitTests.H"

TEUCHOS_UNIT_TEST(Tools, SplitBox)
  {
  int nx, ny, nz;

  HYMLS::Tools::SplitBox(16, 16, 16, 4, nx, ny, nz);
  TEST_EQUALITY(nx, 1);
  TEST_EQUALITY(ny, 2);
  TEST_EQUALITY(nz, 2);

  HYMLS::Tools::SplitBox(32, 32, 32, 8, nx, ny, nz);
  TEST_EQUALITY(nx, 2);
  TEST_EQUALITY(ny, 2);
  TEST_EQUALITY(nz, 2);

  HYMLS::Tools::SplitBox(192, 192, 192, 432, nx, ny, nz);
  TEST_EQUALITY(nx, 6);
  TODO_TEST_EQUALITY(ny, 6);
  TODO_TEST_EQUALITY(nz, 12);

  HYMLS::Tools::SplitBox(160, 160, 160, 256, nx, ny, nz);
  TEST_EQUALITY(nx, 4);
  TEST_EQUALITY(ny, 8);
  TEST_EQUALITY(nz, 8);

  HYMLS::Tools::SplitBox(80, 80, 80, 32, nx, ny, nz);
  TEST_EQUALITY(nx, 2);
  TEST_EQUALITY(ny, 4);
  TEST_EQUALITY(nz, 4);

  HYMLS::Tools::SplitBox(80, 80, 80, 25, nx, ny, nz);
  TEST_EQUALITY(nx, 1);
  TEST_EQUALITY(ny, 5);
  TEST_EQUALITY(nz, 5);
  }
