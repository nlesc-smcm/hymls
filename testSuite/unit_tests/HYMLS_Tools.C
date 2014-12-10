#include "HYMLS_Tools.H"
#include "HYMLS_MatrixUtils.H"

#include "Epetra_Map.h"
#include "Epetra_SerialComm.h"

#include "HYMLS_UnitTests.H"

// A class for which we can set the number of processors that are available.
class SplitBoxComm : public Epetra_SerialComm {
    int numProc_;
public:
    SplitBoxComm(): Epetra_SerialComm(), numProc_() {};
    SplitBoxComm(const SplitBoxComm& Comm): Epetra_SerialComm(Comm), numProc_(Comm.numProc_) {};
    Epetra_Comm *Clone() const {return dynamic_cast<Epetra_Comm *>(new SplitBoxComm(*this));};
    int NumProc() const {return numProc_;}
    int SetNumProc(int num) {numProc_ = num;}
};

TEUCHOS_UNIT_TEST(Tools, SplitBox)
  {
  int nx, ny, nz;

  // Test normal SplitBox behaviour
  HYMLS::Tools::SplitBox(16, 16, 16, 4, nx, ny, nz);
  TEST_EQUALITY(nx, 1);
  TEST_EQUALITY(ny, 2);
  TEST_EQUALITY(nz, 2);

  HYMLS::Tools::SplitBox(32, 32, 32, 8, nx, ny, nz);
  TEST_EQUALITY(nx, 2);
  TEST_EQUALITY(ny, 2);
  TEST_EQUALITY(nz, 2);

  // Previously this gave 9 8 6
  HYMLS::Tools::SplitBox(192, 192, 192, 432, nx, ny, nz, 8, 8, 8);
  TEST_EQUALITY(nx, 6);
  TEST_EQUALITY(ny, 6);
  TEST_EQUALITY(nz, 12);

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

  // Test SplitBox berhaviour combined with MatrixUtils::CreateMap in the way
  // it is used in CartesianPartitioner
  SplitBoxComm comm;
  int nx_in, sx_in, nparts_in, npx;
  Teuchos::RCP<Epetra_Map> map;

  nx_in = 192;
  sx_in = 8;
  nparts_in = 432;
  TEST_EQUALITY(HYMLS::Tools::SplitBox(nx_in, nx_in, nx_in, nparts_in,
                                       nx, ny, nz, sx_in, sx_in, sx_in), 0);
  TEST_EQUALITY(nx, 6);
  TEST_EQUALITY(ny, 6);
  TEST_EQUALITY(nz, 12);

  comm.SetNumProc(nparts_in);
  npx = nx_in / sx_in;

  map = HYMLS::MatrixUtils::CreateMap(npx, npx, npx, 1, 0, comm, nparts_in);
  TEST_EQUALITY(npx / nx * npx / ny * npx / nz, map->NumMyElements());

  TEST_EQUALITY(HYMLS::Tools::SplitBox(160, 160, 160, 256, nx, ny, nz, 8, 8, 8), 1);

  nx_in = 160;
  sx_in = 16;
  nparts_in = 250;
  TEST_EQUALITY(HYMLS::Tools::SplitBox(nx_in, nx_in, nx_in, nparts_in,
                                       nx, ny, nz, sx_in, sx_in, sx_in), 0);
  TEST_EQUALITY(nx, 5);
  TEST_EQUALITY(ny, 5);
  TEST_EQUALITY(nz, 10);

  comm.SetNumProc(nparts_in);
  npx = nx_in / sx_in;

  map = HYMLS::MatrixUtils::CreateMap(npx, npx, npx, 1, 0, comm, nparts_in);
  TEST_EQUALITY(npx / nx * npx / ny * npx / nz, map->NumMyElements());

  nx_in = 80;
  sx_in = 16;
  nparts_in = 25;
  TEST_EQUALITY(HYMLS::Tools::SplitBox(nx_in, nx_in, nx_in, nparts_in,
                                       nx, ny, nz, sx_in, sx_in, sx_in), 0);
  TEST_EQUALITY(nx, 1);
  TEST_EQUALITY(ny, 5);
  TEST_EQUALITY(nz, 5);

  comm.SetNumProc(nparts_in);
  npx = nx_in / sx_in;

  map = HYMLS::MatrixUtils::CreateMap(npx, npx, npx, 1, 0, comm, nparts_in);
  TEST_EQUALITY(npx / nx * npx / ny * npx / nz, map->NumMyElements());
  }
