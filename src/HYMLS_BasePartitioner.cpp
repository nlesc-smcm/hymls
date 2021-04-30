#include "HYMLS_BasePartitioner.hpp"

#include "HYMLS_config.h"

#include "HYMLS_Tools.hpp"
#include "HYMLS_MatrixUtils.hpp"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StrUtils.hpp"
#include "Teuchos_Array.hpp"

#include "Teuchos_StandardParameterEntryValidators.hpp"

#include "Epetra_Map.h"
#include "Epetra_Comm.h"
#include "Epetra_MpiComm.h"
#include "Epetra_Distributor.h"

#ifdef HYMLS_TESTING
#include "HYMLS_Tester.hpp"
#endif

namespace HYMLS {

BasePartitioner::BasePartitioner(Epetra_Comm const &comm, int level)
  :
  comm_(comm.Clone()),
  myLevel_(level)
  {}

void BasePartitioner::SetParameters(Teuchos::ParameterList& params)
  {
  HYMLS_PROF3("BasePartitioner", "setParameterList");

  Teuchos::ParameterList& probList = params.sublist("Problem");
  Teuchos::ParameterList& precList = params.sublist("Preconditioner");

  dim_ = probList.get("Dimension", 3);
  int pvar = -1;

  nx_ = probList.get("nx", -1);
  ny_ = probList.get("ny", nx_);
  nz_ = probList.get("nz", dim_ > 2 ? nx_ : 1);

  if (nx_ == -1)
    Tools::Error("You must presently specify nx, ny (and possibly nz) in the 'Problem' sublist",
      __FILE__, __LINE__);

  bool xperio = false;
  bool yperio = false;
  bool zperio = false;
  xperio = probList.get("x-periodic", xperio);
  if (dim_ >= 1) yperio = probList.get("y-periodic", yperio);
  if (dim_ >= 2) zperio = probList.get("z-periodic", zperio);

  perio_ = GaleriExt::NO_PERIO;

  if (xperio) perio_ = (GaleriExt::PERIO_Flag)(perio_ | GaleriExt::X_PERIO);
  if (yperio) perio_ = (GaleriExt::PERIO_Flag)(perio_ | GaleriExt::Y_PERIO);
  if (zperio) perio_ = (GaleriExt::PERIO_Flag)(perio_ | GaleriExt::Z_PERIO);

  perio_ = (GaleriExt::PERIO_Flag)probList.get("Periodicity", (int)perio_);

  sx_ = -1;
  sy_ = -1;
  sz_ = nz_ > 1 ? -1 : 1;

  if (precList.isParameter("Separator Length (x)"))
    sx_ = precList.get("Separator Length (x)", sx_);
  if (precList.isParameter("Separator Length (y)"))
    sy_ = precList.get("Separator Length (y)", sy_);
  if (precList.isParameter("Separator Length (z)"))
    sz_ = precList.get("Separator Length (z)", sz_);

  if (sx_ == -1)
    sx_ = precList.get("Separator Length", 4);
  if (sy_ == -1)
    sy_ = precList.get("Separator Length", sx_);
  if (sz_ == -1)
    sz_ = precList.get("Separator Length", sx_);

  if (sx_ <= 1)
    Tools::Error("Separator Length not set correctly",
      __FILE__, __LINE__);

  cx_ = -1;
  cy_ = -1;
  cz_ = nz_ > 1 ? -1 : 1;

  if (precList.isParameter("Coarsening Factor (x)"))
    cx_ = precList.get("Coarsening Factor (x)", cx_);
  if (precList.isParameter("Coarsening Factor (y)"))
    cy_ = precList.get("Coarsening Factor (y)", cy_);
  if (precList.isParameter("Coarsening Factor (z)"))
    cz_ = precList.get("Coarsening Factor (z)", cz_);

  if (cx_ == -1)
    cx_ = precList.get("Coarsening Factor", sx_);
  if (cy_ == -1)
    cy_ = precList.get("Coarsening Factor", cx_);
  if (cz_ == -1)
    cz_ = precList.get("Coarsening Factor", cx_);

  if (cx_ <= 1)
    Tools::Error("Coarsening Factor not set correctly",
      __FILE__, __LINE__);

  rx_ = -1;
  ry_ = -1;
  rz_ = -1;

  std::string retainAtLevel = "Retain Nodes at Level " + Teuchos::toString(myLevel_);
  if (precList.isParameter("Retain Nodes (x)"))
    rx_ = precList.get("Retain Nodes (x)", rx_);
  if (precList.isParameter(retainAtLevel + " (x)"))
    rx_ = precList.get(retainAtLevel + " (x)", rx_);
  if (precList.isParameter("Retain Nodes (y)"))
    ry_ = precList.get("Retain Nodes (y)", ry_);
  if (precList.isParameter(retainAtLevel + " (y)"))
    ry_ = precList.get(retainAtLevel + " (y)", ry_);
  if (precList.isParameter("Retain Nodes (z)"))
    rz_ = precList.get("Retain Nodes (z)", rz_);
  if (precList.isParameter(retainAtLevel + " (z)"))
    rz_ = precList.get(retainAtLevel + " (z)", rz_);

  if (rx_ == -1 && precList.isParameter(retainAtLevel))
      rx_ = precList.get(retainAtLevel, rx_);
  if (rx_ == -1)
    rx_ = precList.get("Retain Nodes", rx_);
  if (ry_ == -1 && precList.isParameter(retainAtLevel))
      ry_ = precList.get(retainAtLevel, ry_);
  if (ry_ == -1)
    ry_ = precList.get("Retain Nodes", ry_);
  if (rz_ == -1 && precList.isParameter(retainAtLevel))
      rz_ = precList.get(retainAtLevel, rz_);
  if (rz_ == -1)
    rz_ = precList.get("Retain Nodes", rz_);

  link_retained_nodes_ = precList.get("Eliminate Retained Nodes Together", true);

  link_velocities_ = precList.get("Eliminate Velocities Together", false);

  if (probList.isParameter("Equations"))
    {
    std::string eqn = probList.get("Equations", "Undefined Problem");
    bool is_complex = probList.get("Complex Arithmetic", false);
    int factor = is_complex ? 2 : 1;

    if (eqn == "Laplace")
      {
      if (!is_complex)
        {
        probList.get("Degrees of Freedom", 1);
        probList.sublist("Variable 0").get("Variable Type", "Laplace");
        }
      else
        {
        probList.get("Degrees of Freedom", 2);
        probList.sublist("Variable 0").get("Variable Type", "Laplace");
        probList.sublist("Variable 1").get("Variable Type", "Laplace");
        }
      }
    else if (eqn.rfind("Stokes") == 0 || eqn == "Bous-C")
      {
      if (eqn == "Bous-C")
        {
        probList.get("Degrees of Freedom", dim_ + 2);
        pvar = probList.get("Pressure Variable", dim_ + 1);
        }
      else
        {
        probList.get("Degrees of Freedom", dim_ + 1);
        pvar = probList.get("Pressure Variable", dim_);
        }

      dof_ = probList.get("Degrees of Freedom", 1);
      for (int i = 0; i < dim_ * factor; i++)
        probList.sublist("Variable " + Teuchos::toString(i)).get("Variable Type", "Velocity");

      for (int i = pvar * factor; i < pvar * factor + factor; i++)
        probList.sublist("Variable " + Teuchos::toString(i)).get("Variable Type", "Pressure");

      for (int i = 0; i < dof_; i++)
        if (!probList.isSublist("Variable " + Teuchos::toString(i)))
          probList.sublist("Variable " + Teuchos::toString(i)).get("Variable Type", "Laplace");

#ifdef HYMLS_TESTING
      if (eqn == "Stokes-C")
        probList.get("Test F-Matrix Properties", true);
      else
        probList.get("Test F-Matrix Properties", false);
#endif

      if (eqn == "Stokes-B" || eqn == "Stokes-L" || eqn == "Stokes-T")
        {
        /*
          we assume the following 'augmented B-grid',
          where the @ are dummy p-nodes, * are p-nodes
          and > are v-nodes. To transform this into an
          F-matrix, one has to apply a Givvens rotation
          to the velocity field (giving an F-grid).
          This currently has to be done manually outside
          the solver/preconditioner.

          >---->---->---->>---->---->---->
          @ | *  |  * |  * ||  * |  * | *  |
          >---->---->---->>---->---->---->
          @ | *  |  * |  * ||  * |  * | *  |
          >---->---->---->>---->---->---->
          @ |  * |  * | *  || *  |  * | *  |
          >====>====>====>>====>====>====>
          @ | *  |  * |  * ||  * |  * | *  |
          >---->---->---->>---->---->---->
          @ |  * |  * |  * || *  | *  |  * |
          >---->---->---->>---->---->---->
          @ |  * |  * | *  ||  * | *  |  * |
          >---->---->---->>---->---->---->
          @    @    @    @    @    @     @
        */
        // case of one subdomain per partition not implemented for B-grid
        if (is_complex)
          Tools::Error("complex Stokes-B not implemented", __FILE__, __LINE__);

        retainPressures_ = probList.get("Retained Pressure Nodes", 2);

        if (precList.get("Fix Pressure Level", true))
          {
          // we fix the singularity by inserting a Dirichlet condition for
          // global pressure in cells 0 and 1, since we retain two pressures
          // per subdomain both will be retained until the coarsest grid.
          precList.get("Fix GID 1", factor * pvar);
          precList.get("Fix GID 2", factor * dof_ + factor * pvar);
          }
        }
      else
        {
        if (precList.get("Fix Pressure Level", true))
          {
          // we fix the singularity by inserting a Dirichlet condition for
          // the first global pressure node
          precList.get("Fix GID 1", factor * pvar);
          if (is_complex) precList.get("Fix GID 2", factor * pvar + 1);
          }
        retainPressures_ = probList.get("Retained Pressure Nodes", 1);
        }
      }
    else
      {
      Tools::Error("'Equations' parameter not recognized",
        __FILE__, __LINE__);
      }
    }

  if (!probList.isParameter("Degrees of Freedom"))
    {
    HYMLS::Tools::Error(
      "At this point, the 'Problem' sublist must contain 'Degrees of Freedom'\n"
      "If you do not set 'Equations', you have to set a (among others) this one.\n",
      __FILE__, __LINE__);
    }

  dof_ = probList.get("Degrees of Freedom", 1);
  retainPressures_ = probList.get("Retained Pressure Nodes", 1);

  variableType_.resize(dof_);

  int pcount = 0;
  int vcount = 0;
  for (int i = 0; i < dof_; i++)
    {
    Teuchos::ParameterList& varList = probList.sublist("Variable " + Teuchos::toString(i));
    std::string variableType = varList.get("Variable Type", "Laplace");

    if (variableType == "Laplace")
      variableType_[i] = VariableType::Velocity_V;
    else if (variableType == "Velocity U" || (variableType == "Velocity" && vcount == 0))
      {
      variableType_[i] = VariableType::Velocity_U;
      vcount++;
      }
    else if (variableType == "Velocity V" || (variableType == "Velocity" && vcount == 1))
      {
      variableType_[i] = VariableType::Velocity_V;
      vcount++;
      }
    else if (variableType == "Velocity W" || (variableType == "Velocity" && vcount == 2))
      {
      variableType_[i] = VariableType::Velocity_W;
      vcount++;
      }
    else if (variableType == "Pressure")
      {
      pvar = i;
      variableType_[i] = VariableType::Pressure;
      pcount++;
      }
    else if (variableType == "Interior")
      variableType_[i] = VariableType::Interior;
    else
      Tools::Error("Variable type " + variableType + " does not exist",
                   __FILE__, __LINE__);
    }

  if (pcount > 1)
    Tools::Error("Can only have one 'Pressure' variable",
      __FILE__, __LINE__);

  probList.get("Pressure Variable", pvar);

#ifdef HYMLS_TESTING
  Tester::nx_ = nx_;
  Tester::ny_ = ny_;
  Tester::nz_ = nz_;
  Tester::dim_ = dim_;
  Tester::dof_ = dof_;
  Tester::pvar_ = pvar;
  Tester::doFmatTests_ = probList.get("Test F-Matrix Properties", false);
#endif
  }

void BasePartitioner::SetNextLevelParameters(Teuchos::ParameterList& params) const
  {
  Teuchos::ParameterList& precList = params.sublist("Preconditioner");

  int new_sx = sx_ * cx_;
  int new_sy = sy_ * cy_;
  int new_sz = sz_ * cz_;

  if (precList.isParameter("Separator Length (x)"))
    {
    precList.set("Separator Length (x)", new_sx);
    precList.set("Separator Length (y)", new_sy);
    precList.set("Separator Length (z)", new_sz);
    }
  else
    precList.set("Separator Length", new_sx);

  if (precList.isParameter("Coarsening Factor (x)"))
    {
    precList.set("Coarsening Factor (x)", cx_);
    precList.set("Coarsening Factor (y)", cy_);
    precList.set("Coarsening Factor (z)", cz_);
    }
  else
    precList.set("Coarsening Factor", cx_);
  }

int FindCoarseningFactor(int cx)
  {
  int b = 1;
  while (b < cx)
    {
    for (int p = 0; p < cx; p++)
      if (pow(b, p) == cx)
        return b;
    b += 1;
    }
  return cx;
  }

int BasePartitioner::CreatePIDMap()
  {
  HYMLS_PROF2("BasePartitioner", "CreatePIDMap");

  int sx = sx_;
  int sy = sy_;
  int sz = sz_;

  // If there is only 1 processor everyone is on PID 0.
  int nparts = NumGlobalParts(sx, sy, sz);
  if (comm_->NumProc() == 1 || nparts == 1)
    {
    nprocs_ = 1;
    pidMap_ = Teuchos::rcp(new Teuchos::Array<int>(nparts, 0));
    return 0;
    }

  pidMap_ = Teuchos::rcp(new Teuchos::Array<int>(nparts, -1));
  Teuchos::Array<Teuchos::Array<int> > pidGroups(nparts);
  Teuchos::Array<int> sdPidNum(nparts, 0);

  // Find the smallest possible coarsening factor
  int cx = FindCoarseningFactor(cx_);
  int cy = FindCoarseningFactor(cy_);
  int cz = FindCoarseningFactor(cz_);

  // Increase the subdomain size until the size of the subdomains is
  // the entire domain.
  while (sx < nx_ || sy < ny_ || sz < nz_)
    {
    sx = sx * cx;
    sy = sy * cy;
    if (nz_ > 1)
      sz = sz * cz;
    }

  nparts = NumGlobalParts(sx, sy, sz);
  if (nparts < 1)
    Tools::Error("Amount of subdomains with separator length " +
      Teuchos::toString(sx) + "x" + Teuchos::toString(sy) + "x" + Teuchos::toString(sz) +
      " and domain size " +
      Teuchos::toString(nx_) + "x" + Teuchos::toString(ny_) + "x" + Teuchos::toString(nz_) +
      " is " + Teuchos::toString(nparts) + ".", __FILE__, __LINE__);

  int sx2 = sx;
  int sy2 = sy;
  int sz2 = sz;

  // Loop over subdomain sizes from large to small until all
  // processors have some subdomain assigned to them.
  nprocs_ = 0;
  for (int j = 0; j < 1000; j++)
    {
    nparts = NumGlobalParts(sx, sy, sz);

    int prevNprocs = nprocs_;
    Teuchos::Array<Teuchos::Array<int> > prevPidGroups = pidGroups;

    for (int i = 0; i < nparts; i++)
      {
      int x, y, z;
      // Get the position of the larger subdomain
      GetSubdomainPosition(i, sx, sy, sz, x, y, z);

      x = (x % nx_ + nx_) % nx_;
      y = (y % ny_ + ny_) % ny_;
      z = (z % nz_ + nz_) % nz_;

      // Get first subdomain in the larger subdomain
      int sd = GetSubdomainID(sx_, sy_, sz_, x, y, z);

      if (pidGroups[sd].length() == 0)
        pidGroups[sd].append(nprocs_++);
      }

    // If we don't have enough processors to perform this
    // partitioning, go back to the previous one.
    if (nprocs_ > comm_->NumProc())
      {
      nprocs_ = prevNprocs;
      pidGroups = prevPidGroups;
      break;
      }

    sx2 = sx;
    sy2 = sy;
    sz2 = sz;

    // Refine
    sx = sx / cx;
    sy = sy / cy;
    if (nz_ > 1)
      sz = sz / cz;

    if (sx < sx_ || sy < sy_ || sz < sz_)
      {
      sx = sx2;
      sy = sy2;
      sz = sz2;
      break;
      }
    }

  // Assign leftover processors to groups of domains that already have one
  nparts = NumGlobalParts(sx_, sy_, sz_);
  for (int j = 0; j < 1000; j++)
    {
    if (nprocs_ >= comm_->NumProc())
      break;
    for (int sd = 0; sd < nparts; sd++)
      {
      if (nprocs_ >= comm_->NumProc())
        break;

      if (pidGroups[sd].length() != 0)
        pidGroups[sd].append(nprocs_++);
      }
    }

  // Every subdomain of size sx that is not yet assigned to a
  // processor is assigned to the same processor as the subdomain that
  // is 1 size larger (sx2).
  nparts = NumGlobalParts(sx, sy, sz);
  for (int i = 0; i < nparts; i++)
    {
    int x, y, z;

    // Get the position of the larger subdomain
    GetSubdomainPosition(i, sx, sy, sz, x, y, z);

    x = (x % nx_ + nx_) % nx_;
    y = (y % ny_ + ny_) % ny_;
    z = (z % nz_ + nz_) % nz_;

    // Get first subdomain in the larger subdomain
    int sd = GetSubdomainID(sx_, sy_, sz_, x, y, z);

    // Check if the subdomain at this position already has a PID
    // assigned to it. This may happen at a boundary.
    if ((*pidMap_)[sd] != -1)
      continue;

    // Get ID of the subdomain that is one size larger
    int sd2 = GetSubdomainID(sx2, sy2, sz2, x, y, z);

    // Position of this subdomain.
    GetSubdomainPosition(sd2, sx2, sy2, sz2, x, y, z);

    x = (x % nx_ + nx_) % nx_;
    y = (y % ny_ + ny_) % ny_;
    z = (z % nz_ + nz_) % nz_;

    // First subdomain in this larger subdomain. This one should
    // have been assigned in the previous loop.
    sd2 = GetSubdomainID(sx_, sy_, sz_, x, y, z);

    if (pidGroups[sd2].length() == 0)
      Tools::Error("Invalid subdomain index " + Teuchos::toString(sd) +
        " with subdomain size " +
        Teuchos::toString(sx) + "x" + Teuchos::toString(sy) + "x" + Teuchos::toString(sz) +
        " and position " +
        Teuchos::toString(x) + ", " + Teuchos::toString(y) + ", " + Teuchos::toString(z)
        , __FILE__, __LINE__);

    (*pidMap_)[sd] = pidGroups[sd2][sdPidNum[sd2]++ % pidGroups[sd2].length()];
    }

  // Now fill up all the actual subdomains with processors from a
  // larger subdomain size
  nparts = NumGlobalParts(sx_, sy_, sz_);
  for (int i = 0; i < nparts; i++)
    {
    if ((*pidMap_)[i] == -1)
      {
      int x, y, z;

      // Position of the current subdomain.
      GetSubdomainPosition(i, sx_, sy_, sz_, x, y, z);

      x = (x % nx_ + nx_) % nx_;
      y = (y % ny_ + ny_) % ny_;
      z = (z % nz_ + nz_) % nz_;

      // Check if the subdomain at this position already has a PID
      // assigned to it. This may happen at a boundary.
      int sd = GetSubdomainID(sx_, sy_, sz_, x, y, z);
      if ((*pidMap_)[sd] != -1)
        {
        (*pidMap_)[i] = (*pidMap_)[sd];
        continue;
        }

      // Subdomain ID of the larger subdomain.
      sd = GetSubdomainID(sx, sy, sz, x, y, z);

      // Position of this subdomain.
      GetSubdomainPosition(sd, sx, sy, sz, x, y, z);

      x = (x % nx_ + nx_) % nx_;
      y = (y % ny_ + ny_) % ny_;
      z = (z % nz_ + nz_) % nz_;

      // First subdomain in this larger subdomain. This one should
      // have been assigned in the previous loop.
      sd = GetSubdomainID(sx_, sy_, sz_, x, y, z);

      if ((*pidMap_)[sd] == -1)
        Tools::Error("Invalid subdomain index " + Teuchos::toString(sd) +
          " with subdomain size " +
          Teuchos::toString(sx) + "x" + Teuchos::toString(sy) + "x" + Teuchos::toString(sz) +
          " and position " +
          Teuchos::toString(x) + ", " + Teuchos::toString(y) + ", " + Teuchos::toString(z)
          , __FILE__, __LINE__);

      (*pidMap_)[i] = (*pidMap_)[sd];
      }
    }

  // Redetermine the amount of processors.
  Teuchos::Array<int> pidMap(*pidMap_);
  std::sort(pidMap.begin(), pidMap.end());
  auto end = std::unique(pidMap.begin(), pidMap.end());
  nprocs_ = std::distance(pidMap.begin(), end);

  return 0;
  }

Teuchos::RCP<const Epetra_Map> BasePartitioner::MoveMap(
  Teuchos::RCP<const Epetra_Map> baseMap) const
  {
  HYMLS_PROF2("BasePartitioner", "MoveMap");

  if (destinationPID_ < 0)
    return Teuchos::null;

  Epetra_Comm const &comm = baseMap->Comm();
  const Epetra_MpiComm *epetraMpiComm = dynamic_cast<const Epetra_MpiComm *>(&comm);

  // If we don't have an MPI comm, we can't do anything
  if(!epetraMpiComm)
    return Teuchos::null;

  MPI_Comm newMpiComm;
  MPI_Comm MpiComm = epetraMpiComm->Comm();

  CHECK_ZERO(MPI_Comm_split(MpiComm, destinationPID_, 0, &newMpiComm));
  Teuchos::RCP<Epetra_MpiComm> newEpetraMpiComm = Teuchos::rcp(new Epetra_MpiComm(newMpiComm));

  int failed = false;
  if ((destinationPID_ == comm.MyPID() and newEpetraMpiComm->MyPID() != 0) or
    (newEpetraMpiComm->MyPID() == 0 and destinationPID_ != comm.MyPID()))
    {
    failed = true;
    Tools::Warning("Processor with rank " + Teuchos::toString(comm.MyPID()) +
      " and destination " + Teuchos::toString(destinationPID_) +
      " has rank " + Teuchos::toString(newEpetraMpiComm->MyPID()) +
      " in the gather group. Using send and receive instead of move.",
      __FILE__, __LINE__);
    }
  int globalFailed = false;
  comm.MaxAll(&failed, &globalFailed, 1);
  if (globalFailed)
    {
    CHECK_ZERO(MPI_Comm_free(&newMpiComm));
    return Teuchos::null;
    }

  // Create a map with the new communicator
  hymls_gidx *myGlobalElements;
#ifdef HYMLS_LONG_LONG
  myGlobalElements = baseMap->MyGlobalElements64();
#else
  myGlobalElements = baseMap->MyGlobalElements();
#endif
  Teuchos::RCP<Epetra_Map> gatherMap = Teuchos::rcp(new Epetra_Map(
      (hymls_gidx)-1, baseMap->NumMyElements(), myGlobalElements,
    (hymls_gidx)baseMap->IndexBase64(), *newEpetraMpiComm));

  // Gather on the first processor in each group
  Teuchos::RCP<Epetra_Map> gatheredMap = MatrixUtils::Gather(*gatherMap, 0);

  // Make a new map with the gathered elements
#ifdef HYMLS_LONG_LONG
  myGlobalElements = gatheredMap->MyGlobalElements64();
#else
  myGlobalElements = gatheredMap->MyGlobalElements();
#endif

  Teuchos::RCP<Epetra_Map> out = Teuchos::rcp(new Epetra_Map(
      (hymls_gidx)-1, gatheredMap->NumMyElements(),
    myGlobalElements, (hymls_gidx)baseMap->IndexBase64(), comm));

  CHECK_ZERO(MPI_Comm_free(&newMpiComm));

  return out;
  }

int BasePartitioner::SetDestinationPID(
  Teuchos::RCP<const Epetra_Map> baseMap)
  {
  Epetra_Comm const &comm = baseMap->Comm();
  int myPID = comm.MyPID();

  // In many cases, all GID have to be moved to the same
  // processor. We treat this as a special case
  int destinationPID = myPID;
  if (baseMap->NumMyElements() > 0)
    destinationPID = PID(baseMap->GID64(0));

  for (int lid = 0; lid < baseMap->NumMyElements(); lid++)
    {
    hymls_gidx gid = baseMap->GID64(lid);
    if (PID(gid) != destinationPID)
      destinationPID = -1;
    }

  destinationPID_ = -1;
  CHECK_ZERO(comm_->MinAll(&destinationPID, &destinationPID_, 1));
  if (destinationPID_ > -1)
    destinationPID_ = destinationPID;

  return 0;
  }


Teuchos::RCP<const Epetra_Map> BasePartitioner::RepartitionMap(
  Teuchos::RCP<const Epetra_Map> baseMap) const
  {
  HYMLS_PROF2("BasePartitioner", "RepartitionMap");

  Teuchos::RCP<const Epetra_Map> moveMap = MoveMap(baseMap);
  if (moveMap != Teuchos::null)
    return moveMap;

  Epetra_Comm const &comm = baseMap->Comm();
  int myPID = comm.MyPID();

  // check how many of the owned GIDs in the map need to be
  // moved to someone else
  int numSends = 0;
  for (int lid = 0; lid < baseMap->NumMyElements(); lid++)
    {
    hymls_gidx gid = baseMap->GID64(lid);
    int pid = PID(gid);
    if (pid != myPID)
      numSends++;
    }

  int numLocal = baseMap->NumMyElements() - numSends;

  // determine which GIDs we have to move, and where they will go
  hymls_gidx *sendGIDs = new hymls_gidx[numSends];
  int *sendPIDs = new int[numSends];

  int pos = 0;
  for (int i = 0; i < baseMap->NumMyElements(); i++)
    {
    hymls_gidx gid = baseMap->GID64(i);
    int pid = PID(gid); // global partition ID
    if (pid != myPID)
      {
      if (pos >= numSends)
        {
        Tools::Error("Sanity check failed with gid = " + Teuchos::toString(gid) +
          ", pos = " + Teuchos::toString(pos) + ", numSends = " +
          Teuchos::toString(numSends) + ".", __FILE__, __LINE__);
        }
      sendGIDs[pos] = gid;
      sendPIDs[pos++] = pid;
      }
    }

  Teuchos::RCP<Epetra_Distributor> Distor =
    Teuchos::rcp(comm.CreateDistributor());

  int numRecvs;
  CHECK_ZERO(Distor->CreateFromSends(numSends, sendPIDs, true, numRecvs));

  char* sbuf = reinterpret_cast<char*>(sendGIDs);
  int numRecvChars = static_cast<int>(numRecvs * sizeof(hymls_gidx));
  char* rbuf = new char[numRecvChars];

  CHECK_ZERO(Distor->Do(sbuf, sizeof(hymls_gidx), numRecvChars, rbuf));

  hymls_gidx *recvGIDs = reinterpret_cast<hymls_gidx*>(rbuf);
  if (static_cast<int>(numRecvs * sizeof(hymls_gidx)) != numRecvChars)
    {
    Tools::Error("sanity check failed", __FILE__, __LINE__);
    }

  int NumMyElements = numLocal + numRecvs;
  hymls_gidx *MyGlobalElements = new hymls_gidx[NumMyElements];
  pos = 0;
  for (int i = 0; i < baseMap->NumMyElements(); i++)
    {
    hymls_gidx gid = baseMap->GID64(i);
    if (PID(gid) == myPID)
      MyGlobalElements[pos++] = gid;
    }

  for (int i = 0; i < numRecvs; i++)
    MyGlobalElements[pos+i] = recvGIDs[i];

  std::sort(MyGlobalElements, MyGlobalElements + NumMyElements);

  Teuchos::RCP<const Epetra_Map> ret = Teuchos::rcp(new Epetra_Map(
      -1, NumMyElements, MyGlobalElements, (hymls_gidx)baseMap->IndexBase64(), comm));

  delete [] sendPIDs;
  delete [] sendGIDs;
  delete [] rbuf;
  delete [] MyGlobalElements;

  return ret;
  }

int BasePartitioner::PID(hymls_gidx gid) const
  {
  int i, j, k, var;
  Tools::ind2sub(nx_, ny_, nz_, dof_, gid, i, j, k, var);
  return PID(i, j, k);
  }

int BasePartitioner::PID(int i, int j, int k) const
  {
  return (*pidMap_)[GetSubdomainID(sx_, sy_, sz_, i, j, k)];
  }

  }//namespace
