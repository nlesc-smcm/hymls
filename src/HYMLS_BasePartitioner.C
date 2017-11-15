#include "HYMLS_BasePartitioner.H"
#include "HYMLS_Tools.H"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StrUtils.hpp"
#include "Teuchos_Array.hpp"

#include "Teuchos_StandardParameterEntryValidators.hpp"

#ifdef HYMLS_TESTING
#include "HYMLS_Tester.H"
#endif

namespace HYMLS {

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

  perio_ = probList.get("Periodicity", perio_);

  if (precList.isParameter("Separator Length (x)"))
    {
    sx_ = precList.get("Separator Length (x)", -1);
    sy_ = precList.get("Separator Length (y)", sx_);
    sz_ = precList.get("Separator Length (z)", nz_ > 1 ? sx_ : 1);
    }
  else
    {
    sx_ = precList.get("Separator Length", 4);
    sy_ = sx_;
    sz_ = nz_ > 1 ? sx_ : 1;
    }

  if (sx_ <= 1)
    Tools::Error("Separator Length not set correctly",
      __FILE__, __LINE__);

  if (precList.isParameter("Coarsening Factor (x)"))
    {
    cx_ = precList.get("Coarsening Factor (x)", -1);
    cy_ = precList.get("Coarsening Factor (y)", cx_);
    cz_ = precList.get("Coarsening Factor (z)", nz_ > sz_ ? cx_ : 1);
    }
  else
    {
    cx_ = precList.get("Coarsening Factor", sx_);
    cy_ = cx_;
    cz_ = nz_ > sz_ ? cx_ : 1;
    }

  if (cx_ <= 1)
    Tools::Error("Coarsening Factor not set correctly",
      __FILE__, __LINE__);

  if (probList.isParameter("Equations"))
    {
    std::string eqn = probList.get("Equations", "Undefined Problem");
    bool is_complex = probList.get("Complex Arithmetic", false);
    int factor = is_complex ? 2 : 1;

    if (eqn == "Laplace")
      {
      if (!is_complex)
        {
        probList.set("Degrees of Freedom", 1);
        probList.sublist("Variable 0").set("Variable Type", "Laplace");
        }
      else
        {
        probList.set("Degrees of Freedom", 2);
        probList.sublist("Variable 0").set("Variable Type", "Laplace");
        probList.sublist("Variable 1").set("Variable Type", "Laplace");
        }
      }
    else if (eqn == "Stokes-B" || eqn == "Stokes-C" || eqn == "Bous-C")
      {
      probList.set("Degrees of Freedom", dim_ + 1);
      pvar = dim_;
      if (eqn == "Bous-C")
        {
        probList.set("Degrees of Freedom", dim_ + 2);
        pvar = dim_ + 1;
        }

      dof_ = probList.get("Degrees of Freedom", 1);
      for (int i = 0; i < dim_ * factor; i++)
        probList.sublist("Variable " + Teuchos::toString(i)).set("Variable Type", "Velocity");

      for (int i = pvar * factor; i < pvar * factor + factor; i++)
        probList.sublist("Variable " + Teuchos::toString(i)).set("Variable Type", "Pressure");

      for (int i = 0; i < dof_; i++)
        if (!probList.isSublist("Variable " + Teuchos::toString(i)))
          probList.sublist("Variable " + Teuchos::toString(i)).set("Variable Type", "Laplace");

      if (precList.get("Fix Pressure Level", true))
        {
        // we fix the singularity by inserting a Dirichlet condition for
        // global pressure node 2 
        precList.set("Fix GID 1", factor * pvar);
        if (is_complex) precList.set("Fix GID 2", factor * pvar + 1);
        }
#ifdef HYMLS_TESTING
      probList.set("Test F-Matrix Properties", true);
#endif
      if (eqn == "Stokes-B")
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

        if (precList.get("Fix Pressure Level", true))
          {
          // we fix the singularity by inserting a Dirichlet condition for 
          // global pressure in cells 0 and 1, since we retain two pressures
          // per subdomain both will be retained until the coarsest grid.
          // We use +nx*dof here to skip the dummy P-nodes (@).
          precList.set("Fix GID 1", dim_ + nx_ * dof_);
          precList.set("Fix GID 2", 2 * dim_ + nx_ * dof_);
          }
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

  variableType_.resize(dof_);

  int pcount = 0;
  int vcount = 0;
  for (int i = 0; i < dof_; i++)
    {
    Teuchos::ParameterList& varList = probList.sublist("Variable " + Teuchos::toString(i));
    std::string variableType = varList.get("Variable Type", "Laplace");
    if (variableType == "Laplace")
      variableType_[i] = 1;
    else if (variableType == "Velocity U")
      variableType_[i] = 0;
    else if (variableType == "Velocity V")
      variableType_[i] = 1;
    else if (variableType == "Velocity W")
      variableType_[i] = 2;
    else if (variableType == "Velocity")
      {
      variableType_[i] = vcount;
      vcount++;
      }
    else if (variableType == "Pressure")
      {
      pvar = i;
      variableType_[i] = 3;
      pcount++;
      }
    }

  if (pcount > 1)
    Tools::Error("Can only have one 'Pressure' variable",
      __FILE__, __LINE__);

  if (vcount > 3)
    Tools::Error("Can only have three 'Velocity' variables",
      __FILE__, __LINE__);

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

void BasePartitioner::SetNextLevelParameters(Teuchos::ParameterList& params)
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

  }//namespace
