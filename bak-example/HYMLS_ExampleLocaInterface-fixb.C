#include "HYMLS_ExampleLocaInterface.H"

#include "HYMLS_Preconditioner.H"

#include "HYMLS_Tools.H"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "HYMLS_MatrixUtils.H"
#include "LOCA_Parameter_Vector.H"

#include "Epetra_Map.h"
#include "Epetra_Vector.h"
#include "Epetra_MpiComm.h"
#include "Epetra_CrsMatrix.h"

typedef int gidx_t;

namespace HYMLS {


//! Constructor
exampleLocaInterface::exampleLocaInterface(Teuchos::RCP<Epetra_Comm> comm,
                                           Teuchos::ParameterList& params,
                                           Teuchos::RCP<Teuchos::ParameterList> lsParams)
  :
  dim_(3),
  dof_(2),
  comm_(comm),
  precond_(Teuchos::null),
  pVector_(Teuchos::null)
{
  nx_ = params.get("nx", -1);
  ny_ = params.get("ny", nx_);
  nz_ = params.get("nz", nx_);

  if (nx_==-1)
  {
    Tools::Error("missing parameter \"nx\" in \"Model\" parameter list",__FILE__,__LINE__);
  }

  // create a 3D HYMLS-style cartesian mapi
  rowMap_=HYMLS::MatrixUtils::CreateMap(nx_,ny_,nz_,dof_, 0, *comm_);

  // create clone vector
  currentState_=Teuchos::rcp(new Epetra_Vector(*rowMap_));
 
  // right hand side
  // rhs_ = Teuchos::rcp(new Epetra_Vector(*rowMap_));
  // create the Jacobian pattern.
  // The pattern we assume is a 7-point Laplacian for u and v with a
  // diagonal and off diagonal coupling term and periodic boundary conditions in all three
  // space directions.
  currentJac_=Teuchos::rcp(new Epetra_CrsMatrix(Copy,*rowMap_,8));//maximum 8 nonzero per row
  massMatrix_=Teuchos::rcp(new Epetra_CrsMatrix(Copy,*rowMap_,1));

  for (int i=0; i<massMatrix_->NumMyRows(); i++)
  {
    gidx_t gid = massMatrix_->GRID(i);
    double val=1.0;
    CHECK_ZERO(massMatrix_->InsertGlobalValues(gid,1,&val,&gid));
  }
  CHECK_ZERO(massMatrix_->FillComplete());

  // get initial model parameters
  Teuchos::ParameterList &pList=params.sublist("Starting Parameters");
  r1_=pList.get("r1",1.0);
  r2_=pList.get("r2",1.0);
  D_=pList.get("D",0.0);
  sigma_=pList.get("sigma",0.0);
  alpha_=pList.get("alpha",0.0);
  beta_=pList.get("beta",0.0);
  source_=pList.get("source",0.0); 

  CHECK_ZERO(currentState_->PutScalar(0.0));
  computeJacobian(*currentState_,*currentJac_);

  if (lsParams!=Teuchos::null)
  {
    // set up HYMLS Parameter list
   Teuchos::RCP<Teuchos::ParameterList> hymlsParams
     = Teuchos::rcp(&(lsParams->sublist("HYMLS")), false);

    Teuchos::ParameterList& probList=hymlsParams->sublist("Problem");

    probList.set("nx",nx_);
    probList.set("ny",ny_);
    probList.set("nz",nz_);
    precond_ = Teuchos::rcp(new HYMLS::Preconditioner(currentJac_, hymlsParams));
  }
  else
  {
     precond_ = Teuchos::null;
  }

}

//! Destructor
exampleLocaInterface::~exampleLocaInterface()
{
  //handled by RCP's
}

//get vector of possible continuation parameters for LOCA
Teuchos::RCP<LOCA::ParameterVector> exampleLocaInterface::getParameterVector()
{

  if (pVector_==Teuchos::null)
  {
    pVector_ = Teuchos::rcp(new LOCA::ParameterVector());

    pVector_->addParameter("r1",r1_);
    pVector_->addParameter("r2",r2_);
    pVector_->addParameter("D",D_);
    pVector_->addParameter("sigma",sigma_);
    pVector_->addParameter("alpha",alpha_);
    pVector_->addParameter("beta",beta_);
    pVector_->addParameter("source",source_);

  }
  else
  {
/*
    pVector_->setParameter("r1",r1_);
    pVector_->setParameter("r2",r2_);
    pVector_->setParameter("D",D_);
    pVector_->setParameter("sigma",sigma_);
    pVector_->setParameter("alpha",alpha_);
    pVector_->setParameter("beta",beta_);
*/
  }
  return pVector_;
}

  //! Compute the function, F, given the specified input vector x.  Returns true if computation was successful.
  bool exampleLocaInterface::computeF(const Epetra_Vector& x, Epetra_Vector& F,
                          const FillType fillFlag)
{
  double ind = 1.0/(nx_*nx_);
  gidx_t cols[8];
  double vals[8];
  //Epetra_Vector xx(*x);
//  int row = 0; //count the row
  for (int k=0; k<nz_; k++)
  {
    for (int j=0; j<ny_; j++)
    {    
      for (int i=0; i<nx_; i++)
      {
        for (int var=0; var<2; var++)
        {
  //        row = row + 1;
          // diagonal element (row index)
          int ijk  = Tools::sub2ind(nx_,ny_,nz_,dof_,i,j,k,var);
          // coupling term to other variable
          int ijk2 = Tools::sub2ind(nx_,ny_,nz_,dof_,i,j,k,1-var);

          // 6 nearest neighbors
          int ip1=(i+1)%nx_,jp1=(j+1)%ny_,kp1=(k+1)%nz_;
          int im1=(i==0)?nx_-1:i-1,jm1=(j==0)?ny_-1:j-1, km1=(k==0)?nz_-1:k-1;

          int im1jk  = Tools::sub2ind(nx_,ny_,nz_,dof_,im1,j,k,var);
          int ip1jk  = Tools::sub2ind(nx_,ny_,nz_,dof_,ip1,j,k,var);
          int ijm1k  = Tools::sub2ind(nx_,ny_,nz_,dof_,i,jm1,k,var);
          int ijp1k  = Tools::sub2ind(nx_,ny_,nz_,dof_,i,jp1,k,var);
          int ijkm1  = Tools::sub2ind(nx_,ny_,nz_,dof_,i,j,km1,var);
          int ijkp1  = Tools::sub2ind(nx_,ny_,nz_,dof_,i,j,kp1,var);
          F[(gidx_t)ijk]= 0;
          int p=0;
          if (var==0)
          {
            vals[p]=-D_*sigma_*6*ind+alpha_-alpha_*r1_*x[ijk2]*x[ijk2]-r2_*x[ijk2]; cols[p++]=(gidx_t)ijk;
            vals[p]=1;             cols[p++]=(gidx_t)ijk2;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)im1jk;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ip1jk;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijm1k;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijp1k;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijkm1;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijkp1;                 

          }
          else
          {
            if (k==0 || k==nz_-1)
              {
               vals[p]=-sigma_*6*ind+beta_+alpha_*r1_*source_*x[ijk]+r2_*source_; cols[p++]=(gidx_t)ijk;
              }
            else
              {
               vals[p]=-sigma_*6*ind+beta_+alpha_*r1_*x[ijk2]*x[ijk]+r2_*x[ijk2]; cols[p++]=(gidx_t)ijk;
              }
            vals[p]=-alpha_;    cols[p++]=(gidx_t)ijk2;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)im1jk;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ip1jk;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijm1k;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijp1k;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijkm1;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijkp1;
          }
          for (int h=0; h<8; h++)
          {
            if ((0 <= cols[h] && cols[h] <= 127 && (cols[h]%2 == 0)) ||((896 <= cols[h] && cols[h]<= 1023) && (cols[h]%2 == 0) ))
              {
               F[(gidx_t)ijk]= F[(gidx_t)ijk] + vals[h]*source_;
              }
            else
              {
               F[(gidx_t)ijk]= F[(gidx_t)ijk] + vals[h]*x[cols[h]];
              }
          }
        }
      }
    }
  }

  HYMLS::MatrixUtils::Dump(F,"rhs.txt");

  return true;
}


  /*! Compute Jacobian given the specified input vector x.  Returns
    true if computation was successful.
   */
  bool exampleLocaInterface::computeJacobian(const Epetra_Vector& x, Epetra_Operator& Jac)
  {
    return computeShiftedMatrix(1.0,0.0,x,Jac);
  }

      /*!
     * \brief Call user routine for computing the shifted matrix
     * \f$\alpha J + \beta M\f$ where \f$J\f$ is the Jacobian matrix
     * and \f$M\f$ is the mass matrix.
     */
bool exampleLocaInterface::computeShiftedMatrix(double alpha, double beta,
                const  Epetra_Vector& x,
                 Epetra_Operator& A)
{
  double ind = 1.0/(nx_*nx_);
  gidx_t cols[8];
  double vals[8];
  for (int k=0; k<nz_; k++)
  {
    for (int j=0; j<ny_; j++)
    {
      for (int i=0; i<nx_; i++)
      {
        for (int var=0; var<2; var++)
        {
          // diagonal element (row index)
          int ijk  = Tools::sub2ind(nx_,ny_,nz_,dof_,i,j,k,var);
          // coupling term to other variable
          int ijk2 = Tools::sub2ind(nx_,ny_,nz_,dof_,i,j,k,1-var);

          // 6 nearest neighbors
          int ip1=(i+1)%nx_,jp1=(j+1)%ny_,kp1=(k+1)%nz_;
          int im1=(i==0)?nx_-1:i-1,jm1=(j==0)?ny_-1:j-1, km1=(k==0)?nz_-1:k-1;

          int im1jk  = Tools::sub2ind(nx_,ny_,nz_,dof_,im1,j,k,var);
          int ip1jk  = Tools::sub2ind(nx_,ny_,nz_,dof_,ip1,j,k,var);
          int ijm1k  = Tools::sub2ind(nx_,ny_,nz_,dof_,i,jm1,k,var);
          int ijp1k  = Tools::sub2ind(nx_,ny_,nz_,dof_,i,jp1,k,var);
          int ijkm1  = Tools::sub2ind(nx_,ny_,nz_,dof_,i,j,km1,var);
          int ijkp1  = Tools::sub2ind(nx_,ny_,nz_,dof_,i,j,kp1,var);

          int p=0;
          if (var==0)
          {
            vals[p]=-D_*sigma_*6*ind+alpha_-alpha_*r1_*x[ijk2]*x[ijk2]-r2_*x[ijk2]; cols[p++]=(gidx_t)ijk;
            if (k==0 || k==nz_-1) 
              {
                vals[p]=1-2*alpha_*r1_*source_*x[ijk2]-r2_*source_; cols[p++]=(gidx_t)ijk2;
              }
            else
              {
                vals[p]=1-2*alpha_*r1_*x[ijk]*x[ijk2]-r2_*x[ijk]; cols[p++]=(gidx_t)ijk2;
              }

           // vals[p]=1-2*alpha_*r1_*x[ijk]*x[ijk2]-r2_*x[ijk]; cols[p++]=(gidx_t)ijk2;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)im1jk;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ip1jk;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijm1k;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijp1k;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijkm1;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijkp1;
          }
          else
          {
              if (k==0 || k==nz_-1)
              {
                vals[p]=-sigma_*6*ind+beta_+2*alpha_*r1_*source_*x[ijk]+r2_*source_; cols[p++]=(gidx_t)ijk;

              }
            else
              {
                vals[p]=-sigma_*6*ind+beta_+2*alpha_*r1_*x[ijk2]*x[ijk]+r2_*x[ijk2]; cols[p++]=(gidx_t)ijk;

              }

            //vals[p]=-sigma_*6*ind+beta_+2*alpha_*r1_*x[ijk2]*x[ijk]+r2_*x[ijk2]; cols[p++]=(gidx_t)ijk;
            vals[p]=-alpha+alpha_*r1_*x[ijk]*x[ijk]+r2_*x[ijk]; cols[p++]=(gidx_t)ijk2;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)im1jk;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ip1jk;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijm1k;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijp1k;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijkm1;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijkp1;
          }

          if (!currentJac_->Filled())
          {
            CHECK_ZERO(currentJac_->InsertGlobalValues((gidx_t)ijk,8,vals,cols));
          }
          else
          {
            CHECK_ZERO(currentJac_->ReplaceGlobalValues((gidx_t)ijk,8,vals,cols));
          }
        }
      }
    }
  }
  if (!currentJac_->Filled())
  {
    CHECK_ZERO(currentJac_->FillComplete());
  }
    HYMLS::MatrixUtils::Dump(*currentJac_,"jacMatrix.txt");
  return true; //TODO - put in correct values above
}


  /*//! Computes a user defined preconditioner.
  bool exampleLocaInterface::computePreconditioner(const Epetra_Vector& x,
                     Epetra_Operator& M,
                     Teuchos::ParameterList* precParams)
  {
    //TODO - in which situation does this function get a parameter list, and
    //       at which level?
    if (precParams!=NULL)
    {
      Tools::Warning("parameter list passed to computePreconditioner is ignored",__FILE__,__LINE__);
    }
    CHECK_ZERO(precond_->Compute());
    return true;
  }*/

 // compute preconditioner, which can then be retrieved by getPreconditioner()
bool exampleLocaInterface::computePreconditioner(const Epetra_Vector& x,
                                         Epetra_Operator& Prec,
                                         Teuchos::ParameterList* p)
  {
  // step_counter++;
  //if (step_counter>1) HYMLS::Tools::Error("STOP",__FILE__,__LINE__);
   int result=0;
   cout << "HELLO-3" << endl;

  //DEBUG("enter LocaInterface::computePreconditioner");
  if (precond_ == Teuchos::null)
    {
   
    // no preconditioner parameters passed to constructor
    HYMLS::Tools::Error("No Preconditioner available!",__FILE__,__LINE__);
    }

  if (precond_->IsInitialized()==false)
    {
    CHECK_ZERO(result=precond_->Initialize());
    }

  if (result!=0)
    {
    HYMLS::Tools::Warning("Error code "+Teuchos::toString(result)+" returned when "+
                   " initializing the solver!",__FILE__,__LINE__);
    }


  if (result==0)
    {
    Teuchos::RCP<HYMLS::Preconditioner> hymls =
        Teuchos::rcp_dynamic_cast<HYMLS::Preconditioner>(precond_);
    result=precond_->Compute();
    if (result!=0)
      {
      HYMLS::Tools::Warning("Error code "+Teuchos::toString(result)+" returned when "+
                   " computing the solver!",__FILE__,__LINE__);
      }
    }

  DEBUG("leave LocaInterface::computePreconditioner");

  return (result==0);
  }


    /*!
      \brief Set parameters in the user's application.

      Should be called prior to calling one of the compute functions.
    */
    void exampleLocaInterface::setParameters(const LOCA::ParameterVector& p)
  {
    r1_ = p.getValue("r1");
    r2_ = p.getValue("r2");
    D_ = p.getValue("D");
    sigma_ = p.getValue("sigma");
    alpha_ = p.getValue("alpha");
    beta_ = p.getValue("beta");

  }

    //! Call user's own print routine for vector-parameter pair
    void exampleLocaInterface::printSolution(const Epetra_Vector& x_,
                   double conParam)
  {
    //TODO
  }
        /*!
          \brief Provides data to application for output files.

          This routine is called from Interface::xyzt::printSolution() just
          before the call to Interface::Required::printSolution(x,param),
          and gives the application some indices that can be used for
          creating a unique name/index for the output files.
        */
        void exampleLocaInterface::dataForPrintSolution(const int conStep, const int timeStep,
                                          const int totalTimeSteps)
  {
    //TODO
  }

    //! Perform any preprocessing before a continuation step starts.
    /*!
     * The \c stepStatus argument indicates whether the previous step was
     * successful.  The default implementation here is empty.
     */
    void
    exampleLocaInterface::preProcessContinuationStep(
                 LOCA::Abstract::Iterator::StepStatus stepStatus,
                 LOCA::Epetra::Group& group)
  {
    // do nothing right now
  }

    //! Perform any postprocessing after a continuation step finishes.
    /*!
     * The \c stepStatus argument indicates whether the step was
     * successful. The default implementation here is empty.
     */
    void
    exampleLocaInterface::postProcessContinuationStep(
                 LOCA::Abstract::Iterator::StepStatus stepStatus,
                 LOCA::Epetra::Group& group)
  {
    // do nothing right now
  }

    /*!
      \brief Projects solution to a few scalars for
      multiparameter continuation

      Default implementation is the max norm.
    */
    void exampleLocaInterface::projectToDraw(const NOX::Epetra::Vector& x,
                   double *px) const {
      // use default implementation for now
      px[0] = x.norm(NOX::Abstract::Vector::MaxNorm);
    }

    //! Returns the dimension of the projection to draw array
    int exampleLocaInterface::projectToDrawDimension() const { return 1; }


  } // HYMLS
