#include "HYMLS_ExampleLocaInterface.H"

#include "HYMLS_Preconditioner.H"

#include "HYMLS_Solver.H"
#include "HYMLS_Tools.H"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StrUtils.hpp"
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
  dim_ = params.get("dim",1);
  if (nx_==-1)
  {
    Tools::Error("missing parameter \"nx\" in \"Model\" parameter list",__FILE__,__LINE__);
  }

  // create a 3D HYMLS-style cartesian map
  rowMap_=HYMLS::MatrixUtils::CreateMap(nx_,ny_,nz_,dof_, 0, *comm_);

  // create clone vector
  currentState_=Teuchos::rcp(new Epetra_Vector(*rowMap_));
  rhs_=Teuchos::rcp(new Epetra_Vector(*rowMap_));
  initialSol_=Teuchos::rcp(new Epetra_Vector(*rowMap_));

 //nullspace as the border v in the Jacobian matrix, in case the Jacobian matrix is singular 
  nullSpace_=Teuchos::rcp(new Epetra_MultiVector(*rowMap_,1));

 
  // right hand side
  // rhs_ = Teuchos::rcp(new Epetra_Vector(*rowMap_));
  // create the Jacobian pattern.
  // The pattern we assume is a 7-point Laplacian for u and v with a
  // diagonal and off diagonal coupling term and periodic boundary conditions in all three
  // space directiions.
  currentJac_=Teuchos::rcp(new Epetra_CrsMatrix(Copy,*rowMap_,2*dim_+2));//maximum 8 nonzero per row
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
 
 
  // get continuation parameters 
  cont_param_ = params.get("Parameter Name","Undefined"); 

   /* int length = nx_*ny_*nz_*dof_;
     float inisol[length];
      int i=0;
      char ac[1024];
      FILE *f=NULL;
      // std::string Eigvecfile = "eigenVec.txt";
         
      if((f=fopen("solution-2DW-reorder.txt","r"))==NULL)
       {
         std::cout << "can not open\n";
       }
      else
       {
        while(fgets(ac, 1024, f) != NULL)
         {
          
          fscanf(f,"%f",&inisol[i]); 
          i++;   
         }         
             
       }
       
  if (f) fclose(f);*/
 std::string initialFile = "solution-2DW-reorder.txt";
 HYMLS::MatrixUtils::mmread(initialFile,*currentState_);


  //CHECK_ZERO(currentState_->PutScalar(0.0));
  //HYMLS::MatrixUtils::Random(*currentState_);//range is [-1,1]
   for (int i=0; i<currentState_->MyLength(); i++)
  {
   //if (rowMap_->LID(i) != -1)
   // (*currentState_)[i] =((*currentState_)[i]);
   //(*currentState_)[lid] = inisol[i];
   //(*currentState_)[i] = (*initialSol_)[i];
    cout << (*currentState_)[i] << endl;

  }

  
  // set the nullspace  
  (*nullSpace_)(0)->PutScalar(0.0);
  for (int i=2*nx_*ny_-2*nx_+1;i<nullSpace_->MyLength();i=i+2)
  { 
     (*nullSpace_)[0][i]=1.0;
  }


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

  //! Compute the function, F, gIiven the specified input vector x.  Returns true if computation was successful.
  bool exampleLocaInterface::computeF(const Epetra_Vector& x, Epetra_Vector& F,
                          const FillType fillFlag)
{
  double indl = nx_*nx_/900.0;
  double indr = nx_*nx_/900.0;
  double indu = nx_*nx_/900.0;
  double indd = nx_*nx_/900.0;
  double ind = nx_*nx_/900.0;
  gidx_t cols[2*dim_+2];
  double vals[2*dim_+2];
  cout << "r2=" << r2_ << endl;
 

  //count the row
  for (int ii=0; ii<F.MyLength(); ii++)
  {
        gidx_t ijk = F.Map().GID(ii);
        
        int i,j,k,var;
        Tools::ind2sub(nx_,ny_,nz_,dof_,ijk,i,j,k,var);
  
  //        row = row + 1;
          // diagonal element (row index)
         // int ijk  = Tools::sub2ind(nx_,ny_,nz_,dof_,i,j,k,var);
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
   
          //Neumann boundary condition for the diffusion term 
         /* if(i==0)     indl = 0.0; 
          else indl = nx_*nx_;

          if(i==nx_-1) indr = 0.0;
          else indr = nx_*nx_;

          if(k==0)     indd = 0.0;
          else indd = nx_*nx_;

          if(k==nz_-1) indu = 0.0;
          else indu = nx_*nx_;*/

 /**************** ///  Below is the rhs of Interacting Turing System ************/

          //int lid1 = x->Map().LID(ijk);
          //int lid2 = x->Map().LID(ijk2);
           if (var==0)
          { 
            //int lid2 = x->Map().LID(ijk2);
            vals[p]=-D_*sigma_*(dim_*2)*ind+alpha_-alpha_*r1_*x[ii+1]*x[ii+1]-r2_*x[ii+1]; cols[p++]=(gidx_t)ijk;
            vals[p]=1;             cols[p++]=(gidx_t)ijk2;
            vals[p]=D_*sigma_*indl; cols[p++]=(gidx_t)im1jk;
            vals[p]=D_*sigma_*indr; cols[p++]=(gidx_t)ip1jk;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijm1k;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijp1k;
            if (dim_==3)
            {
              vals[p]=D_*sigma_*indd; cols[p++]=(gidx_t)ijkm1;
              vals[p]=D_*sigma_*indu; cols[p++]=(gidx_t)ijkp1;                 
            }
          }
          else
          {       
            vals[p]=-sigma_*(dim_*2)*ind+beta_+alpha_*r1_*x[ii-1]*x[ii]+r2_*x[ii-1]; cols[p++]=(gidx_t)ijk;
            vals[p]=-alpha_;    cols[p++]=(gidx_t)ijk2;
            vals[p]=sigma_*indl; cols[p++]=(gidx_t)im1jk;
            vals[p]=sigma_*indr; cols[p++]=(gidx_t)ip1jk;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijm1k;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijp1k;
            if (dim_==3)
            {
              vals[p]=sigma_*indd; cols[p++]=(gidx_t)ijkm1;
              vals[p]=sigma_*indu; cols[p++]=(gidx_t)ijkp1;
            }
          }
 
/**************** ///  Below is the rhs of FitzHugh–Nagumo equation ************/
          /* if (var==0)
          {
            vals[p]=-D_*sigma_*4*ind+r1_-x[ijk]*x[ijk]; cols[p++]=(gidx_t)ijk;
            vals[p]=-1;             cols[p++]=(gidx_t)ijk2;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)im1jk;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ip1jk;
            //vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijm1k;
            //vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijp1k;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijkm1;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijkp1;                 

          }
          else
          {       
            vals[p]=-sigma_*4*ind-1.0; cols[p++]=(gidx_t)ijk;
            vals[p]=1;    cols[p++]=(gidx_t)ijk2;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)im1jk;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ip1jk;
            //vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijm1k;
            //vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijp1k;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijkm1;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijkp1;
          }*/
          
         
             
          for (int h=0; h<(2*dim_+2); h++)
          {   
            int lid= x.Map().LID(cols[h]);
            F[ii]= F[ii] + vals[h]*x[lid];
              
          }
          
        
      
    }
        // set the phase condition, assume the last u is zero. 
          F[nx_*ny_*nz_*2-2]=0.0;


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
  double indl = nx_*nx_/900.0;
  double indr = nx_*nx_/900.0;
  double indu = nx_*nx_/900.0;
  double indd = nx_*nx_/900.0;
  double ind = nx_*nx_/900.0;

  gidx_t cols[2*dim_+2];
  double vals[2*dim_+2];
  

   for (int ii=0; ii<currentJac_->NumMyRows(); ii++)
  {
        gidx_t ijk = currentJac_->GRID(ii);
        
        int i,j,k,var;
        Tools::ind2sub(nx_,ny_,nz_,dof_,ijk,i,j,k,var);

          // diagonal element (row index)
          //int ijk  = Tools::sub2ind(nx_,ny_,nz_,dof_,i,j,k,var);
         // cout << "ijk="<< ijk << "gid=" << gid << endl;

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
          //Neumann boundary condition for the diffusion term 
         /* if(i==0)     indl = 0.0; 
          else indl = nx_*nx_;

          if(i==nx_-1) indr = 0.0;
          else indr = nx_*nx_;

          if(k==0)     indd = 0.0;
          else indd = nx_*nx_;

          if(k==nz_-1) indu = 0.0;
          else indu = nx_*nx_;*/


/**************** ///  Below is the rhs of Interacting Turing System ************/
          if (var==0)
          {
            vals[p]=-D_*sigma_*(2*dim_)*ind+alpha_-alpha_*r1_*x[ii+1]*x[ii+1]-r2_*x[ii+1]; cols[p++]=(gidx_t)ijk;
            vals[p]=1-2*alpha_*r1_*x[ii]*x[ii+1]-r2_*x[ii]; cols[p++]=(gidx_t)ijk2;
            vals[p]=D_*sigma_*indl; cols[p++]=(gidx_t)im1jk;
            vals[p]=D_*sigma_*indr; cols[p++]=(gidx_t)ip1jk;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijm1k;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijp1k;
            if (dim_==3)
            {
              vals[p]=D_*sigma_*indd; cols[p++]=(gidx_t)ijkm1;
              vals[p]=D_*sigma_*indu; cols[p++]=(gidx_t)ijkp1;
            }
           
          }
          else
          {
            vals[p]=-sigma_*(2*dim_)*ind+beta_+2*alpha_*r1_*x[ii-1]*x[ii]+r2_*x[ii-1]; cols[p++]=(gidx_t)ijk;
            vals[p]=-alpha_+alpha_*r1_*x[ii]*x[ii]+r2_*x[ii]; cols[p++]=(gidx_t)ijk2;
            vals[p]=sigma_*indl; cols[p++]=(gidx_t)im1jk;
            vals[p]=sigma_*indr; cols[p++]=(gidx_t)ip1jk;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijm1k;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijp1k;
            if (dim_==3)
            {
              vals[p]=sigma_*indd; cols[p++]=(gidx_t)ijkm1;
              vals[p]=sigma_*indu; cols[p++]=(gidx_t)ijkp1;
            }
          }


/**************** ///  Below is the rhs of FitzHugh–Nagumo equation ************/
         /*if (var==0)
          {
            vals[p]=-D_*sigma_*4*ind+r1_-3*x[ijk]*x[ijk]; cols[p++]=(gidx_t)ijk;
            vals[p]=-1;             cols[p++]=(gidx_t)ijk2;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)im1jk;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ip1jk;
            //vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijm1k;
            //vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijp1k;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijkm1;
            vals[p]=D_*sigma_*ind; cols[p++]=(gidx_t)ijkp1;

          }
          else
          {
            vals[p]=-sigma_*4*ind-1.0; cols[p++]=(gidx_t)ijk;
            vals[p]=1;    cols[p++]=(gidx_t)ijk2;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)im1jk;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ip1jk;
            //vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijm1k;
            //vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijp1k;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijkm1;
            vals[p]=sigma_*ind; cols[p++]=(gidx_t)ijkp1;
          }*/

          //set the phase condition
          if (ijk==nx_*ny_*nz_*2-2) 
            {
              vals[0]=1.0;
              vals[1]=0.0;
              vals[2]=0.0;
              vals[3]=0.0;
              vals[4]=0.0;
              vals[5]=0.0;
              if (dim_==3)
              { 
                vals[6]=0.0;
                vals[7]=0.0;
              }
            }
   

          if (!currentJac_->Filled())
          {
            CHECK_ZERO(currentJac_->InsertGlobalValues(ijk,2*dim_+2,vals,cols));
          }
          else
          {
            CHECK_ZERO(currentJac_->ReplaceGlobalValues(ijk,2*dim_+2,vals,cols));
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
    //set border
    hymls->SetBorder(nullSpace_,Teuchos::null,Teuchos::null);

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


  int exampleLocaInterface::write_vector(const Epetra_Vector& vec, std::string filename)                                      
  {                                                                                                                   
    return HYMLS::MatrixUtils::mmwrite(filename,vec);                                                                   
  }                                                                                                                   
                                                                                                                      
// call fortran routine to read vector                                                                                
  int exampleLocaInterface::read_vector(Epetra_MultiVector& vec, std::string filename)                                        
  {                                                                                                                   
    return HYMLS::MatrixUtils::mmread(filename,vec);                                                                    
  }   
    //! Call user's own print routine for vector-parameter pair
  void exampleLocaInterface::printSolution(const Epetra_Vector& x,
                   double conParam)
   {
    int dump_psiome=0;
    //if (output_interval_>=0 || force_backup_)
     //{
    // if (conParam-last_output_>output_interval_ || force_backup_)
      //{i
      std::string paramName=Teuchos::StrUtils::varSubstitute(cont_param_," ","_");
      paramName=Teuchos::StrUtils::varSubstitute(paramName,"-","_");
      std::string filename =
       // "solution_"+paramName+"_"+Teuchos::toString((int)conParam)+".mtx";
       "solution_"+paramName+"_"+Teuchos::toString(conParam)+".mtx";
      this->write_vector(x, filename);
    //  this->write_state("restart.xml",filename);
      //last_output_=conParam;
//      force_backup_=false;
      dump_psiome=1;
      //}
    //}

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
