#ifndef HYMLS_EXAMPLE_LOCA_INTERFACE_H
#define HYMLS_EXAMPLE_LOCA_INTERFACE_H

#include "LOCA_Epetra_Interface_TimeDependent.H"
#include "NOX_Epetra_Interface_Jacobian.H"
#include "NOX_Epetra_Interface_Preconditioner.H"
#include "LOCA_Parameter_Vector.H"
#include "Ifpack_Preconditioner.h"

class Epetra_CrsMatrix;
class Epetra_Vector;
class Epetra_Map;
class Epetra_Comm;


namespace HYMLS {

//! example class for doing continuation using LOCA and HYMLS on the problem
//!
//!     d1^2 Laplace u + f(u) - sigma*v = 0
//!     d2^2 Laplace v +   u - v        = 0
//!
//!     f(u) = lambda*u - u^3 - kappa
//!
//! (cf. https://en.wikipedia.org/wiki/Reaction%E2%80%93diffusion_system)
//! in 3 space dimensions. 
//!
//! Continuation parameters: 
//! d1 (default 1.0)
//! d2 (default 1.0)
//! sigma (default 0.0)
//! lambda (default 0.0)
//! kappa (default 0.0)
//!


//! Another example class for doing continuation using LOCA and HYMLS on the problem
//!
//!     D*sigma* Laplace u + f(u) - sigma*v = 0
//!     sigma*Laplace v +   u - v        = 0
//!
//!     f(u) = lambda*u - u^3 - kappa
//!
//! (cf. https://en.wikipedia.org/wiki/Reaction%E2%80%93diffusion_system)
//! in 3 space dimensions. 
//!
//! Continuation parameters: 
//! d1 (default 1.0)
//! d2 (default 1.0)
//! sigma (default 0.0)
//! lambda (default 0.0)
//! kappa (default 0.0)

class exampleLocaInterface : public LOCA::Epetra::Interface::TimeDependent,
                             public NOX::Epetra::Interface::Jacobian,
                             public NOX::Epetra::Interface::Preconditioner
{

      public:

    //! Constructor
    exampleLocaInterface(Teuchos::RCP<Epetra_Comm> comm,
      Teuchos::ParameterList& modelParams,
      Teuchos::RCP<Teuchos::ParameterList> hymlsParams);

    //! Destructor
    virtual ~exampleLocaInterface();

    //! Compute the function, F, given the specified input vector x.  Returns true if computation was successful.
    virtual bool computeF(const Epetra_Vector& x, Epetra_Vector& F,
                          const FillType fillFlag);

  /*! Compute Jacobian given the specified input vector x.  Returns
    true if computation was successful.
   */
  virtual bool computeJacobian(const Epetra_Vector& x, Epetra_Operator& Jac);

  /*!
  * \brief Call user routine for computing the shifted matrix
  * \f$\alpha J + \beta M\f$ where \f$J\f$ is the Jacobian matrix
  * and \f$M\f$ is the mass matrix.
  */
  virtual bool computeShiftedMatrix(double alpha, double beta,
                         const Epetra_Vector& x,
                                          Epetra_Operator& A);
                                          

  //! Computes a user defined preconditioner.
  virtual bool computePreconditioner(const Epetra_Vector& x,
                     Epetra_Operator& M,
                     Teuchos::ParameterList* precParams = 0);

    //! get pointer to solution vector (also for cloning)
    Teuchos::RCP<Epetra_Vector> getSolution(){return currentState_;}

    //! get pointer to Jacobian. The way loca handles sharing of the Jacobian is
    //! rather obscure, it works perfectly if the computeJacobian function just ignores
    //! its Jac argument, it seems, as long as the "LinearSystem" object has the internal
    //! pointer. Since the pointer in the LinearSystem is never updated, computeJacobian
    //! actually has to fill the object that was returned by the initial call to this
    //! function each time it is called.
    Teuchos::RCP<Epetra_CrsMatrix> getJacobian(){return currentJac_;}
    
    //! get mass matrix
    Teuchos::RCP<Epetra_CrsMatrix> getMassMatrix(){return massMatrix_;}

    //! get a pointer to the preconditioner object computed by computePreconditioner()
    Teuchos::RCP<Epetra_Operator> getPreconditioner(){return precond_;}
    
    //! get current parameter vector
    Teuchos::RCP<LOCA::ParameterVector> getParameterVector();

    // get nullspace
    Teuchos::RCP<Epetra_MultiVector> getNullSpace() const {return nullSpace_;}

    /*!
      \brief Set parameters in the user's application.

      Should be called prior to calling one of the compute functions.
    */
    virtual void setParameters(const LOCA::ParameterVector& p);
  
    virtual  int write_vector(const Epetra_Vector& vec, std::string filename);
    virtual  int read_vector(Epetra_MultiVector& vec, std::string filename);

    //! Call user's own print routine for vector-parameter pair
    virtual void printSolution(const Epetra_Vector& x_,
                   double conParam);

        /*!
          \brief Provides data to application for output files.

          This routine is called from Interface::xyzt::printSolution() just
          before the call to Interface::Required::printSolution(x,param),
          and gives the application some indices that can be used for
          creating a unique name/index for the output files.
        */
        virtual void dataForPrintSolution(const int conStep, const int timeStep,
                                          const int totalTimeSteps);


    //! Perform any preprocessing before a continuation step starts.
    /*!
     * The \c stepStatus argument indicates whether the previous step was
     * successful.  The default implementation here is empty.
     */
    virtual void
    preProcessContinuationStep(
                 LOCA::Abstract::Iterator::StepStatus stepStatus,
                 LOCA::Epetra::Group& group);

    //! Perform any postprocessing after a continuation step finishes.
    /*!
     * The \c stepStatus argument indicates whether the step was
     * successful. The default implementation here is empty.
     */
    virtual void
    postProcessContinuationStep(
                 LOCA::Abstract::Iterator::StepStatus stepStatus,
                 LOCA::Epetra::Group& group);

    /*!
      \brief Projects solution to a few scalars for
      multiparameter continuation

      Default implementation is the max norm.
    */
    virtual void projectToDraw(const NOX::Epetra::Vector& x,
                   double *px) const;

    //! Returns the dimension of the projection to draw array
    virtual int projectToDrawDimension() const;
                                                      

    protected:
    
    //! global grid dimensions
    int nx_,ny_,nz_,dim_,dof_;
    std::string cont_param_;

    //! Communicator object
    Teuchos::RCP<Epetra_Comm> comm_;

    //! map containing all local indices
    Teuchos::RCP<Epetra_Map> rowMap_;
    
    //! map containing all column indices ('map with overlap')
    Teuchos::RCP<Epetra_Map> colMap_;
    
    //! model parameters
    double r1_, r2_, sigma_, D_, alpha_,beta_, source_;
     
    //! state vector
    Teuchos::RCP<Epetra_Vector> currentState_, rhs_, initialSol_;
   
    // eigenvector as perturbation
    Teuchos::RCP<Epetra_MultiVector> Eigvec_;
    //inf norm of the eigenvector
    double infnorm; 
    
    //! Jacobian matrix at current state
    Teuchos::RCP<Epetra_CrsMatrix> currentJac_, massMatrix_;
    
    //! HYMLS preconditioner
    Teuchos::RCP<Ifpack_Preconditioner> precond_;
    
    //! parameter vector
    Teuchos::RCP<LOCA::ParameterVector> pVector_;
 
    // nullspace as border of the Jacobian matrix
    Teuchos::RCP<Epetra_MultiVector> nullSpace_;

};

  } // HYMLS
#endif
