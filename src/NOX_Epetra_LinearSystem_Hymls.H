#ifndef NOX_EPETRA_LINEARSYSTEMHYMLS_H
#define NOX_EPETRA_LINEARSYSTEMHYMLS_H

#include "BelosConfigDefs.hpp"
#include "NOX_Epetra_LinearSystem_AztecOO.H"
#include "Teuchos_ScalarTraits.hpp"

class Epetra_CrsMatrix;

namespace HYMLS {
class Solver;
}

class Epetra_MultiVector;
class Epetra_Operator;


namespace NOX {
//! Improved version of the Epetra support class.
namespace Epetra {

/*! This class overloads LinearSystem_AztecOO, adding the  
    possibility to use Belos for linear systems instead of 
    AztecOO. Right now the Aztec solver will still be      
    constructed, too. Belos parameters can be set in the   
    "Linear Solver"->"Belos" sublist.                      
 */
class LinearSystemHymls : public LinearSystemAztecOO 
  {

public:
  //! Constructor with no Operators.  
  /*! Jacobian Operator will be constructed internally based on the
    parameter "Jacobian Operator".  Defaults to using a 
    NOX::Epetra::MatrixFree object.
   */
  LinearSystemHymls(
    Teuchos::ParameterList& printingParams, 
    Teuchos::ParameterList& linearSolverParams,
    const Teuchos::RCP<NOX::Epetra::Interface::Required>& iReq, 
    const NOX::Epetra::Vector& cloneVector,
    const Teuchos::RCP<NOX::Epetra::Scaling> scalingObject = 
    Teuchos::null);

  //! Constructor with a user supplied Jacobian Operator only.  
  /*! Either there is no preconditioning or the preconditioner will be 
    used/created internally.  The Jacobian (if derived from an 
    Epetra_RowMatrix class can be used with an internal preconditioner. 
    See the parameter key "Preconditioner Operator" for more details.
   */
  LinearSystemHymls(
    Teuchos::ParameterList& printingParams, 
    Teuchos::ParameterList& linearSolverParams, 
    const Teuchos::RCP<NOX::Epetra::Interface::Required>& iReq, 
    const Teuchos::RCP<NOX::Epetra::Interface::Jacobian>& iJac, 
    const Teuchos::RCP<Epetra_Operator>& J,
    const NOX::Epetra::Vector& cloneVector,
    const Teuchos::RCP<NOX::Epetra::Scaling> scalingObject = 
    Teuchos::null);

  //! Constructor with a user supplied Preconditioner Operator only.  
  /*! Jacobian operator will be constructed internally based on the
    parameter "Jacobian Operator" in the parameter list.  See the 
    parameter key "Jacobian Operator" for more details.  Defaults 
    to using a NOX::Epetra::MatrixFree object.
   */
  LinearSystemHymls(
    Teuchos::ParameterList& printingParams, 
    Teuchos::ParameterList& linearSolverParams, 
    const Teuchos::RCP<NOX::Epetra::Interface::Required>& i, 
    const Teuchos::RCP<NOX::Epetra::Interface::Preconditioner>& iPrec,
    const Teuchos::RCP<Epetra_Operator>& M,
    const NOX::Epetra::Vector& cloneVector,
    const Teuchos::RCP<NOX::Epetra::Scaling> scalingObject = Teuchos::null, 
    Teuchos::RCP<Epetra_CrsMatrix> massMat=Teuchos::null);
  
  //! Constructor with user supplied separate objects for the
  //! Jacobian (J) and Preconditioner (M).  
  //! linearSolverParams is the "Linear Solver" sublist of parameter list.
  LinearSystemHymls(
    Teuchos::ParameterList& printingParams, 
    Teuchos::ParameterList& linearSolverParams, 
    const Teuchos::RCP<NOX::Epetra::Interface::Jacobian>& iJac, 
    const Teuchos::RCP<Epetra_Operator>& J,  
    const Teuchos::RCP<NOX::Epetra::Interface::Preconditioner>& iPrec, 
    const Teuchos::RCP<Epetra_Operator>& M,
    const NOX::Epetra::Vector& cloneVector,
    const Teuchos::RCP<NOX::Epetra::Scaling> scalingObject = Teuchos::null,
    Teuchos::RCP<Epetra_CrsMatrix> massMatrix=Teuchos::null);

  //! Destructor.
  virtual ~LinearSystemHymls();


  virtual bool applyJacobianInverse(Teuchos::ParameterList& linearSolverParams, 
				    const NOX::Epetra::Vector& input, 
				    NOX::Epetra::Vector& result);

  //! Reset the linear solver parameters.
  virtual void reset(Teuchos::ParameterList& linearSolverParams);
  
  //! get HYMLS solver
  Teuchos::RCP<HYMLS::Solver> getHymls() const {return hymls_;}

protected:
  
  /*! \brief Sets the epetra Jacobian operator in the Belos object.
     
      It is still called setAztecOO* because NOX/LOCA call that function.
  */

  virtual void setAztecOOJacobian() const;

  /*! \brief Sets the epetra Preconditioner operator in the Belos object.
    
      It is still called setAztecOO* because NOX/LOCA call that function.
  */
  virtual void setAztecOOPreconditioner() const;

  //! overloaded to include setup of deflation
  virtual bool createPreconditioner(const NOX::Epetra::Vector& x, 
                                    Teuchos::ParameterList& linearSolverParams,
                                    bool recomputeGraph) const;

  //! overloaded to include setup of deflation
  virtual bool recomputePreconditioner(const NOX::Epetra::Vector& x, 
                             Teuchos::ParameterList& linearSolverParams) const;


protected:

//!\name HYMLS data structures
//@{

//! iterative solver
Teuchos::RCP<HYMLS::Solver> hymls_;

//! mass matrix
Teuchos::RCP<const Epetra_CrsMatrix> massMatrix_;
  
//@}

};

} // namespace Epetra
} // namespace NOX


#endif
