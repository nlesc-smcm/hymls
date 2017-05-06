#include "HYMLS_config.h"

#include "BelosConfigDefs.hpp"
#include "NOX_Epetra_Scaling.H"


class Epetra_MultiVector;
class Epetra_Operator;


namespace HYMLS {


//! this class extends the NOX diagonal scaling facility to
//! a block 2x2 scaling
class BlockScaling : public NOX::Epetra::Scaling
{
public:

  //!Constructor
  BlockScaling();
   //!Destuctor
  ~BlockScaling();


  //! Computes Row Sum scaling diagonal vectors.  Only needs to be called if a row or column sum scaling has been requested.
  void computeScaling(const Epetra_LinearProblem& problem);

  //get the scaling coeefcients from the model
  void setCoefficients(double Sl11, double Sl12, double Sl21, double Sl22,
                       double Sr11, double Sr12, double Sr21, double Sr22);
  //! Scales the linear system.
  void scaleLinearSystem(Epetra_LinearProblem& problem);

  //! Remove the scaling from the linear system.
  void unscaleLinearSystem(Epetra_LinearProblem& problem);

protected:

  //2x2 right scaling Sr
  double Sr11_, Sr12_, Sr21_, Sr22_;
  //inverse of Sr
  double iSr11_, iSr12_, iSr21_, iSr22_;


  
  //2x2 left scaling Sl=inv(Sr)*some factor
  double Sl11_, Sl12_, Sl21_, Sl22_;
  //inverse of Sl
  double iSl11_, iSl12_, iSl21_, iSl22_;


};




}
