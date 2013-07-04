#ifndef HYMLS_TESTER_H
#define HYMLS_TESTER_H

#include <iostream>
#include "Teuchos_ScalarTraits.hpp"

class Epetra_CrsMatrix;
class Epetra_CrsGraph;
class Epetra_IntVector;

#ifdef TESTING
#define HYMLS_TEST(CALLER,FCN,FILE,LINE) \
{bool status=true; \
try { \
status = HYMLS::Tester:: FCN; \
} TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr,status); \
if (!status) { \
  std::stringstream ss; \
  ss << "HYMLS TEST '"<<#FCN<<"' FAILED!"<<std::endl; \
  ss << "Called by: "<<CALLER<<std::endl; \
  ss << "additional information:"<<std::endl; \
  ss << HYMLS::Tester::msg_.str() << std::endl; \
  HYMLS::Tools::Warning(ss.str(),FILE,LINE); }}
#else
#define HYMLS_TEST(CALLER,FCN,FILE,LINE)
#endif

namespace HYMLS
  {

//! this class tests if the interaction of HYMLS components
//! works as expected. It has a few static member functions that can
//! be used to assert certain properties of matrices and other
//! data structures during run time. To maintain efficiency, these 
//! functions should always be called using the HYMLS_TEST() macro,
//! which does nothing unless -DTESTING is defined.
class Tester
  {

public:

  //! returns true if the input graph (i.e. the sparsity pattern of a matrix) is symmetric
  static bool isSymmetric(const Epetra_CrsGraph& G);

  //! returns true if the input matrix is symmetric
  static bool isSymmetric(const Epetra_CrsMatrix& A);

  //! returns true if the input matrix is an F-matrix, where the 
  //! pressure is each dof'th unknown, starting from pvar
  static bool isFmatrix(const Epetra_CrsMatrix& A, int dof, int pvar);

  //! this test is a specialized test for the 3D Navier-Stokes equations on a C-grid.
  //! It checks that the interior velocities of full conservation tubes only connect 
  //! to interior pressures of the same tube.
  static bool areTubesCorrect(const Epetra_CrsMatrix& K,
                              const Epetra_IntVector& p_nodeType,
                              int dof, int pvar);


  //! tolerance if we have a numerical value that should be small  
  inline static double float_tol(){return 100.0*Teuchos::ScalarTraits<double>::eps();}
  
  //! this message buffer is printed when a test fails
  static std::stringstream msg_;

  //! returns "Tester"
  static std::string Label() {return "Tester";}
  
  };

}// namespace HYMLS

#endif
