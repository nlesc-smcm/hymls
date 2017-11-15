#ifndef HYMLS_TESTER_H
#define HYMLS_TESTER_H

#include "HYMLS_config.h"

#include <iostream>
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_RCP.hpp"

class Epetra_Operator;
class Epetra_CrsMatrix;
class Epetra_CrsGraph;
class Epetra_IntVector;
class Epetra_MultiVector;

#ifdef HYMLS_TESTING
#define HYMLS_TEST(CALLER,FCN,FILE,LINE) \
{bool status=true; \
try { \
HYMLS::Tester::msg_.clear(); \
HYMLS::Tester::msg_ <<"#####################################"<<std::endl; \
HYMLS::Tester::msg_ <<" RUN TEST "<<#FCN<<std::endl; \
HYMLS::Tester::msg_ <<" CALLER:  "<<CALLER<<std::endl; \
HYMLS::Tester::msg_ <<"#####################################"<<std::endl; \
HYMLS::Tools::out() <<std::endl<<HYMLS::Tester::msg_.str()<<std::endl; \
status = HYMLS::Tester:: FCN; \
} TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr,status); \
if (!status) { \
  std::stringstream ss; \
  ss << "HYMLS TEST '"<<#FCN<<"' FAILED!"<<std::endl; \
  ss << "Called by: "<<CALLER<<std::endl; \
  ss << "additional information:"<<std::endl; \
  ss << HYMLS::Tester::msg_.str() << std::endl; \
  HYMLS::Tester::numFailedTests_++; \
  HYMLS::Tools::Warning(ss.str(),FILE,LINE); } \
  HYMLS::Tester::msg_.str("");}
#else
#define HYMLS_TEST(CALLER,FCN,FILE,LINE)
#endif

namespace HYMLS
  {

class HierarchicalMap;
class OverlappingPartitioner;

//! this class tests if the interaction of HYMLS components
//! works as expected. It has a few static member functions that can
//! be used to assert certain properties of matrices and other
//! data structures during run time. To maintain efficiency, these 
//! functions should always be called using the HYMLS_TEST() macro,
//! which does nothing unless -DHYMLS_TESTING is defined.
class Tester
  {

public:

  //!\name the tests
  //!@{

  //! returns true if the input graph (i.e. the sparsity pattern of a matrix) is symmetric
  static bool isSymmetric(const Epetra_CrsGraph& G);

  //! returns true if the input matrix is symmetric
  static bool isSymmetric(const Epetra_CrsMatrix& A);

  //! returns true if the input matrix is the identity matrix
  static bool isIdentity(const Epetra_CrsMatrix& A);

  //! returns true if the input matrix is an F-matrix, where the 
  //! pressure is each dof'th unknown, starting from pvar
  static bool isFmatrix(const Epetra_CrsMatrix& A);

  //! check that the domain decomposition worked, i.e. there are no couplings between 
  //! interior nodes of two different subdomains.
  static bool isDDcorrect(const Epetra_CrsMatrix& A, const HYMLS::OverlappingPartitioner& hid);

  //! check that the vector V is divergence free, i.e. A*V has a zero p-part
  static bool isDivFree(const Epetra_CrsMatrix& A, const Epetra_MultiVector &V, double tol=1e-4);

  //! in the Schur-preconditioner, check that only V-P couplings are dropped which
  //! have been reduced to zero by a transformation
  static bool noPcouplingsDropped(const Epetra_CrsMatrix& transSC,
                                  const HierarchicalMap& sepObject);

  //! test if a given CrsMatrix does not contain numerical zeros except on the diagonal
  static bool noNumericalZeros(const Epetra_CrsMatrix& A);
  //!@}

  //! convert global index to string for output (gid => (i,j,k,v)
  static std::string gid2str(hymls_gidx gid); 

  //! tolerance if we have a numerical value that should be small  
  inline static double float_tol(){return 256.0*Teuchos::ScalarTraits<double>::eps();}
  
  //! this message buffer is printed when a test fails
  static std::stringstream msg_;

  //! returns "Tester"
  static std::string Label() {return "Tester";}

  //! some problem data, should be set once known
  static int dim_,dof_,pvar_;
  
  //! grid dimension
  static int nx_,ny_,nz_;
  
  //! do we have F-matrices
  static bool doFmatTests_;
  
  //! static counter how many tests have failed so far
  static int numFailedTests_;
  
  };

}// namespace HYMLS

#endif
