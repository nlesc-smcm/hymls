#ifndef HYMLS_DENSE_UTILS_H
#define HYMLS_DENSE_UTILS_H

#include "HYMLS_config.h"

#include "Teuchos_RCP.hpp"

class Epetra_MultiVector;
class Epetra_Operator;
class Epetra_SerialDenseMatrix;
class Epetra_SerialDenseVector;

namespace HYMLS {


class DenseUtils {

public: 

                
//! compute all eigenvalues and eigenvectors of the dense matrix 
//! A by using LAPACK routine DGEEV (no symmetry is              
//! taken into account).                                         
//! The eigenvalues are returned as (lambda_r + i*lambda_i).     
static int Eig(const Epetra_SerialDenseMatrix& A,
                     Epetra_SerialDenseVector& lambda_r,
                     Epetra_SerialDenseVector& lambda_i,
                     Epetra_SerialDenseMatrix& right_evecs,
                     Epetra_SerialDenseMatrix& left_evecs);

//! compute all eigenvalues and eigenvectors of the dense matrix 
//! pencio [A,B] by using LAPACK routine DGGEV (no symmetry is   
//! taken into account).                                         
//! The eigenvalues are returned as (alpha_r + i*alpha_i)/beta.  
static int Eig(const Epetra_SerialDenseMatrix& A,
               const Epetra_SerialDenseMatrix& B,
                     Epetra_SerialDenseVector& alpha_r,
                     Epetra_SerialDenseVector& alpha_i,
                     Epetra_SerialDenseVector& beta,
                     Epetra_SerialDenseMatrix& right_evecs,
                     Epetra_SerialDenseMatrix& left_evecs);


//! computes the dense matrix-matrix product of two MultiVectors        
//! and returns the result as a serial dense matrix,                    
//! C = V'W. W and V must be based on the same map and have the         
//! same number of vectors (=columns).                                  
//! If the argument C does not have the right size, it is resized.      
static int MatMul(const Epetra_MultiVector& V, const Epetra_MultiVector& W,
                       Epetra_SerialDenseMatrix& result);

//! given two multivectors V and W, computes V_orth*W and returns the result
//! as a new MultiVector Z. V, W and Z should have the same maps and numbers
//! of vectors (columns). The product is computed as Z=(I-VV')W.            
static int ApplyOrth(const Epetra_MultiVector& V, const Epetra_MultiVector& W,
                           Epetra_MultiVector& Z, Teuchos::RCP<const Epetra_MultiVector> BV=Teuchos::null,
                           bool reverse=false);


//! orthogonalize the columns of A using lapack's QR (Q is put into A, in fact)
static int Orthogonalize(Epetra_SerialDenseMatrix& A);

//! create a MultiVector view of a DenseMatrix
static Teuchos::RCP<Epetra_MultiVector> CreateView(Epetra_SerialDenseMatrix& A);

//! create a DenseMatrix view of a MultiVector
static Teuchos::RCP<Epetra_SerialDenseMatrix> CreateView(Epetra_MultiVector& A);

//! create a MultiVector view of a DenseMatrix
static Teuchos::RCP<const Epetra_MultiVector> CreateView(const Epetra_SerialDenseMatrix& A);

//! create a DenseMatrix view of a MultiVector
static Teuchos::RCP<const Epetra_SerialDenseMatrix> CreateView(const Epetra_MultiVector& A);

private: 

static std::string Label() {return "DenseUtils";}

};

}

#endif
