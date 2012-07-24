#ifndef HYMLS_AUGMENTED_MATRIX_H
#define HYMLS_AUGMENTED_MATRIX_H

#include "Epetra_RowMatrix.h"
#include "Epetra_Map.h"
#include "Teuchos_RCP.hpp"

class Epetra_Comm;
class Epetra_Import;
class Epetra_Export;
class Epetra_Vector;
class Epetra_MultiVector;
class Epetra_SerialDenseMatrix;

namespace HYMLS {

//! implements the matrix A  V  where A is a standard row matrix,
//!                       W' C, 
//! V and W are multivecotrs and C is a serial dense matrix.
class AugmentedMatrix: public Epetra_RowMatrix 
{

 public:

   //! constructor
   AugmentedMatrix(Teuchos::RCP<Epetra_RowMatrix> A,
                   Teuchos::RCP<const Epetra_MultiVector> V,
                   Teuchos::RCP<const Epetra_MultiVector> W,
                   Teuchos::RCP<const Epetra_SerialDenseMatrix> C=Teuchos::null);
   
   //! @name Destructor
  //@{ 
    //! Destructor
    ~AugmentedMatrix() {};

  //@}

  //! get number of vectors in the border
  inline int NumBorderVectors() const 
    {
    return numBorderVectors_;
    }
    
  //! return the number of dense rows on this proc
  inline int NumMyDenseRows() const
    {
    return numMyDenseRows_;
    }
    
  //! get total number of dense rows
  inline int GetNumGlobalDenseRows() const
    {
    return NumBorderVectors();
    }
  
  //! @name Matrix data extraction routines
  //@{ 

    //! Returns the number of nonzero entries in MyRow.
    /*! 
    \param In
           MyRow - Local row.
    \param Out
	   NumEntries - Number of nonzero values present.
	  
    \return Integer error code, set to 0 if successful.
  */
    int NumMyRowEntries(int MyRow, int & NumEntries) const
      {
      NumEntries=-1;
      if (MyRow>=NumMyRows()) return -1;
      if (MyRow>=A_->NumMyRows())
        {
        NumEntries=A_->NumGlobalRows();
        if (C_!=Teuchos::null) NumEntries+=NumBorderVectors();
        return 0;
        }
      int ierr = A_->NumMyRowEntries(MyRow,NumEntries);
      if (ierr) return ierr;
      NumEntries += NumBorderVectors();
      return 0;
      }


    //! Returns the maximum of NumMyRowEntries() over all rows.
    int MaxNumEntries() const
      {
      if (NumMyDenseRows()>0) return NumGlobalCols();
      return A_->MaxNumEntries()+NumBorderVectors();
      }

    //! Returns a copy of the specified local row in user-provided arrays.
    /*! 
    \param In
           MyRow - Local row to extract.
    \param In
	   Length - Length of Values and Indices.
    \param Out
	   NumEntries - Number of nonzero entries extracted.
    \param Out
	   Values - Extracted values for this row.
    \param Out
	   Indices - Extracted local column indices for the corresponding values.
	  
    \return Integer error code, set to 0 if successful.
  */
    int ExtractMyRowCopy(int MyRow, int Length, int & NumEntries, double *Values, int * Indices) const;

    //! Returns a copy of the main diagonal in a user-provided vector.
    /*! 
    \param Out
	   Diagonal - Extracted main diagonal.

    \return Integer error code, set to 0 if successful.
  */
    int ExtractDiagonalCopy(Epetra_Vector & Diagonal) const;
  //@}
  
  //! @name Mathematical functions
  //@{ 

    //! Returns the result of a Epetra_RowMatrix multiplied by a Epetra_MultiVector X in Y.
    /*! 
    \param In
	   TransA -If true, multiply by the transpose of matrix, otherwise just use matrix.
    \param In
	   X - A Epetra_MultiVector of dimension NumVectors to multiply with matrix.
    \param Out
	   Y -A Epetra_MultiVector of dimension NumVectorscontaining result.

    \return Integer error code, set to 0 if successful.
  */
    int Multiply(bool TransA, const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

    //! Returns result of a local-only solve using a triangular Epetra_RowMatrix with Epetra_MultiVectors X and Y.
    /*! This method will perform a triangular solve independently on each processor of the parallel machine.
        No communication is performed.
    \param In
	   Upper -If true, solve Ux = y, otherwise solve Lx = y.
    \param In
	   Trans -If true, solve transpose problem.
    \param In
	   UnitDiagonal -If true, assume diagonal is unit (whether it's stored or not).
    \param In
	   X - A Epetra_MultiVector of dimension NumVectors to solve for.
    \param Out
	   Y -A Epetra_MultiVector of dimension NumVectors containing result.

    \return Integer error code, set to 0 if successful.
  */
    int Solve(bool Upper, bool Trans, bool UnitDiagonal, const Epetra_MultiVector& X, 
		      Epetra_MultiVector& Y) const;

    //! Computes the sum of absolute values of the rows of the Epetra_RowMatrix, results returned in x.
    /*! The vector x will return such that x[i] will contain the inverse of sum of the absolute values of the 
        \e this matrix will be scaled such that A(i,j) = x(i)*A(i,j) where i denotes the global row number of A
        and j denotes the global column number of A.  Using the resulting vector from this function as input to LeftScale()
	will make the infinity norm of the resulting matrix exactly 1.
    \param Out
	   x -A Epetra_Vector containing the row sums of the \e this matrix. 
	   \warning It is assumed that the distribution of x is the same as the rows of \e this.

    \return Integer error code, set to 0 if successful.
  */
    int InvRowSums(Epetra_Vector& x) const;

    //! Scales the Epetra_RowMatrix on the left with a Epetra_Vector x.
    /*! The \e this matrix will be scaled such that A(i,j) = x(i)*A(i,j) where i denotes the row number of A
        and j denotes the column number of A.
    \param In
	   x -A Epetra_Vector to solve for.

    \return Integer error code, set to 0 if successful.
  */
    int LeftScale(const Epetra_Vector& x);

    //! Computes the sum of absolute values of the columns of the Epetra_RowMatrix, results returned in x.
    /*! The vector x will return such that x[j] will contain the inverse of sum of the absolute values of the 
        \e this matrix will be sca such that A(i,j) = x(j)*A(i,j) where i denotes the global row number of A
        and j denotes the global column number of A.  Using the resulting vector from this function as input to 
	RighttScale() will make the one norm of the resulting matrix exactly 1.
    \param Out
	   x -A Epetra_Vector containing the column sums of the \e this matrix. 
	   \warning It is assumed that the distribution of x is the same as the rows of \e this.

    \return Integer error code, set to 0 if successful.
  */
    int InvColSums(Epetra_Vector& x) const;

    //! Scales the Epetra_RowMatrix on the right with a Epetra_Vector x.
    /*! The \e this matrix will be scaled such that A(i,j) = x(j)*A(i,j) where i denotes the global row number of A
        and j denotes the global column number of A.
    \param In
	   x -The Epetra_Vector used for scaling \e this.

    \return Integer error code, set to 0 if successful.
  */
    int RightScale(const Epetra_Vector& x);
  //@}
  
  //! @name Attribute access functions
  //@{

    //! If FillComplete() has been called, this query returns true, otherwise it returns false.
    bool Filled() const {return A_->Filled();}

    //! Returns the infinity norm of the global matrix.
    /* Returns the quantity \f$ \| A \|_\infty\f$ such that
       \f[\| A \|_\infty = \max_{1\lei\len} \sum_{i=1}^m |a_{ij}| \f].
    */ 
    double NormInf() const;

    //! Returns the one norm of the global matrix.
    /* Returns the quantity \f$ \| A \|_1\f$ such that
       \f[\| A \|_1= \max_{1\lej\len} \sum_{j=1}^n |a_{ij}| \f].
    */ 
    double NormOne() const;

    //! Returns the number of nonzero entries in the global matrix.
    /*
      Note that depending on the matrix implementation, it is sometimes
      possible to have some nonzeros that appear on multiple processors.
      In that case, those nonzeros may be counted multiple times (also
      depending on the matrix implementation).
    */
    int NumGlobalNonzeros() const 
      {
      int val=A_->NumGlobalNonzeros();
      val+=2*NumBorderVectors()*A_->NumGlobalRows();
      val+=NumBorderVectors()*NumBorderVectors();
      return val;
      }

    //! Returns the number of global matrix rows.
    int NumGlobalRows() const
      {
      return A_->NumGlobalRows() + NumBorderVectors();
      }

    //! Returns the number of global matrix columns.
    int NumGlobalCols() const
      {
      return A_->NumGlobalCols() + NumBorderVectors();
      }

    //! Returns the number of global nonzero diagonal entries, based on global row/column index comparisons.
    int NumGlobalDiagonals() const
      {
      return A_->NumGlobalDiagonals() + NumBorderVectors();      
      }
    
    //! Returns the number of nonzero entries in the calling processor's portion of the matrix.
    int NumMyNonzeros() const
      {
      int val=A_->NumMyNonzeros() + NumBorderVectors()*A_->NumMyRows();
      val+=NumMyDenseRows()*NumGlobalRows();
      return val;
      }

    //! Returns the number of matrix rows owned by the calling processor.
    int NumMyRows() const 
      {
      return A_->NumMyRows() + NumMyDenseRows();
      }

    //! Returns the number of matrix columns owned by the calling processor.
    int NumMyCols() const 
      {
      return A_->NumMyCols()+NumBorderVectors();
      }

    //! Returns the number of local nonzero diagonal entries, based on global row/column index comparisons.
    int NumMyDiagonals() const
      {
      return A_->NumMyDiagonals() + NumMyDenseRows();
      }

    //! If matrix is lower triangular in local index space, this query returns true, otherwise it returns false.
    bool LowerTriangular() const {return false;};

    //! If matrix is upper triangular in local index space, this query returns true, otherwise it returns false.
    bool UpperTriangular() const {return false;}    

    //! Returns the Epetra_Map object associated with the rows of this matrix.
    const Epetra_Map & RowMatrixRowMap() const
      {
      return *rowMap_;
      }

    //! Returns the Epetra_Map object associated with the columns of this matrix.
    const Epetra_Map & RowMatrixColMap() const
      {
      return *colMap_;
      }

    //! Returns the Epetra_Import object that contains the import operations for distributed operations.
    const Epetra_Import * RowMatrixImporter() const
      {
      return import_.get();
      }
  //@}

  //! \name Epetra_Operator implementation
  //@{
  //! @name Attribute set methods
  //@{ 

    //! If set true, transpose of this operator will be applied.
    /*! This flag allows the transpose of the given operator to be used implicitly.  Setting 
this flag
        affects only the Apply() and ApplyInverse() methods.  If the implementation of this 
interface 
        does not support transpose use, this method should return a value of -1.
      
    \param In
           UseTranspose -If true, multiply by the transpose of operator, otherwise just use 
operator.

    \return Integer error code, set to 0 if successful.  Set to -1 if this implementation 
does not support transpose.
  */
    
    int SetUseTranspose(bool UseTranspose);
  //@}
  
  //! @name Mathematical functions
  //@{ 

    //! Returns the result of a Epetra_Operator applied to a Epetra_MultiVector X in Y.
    /*! 
    \param In
           X - A Epetra_MultiVector of dimension NumVectors to multiply with matrix.
    \param Out
           Y -A Epetra_MultiVector of dimension NumVectors containing result.

    \return Integer error code, set to 0 if successful.
  */
    int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

    //! Returns the result of a Epetra_Operator inverse applied to an Epetra_MultiVector X in Y.
    /*! 
    \param In
           X - A Epetra_MultiVector of dimension NumVectors to solve for.
    \param Out
           Y -A Epetra_MultiVector of dimension NumVectors containing result.

    \return Integer error code, set to 0 if successful.

    \warning In order to work with AztecOO, any implementation of this method must 
              support the case where X and Y are the same object.
  */
    int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

//@}
  //! @name Attribute access functions
  //@{ 

    //! Returns a character string describing the operator
    const char * Label() const {return label_.c_str();}

    //! Returns the current UseTranspose setting.
    bool UseTranspose() const {return useTranspose_;}

    //! Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
    bool HasNormInf() const;

    //! Returns a pointer to the Epetra_Comm communicator associated with this operator.
    const Epetra_Comm & Comm() const {return *comm_;}

    //! Returns the Epetra_Map object associated with the domain of this operator.
    const Epetra_Map & OperatorDomainMap() const {return *domainMap_;}

    //! Returns the Epetra_Map object associated with the range of this operator.
    const Epetra_Map & OperatorRangeMap() const {return *rangeMap_;}
  //@}
  
  //@}

  //! from Epetra_SrcDistObject
  const Epetra_Map& Map() const {return *rowMap_;}
  
protected:

  //!
  Teuchos::RCP<Epetra_RowMatrix> A_;
  //!
  Teuchos::RCP<const Epetra_MultiVector> V_,W_;
  //! W gathered on the owning proc
  Teuchos::RCP<Epetra_MultiVector> Wloc_;
  //!
  Teuchos::RCP<const Epetra_SerialDenseMatrix> C_;
  
  //!
  int numBorderVectors_;
  //!
  int numMyDenseRows_;
  
  //!
  Teuchos::RCP<const Epetra_Comm> comm_;
  
  //!
  Teuchos::RCP<Epetra_Map> rowMap_;
  //!
  Teuchos::RCP<Epetra_Map> colMap_;
  //!
  Teuchos::RCP<Epetra_Map> rangeMap_;
  //!
  Teuchos::RCP<Epetra_Map> domainMap_;
  
  //!
  Teuchos::RCP<Epetra_Import> import_;

  //!
  std::string label_;
  
  //!
  bool useTranspose_;  

};

}

#endif /* EPETRA_ROWMATRIX_H */
