
#include "HYMLS_AugmentedMatrix.H"

#include "Epetra_Comm.h"
#include "Epetra_Import.h"
#include "Epetra_Export.h"
#include "Epetra_Vector.h"
#include "Epetra_MultiVector.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseVector.h"
#include "Epetra_Map.h"

#include "HYMLS_config.h"
#include "HYMLS_Tools.H"
#include "HYMLS_MatrixUtils.H"

namespace HYMLS {


   // constructor
   AugmentedMatrix::AugmentedMatrix(Teuchos::RCP<Epetra_RowMatrix> A,
                   Teuchos::RCP<const Epetra_MultiVector> V,
                   Teuchos::RCP<const Epetra_MultiVector> W,
                   Teuchos::RCP<const Epetra_SerialDenseMatrix> C)
  {
  label_="AugmentedMatrix";
  HYMLS_PROF2(label_,"Constructor");
  useTranspose_=false;
  if (!A->Filled())
    {
    // this is really just so we have a column map already and don't have to woory
    // about rebuilding our column map when A gets filled.
    HYMLS::Tools::Error("AugmentedMatrix: A should be Filled().",__FILE__,__LINE__);
    }
  
  if (V->NumVectors()!=W->NumVectors())
    {
    HYMLS::Tools::Error("V and W in AugmentedMatrix must be of same shape",
        __FILE__,__LINE__);
    }
  if ((V->Map().SameAs(A->RowMatrixRowMap())==false) ||
      (W->Map().SameAs(A->RowMatrixRowMap())==false))
    {
    HYMLS_DEBVAR(A->RowMatrixRowMap());
    HYMLS_DEBVAR(V->Map());
    HYMLS_DEBVAR(W->Map());
    HYMLS::Tools::Error("V and W in AugmentedMatrix must have same map as A",
        __FILE__,__LINE__);
    }
    
  V_=V; W_=W; A_=A;
  numBorderVectors_=V_->NumVectors();

  C_=C;
  if (C_==Teuchos::null)
    {
    C_=Teuchos::rcp(new Epetra_SerialDenseMatrix(numBorderVectors_,numBorderVectors_));
    }
  if (C_->N()!=C_->M() || C_->N()!=numBorderVectors_)
    {
    HYMLS::Tools::Error("shape of C does not mach that of V and W",__FILE__,__LINE__);
    }

  comm_=Teuchos::rcp(&(A_->Comm()), false);

  // the W' and C part are officially owned by the last processor
  numMyDenseRows_=0;
  if (comm_->MyPID()==comm_->NumProc()-1)
    {
    numMyDenseRows_ = numBorderVectors_;
    }
    
  Wloc_ = HYMLS::MatrixUtils::Gather(*W_,comm_->NumProc()-1);

  // create the maps
  const Epetra_Map& rowMapA = A_->RowMatrixRowMap();
  const Epetra_Map& colMapA = A_->RowMatrixColMap();
  if ((rowMapA.SameAs(A_->OperatorDomainMap())==false)
  ||  (rowMapA.SameAs(A_->OperatorRangeMap())==false))
    {
    HYMLS::Tools::Error("AugmentedMatrix: A cannot have different "
                        " row-, range and domain maps right now.",__FILE__,__LINE__);
    }
    
  HYMLS_DEBVAR(A_->NumMyRows());
  HYMLS_DEBVAR(V_->NumVectors());
  HYMLS_DEBVAR(NumMyRows());
  
  hymls_gidx* id = new hymls_gidx[NumMyRows()];
  for (int i=0;i<A_->NumMyRows();i++)
    {
    id[i] = rowMapA.GID64(i);
    }
  hymls_gidx k = rowMapA.MaxAllGID64()+1;
  for (int i=A_->NumMyRows();i<NumMyRows();i++)
    {
    id[i]=k++;
    }
  HYMLS_DEBUG("Construct row map");
  rowMap_ = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1),
      NumMyRows(), id, (hymls_gidx)rowMapA.IndexBase64(), *comm_));
  delete [] id;
  
  int numMyColEntries = NumMyDenseRows() > 0 ? NumGlobalRows64() :
    colMapA.NumMyElements() + numBorderVectors_;
  id = new hymls_gidx[numMyColEntries];
  int pos=0;
  for (int i=0;i<colMapA.NumMyElements();i++)
    {
    id[pos++]=colMapA.GID64(i);
    }
  k=colMapA.MaxAllGID64()+1;
  for (int i=0;i<NumBorderVectors();i++)
    {
    id[pos++]=k++;
    }
  for (int i=0;i<Wloc_->MyLength();i++)
    {
    if (colMapA.MyGID(Wloc_->Map().GID64(i))==false)
    id[pos++]=Wloc_->Map().GID64(i);
    }
  colMap_ = Teuchos::rcp(new Epetra_Map((hymls_gidx)(-1),
      numMyColEntries, id, (hymls_gidx)rowMapA.IndexBase64(), *comm_));
  delete [] id;
  HYMLS_DEBVAR(*rowMap_);
  HYMLS_DEBVAR(*colMap_);
  rangeMap_ = rowMap_;
  domainMap_ = rowMap_;
  import_=Teuchos::rcp(new Epetra_Import(*colMap_,*domainMap_));
  }
     
    // Returns a copy of the specified local row in user-provided arrays.
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
int AugmentedMatrix::ExtractMyRowCopy(int MyRow, int Length, int & NumEntries, 
  double *Values, int * Indices) const
  {
  HYMLS_PROF3(label_,"ExtractMyRowCopy");
  int ierr=NumMyRowEntries(MyRow, NumEntries);
  if (ierr) 
    {
    Tools::Warning("Error "+Teuchos::toString(ierr)+" returned from call NumMyRowEntries",
        __FILE__,__LINE__);
    return ierr;
    }
  if (NumEntries>Length) 
    {
    Tools::Warning("insufficient space provided",__FILE__,__LINE__);
    return -2;
    }
  int lenA=0;
  if (MyRow<A_->NumMyRows())
    {
    HYMLS_DEBUG("   belongs to A V");
    ierr = A_->ExtractMyRowCopy(MyRow,Length,lenA,Values,Indices);
    if (ierr)
      {
      Tools::Warning("error "+Teuchos::toString(ierr)+" returned from A_->ExtractMyRowCopy",
        __FILE__, __LINE__);
      return ierr;
      }
    
    for (int k=0;k<NumBorderVectors();k++)
      {
      Values[lenA+k] = (*V_)[k][MyRow];
      Indices[lenA+k]= colMap_->LID(A_->NumGlobalRows64()+k);
      }
    }
  else
    {
    HYMLS_DEBUG("   belongs to W C");
    int k=MyRow-A_->NumMyRows();
    if (Length!=Wloc_->MyLength() + NumBorderVectors())
      {
      Tools::Warning("dense row length mismatch",__FILE__,__LINE__);
      return -3; // length mismatch for dense row
      }
    for (int i=0;i<Wloc_->MyLength();i++)
      {
      Values[i] = (*Wloc_)[k][i];
      Indices[i]= colMap_->LID(Wloc_->Map().GID64(i));
      }
    for (int i=0;i<NumBorderVectors();i++)
      {
      Values[Wloc_->MyLength()+i]=(*C_)[k][i];
      Indices[Wloc_->MyLength()+i]=colMap_->LID(Wloc_->Map().MaxAllGID64()+i+1);
      }
    }
#ifdef HYMLS_DEBUGGING
for (int i=0;i<NumEntries;i++)
  {
  Tools::deb() << Map().GID64(MyRow) << " " << colMap_->GID64(Indices[i]) << " " << Values[i] << std::endl;
  }
#endif  
  return 0;
  }        

    // Returns a copy of the main diagonal in a user-provided vector.
    /*! 
    \param Out
	   Diagonal - Extracted main diagonal.

    \return Integer error code, set to 0 if successful.
  */
    int AugmentedMatrix::ExtractDiagonalCopy(Epetra_Vector & Diagonal) const
    {
    return -99; // not implemented
    }
  //@}
  
  // @name Mathematical functions
  //@{ 

    // Returns the result of a Epetra_RowMatrix multiplied by a Epetra_MultiVector X in Y.
    /*! 
    \param In
	   TransA -If true, multiply by the transpose of matrix, otherwise just use matrix.
    \param In
	   X - A Epetra_MultiVector of dimension NumVectors to multiply with matrix.
    \param Out
	   Y -A Epetra_MultiVector of dimension NumVectorscontaining result.

    \return Integer error code, set to 0 if successful.
  */
    int AugmentedMatrix::Multiply(bool TransA, const Epetra_MultiVector& X, 
        Epetra_MultiVector& Y) const
  {
  HYMLS_PROF3(label_,"Multiply");  
  return -99; // not implemented
  }        

    // Returns result of a local-only solve using a triangular Epetra_RowMatrix with Epetra_MultiVectors X and Y.
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
    int AugmentedMatrix::Solve(bool Upper, bool Trans, bool UnitDiagonal, 
        const Epetra_MultiVector& X, 
              Epetra_MultiVector& Y) const
  {
  HYMLS_PROF2(label_,"Solve");  
  return -99; // not implemented
  }              

    // Computes the sum of absolute values of the rows of the Epetra_RowMatrix, results returned in x.
    /*! The vector x will return such that x[i] will contain the inverse of sum of the absolute values of the 
        \e this matrix will be scaled such that A(i,j) = x(i)*A(i,j) where i denotes the global row number of A
        and j denotes the global column number of A.  Using the resulting vector from this function as input to LeftScale()
	will make the infinity norm of the resulting matrix exactly 1.
    \param Out
	   x -A Epetra_Vector containing the row sums of the \e this matrix. 
	   \warning It is assumed that the distribution of x is the same as the rows of \e this.

    \return Integer error code, set to 0 if successful.
  */
    int AugmentedMatrix::InvRowSums(Epetra_Vector& x) const
  {
  return -99; // not implemented
  }
    // Scales the Epetra_RowMatrix on the left with a Epetra_Vector x.
    /*! The \e this matrix will be scaled such that A(i,j) = x(i)*A(i,j) where i denotes the row number of A
        and j denotes the column number of A.
    \param In
	   x -A Epetra_Vector to solve for.

    \return Integer error code, set to 0 if successful.
  */
    int AugmentedMatrix::LeftScale(const Epetra_Vector& x)
      {
      return -99; // not implemented      
      }

    // Computes the sum of absolute values of the columns of the Epetra_RowMatrix, results returned in x.
    /*! The vector x will return such that x[j] will contain the inverse of sum of the absolute values of the 
        \e this matrix will be sca such that A(i,j) = x(j)*A(i,j) where i denotes the global row number of A
        and j denotes the global column number of A.  Using the resulting vector from this function as input to 
	RighttScale() will make the one norm of the resulting matrix exactly 1.
    \param Out
	   x -A Epetra_Vector containing the column sums of the \e this matrix. 
	   \warning It is assumed that the distribution of x is the same as the rows of \e this.

    \return Integer error code, set to 0 if successful.
  */
    int AugmentedMatrix::InvColSums(Epetra_Vector& x) const
      {
      return -99; // not implemented      
      }

    // Scales the Epetra_RowMatrix on the right with a Epetra_Vector x.
    /*! The \e this matrix will be scaled such that A(i,j) = x(j)*A(i,j) where i denotes the global row number of A
        and j denotes the global column number of A.
    \param In
	   x -The Epetra_Vector used for scaling \e this.

    \return Integer error code, set to 0 if successful.
  */
    int AugmentedMatrix::RightScale(const Epetra_Vector& x)
      {
      return -99; // not implemented      
      }
  //@}
  
  // @name Attribute access functions
  //@{ 

    // Returns the infinity norm of the global matrix.
    /* Returns the quantity \f$ \| A \|_\infty\f$ such that
       \f[\| A \|_\infty = \max_{1\lei\len} \sum_{i=1}^m |a_{ij}| \f].
    */ 
    double AugmentedMatrix::NormInf() const {return -1.0;}

    // Returns the one norm of the global matrix.
    /* Returns the quantity \f$ \| A \|_1\f$ such that
       \f[\| A \|_1= \max_{1\lej\len} \sum_{j=1}^n |a_{ij}| \f].
    */ 
    double AugmentedMatrix::NormOne() const {return -1.0;}


  // \name Epetra_Operator implementation
  //@{
  // @name Attribute set methods
  //@{ 

    // If set true, transpose of this operator will be applied.
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
    
    int AugmentedMatrix::SetUseTranspose(bool UseTranspose)
      {
      return -99; // not implemented
      }
  //@}
  
  // @name Mathematical functions
  //@{ 

    // Returns the result of a Epetra_Operator applied to a Epetra_MultiVector X in Y.
    /*! 
    \param In
           X - A Epetra_MultiVector of dimension NumVectors to multiply with matrix.
    \param Out
           Y -A Epetra_MultiVector of dimension NumVectors containing result.

    \return Integer error code, set to 0 if successful.
  */
    int AugmentedMatrix::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
      {
      HYMLS_PROF3(label_,"Apply");
      return this->Multiply(useTranspose_,X,Y);
      }

    // Returns the result of a Epetra_Operator inverse applied to an Epetra_MultiVector X in Y.
    /*! 
    \param In
           X - A Epetra_MultiVector of dimension NumVectors to solve for.
    \param Out
           Y -A Epetra_MultiVector of dimension NumVectors containing result.

    \return Integer error code, set to 0 if successful.

    \warning In order to work with AztecOO, any implementation of this method must 
              support the case where X and Y are the same object.
  */
    int AugmentedMatrix::ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
      {
      HYMLS_PROF3(label_,"ApplyInverse");
      return -99; // not implemented
      }

    // Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
    bool AugmentedMatrix::HasNormInf() const {return false;}

  //@}
  
}//namespace HYMLS
