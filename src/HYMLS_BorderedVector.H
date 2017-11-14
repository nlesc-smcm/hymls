#ifndef HYMLS_BORDERED_VECTOR_H
#define HYMLS_BORDERED_VECTOR_H

#include <math.h>
#include <vector>

#include <Trilinos_version.h>

#include <Teuchos_RCP.hpp>

#include "Epetra_MultiVector.h"

#include "BelosMultiVec.hpp"
#include "BelosOperator.hpp"
#include "BelosTypes.hpp"
#include "BelosEpetraAdapter.hpp"

#include "HYMLS_Tools.H"

class Epetra_SerialDenseMatrix;

namespace HYMLS {

class BorderedVector
  {
  //! Pointers to multivector and matrix
  Teuchos::RCP<Epetra_MultiVector> first_;
  Teuchos::RCP<Epetra_MultiVector> second_;

public:
  // default constructor
  BorderedVector() = delete;

  BorderedVector(const Epetra_BlockMap &map1, const Epetra_BlockMap &map2,
    int numVectors, bool zeroOut = true);

  // Copy constructor
  BorderedVector(const BorderedVector &source);

  BorderedVector(const Teuchos::RCP<Epetra_MultiVector> &first,
    const Teuchos::RCP<Epetra_MultiVector> &second);

  BorderedVector(const Teuchos::RCP<Epetra_MultiVector> &first,
    const Teuchos::RCP<Epetra_SerialDenseMatrix> &second);

  BorderedVector(const Epetra_MultiVector &first, const Epetra_MultiVector &second);

  BorderedVector(const Epetra_MultiVector &first, const Epetra_SerialDenseMatrix &second);

  // const
  BorderedVector(Epetra_DataAccess CV, const BorderedVector &source,
    const std::vector<int> &index);

  // nonconst
  BorderedVector(Epetra_DataAccess CV, BorderedVector &source,
    const std::vector<int> &index);

  // const
  BorderedVector(Epetra_DataAccess CV, const BorderedVector &source,
    int startIndex, int numVectors);

  // nonconst
  BorderedVector(Epetra_DataAccess CV, BorderedVector &source,
    int startIndex, int numVectors);

  virtual ~BorderedVector() {}

  // Assignment operator
  BorderedVector &operator=(const BorderedVector &Source);

  // Insertion operator
  friend std::ostream &operator<<(std::ostream &out, const BorderedVector &mv);

  // Get the rcpointers
  Teuchos::RCP<Epetra_MultiVector> First();
  Teuchos::RCP<Epetra_MultiVector> Second();
  Teuchos::RCP<Epetra_MultiVector> Vector();
  Teuchos::RCP<Epetra_SerialDenseMatrix> Border();

  Teuchos::RCP<Epetra_MultiVector> First() const;
  Teuchos::RCP<Epetra_MultiVector> Second() const;
  Teuchos::RCP<Epetra_MultiVector> Vector() const;
  Teuchos::RCP<Epetra_SerialDenseMatrix> Border() const;

  // Method to set the border since the assignment operator does not work
  // properly with the views that we use
  int SetBorder(const Epetra_SerialDenseMatrix &mv2);

  // Get number of vectors in each multivector
  int NumVecs() const;

  // Get the global length of the combined multivector
  int GlobalLength() const;

  // Get the global length of the combined multivector
  long long GlobalLength64() const;

  // Get the local length of the combined multivector
  int MyLength() const;

  // Query the stride
  bool ConstantStride() const;

  // this = alpha*A*B + scalarThis*this
  int Multiply(char transA, char transB, double scalarAB,
    const BorderedVector &A, const Epetra_MultiVector &B,
    double scalarThis);

  // this = scalarA*A + scalarThis*this
  int Update(double scalarA, const BorderedVector &A, double scalarThis);

  // this = scalarA*A + scalarB*B + scalarThis*this
  int Update(double scalarA, const BorderedVector &A,
    double scalarB, const BorderedVector &B, double scalarThis);

  // b[j] := this[j]^T * A[j]
  int Dot(const BorderedVector& A, std::vector<double> &b1) const;

  // result[j] := this[j]^T * A[j]
  int Dot(const BorderedVector& A, double *result) const;

  int Scale(double scalarValue);

  int Norm1(std::vector<double> &result) const;

  int Norm2(double *result) const;

  int Norm2(std::vector<double> &result) const;

  int NormInf(std::vector<double> &result) const;

  int Random();

  int PutScalar(double alpha);

  void Print(std::ostream &os) const;
  };


//------------------------------------------------------------------
// Specialization of MultiVectorTraits for Belos,
//  adapted from BelosEpetraAdapter.hpp, for better documentation go there.
//------------------------------------------------------------------
  }

namespace Belos
  {
template<>
class MultiVecTraits <double, HYMLS::BorderedVector>
  {

public:

  static Teuchos::RCP<HYMLS::BorderedVector>
  Clone (const HYMLS::BorderedVector &mv, const int numVecs)
    {
    TEUCHOS_TEST_FOR_EXCEPTION(
      numVecs <= 0, std::invalid_argument,
      "Belos::MultiVecTraits<double, HYMLS::BorderedVector>::"
      "Clone(mv, numVecs = " << numVecs << "): "
      "outNumVecs must be positive.");

    return Teuchos::rcp
      (new HYMLS::BorderedVector(mv.First()->Map(),
        mv.Second()->Map(), numVecs, false));
    }

  static Teuchos::RCP<HYMLS::BorderedVector>
  CloneCopy (const HYMLS::BorderedVector &mv)
    {
    return Teuchos::rcp(new HYMLS::BorderedVector(mv));
    }

  static Teuchos::RCP<HYMLS::BorderedVector>
  CloneCopy (const HYMLS::BorderedVector &mv, const std::vector<int> &index)
    {
    const int inNumVecs  = mv.NumVecs();
    const int outNumVecs = index.size();
    TEUCHOS_TEST_FOR_EXCEPTION(outNumVecs == 0, std::invalid_argument,
      "Belos::MultiVecTraits<double, HYMLS::BorderedVector>::"
      "CloneCopy(mv, index = {}): At least one vector must be"
      " cloned from mv.");

    if (outNumVecs > inNumVecs)
      {
      std::ostringstream os;
      os << "Belos::MultiVecTraits<double, Combined_Operator>::"
        "CloneCopy(mv, index = {";
      for (int k = 0; k < outNumVecs - 1; ++k)
        os << index[k] << ", ";
      os << index[outNumVecs-1] << "}): There are " << outNumVecs
         << " indices to copy, but only " << inNumVecs << " columns of mv.";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, os.str());
      }

    return Teuchos::rcp(new HYMLS::BorderedVector(Copy, mv, index));
    }

  static Teuchos::RCP<HYMLS::BorderedVector>
  CloneCopy (const HYMLS::BorderedVector &mv, const Teuchos::Range1D &index)
    {
    const int inNumVecs   = mv.NumVecs();
    const int outNumVecs  = index.size();
    const bool validRange = outNumVecs > 0 && index.lbound() >= 0 &&
      index.ubound() < inNumVecs;

    if (! validRange)
      {
      std::ostringstream os;
      os << "Belos::MultiVecTraits<double, HYMLS::BorderedVector>::Clone(mv,"
        "index=[" << index.lbound() << ", " << index.ubound() << "]): ";
      TEUCHOS_TEST_FOR_EXCEPTION(outNumVecs == 0, std::invalid_argument,
        os.str() << "Column index range must be nonempty.");
      TEUCHOS_TEST_FOR_EXCEPTION(index.lbound() < 0, std::invalid_argument,
        os.str() << "Column index range must be nonnegative.");
      TEUCHOS_TEST_FOR_EXCEPTION(index.ubound() >= inNumVecs, std::invalid_argument,
        os.str() << "Column index range must not exceed "
        "number of vectors " << inNumVecs << " in the "
        "input multivector.");
      }

    return Teuchos::rcp
      (new HYMLS::BorderedVector(Copy, mv, index.lbound(), index.size()));

    }

  static Teuchos::RCP<HYMLS::BorderedVector>
  CloneViewNonConst (HYMLS::BorderedVector &mv, const std::vector<int> &index)
    {
    const int inNumVecs  = mv.NumVecs();
    const int outNumVecs = index.size();
    // Simple, inexpensive tests of the index vector.

    TEUCHOS_TEST_FOR_EXCEPTION(outNumVecs == 0, std::invalid_argument,
      "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::"
      "CloneViewNonConst(mv, index = {}): The output view "
      "must have at least one column.");
    if (outNumVecs > inNumVecs)
      {
      std::ostringstream os;
      os << "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::"
        "CloneViewNonConst(mv, index = {";
      for (int k = 0; k < outNumVecs - 1; ++k)
        os << index[k] << ", ";
      os << index[outNumVecs-1] << "}): There are " << outNumVecs
         << " indices to view, but only " << inNumVecs << " columns of mv.";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, os.str());
      }
    return Teuchos::rcp
      (new HYMLS::BorderedVector(View, mv, index));
    }

  static Teuchos::RCP<HYMLS::BorderedVector>
  CloneViewNonConst (HYMLS::BorderedVector& mv, const Teuchos::Range1D& index)
    {
    const bool validRange = index.size() > 0 &&
      index.lbound() >= 0 &&
      index.ubound() < mv.NumVecs();

    if (! validRange)
      {
      std::ostringstream os;
      os << "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::CloneView"
        "NonConst(mv,index=[" << index.lbound() << ", " << index.ubound()
         << "]): ";
      TEUCHOS_TEST_FOR_EXCEPTION(index.size() == 0, std::invalid_argument,
        os.str() << "Column index range must be nonempty.");
      TEUCHOS_TEST_FOR_EXCEPTION(index.lbound() < 0, std::invalid_argument,
        os.str() << "Column index range must be nonnegative.");
      TEUCHOS_TEST_FOR_EXCEPTION(index.ubound() >= mv.NumVecs(),
        std::invalid_argument,
        os.str() << "Column index range must not exceed "
        "number of vectors " << mv.NumVecs() << " in "
        "the input multivector.");
      }
    return Teuchos::rcp
      (new HYMLS::BorderedVector(View, mv, index.lbound(), index.size()));
    }

  static Teuchos::RCP<const HYMLS::BorderedVector>
  CloneView(const HYMLS::BorderedVector& mv, const std::vector<int>& index)
    {
    const int inNumVecs  = mv.NumVecs();
    const int outNumVecs = index.size();

    // Simple, inexpensive tests of the index vector.
    TEUCHOS_TEST_FOR_EXCEPTION(outNumVecs == 0, std::invalid_argument,
      "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::"
      "CloneView(mv, index = {}): The output view "
      "must have at least one column.");
    if (outNumVecs > inNumVecs)
      {
      std::ostringstream os;
      os << "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::"
        "CloneView(mv, index = {";
      for (int k = 0; k < outNumVecs - 1; ++k)
        os << index[k] << ", ";
      os << index[outNumVecs-1] << "}): There are " << outNumVecs
         << " indices to view, but only " << inNumVecs << " columns of mv.";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, os.str());
      }

    return Teuchos::rcp (new HYMLS::BorderedVector(View, mv, index));
    }

  static Teuchos::RCP<HYMLS::BorderedVector>
  CloneView(const HYMLS::BorderedVector &mv, const Teuchos::Range1D &index)
    {
    const bool validRange = index.size() > 0 &&
      index.lbound() >= 0 &&
      index.ubound() < mv.NumVecs();
    if (! validRange)
      {
      std::ostringstream os;
      os << "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::CloneView"
        "(mv,index=[" << index.lbound() << ", " << index.ubound()
         << "]): ";
      TEUCHOS_TEST_FOR_EXCEPTION(index.size() == 0, std::invalid_argument,
        os.str() << "Column index range must be nonempty.");
      TEUCHOS_TEST_FOR_EXCEPTION(index.lbound() < 0, std::invalid_argument,
        os.str() << "Column index range must be nonnegative.");
      TEUCHOS_TEST_FOR_EXCEPTION(index.ubound() >= mv.NumVecs(),
        std::invalid_argument,
        os.str() << "Column index range must not exceed "
        "number of vectors " << mv.NumVecs() << " in "
        "the input multivector.");
      }
    return Teuchos::rcp (new HYMLS::BorderedVector(View, mv, index.lbound(), index.size()));
    }
  static int  GetVecLength     (const HYMLS::BorderedVector& mv) { return mv.GlobalLength(); }
  static int  GetNumberVecs    (const HYMLS::BorderedVector& mv) { return mv.NumVecs(); }
  static bool HasConstantStride(const HYMLS::BorderedVector& mv) { return mv.ConstantStride(); }   static ptrdiff_t GetGlobalLength (const HYMLS::BorderedVector& mv)
    {
    if ( mv.First()->Map().GlobalIndicesLongLong() )
      return static_cast<ptrdiff_t>( mv.GlobalLength64() );
    else
      return static_cast<ptrdiff_t>( mv.GlobalLength() );
    }

  // Epetra style (we should compare this with just a bunch of updates)
  static void MvTimesMatAddMv (const double alpha,
    const HYMLS::BorderedVector& A,
    const Teuchos::SerialDenseMatrix<int,double>& B,
    const double beta,
    HYMLS::BorderedVector& mv)
    {
    // Create Epetra_Multivector from SerialDenseMatrix
    Epetra_LocalMap LocalMap((hymls_gidx)B.numRows(), 0, mv.Second()->Map().Comm());
    Epetra_MultiVector B_Pvec(View, LocalMap, B.values(), B.stride(), B.numCols());

    const int info = mv.Multiply ('N', 'N', alpha, A, B_Pvec, beta);

    TEUCHOS_TEST_FOR_EXCEPTION(
      info != 0, EpetraMultiVecFailure,
      "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::MvTimesMatAddMv: "
      "HYMLS::BorderedVector::multiply() returned a nonzero value info=" << info
      << ".");
    }

  static void
  MvAddMv (const double alpha,
    const HYMLS::BorderedVector& A,
    const double beta,
    const HYMLS::BorderedVector& B,
    HYMLS::BorderedVector& mv)
    {
    const int info = mv.Update (alpha, A, beta, B, 0.0);

    TEUCHOS_TEST_FOR_EXCEPTION(info != 0, EpetraMultiVecFailure,
      "Belos::MultiVecTraits<double, HYMLS::BorderedVector>::MvAddMv: Call to "
      "update() returned a nonzero value " << info << ".");
    }

  static void
  MvScale (HYMLS::BorderedVector& mv,
    const double alpha)
    {
    const int info = mv.Scale(alpha);

    TEUCHOS_TEST_FOR_EXCEPTION(info != 0, EpetraMultiVecFailure,
      "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::MvScale: "
      "HYMLS::BorderedVector::scale() returned a nonzero value info="
      << info << ".");
    }

  //! For all columns j of  mv, set mv[j] = alpha[j] * mv[j].
  static void
  MvScale (HYMLS::BorderedVector &mv,
    const std::vector<double> &alpha)
    {
    // Check to make sure the vector has the same number of entries
    // as the multivector has columns.
    const int numvecs = mv.NumVecs();

    TEUCHOS_TEST_FOR_EXCEPTION(
      (int) alpha.size () != numvecs, EpetraMultiVecFailure,
      "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::MvScale: "
      "Array alpha of scaling coefficients has " << alpha.size ()
      << " entries, which is not the same as the number of columns "
      << numvecs << " in the input multivector mv.");

    int info = 0;
    int startIndex = 0;
    for (int i = 0; i < numvecs; ++i)
      {
      HYMLS::BorderedVector temp_vec(::View, mv, startIndex, 1);
      info = temp_vec.Scale(alpha[i]);

      TEUCHOS_TEST_FOR_EXCEPTION(info != 0, EpetraMultiVecFailure,
        "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::MvScale: "
        "On column " << (i+1) << " of " << numvecs << ", Epetra_Multi"
        "Vector::Scale() returned a nonzero value info=" << info << ".");
      startIndex++;
      }
    }

  //! B := alpha * A^T * mv.
  //! Epetra style
  static void MvTransMv(const double alpha, const HYMLS::BorderedVector &A,
    const HYMLS::BorderedVector &mv, Teuchos::SerialDenseMatrix<int,double> &B)
    {
    // Create Epetra_MultiVector from SerialDenseMatrix
    Epetra_LocalMap LocalMap((hymls_gidx)B.numRows(), 0, mv.Second()->Map().Comm());
    Epetra_MultiVector B_Pvec(View, LocalMap, B.values(), B.stride(), B.numCols());
    int info = B_Pvec.Multiply('T', 'N', alpha, *A.First(), *mv.First(), 0.0);
    info    += B_Pvec.Multiply('T', 'N', alpha, *A.Second(), *mv.Second(), 1.0);

    TEUCHOS_TEST_FOR_EXCEPTION(info != 0, EpetraMultiVecFailure,
      "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::MvTransMv: "
      "HYMLS::BorderedVector::multiply() returned a nonzero value info="
      << info << ".");
    }

  //! For all columns j of mv, set b[j] := mv[j]^T * A[j].
  static void
  MvDot (const HYMLS::BorderedVector &mv,
    const HYMLS::BorderedVector &A,
    std::vector<double> &b)
    {
    const int info = mv.Dot(A, b);

    TEUCHOS_TEST_FOR_EXCEPTION(info != 0, EpetraMultiVecFailure,
      "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::MvDot: "
      "HYMLS::BorderedVector::dot() returned a nonzero value info="
      << info << ".");
    }

  //! For all columns j of mv, set normvec[j] = norm(mv[j]).
  static void
  MvNorm (const HYMLS::BorderedVector &mv,
    std::vector<double> &normvec,
    NormType type = TwoNorm)
    {
    if ((int) normvec.size() >= mv.NumVecs())
      {
      int info = 0;
      switch( type )
        {
        case ( OneNorm ) :
          info = mv.Norm1(normvec);
          break;
        case ( TwoNorm ) :
          info = mv.Norm2(normvec);
          break;
        case ( InfNorm ) :
          info = mv.NormInf(normvec);
          break;
        default:
          break;
        }
      TEUCHOS_TEST_FOR_EXCEPTION(info != 0, EpetraMultiVecFailure,
        "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::MvNorm: "
        "HYMLS::BorderedVector::Norm() returned a nonzero value info="
        << info << ".");
      }
    }

  static void
  SetBlock (const HYMLS::BorderedVector &A,
    const std::vector<int> &index,
    HYMLS::BorderedVector &mv)
    {
    const int inNumVecs  = GetNumberVecs(A);
    const int outNumVecs = index.size();

    if (inNumVecs < outNumVecs)
      {
      std::ostringstream os;
      os << "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::"
        "SetBlock(A, mv, index = {";
      if (outNumVecs > 0)
        {
        for (int k = 0; k < outNumVecs - 1; ++k)
          os << index[k] << ", ";
        os << index[outNumVecs-1];
        }
      os << "}): A has only " << inNumVecs << " columns, but there are "
         << outNumVecs << " indices in the index vector.";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, os.str());
      }

    // Make a view of the columns of mv indicated by the index std::vector.
    Teuchos::RCP<HYMLS::BorderedVector> mv_view = CloneViewNonConst(mv, index);

    // View of columns [0, outNumVecs-1] of the source multivector A.
    // If A has fewer columns than mv_view, then create a view of
    // the first outNumVecs columns of A.
    Teuchos::RCP<const HYMLS::BorderedVector> A_view;
    if (outNumVecs == inNumVecs)
      A_view = Teuchos::rcpFromRef(A); // Const, non-owning RCP
    else
      A_view = CloneView(A, Teuchos::Range1D(0, outNumVecs - 1));

    *mv_view = *A_view;
    }

  static void
  SetBlock (const HYMLS::BorderedVector &A,
    const Teuchos::Range1D &index,
    HYMLS::BorderedVector &mv)
    {
    const int numColsA  = A.NumVecs();
    const int numColsMv = mv.NumVecs();

    // 'index' indexes into mv; it's the index set of the target.
    const bool validIndex = index.lbound() >= 0 && index.ubound() < numColsMv;

    // We can't take more columns out of A than A has.
    const bool validSource = index.size() <= numColsA;

    if (! validIndex || ! validSource)
      {
      std::ostringstream os;
      os << "Belos::MultiVecTraits<double, HYMLS::BorderedVector>::SetBlock"
        "(A, index=[" << index.lbound() << ", " << index.ubound() << "], "
        "mv): ";
      TEUCHOS_TEST_FOR_EXCEPTION(index.lbound() < 0, std::invalid_argument,
        os.str() << "Range lower bound must be nonnegative.");
      TEUCHOS_TEST_FOR_EXCEPTION(index.ubound() >= numColsMv, std::invalid_argument,
        os.str() << "Range upper bound must be less than "
        "the number of columns " << numColsA << " in the "
        "'mv' output argument.");
      TEUCHOS_TEST_FOR_EXCEPTION(index.size() > numColsA, std::invalid_argument,
        os.str() << "Range must have no more elements than"
        " the number of columns " << numColsA << " in the "
        "'A' input argument.");
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Should never get here!");
      }

    // View of columns [index.lbound(), index.ubound()] of the
    // target multivector mv.  We avoid view creation overhead by
    // only creating a view if the index range is different than [0,
    // (# columns in mv) - 1].
    Teuchos::RCP<HYMLS::BorderedVector> mv_view;
    if (index.lbound() == 0 && index.ubound()+1 == numColsMv)
      mv_view = Teuchos::rcpFromRef(mv); // Non-const, non-owning RCP
    else
      mv_view = CloneViewNonConst(mv, index);

    // View of columns [0, index.size()-1] of the source multivector
    // A.  If A has fewer columns than mv_view, then create a view
    // of the first index.size() columns of A.
    Teuchos::RCP<const HYMLS::BorderedVector> A_view;
    if (index.size() == numColsA)
      A_view = Teuchos::rcpFromRef(A); // Const, non-owning RCP
    else
      A_view = CloneView(A, Teuchos::Range1D(0, index.size()-1));

    *mv_view = *A_view;
    }

  static void
  Assign (const HYMLS::BorderedVector& A,
    HYMLS::BorderedVector& mv)
    {
    const int numColsA  = GetNumberVecs(A);
    const int numColsMv = GetNumberVecs(mv);
    if (numColsA > numColsMv)
      {
      std::ostringstream os;
      os << "Belos::MultiVecTraits<double, HYMLS::BorderedVector>::Assign"
        "(A, mv): ";
      TEUCHOS_TEST_FOR_EXCEPTION(numColsA > numColsMv, std::invalid_argument,
        os.str() << "Input multivector 'A' has "
        << numColsA << " columns, but output multivector "
        "'mv' has only " << numColsMv << " columns.");
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "Should never get here!");
      }

    // View of the first [0, numColsA-1] columns of mv.
    Teuchos::RCP<HYMLS::BorderedVector> mv_view;
    if (numColsMv == numColsA)
      mv_view = Teuchos::rcpFromRef(mv); // Non-const, non-owning RCP
    else // numColsMv > numColsA
      mv_view = CloneView(mv, Teuchos::Range1D(0, numColsA - 1));

    *mv_view = A;
    }

  static void MvRandom (HYMLS::BorderedVector& mv)
    {
    TEUCHOS_TEST_FOR_EXCEPTION( mv.Random()!= 0, EpetraMultiVecFailure,
      "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::"
      "MvRandom() call to random() returned a nonzero value.");
    }

  static void MvInit (HYMLS::BorderedVector& mv,
    double alpha = Teuchos::ScalarTraits<double>::zero())
    {
    TEUCHOS_TEST_FOR_EXCEPTION( mv.PutScalar(alpha) != 0, EpetraMultiVecFailure,
      "Belos::MultiVecTraits<double,HYMLS::BorderedVector>::"
      "MvInit() call to putScalar() returned a nonzero value.");
    }

  static void MvPrint (const HYMLS::BorderedVector& mv, std::ostream& os)
    {
    os << mv << std::endl;
    }

  }; // end of specialization

#if TRILINOS_MAJOR_VERSION<12
template<>
class MultiVecTraitsExt<double, HYMLS::BorderedVector>
  {
public:
  //! @name New attribute methods
  //@{

  //! Obtain the vector length of \c mv.
  //! \note This method supersedes GetVecLength, which will be deprecated.
  static ptrdiff_t GetGlobalLength( const HYMLS::BorderedVector& mv )
    {
    if (mv.First()->Map().GlobalIndicesLongLong())
      return static_cast<ptrdiff_t>( mv.GlobalLength64() );
    else
      return static_cast<ptrdiff_t>( mv.GlobalLength() );
    }
  };
#endif


  } // end of Belos namespace
#endif
