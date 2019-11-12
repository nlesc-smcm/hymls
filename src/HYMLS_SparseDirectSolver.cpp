#include "Trilinos_version.h"
#include "HYMLS_SparseDirectSolver.hpp"

#include "HYMLS_config.h"

#ifndef USE_AMESOS

#include "Ifpack_Condest.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"
#include "Epetra_Map.h"
#include "Epetra_Comm.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_RowMatrix.h"
#include "HYMLS_Epetra_Time.h"
#include "Teuchos_ParameterList.hpp"

#include "HYMLS_Tools.hpp"
#include "HYMLS_MatrixUtils.hpp"

#include "Teuchos_StrUtils.hpp"
#include <cstdarg>

#include <fstream>

extern "C" {
#ifdef HAVE_PARDISO
#include "mkl_pardiso.h"
#define iparam(x) pardiso_iparam_[x-1]
#endif

#ifdef HAVE_SUITESPARSE
#include "umfpack.h"
#include "klu.h"
#define T_KLU(xxx) xxx
#define DO_KLU(function) klu_ ## function
#elif TRILINOS_MAJOR_MINOR_VERSION>=121200
#include "trilinos_klu_decl.h"
#define T_KLU(xxx) trilinos_ ## xxx
#define DO_KLU(function) trilinos_klu_ ## function
#else
#include "amesos_klu_decl.h"
#define TRILINOS_KLU_SINGULAR KLU_SINGULAR
#define T_KLU(xxx) xxx
#define DO_KLU(function) amesos_klu_ ## function
#endif

  class KluWrapper
    {
  public:

    T_KLU(klu_symbolic) *Symbolic_;
    T_KLU(klu_numeric) *Numeric_;
    T_KLU(klu_common) *Common_;
    };

  static std::ostream* output_stream;
  static int firstTime=true;

  int my_printf(const char* fmt, ...)
    {
    std::string fmt_string(fmt);

    char formatted_string[fmt_string.length()+10000];
    va_list argptr;
    va_start(argptr,fmt);
    vsprintf(formatted_string,fmt,argptr);
    va_end(argptr);

    *output_stream << formatted_string;
    return 0;
    }
  }

namespace HYMLS {

//==============================================================================
SparseDirectSolver::SparseDirectSolver(Epetra_RowMatrix* Matrix_in) :
  Matrix_(Teuchos::rcp( Matrix_in, false )),
  method_(KLU),
  label_("SparseDirectSolver"),
  IsEmpty_(false),
  IsInitialized_(false),
  IsComputed_(false),
  UseTranspose_(false),
  Condest_(-1.0),
  serialMatrix_(Teuchos::null),
  serialImport_(Teuchos::null),
  ownOrdering_(false), ownScaling_(false),
  pardiso_initialized_(false)
  {
  HYMLS_PROF3(label_,"Constructor");

  output_stream = &Tools::out();
#ifdef HAVE_SUITESPARSE
  amd_printf = &my_printf;
#endif
  MyPID_=Matrix_->Comm().MyPID();

  klu_=new KluWrapper();
  umf_Symbolic_=NULL;
  umf_Numeric_=NULL;
  klu_->Common_=NULL;
  scaLeft_=Teuchos::rcp(new Epetra_Vector(Matrix_->RowMatrixRowMap()));
  scaRight_=Teuchos::rcp(new Epetra_Vector(Matrix_->RowMatrixRowMap()));
  CHECK_ZERO(scaLeft_->PutScalar(1.0));
  CHECK_ZERO(scaRight_->PutScalar(1.0));

// assume that we just have a serial solver built in here
  int N = Matrix_->NumGlobalRows();

  row_perm_.resize(N);
  col_perm_.resize(N);

  for (int i=0;i<N;i++)
    {
    row_perm_[i]=i;
    col_perm_[i]=i;
    }

  }

//==============================================================================
SparseDirectSolver::SparseDirectSolver(const SparseDirectSolver& rhs) :
  Matrix_(Teuchos::rcp( &rhs.Matrix(), false )),
  label_(rhs.Label()),
  IsEmpty_(false),
  IsInitialized_(false),
  IsComputed_(false),
  Condest_(rhs.Condest()),
  pardiso_initialized_(false)
  {
  Tools::Error("not implemented!",__FILE__,__LINE__);
  }

SparseDirectSolver::~SparseDirectSolver()
  {
  HYMLS_PROF3(label_,"Destructor");

  if (method_==KLU)
    {
    if (klu_->Symbolic_)
      {
      DO_KLU(free_symbolic)(&klu_->Symbolic_,klu_->Common_);
      }
    if (klu_->Numeric_)
      {
      DO_KLU(free_numeric)(&klu_->Numeric_,klu_->Common_);
      }
    if (klu_->Common_)
      {
      delete klu_->Common_;
      }
    }
#ifdef HAVE_SUITESPARSE
  else if (method_==UMFPACK)
    {
    if (umf_Symbolic_)
      {
      umfpack_di_free_symbolic (&umf_Symbolic_) ;
      }
    if (umf_Numeric_)
      {
      umfpack_di_free_numeric (&umf_Numeric_) ;
      }
    umf_Info_.resize(0);
    umf_Control_.resize(0);
    }
#endif
#ifdef HAVE_PARDISO
  else if (method_==PARDISO)
    {
    if (pardiso_initialized_)
      {
      int N = serialMatrix_->NumGlobalRows();
      int NumVectors = 1;
      int maxfct = 1; // Max number of factors in memory
      int mnum = 1; // Maxtrix number
      int phase = -1; // Release internal memory
      int msglvl = 0; // No output
      int error = 0;
      double ddum; // Dummy variable

      pardiso(pardiso_pt_, &maxfct, &mnum, &pardiso_mtype_, &phase,
        &N, &Aval_[0], &Ap_[0], &Ai_[0], &pardiso_perm_[0], &NumVectors,
        pardiso_iparam_, &msglvl, &ddum, &ddum, &error);
      }
    }
#endif
  else
    {
    Tools::Warning("destructor not implemented",__FILE__,__LINE__);
    }
  delete klu_;
  }

//==============================================================================
int SparseDirectSolver::SetParameters(Teuchos::ParameterList& params)
  {
  HYMLS_PROF3(label_,"SetParameters");
  std::string choice = params.get("amesos: solver type", "KLU");
  choice = Teuchos::StrUtils::allCaps(choice);
  //~ std::cerr << "choice: " << choice << std::endl;
  method_=KLU; // default - always available.
  std::string label2="KLU";
#ifdef HAVE_SUITESPARSE
  if (choice=="UMFPACK"||choice=="AMESOS_UMFPACK")
    {
    method_=UMFPACK;
    label2="Umfpack";
    }
  else
#endif
#ifdef HAVE_PARDISO
    if (choice=="PARDISO"||choice=="MKL_PARDISO"||choice=="MKL PARDISO")
      {
      method_=PARDISO;
      label2="MKL ParDiSo";
      }
    else
#endif
      if ((choice!="AMESOS_KLU" && choice!="KLU"))
        {
        if (firstTime)
          {
          firstTime=false;
          Tools::Warning("Invalid choice of 'amesos: solver type'. KLU is used as Sparse Solver",__FILE__,__LINE__);
          }
        }

  label_=params.get("Label",label_);
  label_=label_+" ("+label2+")";

  if (method_==KLU)
    {
    klu_->Common_ = new T_KLU(klu_common)();
    DO_KLU(defaults)(klu_->Common_);
    }
#ifdef HAVE_SUITESPARSE
  else if (method_==UMFPACK)
    {
    int prl = params.get("OutputLevel",0);
    umf_Info_.resize(UMFPACK_INFO);
    umf_Control_.resize(UMFPACK_CONTROL);
    umfpack_di_defaults( &umf_Control_[0] ) ;
    umf_Control_[UMFPACK_PRL]=prl;
    }
#endif
#ifdef HAVE_PARDISO
  if (method_==PARDISO)
    {
    // Init pt_ for the first call to PARDISO
    for (int i = 0; i < 64; i++)
      {
      pardiso_pt_[i] = 0;
      }
    // Real unsymmetric by default here
    pardiso_mtype_ = 11;
    pardisoinit(pardiso_pt_, &pardiso_mtype_, pardiso_iparam_);
    iparam(35) = 1; // zero based indexing
    }
#endif

  ownOrdering_=params.get("Custom Ordering",false);
  ownScaling_=params.get("Custom Scaling",false);

  if (ownOrdering_)
    {
//  double pivtol=100*HYMLS_SMALL_ENTRY;
    double pivtol=0.0;
    // cf. (REMARK *) below
    if (method_==KLU)
      {
      /* parameters */
      klu_->Common_->tol = pivtol; /* pivot tolerance for diagonal */
      klu_->Common_->btf = 1;        /* use BTF pre-ordering, or not */
      klu_->Common_->ordering = 2;      // 0: AMD, 1: COLAMD, 2: user-provided P and Q,
      // 3: user-provided function
      klu_->Common_->halt_if_singular = 0;   /* quick halt if matrix is singular */
      }
#ifdef HAVE_SUITESPARSE
    else if (method_==UMFPACK)
      {
      umf_Control_[UMFPACK_ORDERING] = UMFPACK_ORDERING_NONE;
      umf_Control_[UMFPACK_FIXQ] = 1;
      umf_Control_[UMFPACK_SYM_PIVOT_TOLERANCE] = pivtol;
      umf_Control_[UMFPACK_PIVOT_TOLERANCE] = pivtol;
      }
#endif
#ifdef HAVE_PARDISO
    else if (method_==PARDISO)
      {
      iparam(5) = 1; // user permutation
      iparam(10) = 16; // pivot tolerance of 10^(-16)
      }
#endif
    }

  if (ownScaling_)
    {
    if (method_==KLU)
      {
      klu_->Common_->scale = 0 ;    // scale: -1: none, and do not check for errors
      // in the input matrix in KLU_refactor.
      // 0: none, but check for errors,
      // 1: sum, 2: max
      }
#ifdef HAVE_SUITESPARSE
    else if (method_==UMFPACK)
      {
      umf_Control_[UMFPACK_SCALE] = UMFPACK_SCALE_NONE;
      }
#endif
#ifdef HAVE_PARDISO
    else if (method_==PARDISO)
      {
      iparam(11) = 0; // scaling vectors off
      }
#endif
    }

  return(0);
  }

//==============================================================================
int SparseDirectSolver::Initialize()
  {
  HYMLS_PROF3(label_,"Initialize");
  IsEmpty_ = false;
  IsInitialized_ = false;
  IsComputed_ = false;

  if (Matrix_ == Teuchos::null)
    {
    Tools::Error("null matrix",__FILE__,__LINE__);
    }

  // only square matrices
  if (Matrix_->NumGlobalRows() != Matrix_->NumGlobalCols())
    {
    Tools::Error("non-square matrix",__FILE__,__LINE__);
    }

  // if the matrix has a dimension of 0, this is an empty preconditioning object.
  if (Matrix_->NumGlobalRows() == 0) {
    IsEmpty_ = true;
    IsInitialized_ = true;
    return(0);
    }

  // create umfpack
  CHECK_ZERO(this->ConvertToSerial());
  if (ownOrdering_)
    {
    CHECK_ZERO(this->FillReducingOrdering());
    }
  CHECK_ZERO(this->ConvertToCRS());
  if (method_==KLU)
    {
    CHECK_ZERO(this->KluSymbolic());
    }
#ifdef HAVE_SUITESPARSE
  else if (method_==UMFPACK)
    {
    CHECK_ZERO(this->UmfpackSymbolic());
    }
#endif
#ifdef HAVE_PARDISO
  else if (method_==PARDISO)
    {
    CHECK_ZERO(this->PardisoSymbolic());
    }
#endif
  else
    {
    // not implemented
    return -99;
    }
  IsInitialized_ = true;
  return(0);
  }

//==============================================================================
int SparseDirectSolver::Compute()
  {
  HYMLS_PROF3(label_,"Compute");
  if (!IsInitialized())
    CHECK_ZERO(Initialize());

  if (IsEmpty_) {
    IsComputed_ = true;
    return(0);
    }

  IsComputed_ = false;

  if (Matrix_ == Teuchos::null)
    {
    Tools::Error("null matrix",__FILE__,__LINE__);
    }
  if (serialMatrix_.get()!=Matrix_.get())
    {
    if (serialImport_==Teuchos::null)
      {
      Tools::Error("importer is null",__FILE__,__LINE__);
      }
    Teuchos::RCP<Epetra_CrsMatrix> serialCrsMatrix =
      Teuchos::rcp_const_cast<Epetra_CrsMatrix>(
        Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(serialMatrix_));
    if (Teuchos::is_null(serialCrsMatrix))
      {
      Tools::Error("dynamic cast failed - need a CrsMatrix",__FILE__,__LINE__);
      }
    CHECK_ZERO(serialCrsMatrix->Import(*Matrix_,*serialImport_,Insert));
    }
  if (ownScaling_)
    {
    CHECK_ZERO(ComputeScaling());
    }
  CHECK_ZERO(this->ConvertToCRS());
  if (method_==KLU)
    {
    CHECK_ZERO(this->KluNumeric());
    }
#ifdef HAVE_SUITESPARSE
  else if (method_==UMFPACK)
    {
    CHECK_ZERO(this->UmfpackNumeric());
    }
#endif
#ifdef HAVE_PARDISO
  else if (method_==PARDISO)
    {
    CHECK_ZERO(this->PardisoNumeric());
    }
#endif
  else
    {
    return -99; // not implemented
    }

  IsComputed_ = true;
  return(0);
  }

//==============================================================================
int SparseDirectSolver::SetUseTranspose(bool UseTranspose_in)
  {
  UseTranspose_ = UseTranspose_in;
  return(0);
  }

//==============================================================================
int SparseDirectSolver::
Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
  {
  // check for maps? check UseTranspose_?
  return(-99); // not implemented
  CHECK_ZERO(Matrix_->Apply(X,Y));
  return 0;
  }

//==============================================================================
int SparseDirectSolver::
ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
  {
  if (IsEmpty_) {
    return(0);
    }

  if (IsComputed() == false)
    {return -1;}

  if (X.NumVectors() != Y.NumVectors())
    {return -2;}

  // AztecOO gives X and Y pointing to the same memory location,
  // need to create an auxiliary vector, Xcopy
  Teuchos::RCP<const Epetra_MultiVector> Xcopy;
  if (X.Pointers()[0] == Y.Pointers()[0])
    {
    Xcopy = Teuchos::rcp( new Epetra_MultiVector(X) );
    }
  else
    {
    Xcopy = Teuchos::rcp( &X, false );
    }

  if (method_==KLU)
    {
    CHECK_ZERO(this->KluSolve(*Xcopy,Y));
    }
#ifdef HAVE_SUITESPARSE
  else if (method_==UMFPACK)
    {
    CHECK_ZERO(this->UmfpackSolve(*Xcopy,Y));
    }
#endif
#ifdef HAVE_PARDISO
  else if (method_==PARDISO)
    {
    CHECK_ZERO(this->PardisoSolve(*Xcopy,Y));
    }
#endif
  else
    {
    return -99; // not implemented
    }

#ifdef HYMLS_TESTING
  Epetra_MultiVector R(X);
  CHECK_ZERO(Matrix_->Multiply(UseTranspose_,Y,R));
  CHECK_ZERO(R.Update(-1.0,X,1.0));
  double *rnorm2 = new double[X.NumVectors()];
  double *bnorm2 = new double[X.NumVectors()];
  CHECK_ZERO(R.Norm2(rnorm2));
  CHECK_ZERO(X.Norm2(bnorm2));

  double rcond = Condest();

  bool bad_res=false;
  for (int i=0;i<X.NumVectors();i++)
    {
    if (rnorm2[i]/bnorm2[i] > 1.0e-12/rcond)
      {
      bad_res=true;
      Tools::Warning("bad residual found: "+Teuchos::toString(rnorm2[i])+"\n"
        "            norm of rhs, ||b||="+Teuchos::toString(bnorm2[i])+"\n"
        "            condition estimate of matrix: "+Teuchos::toString(1.0/rcond),
        __FILE__,__LINE__);
      }
    }
  if (bad_res)
    {
    this->DumpSolverStatus("umfBadRes",false,Teuchos::rcp(&Y,false),Xcopy);
    }

  delete [] rnorm2;
  delete [] bnorm2;
#endif

  return(0);
  }

//==============================================================================
double SparseDirectSolver::NormInf() const
  {
  return(-1.0);
  }

//==============================================================================
const char* SparseDirectSolver::Label() const
  {
  return((char*)label_.c_str());
  }

//==============================================================================
bool SparseDirectSolver::UseTranspose() const
  {
  return(UseTranspose_);
  }

//==============================================================================
bool SparseDirectSolver::HasNormInf() const
  {
  return(false);
  }

//==============================================================================
const Epetra_Comm & SparseDirectSolver::Comm() const
  {
  return(Matrix_->Comm());
  }

//==============================================================================
const Epetra_Map & SparseDirectSolver::OperatorDomainMap() const
  {
  return(Matrix_->OperatorDomainMap());
  }

//==============================================================================
const Epetra_Map & SparseDirectSolver::OperatorRangeMap() const
  {
  return(Matrix_->OperatorRangeMap());
  }

//==============================================================================
double SparseDirectSolver::Condest(const Ifpack_CondestType CT,
  const int MaxIters, const double Tol,
  Epetra_RowMatrix* Matrix_in)
  {

  if (!IsComputed()) // cannot compute right now
    return(-1.0);

  if (Condest_ == -1.0)
    Condest_ = Ifpack_Condest(*this, CT, MaxIters, Tol, Matrix_in);

  return(Condest_);
  }

//==============================================================================
std::ostream& SparseDirectSolver::Print(std::ostream& os) const
  {
#if 0
  if (!Comm().MyPID()) {
    os << endl;
    os << "================================================================================" << endl;
    os << "SparseDirectSolver: " << Label () << endl << endl;
    os << "Condition number estimate = " << Condest() << endl;
    os << "Global number of rows            = " << Matrix_->NumGlobalRows() << endl;
    os << endl;
    os << "Phase           # calls   Total Time (s)       Total MFlops     MFlops/s" << endl;
    os << "-----           -------   --------------       ------------     --------" << endl;
    os << "Initialize()    "   << std::setw(5) << NumInitialize_
       << "  " << std::setw(15) << InitializeTime_
       << "              0.0              0.0" << endl;
    os << "Compute()       "   << std::setw(5) << NumCompute_
       << "  " << std::setw(15) << ComputeTime_
       << "  " << std::setw(15) << 1.0e-6 * ComputeFlops_;
    if (ComputeTime_ != 0.0)
      os << "  " << std::setw(15) << 1.0e-6 * ComputeFlops_ / ComputeTime_ << endl;
    else
      os << "  " << std::setw(15) << 0.0 << endl;
    os << "ApplyInverse()  "   << std::setw(5) << NumApplyInverse_
       << "  " << std::setw(15) << ApplyInverseTime_
       << "  " << std::setw(15) << 1.0e-6 * ApplyInverseFlops_;
    if (ApplyInverseTime_ != 0.0)
      os << "  " << std::setw(15) << 1.0e-6 * ApplyInverseFlops_ / ApplyInverseTime_ << endl;
    else
      os << "  " << std::setw(15) << 0.0 << endl;
    os << "================================================================================" << endl;
    os << endl;
    }
#endif
  return(os);
  }

// private member functions

int SparseDirectSolver::ConvertToSerial()
  {
  if (Matrix_->Comm().NumProc()==1)
    {
    serialMatrix_=Matrix_;
    }
  else
    {
    Tools::Error("not implemented",__FILE__,__LINE__);
    scaLeft_=Teuchos::rcp(new Epetra_Vector(serialMatrix_->RowMatrixRowMap()));
    scaRight_=Teuchos::rcp(new Epetra_Vector(serialMatrix_->RowMatrixRowMap()));
    CHECK_ZERO(scaLeft_->PutScalar(1.0));
    CHECK_ZERO(scaRight_->PutScalar(1.0));
    }
  return 0;
  }

int SparseDirectSolver::ComputeScaling()
  {
  if (serialMatrix_!=Teuchos::null)
    {
    // find max abs diagonal value
    CHECK_ZERO(serialMatrix_->ExtractDiagonalCopy(*scaRight_));
    CHECK_ZERO(scaRight_->Abs(*scaRight_));
    double dmax;
    CHECK_ZERO(scaRight_->MaxValue(&dmax));
    // this scaling is intended for F-matrices - scale the grad and div
    // parts by a constant to make their entries the same size as the maximum
    // diagonal entry in A.
    for (int i=0;i<scaRight_->MyLength();i++)
      {
      if ((*scaRight_)[i]<=HYMLS_SMALL_ENTRY*dmax)
        {
        (*scaLeft_)[i] = dmax;
        (*scaRight_)[i] = dmax;
        }
      else
        {
        (*scaLeft_)[i]=1.0;
//        (*scaLeft_)[i]=1.0/sqrt((*scaRight_)[i]);
//        (*scaRight_)[i]=(*scaLeft_)[i];
        }
      }
    }
  else
    {
    Tools::Error("not implemented",__FILE__,__LINE__);
    }
  return 0;
  }

// compute fill-reducing ordering before converting to UMFPACK
int SparseDirectSolver::FillReducingOrdering()
  {
  if (IsEmpty_||(MyPID_ != 0)) return 0;
  if (serialMatrix_==Teuchos::null) return -1;

  Teuchos::RCP<const Epetra_CrsMatrix> serialCrsMatrix
    = Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(serialMatrix_);
  if (Teuchos::is_null(serialCrsMatrix))
    {
    Tools::Error("need a CrsMatrix here",__FILE__,__LINE__);
    }
  CHECK_ZERO(HYMLS::MatrixUtils::FillReducingOrdering(*serialCrsMatrix,row_perm_,col_perm_));

  return 0;
  }

//=============================================================================
int SparseDirectSolver::ConvertToCRS()
  {
  HYMLS_PROF3(label_,"ConvertToCRS");
  // Convert matrix to the form that Umfpack expects (Ap, Ai, Aval),
  // only on processor 0. The matrix has already been assembled in
  // serialMatrix_; if only one processor is used, then serialMatrix_
  // points to the problem's matrix.

  // Umfpack expects compressed column storage, but we pass in CRS and
  // then solve tht transposed problem in UmfpackSolve().

  // we do the reordering step here already

  if (MyPID_ == 0)
    {
    int N = serialMatrix_->NumMyRows();
    int nnz= serialMatrix_->NumMyNonzeros();
    Ap_.resize(N+1);
    Ai_.resize(nnz);
    Aval_.resize(nnz);

    int NumEntries = serialMatrix_->MaxNumEntries();
    Teuchos::Array<int> invperm(N);
    for (int i=0;i<N;i++) invperm[col_perm_[i]]=i;
    int len;
    int Ai_index = 0;
    for (int i = 0 ; i < N; i++)
      {
      // get row p[i] from the matrix -> row i in the Umfpack matrix
      int MyRow = row_perm_[i];
      Ap_[i] = Ai_index ;
      CHECK_ZERO(serialMatrix_->ExtractMyRowCopy(MyRow, NumEntries,
          len, &Aval_[Ai_index], &Ai_[Ai_index]));
      for (int j=0;j<len;j++)
        {
        Aval_[Ai_index+j] *= (*scaLeft_)[MyRow] * (*scaRight_)[Ai_[Ai_index+j]];
        Ai_[Ai_index+j] = invperm[Ai_[Ai_index+j]];
        }
      Ai_index += len;
      // sort row entries by column index
      MatrixUtils::SortMatrixRow(&Ai_[Ai_index-len],&Aval_[Ai_index-len],len);
      }
    Ap_[N] = Ai_index;
    }
  return 0;
  }

//////////////////////////////////////////////////////////////////////
// KLU INTERFACE                                                    //
//////////////////////////////////////////////////////////////////////

int SparseDirectSolver::KluSymbolic()
  {
  if (MyPID_!=0) return 0;
  HYMLS_PROF3(label_,"KluSymbolic");

  int N = serialMatrix_->NumGlobalRows();

  if (klu_->Symbolic_)
    DO_KLU(free_symbolic)(&klu_->Symbolic_,klu_->Common_);
  if (ownOrdering_)
    {
    klu_->Symbolic_=DO_KLU(analyze_given)(N, &Ap_[0], &Ai_[0], NULL, NULL, klu_->Common_);
    }
  else
    {
    klu_->Symbolic_=DO_KLU(analyze)(N, &Ap_[0], &Ai_[0], klu_->Common_);
    }
  int status = klu_->Common_->status;

  if (status || (klu_->Symbolic_==NULL))
    {
    HYMLS::Tools::Error("KLU Symbolic Error "+Teuchos::toString(status),__FILE__,__LINE__);
    }

  return status;
  }

//=============================================================================

int SparseDirectSolver::KluNumeric()
  {
  HYMLS_PROF3(label_,"KluNumeric");
  if (MyPID_!=0) return 0;

  if (klu_->Numeric_) DO_KLU(free_numeric)(&klu_->Numeric_,klu_->Common_);

  klu_->Numeric_=DO_KLU(factor)(&Ap_[0], &Ai_[0], &Aval_[0],
    klu_->Symbolic_, klu_->Common_);

  int status = klu_->Common_->status;

  if (status ||(klu_->Numeric_==NULL))
    {
    if (status==TRILINOS_KLU_SINGULAR) this->DumpSolverStatus("kluSingular",false);
    HYMLS::Tools::Error("KLU Numeric Error "+Teuchos::toString(status),__FILE__,__LINE__);
    }
  DO_KLU(rcond)(klu_->Symbolic_,klu_->Numeric_,klu_->Common_);
  Condest_ = klu_->Common_->rcond;
  return status;
  }

//=============================================================================

int SparseDirectSolver::KluSolve(const Epetra_MultiVector& B, Epetra_MultiVector& X) const
  {
  HYMLS_PROF3(label_,"KluSolve");

  if (Matrix_.get()!=serialMatrix_.get()) return -99; // not implemented

  int N = Matrix_->NumMyRows();
  int NumVectors = X.NumVectors();

  Teuchos::RCP<Epetra_MultiVector> serialX = Teuchos::rcp(&X,false);
  Teuchos::RCP<const Epetra_MultiVector> serialB = Teuchos::rcp(&B,false);

  double *xbuf = new double[NumVectors * N];

  const Teuchos::RCP<Epetra_Vector>& sca_l =
    UseTranspose_? scaRight_: scaLeft_;
  const Teuchos::RCP<Epetra_Vector>& sca_r =
    UseTranspose_? scaLeft_: scaRight_;
  const Teuchos::Array<int>& row_perm =
    UseTranspose_? col_perm_: row_perm_;
  const Teuchos::Array<int>& col_perm =
    UseTranspose_? row_perm_: col_perm_;

  int status=0;
  if ( MyPID_ == 0 )
    {
    // Get direct pointers to the arrays, which speeds up the code somewhat
    const double *sca_l_ptr = sca_l->Values();
    const int *row_perm_ptr = row_perm.getRawPtr();
    const double *sca_r_ptr = sca_r->Values();
    const int *col_perm_ptr = col_perm.getRawPtr();

    for (int j = 0 ; j < NumVectors; j++)
      {
      double *xbuf_ptr = xbuf + j * N;
      const double *serialB_ptr = (*serialB)[j];
      for (int i = 0; i < N; i++)
        {
        xbuf_ptr[i] = serialB_ptr[row_perm_ptr[i]] * sca_l_ptr[row_perm_ptr[i]];
        }
      }

    if (UseTranspose() == false)
      {
      DO_KLU(tsolve)(klu_->Symbolic_, klu_->Numeric_, N, NumVectors, xbuf, klu_->Common_);
      }
    else
      {
      DO_KLU(solve)(klu_->Symbolic_, klu_->Numeric_, N, NumVectors, xbuf, klu_->Common_);
      }

    // we now have x(col_perm) in x_buf
    for (int j = 0; j < NumVectors; j++)
      {
      double *serialX_ptr = (*serialX)[j];
      double *xbuf_ptr = xbuf + j * N;
      for (int i = 0; i < N; i++)
        {
        serialX_ptr[col_perm_ptr[i]] = xbuf_ptr[i] * sca_r_ptr[col_perm_ptr[i]];
        }
      }
    status = klu_->Common_->status;
    }

  delete[] xbuf;

  if (serialX.get()!=&X) return -99; //not implemented
  return status;
  }

//////////////////////////////////////////////////////////////////////
// END KLU INTERFACE                                                //
//////////////////////////////////////////////////////////////////////

#ifdef HAVE_SUITESPARSE

//////////////////////////////////////////////////////////////////////
// UMFPACK INTERFACE                                                //
//////////////////////////////////////////////////////////////////////

int SparseDirectSolver::UmfpackSymbolic()
  {
  if (MyPID_!=0) return 0;
  HYMLS_PROF3(label_,"UmfpackSymbolic");

  int N = serialMatrix_->NumGlobalRows();

  if (umf_Symbolic_)
    umfpack_di_free_symbolic (&umf_Symbolic_) ;

  int status = umfpack_di_symbolic (N, N, &Ap_[0],
    &Ai_[0], &Aval_[0],
    &umf_Symbolic_, &umf_Control_[0], &umf_Info_[0]);
  if (status)
    {
#ifdef HYMLS_TESTING
    if (N<=256)
      {
      std::cout << "umfpack matrix: "<<std::endl;
      umfpack_di_report_matrix(N,N,
        &Ap_[0],&Ai_[0],&Aval_[0],0,&umf_Control_[0]);
      }
#endif
    umfpack_di_report_info(&umf_Control_[0], &umf_Info_[0]);
    umfpack_di_report_status(&umf_Control_[0], status);
    HYMLS::Tools::Error("UMFPACK Symbolic Error",__FILE__,__LINE__);
    }

  return 0;
  }

//=============================================================================

int SparseDirectSolver::UmfpackNumeric()
  {
  HYMLS_PROF3(label_,"UmfpackNumeric");
  if (MyPID_!=0) return 0;

  if (umf_Numeric_) umfpack_di_free_numeric (&umf_Numeric_) ;

  int status=umfpack_di_numeric (&Ap_[0], &Ai_[0], &Aval_[0],
    umf_Symbolic_, &umf_Numeric_,
    &umf_Control_[0], &umf_Info_[0]);
  if (status)
    {
    umfpack_di_report_info(&umf_Control_[0], &umf_Info_[0]);
    umfpack_di_report_status(&umf_Control_[0], status);
    HYMLS::Tools::Error("UMFPACK Numeric Error",__FILE__,__LINE__);
    }
  Condest_=umf_Info_[UMFPACK_RCOND];
  double rcond = Condest_;
#ifdef HYMLS_TESTING
  if (rcond>0.0)
    {
    HYMLS_DEBVAR(rcond);
    if (rcond<1.0e-14)
      {
      Tools::Warning("singular matrix encountered!\n"
        "(RCOND="+Teuchos::toString(rcond)+")",
        __FILE__,__LINE__);
      this->DumpSolverStatus("umfSingular",false);
      }
    else if (rcond<1.0e-8)
      {
      Tools::Warning("ill-conditioned matrix encountered!\n"
        "(RCOND="+Teuchos::toString(rcond)+")",
        __FILE__,__LINE__);
      this->DumpSolverStatus("umfIllCond",false);
      }

    }
#endif
  return 0;
  }

//=============================================================================

int SparseDirectSolver::UmfpackSolve(const Epetra_MultiVector& B, Epetra_MultiVector& X) const
  {
  HYMLS_PROF3(label_,"UmfpackSolve");

  if (Matrix_.get()!=serialMatrix_.get()) return -99; // not implemented

  Teuchos::RCP<Epetra_MultiVector> serialX = Teuchos::rcp(&X,false);
  Teuchos::RCP<const Epetra_MultiVector> serialB = Teuchos::rcp(&B,false);

  int N = serialX->MyLength();
  int NumVectors = X.NumVectors();

  double *xbuf = new double[N];
  double *bbuf = new double[N];

  int UmfpackRequest = UseTranspose()?UMFPACK_A:UMFPACK_At;

  int status = 0;
  if ( MyPID_ == 0 )
    {
    // Get direct pointers to the arrays, which speeds up the code somewhat
    const double *sca_l_ptr = scaLeft_->Values();
    const int *row_perm_ptr = row_perm_.getRawPtr();
    const double *sca_r_ptr = scaRight_->Values();
    const int *col_perm_ptr = col_perm_.getRawPtr();

    for (int j = 0 ; j < NumVectors; j++)
      {
      const double *serialB_ptr = (*serialB)[j];
      double *serialX_ptr = (*serialX)[j];
      for (int i = 0; i < N; i++)
        {
        bbuf[i] = serialB_ptr[row_perm_ptr[i]] * sca_l_ptr[row_perm_ptr[i]];
        }

      status = umfpack_di_solve (UmfpackRequest, &Ap_[0],
        &Ai_[0], &Aval_[0],
        xbuf, bbuf,
        const_cast<void*>(umf_Numeric_),
        const_cast<double*>(&umf_Control_[0]),
        const_cast<double*>(&umf_Info_[0]));
      // we now have x(col_perm) in x_buf
      for (int i = 0; i < N; i++)
        {
        serialX_ptr[col_perm_ptr[i]] = xbuf[i] * sca_r_ptr[col_perm_ptr[i]];
        }
      }
    }

  delete[] xbuf;
  delete[] bbuf;

  if (serialX.get()!=&X) return -99; //not implemented
  return 0;
  }

//////////////////////////////////////////////////////////////////////
// END UMFPACK INTERFACE                                            //
//////////////////////////////////////////////////////////////////////

#endif // HAVE_SUITESPARSE

#ifdef HAVE_PARDISO

//////////////////////////////////////////////////////////////////////
// PARDISO INTERFACE                                                //
//////////////////////////////////////////////////////////////////////

int SparseDirectSolver::PardisoSymbolic()
  {
  if (MyPID_!=0) return 0;
  HYMLS_PROF3(label_,"PardisoSymbolic");

  int num_procs = 1;
  char* var = getenv("OMP_NUM_THREADS");
  if (var != NULL)
    {
    sscanf(var, "%d", &num_procs);
    }
  iparam(3) = num_procs;

  int N = serialMatrix_->NumGlobalRows();
  int NumVectors = 1;
  int maxfct = 1; // Max number of factors in memory
  int mnum = 1; // Maxtrix number
  int phase = 11; // Only do analysis
  int msglvl = 0; // No output
  int error = 0;
  double ddum; // Dummy variable

  // We use our own permutation, so just pass a 0,...,N array to PARDISO
  pardiso_perm_.resize(N);
  for (int i=0; i < N; ++i)
    pardiso_perm_[i] = i;

  pardiso(pardiso_pt_, &maxfct, &mnum, &pardiso_mtype_, &phase,
    &N, &Aval_[0], &Ap_[0], &Ai_[0], &pardiso_perm_[0], &NumVectors,
    pardiso_iparam_, &msglvl, &ddum, &ddum, &error);

  pardiso_initialized_ = true;

  if (error)
    {
    HYMLS::Tools::Error("PARDISO Symbolic Error "+Teuchos::toString(error),
      __FILE__,__LINE__);
    }

  return 0;
  }

//=============================================================================

int SparseDirectSolver::PardisoNumeric()
  {
  HYMLS_PROF3(label_,"PardisoNumeric");
  if (MyPID_!=0) return 0;

  int N = serialMatrix_->NumGlobalRows();
  int NumVectors = 1;
  int maxfct = 1; // Max number of factors in memory
  int mnum = 1; // Maxtrix number
  int phase = 22; // Only do numerical factorization
  int msglvl = 0; // No output
  int error = 0;
  double ddum; // Dummy variable

  pardiso(pardiso_pt_, &maxfct, &mnum, &pardiso_mtype_, &phase,
    &N, &Aval_[0], &Ap_[0], &Ai_[0], &pardiso_perm_[0], &NumVectors,
    pardiso_iparam_, &msglvl, &ddum, &ddum, &error);

  if (error)
    {
    HYMLS::Tools::Error("PARDISO Numeric Error "+Teuchos::toString(error),
      __FILE__,__LINE__);
    // condition number is not in PARDISO?
    }
  return 0;
  }

//=============================================================================

int SparseDirectSolver::PardisoSolve(const Epetra_MultiVector& B, Epetra_MultiVector& X) const
  {
  HYMLS_PROF3(label_,"PardisoSolve");

  if (Matrix_.get()!=serialMatrix_.get()) return -99; // not implemented

  int N = Matrix_->NumMyRows();
  int NumVectors = X.NumVectors();

  Teuchos::RCP<Epetra_MultiVector> serialX = Teuchos::rcp(&X,false);
  Teuchos::RCP<const Epetra_MultiVector> serialB = Teuchos::rcp(&B,false);

  double *xbuf = new double[NumVectors * N];
  double *bbuf = new double[NumVectors * N];

  const Teuchos::RCP<Epetra_Vector>& sca_l =
    UseTranspose_? scaRight_: scaLeft_;
  const Teuchos::RCP<Epetra_Vector>& sca_r =
    UseTranspose_? scaLeft_: scaRight_;
  const Teuchos::Array<int>& row_perm =
    UseTranspose_? col_perm_: row_perm_;
  const Teuchos::Array<int>& col_perm =
    UseTranspose_? row_perm_: col_perm_;

  if (UseTranspose_)
    {
    iparam(12) = 1;
    }
  else
    {
    iparam(12) = 0;
    }

  int maxfct = 1; // Max number of factors in memory
  int mnum = 1; // Maxtrix number
  int phase = 33; // Solve, iterative refinement
  int msglvl = 0; // No output
  int error = 0;

  if ( MyPID_ == 0 )
    {
    // Get direct pointers to the arrays, which speeds up the code somewhat
    const double *sca_l_ptr = sca_l->Values();
    const int *row_perm_ptr = row_perm.getRawPtr();
    const double *sca_r_ptr = sca_r->Values();
    const int *col_perm_ptr = col_perm.getRawPtr();

    for (int j = 0 ; j < NumVectors; j++)
      {
      double *bbuf_ptr = bbuf + j * N;
      const double *serialB_ptr = (*serialB)[j];
      for (int i = 0; i < N; i++)
        {
        bbuf[i] = serialB_ptr[row_perm_ptr[i]] * sca_l_ptr[row_perm_ptr[i]];
        }
      }

    pardiso(pardiso_pt_, &maxfct, &mnum, &pardiso_mtype_, &phase,
      &N, &Aval_[0], &Ap_[0], &Ai_[0], &pardiso_perm_[0], &NumVectors,
      pardiso_iparam_, &msglvl, bbuf, xbuf, &error);

    // we now have x(col_perm) in x_buf
    for (int j = 0; j < NumVectors; j++)
      {
      double *xbuf_ptr = xbuf + j * N;
      double *serialX_ptr = (*serialX)[j];
      for (int i = 0; i < N; i++)
        {
        serialX_ptr[col_perm_ptr[i]] = xbuf[i] * sca_r_ptr[col_perm_ptr[i]];
        }
      }
    }

  delete[] xbuf;
  delete[] bbuf;

  if (serialX.get()!=&X) return -99; //not implemented
  return error;
  }

//////////////////////////////////////////////////////////////////////
// END PARDISO INTERFACE                                            //
//////////////////////////////////////////////////////////////////////

#endif // HAVE_PARDISO

void SparseDirectSolver::DumpSolverStatus(std::string filePrefix,
  bool overwrite,
  Teuchos::RCP<const Epetra_MultiVector> X,
  Teuchos::RCP<const Epetra_MultiVector> B) const
  {
  if (overwrite==false)
    {
    std::ifstream ifs((filePrefix+".m").c_str());
    if (ifs)
      {
      Tools::out() << "status info not written, file '"<<filePrefix<<".m exists.\n";
      return;
      }
    }
  Tools::out() << label_ << ": writing status info to " << filePrefix << "* files"<<std::endl;

  int N = Ap_.size()-1;

#ifdef HAVE_SUITESPARSE
  if (method_==UMFPACK)
    {
    std::ofstream umf_fs((filePrefix+"_umfpack.txt").c_str());

    output_stream = &umf_fs;

    int old_prl = umf_Control_[UMFPACK_PRL];
    const_cast<double&>(umf_Control_[UMFPACK_PRL])=5;

    umfpack_di_report_info(&umf_Control_[0],&umf_Info_[0]);
    umfpack_di_report_matrix(N,N,&Ap_[0],&Ai_[0],&Aval_[0],0,&umf_Control_[0]);

    const_cast<double&>(umf_Control_[UMFPACK_PRL])=old_prl;

    output_stream = &Tools::out();
    umf_fs.close();
    }
#endif
  Teuchos::RCP<const Epetra_CrsMatrix> serialCrsMatrix =
    Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(serialMatrix_);
  if (serialCrsMatrix!=Teuchos::null)
    {
    MatrixUtils::Dump(*serialCrsMatrix,filePrefix+"_A.txt");
    }
  std::ofstream ofs((filePrefix+".m").c_str());
  ofs << std::setprecision(16) << std::setw(16) << std::scientific;
  ofs << "p=[";
  for (int i=0;i<row_perm_.size();i++)
    {
    ofs << row_perm_[i]+1 << " ";
    }
  ofs << "];\n";
  ofs << "q=[";
  for (int i=0;i<col_perm_.size();i++)
    {
    ofs << col_perm_[i]+1 << " ";
    }
  ofs << "];\n";
  ofs << "tmp=[...\n";
  for (int i=0;i<serialMatrix_->NumMyRows();i++)
    {
    for (int j=Ap_[i];j<Ap_[i+1];j++)
      {
      ofs << i+1 << " " << Ai_[j]+1 << " " << Aval_[j] << std::endl;
      }
    }
  ofs<<"];"<<std::endl;
  ofs<<"Apq=sparse(tmp(:,1),tmp(:,2),tmp(:,3));\n\n";

  ofs << "S_left = spdiags([...\n";
  for (int i=0;i<scaLeft_->MyLength();i++)
    {
    ofs << (*scaLeft_)[i]<<std::endl;
    }
  ofs << "],0,"<<N<<","<<N<<");\n\n";
  ofs << "S_right = spdiags([...\n";
  for (int i=0;i<scaRight_->MyLength();i++)
    {
    ofs << (*scaRight_)[i]<<std::endl;
    }
  ofs << "],0,"<<N<<","<<N<<");\n\n";

/*
  ofs << "bp=[";
  for (int ii=0;ii<N;ii++) ofs << b_buf[ii] << " ";
  ofs << "]';\n\nxq=[";
  for (int ii=0;ii<N;ii++) ofs << x_buf[ii] << " ";
  ofs << "]';";
*/
  ofs.close();

  if (X!=Teuchos::null)
    {
    MatrixUtils::Dump(*X, filePrefix+"_X.txt");
    }
  if (B!=Teuchos::null)
    {
    MatrixUtils::Dump(*B, filePrefix+"_B.txt");
    }
  }

// return number of nonzeros in original matrix
int SparseDirectSolver::NumGlobalNonzerosA() const
  {
  return Matrix_->NumGlobalNonzeros();
  }

int SparseDirectSolver::NumGlobalNonzerosL() const
  {
    if (method_==KLU)
      return klu_->Numeric_->lnz;
    return 0;
  }

int SparseDirectSolver::NumGlobalNonzerosU() const
  {
    if (method_==KLU)
      return klu_->Numeric_->unz;
    return 0;
  }

  }//namespace HYMLS


#endif // use Amesos instead of our own interface
