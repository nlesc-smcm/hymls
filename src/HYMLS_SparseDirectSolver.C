#include "HYMLS_SparseDirectSolver.H"

#ifdef HAVE_AMESOS_UMFPACK

#include "Ifpack_Condest.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Map.h"
#include "Epetra_Comm.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_Time.h"
#include "Teuchos_ParameterList.hpp"

#include "HYMLS_Tools.H"
#include "HYMLS_MatrixUtils.H"

extern "C" {
#include "umfpack.h"
}

namespace HYMLS {

//==============================================================================
SparseDirectSolver::SparseDirectSolver(Epetra_RowMatrix* Matrix_in) :
  Matrix_(Teuchos::rcp( Matrix_in, false )),
  label_("HYMLS Umfpack"),
  IsEmpty_(false),
  IsInitialized_(false),
  IsComputed_(false),
  UseTranspose_(false),
  NumInitialize_(0),
  NumCompute_(0),
  NumApplyInverse_(0),
  InitializeTime_(0.0),
  ComputeTime_(0.0),
  ApplyInverseTime_(0.0),
  ComputeFlops_(0),
  ApplyInverseFlops_(0),
  Condest_(-1.0),
  serialMatrix_(Teuchos::null),
  serialImport_(Teuchos::null)
{
MyPID_=Matrix_->Comm().MyPID();
umf_Info_.resize(UMFPACK_INFO);
umf_Control_.resize(UMFPACK_CONTROL);
umfpack_di_defaults( &umf_Control_[0] ) ; 
umf_Symbolic_=NULL;
umf_Numeric_=NULL;
}

//==============================================================================
SparseDirectSolver::SparseDirectSolver(const SparseDirectSolver& rhs) :
  Matrix_(Teuchos::rcp( &rhs.Matrix(), false )),
  label_(rhs.Label()),
  IsEmpty_(false),
  IsInitialized_(false),
  IsComputed_(false),
  NumInitialize_(rhs.NumInitialize()),
  NumCompute_(rhs.NumCompute()),
  NumApplyInverse_(rhs.NumApplyInverse()),
  InitializeTime_(rhs.InitializeTime()),
  ComputeTime_(rhs.ComputeTime()),
  ApplyInverseTime_(rhs.ApplyInverseTime()),
  ComputeFlops_(rhs.ComputeFlops()),
  ApplyInverseFlops_(rhs.ApplyInverseFlops()),
  Condest_(rhs.Condest())
{
Tools::Error("not implemented!",__FILE__,__LINE__);
}

SparseDirectSolver::~SparseDirectSolver()
  {
  if (umf_Symbolic_) umfpack_di_free_symbolic (&umf_Symbolic_) ;
  if (umf_Numeric_) umfpack_di_free_numeric (&umf_Numeric_) ;
  }

//==============================================================================
int SparseDirectSolver::SetParameters(Teuchos::ParameterList& List_in)
{

  List_ = List_in;
  std::string choice = List_in.get("amesos: solver type", label_);
  if (choice!="Amesos_Umfpack")
    {
    Tools::Warning("your choice of 'amesos: solver type' is ignored\n"
                   "The HYMLS direct solver always uses Umfpack right now\n",
                   __FILE__,__LINE__);
    }
return(0);
}

//==============================================================================
int SparseDirectSolver::Initialize()
{
START_TIMER2(label_,"Initialize");
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
    ++NumInitialize_;
    return(0);
  }

  // create timer, which also starts it.
  if (Time_ == Teuchos::null)
    Time_ = Teuchos::rcp( new Epetra_Time(Comm()) );

  if (UseTranspose_) Tools::Error("not implemented",__FILE__,__LINE__);

  // create umfpack
  CHECK_ZERO(this->ConvertToSerial());
  CHECK_ZERO(this->FillReducingOrdering());
  CHECK_ZERO(this->ConvertToUmfpackCRS());
  CHECK_ZERO(this->UmfpackSymbolic());

  IsInitialized_ = true;
  ++NumInitialize_;
  InitializeTime_ += Time_->ElapsedTime();
  return(0);
}

//==============================================================================
int SparseDirectSolver::Compute()
{
START_TIMER2(label_,"Compute");
  if (!IsInitialized())
    CHECK_ZERO(Initialize());

  if (IsEmpty_) {
    IsComputed_ = true;
    ++NumCompute_;
    return(0);
  }

  IsComputed_ = false;
  Time_->ResetStartTime();

  if (Matrix_ == Teuchos::null)
    {
    Tools::Error("null matrix",__FILE__,__LINE__);
    }
  if (serialMatrix_.get()!=Teuchos::rcp_dynamic_cast<const Epetra_CrsMatrix>(Matrix_).get())
    {
    if (serialImport_==Teuchos::null)
      {
      Tools::Error("importer is null",__FILE__,__LINE__);
      }
    CHECK_ZERO(serialMatrix_->Import(*Matrix_,*serialImport_,Insert));
    }
  CHECK_ZERO(this->UmfpackNumeric());

  IsComputed_ = true;
  ++NumCompute_;
  ComputeTime_ += Time_->ElapsedTime();
  return(0);
}

//==============================================================================
int SparseDirectSolver::SetUseTranspose(bool UseTranspose_in)
{
UseTranspose_ = UseTranspose_in;
return(99);// I have not checked wether this really works
}

//==============================================================================
int SparseDirectSolver::
Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  // check for maps? check UseTranspose_?
  return(-99);
  CHECK_ZERO(Matrix_->Apply(X,Y));
}

//==============================================================================
int SparseDirectSolver::
ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const
{
  if (IsEmpty_) {
    ++NumApplyInverse_;
    return(0);
  }

  if (IsComputed() == false) 
        {return -1;}

  if (X.NumVectors() != Y.NumVectors())
        {return -2;}
  
  Time_->ResetStartTime();

  // AztecOO gives X and Y pointing to the same memory location,
  // need to create an auxiliary vector, Xcopy
  Teuchos::RCP<const Epetra_MultiVector> Xcopy;
  if (X.Pointers()[0] == Y.Pointers()[0])
    Xcopy = Teuchos::rcp( new Epetra_MultiVector(X) );
  else
    Xcopy = Teuchos::rcp( &X, false );

  CHECK_ZERO(this->UmfpackSolve(*Xcopy,Y));

  ++NumApplyInverse_;
  ApplyInverseTime_ += Time_->ElapsedTime();

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

  return(os);
}

// private member functions

int SparseDirectSolver::ConvertToSerial()
  {
  if (Matrix_->Comm().NumProc()==1)
    {
    serialMatrix_=Teuchos::rcp_dynamic_cast<Epetra_CrsMatrix>(Teuchos::rcp_const_cast<Epetra_RowMatrix>(Matrix_));
    // not implemented unless CRS:
    if (serialMatrix_==Teuchos::null) Tools::Error("need a CrsMatrix here",__FILE__,__LINE__);
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
  int N = serialMatrix_->NumMyRows();
  row_perm_.resize(N);
  col_perm_.resize(N);
  CHECK_ZERO(HYMLS::MatrixUtils::FillReducingOrdering(*serialMatrix_,row_perm_,col_perm_));
  //
  for (int i=0;i<N;i++)
    {
    row_perm_[i]=i;
    col_perm_[i]=i;
    }
  //
  return 0;
  }

//=============================================================================
int SparseDirectSolver::ConvertToUmfpackCRS()
{
START_TIMER2(label_,"ConvertToUmfpackCRS");
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
        Ai_[Ai_index+j] = invperm[Ai_[Ai_index+j]];
        }
      Ai_index += len;
    }

    Ap_[N] = Ai_index; 
  }
  
  return 0;
}

int SparseDirectSolver::UmfpackSymbolic() 
{
if (MyPID_!=0) return 0;
START_TIMER2(label_,"UmfpackSymbolic");  
  
  int N = serialMatrix_->NumGlobalRows();
  double *Control = &umf_Control_[0];
  double *Info = &umf_Info_[0];
  
  if (umf_Symbolic_) 
    umfpack_di_free_symbolic (&umf_Symbolic_) ;

  umf_Control_[UMFPACK_ORDERING] = UMFPACK_ORDERING_NONE;
  CHECK_ZERO(umfpack_di_symbolic (N, N, &Ap_[0], 
                             &Ai_[0], &Aval_[0], 
                             &umf_Symbolic_, Control, Info) );

  return 0;
}
//=============================================================================
int SparseDirectSolver::UmfpackNumeric() 
{
START_TIMER2(label_,"UmfpackNumeric");
if (MyPID_!=0) return 0;

    if (umf_Numeric_) umfpack_di_free_numeric (&umf_Numeric_) ;
  //TODO - this is dangerous in general so we should have a flag like
  //       'TrustMe'
  umf_Control_[UMFPACK_SYM_PIVOT_TOLERANCE] = 0.0;
  umf_Control_[UMFPACK_PIVOT_TOLERANCE] = 0.0;
  CHECK_ZERO(umfpack_di_numeric (&Ap_[0], 
                                     &Ai_[0], 
                                     &Aval_[0], 
                                     umf_Symbolic_, 
                                     &umf_Numeric_, 
                                     &umf_Control_[0], 
                                     &umf_Info_[0]) );
    double rcond = umf_Info_[UMFPACK_RCOND];
#ifdef TESTING
    if (rcond>0.0)
      {
      if (rcond<1.0e-14) Tools::Warning("nearly singular matrix encountered!\n"
                                      "(RCOND="+Teuchos::toString(rcond)+")",
                                      __FILE__,__LINE__);
      else if (rcond<1.0e-6) Tools::Warning("ill-conditioned matrix encountered!\n"
                                      "(RCOND="+Teuchos::toString(rcond)+")",
                                      __FILE__,__LINE__);
      }
#endif

#ifdef DEBUGGING
      int N = serialMatrix_->NumMyRows();
      double* Control=&umf_Control_[0]; //NULL;
      double* Info=&umf_Info_[0]; //NULL;
      double *x_buf = new double[N];
      double *b_buf = new double[N];
      for (int i=0;i<N;i++) b_buf[i] = row_perm_[i]+1.0;
      for (int i=0;i<N;i++) x_buf[i] = -42.0;
      CHECK_ZERO(umfpack_di_solve (UMFPACK_At, &Ap_[0], 
                                     &Ai_[0], &Aval_[0], 
                                     x_buf, b_buf, 
                                     umf_Numeric_,
                                     Control,Info)); 
                                     //const_cast<double*>(&umf_Control_[0]), 
                                     //const_cast<double*>(&umf_Info_[0]));

MatrixUtils::Dump(*serialMatrix_,"testUmf_A.txt");
std::ofstream ofs("testUmf_Apq.m");
ofs << std::setprecision(16) << std::setw(16) << std::scientific;
ofs << "p=[";
for (int i=0;i<row_perm_.size();i++)
  {
  ofs << row_perm_[i]+1 << " ";
  }
ofs << "];\n";
ofs << "q=[";
for (int i=0;i<row_perm_.size();i++)
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
ofs<<"Apq=sparse(tmp(:,1),tmp(:,2),tmp(:,3));\n";
ofs.close();
MatrixUtils::Dump(*serialMatrix_,"testUmf_A.txt"); 
ofs.open("testUmf_vectors.m");
ofs << "bp=[";
for (int ii=0;ii<N;ii++) ofs << b_buf[ii] << " ";
ofs << "];\n\nxq=[";
for (int ii=0;ii<N;ii++) ofs << x_buf[ii] << " ";
ofs << "];";
ofs.close();

delete [] x_buf;
delete [] b_buf;

#endif

  return 0;
}

///////////////////////////////////////

int SparseDirectSolver::UmfpackSolve(const Epetra_MultiVector& B, Epetra_MultiVector& X) const
{ 
START_TIMER2(label_,"UmfpackSolve");

if (Matrix_.get()!=serialMatrix_.get()) return -99; // not implemented

Teuchos::RCP<Epetra_MultiVector> serialX = Teuchos::rcp(&X,false);
Teuchos::RCP<Epetra_MultiVector> serialB = 
        Teuchos::rcp(const_cast<Epetra_MultiVector*>(&B),false);

  double *xvalues ;
  double *bvalues ;
  int xlda,blda;
  int N = serialX->MyLength();  
  double *x_buf = new double[N];
  double *b_buf = new double[N];
  
  int NumVectors = X.NumVectors();

  int UmfpackRequest = UseTranspose()?UMFPACK_A:UMFPACK_At ;
  int status = 0;

  if ( MyPID_ == 0 ) 
    {    
    for ( int j =0 ; j < NumVectors; j++ ) 
      {
      double* Control=NULL;
      double* Info=NULL;
      for (int i=0;i<N;i++) b_buf[i] = (*serialB)[j][row_perm_[i]];
      status = umfpack_di_solve (UmfpackRequest, &Ap_[0], 
                                     &Ai_[0], &Aval_[0], 
                                     x_buf, b_buf, 
                                     const_cast<void*>(umf_Numeric_),
                                     Control,Info); 
                                     //const_cast<double*>(&umf_Control_[0]), 
                                     //const_cast<double*>(&umf_Info_[0]));
      for (int i=0;i<N;i++) (*serialX)[j][i] = x_buf[col_perm_[i]];
      }
    }
  delete [] x_buf;
  delete [] b_buf;

  if (serialX.get()!=&X) return -99; //not implemented
  return 0;
  }
}//namespace HYMLS
#endif
