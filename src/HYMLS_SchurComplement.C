#include "HYMLS_no_debug.H"

#include "HYMLS_SchurComplement.H"
#include "HYMLS_OverlappingPartitioner.H"
#include "HYMLS_SparseDirectSolver.H"
#include "HYMLS_MatrixUtils.H"

#include "Epetra_Comm.h"
#include "Epetra_Map.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"
#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_RowMatrix.h"
#include "Epetra_CrsMatrix.h"
#include "Epetra_FECrsMatrix.h"
#include "Epetra_Import.h"

#include "Ifpack_Container.h"

#include "Teuchos_ParameterList.hpp"
#include "Teuchos_Array.hpp"

#include "EpetraExt_MatrixMatrix.h"
#include "EpetraExt_Reindex_MultiVector.h"


namespace HYMLS {

// operator representation of our Schur complement.
// allows applying the Schur complement of our factorization
// to a vector without actually constructing it.
// Also provides functionality to explicitly construct parts
// of the SC or the whole thing as sparse or dense matrix.

  SchurComplement::SchurComplement(Teuchos::RCP<const Preconditioner> mother, int lev)
       : mother_(mother),
       label_("SchurComplement (level "+Teuchos::toString(lev)+")"),
         comm_(Teuchos::rcp(&(mother->Comm()),false)),
         useTranspose_(false), normInf_(-1.0),
         sparseMatrixRepresentation_(Teuchos::null),
         flopsApply_(0.0), flopsCompute_(0.0)
    {
    START_TIMER3(label_,"Constructor");
    isConstructed_=false;
    // we do a finite-element style assembly of the full matrix    
    const Epetra_Map& map = mother_->Map2();
    const HierarchicalMap& hid = mother_->Partitioner();
    sparseMatrixRepresentation_ = Teuchos::rcp(new 
      Epetra_FECrsMatrix(Copy,map,mother_->Matrix().MaxNumEntries()));
    Scrs_=sparseMatrixRepresentation_;
    sca_left_=Teuchos::rcp(new Epetra_Vector(map));
    sca_right_=Teuchos::rcp(new Epetra_Vector(map));
    sca_left_->PutScalar(1.0);
    sca_right_->PutScalar(1.0);
    }

  // destructor
  SchurComplement::~SchurComplement()
    {
    START_TIMER3(label_,"Destructor");
    }


  // Applies the operator. Here X and Y are based on the map
  // mother_->Map2().
  int SchurComplement::Apply(const Epetra_MultiVector& X,
                  Epetra_MultiVector& Y) const
    {
    START_TIMER2(label_,"Apply");
    int ierr=0;
    if (IsConstructed())
      {
      CHECK_ZERO(Scrs_->Apply(X,Y));
#ifdef FLOPS_COUNT
      flopsApply_+=2*Scrs_->NumGlobalNonzeros();
#endif
      }
    else
      {
      // we now have overlap in the rowMap of the Preconditioner class
      // and I can't oversee if this would still work.
      Tools::Error("distributed SC currently disabled",__FILE__,__LINE__);
#if 0
      // The Schur-complement is given by A22-A21*A11\A12      
      CHECK_ZERO(mother_->ApplyA22(X,Y, &flopsApply_));
      
      // 2) compute y2 = A21*A11\A12*X      
      Epetra_MultiVector Y1(mother_->RowMap(), Y.NumVectors());
      Epetra_MultiVector Z1(mother_->RowMap(), Y.NumVectors());
      Epetra_MultiVector Y2(mother_->Map2(), Y.NumVectors());
      
      CHECK_ZERO(mother_->ApplyA12(X,Y1,&flopsApply_));

      CHECK_ZERO(mother_->ApplyInverseA11(Y1,Z1));
      
      CHECK_ZERO(mother_->ApplyA21(Z1,Y2,&flopsApply_));
      
      // 3) compute Y = Y-Y2
      CHECK_ZERO(Y.Update(-1.0,Y2,1.0));
#ifdef FLOPS_COUNT
      flopsApply_+=Y.GlobalLength()*Y.NumVectors();
#endif
#endif
      }
    return ierr;
    }

  // Apply inverse operator - not implemented.
  int SchurComplement::ApplyInverse(const Epetra_MultiVector& X,
                           Epetra_MultiVector& Y) const
    {
    Tools::Warning("ApplyInverse() not available!",__FILE__,__LINE__);
    return -1;
    }                           

  
  
  
  // construct complete Schur complement as a sparse matrix
  int SchurComplement::Construct()
    {
    START_TIMER3(label_,"Construct (1)");
    const Epetra_Map& map = mother_->Map2();
    const OverlappingPartitioner& hid = mother_->Partitioner();
    
    isConstructed_=true;
    CHECK_ZERO(this->Construct(sparseMatrixRepresentation_));
    Scrs_ = MatrixUtils::DropByValue(sparseMatrixRepresentation_,
        HYMLS_SMALL_ENTRY);
    REPORT_MEM(label_,"SchurComplement",Scrs_->NumGlobalNonzeros(),
                                        Scrs_->NumGlobalNonzeros()+
                                        Scrs_->NumGlobalRows());
    return 0;
    }

  int SchurComplement::Construct(Teuchos::RCP<Epetra_FECrsMatrix> S) const
    {
    START_TIMER2(label_,"Construct FEC");
    Epetra_IntSerialDenseVector indices;
    Epetra_SerialDenseMatrix Sk;
    
    const Epetra_Map& map = mother_->Map2();
    const OverlappingPartitioner& hid = mother_->Partitioner();
    
    if (map.NumGlobalElements()==0) return 0; // empty SC

    if (!S->Filled())
      {

      // start out by just putting the structure together.
      // I do this because the SumInto function will fail 
      // unless the values have been put in already. On the
      // other hand, the Insert function will overwrite stuff
      // we put in previously.
  
      for (int k=0;k<hid.NumMySubdomains();k++)
        {
        CHECK_ZERO(this->Construct(k, indices));
        DEBVAR(k);
        DEBVAR(indices);
        if (indices.Length()!=Sk.N())
          {
          Sk.Shape(indices.Length(),indices.Length());
          }
        int ierr=S->InsertGlobalValues(indices,Sk);
        if (ierr<0) 
          {
          Tools::Warning("error "+Teuchos::toString(ierr)+" returned from call S->InsertGlobalValues",
                          __FILE__,__LINE__);
          return ierr;
          }
        }
  
      DEBUG("SchurComplement: Assembly with all zeros...");
      //assemble without calling FillComplete because we
      // still miss A22 in the pattern
      CHECK_ZERO(S->GlobalAssemble(false));
      }
    else
      {
      CHECK_ZERO(S->PutScalar(0.0));
      }
    
    for (int k=0;k<hid.NumMySubdomains();k++)
      {
      // construct values for separators around subdomain k
      CHECK_ZERO(this->Construct(k, indices));
      CHECK_ZERO(this->Construct(k, Sk, indices, &flopsCompute_));
      
      CHECK_ZERO(S->SumIntoGlobalValues(indices,Sk));
      }
    DEBUG("SchurComplement - GlobalAssembly");
    CHECK_ZERO(S->GlobalAssemble(false));
    //DEBVAR(mother_->A22());
    CHECK_ZERO(EpetraExt::MatrixMatrix::Add(mother_->A22(), false, 1.0, 
                                                *S,-1.0));
    // finish construction by creating local IDs:
    CHECK_ZERO(S->FillComplete());    
    return 0;
    }

  int SchurComplement::Construct(int sd, Epetra_IntSerialDenseVector& inds) const
    {
    START_TIMER2(label_,"Construct ISDV");
    const OverlappingPartitioner& hid=mother_->Partitioner();
    
    if (sd<0 || sd>hid.NumMySubdomains())
      {
      Tools::Warning("Subdomain index out of range!",__FILE__,__LINE__);
      return -1;
      }
    
    int nrows = hid.NumSeparatorElements(sd);
    
    
    // resize input arrays if necessary
    if (inds.Length()!=nrows)
      {
      CHECK_ZERO(inds.Size(nrows));
      }
    
    // create vector with global indices
    int pos=0;

    // loop over all groups except the first (first is interior elements),
    // that is separator groups and retained elements
    for (int grp=1;grp<hid.NumGroups(sd);grp++)
      {
      // loop over all elements of each separator group
      for (int j=0; j<hid.NumElements(sd,grp);j++)
        {
        inds[pos++]=hid.GID(sd,grp,j);
        }
      }
    return 0;
    }



  int SchurComplement::Construct(int sd, Epetra_SerialDenseMatrix& Sk, 
                                        const Epetra_IntSerialDenseVector& inds,
                                        double* count_flops) const
    {
    START_TIMER2(label_,"Construct SDM");
#ifdef FLOPS_COUNT
    double flops=0;
#endif    
    const OverlappingPartitioner& hid=mother_->Partitioner();
    const Epetra_CrsMatrix& A12 = mother_->A12(sd);
    const Epetra_CrsMatrix& A21 = mother_->A21(sd);
    const Epetra_CrsMatrix& A22 = mother_->A22();
    Ifpack_Container& _A11 = mother_->SolverA11(sd);
    Ifpack_SparseContainer<SparseDirectSolver> *sparseA11
        = dynamic_cast<Ifpack_SparseContainer<SparseDirectSolver>*>(&_A11);
    if (sparseA11==NULL)
      {
      Tools::Error("use of dense subdomain solvers not implemented, yet!",
        __FILE__, __LINE__);
      }
    Ifpack_SparseContainer<SparseDirectSolver>& A11 = *sparseA11;
    if (sd<0 || sd>hid.NumMySubdomains())
      {
      Tools::Warning("Subdomain index out of range!",__FILE__,__LINE__);
      return -1;
      }
    
#ifdef TESTING
 // verify that the ID array of the subdomain solver is sorted
 // in ascending order, I think we assume that...
 for (int i=1;i<A11.NumRows();i++)
   {
   if (A11.ID(i)<A11.ID(i-1))
     {
     Tools::Warning("re-indexing of blocks is not supported!",__FILE__,__LINE__);
     }
   }
#endif

    int nrows = hid.NumSeparatorElements(sd);
              
    if (inds.Length()!=nrows)
      {
      return -1; // caller probably did not call Construct(indices)
      }

    if (Sk.M()!=nrows || Sk.N()!=nrows) 
      {
      CHECK_ZERO(Sk.Shape(nrows,nrows));
      }

    if (A11.NumRows()==0)
      {      
      return 0; // has only an A22-contribution (no interior elements)
      }
          
    A11.SetNumVectors(nrows);

    DEBVAR(sd);
    DEBVAR(inds);
    DEBVAR(nrows);

{
    START_TIMER2(label_, "Fill RHS");
    int int_elems = hid.NumInteriorElements(sd);
    int len[int_elems];
    int *indices[int_elems];
    double *values[int_elems];
    // loop over all rows in this subdomain
    for (int i = 0;i<int_elems;i++)
      {
      // get a view of the matrix row (with all separator couplings)
      CHECK_ZERO(A12.ExtractMyRowView(i,len[i],values[i],indices[i]));
      }

    int pos = 0; // position in multi-vector (rhs of subdomain solver)

    const Epetra_Map& A12_ColMap = A12.ColMap();
    // now loop over all the separators around this subdomain
    for (int grp=1;grp<hid.NumGroups(sd);grp++)
      {
      // loop over all elements of each separator group
      for (int j=0; j<hid.NumElements(sd,grp);j++)
        {
        int gcid=hid.GID(sd,grp,j);// global ID of separator node (sd,grp,j)
        double *rhs = &A11.RHS(0, pos);
        // loop over all rows in this subdomain
        for (int i = 0;i<int_elems;i++)
          {
          const int *incices_ptr = indices[i];
          const double *values_ptr = values[i];
          // loop over the matrix row and look for matching entries
          for (int k = 0 ; k < len[i]; k++)
            {
            if (gcid == A12_ColMap.GID(incices_ptr[k]))
              rhs[i] = values_ptr[k];
            }
          }
        pos++;
        }
      }
}

//    DEBUG("Apply A11 inverse...");
#ifdef FLOPS_COUNT    
    double flopsOld=A11.ApplyInverseFlops();
#endif
    IFPACK_CHK_ERR(A11.ApplyInverse());
#ifdef FLOPS_COUNT
    double flopsNew=A11.ApplyInverseFlops();
    //TODO: these flops are counted twice: in Solver->ApplyInverse() they shouldn't 
    //      contribute!
    flops+=flopsNew-flopsOld;
#endif
    
    // get the solution, B=A11\A12, as a MultiVector in the domain map of operator A21

    Teuchos::RCP<const Epetra_MultiVector> lhs = A11.LHS();
    EpetraExt::MultiVector_Reindex reindex(mother_->Map1(sd));
    const Epetra_MultiVector& B = reindex(const_cast<Epetra_MultiVector&>(*lhs));

    // multiply by A21, giving A21*(A11\A12) in a vector based on Map2 (i.e. with a row
    // for each separator element) and a column for each separator node connected to this 
    // subdomain. Some separators may not be on this CPU: those need to be imported 
    // manually later on.

    Epetra_MultiVector Aloc(mother_->Map2(sd),B.NumVectors());
    CHECK_ZERO(A21.Multiply(false,B,Aloc));
#ifdef FLOPS_COUNT    
    flops +=2*B.NumVectors()*A21.NumGlobalNonzeros();
#endif
    // re-index and put into final block
    
//    DEBUG("Copy into Sk matrix");
    const Epetra_Map& mmap = mother_->Map2(sd);
    for (int i = 0; i < nrows; i++)
      {
      const int lrid = mmap.LID(inds[i]);
      for (int j = 0; j < nrows; j++)
        {
        Sk(j,i) = Aloc[j][lrid];
        }
      }

    A11.SetNumVectors(1);
    
//    DEBUG("Block constructed successfully!");
#ifdef FLOPS_COUNT
    if (count_flops!=NULL) *count_flops+=flops;
#endif
    return 0;
    }

Teuchos::RCP<Epetra_Vector> SchurComplement::ConstructLeftScaling(int p_variable)
  {
  START_TIMER2(label_,"ConstructLeftScaling");
  sca_left_->PutScalar(1.0);
  double *val; 
  int* ind;
  int len;
  bool has_pcol;
  double diag;
  const OverlappingPartitioner& hid = mother_->Partitioner();
  const BasePartitioner& BP = hid.Partitioner();
  if (!IsConstructed())
    {
    Tools::Warning("Schur-complement not constructed, using default scaling",__FILE__,__LINE__);
    }
  else
    {
    for (int i=0;i<Scrs_->NumMyRows();i++)
      {
      diag=1.0;
      has_pcol=false;
      CHECK_ZERO(Scrs_->ExtractMyRowView(i,len,val,ind));
      for (int j=0;j<len;j++)
        {
        if (Scrs_->GRID(i)==Scrs_->GCID(ind[j]))
          {
          diag=std::abs(val[j]);
          }
        if (BP.VariableType(Scrs_->GCID(ind[j]))==p_variable)
          {
          if (std::abs(val[j])>1.0e-8) has_pcol=true;
          }
        }
      if ((has_pcol==false) && (diag>1.0e-10))
        {
        (*sca_left_)[i] = 1.0/diag;
        }
      }
    }
  return sca_left_;
  }

int SchurComplement::Scale(Teuchos::RCP<Epetra_Vector> sca_left, Teuchos::RCP<Epetra_Vector> sca_right)
  {
  START_TIMER3(label_,"Scale");
  int ierr=0;
  if (!IsConstructed())    
    {
    ierr=1;
    }
  else
    {
    ierr=sparseMatrixRepresentation_->LeftScale(*sca_left);
    if (ierr==0)
      {
      ierr=sparseMatrixRepresentation_->RightScale(*sca_right);
      }
    }
  return ierr;
  }

int SchurComplement::Unscale(Teuchos::RCP<Epetra_Vector> sca_left, Teuchos::RCP<Epetra_Vector> sca_right)
  {
  START_TIMER3(label_,"Unscale");
  
  int ierr=0;
  if (!IsConstructed())    
    {
    ierr=1;
    }
  else
    {
    Epetra_Vector left(sca_left->Map());
    left.Reciprocal(*sca_left);
    Epetra_Vector right(sca_right->Map());
    right.Reciprocal(*sca_right);
    ierr=sparseMatrixRepresentation_->LeftScale(left);
    if (ierr==0)
      {
      ierr=sparseMatrixRepresentation_->RightScale(right);
      }
    }
  return ierr;
  }

}
