
#include "HYMLS_SchurPreconditioner.H"

#include "HYMLS_Tools.H" 
#include "HYMLS_MatrixUtils.H"

#include "HYMLS_OverlappingPartitioner.H" 
#include "HYMLS_SchurComplement.H" 
#include "HYMLS_Householder.H" 
#include "HYMLS_SepNode.H"

#include "Epetra_Comm.h" 
#include "Epetra_Map.h" 
#include "Epetra_RowMatrix.h" 
#include "Epetra_Import.h" 
#include "Epetra_IntVector.h" 
#include "Epetra_MultiVector.h" 
#include "Epetra_FECrsMatrix.h" 
#include "Epetra_SerialDenseVector.h" 
#include "Ifpack_Amesos.h" 
#include "Teuchos_RCP.hpp" 
#include "Teuchos_ArrayView.hpp" 
#include "Teuchos_ParameterList.hpp" 
#include "Teuchos_StandardCatchMacros.hpp"

#include "Ifpack_DenseContainer.h" 
#include "Ifpack_SparseContainer.h" 
#include "Ifpack_Amesos.h"

#include "EpetraExt_Reindex_CrsMatrix.h" 
#include "EpetraExt_Reindex_MultiVector.h" 
#include "EpetraExt_MatrixMatrix.h"

        namespace HYMLS {


  // constructor
  SchurPreconditioner::SchurPreconditioner(
                Teuchos::RCP<const Epetra_CrsMatrix> S, 
                Teuchos::RCP<const OverlappingPartitioner> hid, 
                Teuchos::RCP<Teuchos::ParameterList> params,
                int level)
      : SchurMatrix_(S), hid_(hid),
        numInitialize_(0),numCompute_(0),numApplyInverse_(0),
        flopsInitialize_(0.0),flopsCompute_(0.0),flopsApplyInverse_(0.0),
        timeInitialize_(0.0),timeCompute_(0.0),timeApplyInverse_(0.0),
        initialized_(false),computed_(false),
        comm_(Teuchos::rcp(&(S->Comm()),false)),
        map_(Teuchos::rcp(&(S->RowMap()),false)),
        normInf_(-1.0), useTranspose_(false), 
        label_("HYMLS::SchurPreconditioner (level "+Teuchos::toString(level)+")"),
        myLevel_(level),
        sparseMatrixOT_(Teuchos::null),        
        matrix_(Teuchos::null),
        nextLevelHID_(Teuchos::null)
    {
    START_TIMER2(label_,"constructor");
    time_=Teuchos::rcp(new Epetra_Time(*comm_));
    
    SetParameters(*params);
          
    DEBVAR(myLevel_);
    DEBVAR(maxLevel_);
          
    int nnzPerRow=SchurMatrix_->MaxNumEntries();

    if (myLevel_==maxLevel_)
      {
      // reindex the reduced system, this seems to be a good idea when
      // solving it using Ifpack_Amesos
      linearMap_ = Teuchos::rcp(new Epetra_Map(map_->NumGlobalElements(),
                                                  map_->NumMyElements(),
                                                  0, map_->Comm()) );

      reindex_ = Teuchos::rcp(new EpetraExt::CrsMatrix_Reindex(*linearMap_));
      reindexMV_ = Teuchos::rcp(new EpetraExt::MultiVector_Reindex(*linearMap_));
      }
    else
      {
      if (hid_==Teuchos::null)
        {
        Tools::Error("not on coarsest level and no HID available!",
                __FILE__,__LINE__);
        }
      if (hid_->NumMySubdomains()==1)
        {
        //This can cause trouble if a processor partition has no local separators at all,
        //     not quite sure why but for now we should avoid this situation.
        Tools::Warning("PID "+Teuchos::toString(comm_->MyPID())+" has only one subdomain "+
        "on level "+Teuchos::toString(myLevel_)+", this case is not well-implemented!",
                __FILE__,__LINE__);
        }
/*        
      std::cout << "PRECONDITIONER LEVEL "<<myLevel_<<std::endl;
      std::cout << *hid_ << std::endl;
*/      
      }

  //TODO: may want to give the user a choice here, currently we just have
  //      Householder.
  OT=Teuchos::rcp(new Householder());

#ifdef TESTING
  dumpVectors_=true;
#endif  
    
  STOP_TIMER2(label_,"constructor");
  }


  // destructor
  SchurPreconditioner::~SchurPreconditioner()
    {
    DEBUG("Destroy "+label_);
    }

  // Ifpack_Preconditioner interface
  

  // Sets all parameters for the preconditioner.
  int SchurPreconditioner::SetParameters(Teuchos::ParameterList& List)
    {
    
    if (params_==Teuchos::null)
      {
      params_ = Teuchos::rcp(new Teuchos::ParameterList(List));
      }    
    else if (params_.get()!=&List)
      {
      *params_ = List;
      }
    
    maxLevel_=params_->sublist("Solver").get("Number of Levels",myLevel_);
    subdivideSeparators_=params_->sublist("Solver").get("Subdivide Separators",false);
    
    int pos=1;
    
    fix_gid_.resize(0);
    
    while (pos>0)
      {
      string label="Fix GID "+Teuchos::toString(pos);
      if (params_->sublist("Solver").isParameter(label))
        {
        fix_gid_.append(params_->sublist("Solver").get(label,0));
        pos++;
        }
      else
        {
        pos=0;
        }
      }
      
    DEBVAR(fix_gid_);
    
    return 0;
    }

  // Computes all it is necessary to initialize the preconditioner.
  int SchurPreconditioner::Initialize()
    {
    START_TIMER(label_,"Initialize");
    time_->ResetStartTime();
    DEBVAR(myLevel_);
    DEBVAR(maxLevel_);
    
    if (myLevel_==maxLevel_)
      {
      CHECK_ZERO(InitializeCoarse());
      }
    else
      {
      CHECK_ZERO(InitializeSeparatorGroups());
      CHECK_ZERO(InitializeOT());
      CHECK_ZERO(TransformAndDrop());
      CHECK_ZERO(InitializeBlocks());
      CHECK_ZERO(InitializeNextLevel())
      }    

    initialized_=true;
    numInitialize_++;

    timeInitialize_+=time_->ElapsedTime();
    STOP_TIMER(label_,"Initialize");
    return 0;
    }


  // Computes all it is necessary to apply the preconditioner.
  int SchurPreconditioner::Compute()
    {
    START_TIMER(label_,"Compute");
    DEBUG("SchurPreconditioner::Compute()");

    if (!(IsInitialized()))
      {
      // the user should normally call Initialize before Compute
      Tools::Error("HYMLS::SchurPreconditioner not initialized. I'll do it for you.",
        __FILE__,__LINE__);
      EPETRA_CHK_ERR(this->Initialize());
      }

  time_->ResetStartTime();

  if (myLevel_==maxLevel_)
    {
    // drop numerical zeros:
    reducedSchur_ = MatrixUtils::DropByValue(SchurMatrix_, 1.0e-8, MatrixUtils::Absolute);

    for (int i=0;i<fix_gid_.length();i++)
      {
      EPETRA_CHK_ERR(MatrixUtils::PutDirichlet(*reducedSchur_,fix_gid_[i]));
      }

    // compute scaling for reduced Schur
    //EPETRA_CHK_ERR(ComputeScaling(*reducedSchur_,reducedSchurScaLeft_,reducedSchurScaRight_));
    CHECK_ZERO(ComputeScaling(*reducedSchur_,reducedSchurScaLeft_,reducedSchurScaRight_));

    DEBUG("scale matrix");
    EPETRA_CHK_ERR(reducedSchur_->LeftScale(*reducedSchurScaLeft_));
    EPETRA_CHK_ERR(reducedSchur_->RightScale(*reducedSchurScaRight_));
    
    DEBUG("reindex matrix");
    //linearMatrix_ = Teuchos::rcp(&((*reindex_)(const_cast<Epetra_CrsMatrix&>(*SchurMatrix_))),false);
    linearMatrix_ = Teuchos::rcp(&((*reindex_)(*reducedSchur_)),false);

#ifdef STORE_MATRICES
    //MatrixUtils::Dump(*linearMatrix_,"ScaledS2.txt");    
#endif
    
    Teuchos::ParameterList& amesosList=params_->sublist("Solver").sublist("Coarse Solver");
    if (amesosList.get("amesos: solver type","Amesos_Klu")=="Amesos_Mumps")
      {
      if (amesosList.sublist("mumps").isParameter("ICNTL(7)"))
        {
        int ordering=amesosList.sublist("mumps").get("ICNTL(7)",7);
        if (ordering==1)
          {
          // construct a feasible ordering for MUMPS
          HYMLS::MatrixUtils::FillReducingOrdering(*reducedSchur_,pivot_order_,
                params_->sublist("Problem").sublist("Problem Definition"));


#ifdef TESTING
        if (comm_->MyPID()==0)
          {
          std::ofstream ofs("ordering2.txt");
          for (int i=0;i<pivot_order_.size();i++)
            {
            ofs << pivot_order_[i]<<std::endl;
            }
          ofs.close();
          }
#endif

          amesosList.sublist("mumps").set("PermIn",&(pivot_order_[0]));
          }
        }
      }
                
    reducedSchurSolver_= Teuchos::rcp(new Ifpack_Amesos(linearMatrix_.get()));
    IFPACK_CHK_ERR(reducedSchurSolver_->SetParameters(amesosList));
  
  
    int status;
    
    try {
    DEBUG("Initialize direct solver");
      IFPACK_CHK_ERR(reducedSchurSolver_->Initialize());
      } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr,status);
    }
  else
    {
    EPETRA_CHK_ERR(TransformAndDrop());
    
#ifdef STORE_MATRICES_disabled
    // dump a reordering for the Schur-complement (for checking in MATLAB)
    Teuchos::RCP<const RecursiveOverlappingPartitioner>
        sepObject = hid_->Spawn(RecursiveOverlappingPartitioner::LocalSeparators);

    std::ofstream ofs("pS.txt");
    std::ofstream ofs1("pS1.txt");
    std::ofstream ofs2("pS2.txt");
    std::ofstream begI("begI.txt");
    std::ofstream begS("begS.txt");
          
    bool linear_indices=true;
    
    Teuchos::RCP<const Epetra_Map> newMap=map_;
    
    if (linear_indices)
      {
      int myLength = SchurMatrix_->NumMyRows();
      newMap=Teuchos::rcp(new Epetra_Map(-1,myLength,0,*comm_));
      }
      
    int off = 0;
    for (int sd=0;sd<hid_->NumMySubdomains();sd++)
      {
      begI << off << std::endl;
      off = off + hid_->NumInteriorElements(sd);
      }
    begI << off << std::endl;
    
    int offset=0;
          
    for (int sep=0;sep<sepObject->NumMySubdomains();sep++)
      {
      for (int grp=0;grp<sepObject->NumGroups(sep);grp++)
        {
        begS << offset << std::endl;
        offset = offset + sepObject->NumElements(sep,grp);
        //begS << sepObject->LID(sep,grp,0)<<std::endl;
        
        // V-sum nodes
        ofs << newMap->GID(map_->LID(sepObject->GID(sep,grp,0))) << std::endl;
        ofs2 << newMap->GID(map_->LID(sepObject->GID(sep,grp,0))) << std::endl;
        // non-Vsum nodes
        for (int j=1;j<sepObject->NumElements(sep,grp);j++)
          {
          ofs << newMap->GID(map_->LID(sepObject->GID(sep,grp,j))) << std::endl;
          ofs1 << newMap->GID(map_->LID(sepObject->GID(sep,grp,j))) << std::endl;
          }
        }
      }
    
    begS << matrix_->NumMyRows()<<std::endl;
    
    ofs.close();
    ofs1.close();
    ofs2.close();
    begI.close();
    begS.close();
    
//    MatrixUtils::Dump(*matrix_,"SchurPreconditioner.txt",false);
    MatrixUtils::Dump(*matrix_,"SchurPreconditionerReindexed.txt",true);
#endif

  // compute LU decompositions of blocks...
  START_TIMER(label_,"factor blocks");
  for (int i=0;i<blockSolver_.size();i++)
    {
    IFPACK_CHK_ERR(blockSolver_[i]->Compute(*matrix_));
    }
  STOP_TIMER(label_,"factor blocks");

  // extract the Vsum part of the preconditioner (reduced Schur)
  EPETRA_CHK_ERR(reducedSchur_->Import(*matrix_, *vsumImporter_, Insert));

  EPETRA_CHK_ERR(reducedSchur_->FillComplete(*vsumMap_,*vsumMap_));

  //TODO: at this point we would like to throw out numerical zeros, I think.
  //      However, that would mean resetting the pointer and invalidating
  //      the reducedSchurSolver. Possibly it is not necessary, though.
//  reducedSchur_ = MatrixUtils::DropByValue(reducedSchur_, 1.0e-14,MatrixUtils::Absolute);
  
#ifdef STORE_MATRICES
    MatrixUtils::Dump(*reducedSchur_,"Schur"+Teuchos::toString(myLevel_+1)+".txt",true);
#endif  

    }

  // compute solver for reduced Schur
  DEBUG("compute coarse solver");
  int ierr=reducedSchurSolver_->Compute();
  if (ierr!=0)
    {
#ifdef STORE_MATRICES
    Teuchos::RCP<const Epetra_CrsMatrix> dumpMatrix
        = reducedSchur_;
    if (myLevel_==maxLevel_)
      {
//      dumpMatrix = linearMatrix_;
      }
      MatrixUtils::Dump(*dumpMatrix,"BadMatrix"+Teuchos::toString(myLevel_)+".txt");
#endif      
    Tools::Error("factorization returned value "+Teuchos::toString(ierr)+
        " on level "+Teuchos::toString(myLevel_),__FILE__,__LINE__);
    }

  computed_ = true;
  timeCompute_ += time_->ElapsedTime();
  numCompute_++;
  STOP_TIMER(label_,"Compute");
  return 0;
  }

/// private init/compute functions /////

int SchurPreconditioner::InitializeBlocks()
  {
  // get an object with only local separators and remote connected separators:
  Teuchos::RCP<const RecursiveOverlappingPartitioner> sepObject
      = hid_->Spawn(RecursiveOverlappingPartitioner::LocalSeparators);
    
    // number of blocks in this preconditioner (except next Schur-complement).
    // Some blocks may ultimately have 0 rows if they had only one element which
    // is retained as a 'Vsum'-node. That doesn't bother the solver, though.
    int numBlocks=0;
    for (int i=0;i<sepObject->NumMySubdomains();i++)
      {
      numBlocks+=sepObject->NumGroups(i);
      }

#ifdef TESTING
    for (int i=0;i<sepObject->NumMySubdomains();i++)
      {
      for (int grp=0;grp<sepObject->NumGroups(i);grp++)
        {
        if (sepObject->NumElements(i,grp)==0)
          {
          HYMLS::Tools::Warning("there is an empty separator, which is probably dangerous",
            __FILE__,__LINE__);
          }
        }
      }
#endif          
        
  // create an array of solvers for all the diagonal blocks
  blockSolver_.resize(numBlocks);
  int blk=0;
  for (int sep=0;sep<sepObject->NumMySubdomains();sep++)
    {
    for (int grp=0;grp<sepObject->NumGroups(sep);grp++)
      {
      // in the spawned sepObject, each local separator is a group of a subdomain.
      // -1 because we remove one Vsum node from each block
      int numRows=sepObject->NumElements(sep,grp)-1; 
      blockSolver_[blk]=Teuchos::rcp(new 
             Ifpack_DenseContainer(numRows));
      IFPACK_CHK_ERR(blockSolver_[blk]->SetParameters(
              params_->sublist("Solver").sublist("Dense Solver")));
      IFPACK_CHK_ERR(blockSolver_[blk]->Initialize());

      for (int j=0; j<numRows; j++)
        {
        int gid=sepObject->GID(sep,grp,j+1); // skip first element, which is a Vsum
        int LRID = map_->LID(gid);
        blockSolver_[blk]->ID(j) = LRID;
        }
      blk++;
      }
    }
  return 0;  
  }


int SchurPreconditioner::InitializeSeparatorGroups()
  {
  START_TIMER2(label_,"InitializeSeparatorGroups");
  if (subdivideSeparators_)
    {
    int dof=params_->sublist("Problem")
                  .sublist("Problem Definition")
                  .get("Degrees of Freedom",-1);
    if (dof==-1)
      {
      HYMLS::Tools::Error("'Degrees of Freedom' parameter not set!",
              __FILE__,__LINE__);
      }
    int pressure=params_->sublist("Solver").get("Subdivide based on variable",-1);
    if (pressure==-1)
      {
      HYMLS::Tools::Error("'Subdivide based on variable' parameter not set!",
              __FILE__,__LINE__);
      }
    Teuchos::RCP<const RecursiveOverlappingPartitioner> sepObject
        = hid_->Spawn(RecursiveOverlappingPartitioner::Separators);
    Teuchos::RCP<Teuchos::Array<HYMLS::SepNode> > sepList 
        = Teuchos::rcp(new Teuchos::Array<HYMLS::SepNode>(sepObject->NumMyElements()));
    Teuchos::Array<int> connectedPs(2); // typically there are exactly two p-couplings
    for (int sep=0;sep<sepObject->NumMySubdomains();sep++)
      {
      int grp=0; // the standard Separator object has local separators as group 0
                 // of its 'subdomains'
      for (int j=0;j<sepObject->NumElements(sep,grp);j++)
        {
        int lsid = sepObject->LID(sep,grp,j);// local separator ID
        int gid = sepObject->GID(sep,grp,j); // global ID
        int lrid = SchurMatrix_->LRID(gid); // local row ID
        int* indices;
        double* values;
        int len;
        int type;
        EPETRA_CHK_ERR(SchurMatrix_->ExtractMyRowView(lrid,len,values,indices));
        int pos=0;
        connectedPs[0]=-1;
        connectedPs[1]=-1;// will remain there if not connected to P-nodes -> own group
        for (int k=0;k<len;k++)
          {
          int gcid = SchurMatrix_->GCID(indices[k]);
          if (MOD(gcid,dof)==pressure)
            {
            if (abs(values[k])>1.0e-8)
              {
              if (pos==0)
                {
                // distinguish between [+1 -1] and [-1 +1] type p-couplings
                type = values[k]>0? 1:0;
                }
              connectedPs[pos++]=gcid;
              }
            if (pos>=2)
              {
              break;
              }
            }
          }
        SepNode S(gid,connectedPs,type);
        (*sepList)[lsid] = S;
        }
      }
      
    hid_->Spawn(RecursiveOverlappingPartitioner::LocalSeparators,sepList);
    }
  else
    {
    hid_->Spawn(RecursiveOverlappingPartitioner::LocalSeparators);
    }

#ifdef DEBUGGING
    std::ofstream ofs1,ofs2;
    if (myLevel_==1)
      {
      ofs1.open("sep_data.m",std::ios::out);
      ofs2.open("lsep_data.m",std::ios::out);
      }
    else
      {
      ofs1.open("sep_data.m",std::ios::app);
      ofs2.open("lsep_data.m",std::ios::app);
      }
    ofs1<<*(hid_->Spawn(RecursiveOverlappingPartitioner::Separators));
    ofs1.close();
    ofs2<<*(hid_->Spawn(RecursiveOverlappingPartitioner::LocalSeparators));
    ofs2.close();
#endif      

  STOP_TIMER2(label_,"InitializeSeparatorGroups");
  return 0;
  }

int SchurPreconditioner::InitializeOT()
  {
  START_TIMER2(label_,"InitializeOT");

  // create orthogonal transform as a sparse matrix representation
  if (sparseMatrixOT_==Teuchos::null)
    {

    // get an object with only local separators and remote connected separators:
    Teuchos::RCP<const RecursiveOverlappingPartitioner> sepObject
        = hid_->Spawn(RecursiveOverlappingPartitioner::LocalSeparators);

    Epetra_IntSerialDenseVector inds;

    sparseMatrixOT_ = Teuchos::rcp(new
        Epetra_CrsMatrix(Copy,*map_,sepObject->NumInteriorElements(0)));
  
    for (int sep=0;sep<sepObject->NumMySubdomains();sep++)
      {
      // the LocalSeparator object has only local separators, but it may
      // have several groups due to splitting of groups (i.e. for the B-grid,
      // where velocities are grouped depending on how they connect ot the pressures)
      for (int grp=0;grp<sepObject->NumGroups(sep);grp++)
        {
        int len = sepObject->NumElements(sep,grp);
        if ((inds.Length()!=len) && (len>0))
          {
          inds.Size(len);
          }
        for (int j=0;j<len;j++)
          {
          inds[j] = sepObject->GID(sep,grp,j);
          }
        if (len>0)
          {
          int ierr=OT->Construct(*sparseMatrixOT_,inds);
          if (ierr<0) 
            {
            Tools::Warning("Error code "+Teuchos::toString(ierr)+" returned from Epetra call!",
                        __FILE__, __LINE__);                        
            return ierr;
            }
          }
        }      
      }
    
    CHECK_ZERO(sparseMatrixOT_->FillComplete())
    }
#ifdef STORE_MATRICES_disabled
  MatrixUtils::Dump(*sparseMatrixOT_, 
        "Householder_level_"+Teuchos::toString(myLevel_)+".txt",false);
  MatrixUtils::Dump(*sparseMatrixOT_, 
        "Householder_level_"+Teuchos::toString(myLevel_)+"Reindexed.txt",true);
#endif  
  STOP_TIMER2(label_,"InitializeOT");
  return 0;
  }
    
  int SchurPreconditioner::InitializeNextLevel()
    {
  START_TIMER2(label_,"InitializeNextLevel");

    Teuchos::RCP<const RecursiveOverlappingPartitioner>
        sepObject = hid_->Spawn(RecursiveOverlappingPartitioner::LocalSeparators);
    
    int numBlocks = 0;
    for (int i=0;i<sepObject->NumMySubdomains();i++)
      {
      numBlocks+=sepObject->NumGroups(i);
      }
    
    DEBVAR(numBlocks);            
    
    // create a map for the reduced Schur-complement. Note that this is a distributed
    // matrix, in contrast to the other diagonal blocks, so we can't use an Ifpack 
    // container.
    int *MyVsumElements = new int[numBlocks]; // one Vsum per block
    int pos=0;
    for (int sep=0;sep<sepObject->NumMySubdomains();sep++)
      {
      DEBVAR(sep)
      for (int grp = 0 ; grp < sepObject->NumGroups(sep) ; grp++)
        {
        DEBVAR(sepObject->GID(sep,grp,0));
        MyVsumElements[pos++] = sepObject->GID(sep,grp,0);
        }
      }
      
      // sort entries. The reduced Schur-complement is a diagonal block
      // of the matrix, and the row- and column map should be consistent
      // so we can use things like Amesos_Klu to solve it.
      // TODO: what are the implications of reordering it here, in particular
      //       if we want to apply the method recursively?
      //
      Teuchos::ArrayView<int> array(MyVsumElements,numBlocks);
      //TODO: I think we don't need this...
      //std::sort(array.begin(),array.end());
  
      vsumMap_=Teuchos::rcp(new Epetra_Map(-1,numBlocks,MyVsumElements,
                                map_->IndexBase(), map_->Comm()));

      DEBUG(label_);
      DEBVAR(*vsumMap_);

      vsumRhs_ = Teuchos::rcp(new Epetra_MultiVector(*vsumMap_,1));
      vsumSol_ = Teuchos::rcp(new Epetra_MultiVector(*vsumMap_,1));

      // the vsums are still distributed and we must
      // form a correct col map
      vsumColMap_ = MatrixUtils::AllGather(*vsumMap_);

      reducedSchur_=Teuchos::rcp(new 
            Epetra_CrsMatrix(Copy,*vsumMap_,*vsumColMap_,numBlocks));
            
      vsumImporter_=Teuchos::rcp(new Epetra_Import(*vsumMap_,*map_));

      // import sparsity pattern for S2
      // extract the Vsum part of the preconditioner (reduced Schur)
      CHECK_ZERO(reducedSchur_->Import(*matrix_, *vsumImporter_, Insert));
      CHECK_ZERO(reducedSchur_->FillComplete(*vsumMap_,*vsumMap_));

      // drop numerical zeros so that the domain decomposition works
      reducedSchur_=MatrixUtils::DropByValue(reducedSchur_,1.0e-14,MatrixUtils::Absolute);
      
      // I think this is required to make the matrix Ifpack-proof:
      reducedSchur_ = MatrixUtils::RemoveColMap(reducedSchur_);
      CHECK_ZERO(reducedSchur_->FillComplete(*vsumMap_,*vsumMap_));
  
#ifdef TESTING
  this->Visualize("hid_data_deb.m",false);
#endif

  DEBVAR("Create solver for reduced Schur");

  nextLevelParams_ = Teuchos::rcp(new Teuchos::ParameterList(*params_));

  if (myLevel_+1!=maxLevel_)
    {
    if (nextLevelHID_==Teuchos::null)
      {
      nextLevelHID_ = hid_->SpawnNextLevel(reducedSchur_,nextLevelParams_);
      }
    
    // create another level of HYMLS::Solver
    reducedSchurSolver_ = Teuchos::rcp(new
        Solver(reducedSchur_, nextLevelParams_, nextLevelHID_, myLevel_+1));
    }
  else
    {
    // fix pressure on coarsest level:
    for (int i=0;i<fix_gid_.length();i++)
      {
      CHECK_ZERO(MatrixUtils::PutDirichlet(*reducedSchur_,fix_gid_[i]));
      }    
    // create another level of HYMLS::SchurPreconditioner, it will figure out
    // itself that it is a direct solver. Therefore it also doesn't need an
    // OverlappingPartitioner object.
    reducedSchurSolver_= Teuchos::rcp(new
        SchurPreconditioner(reducedSchur_,nextLevelHID_,nextLevelParams_,myLevel_+1));
    }

  IFPACK_CHK_ERR(reducedSchurSolver_->SetParameters(*nextLevelParams_));
    
  DEBUG("Initialize solver for reduced Schur");

  int status;
  try {
    IFPACK_CHK_ERR(reducedSchurSolver_->Initialize());
    } TEUCHOS_STANDARD_CATCH_STATEMENTS(true,std::cerr,status);

  STOP_TIMER2(label_,"InitializeNextLevel");
  return 0;
  }

  int SchurPreconditioner::InitializeCoarse()
    {
    // we can't initialize the direct solver for the sparse matrix right now.
    // We have to reindex it first, and the values (or pattern) may not be there, yet.
    // so we build the solver and call just Initialize() before Compute()
    return 0;
    }

  int SchurPreconditioner::TransformAndDrop()
    {
    START_TIMER(label_,"TransformAndDrop");
#ifdef DEBUGGING    
    MatrixUtils::Dump(*SchurMatrix_,"S"+Teuchos::toString(myLevel_)+".txt",true);
    MatrixUtils::Dump(*sparseMatrixOT_,"H"+Teuchos::toString(myLevel_)+".txt",true);
#endif        
    // currently we simply compute T'*S*T using a sparse matmul.
    // I tried more fancy block-variants, but they were quite tedious
    // to implement and also slower than this.
    // We do not actually drop anything here, that happens automatically
    // by the reducedSchur import and the block solvers.
    if (matrix_==Teuchos::null)
      {
      matrix_=OT->Apply(*sparseMatrixOT_, *SchurMatrix_);
      }
    else
      {
      OT->Apply(*matrix_,*sparseMatrixOT_, *SchurMatrix_);
      }

    EPETRA_CHK_ERR(matrix_->FillComplete());
    STOP_TIMER(label_,"TransformAndDrop");
    return 0;
    }


  // Returns true if the  preconditioner has been successfully initialized, false otherwise.
  bool SchurPreconditioner::IsInitialized() const {return initialized_;}
  
  // Returns true if the  preconditioner has been successfully computed, false otherwise.
  bool SchurPreconditioner::IsComputed() const {return computed_;}

  double SchurPreconditioner::InitializeFlops() const
    {
    // the total number of flops is computed each time InitializeFlops() is
    // called. This is becase I also have to add the contribution from each
    // container.
    double total = flopsInitialize_;
    for (int i=0;i<blockSolver_.size();i++)
      {
      if (blockSolver_[i]!=Teuchos::null)
        {
        total+=blockSolver_[i]->InitializeFlops();
        }
      }
    if (reducedSchurSolver_!=Teuchos::null)
      {
      total += reducedSchurSolver_->InitializeFlops();
      }
    return(total);
    }

  double SchurPreconditioner::ComputeFlops() const
    {
    double total = flopsCompute_;
    for (int i=0;i<blockSolver_.size();i++)
      {
      if (blockSolver_[i]!=Teuchos::null)
        {
        total+=blockSolver_[i]->ComputeFlops();
        }
      }
    if (reducedSchurSolver_!=Teuchos::null)
      {
      total += reducedSchurSolver_->ComputeFlops();
      }
    return(total);
    }

  double SchurPreconditioner::ApplyInverseFlops() const
    {
    double total = flopsApplyInverse_;
    for (int i=0;i<blockSolver_.size();i++)
      {
      if (blockSolver_[i]!=Teuchos::null)
        {
        total+=blockSolver_[i]->ApplyInverseFlops();
        }
      }
    if (reducedSchurSolver_!=Teuchos::null)
      {
      total += reducedSchurSolver_->ApplyInverseFlops();
      }
    return(total);
    }


  // Computes the condition number estimate, returns its value.
  double SchurPreconditioner::Condest(const Ifpack_CondestType CT,
                         const int MaxIters,
                         const double Tol,
                         Epetra_RowMatrix* Matrix)
                         {
                         Tools::Warning("not implemented!",__FILE__,__LINE__);
                         return -1.0; // not implemented.
                         }

  // Returns the computed condition number estimate, or -1.0 if not computed.
  double SchurPreconditioner::Condest() const
    {
    Tools::Warning("not implemented!",__FILE__,__LINE__);
    return -1.0;
    }

  int SchurPreconditioner::Apply(const Epetra_MultiVector& X,
                           Epetra_MultiVector& Y) const
    {
    return matrix_->Apply(X,Y);
    }                           

  // Applies the preconditioner to vector X, returns the result in Y.
  int SchurPreconditioner::ApplyInverse(const Epetra_MultiVector& X,
                           Epetra_MultiVector& Y) const
    {
    START_TIMER(label_,"ApplyInverse");
    numApplyInverse_++;
    time_->ResetStartTime();
    if (!IsComputed())
      {
      return -1;
      }

#ifdef TESTING
if (dumpVectors_)
  {
  MatrixUtils::Dump(*(X(0)),"Precond"+Teuchos::toString(myLevel_)+"_Rhs.txt",true);
  }
#endif

    if (myLevel_==maxLevel_)
      {
#ifdef TESTING
      if (Teuchos::is_null(reducedSchurScaLeft_) ||
          Teuchos::is_null(reducedSchurScaRight_)  )
        {
        Tools::Error("Scaling not available (should be created in Compute())",
                        __FILE__,__LINE__);
        }
#endif      

// there was a bug that when reusing a solver for many solves,
// the memory would fill up (on Hopf, at least). This is solved
// by not using the MV reindex object, although this is of course
// a hotfix.
#define MEMLEAK_BUG

#ifndef MEMLEAK_BUG
      Epetra_MultiVector Xcopy(X.Map(),X.NumVectors());
      // left-scale the rhs
      EPETRA_CHK_ERR(Xcopy.Multiply(1.0,*reducedSchurScaLeft_,X,0.0));
      Epetra_MultiVector& linearRhs = (*reindexMV_)(Xcopy);
      Epetra_MultiVector& linearSol = (*reindexMV_)(Y);
#else
      Epetra_MultiVector linearRhs(*linearMap_,X.NumVectors());
      for (int j=0;j<X.NumVectors();j++)
        {
        for (int i=0;i<X.MyLength();i++)
          {
          linearRhs[j][i]=X[j][i] * (*reducedSchurScaLeft_)[i];
          }
        }
      Epetra_MultiVector linearSol(*linearMap_,X.NumVectors());
#endif           

      for (int i=0;i<fix_gid_.length();i++)
        {
        int lid = X.Map().LID(fix_gid_[i]);
        if (lid>0)
          {
          linearRhs[0][lid]=0.0;
          }
        }

      IFPACK_CHK_ERR(reducedSchurSolver_->ApplyInverse(linearRhs,linearSol));
      // unscale the solution
#ifndef MEMLEAK_BUG
      EPETRA_CHK_ERR(Y.Multiply(1.0,*reducedSchurScaRight_,Y,0.0));
#else
      for (int j=0;j<X.NumVectors();j++)
        {
        for (int i=0;i<X.MyLength();i++)
          {
          Y[j][i]=linearSol[j][i] * (*reducedSchurScaRight_)[i];
          }
        }
#endif      
      }
    else
      {
      // (1) Transform right-hand side, B=OT'*X
      Epetra_MultiVector B(X);
    
      ApplyOT(true,B,&flopsApplyInverse_);

      int numBlocks=blockSolver_.size(); // will be 0 on coarsest level
      
      START_TIMER(label_,"solve blocks");
      for (int blk=0;blk<numBlocks;blk++)
        {
        if (X.NumVectors()!=blockSolver_[blk]->NumVectors())
          {
          blockSolver_[blk]->SetNumVectors(X.NumVectors());
          }
        for (int j = 0 ; j < blockSolver_[blk]->NumRows() ; j++)
          {
          int lid = blockSolver_[blk]->ID(j);
          DEBVAR(B.Map().GID(lid));
          for (int k = 0 ; k < X.NumVectors() ; k++)
            {
            blockSolver_[blk]->RHS(j,k) = B[k][lid];
            }
          }
        // apply the inverse of each block. NOTE: flops occurred
        // in ApplyInverse() of each block are summed up in method
        // ApplyInverseFlops().
        IFPACK_CHK_ERR(blockSolver_[blk]->ApplyInverse());

        // copy back into solution vector Y
        for (int j = 0 ; j < blockSolver_[blk]->NumRows() ; j++)
          {
          int lid = blockSolver_[blk]->ID(j);
          DEBVAR(Y.Map().GID(lid));
          for (int k = 0 ; k < X.NumVectors() ; k++)
            {
            Y[k][lid] = blockSolver_[blk]->LHS(j,k);
            }
          }
        }
      STOP_TIMER(label_,"solve blocks");

      // solve reduced Schur-complement problem

      EPETRA_CHK_ERR(vsumRhs_->Import(B,*vsumImporter_,Insert));
      IFPACK_CHK_ERR(reducedSchurSolver_->ApplyInverse(*vsumRhs_,*vsumSol_));
      EPETRA_CHK_ERR(Y.Export(*vsumSol_,*vsumImporter_,Insert));
      
      // transform back
      ApplyOT(false,Y,&flopsApplyInverse_);
      }

#ifdef TESTING
if (dumpVectors_)
  {
  MatrixUtils::Dump(*(Y(0)),"Precond"+Teuchos::toString(myLevel_)+"_Sol.txt",true);
  dumpVectors_=false;
  }
#endif
      
    timeApplyInverse_+=time_->ElapsedTime();

    STOP_TIMER(label_,"ApplyInverse");
    return 0;
    }

  // Returns a pointer to the matrix to be preconditioned.
  // will cause an error if the Schur-complement has not been
  // explicitly constructed!
  const Epetra_RowMatrix& SchurPreconditioner::Matrix() const 
        {
        return *SchurMatrix_;
        }

  // Returns the number of calls to Initialize().
  int SchurPreconditioner::NumInitialize() const {return numInitialize_;}

  // Returns the number of calls to Compute().
  int SchurPreconditioner::NumCompute() const {return numCompute_;}

  // Returns the number of calls to ApplyInverse().
  int SchurPreconditioner::NumApplyInverse() const {return numApplyInverse_;}

  // Returns the time spent in Initialize().
  double SchurPreconditioner::InitializeTime() const {return timeInitialize_;}

  // Returns the time spent in Compute().
  double SchurPreconditioner::ComputeTime() const {return timeCompute_;}

  // Returns the time spent in ApplyInverse().
  double SchurPreconditioner::ApplyInverseTime() const {return timeApplyInverse_;}
  
  // infinity-norm
  double SchurPreconditioner::NormInf() const 
    {
    if (IsComputed())
      {
      // note: this is not really correct anymore, the 
      // matrix_ has all nonzeros (even those that are 
      // ignored in ApplyInverse();
      //return matrix_->NormInf();
      return -1.0;
      }
    else 
      {
      return -1.0;      
      }
    }

  // Prints basic information on iostream. This function is used by operator<<.
  ostream& SchurPreconditioner::Print(std::ostream& os) const
    {
    os << label_ << std::endl;
    return os;
    }

  // apply orthogonal transforms to a vector v
int SchurPreconditioner::ApplyOT(bool trans, Epetra_MultiVector& v, double* flops) const
  {
  START_TIMER2(label_,"ApplyOT");

  if (!subdivideSeparators_)
    {
    // implementation 1: apply OT to views of blocks of the vector. This is 
    // potentially very fast, but works only if the separators are contiguous
    // in the ordering of S (which they are unless you use "Subdivide Separators")


    // this object has each local separator as a group (without overlap),
    // and remote separators connecting to local subdomains as separators
    // (we only access the local separators here)
    Teuchos::RCP<const RecursiveOverlappingPartitioner> sepObject
          = hid_->Spawn(RecursiveOverlappingPartitioner::LocalSeparators);
  
    if (trans)
      {
      for (int sep=0;sep<sepObject->NumMySubdomains();sep++)
        {
        int len = sepObject->NumInteriorElements(sep);
        
        int gid = sepObject->GID(sep,0,0);
        int lid = map_->LID(gid);
        for (int k=0;k<v.NumVectors();k++)
          {
          Epetra_SerialDenseVector block(View,&(v[k][lid]),len);
          OT->ApplyInverse(block);
          }
        }
      }
    else
      {
      for (int sep=0;sep<sepObject->NumMySubdomains();sep++)
        {
        int len = sepObject->NumInteriorElements(sep);
        int gid = sepObject->GID(sep,0,0);
        int lid = map_->LID(gid);
        for (int k=0;k<v.NumVectors();k++)
          {
          Epetra_SerialDenseVector block(View,&(v[k][lid]),len);
          OT->Apply(block);
          }
        }
      }
              
    //TODO: implement flops-counter
    if (flops!=NULL)
      {
      }
    }
  else
    {
    // implementation 2: just apply the constructed sparse matrix.
    // This makes sure that the OT for the matrix and the vectors are
    // consistent.
    if (sparseMatrixOT_==Teuchos::null)
      {
      HYMLS::Tools::Error("orth. transform not available as matrix!",
                  __FILE__, __LINE__);
      }
    Epetra_MultiVector tmp=v;
    if (trans)
      {
      EPETRA_CHK_ERR(OT->ApplyInverse(v,*sparseMatrixOT_,tmp));
      }
    else
      {
      EPETRA_CHK_ERR(OT->Apply(v,*sparseMatrixOT_,tmp));
      }
      
    if (flops!=NULL)
      {
      //TODO: maak dit algemeen voor alle OTs
      *flops += sparseMatrixOT_->NumGlobalNonzeros() * 4 + v.MyLength();
      }
    }
  STOP_TIMER2(label_,"ApplyOT");
  return 0;
  }

int SchurPreconditioner::ComputeScaling(const Epetra_CrsMatrix& A,
                                        Teuchos::RCP<Epetra_Vector>& sca_left,
                                        Teuchos::RCP<Epetra_Vector>& sca_right)
  {
  START_TIMER2(label_,"ComputeScaling");
  
  if (Teuchos::is_null(sca_left))
    {
    sca_left = Teuchos::rcp(new Epetra_Vector(A.RowMap()));
    }

  if (Teuchos::is_null(sca_right))
    {
    sca_right = Teuchos::rcp(new Epetra_Vector(A.RowMap()));
    }
    
  sca_left->PutScalar(1.0);
  sca_right->PutScalar(1.0);

  Epetra_Vector diagA(A.RowMap());
  
  EPETRA_CHK_ERR(A.ExtractDiagonalCopy(diagA));
  EPETRA_CHK_ERR(diagA.Abs(diagA));

  double a,b;
  //EPETRA_CHK_ERR(diagA.MaxValue(&a));
  //EPETRA_CHK_ERR(diagA.MeanValue(&b));
  a = 1048576;

  for (int i=0;i<diagA.MyLength();i++)
    {
    if (diagA[i]>1e-8)
      {
      //(*sca_left)[i] = 1.0/diagA[i];
      }
    else
      {      
      (*sca_right)[i] = a;
      (*sca_left)[i] = a;
      }
    }
    
  //MatrixUtils::Dump(*sca_left, "left_scale.txt");
  //MatrixUtils::Dump(*sca_right, "right_scale.txt");
  
  STOP_TIMER2(label_,"ComputeScaling");
  return 0;
  } 

void SchurPreconditioner::Visualize(std::string mfilename,bool recurse) const
  {
  if (myLevel_<maxLevel_)
    {
    std::ofstream ofs(mfilename.c_str(),std::ios::app);
    for (int i=0;i<comm_->NumProc();i++)
      {
      if (comm_->MyPID()==i)
        {
        ofs << "% rank "<<comm_->MyPID() << ": "<<Label()<<std::endl;
        ofs << "p{"<<myLevel_<<"}{"<<1+i<<"}.vsums=[";
        for (int j=0;j<vsumMap_->NumMyElements();j++)
          {
          ofs << vsumMap_->GID(j) << " ";
          }
        ofs << "];"<<std::endl;
        }
      comm_->Barrier();
      }
    ofs.close();
    if (recurse)
      {
      Teuchos::RCP<const HYMLS::Solver> hymls =
          Teuchos::rcp_dynamic_cast<const HYMLS::Solver>(reducedSchurSolver_);
      if (!Teuchos::is_null(hymls)) hymls->Visualize(mfilename);
      }
    }
  }
}// namespace
