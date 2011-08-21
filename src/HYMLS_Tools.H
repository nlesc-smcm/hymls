#ifndef HYMLS_TOOLS_H
#define HYMLS_TOOLS_H

#ifdef DEBUGGING
#ifndef TESTING
#define TESTING
#endif
#endif

#include <mpi.h>

#include "Epetra_Comm.h"
#include <iostream>
#include "Teuchos_RCP.hpp"
#include "EpetraExt_RowMatrixOut.h"

#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_ParameterList.hpp"


#ifdef DEBUGGING
#define TIMING_LEVEL 10
#ifndef FUNCTION_TRACING
#define FUNCTION_TRACING
#endif
#define PRINT_TIMING false
#else
#define TIMING_LEVEL 2
#define PRINT_TIMING false
#endif

#if TIMING_LEVEL>0
#define START_TIMER(s1,s2) HYMLS::Tools::StartTiming(s1+": "+s2);
#define STOP_TIMER(s1,s2) HYMLS::Tools::StopTiming(s1+": "+s2,PRINT_TIMING);
#else
#define START_TIMER(s1,s2)
#define STOP_TIMER(s1,s2)
#endif

#if TIMING_LEVEL>1
#define START_TIMER2(s1,s2) HYMLS::Tools::StartTiming(s1+": "+s2);
#define STOP_TIMER2(s1,s2) HYMLS::Tools::StopTiming(s1+": "+s2,PRINT_TIMING);
#else
#define START_TIMER2(s1,s2) 
#define STOP_TIMER2(s1,s2) 
#endif

#if TIMING_LEVEL>2
#define START_TIMER3(s1,s2) HYMLS::Tools::StartTiming(s1+": "+s2);
#define STOP_TIMER3(s1,s2) HYMLS::Tools::StopTiming(s1+": "+s2,PRINT_TIMING);
#else
#define START_TIMER3(s1,s2) 
#define STOP_TIMER3(s1,s2) 
#endif

#ifdef DEBUGGING
#ifndef DEBUG
#define DEBUG(s) if (HYMLS::Tools::InitializedIO()) HYMLS::Tools::deb() << s << std::endl;
#endif
#ifndef DEBVAR
#define DEBVAR(s) if (HYMLS::Tools::InitializedIO()) HYMLS::Tools::deb() << #s << " = "<<s << std::endl;
#endif
#else
#ifndef DEBUG
#define DEBUG(s) 
#endif
#ifndef DEBVAR
#define DEBVAR(s) 
#endif
#endif
#ifndef CHECK_ZERO
#define CHECK_ZERO(funcall) {int ierr = funcall;\
if (ierr) {\
std::string msg="Trilinos Error "+Teuchos::toString(ierr)+" returned from call "+std::string(#funcall);\
HYMLS::Tools::Error(msg,__FILE__,__LINE__);}\
}
#endif

#ifndef CHECK_TRUE
#define CHECK_TRUE(funcall) {bool berr = funcall;\
if (!berr) {std::cerr<<"Trilinos call "<<#funcall<<" returned false"<<std::endl;}}
#endif
#ifndef CHECK_NONNEG
#define CHECK_NONNEG(funcall) {int ierr = funcall;\
if (ierr<0) {std::cerr<<"Trilinos Error "<<ierr<<" returned from call "<<#funcall<<std::endl;}}
#endif

//! our own 'modulo' function, which behaves like mod in matlab.
//! the C++ built-in '%' operator returns -1%n=-1 and is therefore
//! not very useful for periodic boundaries...
#ifndef MOD
#define MOD(x,y) (((double)(y)==0.0)? (double)(x): ((double)(x) - floor((double)(x)/((double)(y)))*((double)(y))))
#endif

namespace HYMLS
  {
  
  class Tools
    {
  public:
  
   static void InitializeIO(Teuchos::RCP<const Epetra_Comm> comm,
                            Teuchos::RCP<Teuchos::FancyOStream> output=Teuchos::null,
                            Teuchos::RCP<Teuchos::FancyOStream> debug=Teuchos::null)
     {
     traceLevel_=0;
     comm_=comm;
     if (output==Teuchos::null)
       {
       output_stream = Teuchos::rcp(new 
           Teuchos::FancyOStream(Teuchos::rcp(&std::cout,false)));
           
       output_stream->setOutputToRootOnly(0);
       output_stream->setShowProcRank(true);
       }
     else
       {
       output_stream=output;
       }
#ifdef DEBUGGING
     if (debug==Teuchos::null)
       {
       std::string filename="debug"+Teuchos::toString(comm->MyPID())+".txt";
       Teuchos::RCP<std::ofstream> ofs = Teuchos::rcp(new std::ofstream(filename.c_str()));
       debug_stream = Teuchos::rcp(new Teuchos::FancyOStream(ofs));
       }
     else
       {
       debug_stream=debug;
       }
#endif
     Out("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
     Out("@@ START OF HYMLS OUTPUT                    @@");
     Out("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");

     DEBUG("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
     DEBUG("@@ START OF HYMLS DEBUGGING OUTPUT          @@");
     DEBUG("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
     }

   static void InitializeIO_std(Teuchos::RCP<const Epetra_Comm> comm,
                            Teuchos::RCP<std::ostream> output=Teuchos::null,
                            Teuchos::RCP<std::ostream> debug=Teuchos::null)
     {
     Teuchos::RCP<Teuchos::FancyOStream> fancy_out=Teuchos::null;
     Teuchos::RCP<Teuchos::FancyOStream> fancy_deb=Teuchos::null;
     
     if (output!=Teuchos::null)
       {
       fancy_out = Teuchos::rcp(new Teuchos::FancyOStream(output));
       }
#ifdef DEBUGGING
     if (debug!=Teuchos::null)
       {
       fancy_deb = Teuchos::rcp(new Teuchos::FancyOStream(debug));
       }
#endif
     InitializeIO(comm,fancy_out,fancy_deb);
     }
     
   static bool InitializedIO() {return output_stream!=Teuchos::null;}

   static void Out(std::string msg)
     {
     if (!InitializedIO())
       {
       Warning("Output system not initialized!",__FILE__,__LINE__);
       }
     else
       {       
       (*output_stream)  << msg << std::endl;
#ifdef DEBUGGING
        if (output_stream.get()!=debug_stream.get())
          {
          (*debug_stream)  << msg << std::endl;
          }
#endif       
       }
     }
    
   static void Out(const Epetra_RowMatrix& A, std::string filename)
     {
     EpetraExt::RowMatrixToMatlabFile(filename.c_str(),A);
     }
  
    static void Error(std::string msg, const char* file, int line)
      {
      if (!Teuchos::is_null(output_stream))
        {
        output_stream=Teuchos::null;
        }
#ifdef DEBUGGING
      if (!Teuchos::is_null(output_stream))
        {
        debug_stream=Teuchos::null;
        }
#endif
      std::cerr << "HYMLS Error: "<<msg<<std::endl;
      std::cerr << "(in "<<file<<", line "<<line<<")"<<std::endl;
      DEBUG("HYMLS Error: "<<msg);
      DEBUG("(in "<<file<<", line "<<line<<")");
      MPI_Abort(MPI_COMM_WORLD,-1);
      exit(-1);
      }

    static void Warning(std::string msg, const char* file, int line)
      {
      std::cerr << "HYMLS Warning: "<<msg<<std::endl;
      std::cerr << "(in "<<file<<", line "<<line<<")"<<std::endl;
      DEBUG("HYMLS Warning: "<<msg)
      DEBUG("(in "<<file<<", line "<<line<<")");
      }

  //! converts linear index to cartesian subscripts
  static inline void ind2sub(int nx, int ny, int nz, int dof, 
        int idx, int& i, int& j, int& k, int& var)
    {
#ifdef TESTING    
    if ((idx<0)||(idx>=(nx*ny*nz*dof)))
      {
      std::cerr << "dim=["<<nx<<","<<ny<<","<<nz<<"], dof="<<dof<<": ind="<<idx<<std::endl;
      Tools::Error("ind2sub: Index out of range!",__FILE__,__LINE__);
      }
#endif      
    int rem=idx;
    var=MOD(rem,dof);
    rem=(rem-var)/dof;
    i=MOD(rem,nx);
    rem=(rem-i)/nx;
    j=MOD(rem,ny);
    rem=(rem-j)/ny;
    k=MOD(rem,nz);
    }

  //! converts linear index to cartesian subscripts
  static inline void ind2sub(int nx, int ny, int nz, int idx, int& i, int& j, int& k)
    {
    int dummy;
    ind2sub(nx,ny,nz,1,idx,i,j,k,dummy);
    return;
    }

  //! converts cartesian subscripts to linear index
  static inline int sub2ind(int nx, int ny, int nz, int dof, int i, int j, int k, int var)
    {
#ifdef TESTING    
    if ((i<0)||(i>=nx))
      {
      Tools::Error("sub2ind: i-Index out of range!",__FILE__,__LINE__);
      }
    if ((j<0)||(j>=ny))
      {
      Tools::Error("sub2ind: j-Index out of range!",__FILE__,__LINE__);
      }
    if ((k<0)||(k>=nz))
      {
      Tools::Error("sub2ind: k-Index out of range!",__FILE__,__LINE__);
      }
    if ((k<0)||(var>=dof))
      {
      Tools::Error("sub2ind: var-Index out of range!",__FILE__,__LINE__);
      }
#endif      
    return ((k*ny+j)*nx+i)*dof+var;
    }

  //! converts cartesian subscripts to linear index
  static inline int sub2ind(int nx, int ny, int nz, int i, int j, int k)
    {
    return sub2ind(nx,ny,nz,1,i,j,k,0);
    }


  //! split a cartesian box into nparts subdomains in a somehow 'good' way.
  //! This function just tells you how many subdomains there should be in 
  //! every direction.
  static void SplitBox(int nx, int ny, int nz, int nparts, int& npx, int& npy, int& npz)
    {
    // Factor the number of processors into two dimensions. (nprocs = npN*npM)
    double r1,r2,r3,r;//remainder
    double r_min=1e100;

//    DEBUG("SplitBox("<<nx<<","<<ny<<","<<nz<<")");
//    DEBUG(" into "<<nparts<<" domains"<<std::endl);

    // check all possibilities:
    for (int t1=1; t1<=nparts;t1++)
      for (int t2=1;t2<=(int)(nparts/t1);t2++)
        {
        int t3=(int)(nparts/(t1*t2));        
        r1 = abs((double)(nx/t1) - (double)(ny/t2));
        r2 = abs((double)(nx/t1) - (double)(nz/t3));
        r3 = abs((double)(ny/t2) - (double)(nz/t3));
        r=r1+r2+r3;
        
        if (t1*t2*t3==nparts && r<r_min)
          {
//          DEBUG(t1<<"x"<<t2<<"x"<<t3<<" scores "<<r);
          r_min=r;
          npx=t1;
          npy=t2;
          npz=t3;
          }        
        }    
      }
  
  //! start a timer for a specific part of the code
  //! The timing routine keeps track of total time 
  //! and number of calls for each string you put  
  //! in. The string must be the same when you call
  // StopTiming, of course.                        
  static void StartTiming(string label);

  //! stop timing specific part of the code
  static void StopTiming(string label,bool print=false);
  
  //! print timing results
  static void PrintTiming(std::ostream& os);
    
  static Teuchos::RCP<Teuchos::FancyOStream> getOutputStream()
    {
    if (!InitializedIO())
      {
      Error("I/O not initialized!",__FILE__,__LINE__);
      }
    return output_stream;
    }

  static Teuchos::FancyOStream& out()
    {
    if (!InitializedIO())
      {
      Error("I/O not initialized!",__FILE__,__LINE__);
      }
    return *output_stream;
    }

#ifdef DEBUGGING
  static Teuchos::FancyOStream& deb()
    {
    if (!InitializedIO())
      {
      Error("I/O not initialized!",__FILE__,__LINE__);
      }
    return *debug_stream;
    }
#endif
private: 

    static Teuchos::RCP<Teuchos::FancyOStream> output_stream;

    static Teuchos::RCP<Teuchos::FancyOStream> debug_stream;

    //! parameter list for timing individual parts of the code
    static Teuchos::ParameterList timerList_;
    
    //! communicator for timing
    static Teuchos::RCP<const Epetra_Comm> comm_;
    
    //! for function tracing (nice indented output)        
    static int traceLevel_;
    
    //! get intentation string
    static std::string tabstring(int indent)
      {
      std::string ret="";
      for (int i=0;i<std::max(0,indent);i++)
        {
        ret=ret+"  ";
        }
      return ret;
      }
    
    };
  
  }

#endif
