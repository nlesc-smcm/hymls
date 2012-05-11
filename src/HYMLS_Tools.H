#ifndef HYMLS_TOOLS_H
#define HYMLS_TOOLS_H

#include <mpi.h>
#include <stack>
#include <cstdio>

#include "Epetra_Comm.h"
#include <iostream>
#include "Teuchos_RCP.hpp"
#include "EpetraExt_RowMatrixOut.h"
#include "Epetra_SerialComm.h"

#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_ParameterList.hpp"


#include "HYMLS_Macros.H"
#include "HYMLS_Exception.H"

namespace HYMLS
  {
  
  //! various static functions to do timing, function tracing,
  //! error handling etc.
  class Tools
    {
  public:
  
   friend class Exception;
   
   //! returns the SVN revision number of the hymls directory
   static int Revision();
  
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

     //redirect std::cout
     if (output_stream->getOStream().get()!=&std::cout)
       {
       std::cout.rdbuf(out().rdbuf());
       }

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
      throw HYMLS::Exception(msg,(std::string)file,line);
      }
      
    static void Fatal(std::string msg, const char* file, int line)
      {      
      printFunctionStack(std::cerr);
      PrintMemUsage(std::cerr);
      std::cerr << std::endl;
      std::cerr << "Fatal Error: "<<msg<<std::endl;
      std::cerr << "(in "<<file<<", line "<<line<<")"<<std::endl;
      DEBUG(std::flush);
      out()<<std::flush;
      output_stream=Teuchos::null;
      debug_stream=Teuchos::null;
      
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
    std::string msg1 = "sub2ind: ";
    std::string msg3 = " out of range ";
    if ((i<0)||(i>=nx))
      {
      std::string msg2 = "i-Index "+Teuchos::toString(i);
      std::string msg4 = "[0,"+Teuchos::toString(nx)+"]";
      Tools::Error(msg1+msg2+msg3+msg4,__FILE__,__LINE__);
      }
    if ((j<0)||(j>=ny))
      {
      std::string msg2 = "j-Index "+Teuchos::toString(j);
      std::string msg4 = "[0,"+Teuchos::toString(ny)+"]";
      Tools::Error(msg1+msg2+msg3+msg4,__FILE__,__LINE__);
      }
    if ((k<0)||(k>=nz))
      {
      std::string msg2 = "k-Index "+Teuchos::toString(j);
      std::string msg4 = "[0,"+Teuchos::toString(nz)+"]";
      Tools::Error(msg1+msg2+msg3+msg4,__FILE__,__LINE__);
      }
    if ((var<0)||(var>=dof))
      {
      std::string msg2 = "var-Index "+Teuchos::toString(j);
      std::string msg4 = "[0,"+Teuchos::toString(dof)+"]";
      Tools::Error(msg1+msg2+msg3+msg4,__FILE__,__LINE__);
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
  
  //! report memory usage to a global list. If the same  
  //! label is passed in again, the entry is overwritten 
  //! (it is assumed that the memory has been freed in   
  //! the mean time). 
  static void ReportMemUsage(std::string label, double bytes);
  
  //! report memory usage
  static void PrintMemUsage(std::ostream& os);
    
  static Teuchos::RCP<Teuchos::FancyOStream> getOutputStream()
    {
    if (!InitializedIO())
      {
      Teuchos::RCP<Epetra_SerialComm> comm = Teuchos::rcp(new Epetra_SerialComm());
      Teuchos::RCP<std::ostream> tmp_stream = Teuchos::rcp(&std::cout, false);
      InitializeIO_std(comm,tmp_stream,tmp_stream);
      }
    return output_stream;
    }

  static Teuchos::FancyOStream& out()
    {
    if (!InitializedIO())
      {
      Teuchos::RCP<Epetra_SerialComm> comm = Teuchos::rcp(new Epetra_SerialComm());
      Teuchos::RCP<std::ostream> tmp_stream = Teuchos::rcp(&std::cout, false);
      InitializeIO_std(comm,tmp_stream,tmp_stream);
      }
    return *output_stream;
    }

#ifdef DEBUGGING
  static Teuchos::FancyOStream& deb()
    {
    if (!InitializedIO())
      {
      debug_stream = Teuchos::rcp(new 
           Teuchos::FancyOStream(Teuchos::rcp(&std::cerr,false)));
           
      debug_stream->setOutputToRootOnly(-1);
      debug_stream->setShowProcRank(true);
      }
    return *debug_stream;
    }
#endif
private: 

    static Teuchos::RCP<Teuchos::FancyOStream> output_stream;

    static Teuchos::RCP<Teuchos::FancyOStream> debug_stream;

    //! parameter list for timing individual parts of the code
    static Teuchos::ParameterList timerList_;
    
    //! keeps track of timer numbers (to get the ordering correct)
    static int timerCounter_;
   
    //! parameter list to keep track of memory usage
    static Teuchos::ParameterList memList_;
    
    //! communicator for timing
    static Teuchos::RCP<const Epetra_Comm> comm_;
    
    //! for function tracing (nice indented output)        
    static int traceLevel_;
    
    //! keep track of the function call stack if FUNCTION_TRACING is defined
    static std::stack<string> functionStack_;
    
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
      
    // print the function stack if FUNCTION_TRACING is enabled
    static std::ostream& printFunctionStack(std::ostream& os);
    
    };

//! this object starts a timer when it is constructed and
//! stops it when it is destroyed.
class TimerObject
  {
  public: 
  
  //!
  TimerObject(string s, bool print)
    {
    s_=s;
    print_=print;
    Tools::StartTiming(s_);
    }
  //!
  ~TimerObject()
    {
    Tools::StopTiming(s_,print_);
    }
  private:
  //!
  std::string s_;
  //!
  bool print_;
  };

  
  }

#endif
