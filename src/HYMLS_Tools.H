#ifndef HYMLS_TOOLS_H
#define HYMLS_TOOLS_H

#include "HYMLS_config.h"

#include <mpi.h>
#include <stack>
#include <cstdio>
#include <signal.h>
#include <iostream>

#include "Epetra_Comm.h"
#include "Epetra_MpiComm.h"
#include "Epetra_SerialComm.h"

#include "EpetraExt_RowMatrixOut.h"

#include "Teuchos_RCP.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_stacktrace.hpp"

#include "Ifpack_ConfigDefs.h"

#include "HYMLS_Epetra_Time.h"
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
   static const char* Revision();
  
   static void InitializeIO(Teuchos::RCP<const Epetra_Comm> comm,
                            Teuchos::RCP<Teuchos::FancyOStream> output=Teuchos::null,
                            Teuchos::RCP<Teuchos::FancyOStream> debug=Teuchos::null)
     {
#ifdef HAVE_TEUCHOS_STACKTRACE
Teuchos::print_stack_on_segfault();
#endif     
     traceLevel_=0;
     if (comm == Teuchos::null)
       comm_ = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
     else
       comm_ = comm;
     if (output==Teuchos::null)
       {
       output_stream = Teuchos::rcp(new 
           Teuchos::FancyOStream(Teuchos::rcp(&std::cout,false)));
           
       output_stream->setOutputToRootOnly(0);
       output_stream->setShowProcRank(false);
       }
     else
       {
       output_stream=output;
       }
#ifdef HYMLS_DEBUGGING
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
       rdbuf_bak = std::cout.rdbuf();
       std::cout.rdbuf(out().rdbuf());
       }
     else if (rdbuf_bak!=NULL && rdbuf_bak!=std::cout.rdbuf())
       {
       std::cout.rdbuf(rdbuf_bak);
       }
/*
     Out("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
     Out("@@ START OF HYMLS OUTPUT                    @@");
     Out("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");

     HYMLS_DEBUG("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
     HYMLS_DEBUG("@@ START OF HYMLS HYMLS_DEBUGGING OUTPUT          @@");
     HYMLS_DEBUG("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");     
*/
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
#ifdef HYMLS_DEBUGGING
     if (debug!=Teuchos::null)
       {
       fancy_deb = Teuchos::rcp(new Teuchos::FancyOStream(debug));
       }
#endif
     InitializeIO(comm,fancy_out,fancy_deb);
     }
   
   // sometime sthe I/O gets killed e.g. by Anasazi, restore it to something that
   // works (currently we set the output stream to std::cout)
   static void RestoreIO()
     {
     output_stream = Teuchos::rcp(new 
         Teuchos::FancyOStream(Teuchos::rcp(&std::cerr,false)));
           
     output_stream->setOutputToRootOnly(0);
     output_stream->setShowProcRank(false);
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
#ifdef HYMLS_DEBUGGING
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
      
    static void Fatal(std::string msg, const char* file, int line, bool printStack=false)
      {
      if (printStack) printFunctionStack(std::cerr);
      //PrintMemUsage(std::cerr);
      //std::cerr << std::endl;
      std::cerr << "Fatal Error: "<<msg<<std::endl;
      std::cerr << "(in "<<file<<", line "<<line<<")"<<std::endl;
      HYMLS_DEBUG(std::flush);
      out()<<std::flush;
      output_stream=Teuchos::null;
      debug_stream=Teuchos::null;

      int initialized = 0;
      MPI_Initialized(&initialized);
      if (initialized)
        MPI_Abort(MPI_COMM_WORLD, -1);
      exit(-1);
      }

    static void Warning(std::string msg, const char* file, int line)
      {
      std::cerr << "HYMLS Warning: "<<msg<<std::endl;
      std::cerr << "(in "<<file<<", line "<<line<<")"<<std::endl;
      HYMLS_DEBUG("HYMLS Warning: "<<msg)
      HYMLS_DEBUG("(in "<<file<<", line "<<line<<")");
      }

    //! converts linear index to cartesian subscripts
    static void ind2sub(int nx, int ny, int nz, int dof, 
      hymls_gidx idx, int& i, int& j, int& k, int& var);

    //! converts linear index to cartesian subscripts
    static void ind2sub(int nx, int ny, int nz, hymls_gidx idx, int& i, int& j, int& k);

    //! converts cartesian subscripts to linear index
    static hymls_gidx sub2ind(int nx, int ny, int nz, int dof, int i, int j, int k, int var);

    //! converts cartesian subscripts to linear index
    static hymls_gidx sub2ind(int nx, int ny, int nz, int i, int j, int k);

static void SignalHandler(int signum)
  {
  std::string msg="Caught signal "+Teuchos::toString(signum);
  if (signum==SIGINT) msg+=" (SIGINT, interrupt)";
  else if (signum==SIGSEGV) msg+=" (SIGSEGV, segmentation fault)";
  HYMLS::Tools::Fatal(msg,__FILE__,__LINE__);
  }

  //! split a cartesian box into nparts subdomains in a somehow 'good' way.
  //! This function just tells you how many subdomains there should be in 
  //! every direction. Returns 1 if no splitting was found.
  static int SplitBox(int nx, int ny, int nz, int nparts, int& ndx, int& ndy, int& ndz, int sx = 1, int sy = 1, int sz = 1)
    {
    // Factor the number of processors into two dimensions. (nprocs = npN*npM)

//    HYMLS_DEBUG("SplitBox("<<nx<<","<<ny<<","<<nz<<")");
//    HYMLS_DEBUG(" into "<<nparts<<" domains"<<std::endl);

    double rmin = 1e100;
    int ret = 1;

    int npx = nx / sx;
    int npy = ny / sy;
    int npz = nz / sz;

    std::string s1 = Teuchos::toString(nx) + "x" + Teuchos::toString(ny) + "x" + Teuchos::toString(nz);
    std::string s2 = Teuchos::toString(npx) + "x" + Teuchos::toString(npy) + "x" + Teuchos::toString(npz);

    // check all possibilities:
    for (int t1 = 1; t1 <= nparts; t1++)
      for (int t2 = 1; t2 <= (int)(nparts / t1); t2++)
        {
        int t3 = (int)(nparts/(t1*t2));

        if (t1 * t2 * t3 == nparts)
          {
          std::string s3 = Teuchos::toString(t1) + "x" + Teuchos::toString(t2) + "x" + Teuchos::toString(t3);
          int my_nx = nx / t1;
          int my_ny = ny / t2;
          int my_nz = nz / t3;
          if ((my_nx * t1 != nx) || (my_ny * t2 != ny) || (my_nz * t3 != nz))
            {
            HYMLS_DEBUG("Can't partition a "+s1+" domain into "+s3+" parts.");
            continue;
            }

          int my_npx = npx / t1;
          int my_npy = npy / t2;
          int my_npz = npz / t3;
          if ((my_npx * sx != my_nx) || (my_npy * sy != my_ny) || (my_npz * sz != my_nz))
            {
            HYMLS_DEBUG("Can't partition "+s2+" domains onto "+s3+" processors.");
            continue;
            }

          double r1 = std::abs((double)nx / (double)t1 - (double)ny / (double)t2);
          double r2 = std::abs((double)nx / (double)t1 - (double)nz / (double)t3);
          double r3 = std::abs((double)ny / (double)t2 - (double)nz / (double)t3);
          double r = r1 + r2 + r3;

          if (r < rmin)
            {
            rmin = r;
            ndx = t1;
            ndy = t2;
            ndz = t3;
            ret = 0;
            }
          }
        }
    return ret;
    }

  //! start a timer for a specific part of the code
  //! The timing routine keeps track of total time 
  //! and number of calls for each std::string you put  
  //! in. The std::string must be the same when you call
  // StopTiming, of course.                        
  static Teuchos::RCP<Epetra_Time> StartTiming(std::string const &label);
  
  //! stop timing specific part of the code
  static void StopTiming(std::string const &fname, bool print=false, Teuchos::RCP<Epetra_Time> T=Teuchos::null);

  //! start memory profiling a specific part of the code
  static std::tuple<long long, long long> StartMemory(std::string const &label);

  //! stop memory profiling specific part of the code
  static void StopMemory(std::string const &fname, bool print=false,
                         long long memory=-1, long long max_memory=-1);

#ifdef HYMLS_DEBUGGING
  //! set breakpoint (does nothing if HYMLS_DEBUGGING is not defined)
  //! function is the name of something that is being timed, e.g.
  //! Solver: ApplyInverse", file and line are __FILE__ and __LINE__
  //! where the bp is set, and msg is a message to be printed. When
  //! the specified function is entered, the program is aborted and
  //! the given info is printed.
  static void SetCheckPoint(std::string function, std::string msg,
        std::string file, int line);

  //! returns true if the breakpoint exists (and fills the args),
  //! false otherwise.
  static bool GetCheckPoint(std::string function, std::string& msg,
        std::string& file, int& line);
#endif
  
  //! print timing results
  static void PrintTiming(std::ostream& os);
  
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

#ifdef HYMLS_DEBUGGING
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
    
    static std::streambuf* rdbuf_bak;

    //! parameter list for timing individual parts of the code
    static Teuchos::ParameterList timerList_;

    //! keeps track of timer numbers (to get the ordering correct)
    static int timerCounter_;

    //! parameter list for setting breakpoints
    static Teuchos::ParameterList breakpointList_;
       
    //! parameter list to keep track of memory usage
    static Teuchos::ParameterList memList_;
    
    //! communicator for timing
    static Teuchos::RCP<const Epetra_Comm> comm_;
    
    //! for function tracing (nice indented output)        
    static int traceLevel_;
    
    //! keep track of the function call stack if HYMLS_FUNCTION_TRACING is defined
    static std::stack<std::string> functionStack_;
    
    //! get intentation std::string
    static std::string tabstring(int indent)
      {
      std::string ret="";
      for (int i=0;i<std::max(0,indent);i++)
        {
        ret=ret+"  ";
        }
      return ret;
      }
      
    // print the function stack if HYMLS_FUNCTION_TRACING is enabled
    static std::ostream& printFunctionStack(std::ostream& os);
    
    };

//! this object starts a timer when it is constructed and
//! stops it when it is destroyed.
class TimerObject
  {
  public:
  
  //!
  TimerObject(std::string const &s, bool print);
  //!
  virtual ~TimerObject();

  private:
  //!
  std::string s_;
  //!
  bool print_;
  //!
  Teuchos::RCP<Epetra_Time> T_;
  //!
  size_t memory_used_;
  //!
  size_t memory_allocated_;
  };

  }

#endif
