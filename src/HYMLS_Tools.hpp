#ifndef HYMLS_TOOLS_H
#define HYMLS_TOOLS_H

#include "HYMLS_config.h"

#include <stack>
#include <cstdio>
#include <string>
#include <iosfwd>

#include "Teuchos_RCP.hpp"
#include "Teuchos_FancyOStream.hpp"

class Epetra_Comm;
class Epetra_RowMatrix;
namespace HYMLS { class Epetra_Time; }
namespace Teuchos { class ParameterList; }

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
    Teuchos::RCP<Teuchos::FancyOStream> debug=Teuchos::null);

  static void InitializeIO_std(Teuchos::RCP<const Epetra_Comm> comm,
    Teuchos::RCP<std::ostream> output=Teuchos::null,
    Teuchos::RCP<std::ostream> debug=Teuchos::null);

  // sometime sthe I/O gets killed e.g. by Anasazi, restore it to something that
  // works (currently we set the output stream to std::cout)
  static void RestoreIO();

  static bool InitializedIO();

  static void Out(std::string msg);

  static void Out(const Epetra_RowMatrix& A, std::string filename);

  static void Error(std::string msg, const char* file, int line);

  static void Fatal(std::string msg, const char* file, int line, bool printStack=false);

  static void Warning(std::string msg, const char* file, int line);

  //! converts linear index to cartesian subscripts
  static void ind2sub(int nx, int ny, int nz, int dof,
    hymls_gidx idx, int& i, int& j, int& k, int& var);

  //! converts linear index to cartesian subscripts
  static void ind2sub(int nx, int ny, int nz, hymls_gidx idx, int& i, int& j, int& k);

  //! converts cartesian subscripts to linear index
  static hymls_gidx sub2ind(int nx, int ny, int nz, int dof, int i, int j, int k, int var);

  //! converts cartesian subscripts to linear index
  static hymls_gidx sub2ind(int nx, int ny, int nz, int i, int j, int k);

  static void SignalHandler(int signum);

  //! split a cartesian box into nparts subdomains in a somehow 'good' way.
  //! This function just tells you how many subdomains there should be in
  //! every direction. Returns 1 if no splitting was found.
  static int SplitBox(int nx, int ny, int nz, int nparts, int& ndx, int& ndy, int& ndz,
    int sx = 1, int sy = 1, int sz = 1);

  //! start a timer for a specific part of the code
  //! The timing routine keeps track of total time
  //! and number of calls for each std::string you put
  //! in. The std::string must be the same when you call
  // StopTiming, of course.
  static Teuchos::RCP<Epetra_Time> StartTiming(std::string const &label);

  //! stop timing specific part of the code
  static void StopTiming(std::string const &fname, bool print=false,
    Teuchos::RCP<Epetra_Time> T=Teuchos::null);

  //! start memory profiling a specific part of the code
  static std::tuple<long long, long long> StartMemory(std::string const &label);

  //! stop memory profiling specific part of the code
  static void StopMemory(std::string const &fname, bool print=false,
    long long memory=-1, long long max_memory=-1);

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

  //! print timing results
  static void PrintTiming(std::ostream& os);

  //! report memory usage
  static void PrintMemUsage(std::ostream& os);

  static Teuchos::RCP<Teuchos::FancyOStream> getOutputStream();

  static Teuchos::FancyOStream& out();

  static Teuchos::FancyOStream& deb();
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
  static std::string tabstring(int indent);

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
