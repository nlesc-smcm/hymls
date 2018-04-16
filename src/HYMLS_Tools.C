#include "HYMLS_Tools.H"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include <dlfcn.h>

#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <stack>

using namespace Teuchos;

//overwrite printf to make e.g. SuperLU_DIST write to our streams
extern "C" {
int printf (const char *fmt, ...)
  {
  std::string fmt_string(fmt);
  
  char formatted_string[fmt_string.length()+10000];

    va_list args;
 
    va_start(args,fmt);
 
    int stat = vsprintf(formatted_string, fmt, args);
 
    va_end(args);
    HYMLS::Tools::out() << formatted_string;
    return stat;
  }
}//extern "C"

namespace HYMLS {

RCP<const Epetra_Comm> Tools::comm_=null;
ParameterList Tools::timerList_;
ParameterList Tools::breakpointList_;
ParameterList Tools::memList_;
RCP<FancyOStream> Tools::output_stream = null;
RCP<FancyOStream> Tools::debug_stream = null;
int Tools::traceLevel_=0;
int Tools::timerCounter_=0;
std::stack<std::string> Tools::functionStack_;
std::streambuf* Tools::rdbuf_bak = std::cout.rdbuf();

//////////////////////////////////////////////////////////////////
// Timing functionality                                         //
//////////////////////////////////////////////////////////////////

const char* Tools::Revision()
  {
  return HYMLS_REVISION;
  }

RCP<Epetra_Time> Tools::StartTiming(std::string const &fname)
  {
  RCP<Epetra_Time> T=null;
#ifdef HYMLS_FUNCTION_TRACING
  traceLevel_++;
  functionStack_.push(fname);
#ifdef HYMLS_DEBUGGING
  deb() << "@@@@@ "<<tabstring(traceLevel_)<<"ENTER "<<fname<<" @@@@@"<<std::endl;
  std::string msg;
  std::string file;
  int line;
  if (GetCheckPoint(fname,msg,file,line))
    {
    Tools::out()<<"reached breakpoint: '"+msg+"' in "+fname<<std::endl;
    Tools::out()<<"(set in "<<file<<", line "<<line<<")\n";
    Tools::deb()<<"reached breakpoint: '"+msg+"' in "+fname<<std::endl;
    Tools::deb()<<"(set in "<<file<<", line "<<line<<")\n";
    }
#endif
#endif
  if (InitializedIO())
    {
    T=rcp(new Epetra_Time(*comm_));
    if (timerList_.sublist("timer id").isParameter(fname)==false)
      {
      timerCounter_++;
      timerList_.sublist("timer id").set(fname,timerCounter_);
      }  
    timerList_.sublist("timers").set(fname,T);
    }
    return T;
  }

size_t (*getMem)() = NULL;
size_t (*getMaxMem)() = NULL;

size_t Tools::StartMemory(std::string const &fname)
  {
  long long out = 0;
#ifdef HYMLS_MEMORY_PROFILING

  if (!getMem)
    {
    getMem = (size_t (*)())dlsym(RTLD_DEFAULT, "get_memory_usage");
    getMaxMem = (size_t (*)())dlsym(RTLD_DEFAULT, "get_max_memory_usage");
    }
  if (!getMem)
    {
    getMem = [](){ return (size_t)0; };
    getMaxMem = [](){ return (size_t)0; };
    Tools::Warning("Memory profiler not loaded correctly", __FILE__, __LINE__);
    }

  if (InitializedIO())
    {
    out = getMem();
    memList_.sublist("memory").set(fname, out);
    }
#endif
  return out;
  }

void Tools::StopTiming(std::string const &fname, bool print, RCP<Epetra_Time> T)
  {
#ifdef HYMLS_FUNCTION_TRACING
  // when an exception or other error is encountered,
  // the function printFunctionStack() may be called,
  // which deletes the stack. In that case, stop     
  // function tracing.
  if (functionStack_.size()>0)
    {
    functionStack_.pop();
#ifdef HYMLS_DEBUGGING
    deb() << "@@@@@ "<<tabstring(traceLevel_)<<"LEAVE "<<fname<<" @@@@@"<<std::endl;
#endif
    }
  traceLevel_--;
#endif
  if (T == null)
    {
    T=timerList_.sublist("timers").get(fname,T);
    }
  double elapsed=0;
  if (T!=null)
    {
    elapsed=T->ElapsedTime();
    int ncalls=timerList_.sublist("number of calls").get(fname,0);
    double total_time=timerList_.sublist("total time").get(fname,0.0);
    timerList_.sublist("number of calls").set(fname,ncalls+1);
    timerList_.sublist("total time").set(fname,total_time+elapsed);

    if (print)
      {
      out() << "### timing: "<<fname<<" "<<elapsed<<std::endl;
      }
    }
  }

std::string mem2string(long long value)
  {
  std::string unit = "B";
  if (std::abs(value) > 1.0e3) {value*=1.0e-3; unit="kB";}
  if (std::abs(value) > 1.0e3) {value*=1.0e-3; unit="MB";}
  if (std::abs(value) > 1.0e3) {value*=1.0e-3; unit="GB";}
  if (std::abs(value) > 1.0e3) {value*=1.0e-3; unit="TB";}

  std::ostringstream ss;
  ss << value << " " << unit;
  return ss.str();
  }

void Tools::StopMemory(std::string const &fname, bool print, long long start_memory)
  {
#ifdef HYMLS_MEMORY_PROFILING
  if (start_memory < 0)
    start_memory = memList_.sublist("memory").get(fname, (long long)-1);

  if (start_memory < 0)
    return;

  long long memory = getMem() - start_memory;

  long long total_memory = memList_.sublist("total memory").get(fname, (long long)0);
  memList_.sublist("total memory").set(fname, total_memory + memory);

  long long maximum_memory = memList_.sublist("maximum memory").get(fname, (long long)0);
  memList_.sublist("maximum memory").set(fname, std::max(maximum_memory, memory));

  int ncalls = memList_.sublist("number of calls").get(fname, 0);
  memList_.sublist("number of calls").set(fname, ncalls + 1);

  if (print)
    {
    out() << "### memory: " << fname << " "<< mem2string(memory) << std::endl;
    }
#endif
  }

template<typename charT, typename traits = std::char_traits<charT> >
class center_helper
  {
  std::basic_string<charT, traits> str_;

public:
  center_helper(std::basic_string<charT, traits> str)
    :
    str_(str)
    {}

  template<typename a, typename b>
  friend std::basic_ostream<a, b>& operator<<(std::basic_ostream<a, b>& s, const center_helper<a, b>& c);
  };

template<typename charT, typename traits = std::char_traits<charT> >
center_helper<charT, traits> centered(std::basic_string<charT, traits> str)
  {
  return center_helper<charT, traits>(str);
  }

center_helper<std::string::value_type, std::string::traits_type> centered(const std::string& str) {
  return center_helper<std::string::value_type, std::string::traits_type>(str);
  }

template<typename charT, typename traits>
std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& s, const center_helper<charT, traits>& c)
  {
  s << std::right;
  std::streamsize w = s.width();
  if (w > static_cast<std::streamsize>(c.str_.length()))
    {
    std::streamsize left = (w + c.str_.length()) / 2;
    s.width(left);
    s << c.str_;
    s.width(w - left);
    s << "";
    }
  else
    {
    s << c.str_;
    }
  return s;
  }

void Tools::PrintTiming(std::ostream& os)
  {

  ParameterList& idList=timerList_.sublist("timer id");
  ParameterList& ncallsList=timerList_.sublist("number of calls");
  ParameterList& elapsedList=timerList_.sublist("total time");

  os << std::setfill('=') << std::setw(100) << centered(" TIMING RESULTS ") << std::endl;
  os << std::setfill(' ') << std::setw(100-17*3) << std::left << "Description"
     << std::setfill(' ') << std::setw(17) << std::left << "# Calls"
     << std::setfill(' ') << std::setw(17) << std::left << "Cumulative Time"
     << std::setfill(' ') << std::setw(17) << std::left << "Time/call"
     << std::endl;
  os << std::setfill('=') << std::setw(100) << "" << std::endl;

  // first construct a correctly sorted list according to timer ID
  Teuchos::ParameterList sortedList;
  for (ParameterList::ConstIterator i=ncallsList.begin();i!=ncallsList.end();i++)
    {
    const string& fname = i->first;
    int id = idList.get(fname,0);
    std::stringstream label;
    label << std::setw(6) << std::setfill('0') << id << " " << fname;
    sortedList.set(label.str(),fname);
    }

  for (ParameterList::ConstIterator i=sortedList.begin();i!=sortedList.end();i++)
    {
    const string& label = i->first;
    string fname = sortedList.get(label,"bad label");
    int ncalls = ncallsList.get(fname,0);
    double elapsed = elapsedList.get(fname,0.0);
    os << std::setfill(' ') << std::setw(100-17*3) << std::left << fname
       << std::setfill(' ') << std::setw(17) << std::left << ncalls
       << std::setfill(' ') << std::setw(17) << std::left << elapsed
       << std::setfill(' ') << std::setw(17) << std::left
       << (ncalls > 0 ? elapsed/(double)ncalls : 0.0)
       << std::endl;
    }
  os << std::setfill('=') << std::setw(100) << "" << std::endl;
  }

void Tools::PrintMemUsage(std::ostream& os)
  {
#ifdef HYMLS_MEMORY_PROFILING
  os << std::setfill('=') << std::setw(100) << centered(" MEMORY USAGE ") << std::endl;
  os << std::setfill(' ') << std::setw(100-17*3) << std::left << "Description"
     << std::setfill(' ') << std::setw(17) << std::left << "# Calls"
     << std::setfill(' ') << std::setw(17) << std::left << "Maximum Usage"
     << std::setfill(' ') << std::setw(17) << std::left << "Average Usage"
     << std::endl;
  os << std::setfill('=') << std::setw(100) << "" << std::endl;

  for (auto &i: memList_.sublist("total memory"))
    {
    std::string label = i.first;
    long long total = memList_.sublist("total memory").get(label, (long long)0);
    long long maximum = memList_.sublist("maximum memory").get(label, (long long)0);
    int ncalls = memList_.sublist("number of calls").get(label, 1);
    os << std::setfill(' ') << std::setw(100-17*3) << std::left << label
       << std::setfill(' ') << std::setw(17) << std::left << ncalls
       << std::setfill(' ') << std::setw(17) << std::left << mem2string(maximum)
       << std::setfill(' ') << std::setw(17) << std::left
       << mem2string(ncalls > 0 ? total / ncalls : 0)
       << std::endl;
    }
  os << std::setfill('=') << std::setw(100) << centered(" MAX MEMORY USAGE ") << std::endl;
  os << "Total: " << mem2string(getMaxMem()) << "\n";
  os << std::setfill('=') << std::setw(100) << "" << std::endl;
#endif
  return;
  }

std::ostream& Tools::printFunctionStack(std::ostream& os)
  {
#ifdef HYMLS_FUNCTION_TRACING
  if (functionStack_.size()>0)
    {
    os << "FUNCTION STACK:"<<std::endl;
    while (1)
      {
      os << functionStack_.top() << std::endl;
      functionStack_.pop();
      if (functionStack_.size()==0) break;
      }
    }
#elif defined(HAVE_TEUCHOS_STACKTRACE)
    os << "FUNCTION STACK:"<<std::endl;
    os << Teuchos::get_stacktrace()<<std::endl;
#else
  os << "no function stack available, to get one, compile HYMLS with -DHYMLS_FUNCTION_TRACING or -DHYMLS_TESTING"<<std::endl;
#endif
  return os;
  }

void Tools::ind2sub(int nx, int ny, int nz, int dof, 
  hymls_gidx idx, int& i, int& j, int& k, int& var)
  {
#ifdef HYMLS_TESTING
  if (idx < 0 || idx >= (hymls_gidx)nx * ny * nz * dof)
    {
    std::cerr << "dim=["<<nx<<","<<ny<<","<<nz<<"], dof="<<dof<<": ind="<<idx<<std::endl;
    Tools::Error("ind2sub: Index out of range!",__FILE__,__LINE__);
    }
#endif
  hymls_gidx rem = idx;
  var = rem % dof;
  rem = rem / dof;
  i = rem % nx;
  rem = rem / nx;
  j = rem % ny;
  rem = rem / ny;
  k = rem % nz;
  }

  //! converts linear index to cartesian subscripts
void Tools::ind2sub(int nx, int ny, int nz, hymls_gidx idx, int& i, int& j, int& k)
  {
  int dummy;
  ind2sub(nx,ny,nz,1,idx,i,j,k,dummy);
  return;
  }

  //! converts cartesian subscripts to linear index
hymls_gidx Tools::sub2ind(int nx, int ny, int nz, int dof, int i, int j, int k, int var)
  {
#ifdef HYMLS_TESTING
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
  return (((hymls_gidx)k*ny+j)*nx+i)*dof+var;
  }

//! converts cartesian subscripts to linear index
hymls_gidx Tools::sub2ind(int nx, int ny, int nz, int i, int j, int k)
  {
  return sub2ind(nx,ny,nz,1,i,j,k,0);
  }

#ifdef HYMLS_DEBUGGING
void Tools::SetCheckPoint(std::string fname, std::string msg,
        std::string file, int line)
        {
        breakpointList_.sublist(fname).set("msg",msg);
        breakpointList_.sublist(fname).set("file",file);
        breakpointList_.sublist(fname).set("line",line);
        }

bool Tools::GetCheckPoint(std::string fname, std::string& msg,
        std::string& file, int& line)
        {
        if (breakpointList_.isSublist(fname))
          {
          msg=breakpointList_.sublist(fname).get("msg",msg);
          file=breakpointList_.sublist(fname).get("file",file);
          line=breakpointList_.sublist(fname).get("line",line);
          return true;
          }
        return false;
        }
#endif


TimerObject::TimerObject(std::string const &s, bool print)
  {
  s_=s;
  print_=print;
  T_=Tools::StartTiming(s);
  M_=Tools::StartMemory(s);
  }

TimerObject::~TimerObject()
  {
  Tools::StopTiming(s_, print_, T_);
  Tools::StopMemory(s_, print_, M_);
  }
}
