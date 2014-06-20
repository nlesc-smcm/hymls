#include "HYMLS_Tools.H"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

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
#pragma omp critical (HYMLS_Timing)
{
#ifdef FUNCTION_TRACING
  traceLevel_++;
  functionStack_.push(fname);
#ifdef DEBUGGING
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
}
    return T;
  }


void Tools::StopTiming(std::string const &fname, bool print, RCP<Epetra_Time> T)
  {
#pragma omp critical (HYMLS_Timing)
{
#ifdef FUNCTION_TRACING
  // when an exception or other error is encountered,
  // the function printFunctionStack() may be called,
  // which deletes the stack. In that case, stop     
  // function tracing.
  if (functionStack_.size()>0)
    {
    functionStack_.pop();
#ifdef DEBUGGING
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
  }

void Tools::PrintTiming(std::ostream& os)
  {

  ParameterList& idList=timerList_.sublist("timer id");
  ParameterList& ncallsList=timerList_.sublist("number of calls");
  ParameterList& elapsedList=timerList_.sublist("total time");

  os << "================================== TIMING RESULTS =================================="<<std::endl;
  os << "     Description                              ";
  os << " # Calls \t Cumulative Time \t Time/call\n";
  os << "=========================================================================================="<<std::endl;

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
    os << fname << "\t" <<ncalls<<"\t"<<elapsed<<"\t" 
       << ((ncalls>0)? elapsed/(double)ncalls : 0.0) <<std::endl;
    }
  os << "=========================================================================================="<<std::endl;
  }

void Tools::ReportMemUsage(std::string label, double bytes)
  {
  memList_.set(label,bytes);
  }
  
void Tools::PrintMemUsage(std::ostream& os)
  {
  os << "=================================== MEMORY USAGE ==================================="<<std::endl;
  double total = 0.0;
  double value;
  std::string unit, label;
  for (ParameterList::ConstIterator i=memList_.begin();i!=memList_.end();i++)
    {
    label = i->first;
    unit = "B";
    value=memList_.get(label,0.0);
    total += value;
    if (value > 1.0e3) {value*=1.0e-3; unit="kB";}
    if (value > 1.0e3) {value*=1.0e-3; unit="MB";}
    if (value > 1.0e3) {value*=1.0e-3; unit="GB";}
    if (value > 1.0e3) {value*=1.0e-3; unit="TB";}
    os << label << "\t" <<value<<"\t"<<unit<<"\n"; 
    }
  os << "===================================================================================="<<std::endl;  
  value = total; unit="B";
  label = "TOTAL";
    if (value > 1.0e3) {value*=1.0e-3; unit="kB";}
    if (value > 1.0e3) {value*=1.0e-3; unit="MB";}
    if (value > 1.0e3) {value*=1.0e-3; unit="GB";}
    if (value > 1.0e3) {value*=1.0e-3; unit="TB";}
    os << label << "\t" <<value<<"\t"<<unit<<"\n"; 
  os << "===================================================================================="<<std::endl;  
  return;
  }

std::ostream& Tools::printFunctionStack(std::ostream& os)
  {
#ifdef FUNCTION_TRACING
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
  os << "no function stack available, to get one, compile HYMLS with -DFUNCTION_TRACING or -DTESTING"<<std::endl;
#endif
  return os;
  }

#ifdef DEBUGGING
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
}
