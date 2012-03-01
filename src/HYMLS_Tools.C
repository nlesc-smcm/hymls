#include "HYMLS_Tools.H"
#include "Epetra_Time.h"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

#include <stack>

using namespace Teuchos;

namespace HYMLS {

RCP<const Epetra_Comm> Tools::comm_=null;
ParameterList Tools::timerList_;
RCP<FancyOStream> Tools::output_stream = null;
RCP<FancyOStream> Tools::debug_stream = null;
int Tools::traceLevel_=0;
int Tools::timerCounter_=0;
std::stack<std::string> Tools::functionStack_;

//////////////////////////////////////////////////////////////////
// Timing functionality                                         //
//////////////////////////////////////////////////////////////////

void Tools::StartTiming(string fname)
  {
#ifdef FUNCTION_TRACING
  traceLevel_++;
#ifdef DEBUGGING
  deb() << "@@@@@ "<<tabstring(traceLevel_)<<"ENTER "<<fname<<" @@@@@"<<std::endl;
#endif
  functionStack_.push(fname);
#endif
  if (InitializedIO())
    {
    RCP<Epetra_Time> T=rcp(new Epetra_Time(*comm_));
    if (timerList_.sublist("timer id").isParameter(fname)==false)
      {
      timerCounter_++;
      timerList_.sublist("timer id").set(fname,timerCounter_);
      }  
    timerList_.sublist("timers").set(fname,T);
    }
  }


void Tools::StopTiming(string fname,bool print)
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
  
  RCP<Epetra_Time> T = null;
  T=timerList_.sublist("timers").get(fname,T);
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
#else
  os << "no function stack available, to get one, compile HYMLS with -DFUNCTION_TRACING or -DTESTING"<<std::endl;
#endif
  return os;
  }

}
  
