#include "HYMLS_Tools.H"
#include "Epetra_Time.h"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"

using namespace Teuchos;

namespace HYMLS {

RCP<const Epetra_Comm> Tools::comm_=null;
ParameterList Tools::timerList_;
RCP<FancyOStream> Tools::output_stream = null;
RCP<FancyOStream> Tools::debug_stream = null;

//////////////////////////////////////////////////////////////////
// Timing functionality                                         //
//////////////////////////////////////////////////////////////////

void Tools::StartTiming(string fname)
  {
  if (!InitializedIO())
    {
    Error("no comm available - cannot time!",__FILE__,__LINE__);
    }
#ifdef FUNCTION_TRACING
#ifdef DEBUGGING
  deb() << "@@@@@ ENTER "<<fname<<" @@@@@"<<std::endl;
#else
  out() << "@@@@@ ENTER "<<fname<<" @@@@@"<<std::endl;
#endif  
#endif    
  RCP<Epetra_Time> T=rcp(new Epetra_Time(*comm_));
  timerList_.sublist("timers").set(fname,T);
  }


void Tools::StopTiming(string fname,bool print)
  {

#ifdef FUNCTION_TRACING
#ifdef DEBUGGING
  deb() << "@@@@@ LEAVE "<<fname<<" @@@@@"<<std::endl;
#else
  out() << "@@@@@ LEAVE "<<fname<<" @@@@@"<<std::endl;
#endif  
#endif    
  
  RCP<Epetra_Time> T = null;
  T=timerList_.sublist("timers").get(fname,T);
  double elapsed=0;
  if (T!=null)
    {
    elapsed=T->ElapsedTime();
    }
  
  int ncalls=timerList_.sublist("number of calls").get(fname,0);
  double total_time=timerList_.sublist("total time").get(fname,0.0);
  timerList_.sublist("number of calls").set(fname,ncalls+1);
  timerList_.sublist("total time").set(fname,total_time+elapsed);
  
  if (print)
    {
    out() << "### timing: "<<fname<<" "<<elapsed<<std::endl;
    }
  
  }

void Tools::PrintTiming(std::ostream& os)
  {
  os << "================================== TIMING RESULTS =================================="<<std::endl;
  os << "     Description                              ";
  os << " # Calls \t Cumulative Time \t Time/call\n";
  os << "=========================================================================================="<<std::endl;
  
  ParameterList& ncallsList=timerList_.sublist("number of calls");
  ParameterList& elapsedList=timerList_.sublist("total time");
  for (ParameterList::ConstIterator i=ncallsList.begin();i!=ncallsList.end();i++)
    {
    const string& fname = i->first;
    int ncalls = ncallsList.get(fname,0);
    double elapsed = elapsedList.get(fname,0.0);
    os << fname << "\t" <<ncalls<<"\t"<<elapsed<<"\t" 
       << ((ncalls>0)? elapsed/(double)ncalls : 0.0) <<std::endl;
    }
  os << "=========================================================================================="<<std::endl;
  }

}
