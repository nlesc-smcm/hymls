#include "HYMLS_Exception.H"
#include "HYMLS_Tools.H"

namespace HYMLS {

  Exception::Exception(std::string msg, std::string file, int line) throw()
    {
    msg_=msg;
    file_=file;
    line_=line;
    std::stringstream ss;
    ss<<Tools::printFunctionStack(ss);
    functionStack_=ss.str();
    }

  const char* Exception::what() const throw ()
    {
    std::stringstream ss;
    ss << "Error: "<<msg_<<std::endl;
    ss << "(in "<<file_<<", line "<<line_<<")"<<std::endl;
    ss << std::endl;
    ss << functionStack_ << std::endl;
    ss << std::endl;

    return ss.str().c_str();
    }

  Exception::~Exception() throw() {}


}
  
