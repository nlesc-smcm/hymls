#include "HYMLS_Exception.H"
#include "HYMLS_Tools.H"

namespace HYMLS {

  Exception::Exception(std::string msg, std::string file, int line) throw() :
    msg_(msg),file_(file),line_(line)
    {
    DEBUG("THROWING HYMLS EXCEPTION!");
    DEBVAR(msg);
    DEBVAR(file);
    DEBVAR(line);
    std::stringstream ss;
    Tools::printFunctionStack(ss);
    functionStack_=ss.str();
    }

  const char* Exception::what() const throw ()
    {
    std::stringstream ss;
    ss << "Error: "<<msg_<<"\n";
    ss << "(in "<<file_<<", line "<<line_<<")\n\n";
    ss << functionStack_ << "\n";
    ss << std::endl;

    return ss.str().c_str();
    }

  Exception::~Exception() throw() {}


}
  
