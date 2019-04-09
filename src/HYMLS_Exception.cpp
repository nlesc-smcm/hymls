#include "HYMLS_Exception.hpp"

#include "HYMLS_Tools.hpp"
#include "HYMLS_Macros.hpp"

namespace HYMLS {

  Exception::Exception(std::string msg, std::string file, int line) throw() :
    msg_(msg),file_(file),line_(line)
    {
    HYMLS_DEBUG("THROWING HYMLS EXCEPTION!");
    HYMLS_DEBVAR(msg);
    HYMLS_DEBVAR(file);
    HYMLS_DEBVAR(line);
    {
    std::stringstream ss;
    Tools::printFunctionStack(ss);
    functionStack_=ss.str();
    }
    {
    std::stringstream ss;
    ss << "#### HYMLS EXCEPTION ####\n";
    ss << "Error: "<<msg_<<"\n";
    ss << "(in "<<file_<<", line "<<line_<<")\n\n";
    ss << functionStack_ << "\n";
    ss << std::endl;
    ss << "#########################\n";
    message_=ss.str();
    }

    }

#if 0
// copy constructor
Exception::Exception(const Exception& e)
  {
    msg_=e.msg_;
    file_=e.file_;
    line_=e.line_;
    functionStack_=e.functionStack_;
  }

// move constructor
Exception::Exception(Exception&& e)
  {
    msg_=e.msg_;
    file_=e.file_;
    line_=e.line_;
    functionStack_=e.functionStack_;
    std::stringstream ss;

    ss << "#### HYMLS EXCEPTION ####\n";
    ss << "Error: "<<msg_<<"\n";
    ss << "(in "<<file_<<", line "<<line_<<")\n\n";
    ss << functionStack_ << "\n";
    ss << std::endl;
    ss << "#########################\n";

    // deliberately create a memory leak because I
    // don't get the output otherwise (?)
    message_=ss.str();
  }
#endif

  const char* Exception::what() const throw()
    {
//    std::string* memory_leak=new std::string(message_);
//    return memory_leak->c_str();
    return message_.c_str();
    }

  Exception::~Exception() throw() {}


}
  
