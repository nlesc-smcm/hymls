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
  }
#endif

  const char* Exception::what() const throw()
    {
    std::stringstream ss;

    ss << "#### HYMLS EXCEPTION ####\n";
    ss << "Error: "<<msg_<<"\n";
    ss << "(in "<<file_<<", line "<<line_<<")\n\n";
    ss << functionStack_ << "\n";
    ss << std::endl;
    ss << "#########################\n";

    // deliberately create a memory leak because I
    // don't get the output otherwise (?)
    std::string* str=new std::string(ss.str());
    const char* output = str->c_str();
    return output;
    }

  Exception::~Exception() throw() {}


}
  
