#ifndef HYMLS_EXCEPTION_H
#define HYMLS_EXCEPTION_H

#include <string>
#include <exception>

namespace HYMLS
  {
  

//! when caught, this exception closes the output streams and
//! if available) prints a function stack. Then it gives     
//! the file and line of code where the error occured and    
//! an error message.                                        
class Exception : public std::exception
  {
  
  public:
  
  //!
  Exception(std::string msg, std::string file, int line) throw();

  //!
  const char* what() const throw();

  //!
  ~Exception() throw();
  
  private:
  
  //!
  std::string msg_;
  //!
  std::string file_;
  //!
  int line_;
  //!
  std::string functionStack_;
  };

}

#endif
