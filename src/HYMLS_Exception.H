#ifndef HYMLS_EXCEPTION_H
#define HYMLS_EXCEPTION_H

#include "HYMLS_config.h"

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
  
  //! constructor with message, file name and line number
  Exception(std::string msg, std::string file, int line) throw();

  //! copy constructor
  Exception(const Exception& e)=default;

  //! assignment operator
  Exception& operator=(const Exception& e)=default;

  //! move constructor
  Exception(Exception&& e)=default;
  
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
  //!
  std::string message_;
  };

}

#endif
