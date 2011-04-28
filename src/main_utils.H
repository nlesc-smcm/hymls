#ifndef HYMLS_MAIN_UTILS_H
#define HYMLS_MAIN_UTILS_H

#include <string>
#include "Teuchos_RCP.hpp"

class Epetra_Map;
class Epetra_CrsMatrix;
class Epetra_Vector;

Teuchos::RCP<Epetra_CrsMatrix> read_matrix(std::string datadir, 
                                           std::string file_format, 
                                           Teuchos::RCP<Epetra_Map> map);
        
Teuchos::RCP<Epetra_Vector> read_vector(std::string name,
                                        std::string datadir, 
                                        std::string file_format,
                                        Teuchos::RCP<Epetra_Map> map);
        
        

#endif
