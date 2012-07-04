#ifndef HYMLS_MAIN_UTILS_H
#define HYMLS_MAIN_UTILS_H

#include <string>
#include "Teuchos_RCP.hpp"

class Epetra_Comm;
class Epetra_Map;
class Epetra_CrsMatrix;
class Epetra_Vector;

namespace Teuchos {
class ParameterList;
}

namespace HYMLS {
namespace MainUtils {

Teuchos::RCP<Epetra_CrsMatrix> read_matrix(std::string datadir, 
                                           std::string file_format, 
                                           Teuchos::RCP<Epetra_Map> map);
        
Teuchos::RCP<Epetra_Vector> read_vector(std::string name,
                                        std::string datadir, 
                                        std::string file_format,
                                        Teuchos::RCP<Epetra_Map> map);
                
Teuchos::RCP<Epetra_Map> create_map(const Epetra_Comm& comm,
                                Teuchos::ParameterList& probl_params);

Teuchos::RCP<Epetra_CrsMatrix> create_matrix(const Epetra_Map& map,
                                Teuchos::ParameterList& probl_params,
                                std::string galeriLabel, Teuchos::ParameterList& galeriList);

}
}

#endif
