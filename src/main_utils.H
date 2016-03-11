#ifndef HYMLS_MAIN_UTILS_H
#define HYMLS_MAIN_UTILS_H

#include "HYMLS_config.h"

#include <string>
#include "Teuchos_RCP.hpp"

class Epetra_Comm;
class Epetra_Map;
class Epetra_CrsMatrix;
class Epetra_Vector;
class Epetra_MultiVector;

namespace Teuchos {
class ParameterList;
}

namespace HYMLS {
namespace MainUtils {

Teuchos::RCP<Epetra_CrsMatrix> read_matrix(std::string datadir,
                                           std::string file_format,
                                           Teuchos::RCP<Epetra_Map> map,
                                           std::string name="jac");
        
Teuchos::RCP<Epetra_Vector> read_vector(std::string name,
                                        std::string datadir, 
                                        std::string file_format,
                                        Teuchos::RCP<Epetra_Map> map);
                
Teuchos::RCP<Epetra_Map> create_map(const Epetra_Comm& comm,
                                Teuchos::ParameterList& probl_params);

Teuchos::RCP<Epetra_CrsMatrix> create_matrix(const Epetra_Map& map,
                                Teuchos::ParameterList& probl_params,
                                std::string galeriLabel, Teuchos::ParameterList& galeriList);

// given the "Problem" sublist, constructs a null space, for instance:
// "Null Space Type"="Constant", "Constant P" etc.
Teuchos::RCP<Epetra_MultiVector> create_nullspace(const Epetra_CrsMatrix& A, 
                                                        const std::string& nullSpaceType,
                                                        Teuchos::ParameterList& probl_params);

int MakeSystemConsistent(const Epetra_CrsMatrix& A,
                               Epetra_MultiVector& x_ex,
                               Epetra_MultiVector& b,
                               Teuchos::ParameterList& driverList);
                               

}
}

#endif
