%module HYMLS
%{
#include "Epetra_CrsMatrix.h"

#include "HYMLS_Preconditioner.hpp"
#include "HYMLS_Solver.hpp"
%}

%include "HYMLS_Preconditioner.hpp"
%include "HYMLS_Solver.hpp"

%extend HYMLS::Preconditioner
{
    Preconditioner(Teuchos::RCP<Epetra_RowMatrix> m, Teuchos::RCP<Teuchos::ParameterList> p)
    {
        return new HYMLS::Preconditioner(m, p);
    }
}

%extend HYMLS::Solver
{
    Solver(Teuchos::RCP<Epetra_RowMatrix> m, HYMLS::Preconditioner &o, Teuchos::RCP<Teuchos::ParameterList> p)
    {
        return new HYMLS::Solver(m, Teuchos::rcp(&o, false), p);
    }

    int ApplyInverse(Teuchos::RCP<Epetra_MultiVector> x, Teuchos::RCP<Epetra_MultiVector> y)
    {
        return self->ApplyInverse(*x, *y);
    }
}
