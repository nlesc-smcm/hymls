%module HYMLS
%{
#include "Epetra_CrsMatrix.h"

#include "HYMLS_Tools.hpp"
#include "HYMLS_Preconditioner.hpp"
#include "HYMLS_Solver.hpp"
#include "HYMLS_CartesianPartitioner.hpp"
#include "HYMLS_SkewCartesianPartitioner.hpp"
%}

%include "HYMLS_Tools.hpp"
%include "HYMLS_Preconditioner.hpp"
%include "HYMLS_Solver.hpp"
%include "HYMLS_CartesianPartitioner.hpp"
%include "HYMLS_SkewCartesianPartitioner.hpp"

%extend HYMLS::Tools
{
    static void InitializeIO(Teuchos::RCP<Epetra_Comm> c)
    {
        HYMLS::Tools::InitializeIO(c);
    }
}

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

%extend HYMLS::CartesianPartitioner
{
    CartesianPartitioner(Teuchos::RCP<Teuchos::ParameterList> p, Teuchos::RCP<Epetra_Comm> c)
    {
        return new HYMLS::CartesianPartitioner(Teuchos::null, p, *c);
    }

    Teuchos::RCP<Epetra_Map> Map()
    {
        return Teuchos::rcp(new Epetra_Map(*self->GetMap()));
    }
}

%extend HYMLS::SkewCartesianPartitioner
{
    SkewCartesianPartitioner(Teuchos::RCP<Teuchos::ParameterList> p, Teuchos::RCP<Epetra_Comm> c)
    {
        return new HYMLS::SkewCartesianPartitioner(Teuchos::null, p, *c);
    }

    Teuchos::RCP<Epetra_Map> Map()
    {
        return Teuchos::rcp(new Epetra_Map(*self->GetMap()));
    }
}
