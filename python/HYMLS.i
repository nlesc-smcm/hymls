%module(directors = "1") HYMLS
%{
#include "Epetra_CrsMatrix.h"

#include "HYMLS_Tools.hpp"
#include "HYMLS_Preconditioner.hpp"
#include "HYMLS_BorderedSolver.hpp"
#include "HYMLS_Solver.hpp"
#include "HYMLS_CartesianPartitioner.hpp"
#include "HYMLS_SkewCartesianPartitioner.hpp"
#include "HYMLS_Exception.hpp"
%}

%feature("director:except")
{
    if ($error != NULL) {
        throw Swig::DirectorMethodException();
    }
}
%exception
{
    try
    {
        $action
        if (PyErr_Occurred()) SWIG_fail;
    }
    catch(HYMLS::Exception & e)
    {
        PyErr_SetString(PyExc_Exception, e.what());
        SWIG_fail;
    }
    catch (Swig::DirectorException & e)
    {
        SWIG_fail;
    }
}

%include "HYMLS_Tools.hpp"
%include "HYMLS_Preconditioner.hpp"
%include "HYMLS_BorderedSolver.hpp"
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

    int SetBorder(Teuchos::RCP<Epetra_MultiVector> V)
    {
        return self->SetBorder(V);
    }

    int SetBorder(Teuchos::RCP<Epetra_MultiVector> V, Teuchos::RCP<Epetra_MultiVector> W)
    {
        return self->SetBorder(V, W);
    }

    int SetBorder(Teuchos::RCP<Epetra_MultiVector> V, Teuchos::RCP<Epetra_MultiVector> W, Teuchos::RCP<Epetra_SerialDenseMatrix> C)
    {
        return self->SetBorder(V, W, C);
    }

    int ApplyInverse(Teuchos::RCP<Epetra_MultiVector> x, Teuchos::RCP<Epetra_MultiVector> y)
    {
        return self->ApplyInverse(*x, *y);
    }

    int ApplyInverse(Teuchos::RCP<Epetra_MultiVector> x, Teuchos::RCP<Epetra_SerialDenseMatrix> s,
                     Teuchos::RCP<Epetra_MultiVector> y, Teuchos::RCP<Epetra_SerialDenseMatrix> t)
    {
        return self->ApplyInverse(*x, *s, *y, *t);
    }
}

%extend HYMLS::Solver
{
    Solver(Teuchos::RCP<Epetra_Operator> m, HYMLS::Preconditioner &o, Teuchos::RCP<Teuchos::ParameterList> p)
    {
        return new HYMLS::Solver(m, Teuchos::rcp(&o, false), p);
    }

    int ApplyInverse(Teuchos::RCP<Epetra_MultiVector> x, Teuchos::RCP<Epetra_MultiVector> y)
    {
        return self->ApplyInverse(*x, *y);
    }
}

%extend HYMLS::BorderedSolver
{
    BorderedSolver(Teuchos::RCP<Epetra_Operator> m, Teuchos::RCP<Epetra_Operator> o, Teuchos::RCP<Teuchos::ParameterList> p)
    {
        return new HYMLS::BorderedSolver(m, o, p);
    }

    BorderedSolver(Teuchos::RCP<Epetra_Operator> m, HYMLS::Preconditioner &o, Teuchos::RCP<Teuchos::ParameterList> p)
    {
        return new HYMLS::BorderedSolver(m, Teuchos::rcp(&o, false), p);
    }

    int SetBorder(Teuchos::RCP<Epetra_MultiVector> V)
    {
        return self->SetBorder(V);
    }

    int SetBorder(Teuchos::RCP<Epetra_MultiVector> V, Teuchos::RCP<Epetra_MultiVector> W)
    {
        return self->SetBorder(V, W);
    }

    int SetBorder(Teuchos::RCP<Epetra_MultiVector> V, Teuchos::RCP<Epetra_MultiVector> W, Teuchos::RCP<Epetra_SerialDenseMatrix> C)
    {
        return self->SetBorder(V, W, C);
    }

    int UnsetBorder()
    {
        return self->SetBorder(Teuchos::null);
    }

    void SetOperator(Teuchos::RCP<Epetra_Operator> m)
    {
        self->SetOperator(m);
    }

    void SetMassMatrix(Teuchos::RCP<Epetra_RowMatrix> m)
    {
        self->SetMassMatrix(m);
    }

    int ApplyInverse(Teuchos::RCP<Epetra_MultiVector> x, Teuchos::RCP<Epetra_MultiVector> y)
    {
        return self->ApplyInverse(*x, *y);
    }

    int ApplyInverse(Teuchos::RCP<Epetra_MultiVector> x, Teuchos::RCP<Epetra_SerialDenseMatrix> s,
                     Teuchos::RCP<Epetra_MultiVector> y, Teuchos::RCP<Epetra_SerialDenseMatrix> t)
    {
        return self->ApplyInverse(*x, *s, *y, *t);
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

%exception;
