# HYMLS

HYMLS is a hybrid direct/iterative solver for the Jacobian of the incompressible Navier-Stokes equations on structured grids. The method is based on domain decomposition, where the variables in the interior of the subdomains are eliminated using a direct method and the Schur-complement on the separators is solved using a robust iterative method. The method features a grid-independent convergence rate and does not break down at high Reynolds numbers. The structure-preserving preconditioning technique allows recursive application of the algorithm resulting a a multilevel solver. 

The implementation of HYMLS was done based on the Epetra package in Trilinos. For this reason, we can easily interface with other packages that are available in Trilinos that perform continuation, eigenvalue computation, etc. 

# Building HYMLS

To build HYMLS, you can use cmake. This allows you to build HYMLS anywhere you like, not only in the src directory. So let's say we make a directory ~/build/hymls and hymls is located in ~/hymls/src. Then we can build with

```
cd ~/build/hymls
cmake ~/hymls/src
make
```

Note that HYMLS has to be built with an mpi compiler, and that Trilinos has to be in your PATH so cmake can find it. Those can be set in for instance .bashrc, but in case they are not, one can build with something like

```
cd ~/build/hymls
CXX=mpicxx PATH=$PATH:$HOME/Trilinos cmake ~/hymls/src
make
```

Instead of adding Trilinos to your path, you can also set it through TRILINOS_HOME.

Building with PHIST is done in the same way. You just add it to your PATH. HYMLS will then automatically enable support for the PHIST JDQR solver, and build a libarry, hymls_jada, that can be used by other packages, for instance FVM.
