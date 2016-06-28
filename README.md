# HYMLS

HYMLS is a hybrid direct/iterative solver for the Jacobian of the incompressible Navier-Stokes equations on structured grids. The method is based on domain decomposition, where the variables in the interior of the subdomains are eliminated using a direct method and the Schur-complement on the separators is solved using a robust iterative method. The method features a grid-independent convergence rate and does not break down at high Reynolds numbers. The structure-preserving preconditioning technique allows recursive application of the algorithm resulting a a multilevel solver. 

The implementation of HYMLS was done based on the Epetra package in Trilinos. For this reason, we can easily interface with other packages that are available in Trilinos that perform continuation, eigenvalue computation, etc. 

# Building HYMLS

To build HYMLS, you can use cmake. This allows you to build HYMLS anywhere you like, not only in the src directory. So let's say we make a directory ~/build/hymls and hymls is located in ~/hymls. Then we can build with

```
cd ~/build/hymls
cmake ~/hymls
make
```

Note that HYMLS has to be built with an mpi compiler, and that Trilinos has to be in your PATH so cmake can find it. Those can be set in for instance .bashrc, but in case they are not, one can build with something like

```
cd ~/build/hymls
PATH=$PATH:$HOME/Trilinos/bin cmake ~/hymls
make
```

Instead of adding Trilinos to your path, you can also set it through TRILINOS_HOME.

Building with PHIST is done in the same way. You just add it to your PATH. After this, you can enable support for the PHIST JDQR solver by setting `-DHYMLS_USE_PHIST=On`. A build command may look something like

```
cd ~/build/hymls
PATH=$PATH:$HOME/Trilinos/bin:$HOME/phist/bin cmake -DHYMLS_USE_PHIST=On ~/hymls
make
```

# Installing HYMLS

HYMLS can be installed by calling

```
make install
```

from the build directory. It will be installed in the `CMAKE_INSTALL_PREFIX`. If you want to install HYMLS in `~/local`, the build process might be like this

```
cd ~/build/hymls
PATH=$PATH:$HOME/Trilinos/bin cmake -DCMAKE_INSTALL_PREFIX="${HOME}/local" ~/hymls
make
make test
make install
```

The line "make test" will run all unit and integration tests and exit with a non-zero return value if anything fails.
A more verbose variant is "make check", which runs the same tests but prints the output. The integration tests are run
with 8 MPI processes.
After installing HYMLS, other packages should be able to find it through the cmake config.

# Building the example

The example can be seen as a separate project. After installing HYMLS, the example will be able to find it as long as the install prefix is in your path. So building it could go like

```
cd ~/build/example
PATH=$PATH:${HOME}/local/bin:$HOME/Trilinos/bin cmake ~/hymls/example
make
```

and you can run it with

```
./main ~/hymls/example/params.xml
```