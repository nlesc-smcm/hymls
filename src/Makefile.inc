# general include file for other packages

include ../../mrilucpp/src/Makefile.inc

CXX_FLAGS+=-I${BASEDIR}/hymls/src
MY_LIBRARIES+=-L${BASEDIR}/hymls/ -lhymls_${PLAT}
TRILINOS_LIBRARIES:=-lbelos -lbelosepetra ${TRILINOS_LIBRARIES}
