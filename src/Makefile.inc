# general include file for other packages

include ${HOME}/fredwubs/mrilucpp/src/Makefile.inc

CXX_FLAGS+=-I${BASEDIR}/hymls/src
MY_LIBRARIES+=-L${BASEDIR}/hymls/src/ -lhymls_${PLAT}
TRILINOS_LIBRARIES:=-lbelos -lbelosepetra ${TRILINOS_LIBRARIES}
