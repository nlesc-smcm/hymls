# general include file for other packages

include ${SHARED_DIR}/fredwubs/make.inc/Makefile.${PLAT}

CXX_FLAGS+=-I${SHARED_DIR}/fredwubs/hymls/src
MY_LIBRARIES+=-L${SHARED_DIR}/fredwubs/hymls/src/ -lhymls_${PLAT}
TRILINOS_LIBRARIES:=-lbelos -lbelosepetra ${TRILINOS_LIBRARIES}
