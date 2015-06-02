# general include file for other packages

CXX_FLAGS+=-I${SHARED_DIR}/fredwubs/hymls/src -I${SHARED_DIR}/fredwubs/hymls/src/evp
MY_LIBRARIES+=-L${SHARED_DIR}/fredwubs/hymls/src/ -lhymls_${PLAT}
TRILINOS_LIBRARIES:=-l${TRILIBPREF}belos -l${TRILIBPREF}belosepetra ${TRILINOS_LIBRARIES}
