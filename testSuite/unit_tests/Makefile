include ${SHARED_DIR}/fredwubs/make.inc/Makefile.${PLAT}
include ${SHARED_DIR}/fredwubs/hymls/src/Makefile.inc

#rev=$(shell svn status -u | tail -1 | sed s/Status\ against\ revision://| sed s/\ //g)
rev=\"\"
CXX_FLAGS+=-DHYMLS_REVISION=${rev}

LIBS=${MY_LIBRARIES} ${TRILINOS_LIBRARIES} ${TPL_LIBRARIES} ${EXTRA_LD_FLAGS}

OBJ=    main.o HYMLS_Tools.o

default: main

main: ${OBJ}
	${CXX} ${CXX_FLAGS} -o main ${OBJ} ${INCLUDES} ${LIBS}

.C.o: $$@.C $$@.H
	${CXX} ${CXX_FLAGS} -c $< ${INCLUDES} -o $@

clean:
	-rm *.o main
