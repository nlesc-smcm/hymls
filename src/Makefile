include ${SHARED_DIR}/fredwubs/make.inc/Makefile.${PLAT}
include Makefile.inc

#rev=$(shell svn status -u | tail -1 | sed s/Status\ against\ revision://| sed s/\ //g)
rev="\"$(shell svnversion -n)\""
CXX_FLAGS+=-DHYMLS_REVISION=${rev} -fPIC

LIBS=${MY_LIBRARIES} ${TRILINOS_LIBRARIES} ${TPL_LIBRARIES} ${EXTRA_LD_FLAGS}

OBJ=    HYMLS_Solver.o \
        HYMLS_Preconditioner.o \
        HYMLS_SchurComplement.o \
        HYMLS_SchurPreconditioner.o \
        HYMLS_OverlappingPartitioner.o \
        HYMLS_HierarchicalMap.o \
        HYMLS_BaseCartesianPartitioner.o \
        HYMLS_CartesianPartitioner.o \
        HYMLS_StandardNodeClassifier.o \
        HYMLS_CartesianStokesClassifier.o \
        HYMLS_SepNode.o \
        HYMLS_HyperCube.o \
        HYMLS_MatrixUtils.o \
        HYMLS_DenseUtils.o \
        HYMLS_ProjectedOperator.o \
        HYMLS_BorderedLU.o \
        HYMLS_BelosEpetraOperator.o \
        HYMLS_Epetra_Time.o \
        HYMLS_View_MultiVector.o \
        HYMLS_EpetraExt_ProductOperator.o \
        HYMLS_SparseDirectSolver.o \
        HYMLS_Householder.o \
        HYMLS_AugmentedMatrix.o \
        HYMLS_Tools.o \
        HYMLS_Tester.o \
        HYMLS_PLA.o \
        HYMLS_Exception.o \
        NOX_Epetra_LinearSystem_Hymls.o \
        GaleriExt_Periodic.o \
        main_utils.o \

# Normal library name
LIBNAMEBASE=hymls_${PLAT}
LIBNAME=lib${LIBNAMEBASE}.a

# Name for the eigenvalue library
LIBEIGSNAMEBASE=${LIBNAMEBASE}
LIBEIGSNAME=${LIBNAME}

# Variable used for the phist dependent library rules
LIBPHISTNAMEBASE=hymls_${PLAT}_eigs
LIBPHISTNAME=lib${LIBPHISTNAMEBASE}.a

# Change the eigenvalue library name if we have phist
ifeq (${HAVE_PHIST}, 1)
	LIBEIGSNAMEBASE=${LIBPHISTNAMEBASE}
	LIBEIGSNAME=${LIBPHISTNAME}
endif

default: main integration_tests main_ifpack

main: main.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o main $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 

main_eigs: main_eigs.C ${LIBEIGSNAME}
	${CXX} ${CXX_FLAGS} -o main_eigs $< ${INCLUDES} -L. -l${LIBEIGSNAMEBASE} ${LIBS} 

main_eigs_full: main_eigs_full.C ${LIBEIGSNAME}
	${CXX} ${CXX_FLAGS} -o main_eigs_full $< ${INCLUDES} -L. -l${LIBEIGSNAMEBASE} ${LIBS} 

integration_tests: integration_tests.C ${LIBEIGSNAME}
	${CXX} ${CXX_FLAGS} -o integration_tests $< ${INCLUDES} -L. -l${LIBEIGSNAMEBASE} ${LIBS} 

main_direct: main_direct.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o main_direct $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 

main_ifpack: main_ifpack.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o main_ifpack $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 

test: test.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o test $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 

${LIBNAME}: ${OBJ}
	ar vrcs ${LIBNAME} ${OBJ}

${LIBPHISTNAME}: ${LIBNAME} libjadaCorrectionSolver_impl.so
	ar vrcsP ${LIBPHISTNAME} ${LIBNAME} libjadaCorrectionSolver_impl.so

libjadaCorrectionSolver_impl.so:
	mpicxx -Wall -O3 -shared -fPIC -fopenmp ${CXX_FLAGS} -Wl,-soname,\
	${SHARED_DIR}/fredwubs/hymls/src/libjadaCorrectionSolver_impl.so \
	-o libjadaCorrectionSolver_impl.so \
	evp/jadaCorrectionSolver_impl.C -ldl ${LIBS}

%.o: %.C %.H
	${CXX} ${CXX_FLAGS} -c $< ${INCLUDES} -o $@

%.o: %.cpp %.h
	${CXX} ${CXX_FLAGS} -c $< ${INCLUDES} -o $@

clean:
	-rm *.o ${LIBNAME} main main_direct main_eigs main_eigs_full
