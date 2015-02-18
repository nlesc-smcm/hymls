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
        main_utils.o 
#        HYMLS_SolverContainer.o \
#        HYMLS_ParallelSparseDirectSolver.o \

LIBNAMEBASE=hymls_${PLAT}
LIBNAME=lib${LIBNAMEBASE}.a

default: main integration_tests main_ifpack

main: main.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o main $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 

main_eigs: main_eigs.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o main_eigs $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 

integration_tests: integration_tests.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o integration_tests $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 

main_direct: main_direct.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o main_direct $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 

main_ifpack: main_ifpack.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o main_ifpack $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 

test: test.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o test $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 		
	
${LIBNAME}: ${OBJ}
	ar vrcs ${LIBNAME} ${OBJ}

%.o: %.C %.H
	${CXX} ${CXX_FLAGS} -c $< ${INCLUDES} -o $@

%.o: %.cpp %.h
	${CXX} ${CXX_FLAGS} -c $< ${INCLUDES} -o $@

clean:
	-rm *.o ${LIBNAME} main main_direct
