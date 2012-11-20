include ${SHARED_DIR}/fredwubs/make.inc/Makefile.${PLAT}
include Makefile.inc

rev=$(shell head -n 4 .svn/entries |tail -n 1)
CXX_FLAGS+=-DHYMLS_REVISION=${rev}

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
        HYMLS_View_MultiVector.o \
        HYMLS_EpetraExt_ProductOperator.o \
        HYMLS_SparseDirectSolver.o \
        HYMLS_Householder.o \
        HYMLS_AugmentedMatrix.o \
        HYMLS_Tools.o \
        HYMLS_PLA.o \
        HYMLS_Exception.o \
        NOX_Epetra_LinearSystem_Hymls.o \
        GaleriExt_Periodic.o \
        main_utils.o \

LIBNAMEBASE=hymls_${PLAT}
LIBNAME=lib${LIBNAMEBASE}.a

default: main unit_tests

main: main.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o main $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 

unit_tests: unit_tests.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o unit_tests $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 

main_direct: main_direct.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o main_direct $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 

main_ifpack: main_ifpack.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o main_ifpack $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 

test: test.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o test $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 		
	
${LIBNAME}: ${OBJ}
	ar vrcs ${LIBNAME} ${OBJ}

HYMLS_Tools.o: .svn/entries

%.o: %.C %.H
	${CXX} ${CXX_FLAGS} -c $< ${INCLUDES} 

%.o: %.cpp %.h
	${CXX} ${CXX_FLAGS} -c $< ${INCLUDES} 

clean:
	-rm *.o ${LIBNAME} main main_direct
