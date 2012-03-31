include Makefile.inc

LIBS=${MY_LIBRARIES} ${TRILINOS_LIBRARIES} ${TPL_LIBRARIES} ${EXTRA_LD_FLAGS}

OBJ=    HYMLS_Solver.o \
        HYMLS_Preconditioner.o \
        HYMLS_SchurComplement.o \
        HYMLS_SchurPreconditioner.o \
        HYMLS_OverlappingPartitioner.o \
        HYMLS_RecursiveOverlappingPartitioner.o \
        HYMLS_CartesianPartitioner.o \
        HYMLS_SepNode.o \
        HYMLS_MatrixUtils.o \
        HYMLS_DenseUtils.o \
        HYMLS_ProjectedOperator.o \
        HYMLS_BorderedLU.o \
        HYMLS_View_MultiVector.o \
        HYMLS_SparseDirectSolver.o \
        HYMLS_Householder.o \
        HYMLS_Tools.o \
        HYMLS_PLA.o \
        HYMLS_Exception.o \
        NOX_Epetra_LinearSystem_Hymls.o \
        GaleriExt_Periodic.o \
        main_utils.o \

LIBNAMEBASE=hymls_${PLAT}
LIBNAME=lib${LIBNAMEBASE}.a

default: main 

main: main.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o main $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 

main_direct: main_direct.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o main_direct $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 

main_ifpack: main_ifpack.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o main_ifpack $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 

test: test.C ${LIBNAME}
	${CXX} ${CXX_FLAGS} -o test $< ${INCLUDES} -L. -l${LIBNAMEBASE} ${LIBS} 
	

	
${LIBNAME}: ${OBJ}
	ar vrcs ${LIBNAME} ${OBJ}

%.o: %.C %.H
	${CXX} ${CXX_FLAGS} -c $< ${INCLUDES} 

%.o: %.cpp %.h
	${CXX} ${CXX_FLAGS} -c $< ${INCLUDES} 

clean:
	-rm *.o ${LIBNAME} main main_direct
