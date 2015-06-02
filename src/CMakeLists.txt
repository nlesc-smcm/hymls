cmake_minimum_required(VERSION 2.8)
project(HYMLS)

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
  message(STATUS "Using GNU compiler flags")

  if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_C_FLAGS        "${CMAKE_C_FLAGS} -O2 -g -DTESTING")
    set(CMAKE_CXX_FLAGS      "${CMAKE_CXX_FLAGS} -O2 -g -DTESTING")
    set(CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} -O2 -g -DTESTING")
  endif()

  set(CMAKE_C_FLAGS_RELEASE        "-O3 -ffast-math -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE      "-O3 -ffast-math -march=native")
  set(CMAKE_Fortran_FLAGS_RELEASE  "-O3 -ffast-math -march=native")

  set (CMAKE_C_FLAGS_DEBUG       "-O0 -g -DTESTING -DDEBUGGING")
  set (CMAKE_CXX_FLAGS_DEBUG     "-O0 -g -DTESTING -DDEBUGGING")
  set (CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -DTESTING -DDEBUGGING")

  set (CMAKE_C_FLAGS_RELWITHDEBINFO       "${CMAKE_C_FLAGS_RELEASE}       -g")
  set (CMAKE_CXX_FLAGS_RELWITHDEBINFO     "${CMAKE_CXX_FLAGS_RELEASE}     -g")
  set (CMAKE_Fortran_FLAGS_RELWITHDEBINFO "${CMAKE_Fortran_FLAGS_RELEASE} -g")
elseif (CMAKE_C_COMPILER_ID STREQUAL "Intel")
  message(STATUS "Using Intel compiler flags")

  if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_C_FLAGS        "${CMAKE_C_FLAGS} -O2 -g -mkl -DTESTING")
    set(CMAKE_CXX_FLAGS      "${CMAKE_CXX_FLAGS} -O2 -g -mkl -DTESTING")
    set(CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} -O2 -g -mkl -DTESTING")
  endif()

  set(CMAKE_C_FLAGS_RELEASE        "-O3 -mkl -ffast-math -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE      "-O3 -mkl -ffast-math -march=native")
  set(CMAKE_Fortran_FLAGS_RELEASE  "-O3 -mkl -ffast-math -march=native")

  set (CMAKE_C_FLAGS_DEBUG       "-O0 -g -mkl -DTESTING -DDEBUGGING")
  set (CMAKE_CXX_FLAGS_DEBUG     "-O0 -g -mkl -DTESTING -DDEBUGGING")
  set (CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -mkl -DTESTING -DDEBUGGING")

  set (CMAKE_C_FLAGS_RELWITHDEBINFO       "${CMAKE_C_FLAGS_RELEASE}       -g")
  set (CMAKE_CXX_FLAGS_RELWITHDEBINFO     "${CMAKE_CXX_FLAGS_RELEASE}     -g")
  set (CMAKE_Fortran_FLAGS_RELWITHDEBINFO "${CMAKE_Fortran_FLAGS_RELEASE} -g")

  add_definitions(-DUSE_MKL)
endif()

set(SOURCES HYMLS_Solver
    HYMLS_Preconditioner
    HYMLS_SchurComplement
    HYMLS_SchurPreconditioner
    HYMLS_OverlappingPartitioner
    HYMLS_HierarchicalMap
    HYMLS_BaseCartesianPartitioner
    HYMLS_CartesianPartitioner
    HYMLS_StandardNodeClassifier
    HYMLS_CartesianStokesClassifier
    HYMLS_SepNode
    HYMLS_HyperCube
    HYMLS_MatrixUtils
    HYMLS_DenseUtils
    HYMLS_ProjectedOperator
    HYMLS_BorderedLU
    HYMLS_BelosEpetraOperator
    HYMLS_Epetra_Time
    HYMLS_View_MultiVector
    HYMLS_EpetraExt_ProductOperator
    HYMLS_SparseDirectSolver
    HYMLS_Householder
    HYMLS_AugmentedMatrix
    HYMLS_Tools
    HYMLS_Tester
    HYMLS_PLA
    HYMLS_Exception
    NOX_Epetra_LinearSystem_Hymls
    GaleriExt_Periodic
    main_utils)

include_directories(${PROJECT_SOURCE_DIR})

if (DEFINED ENV{TRILINOS_HOME})
  set(Trilinos_DIR "$ENV{TRILINOS_HOME}/lib/cmake/Trilinos")
endif()
find_package(Trilinos)

include_directories(${Trilinos_INCLUDE_DIRS})
include_directories(${Trilinos_TPL_INCLUDE_DIRS})

link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${Trilinos_TPL_LIBRARY_DIRS})

find_package(PHIST)

if (${PHIST_FOUND})
  add_definitions(-DHAVE_PHIST)
  include_directories(${PHIST_INCLUDE_DIRS})
  link_directories(${PHIST_LIBRARY_DIRS})
endif()

# HYMLS version
execute_process(COMMAND svnversion -n OUTPUT_VARIABLE rev)
add_definitions(-DHYMLS_REVISION="${rev}")

add_library(hymls ${SOURCES})

target_link_libraries(hymls ${Trilinos_LIBRARIES})
target_link_libraries(hymls ${Trilinos_TPL_LIBRARIES})

add_library(hymls_eigs SHARED)
if (${PHIST_FOUND})
  add_library(jada SHARED evp/jadaCorrectionSolver_impl)
  target_link_libraries(jada hymls)
  target_link_libraries(jada ${PHIST_LIBRARIES})

  set_target_properties(hymls PROPERTIES COMPILE_FLAGS "-fPIC")

  target_link_libraries(hymls_eigs jada)
endif()
target_link_libraries(hymls_eigs hymls)

add_executable(main main.C)
target_link_libraries(main hymls)

add_executable(main_eigs_full main_eigs_full.C)
target_link_libraries(main_eigs_full hymls_eigs)

add_executable(integration_tests integration_tests.C)
target_link_libraries(integration_tests hymls_eigs)

if (DEFINED ENV{PLAT})
  # Install to the source directory for now because fvm expects this
  set_target_properties(hymls PROPERTIES OUTPUT_NAME hymls_$ENV{PLAT})
  install(TARGETS hymls ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR} RENAME hymls_$ENV{PLAT})
endif()
