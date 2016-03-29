cmake_minimum_required(VERSION 2.8)
project(HYMLS CXX)

list (APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")
if (CMAKE_MAJOR_VERSION GREATER 2)
  set (CMAKE_CXX_STANDARD 11)
  set(PHIST_HAVE_CXX11 ON)
endif()

option(HYMLS_STORE_MATRICES "dump all matrices, maps etc. encountered (huge overhead)" OFF)
option(HYMLS_DEBUGGING "turns on verbose debugging output" OFF)
option(HYMLS_FUNCTION_TRACING "turns on very verbose output on every function entered/left" OFF)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(STATUS "Using GNU compiler flags")

  if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_C_FLAGS        "${CMAKE_C_FLAGS} -O2 -g -Wall -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS      "${CMAKE_CXX_FLAGS} -O2 -g -Wall -Wno-unknown-pragmas")
    set(CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} -O2 -g -Wall -Wno-unknown-pragmas")
  endif()

  set(BUILD_SHARED_LIBS ON)

  set(CMAKE_C_FLAGS_RELEASE        "-O3 -ffast-math -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE      "-O3 -ffast-math -march=native")
  set(CMAKE_Fortran_FLAGS_RELEASE  "-O3 -ffast-math -march=native")

  set (CMAKE_C_FLAGS_DEBUG       "-O0 -g")
  set (CMAKE_CXX_FLAGS_DEBUG     "-O0 -g")
  set (CMAKE_Fortran_FLAGS_DEBUG "-O0 -g")

  set (CMAKE_C_FLAGS_RELWITHDEBINFO       "${CMAKE_C_FLAGS_RELEASE}       -g")
  set (CMAKE_CXX_FLAGS_RELWITHDEBINFO     "${CMAKE_CXX_FLAGS_RELEASE}     -g")
  set (CMAKE_Fortran_FLAGS_RELWITHDEBINFO "${CMAKE_Fortran_FLAGS_RELEASE} -g")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  message(STATUS "Using Intel compiler flags")

  if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_C_FLAGS        "${CMAKE_C_FLAGS} -O2 -g -mkl")
    set(CMAKE_CXX_FLAGS      "${CMAKE_CXX_FLAGS} -O2 -g -mkl")
    set(CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} -O2 -g -mkl")
  endif()

  set(CMAKE_C_FLAGS_RELEASE        "-O3 -mkl -ffast-math -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE      "-O3 -mkl -ffast-math -march=native")
  set(CMAKE_Fortran_FLAGS_RELEASE  "-O3 -mkl -ffast-math -march=native")

  set (CMAKE_C_FLAGS_DEBUG       "-O0 -g -mkl")
  set (CMAKE_CXX_FLAGS_DEBUG     "-O0 -g -mkl")
  set (CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -mkl")

  set (CMAKE_C_FLAGS_RELWITHDEBINFO       "${CMAKE_C_FLAGS_RELEASE}       -g")
  set (CMAKE_CXX_FLAGS_RELWITHDEBINFO     "${CMAKE_CXX_FLAGS_RELEASE}     -g")
  set (CMAKE_Fortran_FLAGS_RELWITHDEBINFO "${CMAKE_Fortran_FLAGS_RELEASE} -g")

  set(HYMLS_USE_MKL 1 CACHE STRING "use MKL ParDiSo for direct solves")
else()
  message (WARNING "No custom compiler flags for the ${CMAKE_CXX_COMPILER_ID} compiler were found.")
endif()

find_package(MPI REQUIRED)
if (${MPI_FOUND})
  set (CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_CXX_COMPILE_FLAGS})
  set (CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_CXX_LINK_FLAGS})
  set (CMAKE_C_COMPILE_FLAGS ${CMAKE_C_COMPILE_FLAGS} ${MPI_C_COMPILE_FLAGS})
  set (CMAKE_C_LINK_FLAGS ${CMAKE_C_LINK_FLAGS} ${MPI_C_LINK_FLAGS})
  set (CMAKE_Fortran_COMPILE_FLAGS ${CMAKE_Fortran_COMPILE_FLAGS} ${MPI_Fortran_COMPILE_FLAGS})
  set (CMAKE_Fortran_LINK_FLAGS ${CMAKE_Fortran_LINK_FLAGS} ${MPI_Fortran_LINK_FLAGS})
# For now, just include everything
  include_directories (${MPI_INCLUDE_PATH})
  include_directories (${MPI_C_INCLUDE_PATH})
  include_directories (${MPI_CXX_INCLUDE_PATH})
  include_directories (${MPI_Fortran_INCLUDE_PATH})
else()
  message(WARNING "could not find MPI. Trying to compile anyway, presuming the compiler/linker knows where to find it.")
endif()
set(MPIEXEC "mpirun" CACHE STRING "")
set(MPIEXEC_NUMPROC_FLAG "-np" CACHE STRING "")


set(HYMLS_SOURCE HYMLS_Solver
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
    HYMLS_MatrixBlock
    NOX_Epetra_LinearSystem_Hymls
    GaleriExt_Periodic
    main_utils)

set(HYMLS_EIGS_SOURCE evp/dummy.cpp)

# Headers include eachother, so for now just assume we want to install all
# header files that have a source
set(HEADERS)
set(header_extensions ".h" ".H" ".hpp")
foreach(i ${HYMLS_SOURCE})
  foreach(j ${header_extensions})
    if (EXISTS "${PROJECT_SOURCE_DIR}/${i}${j}")
      list(APPEND HEADERS "${i}${j}")
    endif()
  endforeach()
endforeach()
list(APPEND HEADERS "${CMAKE_CURRENT_BINARY_DIR}/HYMLS_config.h")
unset(header_extensions)

# Manually add headers without source file
set(HEADERS ${HEADERS}
  evp/AnasaziHymlsAdapter.hpp
  evp/AnasaziJacobiDavidson.hpp
  evp/AnasaziJacobiDavidsonSolMgr.hpp
  evp/AnasaziPhistSolMgr.hpp
  evp/AnasaziPreconditionerTraits.hpp
  GaleriExt_Cross2DN.h
  GaleriExt_Darcy2D.h
  GaleriExt_Darcy3D.h
  GaleriExt_Star3D.h
  HYMLS_BaseNodeClassifier.H
  HYMLS_BasePartitioner.H
  HYMLS_BorderedSolver.H
  HYMLS_Macros.H
  HYMLS_no_debug.H
  HYMLS_OrthogonalTransform.H
  HYMLS_RestrictedOT.H
  HYMLS_ShiftedOperator.H
  HYMLS_MatrixBlock.H
  )
  
include_directories(${PROJECT_SOURCE_DIR})

if (DEFINED ENV{TRILINOS_HOME})
  set(Trilinos_DIR "$ENV{TRILINOS_HOME}/lib/cmake/Trilinos")
endif()
find_package(Trilinos)

include_directories(${Trilinos_INCLUDE_DIRS})
include_directories(${Trilinos_TPL_INCLUDE_DIRS})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/evp")

link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${Trilinos_TPL_LIBRARY_DIRS})

if (HYMLS_USE_PHIST)
  find_package(phist QUIET CONFIG)
  if (phist_DIR MATCHES "phist_DIR-NOT-FOUND")
    message(WARNING "phist was requested but not found")
    set(HYMLS_USE_PHIST OFF)
  endif()
endif()

# HYMLS version
execute_process(COMMAND git rev-parse --short --sq HEAD WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} OUTPUT_VARIABLE rev)
STRING(REPLACE "'" "\"" HYMLS_REVISION ${rev})

set(HYMLS_TIMING_LEVEL 2 CACHE STRING "which function should be timed/traced (0: none, 1: major functions ... 99: all")


if (CMAKE_BUILD_TYPE MATCHES "Debug")

  message(STATUS "enabling TESTING")
  set(HYMLS_TESTING 1)

  if (HYMLS_STORE_MATRICES)
    message(STATUS "enabling STORE_MATRICES output")
  endif()

  if (HYMLS_DEBUGGING)
    message(STATUS "enabling DEBUGGING output")
  endif()
  if (HYMLS_FUNCTION_TRACING)
    message(STATUS "enabling FUNCTION_TRACING output")
  endif()

endif()

add_library(hymls ${HYMLS_SOURCE})

target_link_libraries(hymls ${Trilinos_LIBRARIES})
target_link_libraries(hymls ${Trilinos_TPL_LIBRARIES})
target_link_libraries(hymls ${MPI_CXX_LIBRARIES})

if (${phist_FOUND})
  message(STATUS "phist found in ${PHIST_INCLUDE_DIRS}")
  include_directories(${PHIST_INCLUDE_DIRS})
  link_directories(${PHIST_LIBRARY_DIRS})
endif()


# create header file to tell the source code about some config-time settings and features it can use
configure_file(${PROJECT_SOURCE_DIR}/HYMLS_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/HYMLS_config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})


# phist is typically compiled with OpenMP support, add the openmp flag to be able to link.
# And even if we compile without, I think our Trilinos or MPI installation require it, so
# set -fopenmp if this is the GNU compiler.
if (HYMLS_USE_OPENMP)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    target_link_libraries(hymls gomp)
  elseif (CMAKE_C_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")  
  endif()
endif()

add_library(hymls_eigs ${HYMLS_EIGS_SOURCE})
if (${phist_FOUND})
  add_library(hymls_jada evp/jadaCorrectionSolver_impl)
  list(APPEND HEADERS evp/jadaCorrectionSolver_impl.H)
  target_link_libraries(hymls_jada hymls)
  target_link_libraries(hymls_jada ${PHIST_LIBRARIES})

#  set_target_properties(hymls PROPERTIES COMPILE_FLAGS "-fPIC")

  target_link_libraries(hymls_eigs hymls_jada)
endif()
target_link_libraries(hymls_eigs hymls)

add_executable(hymls_main main.C)
target_link_libraries(hymls_main hymls)

add_executable(hymls_main_eigs main_eigs.C)
target_link_libraries(hymls_main_eigs hymls_eigs)

set(INCLUDE_INSTALL_DIR include)
set(LIB_INSTALL_DIR lib)
set(BIN_INSTALL_DIR bin)

# Install executables
install(TARGETS hymls_main EXPORT HYMLSTargets RUNTIME DESTINATION ${BIN_INSTALL_DIR})
install(TARGETS hymls_main_eigs EXPORT HYMLSTargets RUNTIME DESTINATION ${BIN_INSTALL_DIR})

# Install libraries
set(library_list)

install(TARGETS hymls_eigs EXPORT HYMLSTargets LIBRARY DESTINATION ${LIB_INSTALL_DIR})
list(APPEND library_list hymls_eigs)

if (${phist_FOUND})
  install(TARGETS hymls_jada EXPORT HYMLSTargets LIBRARY DESTINATION ${LIB_INSTALL_DIR})
  list(APPEND library_list hymls_jada)
endif()

install(TARGETS hymls EXPORT HYMLSTargets LIBRARY DESTINATION ${LIB_INSTALL_DIR})
list(APPEND library_list hymls)

# Install headers
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/HYMLS/HYMLSConfigVersion.cmake"
  VERSION 0 COMPATIBILITY SameMajorVersion
  )

set(ConfigPackageLocation lib/cmake/HYMLS)
configure_package_config_file("${PROJECT_SOURCE_DIR}/../cmake/HYMLSConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/HYMLSConfig.cmake" INSTALL_DESTINATION ${ConfigPackageLocation}
  PATH_VARS INCLUDE_INSTALL_DIR)

install(EXPORT HYMLSTargets FILE HYMLSTargets.cmake DESTINATION ${ConfigPackageLocation})
install(FILES "${PROJECT_BINARY_DIR}/HYMLSConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/HYMLS/HYMLSConfigVersion.cmake"
  DESTINATION ${ConfigPackageLocation}
  )

