list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")

if (CMAKE_MAJOR_VERSION GREATER 2)
  set(CMAKE_CXX_STANDARD 11)
  set(HYMLS_HAVE_CXX11 ON)
endif()

set(BUILD_SHARED_LIBS ON CACHE STRING "build shared library for hymls (libhymls.so rather than libhymls.a)")

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
    CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  message(STATUS "Using GNU compiler flags")

  # try to enable CCache as a 'compiler launcher'
  find_program(CCACHE_FOUND ccache)
  if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
  endif(CCACHE_FOUND)

  if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -Wall -Wno-unknown-pragmas -Wno-deprecated-declarations")
  endif()

  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas -Wno-deprecated-declarations")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  message(STATUS "Using Intel compiler flags")

  if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -mkl")
  endif()

  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mkl -march=native")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -mkl")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")

  set(HYMLS_USE_MKL 1 CACHE STRING "use MKL ParDiSo for direct solves")
else()
  message (WARNING "No custom compiler flags for the ${CMAKE_CXX_COMPILER_ID} compiler were found.")
endif()

find_package(MPI REQUIRED)
if (MPI_FOUND)
  set (CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_CXX_COMPILE_FLAGS})
  set (CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_CXX_LINK_FLAGS})

  # For now, just include everything
  include_directories (${MPI_CXX_INCLUDE_PATH})
  include_directories (${MPI_INCLUDE_PATH})
else()
  message(WARNING "could not find MPI. Trying to compile anyway, presuming the compiler/linker knows where to find it.")
endif()

set(MPI_EXECUTABLE ${MPIEXEC})
if (NOT MPIEXEC)
  set(MPI_EXECUTABLE ${MPIEXEC_EXECUTABLE})
endif()

execute_process(COMMAND mpiexec echo RESULT_VARIABLE status OUTPUT_QUIET)
if(NOT status)
  set(MPI_EXECUTABLE mpiexec)
endif()

execute_process(COMMAND ${MPI_EXECUTABLE} --oversubscribe echo RESULT_VARIABLE status OUTPUT_QUIET)
if (NOT status)
  set(MPI_OVERSUBSCRIBE "--oversubscribe")
endif()

#################
# find Trilinos #
#################

if (DEFINED ENV{TRILINOS_HOME})
  set(Trilinos_DIR "$ENV{TRILINOS_HOME}/lib/cmake/Trilinos")
endif()
find_package(Trilinos REQUIRED CONFIG)

include_directories(${Trilinos_INCLUDE_DIRS})
include_directories(${Trilinos_TPL_INCLUDE_DIRS})
list(APPEND include_list ${Trilinos_INCLUDE_DIRS})

# The PyTrilinos dependency is broken, see e.g. https://github.com/trilinos/Trilinos/issues/866
list(FILTER Trilinos_LIBRARIES EXCLUDE REGEX pytrilinos)

# Link to OpenMP to prevent errors with EPETRA_HAVE_OMP
string(FIND ${Trilinos_CXX_COMPILER_FLAGS} "fopenmp" ENABLE_OPENMP)
if (NOT ${ENABLE_OPENMP} EQUAL -1 OR ${HYMLS_USE_OPENMP})
  find_package(OpenMP)
  set (CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${OpenMP_CXX_FLAGS})
  include_directories (${OpenMP_CXX_INCLUDE_DIRS})
endif()

include(CheckCXXSymbolExists)
list(APPEND CMAKE_REQUIRED_INCLUDES ${Trilinos_INCLUDE_DIRS})
check_cxx_symbol_exists(HAVE_TEUCHOS_COMPLEX "Teuchos_config.h" HAVE_TEUCHOS_COMPLEX)

if (HYMLS_USE_PHIST)
  find_package(phist REQUIRED CONFIG)
endif()
