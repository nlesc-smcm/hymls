cmake_minimum_required(VERSION 3.5)
project(HYMLS CXX)

list (APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/")
if (CMAKE_MAJOR_VERSION GREATER 2)
  set(CMAKE_CXX_STANDARD 11)
  set(HYMLS_HAVE_CXX11 ON)
endif()

option(HYMLS_USE_OPENMP OFF "allow using multi-threading inside HYMLS")

option(HYMLS_USE_PHIST OFF "try to find PHIST and enable it as eigensolver with HYMLS")

option(HYMLS_STORE_MATRICES "dump all matrices, maps etc. encountered (huge overhead)" OFF)
option(HYMLS_DEBUGGING "turns on verbose debugging output" OFF)
option(HYMLS_FUNCTION_TRACING "turns on very verbose output on every function entered/left" OFF)

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

set(BUILD_SHARED_LIBS ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(STATUS "Using GNU compiler flags")

  if (NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_C_FLAGS        "${CMAKE_C_FLAGS} -O2 -g -Wall -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS      "${CMAKE_CXX_FLAGS} -O2 -g -Wall -Wno-unknown-pragmas")
    set(CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} -O2 -g -Wall -Wno-unknown-pragmas")
  endif()

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

# here are the HYMLS headers:
include_directories(${PROJECT_SOURCE_DIR}/src)

# and here are any configured headers, e.g. HYMLS_config.h:
include_directories(${CMAKE_CURRENT_BINARY_DIR}/src/)

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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Trilinos_CXX_COMPILER_FLAGS}")


if (HYMLS_USE_PHIST)
  find_package(phist REQUIRED CONFIG)
endif()

# enable "make test" and "make check"  targets
enable_testing()

add_subdirectory (src)
add_subdirectory (testSuite)

####################
# uninstall target #
####################

configure_file(
    "${PROJECT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)
add_custom_target(uninstall "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake/cmake_uninstall.cmake")

set(CTEST_OPTIONS --force-new-ctest-process --verbose CACHE STRING "ctest options")

# Adding custom test target "check" because it is the only way to show the full output of test errors on console
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} ${CTEST_OPTIONS})

