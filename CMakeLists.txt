cmake_minimum_required(VERSION 3.5.0)
project(HYMLS C CXX)

option(HYMLS_USE_OPENMP OFF "allow using multi-threading inside HYMLS")

option(HYMLS_USE_PHIST OFF "try to find PHIST and enable it as eigensolver with HYMLS")

if (HYMLS_USE_PHIST)
  option(HYMLS_USE_PHIST_CORRECTION_SOLVER ON "If set to OFF (the default), HYMLS provides it's own solver of the Jacobi-Davidson correction equation. If set to ON, the PHIST implementation is used.")
endif()

option(HYMLS_USE_JDQZPP OFF "try to find JDQZPP and enable it as eigensolver with HYMLS")

option(HYMLS_STORE_MATRICES "dump all matrices, maps etc. encountered (huge overhead)" OFF)
option(HYMLS_DEBUGGING "turns on verbose debugging output" OFF)
option(HYMLS_FUNCTION_TRACING "turns on very verbose output on every function entered/left" OFF)
option(HYMLS_MEMORY_PROFILING "report memory usage after calling a function, is really slow" OFF)
option(HYMLS_LONG_LONG "use long long global indices" OFF)

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

include("${PROJECT_SOURCE_DIR}/cmake/common.cmake")

# here are the HYMLS headers:
include_directories(BEFORE ${PROJECT_SOURCE_DIR}/src)

# and here are any configured headers, e.g. HYMLS_config.h:
include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR}/src/)

# enable "make test" and "make check"  targets
enable_testing()

add_subdirectory(src)
add_subdirectory(testSuite)

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

