cmake_minimum_required(VERSION 2.8)
project(hymls CXX)

if (CMAKE_MAJOR_VERSION GREATER 2)
  set(CMAKE_CXX_STANDARD 11)
endif()

find_package(Matlab COMPONENTS MAIN_PROGRAM)

include_directories(${PROJECT_SOURCE_DIR}/src)

find_package(Trilinos)

include_directories(${Trilinos_INCLUDE_DIRS})
include_directories(${Trilinos_TPL_INCLUDE_DIRS})

link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${Trilinos_TPL_LIBRARY_DIRS})

find_package(HYMLS)

include_directories(${HYMLS_INCLUDE_DIRS})
link_directories(${HYMLS_LIBRARY_DIRS})

find_package(MPI REQUIRED)
if (${MPI_FOUND})
  set (CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_CXX_COMPILE_FLAGS})
  set (CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_CXX_LINK_FLAGS})
  include_directories (${MPI_INCLUDE_PATH})
  include_directories (${MPI_CXX_INCLUDE_PATH})
else()
  message(WARNING "could not find MPI. Trying to compile anyway, presuming the compiler/linker knows where to find it.")
endif()

set (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -g)

matlab_add_mex(NAME HYMLS_init SRC HYMLS_init.C)
matlab_add_mex(NAME HYMLS_apply SRC HYMLS_apply.C)
matlab_add_mex(NAME HYMLS_free SRC HYMLS_free.C)

target_link_libraries(HYMLS_init ${Trilinos_LIBRARIES})
target_link_libraries(HYMLS_init ${Trilinos_TPL_LIBRARIES})
target_link_libraries(HYMLS_init ${HYMLS_LIBRARIES})
target_link_libraries(HYMLS_init ${MPI_LIBRARIES})

target_link_libraries(HYMLS_apply ${Trilinos_LIBRARIES})
target_link_libraries(HYMLS_apply ${Trilinos_TPL_LIBRARIES})
target_link_libraries(HYMLS_apply ${HYMLS_LIBRARIES})
target_link_libraries(HYMLS_apply ${MPI_LIBRARIES})

target_link_libraries(HYMLS_free ${Trilinos_LIBRARIES})
target_link_libraries(HYMLS_free ${Trilinos_TPL_LIBRARIES})
target_link_libraries(HYMLS_free ${HYMLS_LIBRARIES})
target_link_libraries(HYMLS_free ${MPI_LIBRARIES})

enable_testing()
matlab_add_unit_test(
    NAME stokes0
    TIMEOUT 30
    UNITTEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/test/test_stokes0.m
    ADDITIONAL_PATH "${CMAKE_CURRENT_BINARY_DIR}:${CMAKE_CURRENT_SOURCE_DIR}"
)
