cmake_minimum_required(VERSION 2.8)

set(SOURCES HYMLS_DenseUtils
  HYMLS_Tester
  HYMLS_Tools
  main)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_BINARY_DIR}/src)

if (DEFINED ENV{TRILINOS_HOME})
  set(Trilinos_DIR "$ENV{TRILINOS_HOME}/lib/cmake/Trilinos")
endif()
find_package(Trilinos)

include_directories(${Trilinos_INCLUDE_DIRS})
include_directories(${Trilinos_TPL_INCLUDE_DIRS})

link_directories(${Trilinos_LIBRARY_DIRS})
link_directories(${Trilinos_TPL_LIBRARY_DIRS})

add_executable(unit_tests ${SOURCES})

target_link_libraries(unit_tests ${Trilinos_LIBRARIES})
target_link_libraries(unit_tests ${Trilinos_TPL_LIBRARIES})

target_link_libraries(unit_tests hymls)
