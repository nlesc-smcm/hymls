cmake_minimum_required(VERSION 2.8)

set(SOURCES HYMLS_DenseUtils
  HYMLS_Tester
  HYMLS_Tools
  main)

message(STATUS "unit_tests/PROJECT_SOURCE_DIR='${PROJECT_SOURCE_DIR}'")

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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Trilinos_CXX_COMPILER_FLAGS}")

add_executable(unit_tests ${SOURCES})

target_link_libraries(unit_tests ${Trilinos_LIBRARIES})
target_link_libraries(unit_tests ${Trilinos_TPL_LIBRARIES})

target_link_libraries(unit_tests hymls)

add_test(NAME unit_tests COMMAND ./unit_tests)
        set_tests_properties(unit_tests PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")
        set_tests_properties(unit_tests PROPERTIES FAIL_REGULAR_EXPRESSION "TEST FAILED")
        set_tests_properties(unit_tests PROPERTIES TIMEOUT 500)
