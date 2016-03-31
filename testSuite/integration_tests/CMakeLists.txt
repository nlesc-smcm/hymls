cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)

file(GLOB XMLFILES "${PROJECT_SOURCE_DIR}/testSuite/integration_tests/*.xml")

# copy the XML input files for our example runs
foreach (xml_file ${XMLFILES})
  configure_file(${xml_file} ${CMAKE_CURRENT_BINARY_DIR}/)
endforeach()

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

if (HYMLS_USE_PHIST)
  find_package(phist QUIET CONFIG)
  if (phist_DIR MATCHES "phist_DIR-NOT-FOUND")
    message(WARNING "phist was requested but not found")
    set(HYMLS_USE_PHIST OFF)
  endif()
endif()

if (${phist_FOUND})
  include_directories(${PHIST_INCLUDE_DIRS})
  link_directories(${PHIST_LIBRARY_DIRS})
endif()

add_executable(integration_tests integration_tests.C)

target_link_libraries(integration_tests ${Trilinos_LIBRARIES})
target_link_libraries(integration_tests ${Trilinos_TPL_LIBRARIES})

if (${phist_FOUND})
  target_link_libraries(integration_tests ${PHIST_LIBRARIES})
endif()

target_link_libraries(integration_tests hymls)
