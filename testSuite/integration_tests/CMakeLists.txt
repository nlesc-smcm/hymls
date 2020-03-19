file(GLOB XMLFILES "${PROJECT_SOURCE_DIR}/testSuite/integration_tests/*.xml")

# copy the XML input files for our example runs
foreach (xml_file ${XMLFILES})
  configure_file(${xml_file} ${CMAKE_CURRENT_BINARY_DIR}/)
endforeach()

add_executable(integration_tests integration_tests.cpp)

if (${phist_FOUND})
  target_link_libraries(integration_tests ${PHIST_LIBRARIES})
endif()

target_link_libraries(integration_tests hymls)

if (NOT ${HYMLS_DEBUGGING})
  add_test(NAME integration_tests COMMAND mpirun -np ${HYMLS_TEST_NPROCS} ./integration_tests)
  set_tests_properties(integration_tests PROPERTIES PASS_REGULAR_EXPRESSION "ALL TESTS PASSED")
  set_tests_properties(integration_tests PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED")
  set_tests_properties(integration_tests PROPERTIES TIMEOUT 600)
  set_tests_properties(integration_tests PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1)
endif()
