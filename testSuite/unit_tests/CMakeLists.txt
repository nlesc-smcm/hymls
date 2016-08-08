set(SOURCES
  HYMLS_CartesianPartitioner
  HYMLS_StandardNodeClassifier
  HYMLS_DenseUtils
  HYMLS_Preconditioner
  HYMLS_Tester
  HYMLS_Tools
  main)

add_executable(unit_tests ${SOURCES})

target_link_libraries(unit_tests ${Trilinos_LIBRARIES})
target_link_libraries(unit_tests ${Trilinos_TPL_LIBRARIES})

target_link_libraries(unit_tests hymls)

add_test(NAME unit_tests_1 COMMAND ./unit_tests)
add_test(NAME unit_tests_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ./unit_tests)
add_test(NAME unit_tests_5 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 5 ./unit_tests)

set(test_list unit_tests_1 unit_tests_4 unit_tests_5)
set_tests_properties(${test_list} PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")
set_tests_properties(${test_list} PROPERTIES FAIL_REGULAR_EXPRESSION "TEST FAILED")
set_tests_properties(${test_list} PROPERTIES TIMEOUT 500)
