set(SOURCES
  HYMLS_DenseUtils
  HYMLS_Preconditioner
  HYMLS_ProjectedOperator
  HYMLS_Solver
  HYMLS_Tester
  HYMLS_Tools
  main)

add_executable(unit_tests ${SOURCES})

target_link_libraries(unit_tests ${Trilinos_LIBRARIES})
target_link_libraries(unit_tests ${Trilinos_TPL_LIBRARIES})

target_link_libraries(unit_tests hymls)

add_test(NAME unit_tests COMMAND ./unit_tests)
        set_tests_properties(unit_tests PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")
        set_tests_properties(unit_tests PROPERTIES FAIL_REGULAR_EXPRESSION "TEST FAILED")
        set_tests_properties(unit_tests PROPERTIES TIMEOUT 500)
