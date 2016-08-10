set(SOURCES
  GaleriExt_Darcy2D
  GaleriExt_Stokes2D
  HYMLS_CartesianPartitioner
  HYMLS_StandardNodeClassifier
  HYMLS_DenseUtils
  HYMLS_Preconditioner
  HYMLS_Tester
  HYMLS_Tools
  HYMLS_UnitTests
  main)

# we use this matrix file for testing the GaleriExt::Matrices::Stokes2D function
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../data/DrivenCavity/32x32/Re0/jac.mtx ${CMAKE_CURRENT_BINARY_DIR}/stokes32x32.mtx)

add_executable(unit_tests ${SOURCES})

target_link_libraries(unit_tests ${Trilinos_LIBRARIES})
target_link_libraries(unit_tests ${Trilinos_TPL_LIBRARIES})

target_link_libraries(unit_tests hymls)

add_test(NAME unit_tests_1 COMMAND ./unit_tests)
add_test(NAME unit_tests_2 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 ./unit_tests)
add_test(NAME unit_tests_3 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 3 ./unit_tests)
add_test(NAME unit_tests_4 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 ./unit_tests)
add_test(NAME unit_tests_5 COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 5 ./unit_tests)

set(test_list unit_tests_1 unit_tests_2 unit_tests_3 unit_tests_4 unit_tests_5)
set_tests_properties(${test_list} PROPERTIES PASS_REGULAR_EXPRESSION "TEST PASSED")
set_tests_properties(${test_list} PROPERTIES FAIL_REGULAR_EXPRESSION "TEST FAILED")
set_tests_properties(${test_list} PROPERTIES TIMEOUT 500)
