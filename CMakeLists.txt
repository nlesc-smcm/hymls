cmake_minimum_required(VERSION 2.8)
project(HYMLS CXX)

option(HYMLS_USE_OPENMP OFF "allow using multi-threading inside HYMLS")

option(HYMLS_USE_PHIST OFF "try to find PHIST and enable it as eigensolver with HYMLS")

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

