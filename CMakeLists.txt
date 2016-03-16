cmake_minimum_required(VERSION 2.8)
project(HYMLS CXX)

option(HYMLS_USE_OPENMP OFF "allow using multi-threading inside HYMLS")

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


