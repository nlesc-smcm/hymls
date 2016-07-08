file(GLOB XMLFILES "${PROJECT_SOURCE_DIR}/testSuite/*.xml")

# copy the XML input files for our example runs
foreach (xml_file ${XMLFILES})
  configure_file(${xml_file} ${CMAKE_CURRENT_BINARY_DIR}/)
endforeach()


add_subdirectory(unit_tests)
add_subdirectory(integration_tests)
