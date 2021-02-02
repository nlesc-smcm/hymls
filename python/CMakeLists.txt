find_package(SWIG)

if (SWIG_FOUND)
  if (POLICY CMP0078)
    cmake_policy(SET CMP0078 NEW)
  endif ()
  if (POLICY CMP0086)
    cmake_policy(SET CMP0086 NEW)
  endif ()

  find_package(Python COMPONENTS Interpreter Development)
  include_directories(${Python_INCLUDE_DIRS})

  if (NOT ${Python_VERSION} VERSION_LESS 3.0)
    list(APPEND CMAKE_SWIG_FLAGS -py3)
  endif ()

  include(${SWIG_USE_FILE})
  set_property(SOURCE HYMLS.i PROPERTY CPLUSPLUS ON)
  swig_add_library(HYMLS LANGUAGE Python SOURCES HYMLS.i)
  swig_link_libraries(HYMLS hymls)

  set(SITE_PACKAGE_DIR "lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/HYMLS")
  install(TARGETS HYMLS DESTINATION ${SITE_PACKAGE_DIR})
  get_property(SUPPORT_FILES TARGET HYMLS PROPERTY SWIG_SUPPORT_FILES)
  install(FILES ${SUPPORT_FILES} DESTINATION ${SITE_PACKAGE_DIR})
  install(FILES init.py DESTINATION ${SITE_PACKAGE_DIR} RENAME __init__.py)
endif ()
