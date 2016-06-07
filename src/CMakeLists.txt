
set(HYMLS_SOURCE HYMLS_Solver
    HYMLS_Preconditioner
    HYMLS_SchurComplement
    HYMLS_SchurPreconditioner
    HYMLS_OverlappingPartitioner
    HYMLS_HierarchicalMap
    HYMLS_BaseCartesianPartitioner
    HYMLS_CartesianPartitioner
    HYMLS_StandardNodeClassifier
    HYMLS_CartesianStokesClassifier
    HYMLS_SepNode
    HYMLS_HyperCube
    HYMLS_MatrixUtils
    HYMLS_DenseUtils
    HYMLS_ProjectedOperator
    HYMLS_BorderedLU
    HYMLS_BelosEpetraOperator
    HYMLS_Epetra_Time
    HYMLS_View_MultiVector
    HYMLS_EpetraExt_ProductOperator
    HYMLS_SparseDirectSolver
    HYMLS_Householder
    HYMLS_AugmentedMatrix
    HYMLS_Tools
    HYMLS_Tester
    HYMLS_PLA
    HYMLS_Exception
    HYMLS_MatrixBlock
    NOX_Epetra_LinearSystem_Hymls
    GaleriExt_Periodic
    main_utils)

set(HEADERS)

# Manually add headers without source file
set(HEADERS ${HEADERS}
  GaleriExt_Cross2DN.h
  GaleriExt_Darcy2D.h
  GaleriExt_Darcy3D.h
  GaleriExt_Star3D.h
  HYMLS_BaseNodeClassifier.H
  HYMLS_BasePartitioner.H
  HYMLS_BorderedSolver.H
  HYMLS_Macros.H
  HYMLS_no_debug.H
  HYMLS_OrthogonalTransform.H
  HYMLS_RestrictedOT.H
  HYMLS_ShiftedOperator.H
  HYMLS_MatrixBlock.H
  )

set(library_list)
set(include_list)

include_directories(${PROJECT_SOURCE_DIR})

if (${phist_FOUND})
  message(STATUS "phist found in ${PHIST_INCLUDE_DIRS}")
  include_directories(${PHIST_INCLUDE_DIRS})
  list(APPEND include_list ${PHIST_INCLUDE_DIRS})

  list(APPEND HYMLS_SOURCE
    HYMLS_PhistWrapper)
  list(APPEND HEADERS
    AnasaziPhistSolMgr.hpp)
endif()

# create header file to tell the source code about some config-time settings and features it can use
configure_file(${PROJECT_SOURCE_DIR}/src/HYMLS_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/HYMLS_config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
list(APPEND HEADERS "${CMAKE_CURRENT_BINARY_DIR}/HYMLS_config.h")

# Add header files belonging to cpp files. Headers include eachother,
#so for now just assume we want to install all header files that have a source.
set(header_extensions ".h" ".H" ".hpp")
foreach(i ${HYMLS_SOURCE})
  foreach(j ${header_extensions})
    if (EXISTS "${PROJECT_SOURCE_DIR}/${i}${j}")
      list(APPEND HEADERS "${i}${j}")
    endif()
  endforeach()
endforeach()
unset(header_extensions)

# phist is typically compiled with OpenMP support, add the openmp flag to be able to link.
# And even if we compile without, I think our Trilinos or MPI installation require it, so
# set -fopenmp if this is the GNU compiler.
if (HYMLS_USE_OPENMP)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    target_link_libraries(hymls gomp)
  elseif (CMAKE_C_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")  
  endif()
endif()

add_library(hymls ${HYMLS_SOURCE})

# Create lists of all libraries and library directories
set(library_dependencies ${Trilinos_LIBRARIES})
list(APPEND library_dependencies ${Trilinos_TPL_LIBRARIES})
list(APPEND library_dependencies ${MPI_CXX_LIBRARIES})

set(library_directories ${Trilinos_LIBRARY_DIRS})
list(APPEND library_directories ${Trilinos_TPL_LIBRARY_DIRS})
list(APPEND library_directories ${MPI_CXX_LIBRARY_DIRSY_DIRS})

if (${phist_FOUND})
  list(APPEND library_dependencies ${PHIST_LIBRARIES})
  list(APPEND library_directories ${PHIST_LIBRARY_DIRS})
endif()

# Find absolute paths to all libraries. See
# https://cmake.org/cmake/help/v3.0/command/link_directories.html
# for the reason why we want this.
set(found_library_dependencies)
foreach(lib ${library_dependencies})
  set(found_${lib})
  if(IS_ABSOLUTE ${lib})
    set(found_${lib} ${lib})
  else()
    find_library(found_${lib} ${lib} ${library_directories})
  endif()
  list(APPEND found_library_dependencies ${found_${lib}})
#  message(STATUS "${lib}")
#  message(STATUS "${found_${lib}}")
endforeach(lib)

# Link HYMLS to all found libraries
target_link_libraries(hymls ${found_library_dependencies})

add_executable(hymls_main main.C)
target_link_libraries(hymls_main hymls)

add_executable(hymls_main_eigs main_eigs.C)
target_link_libraries(hymls_main_eigs hymls)

set(INCLUDE_INSTALL_DIR include)
set(LIB_INSTALL_DIR lib)
set(BIN_INSTALL_DIR bin)

# Install executables
install(TARGETS hymls_main EXPORT HYMLSTargets RUNTIME DESTINATION ${BIN_INSTALL_DIR})
install(TARGETS hymls_main_eigs EXPORT HYMLSTargets RUNTIME DESTINATION ${BIN_INSTALL_DIR})

# Install libraries
install(TARGETS hymls EXPORT HYMLSTargets LIBRARY DESTINATION ${LIB_INSTALL_DIR})
list(APPEND library_list hymls)

# Install headers
install(FILES ${HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/HYMLS/HYMLSConfigVersion.cmake"
  VERSION 0 COMPATIBILITY SameMajorVersion
  )

set(ConfigPackageLocation lib/cmake/HYMLS)
configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/HYMLSConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/HYMLSConfig.cmake" INSTALL_DESTINATION ${ConfigPackageLocation}
  PATH_VARS INCLUDE_INSTALL_DIR)

install(EXPORT HYMLSTargets FILE HYMLSTargets.cmake DESTINATION ${ConfigPackageLocation})
install(FILES "${PROJECT_BINARY_DIR}/HYMLSConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/HYMLS/HYMLSConfigVersion.cmake"
  DESTINATION ${ConfigPackageLocation}
  )
