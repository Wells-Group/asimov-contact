# Top level CMakeLists for custom assemblers
cmake_minimum_required(VERSION 3.21)

# Set project name and version number
project(DOLFINX_CONTACT VERSION 0.8.0.0)

# -----------------------------------------------------------------------------
# Set CMake options, see `cmake --help-policy CMP00xx`

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Do not enable compler-specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wall -Werror -Wextra -pedantic" HAVE_PEDANTIC)
if(HAVE_PEDANTIC)
  list(APPEND DOLFINX_CONTACT_CXX_DEVELOPER_FLAGS
       -Wall;-Werror;-Wextra;-pedantic)
endif()

# Debug flags
check_cxx_compiler_flag(-g HAVE_DEBUG)
if(HAVE_DEBUG)
  list(APPEND DOLFINX_CONTACT_CXX_DEVELOPER_FLAGS -g)
endif()

# Optimisation
check_cxx_compiler_flag(-O2 HAVE_O2_OPTIMISATION)
if(HAVE_O2_OPTIMISATION)
  list(APPEND DOLFINX_CONTACT_CXX_DEVELOPER_FLAGS -O2)
endif()

# Turn off some checks in gcc12 and gcc13 due to false positives with the fmt
# library
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "11.4"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "14.0")
  list(APPEND DOLFINX_CONTACT_CXX_DEVELOPER_FLAGS
       -Wno-array-bounds;-Wno-stringop-overflow)
endif()

# ------------------------------------------------------------------------------
# configurable options for how we want to build

include(FeatureSummary)

option(BUILD_SHARED_LIBS "Build DOLFINX_CONTACT with shared libraries." ON)
add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS
                 "Build DOLFINX_CONTACT with shared libraries.")

option(CMAKE_INSTALL_RPATH_USE_LINK_PATH
       "Add paths to linker search and installed rpath." ON)
add_feature_info(
  CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_INSTALL_RPATH_USE_LINK_PATH
  "Add paths to linker search and installed rpath.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

# Find packages
find_package(DOLFINX 0.8.0.0 REQUIRED)
find_package(Basix 0.8.0.0 REQUIRED)

feature_summary(WHAT ALL)

# Installation of DOLFIN_CONTACT library
add_library(dolfinx_contact)

target_link_libraries(dolfinx_contact PUBLIC Basix::basix)
target_link_libraries(dolfinx_contact PUBLIC dolfinx)
target_link_libraries(dolfinx_contact PRIVATE m)

# Set 'Developer' build type flags
target_compile_options(
  dolfinx_contact
  PRIVATE $<$<CONFIG:Developer>:${DOLFINX_CONTACT_CXX_DEVELOPER_FLAGS}>)

target_sources(
  dolfinx_contact
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/coefficients.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/contact_kernels.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/elasticity.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/geometric_quantities.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/meshtie_kernels.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/SubMesh.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/QuadratureRule.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/Contact.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/MeshTie.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/RayTracing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/KernelData.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/error_handling.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/rigid_surface_kernels.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/point_cloud.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/parallel_mesh_ghosting.cpp)

# Set target include location (for build and installed)
target_include_directories(
  dolfinx_contact
  PUBLIC
    $<INSTALL_INTERFACE:include>
    "$<BUILD_INTERFACE:${DOLFINX_CONTACT_SOURCE_DIR};${DOLFINX_CONTACT_SOURCE_DIR}/dolfinx_contact>"
)

# Set target properties
set_target_properties(
  dolfinx_contact
  PROPERTIES VERSION ${DOLFINX_CONTACT_VERSION}
             SOVERSION
             ${DOLFINX_CONTACT_VERSION_MAJOR}.${DOLFINX_CONTACT_VERSION_MINOR})

# Add version to definitions (public)
target_compile_definitions(
  dolfinx_contact PUBLIC DOLFINX_CONTACT_VERSION="${DOLFINX_CONTACT_VERSION}")

# ------------------------------------------------------------------------------
# Install library and header files

include(GNUInstallDirs)
install(
  FILES Contact.h
        MeshTie.h
        contact_kernels.h
        rigid_surface_kernels.h
        error_handling.h
        utils.h
        coefficients.h
        elasticity.h
        geometric_quantities.h
        meshtie_kernels.h
        parallel_mesh_ghosting.h
        point_cloud.h
        SubMesh.h
        QuadratureRule.h
        RayTracing.h
        KernelData.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx_contact
  COMPONENT Development)

install(
  TARGETS dolfinx_contact
  EXPORT DOLFINX_CONTACTTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development)

# Generate DOLFINTargets.cmake
install(EXPORT DOLFINX_CONTACTTargets
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_contact)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${DOLFINX_CONTACT_SOURCE_DIR}/DOLFINX_CONTACTConfig.cmake.in
  ${CMAKE_BINARY_DIR}/dolfinx_contact/DOLFINX_CONTACTConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_contact)

# Install CMake helper files
install(
  FILES ${CMAKE_BINARY_DIR}/dolfinx_contact/DOLFINX_CONTACTConfig.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_contact
  COMPONENT Development)
