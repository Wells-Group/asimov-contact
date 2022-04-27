# Top level CMakeLists for custom assemblers
cmake_minimum_required(VERSION 3.10)

# Set project name and version number
project(DOLFINX_CONTACT VERSION 0.4.1.0)

#------------------------------------------------------------------------------
# Set CMake options, see `cmake --help-policy CMP00xx`

cmake_policy(VERSION 3.10)
if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()
if (POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

# Use C++17
set(CMAKE_CXX_STANDARD 17)

# Require C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Do not enable compler-specific extensions
set(CMAKE_CXX_EXTENSIONS OFF)

#------------------------------------------------------------------------------
# configurable options for how we want to build

include(FeatureSummary)

option(BUILD_SHARED_LIBS "Build DOLFINX_CONTACT with shared libraries." ON)
add_feature_info(BUILD_SHARED_LIBS BUILD_SHARED_LIBS "Build DOLFINX_CONTACT with shared libraries.")

option(CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath." ON)
add_feature_info(CMAKE_INSTALL_RPATH_USE_LINK_PATH CMAKE_INSTALL_RPATH_USE_LINK_PATH "Add paths to linker search and installed rpath.")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})

# Find packages
find_package(DOLFINX 0.3.1...0.4.1 REQUIRED)
find_package(Basix 0.3.1...0.4.1 REQUIRED)
find_package(xtensor REQUIRED)
find_package(DOLFINX_CUAS 0.3.1.0 REQUIRED)

feature_summary(WHAT ALL)

# Installation of DOLFIN_CONTACT library
add_library(dolfinx_contact "")

# Basix
target_link_libraries(dolfinx_contact PUBLIC Basix::basix)

# DOLFINx
target_link_libraries(dolfinx_contact PUBLIC dolfinx)

include(GNUInstallDirs)
install(FILES Contact.h contact_kernels.hpp utils.h coefficients.h geometric_quantities.h SubMesh.h QuadratureRule.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dolfinx_contact COMPONENT Development)

target_sources(dolfinx_contact PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/coefficients.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/geometric_quantities.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SubMesh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/QuadratureRule.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Contact.cpp
  )

# Set target include location (for build and installed)
target_include_directories(dolfinx_contact PUBLIC
                           $<INSTALL_INTERFACE:include>
                           "$<BUILD_INTERFACE:${DOLFINX_CONTACT_SOURCE_DIR};${DOLFINX_CONTACT_SOURCE_DIR}/dolfinx_contact>")

# Set target properties
set_target_properties(dolfinx_contact PROPERTIES
  VERSION ${DOLFINX_CONTACT_VERSION}
  SOVERSION ${DOLFINX_CONTACT_VERSION_MAJOR}.${DOLFINX_CONTACT_VERSION_MINOR})


# Add version to definitions (public)
target_compile_definitions(dolfinx_contact PUBLIC DOLFINX_CONTACT_VERSION="${DOLFINX_CONTACT_VERSION}")


#------------------------------------------------------------------------------
# Install dolfinx library and header files
install(TARGETS dolfinx_contact
  EXPORT DOLFINX_CONTACTTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT RuntimeExecutables
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT RuntimeLibraries
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT Development
  )

# Generate DOLFINTargets.cmake
install(EXPORT DOLFINX_CONTACTTargets DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_contact)

include(CMakePackageConfigHelpers)

configure_package_config_file(${DOLFINX_CONTACT_SOURCE_DIR}/DOLFINX_CONTACTConfig.cmake.in
${CMAKE_BINARY_DIR}/dolfinx_contact/DOLFINX_CONTACTConfig.cmake
INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_contact)

# Install CMake helper files
install(
  FILES
  ${CMAKE_BINARY_DIR}/dolfinx_contact/DOLFINX_CONTACTConfig.cmake
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/dolfinx_contact
  COMPONENT Development)
