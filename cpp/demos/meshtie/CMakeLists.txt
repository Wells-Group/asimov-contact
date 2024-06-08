cmake_minimum_required(VERSION 3.19)

set(PROJECT_NAME demo_meshtie)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native ${CMAKE_C_FLAGS} -Wall")

find_package(DOLFINX REQUIRED)
find_package(Basix REQUIRED)
find_package(DOLFINX_CONTACT REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(ufl_file linear_elasticity)

# Check for what petsc version we are using
include(CheckSymbolExists)
set(CMAKE_REQUIRED_INCLUDES ${PETSC_INCLUDE_DIRS})
check_symbol_exists(PETSC_USE_COMPLEX petscsystypes.h PETSC_SCALAR_COMPLEX)
check_symbol_exists(PETSC_USE_REAL_DOUBLE petscsystypes.h PETSC_REAL_DOUBLE)

# Add target to compile UFL files
if(PETSC_SCALAR_COMPLEX EQUAL 1)
  message(STATUS "** This demo does not support complex mode")
else()
  if(PETSC_REAL_DOUBLE EQUAL 1)
    set(SCALAR_TYPE "--scalar_type=float64")
  else()
    message(STATUS "** This demo does not support single precision")
  endif()
endif()
# Create compilation command
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ufl_file}.c
  COMMAND ffcx ${SCALAR_TYPE} ${CMAKE_CURRENT_SOURCE_DIR}/${ufl_file}.py -o ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ufl_file}.py
  COMMENT "Compiling ${ufl_file}.py with ${SCALAR_TYPE}"
)


add_executable(${PROJECT_NAME} main.cpp ${ufl_file}.c)

# Copy mesh files into binary directory
function(copy_meshes_to_target_dir target)
  add_custom_command(
        TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "$<TARGET_PROPERTY:${target},SOURCE_DIR>/../meshes"
            "$<TARGET_PROPERTY:${target},BINARY_DIR>"
    )
endfunction()
copy_meshes_to_target_dir(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE dolfinx)
target_link_libraries(${PROJECT_NAME} PRIVATE dolfinx_contact)

