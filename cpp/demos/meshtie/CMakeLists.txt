cmake_minimum_required(VERSION 3.19)

set(PROJECT_NAME demo_meshtie)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Ofast -march=native ${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_C_FLAGS "-Ofast -march=native ${CMAKE_C_FLAGS} -Wall")

find_package(DOLFINX REQUIRED)
find_package(Basix REQUIRED)
find_package(DOLFINX_CONTACT REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(ufl_file linear_elasticity)

# Add target to compile UFL files
MESSAGE(STATUS "PETSC STATUS ${PETSC_SCALAR_COMPLEX} ${PETSC_ARCH}")
if (PETSC_SCALAR_COMPLEX EQUAL 1)
  set(SCALAR_TYPE "--scalar_type=complex128")
endif()
# Create compilation command
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ufl_file}.c
  COMMAND ffcx ${SCALAR_TYPE} ${CMAKE_CURRENT_SOURCE_DIR}/${ufl_file}.py -o ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ufl_file}.py
  COMMENT "Compiling ${ufl_file}.py with ${SCALAR_TYPE}"
)


add_executable(${PROJECT_NAME} main.cpp ${ufl_file}.c)

target_link_libraries(${PROJECT_NAME} PRIVATE dolfinx)
target_link_libraries(${PROJECT_NAME} PRIVATE dolfinx_contact)