cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME demo_meshtie)
project(${PROJECT_NAME} LANGUAGES C CXX)

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(DOLFINX REQUIRED)
find_package(Basix REQUIRED)
find_package(DOLFINX_CONTACT REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if (PETSC_SCALAR_COMPLEX EQUAL 1)
  set(SCALAR_TYPE "--scalar_type=double _Complex")
endif()

# Compile UFL files
set(PYTHONPATH "$ENV{PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}")
set(ufl_file linear_elasticity)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ufl_file}.c
  COMMAND PYTHONPATH=${PYTHONPATH} python3 -m ffcx ${SCALAR_TYPE} ${CMAKE_CURRENT_SOURCE_DIR}/${ufl_file}.py
  COMMENT "Run ffcx on ${ufl_file}.py"
)

add_executable(${PROJECT_NAME} main.cpp ${ufl_file}.c)

# Target libraries
target_link_libraries(${PROJECT_NAME} PRIVATE dolfinx)
target_link_libraries(${PROJECT_NAME} PRIVATE dolfinx_contact)
