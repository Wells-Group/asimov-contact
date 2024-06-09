cmake_minimum_required(VERSION 3.19)

set(PROJECT_NAME demo_meshtie_heattransfer)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 20)

find_package(DOLFINX REQUIRED)
find_package(Basix REQUIRED)
find_package(DOLFINX_CONTACT REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(ufl_file thermo_elasticity)

# Add target to compile UFL files
if (PETSC_SCALAR_COMPLEX EQUAL 1)
  set(SCALAR_TYPE "--scalar_type=double _Complex")
endif()
# Create compilation command
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ufl_file}.c
  COMMAND ffcx ${SCALAR_TYPE} ${CMAKE_CURRENT_SOURCE_DIR}/${ufl_file}.py -o ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${ufl_file}.py
  COMMENT "Compiling ${ufl_file}.py"
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