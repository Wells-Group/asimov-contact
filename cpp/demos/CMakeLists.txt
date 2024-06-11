cmake_minimum_required(VERSION 3.21)

project(dolfinx_contact_demos)

find_package(DOLFINX REQUIRED)
find_package(Basix REQUIRED)
find_package(DOLFINX_CONTACT REQUIRED)

add_subdirectory(meshtie)
add_subdirectory(meshtieHeatEquation)
add_subdirectory(meshtieHeatTransfer)
add_subdirectory(meshtieNewton)
