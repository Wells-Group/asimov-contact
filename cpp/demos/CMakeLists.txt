cmake_minimum_required(VERSION 3.16)
project(dolfinx-contact-demos)

find_package(DOLFINX REQUIRED)
find_package(Basix REQUIRED)
find_package(DOLFINX_CONTACT REQUIRED)

add_subdirectory(meshtie)
add_subdirectory(meshtieHeatEquation)
add_subdirectory(meshtieHeatTransfer)
add_subdirectory(meshtieNewton)