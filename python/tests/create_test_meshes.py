# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from dolfinx_contact.create_mesh import (convert_mesh, create_disk_mesh,
                                         create_sphere_mesh)

# This script creates the test meshes used in test_dolfinx_cuas.py
# run before running pytest
fname = "disk"
create_disk_mesh(filename=f"{fname}.msh")
convert_mesh(fname, "triangle", prune_z=True)
fname = "sphere"
create_sphere_mesh(filename=f"{fname}.msh")
convert_mesh(fname, "tetra")
