# Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse

import dolfinx
import dolfinx.io
import numpy as np
from mpi4py import MPI

from dolfinx_contact.nitsche_rigid_surface import nitsche_rigid_surface
from dolfinx_contact.create_contact_meshes import create_circle_plane_mesh, create_sphere_plane_mesh
from dolfinx_contact.helpers import convert_mesh

if __name__ == "__main__":
    desc = "Compare Nitsche's metood for contact against a straight plane with PETSc SNES"
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--theta", default=1, type=np.float64, dest="theta",
                        help="Theta parameter for Nitsche, 1 symmetric, -1 skew symmetric, 0 Penalty-like")
    parser.add_argument("--gamma", default=1000, type=np.float64, dest="gamma",
                        help="Coercivity/Stabilization parameter for Nitsche condition")
    _solve = parser.add_mutually_exclusive_group(required=False)
    _solve.add_argument('--linear', dest='linear_solver', action='store_true',
                        help="Use linear solver", default=False)
    _3D = parser.add_mutually_exclusive_group(required=False)
    _3D.add_argument('--3D', dest='threed', action='store_true',
                     help="Use 3D mesh", default=False)
    _strain = parser.add_mutually_exclusive_group(required=False)
    _strain.add_argument('--strain', dest='plane_strain', action='store_true',
                         help="Use plane strain formulation", default=False)
    _dirichlet = parser.add_mutually_exclusive_group(required=False)
    _dirichlet.add_argument('--dirichlet', dest='dirichlet', action='store_true',
                            help="Use strong Dirichlet formulation", default=False)
    _E = parser.add_argument("--E", default=1e3, type=np.float64, dest="E",
                             help="Youngs modulus of material")
    _nu = parser.add_argument(
        "--nu", default=0.1, type=np.float64, dest="nu", help="Poisson's ratio")
    _disp = parser.add_argument("--disp", default=0.08, type=np.float64, dest="disp",
                                help="Displacement BC in negative y direction")
    _ref = parser.add_argument("--refinements", default=2, type=np.int32,
                               dest="refs", help="Number of mesh refinements")

    # Parse input arguments or set to defualt values
    args = parser.parse_args()

    # Current formulation uses unilateral contact
    nitsche_parameters = {"gamma": args.gamma, "theta": args.theta}
    nitsche_bc = not args.dirichlet
    physical_parameters = {"E": args.E, "nu": args.nu, "strain": args.plane_strain}
    vertical_displacement = -args.disp
    num_refs = args.refs + 1
    top_value = 1
    threed = args.threed
    bottom_value = 2

    # Load mesh and create identifier functions for the top (Displacement condition)
    # and the bottom (contact condition)
    if threed:
        fname = "sphere"
        create_sphere_plane_mesh(filename=f"{fname}.msh")
        convert_mesh(fname, "tetra")
        convert_mesh(f"{fname}", "triangle", ext="facets")
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        mesh.topology.create_connectivity(tdim - 1, tdim)
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}_facets.xdmf", "r") as xdmf:
            facet_marker = xdmf.read_meshtags(mesh, name="Grid")
        top_value = 2
        bottom_value = 1
        surface_value = 8
        surface_bottom = 7

    else:
        fname = "twomeshes"
        create_circle_plane_mesh(filename=f"{fname}.msh")
        convert_mesh(fname, "triangle", prune_z=True)
        convert_mesh(f"{fname}", "line", ext="facets", prune_z=True)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        mesh.topology.create_connectivity(tdim - 1, tdim)
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}_facets.xdmf", "r") as xdmf:
            facet_marker = xdmf.read_meshtags(mesh, name="Grid")
        top_value = 2
        bottom_value = 4
        surface_value = 9
        surface_bottom = 7

    e_abs = []
    e_rel = []
    dofs_global = []
    rank = MPI.COMM_WORLD.rank
    mesh_data = (facet_marker, top_value, bottom_value, surface_value, surface_bottom)
    # Solve contact problem using Nitsche's method
    u1 = nitsche_rigid_surface(mesh=mesh, mesh_data=mesh_data, physical_parameters=physical_parameters,
                               nitsche_parameters=nitsche_parameters, vertical_displacement=vertical_displacement,
                               nitsche_bc=nitsche_bc)
