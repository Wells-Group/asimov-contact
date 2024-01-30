# Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse

from mpi4py import MPI

import numpy as np

from dolfinx.io import XDMFFile
from dolfinx_contact.meshing import convert_mesh, create_circle_plane_mesh, create_sphere_plane_mesh
from dolfinx_contact.one_sided.nitsche_rigid_surface import nitsche_rigid_surface

if __name__ == "__main__":
    desc = "Compare Nitsche's metood for contact against a straight plane with PETSc SNES"
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--theta", default=1, type=np.float64, dest="theta",
                        help="Theta parameter for Nitsche, 1 symmetric, -1 skew symmetric, 0 Penalty-like")
    parser.add_argument("--gamma", default=10, type=np.float64, dest="gamma",
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
    _disp = parser.add_argument("--disp", default=0.2, type=np.float64, dest="disp",
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
    outdir = "meshes"
    if threed:
        fname = f"{outdir}/sphere"
        create_sphere_plane_mesh(filename=f"{fname}.msh")
        convert_mesh(fname, fname, gdim=3)
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh()
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, tdim)
            facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")
        top_value = 2
        bottom_value = 1
        surface_value = 8
        surface_bottom = 7

    else:
        fname = f"{outdir}/twomeshes"
        create_circle_plane_mesh(filename=f"{fname}.msh")
        convert_mesh(fname, fname, gdim=2)

        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh()
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, tdim)
            facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")

        top_value = 2
        bottom_value = 4
        surface_value = 9
        surface_bottom = 7

    # Solver options
    newton_options = {"relaxation_parameter": 1.0}
    # petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    petsc_options = {"ksp_type": "cg", "pc_type": "gamg", "pc_gamg_coarse_eq_limit": 1000,
                     "mg_levels_ksp_type": "chebyshev", "mg_levels_pc_type": "jacobi",
                     "ksp_view": None}

    # Pack mesh data for Nitsche solver
    mesh_data = (facet_marker, top_value, bottom_value, surface_value, surface_bottom)

    # Solve contact problem using Nitsche's method
    u1 = nitsche_rigid_surface(mesh=mesh, mesh_data=mesh_data, physical_parameters=physical_parameters,
                               nitsche_parameters=nitsche_parameters, vertical_displacement=vertical_displacement,
                               nitsche_bc=nitsche_bc, quadrature_degree=3, petsc_options=petsc_options,
                               newton_options=newton_options)

    with XDMFFile(mesh.comm, "results/u_rigid.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u1)
