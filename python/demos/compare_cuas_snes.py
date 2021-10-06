# Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse

import dolfinx
import dolfinx.io
import numpy as np
import ufl
from mpi4py import MPI

from dolfinx_contact.create_mesh import create_disk_mesh, create_sphere_mesh, convert_mesh
from dolfinx_contact.nitsche_cuas import nitsche_cuas
from dolfinx_contact.snes_against_plane import snes_solver


if __name__ == "__main__":
    description = "Compare Nitsche's method for contact against a straight plane with PETSc SNES"
    parser = argparse.ArgumentParser(description=description,
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
    _cube = parser.add_mutually_exclusive_group(required=False)
    _cube.add_argument('--cube', dest='cube', action='store_true',
                       help="Use Cube/Square", default=False)
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
    _ref = parser.add_argument("--refinements", default=1, type=np.int32,
                               dest="refs", help="Number of mesh refinements")
    _gap = parser.add_argument(
        "--gap", default=0.02, type=np.float64, dest="gap", help="Gap between plane and y=0")

    # Parse input arguments or set to defualt values
    args = parser.parse_args()

    # Current formulation uses unilateral contact
    nitsche_parameters = {"gamma": args.gamma, "theta": args.theta}
    nitsche_bc = not args.dirichlet
    physical_parameters = {"E": args.E, "nu": args.nu, "strain": args.plane_strain}
    vertical_displacement = -args.disp
    num_refs = args.refs + 1
    gap = args.gap
    top_value = 1
    threed = args.threed
    bottom_value = 2
    cube = args.cube

    # Load mesh and create identifier functions for the top (Displacement condition)
    # and the bottom (contact condition)
    if threed:
        if cube:
            mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 10, 10, 20)
        else:
            fname = "sphere"
            create_sphere_mesh(filename=f"{fname}.msh")
            convert_mesh(fname, "tetra")
            with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
                mesh = xdmf.read_mesh(name="Grid")

        def top(x):
            return x[2] > 0.9

        def bottom(x):
            return x[2] < 0.15

    else:
        if cube:
            mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 30, 30)
        else:
            fname = "disk"
            create_disk_mesh(filename=f"{fname}.msh")
            convert_mesh(fname, "triangle", prune_z=True)
            with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
                mesh = xdmf.read_mesh(name="Grid")

        def top(x):
            return x[1] > 0.5

        def bottom(x):
            return x[1] < 0.2

    e_abs = []
    e_rel = []
    dofs_global = []
    rank = MPI.COMM_WORLD.rank
    refs = np.arange(0, num_refs)
    for i in refs:
        if i > 0:
            # Refine mesh
            mesh.topology.create_entities(mesh.topology.dim - 2)
            mesh = dolfinx.mesh.refine(mesh)

        # Create meshtag for top and bottom markers
        tdim = mesh.topology.dim
        top_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, top)
        bottom_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, tdim - 1, bottom)
        top_values = np.full(len(top_facets), top_value, dtype=np.int32)
        bottom_values = np.full(
            len(bottom_facets), bottom_value, dtype=np.int32)
        indices = np.concatenate([top_facets, bottom_facets])
        values = np.hstack([top_values, bottom_values])
        sorted_facets = np.argsort(indices)
        facet_marker = dolfinx.MeshTags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])
        mesh_data = (facet_marker, top_value, bottom_value)

        # Solve contact problem using Nitsche's method
        u1 = nitsche_cuas(mesh=mesh, mesh_data=mesh_data, physical_parameters=physical_parameters,
                          vertical_displacement=vertical_displacement, nitsche_parameters=nitsche_parameters,
                          refinement=i, g=gap, nitsche_bc=nitsche_bc)
        # Solve contact problem using PETSc SNES
        u2 = snes_solver(mesh=mesh, mesh_data=mesh_data, physical_parameters=physical_parameters,
                         vertical_displacement=vertical_displacement, refinement=i, g=gap)

        # Compute the difference (error) between Nitsche and SNES
        V = u1.function_space
        dx = ufl.Measure("dx", domain=mesh)
        error = ufl.inner(u1 - u2, u1 - u2) * dx
        E_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(dolfinx.fem.assemble_scalar(error), op=MPI.SUM))
        u2_norm = ufl.inner(u2, u2) * dx
        u2_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(dolfinx.fem.assemble_scalar(u2_norm), op=MPI.SUM))
        if rank == 0:
            print(f"abs. L2-error={E_L2:.2e}")
            print(f"rel. L2-error={E_L2/u2_L2:.2e}")
        e_abs.append(E_L2)
        e_rel.append(E_L2 / u2_L2)
        dofs_global.append(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)
        error_fn = dolfinx.Function(V)
        error_fn.vector[:] = u1.x.array - u2.x.array
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"results/u_error_{i}.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(error_fn)

    # Output absolute and relative errors of Nitsche compared to SNES
    if rank == 0:
        print(f"Num dofs {dofs_global}")
        print(f"Absolute error {e_abs}")
        print(f"Relative error {e_rel}")
    for i in refs:
        nitsche_timings = dolfinx.cpp.common.timing(f'{i} Solve Nitsche')
        snes_timings = dolfinx.cpp.common.timing(f'{i} Solve SNES')
        if rank == 0:
            print(f"{dofs_global[i]}, Nitsche: {nitsche_timings[1]: 0.2e}"
                  + f" SNES: {snes_timings[1]:0.2e}")
    assert(e_rel[-1] < 1e-3)
    assert(e_abs[-1] < 1e-4)
