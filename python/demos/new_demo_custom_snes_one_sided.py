# Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT
#
# Compare our own Nitsche implementation using custom integration kernels with SNES

import argparse
import os

from mpi4py import MPI

import numpy as np
import ufl
from dolfinx.common import timing
from dolfinx.fem import Function, assemble_scalar, form
from dolfinx.io import XDMFFile
from dolfinx.mesh import (
    create_unit_cube,
    create_unit_square,
    locate_entities_boundary,
    meshtags,
    refine,
)
from dolfinx_contact.meshing import convert_mesh, create_disk_mesh, create_sphere_mesh
from dolfinx_contact.one_sided.nitsche_custom import nitsche_custom
from dolfinx_contact.one_sided.snes_against_plane import snes_solver


def solver(
    theta=1.0,
    gamma=10.0,
    # linear_solver=False,
    threed=False,
    cube=False,
    plane_strain=False,
    dirichlet=False,
    E=1e3,
    nu=0.1,
    disp=0.08,
    refs=1,
    gap=0.02,
):
    # Current formulation uses unilateral contact
    nitsche_parameters = {"gamma": gamma, "theta": theta}
    nitsche_bc = not dirichlet
    physical_parameters = {"E": E, "nu": nu, "strain": plane_strain}
    vertical_displacement = -disp
    num_refs = refs + 1
    top_value = 1
    bottom_value = 2

    petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    # petsc_options = {
    #     "ksp_type": "cg",
    #     "pc_type": "gamg",
    #     "rtol": 1e-6,
    #     "pc_gamg_coarse_eq_limit": 1000,
    #     "mg_levels_ksp_type": "chebyshev",
    #     "mg_levels_pc_type": "jacobi",
    #     "mg_levels_esteig_ksp_type": "cg",
    #     "matptap_via": "scalable",
    #     "ksp_view": None,
    # }

    snes_options = {
        "snes_monitor": None,
        "snes_max_it": 50,
        "snes_max_fail": 10,
        "snes_type": "vinewtonrsls",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-9,
        "snes_view": None,
    }
    # Cannot use GAMG with SNES, see: https://gitlab.com/petsc/petsc/-/issues/829
    petsc_snes = {"ksp_type": "cg", "ksp_rtol": 1e-5, "pc_type": "jacobi"}
    # Load mesh and create identifier functions for the top
    # (Displacement condition) and the bottom (contact condition)
    outdir = "meshes"
    os.system(f"mkdir -p {outdir}")
    if threed:
        if cube:
            mesh = create_unit_cube(MPI.COMM_WORLD, 10, 10, 20)
        else:
            fname = f"{outdir}/sphere"
            create_sphere_mesh(filename=f"{fname}.msh")
            convert_mesh(fname, fname, gdim=3)
            with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
                mesh = xdmf.read_mesh()

        def top(x):
            return x[2] > 0.9

        def bottom(x):
            return x[2] < 0.15

    else:
        if cube:
            mesh = create_unit_square(MPI.COMM_WORLD, 30, 30)
        else:
            fname = f"{outdir}/disk"
            create_disk_mesh(filename=f"{fname}.msh")
            convert_mesh(fname, fname, gdim=2)
            with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
                mesh = xdmf.read_mesh()

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
            mesh, _, _ = refine(mesh)

        # Create meshtag for top and bottom markers
        tdim = mesh.topology.dim
        top_facets = locate_entities_boundary(mesh, tdim - 1, top)
        bottom_facets = locate_entities_boundary(mesh, tdim - 1, bottom)
        top_values = np.full(len(top_facets), top_value, dtype=np.int32)
        bottom_values = np.full(len(bottom_facets), bottom_value, dtype=np.int32)
        indices = np.concatenate([top_facets, bottom_facets])
        values = np.hstack([top_values, bottom_values])
        sorted_facets = np.argsort(indices)
        facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])
        mesh_data = (facet_marker, top_value, bottom_value)

        # Solve contact problem using Nitsche's method
        u1 = nitsche_custom(
            mesh=mesh,
            mesh_data=mesh_data,
            physical_parameters=physical_parameters,
            vertical_displacement=vertical_displacement,
            nitsche_parameters=nitsche_parameters,
            plane_loc=gap,
            nitsche_bc=nitsche_bc,
            petsc_options=petsc_options,
        )
        with XDMFFile(mesh.comm, f"results/u_custom_{i}.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(u1)

        # Solve contact problem using PETSc SNES
        u2 = snes_solver(
            mesh=mesh,
            mesh_data=mesh_data,
            physical_parameters=physical_parameters,
            vertical_displacement=vertical_displacement,
            plane_loc=gap,
            petsc_options=petsc_snes,
            snes_options=snes_options,
        )
        with XDMFFile(mesh.comm, f"results/u_snes_{i}.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(u2)
        # Compute the difference (error) between Nitsche and SNES
        V = u1.function_space
        dx = ufl.Measure("dx", domain=mesh)
        error = form(ufl.inner(u1 - u2, u1 - u2) * dx)
        E_L2 = np.sqrt(mesh.comm.allreduce(assemble_scalar(error), op=MPI.SUM))
        u2_norm = form(ufl.inner(u2, u2) * dx)
        u2_L2 = np.sqrt(mesh.comm.allreduce(assemble_scalar(u2_norm), op=MPI.SUM))
        if rank == 0:
            print(f"abs. L2-error={E_L2:.2e}")
            print(f"rel. L2-error={E_L2 / u2_L2:.2e}")
        e_abs.append(E_L2)
        e_rel.append(E_L2 / u2_L2)
        dofs_global.append(V.dofmap.index_map.size_global * V.dofmap.index_map_bs)
        error_fn = Function(V)
        error_fn.x.array[:] = u1.x.array - u2.x.array
        with XDMFFile(MPI.COMM_WORLD, f"results/u_error_{i}.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_function(error_fn)

    # Output absolute and relative errors of Nitsche compared to SNES
    if rank == 0:
        print(f"Num dofs {dofs_global}")
        print(f"Absolute error {e_abs}")
        print(f"Relative error {e_rel}")
    for i in refs:
        nitsche_timings = timing(f"{dofs_global[i]} Solve Nitsche")
        snes_timings = timing(f"{dofs_global[i]} Solve SNES")
        if rank == 0:
            print(
                f"{dofs_global[i]}, Nitsche: {nitsche_timings[1]: 0.2e}"
                + f" SNES: {snes_timings[1]:0.2e}"
            )
    assert e_rel[-1] < 1e-3
    assert e_abs[-1] < 1e-4


if __name__ == "__main__":
    description = "Compare Nitsche's method for contact against a straight plane with PETSc SNES"
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--theta",
        default=1.0,
        type=float,
        dest="theta",
        choices=[-1.0, 0.0, 1.0],
        help="Theta parameter for Nitsche: skew symmetric (-1), Penalty-like (0), symmetric (1)",
    )
    parser.add_argument(
        "--gamma",
        default=10.0,
        type=float,
        dest="gamma",
        help="Coercivity/Stabilization parameter for Nitsche condition",
    )
    _solve = parser.add_mutually_exclusive_group(required=False)
    _solve.add_argument(
        "--linear",
        dest="linear_solver",
        action="store_true",
        help="Use linear solver",
        default=False,
    )
    _3D = parser.add_mutually_exclusive_group(required=False)
    _3D.add_argument("--3D", dest="threed", action="store_true", help="Use 3D mesh", default=False)
    _cube = parser.add_mutually_exclusive_group(required=False)
    _cube.add_argument(
        "--cube",
        dest="cube",
        action="store_true",
        help="Use Cube/Square",
        default=False,
    )
    _strain = parser.add_mutually_exclusive_group(required=False)
    _strain.add_argument(
        "--strain",
        dest="plane_strain",
        action="store_true",
        help="Use plane strain formulation",
        default=False,
    )
    _dirichlet = parser.add_mutually_exclusive_group(required=False)
    _dirichlet.add_argument(
        "--dirichlet",
        dest="dirichlet",
        action="store_true",
        help="Use strong Dirichlet formulation",
        default=False,
    )
    _E = parser.add_argument(
        "--E", default=1e3, type=np.float64, dest="E", help="Youngs modulus of material"
    )
    _nu = parser.add_argument(
        "--nu", default=0.1, type=np.float64, dest="nu", help="Poisson's ratio"
    )
    _disp = parser.add_argument(
        "--disp",
        default=0.08,
        type=np.float64,
        dest="disp",
        help="Displacement BC in negative y direction",
    )
    _ref = parser.add_argument(
        "--refinements",
        default=1,
        type=np.int32,
        dest="refs",
        help="Number of mesh refinements",
    )
    _gap = parser.add_argument(
        "--gap",
        default=0.02,
        type=np.float64,
        dest="gap",
        help="Gap between plane and y=0",
    )

    # Parse input arguments or set to defualt values
    args = parser.parse_args()

    solver(
        theta=args.theta,
        gamma=args.gamma,
        # linear_solver=args.linear_solver,
        threed=args.threed,
        cube=args.cube,
        plane_strain=args.plane_strain,
        dirichlet=args.dirichlet,
        E=args.E,
        nu=args.nu,
        disp=args.disp,
        refs=args.refs,
        gap=args.gap,
    )


def test_custom_snes():
    solver(
        theta=1.0,
        gamma=10.0,
        threed=False,
        cube=False,
        plane_strain=False,
        dirichlet=False,
        E=1e3,
        nu=0.1,
        disp=0.08,
        refs=1,
        gap=0.02,
    )
