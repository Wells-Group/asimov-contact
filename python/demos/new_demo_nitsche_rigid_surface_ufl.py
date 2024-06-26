# Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse

from mpi4py import MPI

import dolfinx.io.gmshio
import gmsh
import numpy as np
from dolfinx.io import XDMFFile
from dolfinx_contact.meshing import (
    create_circle_plane_mesh,
    create_sphere_plane_mesh,
)
from dolfinx_contact.one_sided.nitsche_rigid_surface import nitsche_rigid_surface


def run_solver(
    theta=1.0,
    gamma=10.0,
    linear_solver=False,
    threed=False,
    plane_strain=False,
    dirichlet=False,
    E=1e3,
    nu=0.1,
    disp=0.08,
    refs=1,
):
    gmsh.initialize()

    # Current formulation uses unilateral contact
    nitsche_parameters = {"gamma": gamma, "theta": theta}
    nitsche_bc = not dirichlet
    physical_parameters = {"E": E, "nu": nu, "strain": plane_strain}
    vertical_displacement = -disp
    # num_refs = refs + 1
    top_value = 1
    bottom_value = 2

    # Load mesh and create identifier functions for the top
    # (Displacement condition) and the bottom (contact condition)
    if threed:
        name = "demo_nitche_rigid_custom"
        model = gmsh.model()
        model.add(name)
        model.setCurrent(name)
        model = create_sphere_plane_mesh(model)
        mesh, _, facet_marker = dolfinx.io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=3)

        top_value = 2
        bottom_value = 1
        surface_value = 8
        surface_bottom = 7
    else:
        name = "circle_plane"
        model = gmsh.model()
        model.add(name)
        model.setCurrent(name)
        model = create_circle_plane_mesh(model)
        mesh, _, facet_marker = dolfinx.io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=2)

        top_value = 2
        bottom_value = 4
        surface_value = 9
        surface_bottom = 7

    gmsh.finalize()

    # Solver options
    newton_options = {"relaxation_parameter": 1.0}
    # petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    petsc_options = {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "pc_gamg_coarse_eq_limit": 1000,
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "ksp_view": None,
    }

    # Pack mesh data for Nitsche solver
    mesh_data = (facet_marker, top_value, bottom_value, surface_value, surface_bottom)

    # Solve contact problem using Nitsche's method
    u1 = nitsche_rigid_surface(
        mesh=mesh,
        mesh_data=mesh_data,
        physical_parameters=physical_parameters,
        nitsche_parameters=nitsche_parameters,
        vertical_displacement=vertical_displacement,
        nitsche_bc=nitsche_bc,
        quadrature_degree=3,
        petsc_options=petsc_options,
        newton_options=newton_options,
    )

    with XDMFFile(mesh.comm, "results/u_rigid.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u1)


if __name__ == "__main__":
    desc = "Compare Nitsche's metood for contact against a straight plane with PETSc SNES"
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--theta",
        default=1,
        type=np.float64,
        dest="theta",
        help="Theta parameter for Nitsche, 1 symmetric, -1 skew symmetric, 0 Penalty-like",
    )
    parser.add_argument(
        "--gamma",
        default=10,
        type=np.float64,
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
        default=0.2,
        type=np.float64,
        dest="disp",
        help="Displacement BC in negative y direction",
    )
    _ref = parser.add_argument(
        "--refinements",
        default=2,
        type=np.int32,
        dest="refs",
        help="Number of mesh refinements",
    )

    # Parse input arguments or set to defualt values
    args = parser.parse_args()

    run_solver(
        theta=args.theta,
        gamma=args.gamma,
        linear_solver=args.linear_solver,
        threed=args.threed,
        plane_strain=args.plane_strain,
        dirichlet=args.dirichlet,
        E=args.E,
        nu=args.nu,
        disp=args.disp,
        refs=args.refs,
    )


def test_nitsche_rigid_ufl():
    run_solver()
