# Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse
import tempfile
from pathlib import Path

from mpi4py import MPI

import dolfinx.io.gmshio
import gmsh
import numpy as np
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx_contact.meshing import (
    convert_mesh_new,
    create_circle_circle_mesh,
    create_circle_plane_mesh,
    create_sphere_plane_mesh,
)
from dolfinx_contact.one_sided.nitsche_rigid_surface_custom import (
    nitsche_rigid_surface_custom,
)


def run_solver(
    theta=1.0,
    gamma=10.0,
    linear_solver=False,
    simplex=False,
    curved=False,
    threed=False,
    plane_strain=False,
    dirichlet=False,
    E=1e3,
    nu=0.1,
    disp=0.08,
    refs=1,
):
    # Current formulation uses unilateral contact
    nitsche_parameters = {"gamma": gamma, "theta": theta}
    # nitsche_bc = not dirichlet
    physical_parameters = {"E": E, "nu": nu, "strain": plane_strain}
    vertical_displacement = -disp
    # num_refs = refs + 1

    # Load mesh and create identifier functions for the top
    # (Displacement condition) and the bottom (contact condition)
    if threed:
        name = "demo_nitsche_rigid"
        model = gmsh.model(name)
        model.add(name)
        model.setCurrent(name)
        model = create_sphere_plane_mesh(model)
        mesh, _, facet_marker = dolfinx.io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=3)

        # with tempfile.TemporaryDirectory() as tmpdirname:
        #     fname = Path(tmpdirname, "sphere.msh")
        #     create_sphere_plane_mesh(filename=fname)
        #     convert_mesh_new(fname, fname.with_suffix(".xdmf"), gdim=3)
        #     with XDMFFile(MPI.COMM_WORLD, fname.with_suffix(".xdmf"), "r") as xdmf:
        #         mesh = xdmf.read_mesh()
        #         tdim = mesh.topology.dim
        #         mesh.topology.create_connectivity(tdim - 1, tdim)
        #         facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")

        # fname = f"{mesh_dir}/sphere"
        # create_sphere_plane_mesh(filename=f"{fname}.msh")
        # convert_mesh(fname, fname, gdim=3)
        # with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        #     mesh = xdmf.read_mesh()
        #     tdim = mesh.topology.dim
        #     mesh.topology.create_connectivity(tdim - 1, tdim)
        #     facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")
        top_value = 2
        bottom_value = 1
        surface_value = 8
        surface_bottom = 7

    else:
        if curved:
            with tempfile.TemporaryDirectory() as tmpdirname:
                fname = Path(tmpdirname, "two_disks.msh")
                create_circle_circle_mesh(filename=fname)
                convert_mesh_new(fname, fname.with_suffix(".xdmf"), gdim=2)
                with XDMFFile(MPI.COMM_WORLD, fname.with_suffix(".xdmf"), "r") as xdmf:
                    mesh = xdmf.read_mesh()

            # fname = f"{mesh_dir}/two_disks"
            # create_circle_circle_mesh(filename=f"{fname}.msh")
            # convert_mesh(fname, fname, gdim=2)
            # with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            #     mesh = xdmf.read_mesh()

            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, tdim)

            def top1(x):
                return x[1] > 0.55

            def bottom1(x):
                return np.logical_and(x[1] < 0.5, x[1] > 0.15)

            def top2(x):
                return np.logical_and(x[1] > -0.3, x[1] < 0.15)

            def bottom2(x):
                return x[1] < -0.35

            top_value = 1
            bottom_value = 2
            surface_value = 3
            surface_bottom = 4
            # Create meshtag for top and bottom markers
            top_facets1 = locate_entities_boundary(mesh, tdim - 1, top1)
            bottom_facets1 = locate_entities_boundary(mesh, tdim - 1, bottom1)
            top_facets2 = locate_entities_boundary(mesh, tdim - 1, top2)
            bottom_facets2 = locate_entities_boundary(mesh, tdim - 1, bottom2)
            top_values = np.full(len(top_facets1), top_value, dtype=np.int32)
            bottom_values = np.full(len(bottom_facets1), bottom_value, dtype=np.int32)

            surface_values = np.full(len(top_facets2), surface_value, dtype=np.int32)
            sbottom_values = np.full(len(bottom_facets2), surface_bottom, dtype=np.int32)
            indices = np.concatenate([top_facets1, bottom_facets1, top_facets2, bottom_facets2])
            values = np.hstack([top_values, bottom_values, surface_values, sbottom_values])
            sorted_facets = np.argsort(indices)
            facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])
        else:
            with tempfile.TemporaryDirectory() as tmpdirname:
                fname = Path(tmpdirname, "twomeshes.msh")
                create_circle_plane_mesh(
                    filename=fname,
                    quads=(not simplex),
                    res=0.05,
                    r=0.3,
                    gap=0.1,
                    height=0.1,
                    length=1.0,
                )
                convert_mesh_new(fname, fname.with_suffix(".xdmf"), gdim=2)
                with XDMFFile(MPI.COMM_WORLD, fname.with_suffix(".xdmf"), "r") as xdmf:
                    mesh = xdmf.read_mesh()
                    tdim = mesh.topology.dim
                    mesh.topology.create_connectivity(tdim - 1, tdim)
                    facet_marker = xdmf.read_meshtags(mesh, "facet_marker")

            def top(x):
                return x[1] > 0.0

            def bottom(x):
                return np.logical_and(x[1] < -0.05, x[1] > -0.35)

            top_value = 1
            bottom_value = 2
            surface_value = 3
            surface_bottom = 4
            # Create meshtag for top and bottom markers
            top_facets1 = locate_entities_boundary(mesh, tdim - 1, top)
            bottom_facets1 = locate_entities_boundary(mesh, tdim - 1, bottom)
            top_facets2 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], -0.4))
            bottom_facets2 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], -0.5))
            top_values = np.full(len(top_facets1), top_value, dtype=np.int32)
            bottom_values = np.full(len(bottom_facets1), bottom_value, dtype=np.int32)

            surface_values = np.full(len(top_facets2), surface_value, dtype=np.int32)
            sbottom_values = np.full(len(bottom_facets2), surface_bottom, dtype=np.int32)
            indices = np.concatenate([top_facets1, bottom_facets1, top_facets2, bottom_facets2])
            values = np.hstack([top_values, bottom_values, surface_values, sbottom_values])
            sorted_facets = np.argsort(indices)
            facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])

    # Solver options
    newton_options = {"relaxation_parameter": 1.0, "max_it": 50}
    # petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    petsc_options = {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "pc_gamg_coarse_eq_limit": 1000,
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "matptap_via": "scalable",
        "ksp_view": None,
    }

    # Pack mesh data for Nitsche solver
    mesh_data = (facet_marker, top_value, bottom_value, surface_value, surface_bottom)

    # Solve contact problem using Nitsche's method
    u1 = nitsche_rigid_surface_custom(
        mesh=mesh,
        mesh_data=mesh_data,
        physical_parameters=physical_parameters,
        nitsche_parameters=nitsche_parameters,
        vertical_displacement=vertical_displacement,
        nitsche_bc=True,
        quadrature_degree=3,
        petsc_options=petsc_options,
        newton_options=newton_options,
    )

    with XDMFFile(mesh.comm, "results/u_custom_rigid.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u1)


if __name__ == "__main__":
    desc = "Nitsche's method with rigid surface using custom assemblers"
    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    _simplex = parser.add_mutually_exclusive_group(required=False)
    _simplex.add_argument(
        "--simplex",
        dest="simplex",
        action="store_true",
        help="Use triangle/test mesh",
        default=False,
    )
    _curved = parser.add_mutually_exclusive_group(required=False)
    _curved.add_argument(
        "--curved",
        dest="curved",
        action="store_true",
        help="Use curved rigid surface",
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
    _E = parser.add_argument("--E", default=1e3, type=np.float64, dest="E", help="Youngs modulus of material")
    _nu = parser.add_argument("--nu", default=0.1, type=np.float64, dest="nu", help="Poisson's ratio")
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
        simplex=args.simplex,
        curved=args.curved,
        threed=args.threed,
        plane_strain=args.plane_strain,
        dirichlet=args.dirichlet,
        E=args.E,
        nu=args.nu,
        disp=args.disp,
        refs=args.refs,
    )


def test_nitsche_rigid_custom():
    run_solver()
