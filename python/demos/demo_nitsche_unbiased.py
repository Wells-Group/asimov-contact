# Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse

import numpy as np
from dolfinx import log
from dolfinx.common import TimingType, list_timings
from dolfinx.fem import Function, VectorFunctionSpace
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary, meshtags
from mpi4py import MPI

from dolfinx_contact import update_geometry
from dolfinx_contact.meshing import (convert_mesh, create_box_mesh_2D,
                                     create_box_mesh_3D,
                                     create_circle_circle_mesh,
                                     create_circle_plane_mesh,
                                     create_cylinder_cylinder_mesh,
                                     create_sphere_plane_mesh)
from dolfinx_contact.unbiased.nitsche_unbiased import nitsche_unbiased

if __name__ == "__main__":
    desc = "Nitsche's method with rigid surface using custom assemblers"
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--theta", default=1, type=np.float64, dest="theta",
                        help="Theta parameter for Nitsche, 1 symmetric, -1 skew symmetric, 0 Penalty-like",
                        choices=[1, -1, 0])
    parser.add_argument("--gamma", default=10, type=np.float64, dest="gamma",
                        help="Coercivity/Stabilization parameter for Nitsche condition")
    parser.add_argument("--quadrature", default=3, type=int, dest="q_degree",
                        help="Quadrature degree used for contact integrals")
    parser.add_argument("--problem", default=1, type=int, dest="problem",
                        help="Which problem to solve: 1. Flat surfaces, 2. One curved surface, 3. Two curved surfaces",
                        choices=[1, 2, 3])
    _3D = parser.add_mutually_exclusive_group(required=False)
    _3D.add_argument('--3D', dest='threed', action='store_true',
                     help="Use 3D mesh", default=False)
    _simplex = parser.add_mutually_exclusive_group(required=False)
    _simplex.add_argument('--simplex', dest='simplex', action='store_true',
                          help="Use triangle/tet mesh", default=False)
    _strain = parser.add_mutually_exclusive_group(required=False)
    _strain.add_argument('--strain', dest='plane_strain', action='store_true',
                         help="Use plane strain formulation", default=False)
    parser.add_argument("--E", default=1e3, type=np.float64, dest="E",
                        help="Youngs modulus of material")
    parser.add_argument("--nu", default=0.1, type=np.float64, dest="nu", help="Poisson's ratio")
    parser.add_argument("--disp", default=0.2, type=np.float64, dest="disp",
                        help="Displacement BC in negative y direction")
    parser.add_argument("--load_steps", default=1, type=np.int32, dest="nload_steps",
                        help="Number of steps for gradual loading")
    parser.add_argument("--res", default=0.1, type=np.float64, dest="res",
                        help="Mesh resolution")

    # Parse input arguments or set to defualt values
    args = parser.parse_args()

    # Current formulation uses unilateral contact
    nitsche_parameters = {"gamma": args.gamma, "theta": args.theta}
    physical_parameters = {"E": args.E, "nu": args.nu, "strain": args.plane_strain}
    threed = args.threed
    problem = args.problem
    nload_steps = args.nload_steps
    simplex = args.simplex

    # Load mesh and create identifier functions for the top (Displacement condition)
    # and the bottom (contact condition)
    if threed:
        displacement = ([0, 0, -args.disp], [0, 0, 0])
        if problem == 1:
            fname = "box_3D"
            create_box_mesh_3D(f"{fname}.msh", simplex)
            ct = "tetra" if simplex else "hexahedron"
            convert_mesh(fname, fname, ct)

            with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
                mesh = xdmf.read_mesh(name="Grid")
            tdim = mesh.topology.dim
            gdim = mesh.geometry.dim
            mesh.topology.create_connectivity(tdim - 1, 0)
            mesh.topology.create_connectivity(tdim - 1, tdim)

            dirichet_bdy_1 = 1
            contact_bdy_1 = 2
            contact_bdy_2 = 3
            dirichlet_bdy_2 = 4
            # Create meshtag for top and bottom markers
            top_facets1 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[2], 0.5))
            bottom_facets1 = locate_entities_boundary(
                mesh, tdim - 1, lambda x: np.isclose(x[2], 0.0))
            top_facets2 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[2], -0.1))
            bottom_facets2 = locate_entities_boundary(
                mesh, tdim - 1, lambda x: np.isclose(x[2], -0.6))
            top_values = np.full(len(top_facets1), dirichet_bdy_1, dtype=np.int32)
            bottom_values = np.full(
                len(bottom_facets1), contact_bdy_1, dtype=np.int32)

            surface_values = np.full(len(top_facets2), contact_bdy_2, dtype=np.int32)
            sbottom_values = np.full(
                len(bottom_facets2), dirichlet_bdy_2, dtype=np.int32)
            indices = np.concatenate([top_facets1, bottom_facets1, top_facets2, bottom_facets2])
            values = np.hstack([top_values, bottom_values, surface_values, sbottom_values])
            sorted_facets = np.argsort(indices)
            facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])

        elif problem == 2:
            fname = "sphere"
            create_sphere_plane_mesh(filename=f"{fname}.msh")
            convert_mesh(fname, fname, "tetra")
            convert_mesh(f"{fname}", f"{fname}_facets", "triangle")
            with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
                mesh = xdmf.read_mesh(name="Grid")
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, 0)
            mesh.topology.create_connectivity(tdim - 1, tdim)
            with XDMFFile(MPI.COMM_WORLD, f"{fname}_facets.xdmf", "r") as xdmf:
                facet_marker = xdmf.read_meshtags(mesh, name="Grid")
            dirichet_bdy_1 = 2
            contact_bdy_1 = 1
            contact_bdy_2 = 8
            dirichlet_bdy_2 = 7

        elif problem == 3:
            fname = "cylinder_cylinder_3D"
            displacement = ([-1, 0, 0], [0, 0, 0])
            create_cylinder_cylinder_mesh(fname, res=args.res, simplex=simplex)
            with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
                mesh = xdmf.read_mesh(name="cylinder_cylinder")
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, 0)
            mesh.topology.create_connectivity(tdim - 1, tdim)

            def right(x):
                return x[0] > 2.2

            def right_contact(x):
                return np.logical_and(x[0] < 2, x[0] > 1.45)

            def left_contact(x):
                return np.logical_and(x[0] > 0.25, x[0] < 1.1)

            def left(x):
                return x[0] < -0.5

            dirichet_bdy_1 = 1
            contact_bdy_1 = 2
            contact_bdy_2 = 3
            dirichlet_bdy_2 = 4
            # Create meshtag for top and bottom markers
            dirichlet_facets_1 = locate_entities_boundary(mesh, tdim - 1, right)
            contact_facets_1 = locate_entities_boundary(mesh, tdim - 1, right_contact)
            contact_facets_2 = locate_entities_boundary(mesh, tdim - 1, left_contact)
            dirchlet_facets_2 = locate_entities_boundary(mesh, tdim - 1, left)

            val0 = np.full(len(dirichlet_facets_1), dirichet_bdy_1, dtype=np.int32)
            val1 = np.full(len(contact_facets_1), contact_bdy_1, dtype=np.int32)
            val2 = np.full(len(contact_facets_2), contact_bdy_2, dtype=np.int32)
            val3 = np.full(len(dirchlet_facets_2), dirichlet_bdy_2, dtype=np.int32)
            indices = np.concatenate([dirichlet_facets_1, contact_facets_1, contact_facets_2, dirchlet_facets_2])
            values = np.hstack([val0, val1, val2, val3])
            sorted_facets = np.argsort(indices)
            facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])

    else:
        displacement = ([0, -args.disp], [0, 0])
        if problem == 1:
            fname = "box_2D"
            create_box_mesh_2D(filename=f"{fname}.msh", quads=not simplex, res=args.res)
            if simplex:
                convert_mesh(fname, f"{fname}.xdmf", "triangle", prune_z=True)
            else:
                convert_mesh(fname, f"{fname}.xdmf", "quad", prune_z=True)
            convert_mesh(f"{fname}", f"{fname}_facets", "line", prune_z=True)

            with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
                mesh = xdmf.read_mesh(name="Grid")
            tdim = mesh.topology.dim
            gdim = mesh.geometry.dim
            mesh.topology.create_connectivity(tdim - 1, 0)
            mesh.topology.create_connectivity(tdim - 1, tdim)
            with XDMFFile(MPI.COMM_WORLD, f"{fname}_facets.xdmf", "r") as xdmf:
                facet_marker = xdmf.read_meshtags(mesh, name="Grid")
            dirichet_bdy_1 = 5
            contact_bdy_1 = 3
            contact_bdy_2 = 9
            dirichlet_bdy_2 = 7

        elif problem == 2:
            fname = "twomeshes"
            if simplex:
                create_circle_plane_mesh(filename=f"{fname}.msh")
                convert_mesh(fname, f"{fname}.xdmf", "triangle", prune_z=True)
            else:
                create_circle_plane_mesh(filename=f"{fname}.msh", quads=True)
                convert_mesh(fname, f"{fname}.xdmf", "quad", prune_z=True)
            convert_mesh(f"{fname}", f"{fname}_facets", "line", prune_z=True)

            with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
                mesh = xdmf.read_mesh(name="Grid")
            tdim = mesh.topology.dim
            gdim = mesh.geometry.dim
            mesh.topology.create_connectivity(tdim - 1, 0)
            mesh.topology.create_connectivity(tdim - 1, tdim)
            with XDMFFile(MPI.COMM_WORLD, f"{fname}_facets.xdmf", "r") as xdmf:
                facet_marker = xdmf.read_meshtags(mesh, name="Grid")
            dirichet_bdy_1 = 2
            contact_bdy_1 = 4
            contact_bdy_2 = 9
            dirichlet_bdy_2 = 7
        elif problem == 3:
            fname = "two_disks"
            if simplex:
                create_circle_circle_mesh(filename=f"{fname}.msh", res=args.res)
                convert_mesh(fname, f"{fname}.xdmf", "triangle", prune_z=True)
            else:
                create_circle_circle_mesh(filename=f"{fname}.msh", quads=True, res=args.res)
                convert_mesh(fname, f"{fname}.xdmf", "quad", prune_z=True)
            convert_mesh(f"{fname}", f"{fname}_facets", "line", prune_z=True)

            with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
                mesh = xdmf.read_mesh(name="Grid")
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, 0)
            mesh.topology.create_connectivity(tdim - 1, tdim)

            def top_dir(x):
                return x[1] > 0.5

            def top_contact(x):
                return np.logical_and(x[1] < 0.49, x[1] > 0.11)

            def bottom_dir(x):
                return x[1] < -0.55

            def bottom_contact(x):
                return np.logical_and(x[1] > -0.45, x[1] < 0.1)

            dirichet_bdy_1 = 1
            contact_bdy_1 = 2
            contact_bdy_2 = 3
            dirichlet_bdy_2 = 4
            # Create meshtag for top and bottom markers
            top_facets1 = locate_entities_boundary(mesh, tdim - 1, top_dir)
            bottom_facets1 = locate_entities_boundary(mesh, tdim - 1, top_contact)
            top_facets2 = locate_entities_boundary(mesh, tdim - 1, bottom_contact)
            bottom_facets2 = locate_entities_boundary(mesh, tdim - 1, bottom_dir)
            dir_val1 = np.full(len(top_facets1), dirichet_bdy_1, dtype=np.int32)
            c_val1 = np.full(len(bottom_facets1), contact_bdy_1, dtype=np.int32)
            surface_values = np.full(len(top_facets2), contact_bdy_2, dtype=np.int32)
            sbottom_values = np.full(len(bottom_facets2), dirichlet_bdy_2, dtype=np.int32)
            indices = np.concatenate([top_facets1, bottom_facets1, top_facets2, bottom_facets2])
            values = np.hstack([dir_val1, c_val1, surface_values, sbottom_values])
            sorted_facets = np.argsort(indices)

            facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])

    with XDMFFile(mesh.comm, "test.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_marker)

    # Solver options
    newton_options = {"relaxation_parameter": 1, "atol": 1e-8, "rtol": 1e-8, "convergence_criterion": "residual"}
    # petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    petsc_options = {"ksp_type": "cgs", "pc_type": "gamg", "pc_gamg_type": "agg", "pc_gamg_coarse_eq_limit": 1000,
                     "pc_gamg_agg_nsmooths": 2,
                     "pc_gamg_sym_graph": True, "mg_levels_ksp_type": "chebyshev", "mg_levels_pc_type": "jacobi",
                     "matptap_via": "scalable", "pc_gamg_square_graph": 2,
                     "pc_gamg_threshold": 1e-1}  # , "ksp_view": None}
    # Add if mg_levels_pc_type: sor
    # "mg_levels_esteig_ksp_type": "cg",
    # Pack mesh data for Nitsche solver
    mesh_data = (facet_marker, dirichet_bdy_1, contact_bdy_1, contact_bdy_2, dirichlet_bdy_2)

    # Solve contact problem using Nitsche's method
    load_increment = np.array(displacement) / nload_steps

    # Define function space for problem
    V = VectorFunctionSpace(mesh, ("CG", 1))
    u1 = None

    # Data to be stored on the unperturb domain at the end of the simulation
    u = Function(V)
    u.x.array[:] = np.zeros(u.x.array[:].shape)
    geometry = mesh.geometry.x[:].copy()

    log.set_log_level(log.LogLevel.OFF)
    num_newton_its = np.zeros(nload_steps, dtype=int)
    num_krylov_its = np.zeros(nload_steps, dtype=int)

    # Load geometry over multiple steps
    for j in range(nload_steps):
        displacement = load_increment

        # Solve contact problem using Nitsche's method
        u1, n, krylov_iterations = nitsche_unbiased(
            mesh=mesh, mesh_data=mesh_data, physical_parameters=physical_parameters,
            nitsche_parameters=nitsche_parameters, displacement=displacement,
            quadrature_degree=args.q_degree, petsc_options=petsc_options,
            newton_options=newton_options)
        num_newton_its[j] = n
        num_krylov_its[j] = krylov_iterations
        with XDMFFile(mesh.comm, f"results/u_unbiased_{j}.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            u1.name = "u"
            xdmf.write_function(u1)

        # Perturb mesh with solution displacement
        update_geometry(u1._cpp_object, mesh)

        # Accumulate displacements
        u.x.array[:] += u1.x.array[:]

    # Reset mesh to initial state and write accumulated solution
    mesh.geometry.x[:] = geometry
    with XDMFFile(mesh.comm, "results/u_unbiased_total.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        u.name = "u"
        xdmf.write_function(u)
    list_timings(mesh.comm, [TimingType.wall])

    print(f"Newton iterations {num_newton_its}, {sum(num_newton_its)}")
    print(f"Krylov iterations {num_krylov_its}, {sum(num_krylov_its)}")
    print(f"Petsc options {petsc_options}")
    print(f"Newton options {newton_options}")
    print(f"Krylov/Newton: {num_krylov_its/num_newton_its}")
    print(f"Krylov/Newton accumulated: {sum(num_krylov_its)/sum(num_newton_its)}")
