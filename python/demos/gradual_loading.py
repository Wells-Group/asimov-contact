import argparse

import numpy as np
from dolfinx.fem import Function, VectorFunctionSpace
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary, meshtags
from mpi4py import MPI

from dolfinx_contact import update_geometry
from dolfinx_contact.meshing import (convert_mesh, create_circle_circle_mesh,
                                     create_circle_plane_mesh,
                                     create_sphere_plane_mesh)
from dolfinx_contact.one_sided.nitsche_rigid_surface_custom import \
    nitsche_rigid_surface_custom

if __name__ == "__main__":
    desc = "Nitsche's method with rigid surface using custom assemblers and apply gradual loading in non-linear solve"
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
    _curved = parser.add_mutually_exclusive_group(required=False)
    _curved.add_argument('--curved', dest='curved', action='store_true',
                         help="Use curved rigid surface", default=False)
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
    _nload_steps = parser.add_argument("--load_steps", default=1, type=np.int32, dest="nload_steps",
                                       help="Number of steps for gradual loading")

    # Parse input arguments or set to defualt values
    args = parser.parse_args()

    # Current formulation uses unilateral contact
    nitsche_parameters = {"gamma": args.gamma, "theta": args.theta}
    nitsche_bc = not args.dirichlet
    physical_parameters = {"E": args.E, "nu": args.nu, "strain": args.plane_strain}
    vertical_displacement = -args.disp
    top_value = 1
    threed = args.threed
    bottom_value = 2
    nload_steps = args.nload_steps
    curved = args.curved
    # Load mesh and create identifier functions for the top (Displacement condition)
    # and the bottom (contact condition)
    mesh_dir = "meshes"
    if threed:
        fname = f"{mesh_dir}/sphere"
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
        if curved:
            fname = f"{mesh_dir}/two_disks"
            create_circle_circle_mesh(filename=f"{fname}.msh")
            convert_mesh(fname, fname, gdim=2)

            with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
                mesh = xdmf.read_mesh()
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, 0)
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
            fname = f"{mesh_dir}/twomeshes"
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

            def top(x):
                return x[1] > 0.5

            def bottom(x):
                return np.logical_and(x[1] < 0.45, x[1] > 0.15)

            top_value = 1
            bottom_value = 2
            surface_value = 3
            surface_bottom = 4
            # Create meshtag for top and bottom markers
            top_facets1 = locate_entities_boundary(mesh, tdim - 1, top)
            bottom_facets1 = locate_entities_boundary(mesh, tdim - 1, bottom)
            top_facets2 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], 0.1))
            bottom_facets2 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], 0.0))
            top_values = np.full(len(top_facets1), top_value, dtype=np.int32)
            bottom_values = np.full(len(bottom_facets1), bottom_value, dtype=np.int32)

            surface_values = np.full(len(top_facets2), surface_value, dtype=np.int32)
            sbottom_values = np.full(len(bottom_facets2), surface_bottom, dtype=np.int32)
            indices = np.concatenate([top_facets1, bottom_facets1, top_facets2, bottom_facets2])
            values = np.hstack([top_values, bottom_values, surface_values, sbottom_values])
            sorted_facets = np.argsort(indices)
            facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])

    # Solver options
    newton_options = {"relaxation_parameter": 1.0}
    # petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    petsc_options = {"ksp_type": "cg", "pc_type": "gamg", "rtol": 1e-6, "pc_gamg_coarse_eq_limit": 1000,
                     "mg_levels_ksp_type": "chebyshev", "mg_levels_pc_type": "jacobi",
                     "mg_levels_esteig_ksp_type": "cg", "matptap_via": "scalable", "ksp_view": None}

    # Pack mesh data for Nitsche solver
    mesh_data = (facet_marker, top_value, bottom_value, surface_value, surface_bottom)

    # Solve contact problem using Nitsche's method
    load_increment = vertical_displacement / nload_steps

    # Define function space for problem
    V = VectorFunctionSpace(mesh, ("Lagrange", 1))
    u1 = None

    # Data to be stored on the unperturb domain at the end of the simulation
    u = Function(V)
    u.x.array[:] = np.zeros(u.x.array[:].shape)
    geometry = mesh.geometry.x[:].copy()

    for j in range(nload_steps):
        displacement = load_increment

        # Solve contact problem using Nitsche's method
        u1 = nitsche_rigid_surface_custom(mesh=mesh, mesh_data=mesh_data, physical_parameters=physical_parameters,
                                          nitsche_parameters=nitsche_parameters, vertical_displacement=displacement,
                                          nitsche_bc=True, quadrature_degree=3, petsc_options=petsc_options,
                                          newton_options=newton_options)
        with XDMFFile(mesh.comm, f"results/u_custom_{j}.xdmf", "w") as xdmf:
            xdmf.write_mesh(mesh)
            u1.name = "u"
            xdmf.write_function(u1)

        # Perturb mesh with solution displacement
        update_geometry(u1._cpp_object, mesh._cpp_object)

        # Accumulate displacements
        u.x.array[:] += u1.x.array[:]

    # Reset mesh to initial state and write accumulated solution
    mesh.geometry.x[:] = geometry
    with XDMFFile(mesh.comm, "results/u_custom_total.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        u.name = "u"
        xdmf.write_function(u)
