# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse
import sys

import numpy as np
from dolfinx import log
import dolfinx.fem as _fem
from dolfinx.common import timing, Timer
from dolfinx.graph import create_adjacencylist
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary, GhostMode, meshtags
from mpi4py import MPI
from petsc4py import PETSc as _PETSc
import ufl

from dolfinx_contact.meshing import convert_mesh, create_christmas_tree_mesh_3D
from dolfinx_contact.unbiased.nitsche_unbiased import nitsche_unbiased
from dolfinx_contact.helpers import lame_parameters, sigma_func, weak_dirichlet, epsilon
from dolfinx_contact.parallel_mesh_ghosting import create_contact_mesh
from dolfinx_contact.cpp import ContactMode

if __name__ == "__main__":
    desc = "Nitsche's method for two elastic bodies using custom assemblers"
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--quadrature", default=5, type=int, dest="q_degree",
                        help="Quadrature degree used for contact integrals")
    parser.add_argument("--res", default=0.2, type=np.float64, dest="res",
                        help="Mesh resolution")

    # Parse input arguments or set to defualt values
    args = parser.parse_args()
    mesh_dir = "meshes"
    fname = f"{mesh_dir}/xmas_tree"

    # Call function that creates mesh using gmsh
    create_christmas_tree_mesh_3D(filename=fname, res=args.res, n1=81, n2=41)

    # Convert gmsh output into xdmf
    convert_mesh(fname, fname, gdim=3)

    # Read in mesh from xdmf file including markers
    # Cell markers:  It is expected that all cells are marked and that
    #                disconnected domains have different markers
    #                This is needed for defining the (near-)nullspace
    #                Currently we have the following cell markers:
    #                Outer box: 2
    #                Tree: 1
    #                These markers are handled automatically and the input to
    #                the contact code does not have to be changed if the
    #                marker values change.
    # Facet markers: This must include markers used for defining boundary
    #                conditions and markers for the contact surfaces
    #                Currently we have the follwoing facet markers:
    #                Dirichlet boundary (outer box): 4
    #                Neumann boundary (bottom of tree): 3
    #                Front/back of tree: 5 (part of this will be used
    #                to constrain rigid body movement in z direction)
    #                Contact surface 1 (tree surface): 6
    #                Contact surface 2 (outer box): 7
    #                If the values of these markers change, the input
    #                to the contact code has to be adjusted

    # When using different tags, change values here
    neumann_bdy = 5
    dirichlet_bdy = 4
    surface_1 = 6
    surface_2 = 7
    z_Dirichlet = 8  # this tag is defined further down, use value different from all
    # input markers

    with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(ghost_mode=GhostMode.none)
        domain_marker = xdmf.read_meshtags(mesh, "cell_marker")
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facet_marker = xdmf.read_meshtags(mesh, "facet_marker")

    # Call function that repartitions mesh for parallel computation
    if mesh.comm.size > 1:
        mesh, facet_marker, domain_marker = create_contact_mesh(
            mesh, facet_marker, domain_marker, [6, 7])

    V = _fem.VectorFunctionSpace(mesh, ("Lagrange", 1))

    # Apply zero Dirichlet boundary conditions in z-direction on part of the xmas-tree
    # Find facets for z-Dirichlet bc
    def identifier(x, z):
        return np.logical_and(np.logical_and(np.isclose(x[2], z),
                                             abs(x[1]) < 0.1), abs(x[0] - 2) < 0.1)
    dirichlet_facets1 = locate_entities_boundary(mesh, tdim - 1, lambda x: identifier(x, 0.0))
    dirichlet_facets2 = locate_entities_boundary(mesh, tdim - 1, lambda x: identifier(x, 1.0))

    # create new facet_marker including z Dirichlet facets
    indices = np.hstack([facet_marker.indices, dirichlet_facets1, dirichlet_facets2])
    values = np.hstack([facet_marker.values, z_Dirichlet * np.ones(len(dirichlet_facets1)
                        + len(dirichlet_facets2), dtype=np.int32)])
    sorted_facets = np.argsort(indices)
    facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])

    dirichlet_dofs = _fem.locate_dofs_topological(V.sub(2), tdim - 1, indices[sorted_facets])
    # Create Dirichlet bdy conditions for preventing rigid body motion in z-direction
    bcs = ([(dirichlet_dofs, 2)], [_fem.Constant(mesh, _PETSc.ScalarType(0))])

    # Functions for Dirichlet and Neuman boundaries, body force
    g = _fem.Constant(mesh, _PETSc.ScalarType((0, 0, 0)))      # zero dirichlet
    t = _fem.Constant(mesh, _PETSc.ScalarType((0.2, 0.5, 0)))  # traction
    f = _fem.Constant(mesh, _PETSc.ScalarType((1.0, 0.5, 0)))  # body force

    ncells = mesh.topology.index_map(tdim).size_local
    indices = np.array(range(ncells), dtype=np.int32)
    values = mesh.comm.rank * np.ones(ncells, dtype=np.int32)
    process_marker = meshtags(mesh, tdim, indices, values)
    process_marker.name = "process_marker"
    gdim = mesh.geometry.dim

    # contact surfaces with tags from marker_offset to marker_offset + 4 * split (split = #segments)
    contact_pairs = [(0, 1), (1, 0)]
    data = np.array([surface_1, surface_2], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    surfaces = create_adjacencylist(data, offsets)

    # zero dirichlet boundary condition on mesh boundary with tag 5

    # Function, TestFunction, TrialFunction and measures
    u = _fem.Function(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)

    # Compute lame parameters
    E = 1e3
    nu = 0.1
    mu_func, lambda_func = lame_parameters(False)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    # Create variational form without contact contributions
    F = ufl.inner(sigma(u), epsilon(v)) * dx

    # Apply weak Dirichlet boundary conditions using Nitsche's method
    gamma = 10
    theta = 1
    F = weak_dirichlet(F, u, g, sigma, E * gamma, theta, ds(4))

    # traction (neumann) boundary condition on mesh boundary with tag 3
    F -= ufl.inner(t, v) * ds(3)

    # body forces
    F -= ufl.inner(f, v) * dx(1)

    # create initial guess
    tree_cells = domain_marker.find(1)

    def _u_initial(x):
        values = np.zeros((gdim, x.shape[1]))
        for i in range(x.shape[1]):
            values[0, i] = 0.1
        return values

    u.interpolate(_u_initial, tree_cells)

    # Solver options
    ksp_tol = 1e-10
    newton_tol = 1e-7
    newton_options = {"relaxation_parameter": 1.0,
                      "atol": newton_tol,
                      "rtol": newton_tol,
                      "convergence_criterion": "residual",
                      "max_it": 50,
                      "error_on_nonconvergence": False}

    # In order to use an LU solver for debugging purposes on small scale problems
    # use the following PETSc options: {"ksp_type": "preonly", "pc_type": "lu"}
    petsc_options = {
        "matptap_via": "scalable",
        "ksp_type": "cg",
        "ksp_rtol": ksp_tol,
        "ksp_atol": ksp_tol,
        "pc_type": "gamg",
        "pc_mg_levels": 3,
        "pc_mg_cycles": 1,   # 1 is v, 2 is w
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "pc_gamg_type": "agg",
        "pc_gamg_coarse_eq_limit": 100,
        "pc_gamg_agg_nsmooths": 1,
        "pc_gamg_threshold": 1e-3,
        "pc_gamg_square_graph": 2,
        "pc_gamg_reuse_interpolation": False
    }

    # Solve contact problem using Nitsche's method
    problem_parameters = {"gamma": np.float64(E * gamma), "theta": np.float64(theta)}
    V0 = _fem.FunctionSpace(mesh, ("DG", 0))
    mu0 = _fem.Function(V0)
    lmbda0 = _fem.Function(V0)
    mu0.interpolate(lambda x: np.full((1, x.shape[1]), mu))
    lmbda0.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))
    solver_outfile = None
    log.set_log_level(log.LogLevel.WARNING)
    rhs_fns = [g, t, f]
    size = mesh.comm.size
    outname = f"results/xmas_{tdim}D_{size}"
    search_mode = [ContactMode.ClosestPoint for i in range(len(contact_pairs))]

    cffi_options = ["-Ofast", "-march=native"]
    jit_options = {"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]}
    with Timer("~Contact: - all"):
        u1, num_its, krylov_iterations, solver_time = nitsche_unbiased(1, ufl_form=F, u=u,
                                                                       mu=mu0, lmbda=lmbda0,
                                                                       rhs_fns=rhs_fns, markers=[
                                                                           domain_marker, facet_marker],
                                                                       contact_data=(surfaces, contact_pairs),
                                                                       bcs=bcs, problem_parameters=problem_parameters,
                                                                       search_method=search_mode,
                                                                       newton_options=newton_options,
                                                                       petsc_options=petsc_options,
                                                                       jit_options=jit_options,
                                                                       outfile=solver_outfile,
                                                                       fname=outname,
                                                                       quadrature_degree=args.q_degree,
                                                                       search_radius=np.float64(-1))

    # write solution to file
    size = mesh.comm.size
    with XDMFFile(mesh.comm, f"results/xmas_{size}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        u1.name = f"u_{size}"
        xdmf.write_function(u1)
    with XDMFFile(mesh.comm, f"results/xmas_partitioning_{size}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(process_marker, mesh.geometry)

    outfile = sys.stdout

    if mesh.comm.rank == 0:
        print("-" * 25, file=outfile)
        print(f"Newton options {newton_options}", file=outfile)
        print(f"num_dofs: {u1.function_space.dofmap.index_map_bs*u1.function_space.dofmap.index_map.size_global}"
              + f", {mesh.topology.cell_types[0]}", file=outfile)
        print(f"Newton solver {timing('~Contact: Newton (Newton solver)')[1]}", file=outfile)
        print(f"Krylov solver {timing('~Contact: Newton (Krylov solver)')[1]}", file=outfile)
        print(f"Newton time: {solver_time}", file=outfile)
        print(f"Newton iterations {num_its}, ", file=outfile)
        print(f"Krylov iterations {krylov_iterations},", file=outfile)
        print("-" * 25, file=outfile)
