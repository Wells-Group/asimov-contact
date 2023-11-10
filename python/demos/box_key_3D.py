# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse
import sys

import dolfinx.fem as _fem
import numpy as np
import ufl
from dolfinx import default_scalar_type, log
from dolfinx.common import Timer, TimingType, list_timings, timing
from dolfinx.graph import adjacencylist
from dolfinx.io import XDMFFile
from dolfinx.mesh import GhostMode, meshtags
from dolfinx_contact.helpers import (epsilon, lame_parameters, sigma_func,
                                     weak_dirichlet)
from dolfinx_contact.parallel_mesh_ghosting import create_contact_mesh
from dolfinx_contact.unbiased.nitsche_unbiased import nitsche_unbiased
from mpi4py import MPI

if __name__ == "__main__":
    desc = "Nitsche's method for two elastic bodies using custom assemblers"
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--quadrature", default=5, type=int, dest="q_degree",
                        help="Quadrature degree used for contact integrals")

    parser.add_argument("--time_steps", default=1, type=np.int32, dest="time_steps",
                        help="Number of pseudo time steps")
    timer = Timer("~Contact: - all")
    # Parse input arguments or set to defualt values
    args = parser.parse_args()
    mesh_dir = "meshes"
    fname = f"{mesh_dir}/box-key"

    WARNING = log.LogLevel.WARNING
    log.set_log_level(WARNING)

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
    #                Neumann boundary (bottom): 3
    #                Contact surface 1 (tree surface): 6
    #                Contact surface 2 (outer box): 7
    #                If the values of these markers change, the input
    #                to the contact code has to be adjusted

    # When using different tags, change values here
    neumann_bdy = 5
    dirichlet_bdy = 4
    surface_1 = 6
    surface_2 = 7

    with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(ghost_mode=GhostMode.none)
        domain_marker = xdmf.read_meshtags(mesh, "cell_marker")
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facet_marker = xdmf.read_meshtags(mesh, "facet_marker")
    log.log(WARNING, "HELLO")

    # Call function that repartitions mesh for parallel computation
    if mesh.comm.size > 1:
        with Timer("~Contact: Add ghosts"):
            mesh, facet_marker, domain_marker = create_contact_mesh(
                mesh, facet_marker, domain_marker, [6, 7])

    V = _fem.VectorFunctionSpace(mesh, ("Lagrange", 1))

    def _torque(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        for i in range(x.shape[1]):
            values[1, i] = 100 * x[2, i]
            values[2, i] = -100 * x[1, i]
        return values

    # Functions for Dirichlet and Neuman boundaries, body force
    g = _fem.Constant(mesh, default_scalar_type((0, 0, 0)))      # zero dirichlet
    t = _fem.Function(V)
    t.interpolate(_torque)  # traction
    f = _fem.Constant(mesh, default_scalar_type((2.0, 0.0, 0)))  # body force

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
    surfaces = adjacencylist(data, offsets)

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
    inner_cells = domain_marker.find(1)

    def _u_initial(x):
        values = np.zeros((gdim, x.shape[1]))
        values[0, :] = 0.1
        return values

    u.interpolate(_u_initial, inner_cells)

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
    problem_parameters = {"gamma": E * gamma, "theta": theta, "mu": mu, "lambda": lmbda}
    solver_outfile = None

    rhs_fns = [g, t, f]
    size = mesh.comm.size
    outname = f"results/boxkey_{tdim}D_{size}"

    cffi_options = ["-Ofast", "-march=native"]
    jit_options = {"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]}
    u1, num_its, krylov_iterations, solver_time = nitsche_unbiased(args.time_steps, ufl_form=F, u=u,
                                                                   rhs_fns=rhs_fns,
                                                                   markers=[domain_marker, facet_marker],
                                                                   contact_data=(surfaces, contact_pairs),
                                                                   bcs=[], bc_fns=[],
                                                                   problem_parameters=problem_parameters,
                                                                   raytracing=False,
                                                                   newton_options=newton_options,
                                                                   petsc_options=petsc_options,
                                                                   jit_options=jit_options,
                                                                   outfile=solver_outfile,
                                                                   fname=outname,
                                                                   quadrature_degree=args.q_degree,
                                                                   search_radius=np.float64(-1))

    timer.stop()
    # write solution to file
    size = mesh.comm.size
    with XDMFFile(mesh.comm, f"results/box_{size}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        u1.name = f"u_{size}"
        xdmf.write_function(u1)
    with XDMFFile(mesh.comm, f"results/box_partitioning_{size}.xdmf", "w") as xdmf:
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

    list_timings(MPI.COMM_WORLD, [TimingType.wall])
