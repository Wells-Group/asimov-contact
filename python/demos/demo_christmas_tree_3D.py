# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse
import sys

from mpi4py import MPI
from petsc4py.PETSc import InsertMode, ScatterMode  # type: ignore

import dolfinx.fem as _fem
import numpy as np
import ufl
from dolfinx import default_scalar_type, log
from dolfinx.common import Timer, timed, timing
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_vector,
    set_bc,
)
from dolfinx.graph import adjacencylist
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.mesh import GhostMode, locate_entities_boundary, meshtags
from dolfinx_contact.cpp import ContactMode
from dolfinx_contact.general_contact.contact_problem import ContactProblem, FrictionLaw
from dolfinx_contact.helpers import (
    epsilon,
    lame_parameters,
    rigid_motions_nullspace_subdomains,
    sigma_func,
    weak_dirichlet,
)
from dolfinx_contact.meshing import convert_mesh, create_christmas_tree_mesh_3D
from dolfinx_contact.newton_solver import NewtonSolver
from dolfinx_contact.parallel_mesh_ghosting import create_contact_mesh

if __name__ == "__main__":
    desc = "Nitsche's method for two elastic bodies using custom assemblers"
    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--quadrature",
        default=5,
        type=int,
        dest="q_degree",
        help="Quadrature degree used for contact integrals",
    )
    parser.add_argument("--res", default=0.2, type=np.float64, dest="res", help="Mesh resolution")
    parser.add_argument(
        "--load_steps",
        default=1,
        type=np.int32,
        dest="nload_steps",
        help="Number of steps for gradual loading",
    )
    # Parse input arguments or set to default values
    args = parser.parse_args()
    nload_steps = args.nload_steps
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
            mesh, facet_marker, domain_marker, [surface_1, surface_2]
        )

    V = _fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))

    # Apply zero Dirichlet boundary conditions in z-direction on part of the xmas-tree
    # Find facets for z-Dirichlet bc
    def identifier(x, z):
        return np.logical_and(np.logical_and(np.isclose(x[2], z), abs(x[1]) < 0.1), abs(x[0] - 2) < 0.1)

    dirichlet_facets1 = locate_entities_boundary(mesh, tdim - 1, lambda x: identifier(x, 0.0))
    dirichlet_facets2 = locate_entities_boundary(mesh, tdim - 1, lambda x: identifier(x, 1.0))

    # create new facet_marker including z Dirichlet facets
    indices = np.hstack([facet_marker.indices, dirichlet_facets1, dirichlet_facets2])
    values = np.hstack(
        [
            facet_marker.values,
            z_Dirichlet * np.ones(len(dirichlet_facets1) + len(dirichlet_facets2), dtype=np.int32),
        ]
    )
    sorted_facets = np.argsort(indices)
    facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])

    dirichlet_dofs = _fem.locate_dofs_topological(V.sub(2), tdim - 1, indices[sorted_facets])
    # Create Dirichlet bdy conditions for preventing rigid body motion in z-direction
    dofs = _fem.locate_dofs_topological(V.sub(2), mesh.topology.dim - 1, facet_marker.find(z_Dirichlet))
    g0 = _fem.Constant(mesh, default_scalar_type(0))
    bcs = [_fem.dirichletbc(g0, dofs, V.sub(2))]
    bc_fns = [g0]
    # Functions for Dirichlet and Neuman boundaries, body force
    t_val = (0.2, 0.5, 0)
    f_val = (1.0, 0.5, 0)
    g = _fem.Constant(mesh, default_scalar_type((0, 0, 0)))  # zero dirichlet
    t = _fem.Constant(mesh, default_scalar_type(t_val))  # traction
    f = _fem.Constant(mesh, default_scalar_type(f_val))  # body force

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
    print(surface_1, surface_2)

    # Function, TestFunction, TrialFunction and measures
    u = _fem.Function(V)
    du = _fem.Function(V)
    w = ufl.TrialFunction(V)
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
    gamma = 20
    theta = 1
    F = weak_dirichlet(F, u, g, sigma, E * gamma, theta, ds(4))

    # traction (neumann) boundary condition on mesh boundary with tag 3
    F -= ufl.inner(t, v) * ds(3)

    # body forces
    F -= ufl.inner(f, v) * dx(1)

    F = ufl.replace(F, {u: u + du})

    J = ufl.derivative(F, du, w)

    # compiler options to improve performance
    cffi_options = ["-Ofast", "-march=native"]
    jit_options = {"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]}
    # compiled forms for rhs and tangen system
    F_compiled = _fem.form(F, jit_options=jit_options)
    J_compiled = _fem.form(J, jit_options=jit_options)
    # create initial guess
    tree_cells = domain_marker.find(1)

    def _u_initial(x):
        values = np.zeros((gdim, x.shape[1]))
        for i in range(x.shape[1]):
            values[0, i] = 0.05
        return values

    du.interpolate(_u_initial, tree_cells)

    # Solver options
    ksp_tol = 1e-10
    newton_tol = 1e-7
    newton_options = {
        "relaxation_parameter": 1.0,
        "atol": newton_tol,
        "rtol": newton_tol,
        "convergence_criterion": "residual",
        "max_it": 50,
        "error_on_nonconvergence": False,
    }

    # In order to use an LU solver for debugging purposes on small scale problems
    # use the following PETSc options: {"ksp_type": "preonly", "pc_type": "lu"}
    petsc_options = {
        "matptap_via": "scalable",
        "ksp_type": "cg",
        "ksp_rtol": ksp_tol,
        "ksp_atol": ksp_tol,
        "pc_type": "gamg",
        "pc_mg_levels": 3,
        "pc_mg_cycles": 1,  # 1 is v, 2 is w
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "pc_gamg_type": "agg",
        "pc_gamg_coarse_eq_limit": 100,
        "pc_gamg_agg_nsmooths": 1,
        "pc_gamg_threshold": 1e-3,
        "pc_gamg_square_graph": 2,
        "pc_gamg_reuse_interpolation": False,
        "ksp_norm_type": "unpreconditioned",
    }

    # Solve contact problem using Nitsche's method
    V0 = _fem.functionspace(mesh, ("DG", 0))
    mu0 = _fem.Function(V0)
    lmbda0 = _fem.Function(V0)
    mu0.interpolate(lambda x: np.full((1, x.shape[1]), mu))
    lmbda0.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))

    # create contact solver
    search_mode = [ContactMode.ClosestPoint for _ in range(len(contact_pairs))]
    contact_problem = ContactProblem([facet_marker], surfaces, contact_pairs, mesh, args.q_degree, search_mode)
    contact_problem.generate_contact_data(
        FrictionLaw.Frictionless,
        V,
        {"u": u, "du": du, "mu": mu0, "lambda": lmbda0},
        E * gamma,
        theta,
    )
    solver_outfile = None
    log.set_log_level(log.LogLevel.WARNING)
    rhs_fns = [g, t, f]
    size = mesh.comm.size
    outname = f"results/xmas_{tdim}D_{size}"

    # define functions for newton solver
    def compute_coefficients(x, coeffs):
        du.x.scatter_forward()
        contact_problem.update_contact_data(du)

    @timed("~Contact: Assemble residual")
    def compute_residual(x, b, coeffs):
        b.zeroEntries()
        b.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        with Timer("~~Contact: Contact contributions (in assemble vector)"):
            contact_problem.assemble_vector(b, V)
        with Timer("~~Contact: Standard contributions (in assemble vector)"):
            assemble_vector(b, F_compiled)

        # Apply boundary condition
        if len(bcs) > 0:
            apply_lifting(b, [J_compiled], bcs=[bcs], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=InsertMode.ADD, mode=ScatterMode.REVERSE)
        if len(bcs) > 0:
            set_bc(b, bcs, x, -1.0)

    @timed("~Contact: Assemble matrix")
    def compute_jacobian_matrix(x, a_mat, coeffs):
        a_mat.zeroEntries()
        with Timer("~~Contact: Contact contributions (in assemble matrix)"):
            contact_problem.assemble_matrix(a_mat, V)
        with Timer("~~Contact: Standard contributions (in assemble matrix)"):
            assemble_matrix(a_mat, J_compiled, bcs=bcs)
        a_mat.assemble()

    # create vector and matrix
    A = contact_problem.create_matrix(J_compiled)
    b = create_vector(F_compiled)

    # Set up snes solver for nonlinear solver
    newton_solver = NewtonSolver(mesh.comm, A, b, contact_problem.coeffs)
    # Set matrix-vector computations
    newton_solver.set_residual(compute_residual)
    newton_solver.set_jacobian(compute_jacobian_matrix)
    newton_solver.set_coefficients(compute_coefficients)

    # Set rigid motion nullspace
    null_space = rigid_motions_nullspace_subdomains(V, domain_marker, np.unique(domain_marker.values), num_domains=2)
    newton_solver.A.setNearNullSpace(null_space)

    # Set Newton solver options
    newton_solver.set_newton_options(newton_options)

    # Set Krylov solver options
    newton_solver.set_krylov_options(petsc_options)
    # initialise vtx writer
    u.name = "u"
    vtx = VTXWriter(mesh.comm, f"{outname}_{nload_steps}_step.bp", [u], "bp4")
    vtx.write(0)
    num_newton_its = np.zeros(nload_steps, dtype=int)
    num_krylov_its = np.zeros(nload_steps, dtype=int)
    newton_time = np.zeros(nload_steps, dtype=np.float64)
    for i in range(nload_steps):
        t.value[:] = (i + 1) * np.array(t_val) / nload_steps
        f.value[:] = (i + 1) * np.array(f_val) / nload_steps
        timing_str = f"~Contact: {i+1} Newton Solver"
        with Timer(timing_str):
            n, converged = newton_solver.solve(du, write_solution=True)
        num_newton_its[i] = n
        newton_time[i] = timing(timing_str)[1]
        num_krylov_its[i] = newton_solver.krylov_iterations
        du.x.scatter_forward()
        u.x.array[:] += du.x.array[:]
        contact_problem.update_contact_detection(u)
        A = contact_problem.create_matrix(J_compiled)
        A.setNearNullSpace(null_space)
        newton_solver.set_petsc_matrix(A)
        du.x.array[:] = 0.05 * du.x.array[:]
        contact_problem.update_contact_data(du)
        vtx.write(i + 1)
    vtx.close()

    VDG = _fem.functionspace(mesh, ("DG", 1, (gdim,)))
    u1 = _fem.Function(VDG)
    u1.interpolate(u)
    # write solution to file
    size = mesh.comm.size
    with XDMFFile(mesh.comm, f"results/xmas_partitioning_{size}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(process_marker, mesh.geometry)

    W = _fem.functionspace(mesh, ("DG", 1))
    sigma_vm_h = _fem.Function(W)
    sigma_dev = sigma(u1) - (1 / 3) * ufl.tr(sigma(u1)) * ufl.Identity(len(u1))
    sigma_vm = ufl.sqrt((3 / 2) * ufl.inner(sigma_dev, sigma_dev))
    sigma_vm_h.name = "vonMises"
    sigma_vm_expr = _fem.Expression(sigma_vm, W.element.interpolation_points())
    sigma_vm_h.interpolate(sigma_vm_expr)
    vtx = VTXWriter(mesh.comm, f"results/xmas_{size}.bp", [u1, sigma_vm_h], "bp4")
    vtx.write(0)
    vtx.close()

    outfile = sys.stdout

    if mesh.comm.rank == 0:
        print("-" * 25, file=outfile)
        print(f"Newton options {newton_options}", file=outfile)
        print(
            f"num_dofs: {u1.function_space.dofmap.index_map_bs*u1.function_space.dofmap.index_map.size_global}"
            + f", {mesh.topology.cell_type}",
            file=outfile,
        )
        print(
            f"Newton solver {timing('~Contact: Newton (Newton solver)')[1]}",
            file=outfile,
        )
        print(
            f"Krylov solver {timing('~Contact: Newton (Krylov solver)')[1]}",
            file=outfile,
        )
        print(f"Newton time: {newton_time}", file=outfile)
        print(f"Newton iterations {num_newton_its}, ", file=outfile)
        print(f"Krylov iterations {num_krylov_its},", file=outfile)
        print("-" * 25, file=outfile)
