# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse
import sys

from mpi4py import MPI
from petsc4py.PETSc import InsertMode, ScatterMode  # type: ignore

import dolfinx.fem as _fem
import dolfinx.io.gmshio
import gmsh
import numpy as np
import ufl
from dolfinx import default_scalar_type, log
from dolfinx.common import Timer, TimingType, list_timings, timed, timing
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_vector,
    set_bc,
)
from dolfinx.graph import adjacencylist
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx_contact.cpp import ContactMode, find_candidate_surface_segment
from dolfinx_contact.general_contact.contact_problem import ContactProblem, FrictionLaw
from dolfinx_contact.helpers import (
    epsilon,
    lame_parameters,
    rigid_motions_nullspace_subdomains,
    sigma_func,
    weak_dirichlet,
)
from dolfinx_contact.meshing import (
    create_christmas_tree_mesh,
    create_christmas_tree_mesh_3D,
)
from dolfinx_contact.newton_solver import NewtonSolver
from dolfinx_contact.parallel_mesh_ghosting import create_contact_mesh


# if __name__ == "__main__":
def run_solver(
    theta=1.0,
    gamma=10.0,
    q_degree=5,
    set_timing=False,
    ksp_view=False,
    E=1e3,
    nu=0.1,
    res=0.2,
    radius=0.5,
    outfile=None,
    split=1,
    nload_steps=1,
    raytracing=False,
    threed=False,
):
    gmsh.initialize()

    if threed:
        name = "xmas_3D"
        model = gmsh.model()
        model.add(name)
        model.setCurrent(name)
        model = create_christmas_tree_mesh_3D(model, res=res, split=split, n1=81, n2=41)
        mesh, domain_marker, facet_marker = dolfinx.io.gmshio.model_to_mesh(
            model, MPI.COMM_WORLD, 0, gdim=3
        )

        tdim = mesh.topology.dim

        marker_offset = 6
        if mesh.comm.size > 1:
            mesh, facet_marker, domain_marker = create_contact_mesh(
                mesh,
                facet_marker,
                domain_marker,
                [marker_offset + i for i in range(2 * split)],
            )

        V = _fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
        # Apply zero Dirichlet boundary conditions in z-direction on part of the xmas-tree

        # Find facets for z-Dirichlet bc
        def identifier(x, z):
            return np.logical_and(
                np.logical_and(np.isclose(x[2], z), abs(x[1]) < 0.1),
                abs(x[0] - 2) < 0.1,
            )

        dirichlet_facets1 = locate_entities_boundary(mesh, tdim - 1, lambda x: identifier(x, 0.0))
        dirichlet_facets2 = locate_entities_boundary(mesh, tdim - 1, lambda x: identifier(x, 1.0))

        # create new facet_marker including z Dirichlet facets
        z_Dirichlet = marker_offset + 4 * split + 1
        indices = np.hstack([facet_marker.indices, dirichlet_facets1, dirichlet_facets2])
        values = np.hstack(
            [
                facet_marker.values,
                z_Dirichlet
                * np.ones(len(dirichlet_facets1) + len(dirichlet_facets2), dtype=np.int32),
            ]
        )
        sorted_facets = np.argsort(indices)
        facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])
        # Create Dirichlet bdy conditions
        dofs = _fem.locate_dofs_topological(
            V.sub(2), mesh.topology.dim - 1, facet_marker.find(z_Dirichlet)
        )
        gz = _fem.Constant(mesh, default_scalar_type(0))
        bcs = [_fem.dirichletbc(gz, dofs, V.sub(2))]
        # bc_fns: list[typing.Union[_fem.Constant, _fem.Function]] = [gz]
        g = _fem.Constant(mesh, default_scalar_type((0, 0, 0)))  # zero dirichlet
        t_val = [0.2, 0.5, 0.0]
        f_val = [1.0, 0.5, 0.0]
        t = _fem.Constant(mesh, default_scalar_type(t_val))  # traction
        f = _fem.Constant(mesh, default_scalar_type(f_val))  # body force

    else:
        name = "xmas_2D"
        model = gmsh.model()
        model.add(name)
        model.setCurrent(name)
        model = create_christmas_tree_mesh(model, res=res, split=split)
        mesh, domain_marker, facet_marker = dolfinx.io.gmshio.model_to_mesh(
            model, MPI.COMM_WORLD, 0, gdim=2
        )

        tdim = mesh.topology.dim

        marker_offset = 5
        if mesh.comm.size > 1:
            mesh, facet_marker, domain_marker = create_contact_mesh(
                mesh,
                facet_marker,
                domain_marker,
                [marker_offset + i for i in range(2 * split)],
            )

        V = _fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
        bcs = []
        # bc_fns = []
        g = _fem.Constant(mesh, default_scalar_type((0, 0)))  # zero Dirichlet
        t_val = [0.2, 0.5]
        f_val = [1.0, 0.5]
        t = _fem.Constant(mesh, default_scalar_type(t_val))  # traction
        f = _fem.Constant(mesh, default_scalar_type(f_val))  # body force

    gmsh.finalize()

    ncells = mesh.topology.index_map(tdim).size_local
    indices = np.array(range(ncells), dtype=np.int32)
    values = mesh.comm.rank * np.ones(ncells, dtype=np.int32)
    process_marker = meshtags(mesh, tdim, indices, values)
    process_marker.name = "process_marker"
    gdim = mesh.geometry.dim
    # create meshtags for candidate segments
    mts = [domain_marker, facet_marker]
    cand_facets_0 = np.sort(np.hstack([facet_marker.find(marker_offset + i) for i in range(split)]))
    cand_facets_1 = np.sort(
        np.hstack([facet_marker.find(marker_offset + split + i) for i in range(split)])
    )

    for i in range(split):
        fcts = np.array(
            find_candidate_surface_segment(
                mesh._cpp_object,
                facet_marker.find(marker_offset + split + i),
                cand_facets_0,
                0.8,
            ),
            dtype=np.int32,
        )
        vls = np.full(len(fcts), marker_offset + 2 * split + i, dtype=np.int32)
        mts.append(meshtags(mesh, tdim - 1, fcts, vls))

    for i in range(split):
        fcts = np.array(
            find_candidate_surface_segment(
                mesh._cpp_object,
                facet_marker.find(marker_offset + i),
                cand_facets_1,
                0.8,
            ),
            dtype=np.int32,
        )
        vls = np.full(len(fcts), marker_offset + 3 * split + i, dtype=np.int32)
        mts.append(meshtags(mesh, tdim - 1, fcts, vls))

    # contact surfaces with tags from marker_offset to marker_offset + 4 * split (split = #segments)
    data = np.arange(marker_offset, marker_offset + 4 * split, dtype=np.int32)
    offsets = np.concatenate(
        [
            np.array([0, 2 * split], dtype=np.int32),
            np.arange(2 * split + 1, 4 * split + 1, dtype=np.int32),
        ]
    )
    surfaces = adjacencylist(data, offsets)
    # zero dirichlet boundary condition on mesh boundary with tag 5

    # Function, TestFunction, TrialFunction and measures
    u = _fem.Function(V)
    du = _fem.Function(V)
    v = ufl.TestFunction(V)
    w = ufl.TrialFunction(V)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)

    # Compute lame parameters
    mu_func, lambda_func = lame_parameters(False)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    # Create variational form without contact contributions
    F = ufl.inner(sigma(u), epsilon(v)) * dx

    # Apply weak Dirichlet boundary conditions using Nitsche's method
    F = weak_dirichlet(F, u, g, sigma, E * gamma, theta, ds(4))

    # traction (neumann) boundary condition on mesh boundary with tag 3
    F -= ufl.inner(t, v) * ds(3)

    # body forces
    F -= ufl.inner(f, v) * dx(1)

    F = ufl.replace(F, {u: u + du})
    J = ufl.derivative(F, du, w)

    # compiler options to improve performance
    jit_options = {"cffi_extra_compile_args": [], "cffi_libraries": ["m"]}
    # compiled forms for rhs and tangen system
    F_compiled = _fem.form(F, jit_options=jit_options)
    J_compiled = _fem.form(J, jit_options=jit_options)

    contact_pairs = []
    for i in range(split):
        contact_pairs.append((i, 3 * split + i))
        contact_pairs.append((split + i, 2 * split + i))

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

    # In order to use an LU solver for debugging purposes on small scale
    # problems use the following PETSc options: {"ksp_type": "preonly",
    # "pc_type": "lu"}
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
    # solver_outfile = outfile if ksp_view else None
    log.set_log_level(log.LogLevel.OFF)
    # rhs_fns = [g, t, f]
    size = mesh.comm.size
    outname = f"results/xmas_{tdim}D_{size}"

    # create contact solver
    if raytracing:
        search_mode = [ContactMode.Raytracing for _ in range(len(contact_pairs))]
    else:
        search_mode = [ContactMode.ClosestPoint for _ in range(len(contact_pairs))]

    contact_problem = ContactProblem(
        mts[1:], surfaces, contact_pairs, mesh, q_degree, search_mode, radius
    )
    contact_problem.generate_contact_data(
        FrictionLaw.Frictionless,
        V,
        {"u": u, "du": du, "mu": mu0, "lambda": lmbda0},
        E * gamma,
        theta,
    )
    # solver_outfile = None
    log.set_log_level(log.LogLevel.WARNING)
    # rhs_fns = [g, t, f]
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
            apply_lifting(b, [J_compiled], bcs=[bcs], x0=[x], alpha=-1.0)
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
    null_space = rigid_motions_nullspace_subdomains(
        V, domain_marker, np.unique(domain_marker.values), 2
    )
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
        timing_str = f"~Contact: {i + 1} Newton Solver"
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
    with XDMFFile(mesh.comm, f"results/xmas_partitioning_{size}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(process_marker, mesh.geometry)

    if set_timing:
        list_timings(mesh.comm, [TimingType.wall])

    if outfile is None:
        outfile = sys.stdout
    else:
        outfile = open(outfile, "a")

    if mesh.comm.rank == 0:
        print("-" * 25, file=outfile)
        print(f"Newton options {newton_options}", file=outfile)
        print(
            f"num_dofs: {
                u1.function_space.dofmap.index_map_bs
                * u1.function_space.dofmap.index_map.size_global
            }"
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


if __name__ == "__main__":
    desc = "Nitsche's method for two elastic bodies using custom assemblers"
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--theta",
        default=1.0,
        type=float,
        dest="theta",
        help="Theta parameter for Nitsche, 1 symmetric, -1 skew symmetric, 0 Penalty-like",
        choices=[1.0, -1.0, 0.0],
    )
    parser.add_argument(
        "--gamma",
        default=10,
        type=np.float64,
        dest="gamma",
        help="Coercivity/Stabilization parameter for Nitsche condition",
    )
    parser.add_argument(
        "--quadrature",
        default=5,
        type=int,
        dest="q_degree",
        help="Quadrature degree used for contact integrals",
    )
    _3D = parser.add_mutually_exclusive_group(required=False)
    _3D.add_argument("--3D", dest="threed", action="store_true", help="Use 3D mesh", default=False)
    _timing = parser.add_mutually_exclusive_group(required=False)
    _timing.add_argument(
        "--timing",
        dest="timing",
        action="store_true",
        help="List timings",
        default=False,
    )
    _ksp = parser.add_mutually_exclusive_group(required=False)
    _ksp.add_argument(
        "--ksp-view",
        dest="ksp",
        action="store_true",
        help="List ksp options",
        default=False,
    )
    parser.add_argument(
        "--E", default=1e3, type=np.float64, dest="E", help="Youngs modulus of material"
    )
    parser.add_argument("--nu", default=0.1, type=np.float64, dest="nu", help="Poisson's ratio")
    parser.add_argument("--res", default=0.2, type=np.float64, dest="res", help="Mesh resolution")
    parser.add_argument(
        "--radius",
        default=0.5,
        type=np.float64,
        dest="radius",
        help="Search radius for ray-tracing",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        required=False,
        help="File for appending results",
        dest="outfile",
    )
    parser.add_argument(
        "--split",
        type=np.int32,
        default=1,
        required=False,
        help="number of surface segments",
        dest="split",
    )
    parser.add_argument(
        "--load_steps",
        default=1,
        type=np.int32,
        dest="nload_steps",
        help="Number of loading steps",
    )
    _raytracing = parser.add_mutually_exclusive_group(required=False)
    _raytracing.add_argument(
        "--raytracing",
        dest="raytracing",
        action="store_true",
        help="Use raytracing for contact search.",
        default=False,
    )
    # Parse input arguments or set to defualt values
    args = parser.parse_args()

    threed = args.threed
    split = args.split

    run_solver(
        args.theta,
        args.gamma,
        args.q_degree,
        args.timing,
        args.ksp,
        args.E,
        args.nu,
        args.res,
        args.radius,
        args.outfile,
        args.split,
        args.nload_steps,
        args.raytracing,
        args.threed,
    )


def test_christmas_tree():
    run_solver()
