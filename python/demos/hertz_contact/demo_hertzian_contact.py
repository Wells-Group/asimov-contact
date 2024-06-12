# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

# import argparse

# from mpi4py import MPI
# from petsc4py.PETSc import InsertMode, ScatterMode  # type: ignore

# import numpy as np
# import numpy.typing as npt
# import ufl
# from dolfinx import default_scalar_type
# from dolfinx.common import Timer, timed
# from dolfinx.fem import (
#     Constant,
#     Function,
#     dirichletbc,
#     form,
#     functionspace,
#     locate_dofs_topological,
# )
# from dolfinx.fem.petsc import (
#     apply_lifting,
#     assemble_matrix,
#     assemble_vector,
#     create_vector,
#     set_bc,
# )
# from dolfinx.geometry import bb_tree, compute_closest_entity
# from dolfinx.graph import adjacencylist
# from dolfinx.io import VTXWriter, XDMFFile
# from dolfinx.mesh import Mesh
# from dolfinx_contact.cpp import ContactMode
# from dolfinx_contact.general_contact.contact_problem import ContactProblem, FrictionLaw
# from dolfinx_contact.helpers import epsilon, lame_parameters, rigid_motions_nullspace_subdomains, sigma_func
# from dolfinx_contact.meshing import (
#     convert_mesh,
#     create_circle_plane_mesh,
#     create_halfdisk_plane_mesh,
#     create_halfsphere_box_mesh,
# )
# from dolfinx_contact.newton_solver import NewtonSolver
# from dolfinx_contact.output import ContactWriter


# def closest_node_in_mesh(mesh: Mesh, point: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
#     points = np.reshape(point, (1, 3))
#     bounding_box = bb_tree(mesh, 0)
#     node = compute_closest_entity(bounding_box, bounding_box, mesh, points[0])
#     return node


if __name__ == "__main__":
    print("Demo needs updating. Exiting.")
    exit(0)

    """
    desc = "Example for verifying correctness of code"
    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--quadrature",
        default=5,
        type=int,
        dest="q_degree",
        help="Quadrature degree used for contact integrals",
    )
    parser.add_argument(
        "--order",
        default=1,
        type=int,
        dest="order",
        help="Order of mesh geometry",
        choices=[1, 2],
    )
    _3D = parser.add_mutually_exclusive_group(required=False)
    _3D.add_argument("--3D", dest="threed", action="store_true", help="Use 3D mesh", default=False)
    _simplex = parser.add_mutually_exclusive_group(required=False)
    _simplex.add_argument(
        "--simplex",
        dest="simplex",
        action="store_true",
        help="Use triangle/tet mesh",
        default=False,
    )
    parser.add_argument("--res", default=0.1, type=np.float64, dest="res", help="Mesh resolution")
    parser.add_argument(
        "--problem",
        default=1,
        type=int,
        dest="problem",
        help="Which problem to solve: 1. Volume force, 2. Surface force",
        choices=[1, 2],
    )
    _chouly = parser.add_mutually_exclusive_group(required=False)
    _chouly.add_argument(
        "--chouly",
        dest="chouly",
        action="store_true",
        help="Use parameters from Chouly paper",
        default=False,
    )
    # Parse input arguments or set to defualt values
    args = parser.parse_args()
    threed = args.threed
    simplex = args.simplex
    problem = args.problem
    mesh_dir = "../meshes"
    steps = 4

    # Problem paramters
    if args.chouly:
        R = 0.25
        L = 1.0
        H = 1.0
        if threed:
            load = 4 * 0.25 * np.pi * R**3 / 3.0
        else:
            load = 0.25 * np.pi * R**2
        E1 = 2.5
        E2 = 2.5
        nu1 = 0.25
        nu2 = 0.25
    else:
        R = 8
        L = 20
        H = 20
        load = 2 * R * 0.625
        E1 = 200
        E2 = 200
        nu1 = 0.3
        nu2 = 0.3

    distributed_load = load
    gap = 0.0

    # lame parameters
    mu_func, lambda_func = lame_parameters(True)
    mu1 = mu_func(E1, nu1)
    mu2 = mu_func(E2, nu2)
    lmbda1 = lambda_func(E1, nu1)
    lmbda2 = lambda_func(E2, nu2)
    Estar = E1 * E2 / (E2 * (1 - nu1**2) + E1 * (1 - nu2**2))

    if threed:
        if problem == 1:
            outname = "../results/hertz1_3D_simplex"
            fname = f"{mesh_dir}/hertz1_3D_simplex"
            create_halfsphere_box_mesh(
                filename=f"{fname}.msh",
                res=args.res,
                order=args.order,
                r=R,
                height=H,
                length=L,
                width=L,
                gap=gap,
            )
            neumann_bdy = 2
            contact_bdy_1 = 1
            contact_bdy_2 = 8
            dirichlet_bdy = 7
        else:
            outname = "../results/hertz2_3D_simplex"
            fname = f"{mesh_dir}/hertz2_3D_simplex"
            create_halfsphere_box_mesh(
                filename=f"{fname}.msh",
                res=args.res,
                order=args.order,
                r=R,
                height=H,
                length=L,
                width=L,
                gap=gap,
            )
            neumann_bdy = 2
            contact_bdy_1 = 1
            contact_bdy_2 = 8
            dirichlet_bdy = 7

        convert_mesh(fname, f"{fname}.xdmf", gdim=3)
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh()
            domain_marker = xdmf.read_meshtags(mesh, name="cell_marker")
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, tdim)
            facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")

        V = functionspace(mesh, ("CG", args.order, (mesh.geometry.dim,)))

        node1 = closest_node_in_mesh(mesh, np.array([0.0, 0.0, 0.0], dtype=np.float64))
        node2 = closest_node_in_mesh(mesh, np.array([0.0, 0.0, -R / 2.0], dtype=np.float64))
        node3 = closest_node_in_mesh(mesh, np.array([0.0, R / 2.0, -R / 2.0], dtype=np.float64))
        dirichlet_nodes = np.array([node1, node2, node3], dtype=np.int32)
        dirichlet_dofs1 = locate_dofs_topological(V.sub(0), 0, dirichlet_nodes)
        dirichlet_dofs2 = locate_dofs_topological(V.sub(1), 0, dirichlet_nodes)
        dirichlet_dofs3 = locate_dofs_topological(V, mesh.topology.dim - 1, facet_marker.find(dirichlet_bdy))

        g0 = Constant(mesh, default_scalar_type(0.0))
        g = Constant(mesh, default_scalar_type((0.0, 0.0, 0.0)))
        bcs = [
            dirichletbc(g0, dirichlet_dofs1, V.sub(0)),
            dirichletbc(g0, dirichlet_dofs2, V.sub(1)),
            dirichletbc(g, dirichlet_dofs3, V),
        ]
        bc_fns = [g0, g]

        if problem == 1:
            distributed_load = 3 * load / (4 * np.pi * R**3)
            f = Constant(mesh, default_scalar_type((0.0, 0.0, -distributed_load)))
            t = Constant(mesh, default_scalar_type((0.0, 0.0, 0.0)))
        else:
            distributed_load = load / (np.pi * R**2)
            f = Constant(mesh, default_scalar_type((0.0, 0.0, 0.0)))
            t = Constant(mesh, default_scalar_type((0.0, 0.0, -distributed_load)))

        a = np.cbrt(3.0 * load * R / (4 * Estar))
        force = 4 * a**3 * Estar / (3 * R)
        p0 = 3 * force / (2 * np.pi * a**2)

        def _pressure(x, p0, a):
            vals = np.zeros(x.shape[1])
            for i in range(x.shape[1]):
                rsquared = x[0][i] ** 2 + x[1][i] ** 2
                if rsquared < a**2:
                    vals[i] = p0 * np.sqrt(1 - rsquared / a**2)
            return vals
    else:
        if problem == 1:
            outname = "../../results/hertz1_2D_simplex_RR" if simplex else "../results/hertz1_2D_quads_RR"
            fname = f"{mesh_dir}/hertz1_2D_simplex" if simplex else f"{mesh_dir}/hertz1_2D_quads"
            create_circle_plane_mesh(f"{fname}.msh", not simplex, args.res, args.order, R, H, L, gap)
            contact_bdy_1 = 10
            contact_bdy_2 = 6
            dirichlet_bdy = 4
            neumann_bdy = 8
        else:
            outname = "../results/hertz2_2D_simplex_RR" if simplex else "../results/hertz2_2D_quads_RR"
            fname = f"{mesh_dir}/hertz2_2D_simplex" if simplex else f"{mesh_dir}/hertz2_2D_quads"
            create_halfdisk_plane_mesh(
                filename=f"{fname}.msh",
                res=args.res,
                order=args.order,
                quads=not simplex,
                r=R,
                height=H,
                length=L,
                gap=gap,
            )
            contact_bdy_1 = 7
            contact_bdy_2 = 6
            dirichlet_bdy = 4
            neumann_bdy = 8

        convert_mesh(fname, f"{fname}.xdmf", gdim=2)
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh()
            domain_marker = xdmf.read_meshtags(mesh, name="cell_marker")
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, tdim)
            facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")

        V = functionspace(mesh, ("CG", args.order, (mesh.geometry.dim,)))

        node1 = closest_node_in_mesh(mesh, np.array([0.0, -R / 2.5, 0.0], dtype=np.float64))
        node2 = closest_node_in_mesh(mesh, np.array([0.0, -R / 5.0, 0.0], dtype=np.float64))
        dirichlet_nodes = np.hstack([node1, node2])
        dirichlet_dofs1 = locate_dofs_topological(V.sub(0), 0, dirichlet_nodes)
        dirichlet_dofs2 = locate_dofs_topological(V, mesh.topology.dim - 1, facet_marker.find(dirichlet_bdy))

        g0 = Constant(mesh, default_scalar_type(0.0))
        g = Constant(mesh, default_scalar_type((0, 0)))
        bcs = [
            dirichletbc(g0, dirichlet_dofs1, V.sub(0)),
            dirichletbc(g, dirichlet_dofs2, V),
        ]
        bc_fns = [g0, g]

        if problem == 1:
            distributed_load = load / (np.pi * R**2)
            f = Constant(mesh, default_scalar_type((0.0, -distributed_load)))
            t = Constant(mesh, default_scalar_type((0.0, 0.0)))
        else:
            distributed_load = load / (2 * R)
            f = Constant(mesh, default_scalar_type((0.0, 0.0)))
            t = Constant(mesh, default_scalar_type((0.0, -distributed_load)))

        a = 2 * np.sqrt(R * load / (np.pi * Estar))
        p0 = 2 * load / (np.pi * a)

        def _pressure(x, p0, a):
            vals = np.zeros(x.shape[1])
            for i in range(x.shape[1]):
                if abs(x[0][i]) < a:
                    vals[i] = p0 * np.sqrt(1 - x[0][i] ** 2 / a**2)
            return vals

    # Solver options
    ksp_tol = 1e-12
    newton_tol = 1e-7
    newton_options = {
        "relaxation_parameter": 1.0,
        "atol": newton_tol,
        "rtol": newton_tol,
        "convergence_criterion": "residual",
        "max_it": 200,
        "error_on_nonconvergence": True,
    }

    # for debugging use petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
    petsc_options = {
        "matptap_via": "scalable",
        "ksp_type": "gmres",
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
        "ksp_initial_guess_nonzero": False,
        "ksp_norm_type": "unpreconditioned",
    }

    # DG-0 funciton for material
    V0 = functionspace(mesh, ("DG", 0))
    mu = Function(V0)
    lmbda = Function(V0)
    disk_cells = domain_marker.find(1)
    block_cells = domain_marker.find(2)
    mu.interpolate(lambda x: np.full((1, x.shape[1]), mu2), disk_cells)
    mu.interpolate(lambda x: np.full((1, x.shape[1]), mu1), block_cells)
    lmbda.interpolate(lambda x: np.full((1, x.shape[1]), lmbda2), disk_cells)
    lmbda.interpolate(lambda x: np.full((1, x.shape[1]), lmbda1), block_cells)

    sigma = sigma_func(mu, lmbda)
    # Set initial condition

    # Pack mesh data for Nitsche solver
    contact = [(0, 1), (1, 0)]
    data = np.array([contact_bdy_1, contact_bdy_2], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    surfaces = adjacencylist(data, offsets)

    # Function, TestFunction, TrialFunction and measures
    u = Function(V)
    du = Function(V)
    v = ufl.TestFunction(V)
    w = ufl.TrialFunction(V)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)

    # Create variational form without contact contributions
    F = ufl.inner(sigma(u), epsilon(v)) * dx

    # body forces
    F -= ufl.inner(f, v) * dx(1) + ufl.inner(t, v) * ds(neumann_bdy)

    F = ufl.replace(F, {u: u + du})
    J = ufl.derivative(F, du, w)

    # compiler options to improve performance
    cffi_options = ["-Ofast", "-march=native"]
    jit_options = {"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]}
    # compiled forms for rhs and tangen system
    F_compiled = form(F, jit_options=jit_options)
    J_compiled = form(J, jit_options=jit_options)

    # Nitsche parameters
    gamma = E2 * 100 * args.order**2
    theta = 1

    # create initial guess
    def _u_initial(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[-1] = -H / 100 - gap
        return values

    search_mode = [ContactMode.ClosestPoint, ContactMode.ClosestPoint]

    # create contact solver
    contact_problem = ContactProblem([facet_marker], surfaces, contact, mesh, args.q_degree, search_mode)
    contact_problem.generate_contact_data(
        FrictionLaw.Frictionless,
        V,
        {"u": u, "du": du, "mu": mu, "lambda": lmbda},
        gamma,
        theta,
    )
    solver_outfile = None
    du.interpolate(_u_initial, disk_cells)

    # create vector and matrix
    A = contact_problem.create_matrix(J_compiled)
    b = create_vector(F_compiled)

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

    # Set up snes solver for nonlinear solver
    newton_solver = NewtonSolver(mesh.comm, A, b, contact_problem.coeffs)
    # Set matrix-vector computations
    newton_solver.set_residual(compute_residual)
    newton_solver.set_jacobian(compute_jacobian_matrix)
    newton_solver.set_coefficients(compute_coefficients)

    # Set rigid motion nullspace
    null_space = rigid_motions_nullspace_subdomains(
        V,
        domain_marker,
        np.unique(domain_marker.values),
        num_domains=len(np.unique(domain_marker.values)),
    )
    newton_solver.A.setNearNullSpace(null_space)

    # Set Newton solver options
    newton_solver.set_newton_options(newton_options)

    # Set Krylov solver options
    newton_solver.set_krylov_options(petsc_options)

    if not threed:
        writer = ContactWriter(
            mesh,
            contact_problem,
            u,
            contact,
            contact_problem.coeffs,
            args.order,
            simplex,
            [(tdim - 1, 0), (tdim - 1, -R)],
            f"{outname}_{steps}_step",
        )
    # initialise vtx writer
    vtx = VTXWriter(mesh.comm, f"{outname}_{steps}_step.bp", [u], "bp4")
    vtx.write(0)
    for i in range(steps):
        if problem == 1:
            f.value[-1] = -distributed_load * (i + 1) / steps
        else:
            t.value[-1] = -distributed_load * (i + 1) / steps
        n, converged = newton_solver.solve(du, write_solution=True)
        du.x.scatter_forward()
        u.x.array[:] += du.x.array[:]
        a = 2 * np.sqrt(R * load * (i + 1) / (steps * np.pi * Estar))
        p0 = 2 * load * (i + 1) / (steps * np.pi * a)
        if not threed:
            writer.write(
                i + 1,
                lambda x, pi=p0, ai=a: _pressure(x, pi, ai),
                lambda x: np.zeros(x.shape[1]),
            )
        contact_problem.update_contact_detection(u)
        A = contact_problem.create_matrix(J_compiled)
        A.setNearNullSpace(null_space)
        newton_solver.set_petsc_matrix(A)
        du.x.array[:] = 0.1 * du.x.array[:]
        contact_problem.update_contact_data(du)
        vtx.write(i + 1)

    vtx.close()
    """
