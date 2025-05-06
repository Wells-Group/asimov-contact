# Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse
import sys
from pathlib import Path

from mpi4py import MPI
from petsc4py.PETSc import InsertMode, ScatterMode  # type: ignore

import dolfinx.io.gmshio
import gmsh
import numpy as np
import ufl
from dolfinx import default_scalar_type, log
from dolfinx.common import Timer, TimingType, list_timings, timed, timing
from dolfinx.fem import (
    Constant,
    Expression,
    Function,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
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
from dolfinx_contact.cpp import ContactMode
from dolfinx_contact.general_contact.contact_problem import ContactProblem, FrictionLaw
from dolfinx_contact.helpers import (
    epsilon,
    lame_parameters,
    rigid_motions_nullspace_subdomains,
    sigma_func,
    weak_dirichlet,
)
from dolfinx_contact.meshing import (
    create_box_mesh_3D,
    create_circle_circle_mesh,
    create_circle_plane_mesh,
    create_cylinder_cylinder_mesh,
    create_gmsh_box_mesh_2D,
    create_sphere_plane_mesh,
)
from dolfinx_contact.newton_solver import NewtonSolver
from dolfinx_contact.parallel_mesh_ghosting import create_contact_mesh


def run_soler(args):
    # Current formulation uses bilateral contact
    threed = args.threed
    problem = args.problem
    nload_steps = args.nload_steps
    simplex = args.simplex
    timing_disp = args.timing
    order = args.order
    disp = args.disp
    res = args.res
    E = args.E
    nu = args.nu
    gamma = args.gamma
    theta = args.theta
    plane_strain = args.plane_strain
    q_degree = args.q_degree
    coulomb = args.coulomb
    raytracing = args.raytracing
    outfile = args.outfile
    lifting = args.lifting
    # ksp_v = args.ksp
    fric_val = args.fric

    mesh_dir = "meshes"
    # triangle_ext = {1: "", 2: "6", 3: "10"}
    # tetra_ext = {1: "", 2: "10", 3: "20"}
    # hex_ext = {1: "", 2: "27"}
    # quad_ext = {1: "", 2: "9", 3: "16"}
    # line_ext = {1: "", 2: "3", 3: "4"}
    if order > 2:
        raise NotImplementedError("More work in DOLFINx (SubMesh) required for this to work.")
    # Load mesh and create identifier functions for the top (Displacement condition)
    # and the bottom (contact condition)

    gmsh.initialize()

    fname = Path("nitsche_unbiased/mesh.msh")
    fname.parent.mkdir(exist_ok=True)
    if threed:
        displacement = np.array([[0, 0, -disp], [0, 0, 0]])
        if problem == 1:
            outname = "results/problem1_3D_simplex" if simplex else "results/problem1_3D_hex"

            name = "box_3D"
            model = gmsh.model()
            model.add(name)
            model.setCurrent(name)
            model = create_box_mesh_3D(model, simplex, order=order)
            mesh, domain_marker, _ = dolfinx.io.gmshio.model_to_mesh(
                model, MPI.COMM_WORLD, 0, gdim=3
            )
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, 0)
            mesh.topology.create_connectivity(tdim - 1, tdim)

            dirichlet_bdy_1 = 1
            contact_bdy_1 = 2
            contact_bdy_2 = 3
            dirichlet_bdy_2 = 4
            # Create meshtag for top and bottom markers
            top_facets1 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[2], 0.5))
            bottom_facets1 = locate_entities_boundary(
                mesh, tdim - 1, lambda x: np.isclose(x[2], 0.0)
            )
            top_facets2 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[2], -0.1))
            bottom_facets2 = locate_entities_boundary(
                mesh, tdim - 1, lambda x: np.isclose(x[2], -0.6)
            )
            top_values = np.full(len(top_facets1), dirichlet_bdy_1, dtype=np.int32)
            bottom_values = np.full(len(bottom_facets1), contact_bdy_1, dtype=np.int32)

            surface_values = np.full(len(top_facets2), contact_bdy_2, dtype=np.int32)
            sbottom_values = np.full(len(bottom_facets2), dirichlet_bdy_2, dtype=np.int32)
            indices = np.concatenate([top_facets1, bottom_facets1, top_facets2, bottom_facets2])
            values = np.hstack([top_values, bottom_values, surface_values, sbottom_values])
            sorted_facets = np.argsort(indices)
            facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])

        elif problem == 2:
            outname = "results/problem2_3D_simplex" if simplex else "results/problem2_3D_hex"
            name = "problem2_3D"
            model = gmsh.model()
            model.add(name)
            model.setCurrent(name)
            model = create_sphere_plane_mesh(model, order=order, res=res)
            mesh, domain_marker, facet_marker = dolfinx.io.gmshio.model_to_mesh(
                model, MPI.COMM_WORLD, 0, gdim=3
            )
            dirichlet_bdy_1 = 2
            contact_bdy_1 = 1
            contact_bdy_2 = 8
            dirichlet_bdy_2 = 7

        elif problem == 3:
            outname = "results/problem3_3D_simplex" if simplex else "results/problem3_3D_hex"
            displacement = np.array([[-1, 0, 0], [0, 0, 0]])

            name = "Cylinder-cylinder-mesh"
            model = gmsh.model()
            model.add(name)
            model.setCurrent(name)
            model = create_cylinder_cylinder_mesh(model, res=res, simplex=simplex)
            mesh, domain_marker, _ = dolfinx.io.gmshio.model_to_mesh(
                model, MPI.COMM_WORLD, 0, gdim=3
            )
            mesh.name = "cylinder_cylinder"
            domain_marker.name = "domain_marker"
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, tdim)

            def right(x):
                return x[0] > 2.2

            def right_contact(x):
                return np.logical_and(x[0] < 2, x[0] > 1.45)

            def left_contact(x):
                return np.logical_and(x[0] > 0.25, x[0] < 1.1)

            def left(x):
                return x[0] < -0.5

            dirichlet_bdy_1 = 1
            contact_bdy_1 = 2
            contact_bdy_2 = 3
            dirichlet_bdy_2 = 4
            # Create meshtag for top and bottom markers
            dirichlet_facets_1 = locate_entities_boundary(mesh, tdim - 1, right)
            contact_facets_1 = locate_entities_boundary(mesh, tdim - 1, right_contact)
            contact_facets_2 = locate_entities_boundary(mesh, tdim - 1, left_contact)
            dirchlet_facets_2 = locate_entities_boundary(mesh, tdim - 1, left)

            val0 = np.full(len(dirichlet_facets_1), dirichlet_bdy_1, dtype=np.int32)
            val1 = np.full(len(contact_facets_1), contact_bdy_1, dtype=np.int32)
            val2 = np.full(len(contact_facets_2), contact_bdy_2, dtype=np.int32)
            val3 = np.full(len(dirchlet_facets_2), dirichlet_bdy_2, dtype=np.int32)
            indices = np.concatenate(
                [
                    dirichlet_facets_1,
                    contact_facets_1,
                    contact_facets_2,
                    dirchlet_facets_2,
                ]
            )
            values = np.hstack([val0, val1, val2, val3])
            sorted_facets = np.argsort(indices)
            facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])

    else:
        displacement = np.array([[0, -disp], [0, 0]])
        if problem == 1:
            outname = "results/problem1_2D_simplex" if simplex else "results/problem1_2D_quads"
            name = "box_mesh_2D"
            model = gmsh.model()
            model.add(name)
            model.setCurrent(name)
            model = create_gmsh_box_mesh_2D(model, quads=not simplex, res=res, order=order)
            mesh, domain_marker, facet_marker = dolfinx.io.gmshio.model_to_mesh(
                model, MPI.COMM_WORLD, 0, gdim=2
            )

            dirichlet_bdy_1 = 5
            contact_bdy_1 = 3
            contact_bdy_2 = 9
            dirichlet_bdy_2 = 7

        elif problem == 2:
            outname = "results/problem2_2D_simplex" if simplex else "results/problem2_2D_quads"
            name = "problem2_2D"
            model = gmsh.model()
            model.add(name)
            model.setCurrent(name)
            model = create_circle_plane_mesh(
                model,
                quads=not simplex,
                res=res,
                order=order,
                r=0.3,
                gap=0.1,
                height=0.1,
                length=1.0,
            )
            mesh, domain_marker, facet_marker = dolfinx.io.gmshio.model_to_mesh(
                model, MPI.COMM_WORLD, 0, gdim=2
            )
            dirichlet_bdy_1 = 8
            contact_bdy_1 = 10
            contact_bdy_2 = 6
            dirichlet_bdy_2 = 4
        elif problem == 3:
            outname = "results/problem3_2D_simplex" if simplex else "results/problem3_2D_quads"
            name = "problem2_2D"
            model = gmsh.model()
            model.add(name)
            model.setCurrent(name)
            model = create_circle_circle_mesh(model, quads=(not simplex), res=res, order=order)
            mesh, domain_marker, _ = dolfinx.io.gmshio.model_to_mesh(
                model, MPI.COMM_WORLD, 0, gdim=2
            )
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

            dirichlet_bdy_1 = 1
            contact_bdy_1 = 2
            contact_bdy_2 = 3
            dirichlet_bdy_2 = 4
            # Create meshtag for top and bottom markers
            top_facets1 = locate_entities_boundary(mesh, tdim - 1, top_dir)
            bottom_facets1 = locate_entities_boundary(mesh, tdim - 1, top_contact)
            top_facets2 = locate_entities_boundary(mesh, tdim - 1, bottom_contact)
            bottom_facets2 = locate_entities_boundary(mesh, tdim - 1, bottom_dir)
            dir_val1 = np.full(len(top_facets1), dirichlet_bdy_1, dtype=np.int32)
            c_val1 = np.full(len(bottom_facets1), contact_bdy_1, dtype=np.int32)
            surface_values = np.full(len(top_facets2), contact_bdy_2, dtype=np.int32)
            sbottom_values = np.full(len(bottom_facets2), dirichlet_bdy_2, dtype=np.int32)
            indices = np.concatenate([top_facets1, bottom_facets1, top_facets2, bottom_facets2])
            values = np.hstack([dir_val1, c_val1, surface_values, sbottom_values])
            sorted_facets = np.argsort(indices)

            facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])

    if mesh.comm.size > 1:
        mesh, facet_marker, domain_marker = create_contact_mesh(
            mesh, facet_marker, domain_marker, [contact_bdy_1, contact_bdy_2], 2.0
        )

    gmsh.finalize()

    tdim = mesh.topology.dim
    ncells = mesh.topology.index_map(tdim).size_local
    indices = np.array(range(ncells), dtype=np.int32)
    values = mesh.comm.rank * np.ones(ncells, dtype=np.int32)
    process_marker = meshtags(mesh, tdim, indices, values)
    process_marker.name = "process_marker"
    domain_marker.name = "cell_marker"
    facet_marker.name = "facet_marker"
    with XDMFFile(mesh.comm, f"{mesh_dir}/test.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(domain_marker, mesh.geometry)
        xdmf.write_meshtags(facet_marker, mesh.geometry)

    # Solver options
    ksp_tol = 1e-10
    newton_tol = 1e-7
    newton_options = {
        "relaxation_parameter": 1,
        "atol": newton_tol,
        "rtol": newton_tol,
        "convergence_criterion": "residual",
        "max_it": 200,
        "error_on_nonconvergence": True,
    }
    # petsc_options = {"ksp_type": "preonly", "pc_type": "lu"}
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
        "pc_gamg_coarse_eq_limit": 1000,
        "pc_gamg_agg_nsmooths": 1,
        "pc_gamg_threshold": 0.015,
        "pc_gamg_square_graph": 2,
        "pc_gamg_reuse_interpolation": True,
        "ksp_norm_type": "unpreconditioned",
    }
    # Pack mesh data for Nitsche solver
    dirichlet_vals = [dirichlet_bdy_1, dirichlet_bdy_2]
    contact = [(1, 0), (0, 1)]
    data = np.array([contact_bdy_1, contact_bdy_2], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    surfaces = adjacencylist(data, offsets)

    # Function, TestFunction, TrialFunction and measures
    V = functionspace(mesh, ("Lagrange", order, (mesh.geometry.dim,)))
    u = Function(V)
    du = Function(V)
    v = ufl.TestFunction(V)
    w = ufl.TrialFunction(V)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
    # Compute lame parameters
    mu_func, lambda_func = lame_parameters(plane_strain)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    # Create variational form without contact contributions
    F = ufl.inner(sigma(u), epsilon(v)) * dx

    mesh.topology.create_connectivity(tdim, tdim)

    # Nitsche parameters
    gdim = mesh.geometry.dim
    bcs = []
    bc_fns = []

    for k in range(displacement.shape[0]):
        d = displacement[k, :]
        tag = dirichlet_vals[k]
        g = Constant(mesh, default_scalar_type(tuple(d[i] for i in range(gdim))))
        bc_fns.append(g)
        if lifting:
            dofs = locate_dofs_topological(V, tdim - 1, facet_marker.find(tag))
            bcs.append(dirichletbc(g, dofs, V))
        else:
            F = weak_dirichlet(F, u, g, sigma, E * gamma * order**2, theta, ds(tag))

    F = ufl.replace(F, {u: u + du})
    J = ufl.derivative(F, du, w)

    # compiler options to improve performance
    jit_options = {"cffi_extra_compile_args": [], "cffi_libraries": ["m"]}
    # compiled forms for rhs and tangen system
    F_compiled = form(F, jit_options=jit_options)
    J_compiled = form(J, jit_options=jit_options)

    log.set_log_level(log.LogLevel.WARNING)
    num_newton_its = np.zeros(nload_steps, dtype=int)
    num_krylov_its = np.zeros(nload_steps, dtype=int)
    newton_time = np.zeros(nload_steps, dtype=np.float64)

    # solver_outfile = outfile if ksp_v else None

    V0 = functionspace(mesh, ("DG", 0))

    mu0 = Function(V0)
    lmbda0 = Function(V0)
    fric = Function(V0)
    mu0.interpolate(lambda x: np.full((1, x.shape[1]), mu))
    lmbda0.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))
    fric.interpolate(lambda x: np.full((1, x.shape[1]), fric_val))

    if raytracing:
        search_mode = [ContactMode.Raytracing for _ in range(len(contact))]
    else:
        search_mode = [ContactMode.ClosestPoint for _ in range(len(contact))]

    # create contact solver
    contact_problem = ContactProblem([facet_marker], surfaces, contact, mesh, q_degree, search_mode)
    if coulomb:
        friction_law = FrictionLaw.Coulomb
    else:
        friction_law = FrictionLaw.Tresca
    contact_problem.generate_contact_data(
        friction_law,
        V,
        {"u": u, "du": du, "mu": mu0, "lambda": lmbda0, "fric": fric},
        E * gamma * order**2,
        theta,
    )

    # define functions for newton solver
    def compute_coefficients(x, coeffs):
        size_local = V.dofmap.index_map.size_local
        bs = V.dofmap.index_map_bs
        du.x.array[: size_local * bs] = x.array_r[: size_local * bs]
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
    a_mat = contact_problem.create_matrix(J_compiled)
    b = create_vector(F_compiled)

    # Set up snes solver for nonlinear solver
    newton_solver = NewtonSolver(mesh.comm, a_mat, b, contact_problem.coeffs)
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

    # initialise vtx writer
    u.name = "u"
    vtx = VTXWriter(mesh.comm, f"{outname}_{nload_steps}_step.bp", [u], "bp4")
    vtx.write(0)
    for i in range(nload_steps):
        for k, g in enumerate(bc_fns):
            if len(bcs) > 0:
                g.value[:] = displacement[k, :] / nload_steps
            else:
                g.value[:] = (i + 1) * displacement[k, :] / nload_steps
        timing_str = f"~Contact: {i + 1} Newton Solver"
        with Timer(timing_str):
            n, converged = newton_solver.solve(du, write_solution=True)
        num_newton_its[i] = n

        du.x.scatter_forward()
        u.x.array[:] += du.x.array[:]
        contact_problem.update_contact_detection(u)
        a_mat = contact_problem.create_matrix(J_compiled)
        a_mat.setNearNullSpace(null_space)
        newton_solver.set_petsc_matrix(a_mat)
        du.x.array[:] = 0.1 * du.x.array[:]
        contact_problem.update_contact_data(du)
        vtx.write(i + 1)
    vtx.close()

    sigma_dev = sigma(u) - (1 / 3) * ufl.tr(sigma(u)) * ufl.Identity(len(u))
    sigma_vm = ufl.sqrt((3 / 2) * ufl.inner(sigma_dev, sigma_dev))
    W = functionspace(mesh, ("Discontinuous Lagrange", order - 1))
    sigma_vm_expr = Expression(sigma_vm, W.element.interpolation_points())
    sigma_vm_h = Function(W)
    sigma_vm_h.interpolate(sigma_vm_expr)
    sigma_vm_h.name = "vonMises"
    with XDMFFile(mesh.comm, "results/u_unbiased_total.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        u.name = "u"
        xdmf.write_function(u)

    with XDMFFile(mesh.comm, "results/u_unbiased_vonMises.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(sigma_vm_h)

    with XDMFFile(mesh.comm, "results/partitioning.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(process_marker, mesh.geometry)
    if timing_disp:
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
                u.function_space.dofmap.index_map_bs * u.function_space.dofmap.index_map.size_global
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
        print(f"Newton iterations {num_newton_its}, {sum(num_newton_its)}", file=outfile)
        print(f"Krylov iterations {num_krylov_its}, {sum(num_krylov_its)}", file=outfile)
        print("-" * 25, file=outfile)

    if outfile is not None:
        outfile.close()


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
        type=float,
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
    parser.add_argument(
        "--problem",
        default=1,
        type=int,
        dest="problem",
        help="Which problem to solve: 1. Flat surfaces, 2. One curved surface, 3. "
        "Two curved surfaces",
        choices=[1, 2, 3],
    )
    parser.add_argument(
        "--order",
        default=1,
        type=int,
        dest="order",
        help="Order of mesh geometry",
        choices=[1, 2, 3],
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
    _simplex = parser.add_mutually_exclusive_group(required=False)
    _simplex.add_argument(
        "--simplex",
        dest="simplex",
        action="store_true",
        help="Use triangle/tet mesh",
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
    parser.add_argument(
        "--E", default=1e3, type=np.float64, dest="E", help="Youngs modulus of material"
    )
    parser.add_argument("--nu", default=0.1, type=np.float64, dest="nu", help="Poisson's ratio")
    parser.add_argument(
        "--disp",
        default=0.2,
        type=np.float64,
        dest="disp",
        help="Displacement BC in negative y direction",
    )
    parser.add_argument(
        "--radius",
        default=0.5,
        type=np.float64,
        dest="radius",
        help="Search radius for ray-tracing",
    )
    parser.add_argument(
        "--load_steps",
        default=1,
        type=np.int32,
        dest="nload_steps",
        help="Number of steps for gradual loading",
    )
    parser.add_argument("--res", default=0.1, type=np.float64, dest="res", help="Mesh resolution")
    parser.add_argument(
        "--friction",
        default=0.0,
        type=np.float64,
        dest="fric",
        help="Friction coefficient for Tresca friction",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        required=False,
        help="File for appending results",
        dest="outfile",
    )
    _lifting = parser.add_mutually_exclusive_group(required=False)
    _lifting.add_argument(
        "--lifting",
        dest="lifting",
        action="store_true",
        help="Apply lifting (strong enforcement of Dirichlet condition",
        default=False,
    )
    _raytracing = parser.add_mutually_exclusive_group(required=False)
    _raytracing.add_argument(
        "--raytracing",
        dest="raytracing",
        action="store_true",
        help="Use raytracing for contact search.",
        default=False,
    )
    _coulomb = parser.add_mutually_exclusive_group(required=False)
    _coulomb.add_argument(
        "--coulomb",
        dest="coulomb",
        action="store_true",
        help="Use coulomb friction kernel. This requires --friction=[nonzero float value].",
        default=False,
    )

    # Parse input arguments or set to defualt values
    args = parser.parse_args()

    run_soler(args)
