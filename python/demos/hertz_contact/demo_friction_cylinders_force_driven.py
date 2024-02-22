# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse

import numpy as np
import ufl
from dolfinx import default_scalar_type
from dolfinx.io import XDMFFile, VTXWriter
from dolfinx.fem import (Constant, dirichletbc, form,
                         Function, FunctionSpace,
                         VectorFunctionSpace, locate_dofs_topological)
from dolfinx.fem.petsc import set_bc, apply_lifting, assemble_matrix, assemble_vector, create_vector
from dolfinx.graph import adjacencylist
from dolfinx.mesh import locate_entities
from mpi4py import MPI
from petsc4py.PETSc import InsertMode, ScatterMode  # type: ignore

from dolfinx_contact.helpers import (epsilon, sigma_func, lame_parameters,
                                     rigid_motions_nullspace)
from dolfinx_contact.meshing import (convert_mesh,
                                     create_quarter_disks_mesh)
from dolfinx_contact.newton_solver import NewtonSolver
from dolfinx_contact.cpp import ContactMode

from dolfinx_contact.general_contact.contact_problem import ContactProblem, FrictionLaw
from dolfinx_contact.output import ContactWriter

if __name__ == "__main__":
    desc = "Friction example with two elastic cylinders for verifying correctness of code"
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--quadrature", default=5, type=int, dest="q_degree",
                        help="Quadrature degree used for contact integrals")
    parser.add_argument("--order", default=1, type=int, dest="order",
                        help="Order of mesh geometry", choices=[1, 2])
    _simplex = parser.add_mutually_exclusive_group(required=False)
    _simplex.add_argument('--simplex', dest='simplex', action='store_true',
                          help="Use triangle/tet mesh", default=False)
    parser.add_argument("--res", default=0.06, type=np.float64, dest="res",
                        help="Mesh resolution")

    # Parse input arguments or set to defualt values
    args = parser.parse_args()
    simplex = args.simplex
    mesh_dir = "../meshes"

    # Problem parameters
    R = 8
    gap = 0.01
    p = 0.625
    E = 200
    nu = 0.3
    Estar = E / (2 * (1 - nu**2))
    mu_func, lambda_func = lame_parameters(True)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    fric_val = 0.2
    q_tan = 0.05851

    # Create mesh
    name = "force_driven_cylinders"
    outname = f"../results/{name}_simplex" if simplex else f"results/{name}_quads"
    fname = f"{mesh_dir}/{name}_simplex" if simplex else f"{mesh_dir}/{name}_quads"
    create_quarter_disks_mesh(f"{fname}.msh", args.res, args.order, not simplex, R, gap)

    convert_mesh(fname, f"{fname}.xdmf", gdim=2)
    with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()
        domain_marker = xdmf.read_meshtags(mesh, name="cell_marker")
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, tdim)
        facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")

    vals = facet_marker.values
    facet_marker.values[vals == 5] = 4
    facet_marker.values[vals == 8] = 7
    facet_marker.values[vals == 11] = 10
    facet_marker.values[vals == 14] = 13

    contact_bdy_1 = 4
    contact_bdy_2 = 10
    top = 7
    bottom = 13

    # Solver options
    ksp_tol = 1e-10
    newton_tol = 1e-7
    newton_options = {"relaxation_parameter": 1.0,
                      "atol": newton_tol,
                      "rtol": newton_tol,
                      "convergence_criterion": "residual",
                      "max_it": 200,
                      "error_on_nonconvergence": True}

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
        "pc_gamg_reuse_interpolation": False,
        "ksp_initial_guess_nonzero": False,
        "ksp_norm_type": "unpreconditioned"
    }

    # Step 1: frictionless contact
    V = VectorFunctionSpace(mesh, ("CG", args.order))
    # boundary conditions
    t = Constant(mesh, default_scalar_type((0.0, -p)))
    g = Constant(mesh, default_scalar_type((0.0)))

    symmetry_nodes = locate_entities(mesh, 0, lambda x: np.logical_and(np.isclose(x[0], 0), x[1] >= -8))
    dofs_symmetry = locate_dofs_topological(V.sub(0), 0, symmetry_nodes)
    dofs_bottom = locate_dofs_topological(V, 1, facet_marker.find(bottom))
    dofs_top = locate_dofs_topological(V, 1, facet_marker.find(top))

    g_top = Constant(mesh, default_scalar_type((0.0, -0.1)))
    bcs = [dirichletbc(g, dofs_symmetry, V.sub(0)),
           dirichletbc(Constant(mesh, default_scalar_type((0.0, 0.0))), dofs_bottom, V)]

    # DG-0 funciton for material
    V0 = FunctionSpace(mesh, ("DG", 0))
    mu_dg = Function(V0)
    lmbda_dg = Function(V0)
    mu_dg.interpolate(lambda x: np.full((1, x.shape[1]), mu))
    lmbda_dg.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))
    sigma = sigma_func(mu, lmbda)

    # Pack mesh data for Nitsche solver
    contact_pairs = [(0, 1), (1, 0)]
    data = np.array([contact_bdy_1, contact_bdy_2], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    surfaces = adjacencylist(data, offsets)

    # Function, TestFunction, TrialFunction and measures
    u = Function(V)
    v = ufl.TestFunction(V)
    du = Function(V)
    w = ufl.TrialFunction(V)
    dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)

    # Create variational form without contact contributions
    F = ufl.inner(sigma(u), epsilon(v)) * dx

    # body forces
    F -= ufl.inner(t, v) * ds(top)

    F = ufl.replace(F, {u: u + du})

    J = ufl.derivative(F, du, w)

    # compiler options to improve performance
    cffi_options = ["-Ofast", "-march=native"]
    jit_options = {"cffi_extra_compile_args": cffi_options,
                   "cffi_libraries": ["m"]}
    # compiled forms for rhs and tangen system
    F_compiled = form(F, jit_options=jit_options)
    J_compiled = form(J, jit_options=jit_options)

    # Set up force postprocessing
    n = ufl.FacetNormal(mesh)
    ex = Constant(mesh, default_scalar_type((1.0, 0.0)))
    ey = Constant(mesh, default_scalar_type((0.0, 1.0)))
    Rx_form = form(ufl.inner(sigma(u) * n, ex) * ds(top))
    Ry_form = form(ufl.inner(sigma(u) * n, ey) * ds(top))

    def _tangent(x, p, a, c):
        vals = np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            if abs(x[0][i]) <= c:
                vals[i] = fric_val * 4 * R * p / (np.pi * a**2) * (np.sqrt(a**2 - x[0]
                                                                           [i]**2) - np.sqrt(c**2 - x[0][i]**2))
            elif abs(x[0][i]) < a:
                vals[i] = fric_val * 4 * R * p / (np.pi * a**2) * (np.sqrt(a**2 - x[0][i]**2))
        return vals

    def _pressure(x, p0, a):
        vals = np.zeros(x.shape[1])
        for i in range(x.shape[1]):
            if abs(x[0][i]) < a:
                vals[i] = p0 * np.sqrt(1 - x[0][i]**2 / a**2)
        return vals

    top_cells = domain_marker.find(1)

    # u.interpolate(_u_initial, top_cells)
    search_mode = [ContactMode.ClosestPoint, ContactMode.ClosestPoint]

    # Solve contact problem using Nitsche's method
    steps1 = 4
    contact_problem = ContactProblem([facet_marker], surfaces, contact_pairs, mesh, args.q_degree, search_mode)
    contact_problem.generate_contact_data(FrictionLaw.Frictionless, V, {"u": u, "du": du, "mu": mu_dg,
                                                                        "lambda": lmbda_dg}, E * 100 * args.order**2, 1)

    h = contact_problem.h_surfaces()[1]
    # create initial guess

    def _u_initial(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[-1] = -0.01 - gap
        return values

    du.interpolate(_u_initial, top_cells)

    # define functions for newton solver
    def compute_coefficients(x, coeffs):
        du.x.scatter_forward()
        contact_problem.update_contact_data(du)

    def compute_residual(x, b, coeffs):
        b.zeroEntries()
        b.ghostUpdate(addv=InsertMode.INSERT,
                      mode=ScatterMode.FORWARD)
        contact_problem.assemble_vector(b, V)
        assemble_vector(b, F_compiled)

        # Apply boundary condition
        if len(bcs) > 0:
            apply_lifting(
                b, [J_compiled], bcs=[bcs], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=InsertMode.ADD,
                      mode=ScatterMode.REVERSE)
        if len(bcs) > 0:
            set_bc(b, bcs, x, -1.0)

    def compute_jacobian_matrix(x, a_mat, coeffs):
        a_mat.zeroEntries()
        contact_problem.assemble_matrix(a_mat, V)
        assemble_matrix(a_mat, J_compiled, bcs=bcs)
        a_mat.assemble()

    writer = ContactWriter(mesh, contact_problem, u, contact_pairs,
                           contact_problem.coeffs, args.order, simplex,
                           [(tdim - 1, 0), (tdim - 1, -R)],
                           f"{outname}")
    # initialise vtx writer
    vtx = VTXWriter(mesh.comm, f"{outname}.bp", [u], "bp4")
    vtx.write(0)
    # create vector and matrix
    A = contact_problem.create_matrix(J_compiled)
    b = create_vector(F_compiled)

    # Set up newton solver for nonlinear solver
    newton_solver = NewtonSolver(mesh.comm, A, b, contact_problem.coeffs)
    # Set matrix-vector computations
    newton_solver.set_residual(compute_residual)
    newton_solver.set_jacobian(compute_jacobian_matrix)
    newton_solver.set_coefficients(compute_coefficients)

    # Set rigid motion nullspace
    null_space = rigid_motions_nullspace(V)
    newton_solver.A.setNearNullSpace(null_space)

    # Set Newton solver options
    newton_solver.set_newton_options(newton_options)

    # Set Krylov solver options
    newton_solver.set_krylov_options(petsc_options)
    newton_steps1 = []

    for i in range(steps1):

        val = -p * (i + 1) / steps1  # -0.2 / steps1  #
        t.value[1] = val
        print(f"Fricitionless part: Step {i+1} of {steps1}----------------------------------------------")
        set_bc(du.vector, bcs)
        n, converged = newton_solver.solve(du, write_solution=True)
        newton_steps1.append(n)
        du.x.scatter_forward()
        u.x.array[:] += du.x.array[:]
        # Compute forces
        pr = abs(val)
        q = 0.0
        load = pr * 2 * R
        a = 2 * np.sqrt(R * load / (2 * np.pi * Estar))
        p0 = 2 * load / (np.pi * a)
        print(pr, q)
        # print(val, 0)
        c = a * np.sqrt(1 - 0 / (fric_val * pr))
        writer.write(i + 1, lambda x, pi=p0, ai=a: _pressure(x, pi, ai),
                     lambda x, pi=pr, ai=a, ci=c: _tangent(x, pi, ai, ci))
        vtx.write(i + 1)

        contact_problem.update_contact_detection(u)
        A = contact_problem.create_matrix(J_compiled)
        A.setNearNullSpace(null_space)
        newton_solver.set_petsc_matrix(A)
        du.x.array[:] = 0.1 * du.x.array[:]
        contact_problem.update_contact_data(du)

    # # Step 2: Frictional contact
    ksp_tol = 1e-12
    petsc_options = {
        "matptap_via": "scalable",
        "ksp_type": "gmres",
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
        "pc_gamg_reuse_interpolation": False,
        "ksp_initial_guess_nonzero": False,
        "ksp_norm_type": "unpreconditioned"
    }

    dofs_constraint = locate_dofs_topological(V.sub(1), 1, facet_marker.find(top))
    g_top = Constant(mesh, default_scalar_type((0.0, 0.0)))
    bcs = [dirichletbc(Constant(mesh, default_scalar_type((0.0, 0.0))), dofs_bottom, V),
           dirichletbc(Constant(mesh, default_scalar_type(0.0)), dofs_constraint, V.sub(1))]

    fric = Function(V0)
    fric.interpolate(lambda x: np.full((1, x.shape[1]), fric_val))
    contact_problem.generate_contact_data(FrictionLaw.Coulomb, V, {"u": u, "du": du, "mu": mu_dg,
                                                                   "lambda": lmbda_dg, "fric": fric},
                                          E * 10 * args.order**2, 1)
    newton_solver.update_krylov_solver(petsc_options)
    steps2 = 8

    newton_steps2 = []
    for i in range(steps2):
        print(f"Fricitional part: Step {i+1} of {steps2}----------------------------------------------")
        # print(problem1.du.x.array[:])
        set_bc(du.vector, bcs)
        val = q_tan * (i + 1) / steps2
        t.value[0] = val
        n, converged = newton_solver.solve(du, write_solution=True)
        newton_steps2.append(n)
        du.x.scatter_forward()
        u.x.array[:] += du.x.array[:]
        # Compute forces
        pr = abs(p)
        q = abs(val)
        load = 2 * R * abs(pr)
        a = 2 * np.sqrt(R * load / (2 * np.pi * Estar))
        p0 = 2 * load / (np.pi * a)
        print(pr, q)
        c = a * np.sqrt(1 - q / (fric_val * abs(pr)))
        writer.write(steps1 + i + 1, lambda x, pi=p0, ai=a: _pressure(x, pi, ai),
                     lambda x, pi=pr, ai=a, ci=c: _tangent(x, pi, ai, ci))
        vtx.write(steps1 + 1 + i)
        contact_problem.update_contact_detection(u)
        A = contact_problem.create_matrix(J_compiled)
        A.setNearNullSpace(null_space)
        newton_solver.set_petsc_matrix(A)
        # take a fraction of du as initial guess
        # this is to ensure non-singular matrices in the case of no Dirichlet boundary
        du.x.array[:] = 0.1 * du.x.array[:]
        contact_problem.update_contact_data(du)
    vtx.close()

    print("Newton iterations: ")
    print(newton_steps1)
    print(newton_steps2)
