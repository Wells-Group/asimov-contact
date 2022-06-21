# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import dolfinx.fem as _fem
import dolfinx.mesh as _mesh
from dolfinx.graph import create_adjacencylist
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary, meshtags
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx_contact.helpers import (epsilon, lame_parameters,
                                     rigid_motions_nullspace, sigma_func)
from dolfinx_contact.meshing import create_split_box_2D
from dolfinx_contact.meshtie import nitsche_meshtie


def u_fun(x, c):
    u2 = c * np.sin(2 * np.pi * x[0] / 5) * np.sin(2 * np.pi * x[1])
    return [np.zeros(len(x[0])), u2, np.zeros(len(x[0]))]


def fun(x, c, mu, lmbda):
    a = 2 * np.pi / 5
    b = 2 * np.pi
    f1 = -(lmbda + mu) * a * b * np.cos(a * x[0]) * np.cos(b * x[1])

    f2 = (mu * a**2 + (2 * mu + lmbda) * b**2) * np.sin(a * x[0]) * np.sin(b * x[1])

    return [c * f1, c * f2, np.zeros(len(x[0]))]


def unsplit_domain():
    errors = []
    for N in [5, 10, 20, 40, 80, 160]:
        mesh = _mesh.create_rectangle(MPI.COMM_WORLD, points=((0.0, 0.0), (5.0, 1.0)), n=(5 * N, N),
                                      cell_type=_mesh.CellType.triangle)
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        mesh.topology.create_connectivity(tdim - 1, tdim)

        # Compute lame parameters
        E = 1e3
        nu = 0.1
        mu_func, lambda_func = lame_parameters(False)
        mu = mu_func(E, nu)
        lmbda = lambda_func(E, nu)
        sigma = sigma_func(mu, lmbda)

        # Functions space and FEM functions
        V = _fem.VectorFunctionSpace(mesh, ("CG", 1))
        f = _fem.Function(V)
        c = 0.01
        f.interpolate(lambda x: fun(x, c, mu, lmbda))
        v = ufl.TestFunction(V)
        u = ufl.TrialFunction(V)

        # Boundary conditions
        facets = _mesh.locate_entities_boundary(mesh, dim=1, marker=lambda x: np.full(len(x[0]), True))
        bc = _fem.dirichletbc(np.zeros(tdim, dtype=PETSc.ScalarType),
                              _fem.locate_dofs_topological(V, entity_dim=tdim - 1, entities=facets), V=V)

        dx = ufl.Measure("dx", domain=mesh)
        J = _fem.form(ufl.inner(sigma(u), epsilon(v)) * dx)
        F = _fem.form(ufl.inner(f, v) * dx)

        A = _fem.petsc.assemble_matrix(J, bcs=[bc])
        A.assemble()

        # Set null-space
        null_space = rigid_motions_nullspace(V)
        A.setNearNullSpace(null_space)

        b = _fem.petsc.assemble_vector(F)
        _fem.petsc.apply_lifting(b, [J], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        _fem.petsc.set_bc(b, [bc])

        # Set solver options
        opts = PETSc.Options()
        opts["ksp_type"] = "cg"
        opts["ksp_rtol"] = 1.0e-10
        opts["pc_type"] = "gamg"

        # Use Chebyshev smoothing for multigrid
        opts["mg_levels_ksp_type"] = "chebyshev"
        opts["mg_levels_pc_type"] = "jacobi"

        # Improve estimate of eigenvalues for Chebyshev smoothing
        opts["mg_levels_esteig_ksp_type"] = "cg"
        opts["mg_levels_ksp_chebyshev_esteig_steps"] = 20

        # Create PETSc Krylov solver and turn convergence monitoring on
        solver = PETSc.KSP().create(mesh.comm)
        solver.setFromOptions()

        # Set matrix operator
        solver.setOperators(A)

        uh = _fem.Function(V)

        # Set a monitor, solve linear system, and display the solver
        # configuration
        solver.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))
        solver.solve(b, uh.vector)
        solver.view()

        # Scatter forward the solution vector to update ghost values
        uh.x.scatter_forward()

        # Error computation
        mesh_err = _mesh.create_rectangle(MPI.COMM_WORLD, points=((0.0, 0.0), (5.0, 1.0)), n=(5 * N, N),
                                          cell_type=_mesh.CellType.triangle)
        V_err = _fem.VectorFunctionSpace(mesh_err, ("CG", 5))
        u_ex = _fem.Function(V_err)
        u_ex.interpolate(lambda x: u_fun(x, c))

        dx = ufl.Measure("dx", domain=mesh_err)
        error_form = _fem.form(ufl.inner(u_ex - uh, u_ex - uh) * dx)
        error = _fem.assemble_scalar(error_form)
        errors.append(np.sqrt(mesh_err.comm.allreduce(error, op=MPI.SUM)))

    errors = np.array(errors)
    print((np.log(errors[0:5]) - np.log(errors[1:6])) / np.log(2))


def test_meshtie():
    res = 0.8
    c = 0.01
    # nitsche parameters
    nitsche_parameters = {"gamma": 10, "theta": 1, "lift_bc": False}
    # Compute lame parameters
    E = 1e3
    nu = 0.1
    mu_func, lambda_func = lame_parameters(False)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    physical_parameters = {"E": E, "nu": nu, "strain": False}

    # Solver options
    ksp_tol = 1e-10
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
        "pc_gamg_sym_graph": True,
        "pc_gamg_threshold": 1e-3,
        "pc_gamg_square_graph": 2,
    }
    errors = []
    times = []
    iterations = []
    runs = 7
    for i in range(1, runs + 1):
        fname = "beam"
        create_split_box_2D("beam", res)
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
        tdim = mesh.topology.dim
        gdim = mesh.geometry.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        mesh.topology.create_connectivity(tdim - 1, tdim)
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            domain_marker = xdmf.read_meshtags(mesh, name="domain_marker")
            facet_marker = xdmf.read_meshtags(mesh, name="contact_facets")

        def force_func(x): return fun(x, c, mu, lmbda)
        def zero_dirichlet(x): return np.zeros((gdim, x.shape[1]))
        dirichlet = [(3, zero_dirichlet), (5, zero_dirichlet)]
        body_forces = [(1, force_func), (2, force_func)]

        contact = [(1, 0), (0, 1)]
        data = np.array([4, 6], dtype=np.int32)
        offsets = np.array([0, 2], dtype=np.int32)
        surfaces = create_adjacencylist(data, offsets)
        # Solve contact problem using Nitsche's method
        u1, its, solver_time = nitsche_meshtie(
            mesh=mesh, mesh_tags=[facet_marker], domain_marker=domain_marker,
            surfaces=surfaces, dirichlet=dirichlet, neumann=[], contact_pairs=contact,
            body_forces=body_forces, physical_parameters=physical_parameters,
            nitsche_parameters=nitsche_parameters,
            quadrature_degree=3, petsc_options=petsc_options)

        V_err = _fem.VectorFunctionSpace(mesh, ("CG", 5))
        u_ex = _fem.Function(V_err)
        u_ex.interpolate(lambda x: u_fun(x, c))

        dx = ufl.Measure("dx", domain=mesh)
        error_form = _fem.form(ufl.inner(u_ex - u1, u_ex - u1) * dx)
        error = _fem.assemble_scalar(error_form)
        errors.append(np.sqrt(mesh.comm.allreduce(error, op=MPI.SUM)))
        res = 0.5 * res
        iterations.append(its)
        times.append(solver_time)
    errors = np.array(errors)
    print(errors)
    print((np.log(errors[0:runs - 1]) - np.log(errors[1:runs])) / np.log(2))
    print(times)
    print(iterations)


test_meshtie()
