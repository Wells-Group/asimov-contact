# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import dolfinx.fem as _fem
import dolfinx.mesh as _mesh
from dolfinx.common import Timer, timing
from dolfinx.graph import create_adjacencylist
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary, meshtags
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx_contact.helpers import (epsilon, lame_parameters,
                                     rigid_motions_nullspace, sigma_func)
from dolfinx_contact.meshing import create_split_box_2D, create_split_box_3D, horizontal_sin
from dolfinx_contact.meshtie import nitsche_meshtie
from IPython import embed


def u_fun_2D(x, c, gdim):
    u2 = c * np.sin(2 * np.pi * x[0] / 5) * np.sin(2 * np.pi * x[1])
    vals = np.zeros((gdim, x.shape[1]))
    vals[1, :] = u2[:]
    return vals


def fun_2D(x, c, mu, lmbda, gdim):
    a = 2 * np.pi / 5
    b = 2 * np.pi

    vals = np.zeros((gdim, x.shape[1]))
    f1 = -(lmbda + mu) * a * b * np.cos(a * x[0]) * np.cos(b * x[1])

    f2 = (mu * a**2 + (2 * mu + lmbda) * b**2) * np.sin(a * x[0]) * np.sin(b * x[1])
    vals[0, :] = c * f1[:]
    vals[1, :] = c * f2[:]

    return vals


def u_fun_3D(x, d, gdim):
    u2 = d * np.sin(2 * np.pi * x[0] / 5) * np.sin(2 * np.pi * x[1]) * np.sin(2 * np.pi * x[2])
    vals = np.zeros((gdim, x.shape[1]))
    vals[1, :] = u2[:]
    return vals


def fun_3D(x, d, mu, lmbda, gdim):
    a = 2 * np.pi / 5
    b = 2 * np.pi
    c = 2 * np.pi
    f1 = -(lmbda + mu) * a * b * np.cos(a * x[0]) * np.cos(b * x[1]) * np.sin(c * x[2])

    f2 = (mu * (a**2 + c**2) + (2 * mu + lmbda) * b**2) * np.sin(a * x[0]) * np.sin(b * x[1]) * np.sin(c * x[2])

    f3 = -(lmbda + mu) * b * c * np.sin(a * x[0]) * np.cos(b * x[1]) * np.cos(c * x[2])
    vals = np.zeros((gdim, x.shape[1]))
    vals[0, :] = c * f1[:]
    vals[1, :] = c * f2[:]
    vals[2, :] = c * f3[:]
    return vals


def unsplit_domain(threed=False):
    errors = []
    NN = [4, 8, 15, 30, 58]
    ndofs = []
    times = []
    its = []
    for N in NN:
        if threed:
            mesh = _mesh.create_box(MPI.COMM_WORLD, points=((0.0, 0.0, 0.0), (5.0, 1.0, 1.0)), n=(5 * N, N, N),
                                    cell_type=_mesh.CellType.tetrahedron)
            fun = fun_3D
            u_fun = u_fun_3D
        else:
            mesh = _mesh.create_rectangle(MPI.COMM_WORLD, points=((0.0, 0.0), (5.0, 1.0)), n=(5 * N, N),
                                          cell_type=_mesh.CellType.triangle)
            fun = fun_2D
            u_fun = u_fun_2D
        tdim = mesh.topology.dim
        gdim = mesh.geometry.dim
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
        ndofs.append(V.dofmap.index_map_bs * V.dofmap.index_map.size_global)
        f = _fem.Function(V)
        c = 0.01
        f.interpolate(lambda x: fun(x, c, mu, lmbda, gdim))
        v = ufl.TestFunction(V)
        u = ufl.TrialFunction(V)

        # Boundary conditions
        facets = _mesh.locate_entities_boundary(mesh, dim=tdim - 1, marker=lambda x: np.full(len(x[0]), True))
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
        timing_str = f"~Krylov Solver"
        with Timer(timing_str):
            solver.solve(b, uh.vector)

        times.append(timing(timing_str)[1])
        its.append(solver.getIterationNumber())

        # Scatter forward the solution vector to update ghost values
        uh.x.scatter_forward()

        # Error computation
        V_err = _fem.VectorFunctionSpace(mesh, ("CG", 5))
        u_ex = _fem.Function(V_err)
        u_ex.interpolate(lambda x: u_fun(x, c, gdim))

        error_form = _fem.form(ufl.inner(u_ex - uh, u_ex - uh) * dx)
        error = _fem.assemble_scalar(error_form)
        errors.append(np.sqrt(mesh.comm.allreduce(error, op=MPI.SUM)))

    errors = np.array(errors)
    print("L2-error: ", errors)
    print("Number of dofs: ", ndofs)
    print("Linear solver time: ", times)
    print("Krylov iterations: ", its)
    print("Convergence rates: ", -(np.log(errors[0:- 1]) - np.log(errors[1:])) / (np.log(NN[0:-1]) - np.log(NN[1:])))


def test_meshtie(threed=False, simplex=True, runs=5):
    if simplex:
        res = 0.8
    else:
        res = 2.4
    num_segments = (2 * np.ceil(5.0 / 1.2).astype(np.int32), 2 * np.ceil(5.0 / (1.2 * 0.7)).astype(np.int32))
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
        "pc_type": "gamg",
        "mg_levels_ksp_type": "chebyshev",
        "mg_levels_pc_type": "jacobi",
        "mg_levels_esteig_ksp_type": "cg",
        "pc_gamg_coarse_eq_limit": 100,
        "mg_levels_ksp_chebyshev_esteig_steps": 20
    }
    errors = []
    times = []
    iterations = []
    dofs = []
    for i in range(1, runs + 1):
        if threed:
            fname = "beam3D"
            create_split_box_3D(fname, res=res, L=5.0, H=1.0, W=1.0, domain_1=[0, 1, 5, 4], domain_2=[4, 5, 2, 3], x0=[
                0, 0.5], x1=[5.0, 0.7], curve_fun=horizontal_sin, num_segments=num_segments, hex=not simplex)
            fun = fun_3D
            u_fun = u_fun_3D
        else:
            fname = "beam"
            create_split_box_2D(fname, res=res, L=5.0, H=1.0, domain_1=[0, 1, 5, 4], domain_2=[4, 5, 2, 3], x0=[
                0, 0.5], x1=[5.0, 0.7], curve_fun=horizontal_sin, num_segments=num_segments, quads=not simplex)
            fun = fun_2D
            u_fun = u_fun_2D
        embed()
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
        tdim = mesh.topology.dim
        gdim = mesh.geometry.dim
        mesh.topology.create_connectivity(tdim - 1, 0)
        mesh.topology.create_connectivity(tdim - 1, tdim)
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            domain_marker = xdmf.read_meshtags(mesh, name="domain_marker")
            facet_marker = xdmf.read_meshtags(mesh, name="contact_facets")

        def force_func(x): return fun(x, c, mu, lmbda, gdim)
        def zero_dirichlet(x): return np.zeros((gdim, x.shape[1]))
        dirichlet = [(3, zero_dirichlet), (5, zero_dirichlet)]
        body_forces = [(1, force_func), (2, force_func)]

        contact = [(1, 0), (0, 1)]
        data = np.array([4, 6], dtype=np.int32)
        offsets = np.array([0, 2], dtype=np.int32)
        surfaces = create_adjacencylist(data, offsets)
        # Solve contact problem using Nitsche's method
        u1, its, solver_time, ndofs = nitsche_meshtie(
            mesh=mesh, mesh_tags=[facet_marker], domain_marker=domain_marker,
            surfaces=surfaces, dirichlet=dirichlet, neumann=[], contact_pairs=contact,
            body_forces=body_forces, physical_parameters=physical_parameters,
            nitsche_parameters=nitsche_parameters,
            quadrature_degree=3, petsc_options=petsc_options)

        V_err = _fem.VectorFunctionSpace(mesh, ("CG", 5))
        u_ex = _fem.Function(V_err)
        u_ex.interpolate(lambda x: u_fun(x, c, gdim))

        dx = ufl.Measure("dx", domain=mesh)
        error_form = _fem.form(ufl.inner(u_ex - u1, u_ex - u1) * dx)
        error = _fem.assemble_scalar(error_form)
        errors.append(np.sqrt(mesh.comm.allreduce(error, op=MPI.SUM)))
        res = 0.5 * res
        num_segments = (2 * num_segments[0], 2 * num_segments[1])
        iterations.append(its)
        times.append(solver_time)
        dofs.append(ndofs)
    errors = np.array(errors)
    print("L2 errors; ", errors)
    print("Convergence rates: ", (np.log(errors[0:runs - 1]) - np.log(errors[1:runs])) / (np.log(2)))
    print("Solver time: ", times)
    print("Krylov iterations: ", iterations)
    print("Number of dofs: ", dofs)


# unsplit_domain(threed=False)
test_meshtie(simplex=True, threed=True, runs=1)
