# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from typing import Tuple

import basix
import dolfinx.common as _common
import dolfinx.fem as _fem
import dolfinx.io as _io
import dolfinx.log as _log
import dolfinx.mesh as _mesh
import dolfinx_cuas
import dolfinx_cuas.cpp
import numpy as np
import ufl
from petsc4py import PETSc

import dolfinx_contact
import dolfinx_contact.cpp
from dolfinx_contact.helpers import (epsilon, lame_parameters,
                                     rigid_motions_nullspace, sigma_func)

__all__ = ["nitsche_rigid_surface_cuas"]
kt = dolfinx_contact.cpp.Kernel


def nitsche_rigid_surface_cuas(mesh: _mesh.Mesh, mesh_data: Tuple[_mesh.MeshTags, int, int, int, int],
                               physical_parameters: dict, refinement: int = 0,
                               nitsche_parameters: dict = {"gamma": 1, "theta": 1, "s": 0}, g: float = 0.0,
                               vertical_displacement: float = -0.1, nitsche_bc: bool = True, initGuess=None):
    (facet_marker, top_value, bottom_value, surface_value, surface_bottom) = mesh_data
    # write mesh and facet markers to xdmf
    with _io.XDMFFile(mesh.comm, f"results/mf_cuas_{refinement}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_marker)
    # quadrature degree
    q_deg = 3
    # Nitche parameters and variables
    theta = nitsche_parameters["theta"]
    # s = nitsche_parameters["s"]
    gamma = nitsche_parameters["gamma"] * physical_parameters["E"]
    gdim = mesh.geometry.dim
    n_vec = np.zeros(gdim)
    n_vec[gdim - 1] = 1
    # n_2 = ufl.as_vector(n_vec)  # Normal of plane (projection onto other body)
    n = ufl.FacetNormal(mesh)

    V = _fem.VectorFunctionSpace(mesh, ("CG", 1))
    E = physical_parameters["E"]
    nu = physical_parameters["nu"]
    mu_func, lambda_func = lame_parameters(physical_parameters["strain"])
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)
    bottom_facets = facet_marker.indices[facet_marker.values == bottom_value]

    u = _fem.Function(V)
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    # Initial condition
    def _u_initial(x):
        values = np.zeros((gdim, x.shape[1]))
        values[-1] = -vertical_displacement
        return values

    if initGuess is None:
        u.interpolate(_u_initial)
    else:
        u.x.array[:] = initGuess.x.array[:]

    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh,  # metadata=metadata,
                     subdomain_data=facet_marker)
    a = ufl.inner(sigma(du), epsilon(v)) * dx
    L = ufl.inner(sigma(u), epsilon(v)) * dx

    h = ufl.Circumradius(mesh)
    # Nitsche for Dirichlet, another theta-scheme.
    # https://doi.org/10.1016/j.cma.2018.05.024
    if nitsche_bc:
        disp_vec = np.zeros(gdim)
        disp_vec[gdim - 1] = vertical_displacement
        u_D = ufl.as_vector(disp_vec)
        L += - ufl.inner(sigma(u) * n, v) * ds(top_value)\
             - theta * ufl.inner(sigma(v) * n, u - u_D) * \
            ds(top_value) + gamma / h * ufl.inner(u - u_D, v) * ds(top_value)

        a += - ufl.inner(sigma(du) * n, v) * ds(top_value)\
            - theta * ufl.inner(sigma(v) * n, du) * \
            ds(top_value) + gamma / h * ufl.inner(du, v) * ds(top_value)
        # Nitsche bc for rigid plane
        disp_plane = np.zeros(gdim)
        u_D_plane = ufl.as_vector(disp_plane)
        L += - ufl.inner(sigma(u) * n, v) * ds(surface_bottom)\
             - theta * ufl.inner(sigma(v) * n, u - u_D_plane) * \
            ds(surface_bottom) + gamma / h * ufl.inner(u - u_D_plane, v) * ds(surface_bottom)
        a += - ufl.inner(sigma(du) * n, v) * ds(surface_bottom)\
            - theta * ufl.inner(sigma(v) * n, du) * \
            ds(surface_bottom) + gamma / h * ufl.inner(du, v) * ds(surface_bottom)
    else:
        print("Dirichlet bc not implemented in custom assemblers yet.")

    # Custom assembly
    _log.set_log_level(_log.LogLevel.OFF)
    q_rule = dolfinx_cuas.cpp.QuadratureRule(mesh.topology.cell_type, q_deg,
                                             mesh.topology.dim - 1, basix.QuadratureType.Default)
    consts = np.array([gamma, theta])
    consts = np.hstack((consts, n_vec))

    def lmbda_func2(x):
        values = np.zeros((1, x.shape[1]))
        for i in range(x.shape[1]):
            for j in range(1):
                values[j, i] = lmbda
        return values

    def mu_func2(x):
        values = np.zeros((1, x.shape[1]))
        for i in range(x.shape[1]):
            for j in range(1):
                values[j, i] = mu
        return values
    V2 = _fem.FunctionSpace(mesh, ("DG", 0))
    lmbda2 = _fem.Function(V2)
    lmbda2.interpolate(lmbda_func2)
    mu2 = _fem.Function(V2)
    mu2.interpolate(mu_func2)

    integral = _fem.IntegralType.exterior_facet
    integral_entities = dolfinx_cuas.cpp.compute_active_entities(mesh, bottom_facets, integral)

    coeffs = dolfinx_cuas.cpp.pack_coefficients([mu2._cpp_object, lmbda2._cpp_object], integral_entities)
    h_facets = dolfinx_contact.pack_circumradius_facet(mesh, bottom_facets)
    contact = dolfinx_contact.cpp.Contact(facet_marker, bottom_value, surface_value, V._cpp_object)
    contact.set_quadrature_degree(q_deg)
    contact.create_distance_map(0)
    g_vec = contact.pack_gap(0)
    n_surf = contact.pack_ny(0, g_vec)

    coeffs = np.hstack([coeffs, h_facets, g_vec, n_surf])

    # RHS
    L_cuas = _fem.Form(L)
    kernel_rhs = dolfinx_contact.cpp.generate_contact_kernel(V._cpp_object, kt.Rhs, q_rule,
                                                             [u._cpp_object, mu2._cpp_object, lmbda2._cpp_object],
                                                             False)

    def create_b():
        return _fem.create_vector(L_cuas)

    def F(x, b):
        u.vector[:] = x.array
        u_packed = dolfinx_cuas.cpp.pack_coefficients([u._cpp_object], integral_entities)
        c = np.hstack([u_packed, coeffs])
        dolfinx_cuas.assemble_vector(b, V, bottom_facets, kernel_rhs, c, consts, integral)
        _fem.assemble_vector(b, L_cuas)

    # Jacobian
    a_cuas = _fem.Form(a)
    kernel_J = dolfinx_contact.cpp.generate_contact_kernel(
        V._cpp_object, kt.Jac, q_rule, [u._cpp_object, mu2._cpp_object, lmbda2._cpp_object], False)

    def create_A():
        return _fem.create_matrix(a_cuas)

    def A(x, A):
        u.vector[:] = x.array
        u_packed = dolfinx_cuas.cpp.pack_coefficients([u._cpp_object], integral_entities)
        c = np.hstack([u_packed, coeffs])
        dolfinx_cuas.assemble_matrix(A, V, bottom_facets, kernel_J, c, consts, integral)
        _fem.assemble_matrix(A, a_cuas)

    problem = dolfinx_cuas.NonlinearProblemCUAS(F, A, create_b, create_A)
    # DEBUG: Write each step of Newton iterations
    # problem.i = 0
    # xdmf = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/tmp_sol.xdmf", "w")
    # xdmf.write_mesh(mesh)

    solver = dolfinx_cuas.NewtonSolver(mesh.comm, problem)
    null_space = rigid_motions_nullspace(V)
    solver.A.setNearNullSpace(null_space)

    # Set Newton solver options
    solver.atol = 1e-9
    solver.rtol = 1e-9
    solver.convergence_criterion = "incremental"
    solver.max_it = 200
    solver.error_on_nonconvergence = True
    solver.relaxation_parameter = 1

    # Define solver and options
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    # DEBUG: Use linear solver
    opts[f"{option_prefix}ksp_type"] = "preonly"
    opts[f"{option_prefix}pc_type"] = "lu"

    # opts[f"{option_prefix}ksp_type"] = "cg"
    # opts[f"{option_prefix}pc_type"] = "gamg"
    # opts[f"{option_prefix}rtol"] = 1.0e-6
    # opts[f"{option_prefix}pc_gamg_coarse_eq_limit"] = 1000
    # opts[f"{option_prefix}mg_levels_ksp_type"] = "chebyshev"
    # opts[f"{option_prefix}mg_levels_pc_type"] = "jacobi"
    # opts[f"{option_prefix}mg_levels_esteig_ksp_type"] = "cg"
    # opts[f"{option_prefix}matptap_via"] = "scalable"
    # View solver options
    # opts[f"{option_prefix}ksp_view"] = None
    ksp.setFromOptions()

    # Solve non-linear problem
    _log.set_log_level(_log.LogLevel.INFO)
    with _common.Timer(f"{refinement} Solve Nitsche"):
        n, converged = solver.solve(u)
    u.x.scatter_forward()
    if solver.error_on_nonconvergence:
        assert(converged)
    print(f"{V.dofmap.index_map_bs*V.dofmap.index_map.size_global}, Number of interations: {n:d}")

    with _io.XDMFFile(mesh.comm, f"results/u_cuas_rigid_{refinement}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        u.name = "u"
        xdmf.write_function(u)

    return u
