# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from typing import Tuple

import basix
import dolfinx_cuas
import dolfinx_cuas.cpp
import numpy as np
import ufl
from dolfinx import common as _common
from dolfinx import fem as _fem
from dolfinx import io as _io
from dolfinx import log as _log
from dolfinx import mesh as dmesh
from petsc4py import PETSc as _PETSc

import dolfinx_contact
import dolfinx_contact.cpp
from dolfinx_contact.helpers import (epsilon, lame_parameters,
                                     rigid_motions_nullspace, sigma_func)

__all__ = ["nitsche_cuas"]

kt = dolfinx_contact.cpp.Kernel


def nitsche_cuas(mesh: dmesh.Mesh, mesh_data: Tuple[dmesh.MeshTags, int, int],
                 physical_parameters: dict, refinement: int = 0,
                 nitsche_parameters: dict = {"gamma": 1, "theta": 1, "s": 0}, g: float = 0.0,
                 vertical_displacement: float = -0.1, nitsche_bc: bool = True):
    (facet_marker, top_value, bottom_value) = mesh_data

    # quadrature degree
    q_deg = 5
    # Nitche parameters and variables
    theta = nitsche_parameters["theta"]
    # s = nitsche_parameters["s"]
    gamma = nitsche_parameters["gamma"] * physical_parameters["E"]
    n_vec = np.zeros(mesh.geometry.dim)
    n_vec[mesh.geometry.dim - 1] = -1
    n_2 = ufl.as_vector(n_vec)  # Normal of plane (projection onto other body)
    n = ufl.FacetNormal(mesh)

    V = _fem.VectorFunctionSpace(mesh, ("CG", 1))
    E = physical_parameters["E"]
    nu = physical_parameters["nu"]
    mu_func, lambda_func = lame_parameters(physical_parameters["strain"])
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)
    bottom_facets = facet_marker.indices[facet_marker.values == bottom_value]

    def sigma_n(v):
        # NOTE: Different normals, see summary paper
        return ufl.dot(sigma(v) * n, n_2)

    u = _fem.Function(V)
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    # Initial condition
    def _u_initial(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[-1] = -0.01 - g
        return values

    u = _fem.Function(V)
    v = ufl.TestFunction(V)
    u.interpolate(_u_initial)

    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh,  # metadata=metadata,
                     subdomain_data=facet_marker)
    a = ufl.inner(sigma(du), epsilon(v)) * dx
    L = ufl.inner(sigma(u), epsilon(v)) * dx

    h = ufl.Circumradius(mesh)
    # Nitsche for Dirichlet, another theta-scheme.
    # https://doi.org/10.1016/j.cma.2018.05.024
    if nitsche_bc:
        disp_vec = np.zeros(mesh.geometry.dim)
        disp_vec[mesh.geometry.dim - 1] = vertical_displacement
        u_D = ufl.as_vector(disp_vec)
        L += - ufl.inner(sigma(u) * n, v) * ds(top_value)\
             - theta * ufl.inner(sigma(v) * n, u - u_D) * \
            ds(top_value) + gamma / h * ufl.inner(u - u_D, v) * ds(top_value)
        a += - ufl.inner(sigma(du) * n, v) * ds(top_value)\
            - theta * ufl.inner(sigma(v) * n, du) * \
            ds(top_value) + gamma / h * ufl.inner(du, v) * ds(top_value)
    else:
        print("Dirichlet bc not implemented in custom assemblers yet.")

    # Custom assembly
    q_rule = dolfinx_cuas.cpp.QuadratureRule(mesh.topology.cell_type, q_deg,
                                             mesh.topology.dim - 1, basix.QuadratureType.Default)
    consts = np.array([gamma * E, theta])
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
    u.interpolate(_u_initial)

    integral = _fem.IntegralType.exterior_facet
    integral_entities = dolfinx_cuas.cpp.compute_active_entities(mesh, bottom_facets, integral)
    coeffs = dolfinx_cuas.cpp.pack_coefficients([mu2._cpp_object, lmbda2._cpp_object], integral_entities)
    h_facets = dolfinx_contact.cpp.pack_circumradius_facet(mesh, bottom_facets)
    contact = dolfinx_contact.cpp.Contact(facet_marker, bottom_value, top_value, V._cpp_object)
    contact.set_quadrature_degree(q_deg)
    g_vec = contact.pack_gap_plane(0, -g)
    coeffs = np.hstack([coeffs, h_facets, g_vec])

    # RHS
    L_cuas = _fem.Form(L)
    kernel_rhs = dolfinx_contact.cpp.generate_contact_kernel(V._cpp_object, kt.Rhs, q_rule,
                                                             [u._cpp_object, mu2._cpp_object, lmbda2._cpp_object])

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
        V._cpp_object, kt.Jac, q_rule, [u._cpp_object, mu2._cpp_object, lmbda2._cpp_object])

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
    # xdmf = dolfinx._io.XDMFFile(mesh.comm, "results/tmp_sol.xdmf", "w")
    # xdmf.write_mesh(mesh)

    solver = dolfinx_cuas.NewtonSolver(mesh.comm, problem)
    null_space = rigid_motions_nullspace(V)
    solver.A.setNearNullSpace(null_space)

    # Set Newton solver options
    solver.atol = 1e-9
    solver.rtol = 1e-9
    solver.convergence_criterion = "incremental"
    solver.max_it = 50
    solver.error_on_nonconvergence = True
    solver.relaxation_parameter = 1.0

    def _u_initial(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[-1] = -0.01 - g
        return values

    # Set initial_condition:
    u.interpolate(_u_initial)

    # Define solver and options
    ksp = solver.krylov_solver
    opts = _PETSc.Options()
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

    with _io.XDMFFile(mesh.comm, f"results/u_cuas_{refinement}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)

    return u
