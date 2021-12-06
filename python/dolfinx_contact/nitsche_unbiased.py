# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from typing import Tuple

import dolfinx.common as _common
import dolfinx.fem as _fem
import dolfinx.log as _log
import dolfinx.mesh as _mesh
import dolfinx_cuas
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx_contact
import dolfinx_contact.cpp
from dolfinx_contact.helpers import epsilon, lame_parameters, sigma_func

kt = dolfinx_contact.cpp.Kernel


def nitsche_unbiased(mesh: _mesh.Mesh, mesh_data: Tuple[_mesh.MeshTags, int, int, int, int],
                     physical_parameters: dict, refinement: int = 0,
                     nitsche_parameters: dict = {},
                     vertical_displacement: float = -0.1, nitsche_bc: bool = True, initGuess=None,
                     quadrature_degree: int = 3):

    # Compute lame parameters
    plane_strain = physical_parameters.get("strain", False)
    E = physical_parameters.get("E", 1e3)
    nu = physical_parameters.get("nu", 0.1)
    mu_func, lambda_func = lame_parameters(plane_strain)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    # Nitche parameters and variables
    theta = nitsche_parameters.get("theta", 1)
    gamma = E * nitsche_parameters.get("gamma", 10)

    (facet_marker, top_value, bottom_value, surface_value, surface_bottom) = mesh_data

    # Functions space and FEM functions
    V = _fem.VectorFunctionSpace(mesh, ("CG", 1))
    gdim = mesh.geometry.dim
    u = _fem.Function(V)
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    h = ufl.Circumradius(mesh)
    n = ufl.FacetNormal(mesh)
    # Integration measure and ufl part of linear/bilinear form
    metadata = {"quadrature_degree": quadrature_degree}
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh, metadata=metadata,
                     subdomain_data=facet_marker)
    a = ufl.inner(sigma(du), epsilon(v)) * dx - 0.5 * theta * h / gamma * ufl.inner(sigma(du) * n, sigma(v) * n) * \
        ds(bottom_value) - 0.5 * theta * h / gamma * ufl.inner(sigma(du) * n, sigma(v) * n) * ds(surface_value)
    L = ufl.inner(sigma(u), epsilon(v)) * dx - 0.5 * theta * h / gamma * ufl.inner(sigma(u) * n, sigma(v) * n) * \
        ds(bottom_value) - 0.5 * theta * h / gamma * ufl.inner(sigma(u) * n, sigma(v) * n) * ds(surface_value)

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
        # disp_plane[gdim - 1] = -0.5 * vertical_displacement
        u_D_plane = ufl.as_vector(disp_plane)
        L += - ufl.inner(sigma(u) * n, v) * ds(surface_bottom)\
             - theta * ufl.inner(sigma(v) * n, u - u_D_plane) * \
            ds(surface_bottom) + gamma / h * ufl.inner(u - u_D_plane, v) * ds(surface_bottom)
        a += - ufl.inner(sigma(du) * n, v) * ds(surface_bottom)\
            - theta * ufl.inner(sigma(v) * n, du) * \
            ds(surface_bottom) + gamma / h * ufl.inner(du, v) * ds(surface_bottom)
    else:
        raise RuntimeError("Strong Dirichlet bc's are not implemented in custom assemblers yet.")

    # Custom assembly
    _log.set_log_level(_log.LogLevel.OFF)
    # create contact class
    contact = dolfinx_contact.cpp.Contact(facet_marker, bottom_value, surface_value, V._cpp_object)
    contact.set_quadrature_degree(quadrature_degree)
    contact.create_distance_map(0)
    contact.create_distance_map(1)
    # pack constants
    consts = np.array([gamma, theta])

    # Pack material parameters mu and lambda on each contact surface
    V2 = _fem.FunctionSpace(mesh, ("DG", 0))
    lmbda2 = _fem.Function(V2)
    lmbda2.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))
    mu2 = _fem.Function(V2)
    mu2.interpolate(lambda x: np.full((1, x.shape[1]), mu))
    bottom_facets = facet_marker.indices[facet_marker.values == bottom_value]
    surface_facets = facet_marker.indices[facet_marker.values == surface_value]

    integral = _fem.IntegralType.exterior_facet
    bottom_entities = dolfinx_cuas.compute_active_entities(mesh, bottom_facets, integral)
    material_bottom = dolfinx_cuas.pack_coefficients([mu2, lmbda2], bottom_entities)

    surface_entities = dolfinx_cuas.compute_active_entities(mesh, surface_facets, integral)
    material_top = dolfinx_cuas.pack_coefficients([mu2, lmbda2], surface_entities)

    # Pack circumradius on each surface
    h_0 = dolfinx_contact.pack_circumradius_facet(mesh, bottom_facets)
    h_1 = dolfinx_contact.pack_circumradius_facet(mesh, surface_facets)

    # Pack gap and test functions on each surface
    gap_0 = contact.pack_gap(0)
    test_fn_0 = contact.pack_test_functions(0, gap_0)
    gap_1 = contact.pack_gap(1)
    test_fn_1 = contact.pack_test_functions(1, gap_1)

    # Concatenate all coeffs
    coeff_0 = np.hstack([material_bottom, h_0, gap_0, test_fn_0])
    coeff_1 = np.hstack([material_top, h_1, gap_1, test_fn_1])

    # Assemble jacobian
    a_cuas = _fem.Form(a)
    kernel_jac = contact.generate_kernel(kt.Jac)

    def create_A():
        return contact.create_matrix(a_cuas._cpp_object)

    def A(x, A):
        u.vector[:] = x.array
        u_opp_0 = contact.pack_u_contact(0, u._cpp_object, gap_0)
        u_opp_1 = contact.pack_u_contact(1, u._cpp_object, gap_1)
        u_0 = dolfinx_cuas.pack_coefficients([u], bottom_entities)
        u_1 = dolfinx_cuas.pack_coefficients([u], surface_entities)
        c_0 = np.hstack([coeff_0, u_0, u_opp_0])
        c_1 = np.hstack([coeff_1, u_1, u_opp_1])
        contact.assemble_matrix(A, [], 0, kernel_jac, c_0, consts)
        # To create onesided contact assembly remove comment
        # contact.assemble_matrix(A, [], 0, kernel_jac, c_0, consts)
        # To create rigid assembly comment next line
        contact.assemble_matrix(A, [], 1, kernel_jac, c_1, consts)
        _fem.assemble_matrix(A, a_cuas)

    # assemble rhs
    L_cuas = _fem.Form(L)
    kernel_rhs = contact.generate_kernel(kt.Rhs)

    def create_b():
        return _fem.create_vector(L_cuas)

    def F(x, b):
        u.vector[:] = x.array
        u_opp_0 = contact.pack_u_contact(0, u._cpp_object, gap_0)
        u_opp_1 = contact.pack_u_contact(1, u._cpp_object, gap_1)
        u_0 = dolfinx_cuas.pack_coefficients([u], bottom_entities)
        u_1 = dolfinx_cuas.pack_coefficients([u], surface_entities)
        c_0 = np.hstack([coeff_0, u_0, u_opp_0])
        c_1 = np.hstack([coeff_1, u_1, u_opp_1])
        contact.assemble_vector(b, 0, kernel_rhs, c_0, consts)
        # To create onesided contact assembly remove comment
        # contact.assemble_vector(b, 0, kernel_rhs, c_0, consts)
        # To create rigid assembly comment next line
        contact.assemble_vector(b, 1, kernel_rhs, c_1, consts)
        _fem.assemble_vector(b, L_cuas)

    problem = dolfinx_cuas.NonlinearProblemCUAS(F, A, create_b, create_A)
    # DEBUG: Write each step of Newton iterations
    # problem.i = 0
    # xdmf = dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/tmp_sol.xdmf", "w")
    # xdmf.write_mesh(mesh)

    # Set initial guess
    if initGuess is None:
        u.x.array[:] = 0
    else:
        u.x.array[:] = initGuess.x.array[:]

    solver = dolfinx_cuas.NewtonSolver(MPI.COMM_WORLD, problem)
    # null_space = rigid_motions_nullspace(V)
    # solver.A.setNearNullSpace(null_space)
    # Set Newton solver options
    solver.atol = 1e-9
    solver.rtol = 1e-9
    solver.convergence_criterion = "incremental"
    solver.max_it = 200
    solver.error_on_nonconvergence = True
    solver.relaxation_parameter = 0.6

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

    return u
