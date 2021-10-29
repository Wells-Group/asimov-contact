# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import dolfinx
import dolfinx.io
import dolfinx_contact
import dolfinx_contact.cpp
import numpy as np
import ufl
from typing import Tuple
from dolfinx_contact.helpers import (epsilon, lame_parameters, sigma_func)

import scipy.sparse
import matplotlib.pylab as plt

kt = dolfinx_contact.cpp.Kernel


def nitsche_unbiased(mesh: dolfinx.cpp.mesh.Mesh, mesh_data: Tuple[dolfinx.MeshTags, int, int, int, int],
                     physical_parameters: dict, refinement: int = 0,
                     nitsche_parameters: dict = {"gamma": 1, "theta": 1},
                     vertical_displacement: float = -0.1, nitsche_bc: bool = True, initGuess=None):
    (facet_marker, top_value, bottom_value, surface_value, surface_bottom) = mesh_data
    # quadrature degree
    q_deg = 3
    # Nitche parameters and variables
    theta = nitsche_parameters["theta"]
    gamma = nitsche_parameters["gamma"] * physical_parameters["E"]

    # elasticity parameters
    E = physical_parameters["E"]
    nu = physical_parameters["nu"]
    mu_func, lambda_func = lame_parameters(physical_parameters["strain"])
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    # Functions space and FEM functions
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))
    gdim = mesh.geometry.dim
    u = dolfinx.Function(V)
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    # Initial condition
    def _u_initial(x):
        values = np.zeros((gdim, x.shape[1]))
        for i in range(x.shape[1]):
            if x[-1, i] > 0:
                values[-1, i] = -vertical_displacement
            else:
                values[-1, i] = vertical_displacement
        return values

    if initGuess is None:
        u.interpolate(_u_initial)
    else:
        u.x.array[:] = initGuess.x.array[:]

    # integration measure and ufl part of linear/bilinear form
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh,  # metadata=metadata,
                     subdomain_data=facet_marker)
    a = ufl.inner(sigma(du), epsilon(v)) * dx
    L = ufl.inner(sigma(u), epsilon(v)) * dx

    h = ufl.Circumradius(mesh)
    n = ufl.FacetNormal(mesh)
    # Nitsche for Dirichlet, another theta-scheme.
    # https://doi.org/10.1016/j.cma.2018.05.024
    if nitsche_bc:
        disp_vec = np.zeros(gdim)
        disp_vec[gdim - 1] = 0.5 * vertical_displacement
        u_D = ufl.as_vector(disp_vec)
        L += - ufl.inner(sigma(u) * n, v) * ds(top_value)\
             - theta * ufl.inner(sigma(v) * n, u - u_D) * \
            ds(top_value) + gamma / h * ufl.inner(u - u_D, v) * ds(top_value)

        a += - ufl.inner(sigma(du) * n, v) * ds(top_value)\
            - theta * ufl.inner(sigma(v) * n, du) * \
            ds(top_value) + gamma / h * ufl.inner(du, v) * ds(top_value)
        # Nitsche bc for rigid plane
        disp_plane = np.zeros(gdim)
        disp_plane[gdim - 1] = - 0.5 * vertical_displacement
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
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.OFF)
    # create contact class
    contact = dolfinx_contact.cpp.Contact(facet_marker, bottom_value, surface_value, V._cpp_object)
    contact.set_quadrature_degree(q_deg)
    contact.create_distance_map(0)
    contact.create_distance_map(1)
    # pack constants
    consts = np.array([gamma * E, theta])

    # Pack all coefficients
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

    V2 = dolfinx.FunctionSpace(mesh, ("DG", 0))
    lmbda2 = dolfinx.Function(V2)
    lmbda2.interpolate(lmbda_func2)
    mu2 = dolfinx.Function(V2)
    mu2.interpolate(mu_func2)

    mu_packed_0 = contact.pack_coefficient_dofs(0, mu2._cpp_object)
    mu_packed_1 = contact.pack_coefficient_dofs(1, mu2._cpp_object)
    lmbda_packed_0 = contact.pack_coefficient_dofs(0, lmbda2._cpp_object)
    lmbda_packed_1 = contact.pack_coefficient_dofs(1, lmbda2._cpp_object)

    bottom_facets = facet_marker.indices[facet_marker.values == bottom_value]
    surface_facets = facet_marker.indices[facet_marker.values == surface_value]
    h_0 = dolfinx_contact.cpp.pack_circumradius_facet(mesh, bottom_facets)
    h_1 = dolfinx_contact.cpp.pack_circumradius_facet(mesh, surface_facets)

    gap_0 = contact.pack_gap(0)
    test_fn_0 = contact.pack_test_functions(0, gap_0)
    gap_1 = contact.pack_gap(1)
    test_fn_1 = contact.pack_test_functions(1, gap_1)

    u_opp_0 = contact.pack_u_contact(0, u._cpp_object, gap_0)
    u_opp_1 = contact.pack_u_contact(1, u._cpp_object, gap_1)
    u_0 = contact.pack_coefficient_dofs(0, u._cpp_object)
    u_1 = contact.pack_coefficient_dofs(1, u._cpp_object)

    coeff_0 = np.hstack([mu_packed_0, lmbda_packed_0, h_0, gap_0, test_fn_0, u_0, u_opp_0])
    coeff_1 = np.hstack([mu_packed_1, lmbda_packed_1, h_1, gap_1, test_fn_1, u_1, u_opp_1])

    # assemble jacobian
    a_cuas = dolfinx.fem.Form(a)
    A = contact.create_matrix(a_cuas._cpp_object)
    kernel_0 = contact.generate_kernel(0, kt.Jac)
    kernel_1 = contact.generate_kernel(1, kt.Jac)

    contact.assemble_matrix(A, [], 0, kernel_0, coeff_0, consts)
    contact.assemble_matrix(A, [], 1, kernel_1, coeff_1, consts)
    A.assemble()

    # Create scipy CSR matrices
    ai, aj, av = A.getValuesCSR()
    A_sp = scipy.sparse.csr_matrix((av, aj, ai), shape=A.getSize())
    plt.spy(A_sp)
    plt.savefig("test.png")
