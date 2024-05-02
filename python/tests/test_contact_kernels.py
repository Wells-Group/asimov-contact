# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:   MIT

from mpi4py import MPI

import numpy as np
import pytest

import basix
import dolfinx.mesh
import dolfinx_contact
import dolfinx_contact.cpp
import ufl
from dolfinx.fem import Function, IntegralType, form, functionspace
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_matrix, create_vector
from dolfinx.graph import adjacencylist
from dolfinx.mesh import create_unit_cube, create_unit_square, locate_entities_boundary, meshtags

kt = dolfinx_contact.cpp.Kernel
compare_matrices = dolfinx_contact.helpers.compare_matrices


# R_minus(x) returns x if negative zero else
def R_minus(x):
    abs_x = abs(x)
    return 0.5 * (x - abs_x)


@pytest.mark.parametrize("kernel_type", [kt.Rhs])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("P", [1, 2, 3, 4])
@pytest.mark.parametrize("Q", [0, 1, 2])
def test_vector_surface_kernel(dim, kernel_type, P, Q):
    N = 20 if dim == 2 else 5
    mesh = create_unit_square(MPI.COMM_WORLD, N, N) if dim == 2 else create_unit_cube(MPI.COMM_WORLD, N, N, N)

    # Find facets on boundary to integrate over
    facets = locate_entities_boundary(mesh, mesh.topology.dim - 1,
                                      lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                              np.isclose(x[0], 1.0)))
    facets = np.sort(facets)
    values = np.ones(len(facets), dtype=np.int32)
    ft = meshtags(mesh, mesh.topology.dim - 1, facets, values)

    # Define variational form
    V = functionspace(mesh, ("Lagrange", P, (mesh.geometry.dim, )))

    def f(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        for i in range(mesh.geometry.dim):
            values[i] = 0.5 - x[i]
        return values

    def lmbda_func(x):
        values = np.zeros((1, x.shape[1]))
        for i in range(x.shape[1]):
            for j in range(1):
                values[j, i] = x[j, i] + 2
        return values

    def mu_func(x):
        values = np.zeros((1, x.shape[1]))
        for i in range(x.shape[1]):
            for j in range(1):
                values[j, i] = np.sin(x[j, i]) + 2
        return values

    u = Function(V)
    u.interpolate(f)
    h = ufl.Circumradius(mesh)
    v = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    dx = ufl.Measure("dx", domain=mesh)

    V2 = functionspace(mesh, ("Discontinuous Lagrange", Q))
    lmbda = Function(V2)
    lmbda.interpolate(lmbda_func)
    mu = Function(V2)
    mu.interpolate(mu_func)

    # Nitsche parameters
    theta = 1.0
    gamma = 20

    n_vec = np.zeros(mesh.geometry.dim)
    n_vec[mesh.geometry.dim - 1] = 1
    # FIXME: more general definition of n_2 needed for surface that is not a horizontal rectangular box.
    n_2 = ufl.as_vector(n_vec)  # Normal of plane (projection onto other body)
    n = ufl.FacetNormal(mesh)

    def epsilon(v):
        return ufl.sym(ufl.grad(v))

    def sigma(v):
        return (2.0 * mu * epsilon(v) + lmbda * ufl.tr(epsilon(v)) * ufl.Identity(len(v)))
        # return ufl.tr(epsilon(v)) * ufl.Identity(len(v))

    def sigma_n(v):
        # NOTE: Different normals, see summary paper
        return ufl.dot(sigma(v) * n, (-n_2))

    # Mimicking the plane y=-g
    g = 0.1
    x = ufl.SpatialCoordinate(mesh)
    gap = x[mesh.geometry.dim - 1] + g
    L = ufl.inner(sigma(u), epsilon(v)) * dx
    L += - h * theta / gamma * sigma_n(u) * sigma_n(v) * ds(1)
    L += h / gamma * R_minus(sigma_n(u) + (gamma / h) * (gap - ufl.dot(u, (-n_2)))) * \
        (theta * sigma_n(v) - (gamma / h) * ufl.dot(v, (-n_2))) * ds(1)
    # Compile UFL form
    cffi_options = ["-Ofast", "-march=native"]
    L = form(L, jit_options={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    b = create_vector(L)

    # Normal assembly
    b.zeroEntries()
    assemble_vector(b, L)
    b.assemble()

    # Custom assembly
    # num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    consts = np.array([gamma, theta])
    consts = np.hstack((consts, n_vec))
    integral_entities, num_local = dolfinx_contact.compute_active_entities(
        mesh._cpp_object, ft.indices, IntegralType.exterior_facet)
    integral_entities = integral_entities[:num_local]
    mu_packed = dolfinx_contact.cpp.pack_coefficient_quadrature(mu._cpp_object, 0, integral_entities)
    lmbda_packed = dolfinx_contact.cpp.pack_coefficient_quadrature(lmbda._cpp_object, 0, integral_entities)
    u_packed = dolfinx_contact.cpp.pack_coefficient_quadrature(u._cpp_object, 2 * P
                                                               + Q + 1, integral_entities)
    grad_u_packed = dolfinx_contact.cpp.pack_gradient_quadrature(u._cpp_object, 2 * P
                                                                 + Q + 1, integral_entities)
    h_facets = dolfinx_contact.pack_circumradius(mesh._cpp_object, integral_entities)
    data = np.array([1], dtype=np.int32)
    offsets = np.array([0, 1], dtype=np.int32)
    surfaces = adjacencylist(data, offsets)
    search_mode = [dolfinx_contact.cpp.ContactMode.ClosestPoint]
    contact = dolfinx_contact.cpp.Contact([ft._cpp_object], surfaces, [(0, 0)],
                                          mesh._cpp_object, search_mode, quadrature_degree=2 * P + Q + 1)
    contact.create_distance_map(0)
    g_vec = contact.pack_gap_plane(0, -g)
    # FIXME: assuming all facets are the same type
    q_rule = dolfinx_contact.QuadratureRule(mesh.topology.cell_type, 2 * P
                                            + Q + 1, mesh.topology.dim - 1, basix.QuadratureType.Default)
    coeffs = np.hstack([mu_packed, lmbda_packed, h_facets, g_vec, u_packed, grad_u_packed])

    L_custom = ufl.inner(sigma(u), epsilon(v)) * dx
    L_custom = form(L_custom)
    b2 = create_vector(L_custom)
    kernel = dolfinx_contact.cpp.generate_rigid_surface_kernel(V._cpp_object, kernel_type, q_rule)

    b2.zeroEntries()
    contact.assemble_vector(b2, 0, kernel, coeffs, consts, V._cpp_object)
    assemble_vector(b2, L_custom)
    b2.assemble()
    assert np.allclose(b.array, b2.array)


@pytest.mark.parametrize("kernel_type", [kt.Jac])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("P", [1, 2, 3, 4])
@pytest.mark.parametrize("Q", [0, 1, 2])
def test_matrix_surface_kernel(dim, kernel_type, P, Q):
    N = 20 if dim == 2 else 5
    mesh = create_unit_square(MPI.COMM_WORLD, N, N) if dim == 2 else create_unit_cube(MPI.COMM_WORLD, N, N, N)

    # Find facets on boundary to integrate over
    facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1,
                                                   lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                           np.isclose(x[0], 1.0)))
    facets = np.sort(facets)
    values = np.ones(len(facets), dtype=np.int32)
    ft = meshtags(mesh, mesh.topology.dim - 1, facets, values)

    # Define variational form
    V = functionspace(mesh, ("Lagrange", P, (mesh.geometry.dim, )))
    du = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    dx = ufl.Measure("dx", domain=mesh)

    def f(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        for i in range(x.shape[1]):
            for j in range(mesh.geometry.dim):
                values[j, i] = np.sin(x[j, i]) + x[j, i]
        return values

    def lmbda_func(x):
        values = np.zeros((1, x.shape[1]))
        for i in range(x.shape[1]):
            for j in range(1):
                values[j, i] = 50 * np.sin(np.max(x[:, i]))
        return values

    def mu_func(x):
        values = np.zeros((1, x.shape[1]))
        for i in range(x.shape[1]):
            for j in range(1):
                values[j, i] = np.sin(x[j, i])
        return values

    u = Function(V)
    u.interpolate(f)
    V2 = functionspace(mesh, ("Discontinuous Lagrange", Q))
    lmbda = Function(V2)
    lmbda.interpolate(lmbda_func)
    mu = Function(V2)
    mu.interpolate(mu_func)

    # Nitsche parameters
    theta = 1.0
    gamma = 20

    n_vec = np.zeros(mesh.geometry.dim)
    n_vec[mesh.geometry.dim - 1] = 1
    # FIXME: more general definition of n_2 needed for surface that is not a horizontal rectangular box.
    n_2 = ufl.as_vector(n_vec)  # Normal of plane (projection onto other body)
    n = ufl.FacetNormal(mesh)

    def epsilon(v):
        return ufl.sym(ufl.grad(v))

    def sigma(v):
        return (2.0 * mu * epsilon(v) + lmbda * ufl.tr(epsilon(v)) * ufl.Identity(len(v)))
        # return ufl.tr(epsilon(v)) * ufl.Identity(len(v))

    def sigma_n(v):
        # NOTE: Different normals, see summary paper
        return ufl.dot(sigma(v) * n, (-n_2))

    # Mimicking the plane y=-g
    g = 0.1
    x = ufl.SpatialCoordinate(mesh)
    gap = x[mesh.geometry.dim - 1] + g
    h = ufl.Circumradius(mesh)
    q = sigma_n(u) + gamma / h * (gap - ufl.dot(u, (-n_2)))
    a = ufl.inner(sigma(du), epsilon(v)) * dx
    a += - h * theta / gamma * sigma_n(du) * sigma_n(v) * ds(1)
    a += h / gamma * 0.5 * (1 - ufl.sign(q)) * (sigma_n(du) - gamma / h * ufl.dot(du, (-n_2))) * \
        (theta * sigma_n(v) - gamma / h * ufl.dot(v, (-n_2))) * ds(1)
    # Compile UFL form
    cffi_options = ["-Ofast", "-march=native"]
    a = form(a, jit_options={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    A = create_matrix(a)

    # Normal assembly
    A.zeroEntries()
    assemble_matrix(A, a)
    A.assemble()

    # Custom assembly
    q_rule = dolfinx_contact.QuadratureRule(mesh.topology.cell_type, 2 * P
                                            + Q + 1, mesh.topology.dim - 1, basix.QuadratureType.Default)
    consts = np.array([gamma, theta])
    consts = np.hstack((consts, n_vec))
    integral_entities, num_local = dolfinx_contact.compute_active_entities(
        mesh._cpp_object, facets, IntegralType.exterior_facet)
    integral_entities = integral_entities[:num_local]
    mu_packed = dolfinx_contact.cpp.pack_coefficient_quadrature(mu._cpp_object, 0, integral_entities)
    lmbda_packed = dolfinx_contact.cpp.pack_coefficient_quadrature(lmbda._cpp_object, 0, integral_entities)
    u_packed = dolfinx_contact.cpp.pack_coefficient_quadrature(u._cpp_object, 2 * P
                                                               + Q + 1, integral_entities)
    grad_u_packed = dolfinx_contact.cpp.pack_gradient_quadrature(u._cpp_object, 2 * P
                                                                 + Q + 1, integral_entities)
    h_facets = dolfinx_contact.pack_circumradius(mesh._cpp_object, integral_entities)
    data = np.array([1], dtype=np.int32)
    offsets = np.array([0, 1], dtype=np.int32)
    surfaces = adjacencylist(data, offsets)
    search_mode = [dolfinx_contact.cpp.ContactMode.ClosestPoint]
    contact = dolfinx_contact.cpp.Contact([ft._cpp_object], surfaces, [(0, 0)],
                                          mesh._cpp_object, search_mode, quadrature_degree=2 * P + Q + 1)
    contact.create_distance_map(0)
    g_vec = contact.pack_gap_plane(0, -g)
    coeffs = np.hstack([mu_packed, lmbda_packed, h_facets, g_vec, u_packed, grad_u_packed])
    a_custom = ufl.inner(sigma(du), epsilon(v)) * dx
    a_custom = form(a_custom)
    B = create_matrix(a_custom)
    kernel = dolfinx_contact.cpp.generate_rigid_surface_kernel(
        V._cpp_object, kernel_type, q_rule)
    B.zeroEntries()

    contact.assemble_matrix(B, 0, kernel, coeffs, consts, V._cpp_object)
    assemble_matrix(B, a_custom)
    B.assemble()

    # Compare matrices, first norm, then entries
    assert np.isclose(A.norm(), B.norm())
    compare_matrices(A, B, atol=1e-8)
