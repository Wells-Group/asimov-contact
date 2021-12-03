# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:   LGPL-3.0-or-later

import basix
import dolfinx_contact
import dolfinx_contact.cpp
import dolfinx_cuas
import dolfinx.mesh
import dolfinx_cuas.cpp
import numpy as np
import pytest
import ufl
from dolfinx.fem import (Function, FunctionSpace, IntegralType,
                         VectorFunctionSpace, assemble_matrix, assemble_vector,
                         create_matrix, create_vector, Form)
from dolfinx.generation import UnitCubeMesh, UnitSquareMesh
from dolfinx.mesh import locate_entities_boundary, MeshTags
from mpi4py import MPI

kt = dolfinx_contact.cpp.Kernel
compare_matrices = dolfinx_cuas.utils.compare_matrices


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
    mesh = UnitSquareMesh(MPI.COMM_WORLD, N, N) if dim == 2 else UnitCubeMesh(MPI.COMM_WORLD, N, N, N)

    # Find facets on boundary to integrate over
    facets = locate_entities_boundary(mesh, mesh.topology.dim - 1,
                                      lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                              np.isclose(x[0], 1.0)))
    values = np.ones(len(facets), dtype=np.int32)
    ft = MeshTags(mesh, mesh.topology.dim - 1, facets, values)

    # Define variational form
    V = VectorFunctionSpace(mesh, ("CG", P))

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

    V2 = FunctionSpace(mesh, ("DG", Q))
    lmbda = Function(V2)
    lmbda.interpolate(lmbda_func)
    mu = Function(V2)
    mu.interpolate(mu_func)

    # Nitsche parameters
    theta = 1.0
    gamma = 20

    n_vec = np.zeros(mesh.geometry.dim)
    n_vec[mesh.geometry.dim - 1] = -1
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
        return ufl.dot(sigma(v) * n, n_2)

    # Mimicking the plane y=-g
    g = 0.1
    x = ufl.SpatialCoordinate(mesh)
    gap = x[mesh.geometry.dim - 1] + g
    L = ufl.inner(sigma(u), epsilon(v)) * dx
    L += - h * theta / gamma * sigma_n(u) * sigma_n(v) * ds(1)
    L += h / gamma * R_minus(sigma_n(u) + (gamma / h) * (gap - ufl.dot(u, n_2))) * \
        (theta * sigma_n(v) - (gamma / h) * ufl.dot(v, n_2)) * ds(1)
    # Compile UFL form
    cffi_options = ["-O2", "-march=native"]
    L = Form(L, jit_parameters={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    b = create_vector(L)

    # Normal assembly
    b.zeroEntries()
    assemble_vector(b, L)
    b.assemble()

    # Custom assembly
    # num_local_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    consts = np.array([gamma, theta])
    consts = np.hstack((consts, n_vec))
    integral_entities = dolfinx_cuas.cpp.compute_active_entities(mesh, ft.indices, IntegralType.exterior_facet)
    coeffs = dolfinx_cuas.cpp.pack_coefficients([u._cpp_object, mu._cpp_object, lmbda._cpp_object], integral_entities)
    h_facets = dolfinx_contact.cpp.pack_circumradius_facet(mesh, facets)
    contact = dolfinx_contact.cpp.Contact(ft, 1, 1, V._cpp_object)
    contact.set_quadrature_degree(2 * P + Q + 1)
    g_vec = contact.pack_gap_plane(0, -g)
    # FIXME: assuming all facets are the same type
    q_rule = dolfinx_cuas.cpp.QuadratureRule(mesh.topology.cell_type, 2 * P
                                             + Q + 1, mesh.topology.dim - 1, basix.QuadratureType.Default)
    coeffs = np.hstack([coeffs, h_facets, g_vec])

    L_cuas = ufl.inner(sigma(u), epsilon(v)) * dx
    L_cuas = Form(L_cuas)
    b2 = create_vector(L_cuas)
    kernel = dolfinx_contact.cpp.generate_contact_kernel(V._cpp_object, kernel_type, q_rule,
                                                         [u._cpp_object, mu._cpp_object, lmbda._cpp_object])

    b2.zeroEntries()
    dolfinx_cuas.assemble_vector(b2, V, ft.indices, kernel, coeffs, consts, IntegralType.exterior_facet)
    assemble_vector(b2, L_cuas)
    b2.assemble()
    assert np.allclose(b.array, b2.array)


@pytest.mark.parametrize("kernel_type", [kt.Jac])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("P", [1, 2, 3, 4])
@pytest.mark.parametrize("Q", [0, 1, 2])
def test_matrix_surface_kernel(dim, kernel_type, P, Q):
    N = 20 if dim == 2 else 5
    mesh = UnitSquareMesh(MPI.COMM_WORLD, N, N) if dim == 2 else UnitCubeMesh(MPI.COMM_WORLD, N, N, N)

    # Find facets on boundary to integrate over
    facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1,
                                                   lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                           np.isclose(x[0], 1.0)))
    values = np.ones(len(facets), dtype=np.int32)
    ft = MeshTags(mesh, mesh.topology.dim - 1, facets, values)

    # Define variational form
    V = VectorFunctionSpace(mesh, ("CG", P))
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
    V2 = FunctionSpace(mesh, ("DG", Q))
    lmbda = Function(V2)
    lmbda.interpolate(lmbda_func)
    mu = Function(V2)
    mu.interpolate(mu_func)

    # Nitsche parameters
    theta = 1.0
    gamma = 20

    n_vec = np.zeros(mesh.geometry.dim)
    n_vec[mesh.geometry.dim - 1] = -1
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
        return ufl.dot(sigma(v) * n, n_2)

    # Mimicking the plane y=-g
    g = 0.1
    x = ufl.SpatialCoordinate(mesh)
    gap = x[mesh.geometry.dim - 1] + g
    h = ufl.Circumradius(mesh)
    q = sigma_n(u) + gamma / h * (gap - ufl.dot(u, n_2))
    a = ufl.inner(sigma(du), epsilon(v)) * dx
    a += - h * theta / gamma * sigma_n(du) * sigma_n(v) * ds(1)
    a += h / gamma * 0.5 * (1 - ufl.sign(q)) * (sigma_n(du) - gamma / h * ufl.dot(du, n_2)) * \
        (theta * sigma_n(v) - gamma / h * ufl.dot(v, n_2)) * ds(1)
    # Compile UFL form
    cffi_options = ["-O2", "-march=native"]
    a = Form(a, jit_parameters={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
    A = create_matrix(a)

    # Normal assembly
    A.zeroEntries()
    assemble_matrix(A, a)
    A.assemble()

    # Custom assembly
    q_rule = dolfinx_cuas.cpp.QuadratureRule(mesh.topology.cell_type, 2 * P
                                             + Q + 1, mesh.topology.dim - 1, basix.QuadratureType.Default)
    consts = np.array([gamma, theta])
    consts = np.hstack((consts, n_vec))
    integral_entities = dolfinx_cuas.cpp.compute_active_entities(mesh, facets, IntegralType.exterior_facet)
    coeffs = dolfinx_cuas.cpp.pack_coefficients([u._cpp_object, mu._cpp_object, lmbda._cpp_object], integral_entities)
    h_facets = dolfinx_contact.cpp.pack_circumradius_facet(mesh, facets)
    contact = dolfinx_contact.cpp.Contact(ft, 1, 1, V._cpp_object)
    contact.set_quadrature_degree(2 * P + Q + 1)
    g_vec = contact.pack_gap_plane(0, -g)
    coeffs = np.hstack([coeffs, h_facets, g_vec])
    a_cuas = ufl.inner(sigma(du), epsilon(v)) * dx
    a_cuas = Form(a_cuas)
    B = create_matrix(a_cuas)
    kernel = dolfinx_contact.cpp.generate_contact_kernel(
        V._cpp_object, kernel_type, q_rule, [u._cpp_object, mu._cpp_object, lmbda._cpp_object])
    B.zeroEntries()
    dolfinx_cuas.assemble_matrix(B, V, ft.indices, kernel, coeffs, consts, IntegralType.exterior_facet)
    assemble_matrix(B, a_cuas)
    B.assemble()

    # Compare matrices, first norm, then entries
    assert np.isclose(A.norm(), B.norm())
    compare_matrices(A, B, atol=1e-8)
