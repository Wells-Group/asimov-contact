# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from dolfinx_contact.helpers import (epsilon, lame_parameters, sigma_func, R_minus)

import dolfinx
import dolfinx.io
import dolfinx_cuas
import dolfinx_cuas.cpp
import dolfinx_contact
import dolfinx_contact.cpp
import numpy as np
import ufl
import pytest
from mpi4py import MPI

kt = dolfinx_contact.cpp.Kernel
it = dolfinx.cpp.fem.IntegralType
compare_matrices = dolfinx_cuas.utils.compare_matrices


# This tests compares custom assembly and ufl based assembly
# of the rhs and jacobi matrix for contact
# with a rigid surface for a given initial value
@pytest.mark.parametrize("theta", [1, -1, 0])
@pytest.mark.parametrize("gamma", [10, 1000])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("gap", [0.02, -0.01])
def test_contact_kernel(theta, gamma, dim, gap):
    # Problem parameters
    num_refs = 1
    top_value = 1
    bottom_value = 2
    E = 1e3
    nu = 0.1
    g = gap
    q_deg = 2

    # Load mesh and create identifier functions for the top (Displacement condition)
    # and the bottom (contact condition)
    if dim == 3:
        fname = "sphere"
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")

        def top(x):
            return x[2] > 0.9

        def bottom(x):
            return x[2] < 0.15

    else:
        fname = "disk"
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")

        def top(x):
            return x[1] > 0.5

        def bottom(x):
            return x[1] < 0.2

    refs = np.arange(0, num_refs)
    for i in refs:
        if i > 0:
            # Refine mesh
            mesh.topology.create_entities(mesh.topology.dim - 2)
            mesh = dolfinx.mesh.refine(mesh)

        # Create meshtag for top and bottom markers
        tdim = mesh.topology.dim
        top_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, top)
        bottom_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, tdim - 1, bottom)
        top_values = np.full(len(top_facets), top_value, dtype=np.int32)
        bottom_values = np.full(
            len(bottom_facets), bottom_value, dtype=np.int32)
        indices = np.concatenate([top_facets, bottom_facets])
        values = np.hstack([top_values, bottom_values])
        sorted_facets = np.argsort(indices)
        facet_marker = dolfinx.MeshTags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])
        bottom_facets = np.sort(bottom_facets)

        V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))
        u = dolfinx.Function(V)
        v = ufl.TestFunction(V)
        du = ufl.TrialFunction(V)

        # Initial condition
        def _u_initial(x):
            values = np.zeros((mesh.geometry.dim, x.shape[1]))
            values[-1] = -0.04
            return values

        # Assemble rhs dolfinx
        h = ufl.Circumradius(mesh)
        gammah = gamma * E / h
        n_vec = np.zeros(mesh.geometry.dim)
        n_vec[mesh.geometry.dim - 1] = 1
        n_2 = ufl.as_vector(n_vec)  # Normal of plane (projection onto other body)
        n = ufl.FacetNormal(mesh)
        mu_func, lambda_func = lame_parameters(False)
        mu = mu_func(E, nu)
        lmbda = lambda_func(E, nu)
        sigma = sigma_func(mu, lmbda)

        def sigma_n(v):
            # NOTE: Different normals, see summary paper
            return ufl.dot(sigma(v) * n, n_2)

        # Mimicking the plane y=-g
        x = ufl.SpatialCoordinate(mesh)
        gap = x[mesh.geometry.dim - 1] + g
        g_vec = [i for i in range(mesh.geometry.dim)]
        g_vec[mesh.geometry.dim - 1] = gap

        u = dolfinx.Function(V)
        v = ufl.TestFunction(V)
        u.interpolate(_u_initial)
        metadata = {}  # {"quadrature_degree": q_deg}
        dx = ufl.Measure("dx", domain=mesh, metadata=metadata)
        ds = ufl.Measure("ds", domain=mesh, metadata=metadata,
                         subdomain_data=facet_marker)
        a = ufl.inner(sigma(u), epsilon(v)) * dx
        # L = ufl.inner(dolfinx.Constant(mesh, [0, ] * mesh.geometry.dim), v) * dx

        # Derivation of one sided Nitsche with gap function
        F = a - theta / gammah * sigma_n(u) * sigma_n(v) * ds(bottom_value)
        F += 1 / gammah * R_minus(sigma_n(u) + gammah * (gap + ufl.dot(u, n_2))) * \
            (theta * sigma_n(v) + gammah * ufl.dot(v, n_2)) * ds(bottom_value)

        u.interpolate(_u_initial)
        L = dolfinx.fem.Form(F)
        b = dolfinx.fem.create_vector(L)

        # Normal assembly
        b.zeroEntries()
        dolfinx.fem.assemble_vector(b, L)
        b.assemble()

        q = sigma_n(u) + gammah * (gap + ufl.dot(u, n_2))
        J = ufl.inner(sigma(du), epsilon(v)) * ufl.dx - theta / gammah * sigma_n(du) * sigma_n(v) * ds(bottom_value)
        J += 1 / gammah * 0.5 * (1 - ufl.sign(q)) * (sigma_n(du) + gammah * ufl.dot(du, n_2)) * \
            (theta * sigma_n(v) + gammah * ufl.dot(v, n_2)) * ds(bottom_value)
        a = dolfinx.fem.Form(J)
        A = dolfinx.fem.create_matrix(a)

        # Normal assembly
        A.zeroEntries()
        dolfinx.fem.assemble_matrix(A, a)
        A.assemble()

        # Custom assembly
        # FIXME: assuming all facets are the same type
        q_rule = dolfinx_cuas.cpp.QuadratureRule(mesh.topology.cell_type, q_deg, mesh.topology.dim - 1, "default")
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
        V2 = dolfinx.FunctionSpace(mesh, ("DG", 0))
        lmbda2 = dolfinx.Function(V2)
        lmbda2.interpolate(lmbda_func2)
        mu2 = dolfinx.Function(V2)
        mu2.interpolate(mu_func2)
        u.interpolate(_u_initial)
        coeffs = dolfinx_cuas.cpp.pack_coefficients([u._cpp_object, mu2._cpp_object, lmbda2._cpp_object])
        h_facets = dolfinx_contact.cpp.pack_circumradius_facet(mesh, bottom_facets)
        h_cells = dolfinx_contact.cpp.facet_to_cell_data(mesh, bottom_facets, h_facets, 1)
        contact = dolfinx_contact.cpp.Contact(facet_marker, bottom_value, top_value, V._cpp_object)
        contact.set_quadrature_degree(q_deg)
        g_vec = contact.pack_gap_plane(0, g)
        # FIXME: Assumptions that does not work for prisms
        g_vec_c = dolfinx_contact.cpp.facet_to_cell_data(mesh, bottom_facets, g_vec, dim * q_rule.weights(0).size)
        coeffs = np.hstack([coeffs, h_cells, g_vec_c])
        # RHS
        L_cuas = ufl.inner(sigma(u), epsilon(v)) * dx
        L_cuas = dolfinx.fem.Form(L_cuas)
        b2 = dolfinx.fem.create_vector(L_cuas)
        kernel = dolfinx_contact.cpp.generate_contact_kernel(V._cpp_object, kt.NitscheRigidSurfaceRhs, q_rule,
                                                             [u._cpp_object, mu2._cpp_object, lmbda2._cpp_object])
        b2.zeroEntries()
        dolfinx_cuas.assemble_vector(b2, V, bottom_facets, kernel, coeffs, consts, it.exterior_facet)
        dolfinx.fem.assemble_vector(b2, L_cuas)
        b2.assemble()
        # Jacobian
        a_cuas = ufl.inner(sigma(du), epsilon(v)) * dx
        a_cuas = dolfinx.fem.Form(a_cuas)
        B = dolfinx.fem.create_matrix(a_cuas)
        kernel = dolfinx_contact.cpp.generate_contact_kernel(
            V._cpp_object, kt.NitscheRigidSurfaceJac, q_rule, [u._cpp_object, mu2._cpp_object, lmbda2._cpp_object])
        B.zeroEntries()
        dolfinx_cuas.assemble_matrix(B, V, bottom_facets, kernel, coeffs, consts, it.exterior_facet)
        dolfinx.fem.assemble_matrix(B, a_cuas)
        B.assemble()

        # Compare matrices, first norm, then entries
        assert np.isclose(A.norm(), B.norm())
        compare_matrices(A, B, atol=1e-7)
