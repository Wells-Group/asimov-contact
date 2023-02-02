# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import basix
import dolfinx.fem
import dolfinx.io
import dolfinx.mesh
import numpy as np
import pytest
import ufl
from mpi4py import MPI
from dolfinx.graph import create_adjacencylist
import os
import dolfinx_contact
import dolfinx_contact.cpp
from dolfinx_contact.helpers import (R_minus, epsilon, lame_parameters,
                                     sigma_func, compare_matrices)
from dolfinx_contact.meshing import (convert_mesh, create_disk_mesh,
                                     create_sphere_mesh)

kt = dolfinx_contact.cpp.Kernel


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
    mesh_dir = "meshes"
    os.system(f"mkdir -p {mesh_dir}")
    if dim == 3:
        fname = f"{mesh_dir}/sphere"
        create_sphere_mesh(filename=f"{fname}.msh")
        convert_mesh(fname, fname, gdim=3)

        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh()

        def top(x):
            return x[2] > 0.9

        def bottom(x):
            return x[2] < 0.15

    else:
        fname = f"{mesh_dir}/disk"
        create_disk_mesh(filename=f"{fname}.msh")
        convert_mesh(fname, fname, gdim=2)
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh()

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
        facet_marker = dolfinx.mesh.meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])
        bottom_facets = np.sort(bottom_facets)

        V = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", 1))
        u = dolfinx.fem.Function(V)
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
            return ufl.dot(sigma(v) * n, (-n_2))

        # Mimicking the plane y=g
        x = ufl.SpatialCoordinate(mesh)
        gap = x[mesh.geometry.dim - 1] - g
        g_vec = [i for i in range(mesh.geometry.dim)]
        g_vec[mesh.geometry.dim - 1] = gap

        u.interpolate(_u_initial)
        metadata = {}  # {"quadrature_degree": q_deg}
        dx = ufl.Measure("dx", domain=mesh, metadata=metadata)
        ds = ufl.Measure("ds", domain=mesh, metadata=metadata,
                         subdomain_data=facet_marker)
        a = ufl.inner(sigma(u), epsilon(v)) * dx

        # Derivation of one sided Nitsche with gap function
        F = a - theta / gammah * sigma_n(u) * sigma_n(v) * ds(bottom_value)
        F += 1 / gammah * R_minus(sigma_n(u) + gammah * (gap - ufl.dot(u, (-n_2)))) * \
            (theta * sigma_n(v) - gammah * ufl.dot(v, (-n_2))) * ds(bottom_value)

        L = dolfinx.fem.form(F)
        b = dolfinx.fem.petsc.create_vector(L)

        # Normal assembly
        b.zeroEntries()
        dolfinx.fem.petsc.assemble_vector(b, L)
        b.assemble()

        q = sigma_n(u) + gammah * (gap - ufl.dot(u, (-n_2)))
        J = ufl.inner(sigma(du), epsilon(v)) * ufl.dx - theta / gammah * sigma_n(du) * sigma_n(v) * ds(bottom_value)
        J += 1 / gammah * 0.5 * (1 - ufl.sign(q)) * (sigma_n(du) - gammah * ufl.dot(du, (-n_2))) * \
            (theta * sigma_n(v) - gammah * ufl.dot(v, (-n_2))) * ds(bottom_value)
        a = dolfinx.fem.form(J)
        A = dolfinx.fem.petsc.create_matrix(a)

        # Normal assembly
        A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A, a)
        A.assemble()

        # Custom assembly
        # FIXME: assuming all facets are the same type
        q_rule = dolfinx_contact.QuadratureRule(mesh.topology.cell_type, q_deg,
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
        V2 = dolfinx.fem.FunctionSpace(mesh, ("DG", 0))
        lmbda2 = dolfinx.fem.Function(V2)
        lmbda2.interpolate(lmbda_func2)
        mu2 = dolfinx.fem.Function(V2)
        mu2.interpolate(mu_func2)
        u.interpolate(_u_initial)
        integral_entities, num_local = dolfinx_contact.compute_active_entities(
            mesh._cpp_object, bottom_facets, dolfinx.fem.IntegralType.exterior_facet)
        integral_entities = integral_entities[:num_local]

        mu_packed = dolfinx_contact.cpp.pack_coefficient_quadrature(mu2._cpp_object, 0, integral_entities)
        lmbda_packed = dolfinx_contact.cpp.pack_coefficient_quadrature(lmbda2._cpp_object, 0, integral_entities)
        u_packed = dolfinx_contact.cpp.pack_coefficient_quadrature(u._cpp_object, q_deg, integral_entities)
        grad_u_packed = dolfinx_contact.cpp.pack_gradient_quadrature(u._cpp_object, q_deg, integral_entities)
        h_facets = dolfinx_contact.pack_circumradius(mesh._cpp_object, integral_entities)
        data = np.array([bottom_value, top_value], dtype=np.int32)
        offsets = np.array([0, 2], dtype=np.int32)
        surfaces = create_adjacencylist(data, offsets)
        contact = dolfinx_contact.cpp.Contact([facet_marker], surfaces, [(0, 1)],
                                              V._cpp_object, quadrature_degree=q_deg)
        g_vec = contact.pack_gap_plane(0, g)
        coeffs = np.hstack([mu_packed, lmbda_packed, h_facets, g_vec, u_packed, grad_u_packed])
        # RHS
        L_custom = ufl.inner(sigma(u), epsilon(v)) * dx
        L_custom = dolfinx.fem.form(L_custom)
        b2 = dolfinx.fem.petsc.create_vector(L_custom)
        kernel = dolfinx_contact.cpp.generate_contact_kernel(V._cpp_object, kt.Rhs, q_rule)
        b2.zeroEntries()
        contact_assembler = dolfinx_contact.cpp.Contact(
            [facet_marker], surfaces, [(0, 1)], V._cpp_object, quadrature_degree=q_deg)
        contact_assembler.create_distance_map(0)

        contact_assembler.assemble_vector(b2, 0, kernel, coeffs, consts)
        dolfinx.fem.petsc.assemble_vector(b2, L_custom)
        b2.assemble()
        # Jacobian
        a_custom = ufl.inner(sigma(du), epsilon(v)) * dx
        a_custom = dolfinx.fem.form(a_custom)
        B = contact_assembler.create_matrix(a_custom)
        kernel = dolfinx_contact.cpp.generate_contact_kernel(
            V._cpp_object, kt.Jac, q_rule)
        B.zeroEntries()
        contact_assembler.assemble_matrix(B, [], 0, kernel, coeffs, consts)
        dolfinx.fem.petsc.assemble_matrix(B, a_custom)
        B.assemble()
        assert np.allclose(b.array, b2.array)
        # Compare matrices, first norm, then entries
        assert np.isclose(A.norm(), B.norm())
        compare_matrices(A, B, atol=1e-7)
