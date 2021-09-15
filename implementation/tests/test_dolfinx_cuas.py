# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from dolfinx_contact.helpers import (epsilon, lame_parameters, rigid_motions_nullspace, sigma_func, R_minus)

import dolfinx
import dolfinx.io
import numpy as np
import ufl
import pytest
from mpi4py import MPI


@pytest.mark.parametrize("theta", [0, 1, -1])
@pytest.mark.parametrize("gamma", [10])  # , 100, 1000])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("disp", [0.0])  # , 0.8, 2.0])
@pytest.mark.parametrize("gap", [0.0])  # , 0.02, -0.01])
def test_contact_kernel(theta, gamma, dim, disp, gap):
    # Problem parameters
    num_refs = 0
    top_value = 1
    bottom_value = 2
    E = 1e3
    nu = 0.1
    g = gap

    # Load mesh and create identifier functions for the top (Displacement condition)
    # and the bottom (contact condition)
    if dim == 3:
        fname = "sphere"
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
        #mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, 10, 10, 20)

        # def top(x):
        #     return x[2] > 0.99

        # def bottom(x):
        #     return x[2] < 0.5
        def top(x):
            return x[2] > 0.9

        def bottom(x):
            return x[2] < 0.15

    else:
        fname = "disk"
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh(name="Grid")
        # mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 30, 30)

        # def top(x):
        #     return x[1] > 0.99

        def top(x):
            return x[1] > 0.5

        def bottom(x):
            return x[1] < 0.2

    rank = MPI.COMM_WORLD.rank
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
        facet_marker = dolfinx.MeshTags(mesh, tdim - 1, indices, values)

        # Dirichlet boundary conditions
        V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))
        u = dolfinx.Function(V)
        v = ufl.TestFunction(V)
        du = ufl.TrialFunction(V)

        def _u_D(x):
            values = np.zeros((mesh.geometry.dim, x.shape[1]))
            values[mesh.geometry.dim - 1] = disp
            return values
        u_D = dolfinx.Function(V)
        u_D.interpolate(_u_D)
        u_D.name = "u_D"
        u_D.x.scatter_forward()
        tdim = mesh.topology.dim
        dirichlet_dofs = dolfinx.fem.locate_dofs_topological(
            V, tdim - 1, facet_marker.indices[facet_marker.values == top_value])
        bc = dolfinx.DirichletBC(u_D, dirichlet_dofs)
        bcs = [bc]

        # Initial condition
        def _u_initial(x):
            values = np.zeros((mesh.geometry.dim, x.shape[1]))
            values[-1] = -0.01 - g
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
            return -ufl.dot(sigma(v) * n, n_2)

        # Mimicking the plane y=-g
        x = ufl.SpatialCoordinate(mesh)
        gap = x[mesh.geometry.dim - 1] + g
        g_vec = [i for i in range(mesh.geometry.dim)]
        g_vec[mesh.geometry.dim - 1] = gap

        u = dolfinx.Function(V)
        v = ufl.TestFunction(V)
        # metadata = {"quadrature_degree": 5}
        dx = ufl.Measure("dx", domain=mesh)
        ds = ufl.Measure("ds", domain=mesh,  # metadata=metadata,
                         subdomain_data=facet_marker)
        a = ufl.inner(sigma(u), epsilon(v)) * dx
        L = ufl.inner(dolfinx.Constant(mesh, [0, ] * mesh.geometry.dim), v) * dx

        # Derivation of one sided Nitsche with gap function
        F = a - theta / gammah * sigma_n(u) * sigma_n(v) * ds(bottom_value) - L
        F += 1 / gammah * R_minus(sigma_n(u) + gammah * (gap + ufl.dot(u, n_2))) * \
            (theta * sigma_n(v) + gammah * ufl.dot(v, n_2)) * ds(bottom_value)

        u.interpolate(_u_initial)
        L = dolfinx.fem.Form(F, jit_parameters={"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]})
        b = dolfinx.fem.create_vector(L)

        # Normal assembly
        b.zeroEntries()
        dolfinx.fem.assemble_vector(b, L)
        # # Apply boundary conditions to the rhs
        # dolfinx.fem.apply_lifting(b, [self._a], [self.bcs])
        # self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        # fem.set_bc(self._b, self.bcs)
        # b.assemble()
