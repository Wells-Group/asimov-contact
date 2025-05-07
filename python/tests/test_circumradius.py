# Copyright (C) 2021 Sarah Roggendorf and Jørgen S. Dokken
#
# SPDX-License-Identifier:   MIT

from mpi4py import MPI

import dolfinx_contact.cpp
import numpy as np
import pytest
from dolfinx.fem import Function, IntegralType, functionspace
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_cube, create_unit_square, locate_entities_boundary
from ufl import Circumradius, Measure, TestFunction, TrialFunction


@pytest.mark.parametrize("dim", [2, 3])
def test_circumradius(dim):
    if dim == 3:
        N = 5
        mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N)
    else:
        N = 25
        mesh = create_unit_square(MPI.COMM_WORLD, N, N)

    # Perturb geometry to get spatially varying circumradius
    V = functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
    u = Function(V)
    if dim == 3:
        u.interpolate(
            lambda x: (0.1 * (x[1] > 0.5), 0.1 * np.sin(2 * np.pi * x[2]), np.zeros(x.shape[1]))
        )
    else:
        u.interpolate(
            lambda x: (
                0.1 * np.cos(x[1]) * (x[0] > 0.2),
                0.1 * x[1] + 0.1 * np.sin(2 * np.pi * x[0]),
            )
        )
    dolfinx_contact.update_geometry(u._cpp_object, mesh._cpp_object)

    mesh.topology.create_connectivity(dim - 1, dim)
    f_to_c = mesh.topology.connectivity(dim - 1, dim)

    # Find facets on boundary to integrate over

    facets = locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True, dtype=bool)
    )

    sorted = np.argsort(facets)
    facets = facets[sorted]
    h1 = Circumradius(mesh)
    V = functionspace(mesh, ("DG", 0))
    v = TestFunction(V)
    u = TrialFunction(V)
    dx = Measure("dx", domain=mesh)
    a = u * v * dx
    L = h1 * v * dx
    problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    h2 = np.zeros(facets.size)
    cells = []
    for i, facet in enumerate(facets):
        cell = f_to_c.links(facet)[0]
        h2[i] = uh.x.array[cell]
        cells.append(cell)

    active_facets, num_local = dolfinx_contact.cpp.compute_active_entities(
        mesh._cpp_object, facets, IntegralType.exterior_facet
    )
    active_facets = active_facets[:num_local, :]
    h = dolfinx_contact.pack_circumradius(mesh._cpp_object, active_facets).reshape(-1)

    # sort h2, compute_active_entities sorts by cells
    indices = np.argsort(cells)
    eps = np.finfo(mesh.geometry.x.dtype).eps
    np.testing.assert_allclose(h, h2[indices], atol=eps)
