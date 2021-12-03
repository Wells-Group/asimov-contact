# Copyright (C) 2021 Sarah Roggendorf
# SPDX-License-Identifier:   LGPL-3.0-or-later

import numpy as np
import pytest
import ufl
from dolfinx.fem import FunctionSpace, LinearProblem
from dolfinx.generation import UnitCubeMesh, UnitSquareMesh
from dolfinx.mesh import locate_entities_boundary
from mpi4py import MPI

import dolfinx_contact.cpp


@pytest.mark.parametrize("dim", [2, 3])
def test_circumradius(dim):
    if dim == 3:
        N = 5
        mesh = UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
    else:
        N = 10
        mesh = UnitSquareMesh(MPI.COMM_WORLD, N, N)

    mesh.topology.create_connectivity(dim - 1, dim)
    f_to_c = mesh.topology.connectivity(dim - 1, dim)

    # Find facets on boundary to integrate over
    facets = locate_entities_boundary(mesh, mesh.topology.dim - 1,
                                      lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                              np.isclose(x[0], 1.0)))
    sorted = np.argsort(facets)
    facets = facets[sorted]
    h1 = ufl.Circumradius(mesh)
    V = FunctionSpace(mesh, ("DG", 0))
    v = ufl.TestFunction(V)
    u = ufl.TrialFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    a = u * v * dx
    L = h1 * v * dx
    problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    h2 = np.zeros(facets.size)
    for i, facet in enumerate(facets):
        cell = f_to_c.links(facet)[0]
        h2[i] = uh.vector[cell]

    h = dolfinx_contact.pack_circumradius_facet(mesh, facets)
    assert np.allclose(h, h2)
