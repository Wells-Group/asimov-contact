# Copyright (C) 2021 Sarah Roggendorf and Jørgen S. Dokken
#
# SPDX-License-Identifier:   MIT

import numpy as np
import pytest
import ufl
from dolfinx.fem import Function, FunctionSpace, LinearProblem
from dolfinx.mesh import create_unit_cube, create_unit_square, locate_entities_boundary
from mpi4py import MPI

import dolfinx_contact.cpp


@pytest.mark.parametrize("dim", [2, 3])
def test_circumradius(dim):
    if dim == 3:
        N = 5
        mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N)
    else:
        N = 25
        mesh = create_unit_square(MPI.COMM_WORLD, N, N)

    # Perturb geometry to get spatially varying circumradius
    V = FunctionSpace(mesh, ufl.VectorElement("CG", mesh.ufl_cell(), 1))
    u = Function(V)
    if dim == 3:
        u.interpolate(lambda x: (0.1*(x[1]>0.5), 0.1*np.sin(2*np.pi*x[2]), np.zeros(x.shape[1])))
    else:
        u.interpolate(lambda x: (0.1*np.cos(x[1])*(x[0]>0.2), 0.1*x[1]+0.1*np.sin(2*np.pi*x[0])))
    mesh.geometry.x[:,:mesh.geometry.dim] += u.compute_point_values()
    

    mesh.topology.create_connectivity(dim - 1, dim)
    f_to_c = mesh.topology.connectivity(dim - 1, dim)

    # Find facets on boundary to integrate over

    facets  = locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True, dtype=bool))

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
    h = dolfinx_contact.pack_circumradius_facet(mesh, facets).reshape(-1)
    assert np.allclose(h, h2)
