# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT


import numpy as np
import pytest
import ufl
from dolfinx.graph import adjacencylist
from dolfinx import fem, graph
from dolfinx import mesh as msh
from mpi4py import MPI

import dolfinx_contact


@pytest.mark.skipif(MPI.COMM_WORLD.size > 1,
                    reason="This test should only be run in serial.")
def test_pack_u():
    """
    Test that evaluation of a function u on the opposite surface is correct

    """
    # Create mesh consisting of 4 triangles, where they are grouped in two
    # disconnected regions (with two cells in each region)
    points = np.array([[0, -1], [0.5, -1], [0, -2], [1, -1],
                       [0, 1], [0.3, 1], [2, 2], [1, 1]], dtype=np.float64)
    cells = np.array([[0, 1, 2], [4, 5, 6], [1, 3, 2], [5, 6, 7]], dtype=np.int64)

    cell_type = msh.CellType.triangle
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", ufl.Cell(cell_type.name), 1))
    cells = graph.adjacencylist(cells)
    part = msh.create_cell_partitioner(msh.GhostMode.none)
    mesh = msh.create_mesh(MPI.COMM_WORLD, cells, points, domain, part)

    def f(x):
        vals = np.zeros((2, x.shape[1]))
        vals[0] = 0.1 * x[0]
        vals[1] = 0.23 * x[1]
        return vals

    # Compute function that is known on each side
    V = fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
    u = fem.Function(V)
    u.interpolate(f)

    # Mark two facets on each side of gap
    mesh.topology.create_connectivity(2, 1)
    s0 = msh.locate_entities_boundary(mesh, 1, lambda x: np.isclose(x[1], 1))
    v0 = np.full(len(s0), 1, dtype=np.int32)
    s1 = msh.locate_entities_boundary(mesh, 1, lambda x: np.isclose(x[1], -1))
    v1 = np.full(len(s1), 2, dtype=np.int32)
    ss = [s0, s1]

    facets = np.hstack([s0, s1])
    values = np.hstack([v0, v1])
    arg_sort = np.argsort(facets)
    facet_marker = msh.meshtags(mesh, 1, facets[arg_sort], values[arg_sort])

    data = np.array([1, 2], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    surfaces = adjacencylist(data, offsets)

    # Compute contact class
    quadrature_degree = 2
    contact = dolfinx_contact.cpp.Contact([facet_marker._cpp_object], surfaces, [(0, 1), (1, 0)],
                                          mesh._cpp_object, quadrature_degree=quadrature_degree)
    contact.create_distance_map(0)
    contact.create_distance_map(1)

    for i in range(2):
        gap = contact.pack_gap(i)
        u_opposite = contact.pack_u_contact(i, u._cpp_object)

        gap_reshape = gap.reshape(len(ss[i]), -2, 2)

        q_points = np.zeros_like(gap_reshape)
        for j in range(len(s0)):
            q_points[j] = contact.qp_phys(i, j)
        new_points = q_points + gap_reshape

        for j in range(len(s0)):
            f_exact = f(new_points[j].T).T.reshape(-1)
            assert np.allclose(f_exact, u_opposite[j])
