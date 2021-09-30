# Copyright (C) 2021 Sarah Roggendorf
# SPDX-License-Identifier:   LGPL-3.0-or-later

import dolfinx
import dolfinx_contact.cpp
import numpy as np
from mpi4py import MPI
import pytest
import ufl


@pytest.mark.parametrize("dim", [2, 3])
# Caution this test is very slow. Use small N!!
def test_circumradius(dim):
    if dim == 3:
        N = 5
        mesh = dolfinx.UnitCubeMesh(MPI.COMM_WORLD, N, N, N)
    else:
        N = 10
        mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, N, N)

    # Find facets on boundary to integrate over
    facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim - 1,
                                                   lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                                           np.isclose(x[0], 1.0)))
    values = np.arange(len(facets), dtype=np.int32)
    ft = dolfinx.MeshTags(mesh, mesh.topology.dim - 1, facets, values)
    h1 = ufl.Circumradius(mesh)
    V = dolfinx.FunctionSpace(mesh, ("DG", 0))
    v = ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    h2 = []
    for i, facet in enumerate(facets):
        L = h1 * v * ds(i)

        # Compile UFL form
        L = dolfinx.fem.Form(L)
        b = dolfinx.fem.create_vector(L)

        # Normal assembly
        b.zeroEntries()
        dolfinx.fem.assemble_vector(b, L)
        b.assemble()
        L2 = v * ds(i)

        # Compile UFL form
        L2 = dolfinx.fem.Form(L2)
        b2 = dolfinx.fem.create_vector(L2)

        # Normal assembly
        b2.zeroEntries()
        dolfinx.fem.assemble_vector(b2, L2)
        b2.assemble()
        h2.append(np.max(b.array) / np.max(b2.array))

    h = dolfinx_contact.cpp.pack_circumradius_facet(mesh, facets)
    assert np.allclose(h, h2)
