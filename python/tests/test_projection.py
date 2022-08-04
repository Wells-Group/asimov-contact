# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:   MIT
#
# This test verifies the closest point projection. If we consider the closest point projection Pi(x)
# mapping a point x on surface 0 to a point Pi(x) on surface 1, then Pi(x) - x should be orthogonal
# to surface 1 in Pi(x) and point inwards. The normalised version (Pi(x) - x)||Pi(x)-x|| should therefore
# be the same as the outward unit normal in the point Pi(x) with the opposite sign.

import dolfinx.fem as _fem
import numpy as np
import pytest
from dolfinx.graph import create_adjacencylist
from dolfinx.io import XDMFFile
from dolfinx.mesh import meshtags, locate_entities_boundary
from mpi4py import MPI
import os
import dolfinx_contact
import dolfinx_contact.cpp
from dolfinx_contact.meshing import convert_mesh, create_box_mesh_2D, create_box_mesh_3D

os.system("mkdir -p meshes")


@pytest.mark.parametrize("q_deg", range(1, 4))
@pytest.mark.parametrize("surf", [0, 1])
@pytest.mark.parametrize("dim", [2, 3])
def test_projection(q_deg, surf, dim):

    # Create mesh
    if dim == 2:
        fname = "meshes/box_2D"
        create_box_mesh_2D(filename=f"{fname}.msh", res=1.0)

    else:
        fname = "meshes/box_3D"
        create_box_mesh_3D(filename=f"{fname}.msh", res=1.0)

    convert_mesh(fname, fname, gdim=dim)

    # Read in mesh
    with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh()

    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    mesh.topology.create_connectivity(tdim - 1, 0)
    mesh.topology.create_connectivity(tdim - 1, tdim)

    # Surface parameters see contact_meshes.py
    L = 0.5
    delta = 0.1
    disp = -0.6
    H = 0.5

    # Define surfaces
    def surface_0(x):
        if dim == 2:
            return np.logical_and(np.isclose(x[1], delta * (x[0] + delta) / L), x[1] < delta + 1e-5)
        else:
            return np.isclose(x[2], 0)

    def surface_1(x):
        return np.isclose(x[dim - 1], disp + H)

    # define restriced range for x coordinate to ensure closest point is on interior of opposite surface
    def x_range(x):
        return np.logical_and(x[0] > delta, x[0] < L - delta)

    surface_0_val = 1
    surface_1_val = 2

    # Create meshtags for surfaces
    # restrict range of x coordinate for origin surface
    if surf == 0:
        facets_0 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.logical_and(surface_0(x), x_range(x)))
        facets_1 = locate_entities_boundary(mesh, tdim - 1, surface_1)
    else:
        facets_0 = locate_entities_boundary(mesh, tdim - 1, surface_0)
        facets_1 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.logical_and(surface_1(x), x_range(x)))

    values_0 = np.full(len(facets_0), surface_0_val, dtype=np.int32)
    values_1 = np.full(len(facets_1), surface_1_val, dtype=np.int32)
    indices = np.concatenate([facets_0, facets_1])
    values = np.hstack([values_0, values_1])
    sorted_ind = np.argsort(indices)
    facet_marker = meshtags(mesh, tdim - 1, indices[sorted_ind], values[sorted_ind])

    # Functions space
    V = _fem.VectorFunctionSpace(mesh, ("CG", 1))

    # Create contact class, gap function and normals
    data = np.array([surface_0_val, surface_1_val], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    surfaces = create_adjacencylist(data, offsets)
    contact = dolfinx_contact.cpp.Contact([facet_marker], surfaces, [(0, 1), (1, 0)],
                                          V._cpp_object, quadrature_degree=q_deg)
    contact.create_distance_map(surf)
    gap = contact.pack_gap(surf)
    normals = contact.pack_ny(surf)

    # Compute dot product and normalise
    n_dot = np.zeros((gap.shape[0], gap.shape[1] // gdim))
    for facet in range(gap.shape[0]):
        for q in range(gap.shape[1] // gdim):
            g = gap[facet, q * gdim:(q + 1) * gdim]
            n = -normals[facet, q * gdim:(q + 1) * gdim]
            n_norm = np.linalg.norm(n)
            g_norm = np.linalg.norm(g)
            for i in range(gdim):
                n_dot[facet, q] += g[i] * n[i] / (n_norm * g_norm)

    # Test if angle between -normal and gap function is less than 6.5 degrees
    # Is better accuracy needed?
    assert np.allclose(n_dot, np.ones(n_dot.shape))
