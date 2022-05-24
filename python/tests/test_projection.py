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
from dolfinx.io import XDMFFile
from dolfinx.mesh import meshtags, locate_entities_boundary
from mpi4py import MPI

import dolfinx_contact
import dolfinx_contact.cpp
from dolfinx_contact.meshing import convert_mesh, create_circle_circle_mesh, create_sphere_sphere_mesh


@pytest.mark.parametrize("q_deg", range(1, 4))
@pytest.mark.parametrize("surf", [0, 1])
@pytest.mark.parametrize("dim", [2, 3])
def test_projection(q_deg, surf, dim):

    # Create mesh
    if dim == 2:
        fname = "two_disks"
        create_circle_circle_mesh(filename=f"{fname}.msh", res=0.025)
        convert_mesh(fname, fname, "triangle", prune_z=True)
        convert_mesh(f"{fname}", f"{fname}_facets", "line", prune_z=True)
    else:
        fname = "two_spheres"
        create_sphere_sphere_mesh(filename=f"{fname}.msh")
        convert_mesh(fname, fname, "tetra")
        convert_mesh(f"{fname}", f"{fname}_facets", "triangle")

    # Read in mesh
    with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    mesh.topology.create_connectivity(tdim - 1, 0)
    mesh.topology.create_connectivity(tdim - 1, tdim)

    # Define surfaces
    def surface_0(x):
        if surf == 0:
            return np.logical_and(x[tdim - 1] < 0.3, x[tdim - 1] > 0.15)
        else:
            return np.logical_and(x[tdim - 1] < 0.5, x[tdim - 1] > 0.15)

    def surface_1(x):
        if surf == 0:
            return np.logical_and(x[tdim - 1] > -0.3, x[tdim - 1] < 0.15)
        else:
            return np.logical_and(x[tdim - 1] > -0.5, x[tdim - 1] < 0.15)

    surface_0_val = 1
    surface_1_val = 2

    # Create meshtags for surface
    facets_0 = locate_entities_boundary(mesh, tdim - 1, surface_0)
    facets_1 = locate_entities_boundary(mesh, tdim - 1, surface_1)
    values_0 = np.full(len(facets_0), surface_0_val, dtype=np.int32)
    values_1 = np.full(len(facets_1), surface_1_val, dtype=np.int32)
    indices = np.concatenate([facets_0, facets_1])
    values = np.hstack([values_0, values_1])
    sorted_ind = np.argsort(indices)
    facet_marker = meshtags(mesh, tdim - 1, indices[sorted_ind], values[sorted_ind])

    # Functions space
    V = _fem.VectorFunctionSpace(mesh, ("CG", 1))

    # Create contact class, gap function and normals
    contact = dolfinx_contact.cpp.Contact(facet_marker, [surface_0_val, surface_1_val], V._cpp_object)
    contact.set_quadrature_degree(q_deg)
    if surf == 0:
        contact.create_distance_map(surf, 1)
    else:
        contact.create_distance_map(surf, 0)
    gap = contact.pack_gap(surf)
    normals = contact.pack_ny(surf, gap)

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
    assert(np.allclose(n_dot, np.ones(n_dot.shape), atol=1 - np.cos(np.deg2rad(6.5))))
