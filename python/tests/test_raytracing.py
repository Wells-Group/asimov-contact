# Copyright (C) 2022 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT
#
# This test check that the ray-tracing routines give the same result as the closest point projection
# when a solution is found (It is not guaranteed that a ray hits the mesh on all processes in parallel)

import dolfinx_contact
from mpi4py import MPI
import dolfinx
import numpy as np
import pytest
import ufl


@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.hexahedron, dolfinx.mesh.CellType.tetrahedron])
def test_raytracing_3D(cell_type):
    origin = [0.51, 0.33, -1]
    normal = [0, 0, 1]

    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 15, 15, 15, cell_type)
    tdim = mesh.topology.dim
    facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[2], 0))

    integral_pairs = dolfinx_contact.cpp.compute_active_entities(mesh, facets, dolfinx.fem.IntegralType.exterior_facet)

    status, cell_idx, points = dolfinx_contact.cpp.raytracing(
        mesh, origin, normal, integral_pairs, 10, 1e-6)

    if status > 0:
        # Create structures needed for closest point projections
        boundary_cells = dolfinx.mesh.compute_incident_entities(mesh, facets, tdim - 1, tdim)
        bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, tdim, boundary_cells)
        midpoint_tree = dolfinx.cpp.geometry.create_midpoint_tree(mesh, tdim, boundary_cells)

        # Find closest cell using closest point projection
        closest_cell = dolfinx.geometry.compute_closest_entity(
            bb_tree, midpoint_tree, mesh, np.reshape(origin, (1, 3)))[0]
        assert integral_pairs[cell_idx][0] == closest_cell

        # Compute actual distance between cell and point using GJK
        cell_dofs = mesh.geometry.dofmap.links(closest_cell)
        cell_geometry = np.empty((len(cell_dofs), 3), dtype=np.float64)
        for i, dof in enumerate(cell_dofs):
            cell_geometry[i, :] = mesh.geometry.x[dof, :]
        distance = dolfinx.geometry.compute_distance_gjk(cell_geometry, origin)
        assert np.allclose(points[0], origin + distance)


@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.hexahedron, dolfinx.mesh.CellType.tetrahedron])
def test_raytracing_3D_corner(cell_type):
    origin = [1.5, 1.5, 1.5]
    normal = [1, 1, 1]

    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 13, 11, 12, cell_type)
    tdim = mesh.topology.dim
    facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[2], 1))

    integral_pairs = dolfinx_contact.cpp.compute_active_entities(mesh, facets, dolfinx.fem.IntegralType.exterior_facet)
    status, cell_idx, points = dolfinx_contact.cpp.raytracing(
        mesh, origin, normal, integral_pairs, 10, 1e-6)
    if status > 0:
        # Create structures needed for closest point projections
        boundary_cells = dolfinx.mesh.compute_incident_entities(mesh, facets, tdim - 1, tdim)
        bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, tdim, boundary_cells)
        midpoint_tree = dolfinx.cpp.geometry.create_midpoint_tree(mesh, tdim, boundary_cells)

        # Find closest cell using closest point projection
        closest_cell = dolfinx.geometry.compute_closest_entity(
            bb_tree, midpoint_tree, mesh, np.reshape(origin, (1, 3)))[0]

        # Compute actual distance between cell and point using GJK
        cell_dofs = mesh.geometry.dofmap.links(closest_cell)
        cell_geometry = np.empty((len(cell_dofs), 3), dtype=np.float64)
        for i, dof in enumerate(cell_dofs):
            cell_geometry[i, :] = mesh.geometry.x[dof, :]
        distance = dolfinx.geometry.compute_distance_gjk(cell_geometry, origin)
        assert np.allclose(points[0], origin + distance)


@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral])
def test_raytracing_2D(cell_type):
    origin = [0.273, -0.5]
    normal = [0, 1]

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 15, 15, cell_type)
    tdim = mesh.topology.dim
    facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], 0))

    integral_pairs = dolfinx_contact.cpp.compute_active_entities(mesh, facets, dolfinx.fem.IntegralType.exterior_facet)

    status, cell_idx, points = dolfinx_contact.cpp.raytracing(
        mesh, origin, normal, integral_pairs, 10, 1e-6)

    if status > 0:
        # Create structures needed for closest point projections
        boundary_cells = dolfinx.mesh.compute_incident_entities(mesh, facets, tdim - 1, tdim)
        bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, tdim, boundary_cells)
        midpoint_tree = dolfinx.cpp.geometry.create_midpoint_tree(mesh, tdim, boundary_cells)
        op = np.array([origin[0], origin[1], 0])
        # Find closest cell using closest point projection
        closest_cell = dolfinx.geometry.compute_closest_entity(
            bb_tree, midpoint_tree, mesh, op)[0]
        assert integral_pairs[cell_idx][0] == closest_cell

        # Compute actual distance between cell and point using GJK
        cell_dofs = mesh.geometry.dofmap.links(closest_cell)
        cell_geometry = np.empty((len(cell_dofs), 3), dtype=np.float64)
        for i, dof in enumerate(cell_dofs):
            cell_geometry[i, :] = mesh.geometry.x[dof, :]
        distance = dolfinx.geometry.compute_distance_gjk(cell_geometry, origin)
        assert np.allclose(points[0], origin + distance[:2])


@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral])
def test_raytracing_2D_corner(cell_type):
    origin = [1.5, 1.5]
    normal = [-1, 1]

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 13, 11, cell_type)
    tdim = mesh.topology.dim
    facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], 1))

    integral_pairs = dolfinx_contact.cpp.compute_active_entities(mesh, facets, dolfinx.fem.IntegralType.exterior_facet)
    status, cell_idx, points = dolfinx_contact.cpp.raytracing(
        mesh, origin, normal, integral_pairs, 10, 1e-6)
    if status > 0:
        # Create structures needed for closest point projections
        boundary_cells = dolfinx.mesh.compute_incident_entities(mesh, facets, tdim - 1, tdim)
        bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, tdim, boundary_cells)
        midpoint_tree = dolfinx.cpp.geometry.create_midpoint_tree(mesh, tdim, boundary_cells)

        # Find closest cell using closest point projection
        op = np.array([origin[0], origin[1], 0])
        closest_cell = dolfinx.geometry.compute_closest_entity(
            bb_tree, midpoint_tree, mesh, op)[0]

        # Compute actual distance between cell and point using GJK
        cell_dofs = mesh.geometry.dofmap.links(closest_cell)
        cell_geometry = np.empty((len(cell_dofs), 3), dtype=np.float64)
        for i, dof in enumerate(cell_dofs):
            cell_geometry[i, :] = mesh.geometry.x[dof, :]
        distance = dolfinx.geometry.compute_distance_gjk(cell_geometry, origin)
        assert np.allclose(points[0], origin + distance[:2])


def test_raytracing_manifold():
    cell_type = dolfinx.mesh.CellType.quadrilateral
    x = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.5], [1, 1, 0.5]])
    cells = np.array([[0, 1, 2, 3]], dtype=np.int32)

    cell = ufl.Cell(cell_type.name, geometric_dimension=2)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)

    origin = [1.5, 2, 3.5]
    normal = [0.5, 1, 1.5]

    tdim = mesh.topology.dim
    facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], 1))

    integral_pairs = dolfinx_contact.cpp.compute_active_entities(mesh, facets, dolfinx.fem.IntegralType.exterior_facet)
    status, cell_idx, points = dolfinx_contact.cpp.raytracing(
        mesh, origin, normal, integral_pairs, 10, 1e-6)
    print(facets, points)
    if status > 0:
        # Create structures needed for closest point projections
        boundary_cells = dolfinx.mesh.compute_incident_entities(mesh, facets, tdim - 1, tdim)
        bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, tdim, boundary_cells)
        midpoint_tree = dolfinx.cpp.geometry.create_midpoint_tree(mesh, tdim, boundary_cells)

        # Find closest cell using closest point projection
        op = np.array([origin[0], origin[1], 0])
        closest_cell = dolfinx.geometry.compute_closest_entity(
            bb_tree, midpoint_tree, mesh, op)[0]

        # Compute actual distance between cell and point using GJK
        cell_dofs = mesh.geometry.dofmap.links(closest_cell)
        cell_geometry = np.empty((len(cell_dofs), 3), dtype=np.float64)
        for i, dof in enumerate(cell_dofs):
            cell_geometry[i, :] = mesh.geometry.x[dof, :]
        distance = dolfinx.geometry.compute_distance_gjk(cell_geometry, origin)
        assert np.allclose(points[0], origin + distance[:2])
