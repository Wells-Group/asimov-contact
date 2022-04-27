# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:   MIT

import numpy as np
import pytest
import ufl
from dolfinx.cpp.mesh import to_type
from dolfinx.io import XDMFFile
import dolfinx.fem as _fem
from dolfinx.mesh import (CellType, create_mesh, locate_entities_boundary, meshtags)
from mpi4py import MPI

import dolfinx_contact
import dolfinx_contact.cpp


def create_functionspaces(ct, gap, delta):
    ''' This is a helper function to create the two element function spaces
        for custom assembly using quads, triangles, hexes and tetrahedra'''
    cell_type = to_type(ct)
    if cell_type == CellType.quadrilateral:
        x_1 = np.array([[0, 0], [0.8, 0], [0.1, 1.3], [0.7, 1.2]])
        x_2 = np.array([[0, 0], [0.8, 0], [-0.1, -1.2], [0.8, -1.1]])
        for point in x_2:
            point[0] += delta
            point[1] -= gap
        x_3 = np.array([x_2[0].copy() + [1.6, 0], x_2[2].copy() + [1.6, 0]])
        x = np.vstack([x_1, x_2, x_3])
        cells = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [5, 8, 7, 9]], dtype=np.int32)
    elif cell_type == CellType.triangle:
        x = np.array([[0, 0, 0], [0.8, 0, 0], [0.3, 1.3, 0.0], [
            0 + delta, -gap, 0], [0.8 + delta, -gap, 0], [0.4 + delta, -1.2 - gap, 0.0]])
        for point in x:
            point[2] = 3 * point[0] + 2 * point[1]  # plane given by z = 3x +2y
        cells = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    elif cell_type == CellType.tetrahedron:
        x = np.array([[0, 0, 0], [1.1, 0, 0], [0.3, 1.0, 0], [1, 1.2, 1.5], [
            0 + delta, 0, -gap], [1.1 + delta, 0, -gap], [0.3 + delta, 1.0, -gap], [0.8 + delta, 1.2, -1.6 - gap]])
        cells = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    elif cell_type == CellType.hexahedron:

        x_1 = np.array([[0, 0, 0], [1.1, 0, 0], [0.1, 1, 0], [1, 1.2, 0],
                        [0, 0, 1.2], [1.0, 0, 1], [0, 1, 1], [1, 1, 1]])
        x_2 = np.array([[0, 0, -1.2], [1.0, 0, -1.3], [0.1, 1, -1], [1, 1, -1],
                        [0, 0, 0], [1.1, 0, 0], [0.1, 1, 0], [1, 1.2, 0]])
        for point in x_2:
            point[0] += delta
            point[2] -= gap
        x_3 = np.array([x_2[0].copy() + [2.0, 0, 0], x_2[2].copy() + [2.0, 0, 0],
                       x_2[4].copy() + [2.0, 0, 0], x_2[6].copy() + [2.0, 0, 0]])
        x = np.vstack([x_1, x_2, x_3])
        cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15],
                          [9, 16, 10, 17, 13, 18, 15, 19]], dtype=np.int32)
    else:
        raise ValueError(f"Unsupported mesh type {ct}")

    cell = ufl.Cell(ct, geometric_dimension=x.shape[1])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
    mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
    el = ufl.VectorElement("CG", mesh.ufl_cell(), 1)
    V = _fem.FunctionSpace(mesh, el)
    with XDMFFile(mesh.comm, "test_mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
    return V


@ pytest.mark.parametrize("ct", ["quadrilateral", "triangle", "tetrahedron", "hexahedron"])
@ pytest.mark.parametrize("gap", [0.5, -0.5])
@ pytest.mark.parametrize("q_deg", [1, 2, 3])
@ pytest.mark.parametrize("delta", [0.0, -0.5])
@ pytest.mark.parametrize("surface", [0, 1])
def test_pack_test_fn(ct, gap, q_deg, delta, surface):

    # Create function space
    V = create_functionspaces(ct, gap, delta)

    # Retrieve mesh and mesh data
    mesh = V.mesh
    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    cmap = mesh.geometry.cmap
    geometry_dofmap = mesh.geometry.dofmap

    # locate facets
    facets1 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[tdim - 1], 0))
    facets2 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[tdim - 1], -gap))
    facets = [facets1, facets2]

    # create meshtags
    val0 = np.full(len(facets1), 0, dtype=np.int32)
    val1 = np.full(len(facets2), 1, dtype=np.int32)
    values = np.hstack([val0, val1])
    indices = np.concatenate([facets1, facets2])
    sorted_facets = np.argsort(indices)
    facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])

    # create contact class
    opposites = [1, 0]
    s = surface
    o = opposites[surface]
    contact = dolfinx_contact.cpp.Contact(facet_marker, [0, 1], V._cpp_object)
    contact.set_quadrature_degree(q_deg)
    contact.create_distance_map(s, o)

    # Pack gap on surface, pack test functions and jacobian on opposite surface
    gap = contact.pack_gap(s)
    test_fn = contact.pack_test_functions(s, gap, 1)
    surf_der = contact.pack_surface_derivatives(s, gap)

    # Retrieve surface facets
    s_facets = np.sort(facets[s])
    lookup = contact.facet_map(s)

    # Create facet to cell connectivity
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mesh.topology.create_connectivity(tdim, tdim - 1)
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)

    # loop over facets in surface
    for f in range(len(s_facets)):

        # Compute evaluation points
        qp_phys = contact.qp_phys(s, f)
        num_q_points = qp_phys.shape[0]
        points = np.zeros((num_q_points, 3))

        for i in range(num_q_points):
            for j in range(gdim):
                points[i, j] = qp_phys[i, j] + gap[f][i * gdim + j]

        # retrieve connected facets
        connected_facets = lookup.links(f)
        unique_facets = np.unique(np.sort(connected_facets))

        # loop over unique connected facets
        for link, facet_o in enumerate(unique_facets):

            # retrieve cell index and cell dofs for facet_o
            cell = f_to_c.links(facet_o)
            dofs = V.dofmap.cell_dofs(cell)

            # find quadrature points linked to facet_o
            q_indices = np.argwhere(connected_facets == facet_o)
            zero_ind = np.argwhere(connected_facets != facet_o)

            # retrieve cell geometry and compute pull back of physical points to reference cell
            gdofs = geometry_dofmap.links(cell)
            xg = mesh.geometry.x[gdofs]
            x_ref = cmap.pull_back(points, xg)

            bs = V.dofmap.index_map_bs
            for i, dof in enumerate(dofs):
                for k in range(bs):
                    # Create fem function that is identical with desired test function
                    v = _fem.Function(V)
                    v.x.array[:] = 0
                    v.x.array[dof * bs + k] = 1

                    # Create expression vor evaluating test function and evaluate
                    expr = _fem.Expression(v, x_ref)
                    expr_vals = expr.eval([cell])

                    # Create expression vor evaluating derivative of test function and evaluate
                    expr2 = _fem.Expression(ufl.grad(v.sub(k)), x_ref)
                    expr_vals2 = expr2.eval([cell])

                    # loop over quadrature points connected to facet_o
                    for q in q_indices:

                        # compare values of test functions
                        offset = link * num_q_points * len(dofs) * bs + i * num_q_points * bs
                        assert(np.isclose(expr_vals[0][q * bs + k], test_fn[f][offset + q * bs + k]))

                        # retrieve jacobian from surf_der and compute J^T*dv to compare values of derivatives
                        J = np.zeros((gdim, tdim))
                        dv_ref = np.zeros(tdim)
                        offset = len(dofs) * num_q_points * bs * num_q_points + link * len(dofs) * bs * num_q_points
                        for n in range(tdim):
                            # retrieve values from packed derivativs
                            dv_ref[n] = test_fn[f][offset + n
                                                   * len(dofs) * num_q_points * bs * num_q_points
                                                   + i * num_q_points * bs + q * bs + k]
                            for m in range(gdim):
                                # retrieve entries of jacobian
                                J[m, n] = surf_der[f][q * tdim * gdim + m * tdim + n]

                        # retrieve dv from expression values
                        dv = np.zeros(gdim)
                        for m in range(gdim):
                            dv[m] = expr_vals2[0][q * gdim + m]

                        # compare J^T*dv and compare with packed derivatives
                        assert(np.allclose(np.dot(np.transpose(J), dv), dv_ref))

                    # loop over quadrature points connected to different facet
                    for q in zero_ind:

                        # ensure values are zero if q not connected to quadrature point
                        offset = link * num_q_points * len(dofs) * bs + i * num_q_points * bs
                        assert(np.isclose(0, test_fn[f][offset + q * bs + k]))

                        offset = len(dofs) * num_q_points * bs * num_q_points + link * len(dofs) * bs * num_q_points
                        for n in range(tdim):
                            # retrieve values from packed derivativs
                            dv_ref[n] = test_fn[f][offset + n
                                                   * len(dofs) * num_q_points * bs * num_q_points
                                                   + i * num_q_points * bs + q * bs + k]
                            assert(np.isclose(0, dv_ref[n]))


@ pytest.mark.parametrize("ct", ["quadrilateral", "triangle", "tetrahedron", "hexahedron"])
@ pytest.mark.parametrize("gap", [0.5, -0.5])
@ pytest.mark.parametrize("q_deg", [1, 2, 3])
@ pytest.mark.parametrize("delta", [0.0, -0.5])
@ pytest.mark.parametrize("surface", [0, 1])
def test_pack_u(ct, gap, q_deg, delta, surface):

    # Create function space
    V = create_functionspaces(ct, gap, delta)

    # Retrieve mesh and mesh data
    mesh = V.mesh
    tdim = mesh.topology.dim
    gdim = mesh.geometry.dim
    cmap = mesh.geometry.cmap
    geometry_dofmap = mesh.geometry.dofmap
    bs = V.dofmap.index_map_bs

    # locate facets
    facets1 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[tdim - 1], 0))
    facets2 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[tdim - 1], -gap))
    facets = [facets1, facets2]

    # create meshtags
    val0 = np.full(len(facets1), 0, dtype=np.int32)
    val1 = np.full(len(facets2), 1, dtype=np.int32)
    values = np.hstack([val0, val1])
    indices = np.concatenate([facets1, facets2])
    sorted_facets = np.argsort(indices)
    facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])

    def func(x):
        vals = np.zeros((gdim, x.shape[1]))
        vals[0] = 0.1 * x[0]
        vals[1] = 0.23 * x[1]
        return vals

    # Compute function that is known on each side
    u = _fem.Function(V)
    u.interpolate(func)

    # create contact class
    opposites = [1, 0]
    s = surface
    o = opposites[surface]
    contact = dolfinx_contact.cpp.Contact(facet_marker, [0, 1], V._cpp_object)
    contact.set_quadrature_degree(q_deg)
    contact.create_distance_map(s, o)

    # Pack gap on surface, pack u opposite surface
    gap = contact.pack_gap(s)
    u_opposite = contact.pack_u_contact(s, u._cpp_object, gap, 1)
    surf_der = contact.pack_surface_derivatives(s, gap)

    # Retrieve surface facets
    s_facets = np.sort(facets[s])
    lookup = contact.facet_map(s)

    # Create facet to cell connectivity
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mesh.topology.create_connectivity(tdim, tdim - 1)
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)

    # loop over facets in surface
    for f in range(len(s_facets)):

        # Compute evaluation points
        qp_phys = contact.qp_phys(s, f)
        num_q_points = qp_phys.shape[0]
        points = np.zeros((num_q_points, 3))

        for i in range(num_q_points):
            for j in range(gdim):
                points[i, j] = qp_phys[i, j] + gap[f][i * gdim + j]

        # retrieve connected facets
        connected_facets = lookup.links(f)
        unique_facets = np.unique(np.sort(connected_facets))

        # loop over unique connected facets
        for facet_o in unique_facets:

            # retrieve cell index and cell dofs for facet_o
            cell = f_to_c.links(facet_o)[0]

            # find quadrature points linked to facet_o
            q_indices = np.argwhere(connected_facets == facet_o).reshape(-1)

            # retrieve cell geometry and compute pull back of physical points to reference cell
            gdofs = geometry_dofmap.links(cell)
            xg = mesh.geometry.x[gdofs]
            pts = np.array(points[q_indices, :]).reshape((len(q_indices), 3))
            x_ref = cmap.pull_back(pts, xg)

            # use expression to evaluate u
            expr = _fem.Expression(u, x_ref)
            expr_vals = expr.eval([cell]).reshape(-1)

            # extract values from u_opposite
            vals = np.zeros(len(q_indices) * bs)
            for i, q in enumerate(q_indices):
                vals[i * bs:(i + 1) * bs] = u_opposite[f][q * bs:(q + 1) * bs]

            # compare expression and packed u
            assert(np.allclose(expr_vals, vals))

            # loop over block
            for k in range(bs):

                # use expression to evaluate gradient
                expr = _fem.Expression(ufl.grad(u.sub(k)), x_ref)
                expr_vals = expr.eval([cell]).reshape(-1)

                # extract jacobian from surf_der and gradient from u_opposite and expr_vals
                for i, q in enumerate(q_indices):
                    # gradient from expression
                    vals1 = np.zeros(gdim)
                    for j in range(gdim):
                        vals1[j] = expr_vals[i * gdim + j]

                    # jacobian and gradient from packed data
                    J = np.zeros((gdim, tdim))
                    vals2 = np.zeros(tdim)
                    for j in range(tdim):
                        index = (j + 1) * num_q_points * bs + q * bs + k
                        vals2[j] = u_opposite[f][index]
                        # retrieve jacobian
                        for n in range(gdim):
                            J[n, j] = surf_der[f][q * tdim * gdim + n * tdim + j]

                # compare gradient from expression and u_opposite
                assert(np.allclose(np.dot(np.transpose(J), vals1), vals2))
