# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:   MIT

from mpi4py import MPI

import basix.ufl
import dolfinx.fem as _fem
import dolfinx_contact
import dolfinx_contact.cpp
import numpy as np
import pytest
import ufl
from dolfinx.cpp.mesh import to_type
from dolfinx.graph import adjacencylist
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_mesh, locate_entities_boundary, meshtags


def create_functionspaces(tempdir, ct, gap, delta, disp):
    """This is a helper function to create the two element function spaces
    for custom assembly using quads, triangles, hexes and tetrahedra."""
    cell_type = to_type(ct)
    if cell_type == CellType.quadrilateral:
        x_1 = np.array([[0, 0], [0.8, 0], [0.1, 1.3], [0.7, 1.2]])
        x_2 = np.array([[0, 0], [0.8, 0], [-0.1, -1.2], [0.8, -1.1]])
        for point in x_2:
            point[0] += delta
            point[1] -= gap
        x_3 = np.array([x_2[0].copy() + [1.6, 0], x_2[2].copy() + [1.6, 0]])
        x = np.vstack([x_1, x_2, x_3])
        cells = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [5, 8, 7, 9]], dtype=np.int64)
    elif cell_type == CellType.triangle:
        x = np.array(
            [
                [0, 0, 0],
                [0.8, 0, 0],
                [0.3, 1.3, 0.0],
                [0 + delta, -gap, 0],
                [0.8 + delta, -gap, 0],
                [0.4 + delta, -1.2 - gap, 0.0],
            ]
        )
        for point in x:
            point[2] = 3 * point[0] + 2 * point[1]  # plane given by z = 3x +2y
        cells = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    elif cell_type == CellType.tetrahedron:
        x = np.array(
            [
                [0, 0, 0],
                [1.1, 0, 0],
                [0.3, 1.0, 0],
                [1, 1.2, 1.5],
                [0 + delta, 0, -gap],
                [1.1 + delta, 0, -gap],
                [0.3 + delta, 1.0, -gap],
                [0.8 + delta, 1.2, -1.6 - gap],
            ]
        )
        cells = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    elif cell_type == CellType.hexahedron:
        x_1 = np.array(
            [
                [0, 0, 0],
                [1.1, 0, 0],
                [0.1, 1, 0],
                [1, 1.2, 0],
                [0, 0, 1.2],
                [1.0, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
            ]
        )
        x_2 = np.array(
            [
                [0, 0, -1.2],
                [1.0, 0, -1.3],
                [0.1, 1, -1],
                [1, 1, -1],
                [0, 0, 0],
                [1.1, 0, 0],
                [0.1, 1, 0],
                [1, 1.2, 0],
            ]
        )
        for point in x_2:
            point[0] += delta
            point[2] -= gap
        x_3 = np.array(
            [
                x_2[0].copy() + [2.0, 0, 0],
                x_2[2].copy() + [2.0, 0, 0],
                x_2[4].copy() + [2.0, 0, 0],
                x_2[6].copy() + [2.0, 0, 0],
            ]
        )
        x = np.vstack([x_1, x_2, x_3])
        cells = np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [8, 9, 10, 11, 12, 13, 14, 15],
                [9, 16, 10, 17, 13, 18, 15, 19],
            ],
            dtype=np.int64,
        )
    else:
        raise ValueError(f"Unsupported mesh type {ct}")
    el = basix.ufl.element("Lagrange", ct, 1, shape=(x.shape[1],))
    domain = ufl.Mesh(el)
    mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
    if disp:
        el = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,))
    else:
        el = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1)
    V = _fem.functionspace(mesh, el)
    with XDMFFile(mesh.comm, tempdir / "test_mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
    return V


def compare_test_fn(fn_space, test_fn, grad_test_fn, q_indices, link, x_ref, cell):
    # Retrieve mesh and mesh data
    mesh = fn_space.mesh
    gdim = mesh.geometry.dim
    bs = fn_space.dofmap.index_map_bs
    dofs = fn_space.dofmap.cell_dofs(cell[0])

    num_q_points = x_ref.shape[0]
    cell_arr = np.array(cell, dtype=np.int32)
    eps = 2e2 * np.finfo(x_ref.dtype).eps
    for i, dof in enumerate(dofs):
        for k in range(bs):
            # Create fem function that is identical with desired test function
            v = _fem.Function(fn_space)
            v.x.array[:] = 0
            v.x.array[dof * bs + k] = 1

            # Create expression for evaluating test function and evaluate
            expr = _fem.Expression(v, x_ref)
            expr_vals = expr.eval(mesh, cell_arr)

            # Create expression for evaluating derivative of test
            # function and evaluate
            if bs == 1:
                expr2 = _fem.Expression(ufl.grad(v), x_ref)
            else:
                expr2 = _fem.Expression(ufl.grad(v.sub(k)), x_ref)
            expr_vals2 = expr2.eval(mesh, cell)
            # compare values of test functions
            offset = link * num_q_points * len(dofs) * bs + i * num_q_points * bs
            np.testing.assert_allclose(
                expr_vals[0][q_indices * bs + k], test_fn[offset + q_indices * bs + k], atol=eps
            )
            # retrieve dv from expression values and packed test fn
            dv1 = np.zeros((len(q_indices), gdim))
            dv2 = np.zeros((len(q_indices), gdim))
            offset = link * num_q_points * len(dofs) * gdim + i * gdim * num_q_points
            for m in range(gdim):
                dv1[:, m] = expr_vals2[0][q_indices * gdim + m]
                dv2[:, m] = grad_test_fn[offset + q_indices * gdim + m]
            np.testing.assert_allclose(dv1, dv2, atol=eps)


def assert_zero_test_fn(fn_space, test_fn, grad_test_fn, num_q_points, zero_ind, link, cell):
    # Retrieve mesh and mesh data
    mesh = fn_space.mesh
    gdim = mesh.geometry.dim
    bs = fn_space.dofmap.index_map_bs
    dofs = fn_space.dofmap.cell_dofs(cell[0])
    for i in range(len(dofs)):
        for k in range(bs):
            # ensure values are zero if q not connected to quadrature point
            offset = link * num_q_points * len(dofs) * bs + i * num_q_points * bs
            np.testing.assert_allclose(0, test_fn[offset + zero_ind * bs + k])
            # retrieve dv from expression values and packed test fn
            if len(zero_ind) > 0:
                dv2 = np.zeros((len(zero_ind), gdim))
                offset = link * num_q_points * len(dofs) * gdim + i * gdim * num_q_points
                for m in range(gdim):
                    dv2[:, m] = grad_test_fn[offset + zero_ind * gdim + m]
                np.testing.assert_allclose(np.zeros((len(zero_ind), gdim)), dv2)


def compare_u(fn_space, u, u_opposite, grad_u_opposite, q_indices, x_ref, cell):
    bs = fn_space.dofmap.index_map_bs
    gdim = fn_space.mesh.geometry.dim
    mesh = fn_space.mesh

    # use expression to evaluate u
    expr = _fem.Expression(u, x_ref[q_indices, :])
    expr_vals = expr.eval(mesh, np.array(cell, dtype=np.int32))

    # extract values from u_opposite
    vals = np.zeros(len(q_indices) * bs)
    for i, q in enumerate(q_indices):
        vals[i * bs : (i + 1) * bs] = u_opposite[q * bs : (q + 1) * bs]

    # compare expression and packed u
    eps = np.finfo(mesh.geometry.x.dtype).eps
    np.testing.assert_allclose(expr_vals.flatten(), vals, atol=eps)
    # cell_arr = np.array(cell, dtype=np.int32).reshape(-1)
    # loop over block
    for k in range(bs):
        # use expression to evaluate gradient
        if bs == 1:
            expr = _fem.Expression(ufl.grad(u), x_ref[q_indices, :])
        else:
            expr = _fem.Expression(ufl.grad(u.sub(k)), x_ref[q_indices, :])
        expr_vals = expr.eval(mesh, cell).reshape(-1)

        # extract jacobian from surf_der and gradient from u_opposite and expr_vals
        for i, q in enumerate(q_indices):
            # gradient from expression
            vals1 = np.zeros(gdim)
            for j in range(gdim):
                vals1[j] = expr_vals[i * gdim + j]

            vals2 = np.zeros(gdim)
            for j in range(gdim):
                index = gdim * bs * q + k * gdim + j
                vals2[j] = grad_u_opposite[index]

        # compare gradient from expression and u_opposite
        np.testing.assert_allclose(vals1, vals2, atol=eps)


@pytest.mark.parametrize("ct", ["quadrilateral", "tetrahedron", "hexahedron", "triangle"])
@pytest.mark.parametrize("gap", [0.5, -0.5])
@pytest.mark.parametrize("q_deg", [1, 2, 3])
@pytest.mark.parametrize("delta", [0.0, -0.5])
@pytest.mark.parametrize("surface", [0, 1])
@pytest.mark.parametrize("disp", [True, False])
def test_packing(tmp_path, ct, gap, q_deg, delta, surface, disp):
    # Create function space
    V = create_functionspaces(tmp_path, ct, gap, delta, disp)

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

    if disp:

        def func(x):
            vals = np.zeros((gdim, x.shape[1]))
            vals[0] = x[0] ** 2
            vals[1] = 0.23 * x[1]
            return vals
    else:

        def func(x):
            vals = np.zeros((1, x.shape[1]))
            vals[0] = np.sin(x[0])
            return vals

    # Compute function that is known on each side
    u = _fem.Function(V)
    u.interpolate(func)

    # create contact class
    opposites = [1, 0]
    s = surface
    o = opposites[surface]
    data = np.array([0, 1], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    surfaces = adjacencylist(data, offsets)
    search_mode = [dolfinx_contact.cpp.ContactMode.ClosestPoint]
    contact = dolfinx_contact.cpp.Contact(
        [facet_marker._cpp_object],
        surfaces,
        [(s, o)],
        mesh._cpp_object,
        search_mode,
        quadrature_degree=q_deg,
    )
    if disp:
        contact.update_submesh_geometry(u._cpp_object)
    contact.create_distance_map(0)

    # Pack gap on surface, pack test functions, u on opposite surface
    gap = contact.pack_gap(0)
    test_fn = contact.pack_test_functions(0, V._cpp_object)
    u_packed = contact.pack_u_contact(0, u._cpp_object)
    grad_test_fn = contact.pack_grad_test_functions(0, V._cpp_object)
    grad_u = contact.pack_grad_u_contact(0, u._cpp_object)

    # Retrieve surface facets
    s_facets = np.sort(facets[s])
    lookup = contact.facet_map(0)

    # Create facet to cell connectivity
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mesh.topology.create_connectivity(tdim, tdim - 1)
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)

    # loop over facets in surface
    for f in range(len(s_facets)):
        # Compute evaluation points
        qp_phys = contact.qp_phys(s, f)
        num_q_points = qp_phys.shape[0]
        points = np.zeros((num_q_points, gdim))

        points[:, :gdim] = qp_phys[:, :gdim] + gap[f].reshape((num_q_points, gdim))
        if disp:
            points[:, :gdim] = points[:, :gdim] - u_packed[f].reshape((num_q_points, gdim))

        # retrieve connected facets
        connected_facets = lookup.links(f)
        unique_facets = np.unique(np.sort(connected_facets))

        # loop over unique connected facets
        for link, facet_o in enumerate(unique_facets):
            # retrieve cell index and cell dofs for facet_o
            cell = f_to_c.links(facet_o)

            # find quadrature points linked to facet_o
            q_indices = np.argwhere(connected_facets == facet_o).reshape(-1)
            zero_ind = np.argwhere(connected_facets != facet_o).reshape(-1)

            # retrieve cell geometry and compute pull back of physical points to reference cell
            gdofs = geometry_dofmap[cell][0]
            xg = mesh.geometry.x[gdofs, :gdim]
            x_ref = cmap.pull_back(points, xg)

            compare_test_fn(V, test_fn[f], grad_test_fn[f], q_indices, link, x_ref, cell)
            compare_u(V, u, u_packed[f], grad_u[f], q_indices, x_ref, cell)
            assert_zero_test_fn(V, test_fn[f], grad_test_fn[f], num_q_points, zero_ind, link, cell)
