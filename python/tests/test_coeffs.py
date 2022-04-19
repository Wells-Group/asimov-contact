# Copyright (C) 2021 Jørgen S. Dokken, Sarah Roggendorf
#
# SPDX-License-Identifier:   MIT

import basix
import dolfinx_cuas.cpp
import numpy as np
import pytest
from dolfinx.fem import (Expression, Function, FunctionSpace, IntegralType,
                         VectorFunctionSpace)
from dolfinx.mesh import (create_unit_cube, create_unit_square,
                          locate_entities_boundary)
from mpi4py import MPI
from ufl import FiniteElement, MixedElement, VectorElement

import dolfinx_contact.cpp


@pytest.mark.parametrize("quadrature_degree", range(1, 6))
@pytest.mark.parametrize("degree", range(1, 6))
@pytest.mark.parametrize("space", ["CG", "N1curl", "DG"])
def test_pack_coeff_at_quadrature(quadrature_degree, space, degree):
    N = 15
    mesh = create_unit_square(MPI.COMM_WORLD, N, N)
    if space == "CG":
        V = VectorFunctionSpace(mesh, (space, degree))
    elif space == "N1curl":
        V = FunctionSpace(mesh, (space, degree))
    elif space == "DG":
        V = FunctionSpace(mesh, (space, degree - 1))
    else:
        raise RuntimeError("Unsupported space")

    v = Function(V)
    if space == "DG":
        v.interpolate(lambda x: x[0] < 0.5 + x[1])
    else:
        v.interpolate(lambda x: (x[1], -x[0]))

    # Pack coeffs with cuas
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    integration_entities = dolfinx_cuas.compute_active_entities(mesh, cells,
                                                                IntegralType.cell)
    coeffs_cuas = dolfinx_contact.cpp.pack_coefficient_quadrature(
        v._cpp_object, quadrature_degree, integration_entities)

    # Use prepare quadrature points and geometry for eval
    quadrature_points, wts = basix.make_quadrature(
        basix.QuadratureType.Default, basix.CellType.triangle, quadrature_degree)

    # Use Expression to verify packing
    expr = Expression(v, quadrature_points)
    expr_vals = expr.eval(cells)

    assert np.allclose(coeffs_cuas, expr_vals)


@pytest.mark.parametrize("quadrature_degree", range(1, 6))
@pytest.mark.parametrize("degree", range(1, 6))
@pytest.mark.parametrize("space", ["CG", "DG", "N1curl"])
def test_pack_coeff_on_facet(quadrature_degree, space, degree):
    N = 15
    mesh = create_unit_square(MPI.COMM_WORLD, N, N)
    if space == "CG":
        V = VectorFunctionSpace(mesh, (space, degree))
    elif space == "N1curl":
        V = FunctionSpace(mesh, (space, degree))
    elif space == "DG":
        V = FunctionSpace(mesh, (space, degree - 1))
    else:
        raise RuntimeError("Unsupported space")

    v = Function(V)
    if space == "DG":
        v.interpolate(lambda x: x[0] < 0.5 + x[1])
    else:
        v.interpolate(lambda x: (x[1], -x[0]))

    # Find facets on boundary to integrate over
    facets = locate_entities_boundary(mesh, mesh.topology.dim - 1,
                                      lambda x: np.logical_or(np.isclose(x[0], 0.0),
                                                              np.isclose(x[0], 1.0)))

    # Pack coeffs with cuas
    integration_entities = dolfinx_cuas.compute_active_entities(mesh, facets,
                                                                IntegralType.exterior_facet)
    coeffs_cuas = dolfinx_contact.cpp.pack_coefficient_quadrature(
        v._cpp_object, quadrature_degree, integration_entities)

    # Use prepare quadrature points and geometry for eval
    qp_test, wts = basix.make_quadrature(basix.QuadratureType.Default,
                                         basix.CellType.triangle, quadrature_degree)
    x_g = mesh.geometry.x
    tdim = mesh.topology.dim
    fdim = tdim - 1
    coord_dofs = mesh.geometry.dofmap

    # Connectivity to evaluate at quadrature points
    mesh.topology.create_connectivity(fdim, tdim)
    f_to_c = mesh.topology.connectivity(fdim, tdim)
    mesh.topology.create_connectivity(tdim, fdim)
    c_to_f = mesh.topology.connectivity(tdim, fdim)

    q_rule = dolfinx_cuas.cpp.QuadratureRule(mesh.topology.cell_type, quadrature_degree, mesh.topology.dim - 1)
    # Eval for each cell
    for index, facet in enumerate(facets):
        cell = f_to_c.links(facet)[0]
        xg = x_g[coord_dofs.links(cell)]
        # find local index of facet
        cell_facets = c_to_f.links(cell)
        local_index = np.where(cell_facets == facet)[0]
        quadrature_points = q_rule.points(local_index)
        x = mesh.geometry.cmap.push_forward(quadrature_points, xg)
        v_ex = v.eval(x, np.full(x.shape[0], cell))

        # Compare
        assert(np.allclose(v_ex.reshape(-1), coeffs_cuas[index]))


@pytest.mark.parametrize("quadrature_degree", range(1, 5))
@pytest.mark.parametrize("degree", range(1, 5))
def test_sub_coeff(quadrature_degree, degree):
    N = 10
    mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N)
    el = FiniteElement("N1curl", mesh.ufl_cell(), degree)
    v_el = VectorElement("CG", mesh.ufl_cell(), degree)
    V = FunctionSpace(mesh, MixedElement([v_el, el]))

    v = Function(V)
    v.sub(0).interpolate(lambda x: (x[1], -x[0], 3 * x[2]))
    v.sub(1).interpolate(lambda x: (-5 * x[2], x[1], x[0]))

    # Pack coeffs with cuas
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    integration_entities = dolfinx_cuas.compute_active_entities(mesh, cells,
                                                                IntegralType.cell)

    # Use prepare quadrature points and geometry for eval
    quadrature_points, wts = basix.make_quadrature(
        basix.QuadratureType.Default, basix.CellType.tetrahedron, quadrature_degree)
    x_g = mesh.geometry.x
    coord_dofs = mesh.geometry.dofmap
    num_sub_spaces = V.num_sub_spaces
    for i in range(num_sub_spaces):
        vi = v.sub(i)
        coeffs_cuas = dolfinx_contact.cpp.pack_coefficient_quadrature(
            vi._cpp_object, quadrature_degree, integration_entities)

        # Use Expression to verify packing
        expr = Expression(vi, quadrature_points)
        expr_vals = expr.eval(cells)

        assert np.allclose(coeffs_cuas, expr_vals)
