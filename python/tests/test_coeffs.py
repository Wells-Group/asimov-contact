# Copyright (C) 2021 Jørgen S. Dokken, Sarah Roggendorf
#
# SPDX-License-Identifier:   MIT

import basix
import numpy as np
import pytest
from dolfinx.fem import (Expression, Function, FunctionSpace, IntegralType,
                         VectorFunctionSpace)
from dolfinx.mesh import (CellType, create_unit_cube, create_unit_square,
                          locate_entities_boundary, to_string)
from mpi4py import MPI
from ufl import FiniteElement, MixedElement, VectorElement, grad

import dolfinx_contact.cpp


@pytest.mark.parametrize("ct", [CellType.triangle, CellType.quadrilateral])
@pytest.mark.parametrize("quadrature_degree", range(1, 6))
@pytest.mark.parametrize("degree", range(1, 6))
@pytest.mark.parametrize("space", ["Lagrange", "N1curl", "DG"])
def test_pack_coeff_at_quadrature(ct, quadrature_degree, space, degree):
    N = 15
    mesh = create_unit_square(MPI.COMM_WORLD, N, N, cell_type=ct)
    if space == "Lagrange":
        V = VectorFunctionSpace(mesh, (space, degree))
    elif space == "N1curl":
        if ct == CellType.quadrilateral:
            space = "RTCE"
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

    # Create quadrature points for integration on facets
    ct = mesh.topology.cell_types[0]

    # Pack coeffs
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    integration_entities, num_local = dolfinx_contact.compute_active_entities(mesh._cpp_object, cells,
                                                                              IntegralType.cell)
    integration_entities = integration_entities[:num_local]
    coeffs = dolfinx_contact.cpp.pack_coefficient_quadrature(
        v._cpp_object, quadrature_degree, integration_entities)

    # Use prepare quadrature points and geometry for eval
    quadrature_points, _ = basix.make_quadrature(basix.cell.string_to_type(to_string(ct)),
                                                 quadrature_degree, basix.QuadratureType.Default)

    # Use Expression to verify packing
    expr = Expression(v, quadrature_points)
    expr_vals = expr.eval(mesh, cells)
    assert np.allclose(coeffs, expr_vals)

    if space not in ['N1curl', 'RTCE']:
        coeffs = dolfinx_contact.cpp.pack_gradient_quadrature(
            v._cpp_object, quadrature_degree, integration_entities)
        if space == 'DG' and degree == 1:
            assert (np.allclose(0.0, coeffs))
        else:
            expr = Expression(grad(v), quadrature_points)
            expr_vals = expr.eval(mesh, cells)
            assert np.allclose(coeffs, expr_vals)


@pytest.mark.parametrize("quadrature_degree", range(1, 6))
@pytest.mark.parametrize("degree", range(1, 6))
@pytest.mark.parametrize("space", ["Lagrange", "DG", "N1curl"])
def test_pack_coeff_on_facet(quadrature_degree, space, degree):
    N = 15
    mesh = create_unit_square(MPI.COMM_WORLD, N, N)
    if space == "Lagrange":
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
    # Compuate integration entitites
    integration_entities, num_local = dolfinx_contact.compute_active_entities(mesh._cpp_object, facets,
                                                                              IntegralType.exterior_facet)
    integration_entities = integration_entities[:num_local]

    coeffs = dolfinx_contact.cpp.pack_coefficient_quadrature(
        v._cpp_object, quadrature_degree, integration_entities)
    cstride = coeffs.shape[1]
    tdim = mesh.topology.dim
    fdim = tdim - 1

    # Create quadrature points for integration on facets
    ct = mesh.topology.cell_types[0]
    q_rule = dolfinx_contact.QuadratureRule(ct, quadrature_degree, fdim)

    # Compute coefficients at quadrature points using Expression
    q_points = q_rule.points()
    expr = Expression(v, q_points)
    expr_vals = expr.eval(mesh, integration_entities[:, 0])

    for i, entity in enumerate(integration_entities):
        local_index = entity[1]
        assert np.allclose(coeffs[i],
                           expr_vals[i, cstride * local_index:cstride * (local_index + 1)])
    if space not in ['N1curl', 'RTCE']:
        coeffs = dolfinx_contact.cpp.pack_gradient_quadrature(
            v._cpp_object, quadrature_degree, integration_entities)
        if space == 'DG' and degree == 1:
            assert (np.allclose(0.0, coeffs))
        else:
            gdim = mesh.geometry.dim
            expr = Expression(grad(v), q_points)
            expr_vals = expr.eval(mesh, integration_entities[:, 0])
            for i, entity in enumerate(integration_entities):
                local_index = entity[1]
                assert np.allclose(coeffs[i],
                                   expr_vals[i, gdim * cstride * local_index:gdim * cstride * (local_index + 1)])


@pytest.mark.parametrize("quadrature_degree", range(1, 5))
@pytest.mark.parametrize("degree", range(1, 5))
def test_sub_coeff(quadrature_degree, degree):
    N = 10
    mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N)
    el = FiniteElement("N1curl", mesh.ufl_cell(), degree)
    v_el = VectorElement("Lagrange", mesh.ufl_cell(), degree)
    V = FunctionSpace(mesh, MixedElement([v_el, el]))

    v = Function(V)
    v.sub(0).interpolate(lambda x: (x[1], -x[0], 3 * x[2]))
    v.sub(1).interpolate(lambda x: (-5 * x[2], x[1], x[0]))

    # Pack coeffs
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    integration_entities, num_local = dolfinx_contact.compute_active_entities(mesh._cpp_object, cells,
                                                                              IntegralType.cell)
    integration_entities = integration_entities[:num_local]

    # Use prepare quadrature points and geometry for eval
    quadrature_points, wts = basix.make_quadrature(
        basix.CellType.tetrahedron, quadrature_degree, basix.QuadratureType.Default)
    num_sub_spaces = V.num_sub_spaces
    for i in range(num_sub_spaces):
        vi = v.sub(i)
        coeffs = dolfinx_contact.cpp.pack_coefficient_quadrature(
            vi._cpp_object, quadrature_degree, integration_entities)

        # Use Expression to verify packing
        expr = Expression(vi, quadrature_points)
        expr_vals = expr.eval(mesh, cells)

        assert np.allclose(coeffs, expr_vals)


@pytest.mark.parametrize("quadrature_degree", range(1, 5))
@pytest.mark.parametrize("degree", range(1, 5))
def test_sub_coeff_grad(quadrature_degree, degree):
    N = 10
    mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N)
    el = FiniteElement("DG", mesh.ufl_cell(), degree)
    v_el = VectorElement("Lagrange", mesh.ufl_cell(), degree)
    V = FunctionSpace(mesh, MixedElement([v_el, el]))

    v = Function(V)
    v.sub(0).interpolate(lambda x: (x[1], -x[0], 3 * x[2]))
    v.sub(1).interpolate(lambda x: x[0] < 0.5 + x[1])

    # Pack coeffs
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    cells = np.arange(num_cells, dtype=np.int32)
    integration_entities, num_local = dolfinx_contact.compute_active_entities(mesh._cpp_object, cells,
                                                                              IntegralType.cell)
    integration_entities = integration_entities[:num_local]

    # Use prepare quadrature points and geometry for eval
    quadrature_points, wts = basix.make_quadrature(
        basix.CellType.tetrahedron, quadrature_degree, basix.QuadratureType.Default)
    num_sub_spaces = V.num_sub_spaces
    for i in range(num_sub_spaces):
        vi = v.sub(i)
        coeffs = dolfinx_contact.cpp.pack_gradient_quadrature(
            vi._cpp_object, quadrature_degree, integration_entities)

        # Use Expression to verify packing
        expr = Expression(grad(vi), quadrature_points)
        expr_vals = expr.eval(mesh, cells)

        assert np.allclose(coeffs, expr_vals)
