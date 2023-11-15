# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT
#
# This tests the custom assembly for the unbiased Nitsche formulation in a special case
# that can be expressed using ufl:
# We consider a very simple test case made up of two disconnected elements with a constant
# gap in x[tdim-1]-direction. The contact surfaces are made up of exactly one edge
# from each element that are perfectly aligned such that the quadrature points only
# differ in the x[tdim-1]-direction by the given gap.
# For comparison, we consider a DG function space on a mesh that is constructed by
# removing the gap between the elements and merging the edges making up the contact
# surface into one. This allows us to use DG-functions and ufl to formulate the contact
# terms in the variational form by suitably adjusting the deformation u and using the given
# constant gap.


import numpy as np
import scipy
import pytest
import ufl
from basix.ufl import element
from dolfinx.cpp.mesh import to_type
import dolfinx.fem as _fem
from dolfinx.graph import adjacencylist
from dolfinx.mesh import (CellType, locate_entities_boundary, locate_entities, create_mesh,
                          compute_midpoints, meshtags)
from mpi4py import MPI

import dolfinx_contact
import dolfinx_contact.cpp
from dolfinx_contact.helpers import (R_minus, dR_minus, R_plus, dR_plus, epsilon,
                                     lame_parameters, sigma_func, tangential_proj,
                                     ball_projection, d_ball_projection,
                                     d_alpha_ball_projection)

kt = dolfinx_contact.cpp.Kernel


def tied_dg(u0, v0, h, n, gamma, theta, sigma, dS):
    F = gamma / h('+') * ufl.inner(ufl.jump(u0), ufl.jump(v0)) * dS + \
        gamma / h('-') * ufl.inner(ufl.jump(u0), ufl.jump(v0)) * dS -\
        ufl.inner(ufl.avg(sigma(u0)) * n('+'), ufl.jump(v0)) * dS +\
        ufl.inner(ufl.avg(sigma(u0)) * n('-'), ufl.jump(v0)) * dS -\
        theta * ufl.inner(ufl.avg(sigma(v0)) * n('+'), ufl.jump(u0)) * dS +\
        theta * ufl.inner(ufl.avg(sigma(v0)) * n('-'), ufl.jump(u0)) * dS
    return 0.5 * F


def DG_rhs_plus(u0, v0, h, n, gamma, theta, sigma, gap, dS):
    # This version of the ufl form agrees with the formulation in https://doi.org/10.1007/s00211-018-0950-x
    def Pn_g(u, a, b):
        return ufl.dot(u(a) - u(b), -n(b)) - gap - (h(a) / gamma) * ufl.dot(sigma(u(a)) * n(a), -n(b))

    def Pn_gtheta(v, a, b):
        return ufl.dot(v(a) - v(b), -n(b)) - theta * (h(a) / gamma) * ufl.dot(sigma(v(a)) * n(a), -n(b))

    F = 0.5 * (gamma / h('+')) * R_plus(Pn_g(u0, '+', '-')) * Pn_gtheta(v0, '+', '-') * dS

    F += 0.5 * (gamma / h('-')) * R_plus(Pn_g(u0, '-', '+')) * Pn_gtheta(v0, '-', '+') * dS

    return F


def DG_rhs_minus(u0, v0, h, n, gamma, theta, sigma, gap, dS):
    # This version of the ufl form agrees with its one-sided equivalent in nitsche_ufl.py
    def Pn_g(u, a, b):
        return ufl.dot(sigma(u(a)) * n(a), -n(b)) + (gamma / h(a)) * (gap - ufl.dot(u(a) - u(b), -n(b)))

    def Pn_gtheta(v, a, b):
        return theta * ufl.dot(sigma(v(a)) * n(a), -n(b)) - (gamma / h(a)) * ufl.dot(v(a) - v(b), -n(b))

    F = 0.5 * (h('+') / gamma) * R_minus(Pn_g(u0, '+', '-')) * Pn_gtheta(v0, '+', '-') * dS

    F += 0.5 * (h('-') / gamma) * R_minus(Pn_g(u0, '-', '+')) * Pn_gtheta(v0, '-', '+') * dS

    return F


def DG_jac_plus(u0, v0, w0, h, n, gamma, theta, sigma, gap, dS):
    # This version of the ufl form agrees with the formulation in https://doi.org/10.1007/s00211-018-0950-x
    def Pn_g(u, a, b):
        return ufl.dot(u(a) - u(b), -n(b)) - gap - (h(a) / gamma) * ufl.dot(sigma(u(a)) * n(a), -n(b))

    def Pn_gtheta(v, a, b, t):
        return ufl.dot(v(a) - v(b), -n(b)) - t * (h(a) / gamma) * ufl.dot(sigma(v(a)) * n(a), -n(b))

    J = 0.5 * (gamma / h('+')) * dR_plus(Pn_g(u0, '+', '-')) * \
        Pn_gtheta(w0, '+', '-', 1.0) * Pn_gtheta(v0, '+', '-', theta) * dS

    J += 0.5 * (gamma / h('-')) * dR_plus(Pn_g(u0, '-', '+')) * \
        Pn_gtheta(w0, '-', '+', 1.0) * Pn_gtheta(v0, '-', '+', theta) * dS

    return J


def DG_jac_minus(u0, v0, w0, h, n, gamma, theta, sigma, gap, dS):
    # This version of the ufl form agrees with its one-sided equivalent in nitsche_ufl.py
    def Pn_g(u, a, b):
        return ufl.dot(sigma(u(a)) * n(a), -n(b)) + (gamma / h(a)) * (gap - ufl.dot(u(a) - u(b), -n(b)))

    def Pn_gtheta(v, a, b, t):
        return t * ufl.dot(sigma(v(a)) * n(a), -n(b)) - (gamma / h(a)) * ufl.dot(v(a) - v(b), -n(b))

    J = 0.5 * (h('+') / gamma) * dR_minus(Pn_g(u0, '+', '-')) * \
        Pn_gtheta(w0, '+', '-', 1.0) * Pn_gtheta(v0, '+', '-', theta) * dS

    J += 0.5 * (h('-') / gamma) * dR_minus(Pn_g(u0, '-', '+')) * \
        Pn_gtheta(w0, '-', '+', 1.0) * Pn_gtheta(v0, '-', '+', theta) * dS

    return J


def DG_rhs_tresca(u0, v0, h, n, gamma, theta, sigma, fric, dS, gdim):
    """
    UFL version of the Tresca friction term for the unbiased Nitsche formulation
    """
    def Pt_g(u, a, b, c):
        return tangential_proj(u(a) - u(b) - h(a) * c * sigma(u(a)) * n(a), -n(b))
    return 0.5 * gamma / h('+') * ufl.dot(ball_projection(Pt_g(u0, '+', '-', 1. / gamma), fric * h('+') / gamma, gdim),
                                          Pt_g(v0, '+', '-', theta / gamma)) * dS\
        + 0.5 * gamma / h('-') * ufl.dot(ball_projection(Pt_g(u0, '-', '+', 1. / gamma), fric * h('-') / gamma, gdim),
                                         Pt_g(v0, '-', '+', theta / gamma)) * dS


def DG_jac_tresca(u0, v0, w0, h, n, gamma, theta, sigma, fric, dS, gdim):
    """
    UFL version of the Jacobian for the Tresca friction term for the unbiased Nitsche formulation
    """
    def Pt_g(u, a, b, c):
        return tangential_proj(u(a) - u(b) - h(a) * c * sigma(u(a)) * n(a), -n(b))

    J = 0.5 * gamma / h('+') * ufl.dot(d_ball_projection(Pt_g(u0, '+', '-', 1. / gamma), fric * h('+') / gamma, gdim)
                                       * Pt_g(w0, '+', '-', 1. / gamma), Pt_g(v0, '+', '-', theta / gamma)) * dS
    J += 0.5 * gamma / h('-') * ufl.dot(d_ball_projection(Pt_g(u0, '-', '+', 1. / gamma), fric * h('-') / gamma, gdim)
                                        * Pt_g(w0, '-', '+', 1. / gamma), Pt_g(v0, '-', '+', theta / gamma)) * dS

    return J


def DG_rhs_coulomb(u0, v0, h, n, gamma, theta, sigma, gap, fric, dS, gdim):
    """
    UFL version of the Coulomb friction term for the unbiased Nitsche formulation
    """
    def Pn_g(u, a, b):
        return ufl.dot(u(a) - u(b), -n(b)) - gap - (h(a) / gamma) * ufl.dot(sigma(u(a)) * n(a), -n(b))

    def Pt_g(u, a, b, c):
        return tangential_proj(u(a) - u(b) - h(a) * c * sigma(u(a)) * n(a), -n(b))

    Pn_u_plus = R_plus(Pn_g(u0, '+', '-'))
    Pn_u_minus = R_plus(Pn_g(u0, '-', '+'))
    return 0.5 * gamma / h('+') * ufl.dot(ball_projection(Pt_g(u0, '+', '-', 1. / gamma),
                                                          Pn_u_plus * fric * h('+') / gamma, gdim),
                                          Pt_g(v0, '+', '-', theta / gamma)) * dS\
        + 0.5 * gamma / h('-') * ufl.dot(ball_projection(Pt_g(u0, '-', '+', 1. / gamma),
                                                         Pn_u_minus * fric * h('-') / gamma, gdim),
                                         Pt_g(v0, '-', '+', theta / gamma)) * dS


def DG_jac_coulomb(u0, v0, w0, h, n, gamma, theta, sigma, gap, fric, dS, gdim):
    """
    UFL version of the Jacobian for the Coulomb friction term for the unbiased Nitsche formulation
    """
    def Pn_g(u, a, b):
        return ufl.dot(u(a) - u(b), -n(b)) - gap - (h(a) / gamma) * ufl.dot(sigma(u(a)) * n(a), -n(b))

    def Pn_gtheta(v, a, b, t):
        return ufl.dot(v(a) - v(b), -n(b)) - t * (h(a) / gamma) * ufl.dot(sigma(v(a)) * n(a), -n(b))

    def Pt_g(u, a, b, c):
        return tangential_proj(u(a) - u(b) - h(a) * c * sigma(u(a)) * n(a), -n(b))

    Pn_u_plus = R_plus(Pn_g(u0, '+', '-'))
    Pn_u_minus = R_plus(Pn_g(u0, '-', '+'))

    J = 0.5 * gamma / h('+') * ufl.dot(d_ball_projection(Pt_g(u0, '+', '-', 1. / gamma),
                                                         Pn_u_plus * fric * h('+') / gamma, gdim)
                                       * Pt_g(w0, '+', '-', 1. / gamma), Pt_g(v0, '+', '-', theta / gamma)) * dS
    J += 0.5 * gamma / h('-') * ufl.dot(d_ball_projection(Pt_g(u0, '-', '+', 1. / gamma),
                                                          Pn_u_minus * fric * h('-') / gamma, gdim)
                                        * Pt_g(w0, '-', '+', 1. / gamma), Pt_g(v0, '-', '+', theta / gamma)) * dS

    d_alpha_plus = d_alpha_ball_projection(Pt_g(u0, '+', '-', 1. / gamma), Pn_u_plus * fric * h('+') / gamma,
                                           dR_plus(Pn_g(u0, '+', '-')) * fric * h('+') / gamma, gdim)
    d_alpha_minus = d_alpha_ball_projection(Pt_g(u0, '-', '+', 1. / gamma), Pn_u_plus * fric * h('-') / gamma,
                                            dR_plus(Pn_g(u0, '-', '+')) * fric * h('-') / gamma, gdim)
    J += 0.5 * gamma / h('+') * Pn_gtheta(w0, '+', '-', 1.0) * \
        ufl.dot(d_alpha_plus, Pt_g(v0, '+', '-', theta / gamma)) * dS
    J += 0.5 * gamma / h('-') * Pn_gtheta(w0, '-', '+', 1.0) * \
        ufl.dot(d_alpha_minus, Pt_g(v0, '-', '+', theta / gamma)) * dS
    return J

def compute_dof_permutations_all(V_dg, V_cg, gap):
    '''The meshes used for the two different formulations are
       created independently of each other. Therefore we need to
       determine how to map the dofs from one mesh to the other in
       order to compare the results'''
    mesh_dg = V_dg.mesh
    mesh_cg = V_cg.mesh
    bs = V_cg.dofmap.index_map_bs
    tdim = mesh_dg.topology.dim
    mesh_dg.topology.create_connectivity(tdim - 1, tdim)
    f_to_c_dg = mesh_dg.topology.connectivity(tdim - 1, tdim)

    mesh_cg.topology.create_connectivity(tdim - 1, tdim)
    mesh_cg.topology.create_connectivity(tdim, tdim - 1)
    f_to_c_cg = mesh_cg.topology.connectivity(tdim - 1, tdim)
    c_to_f_cg = mesh_cg.topology.connectivity(tdim, tdim - 1)
    x_cg = V_cg.tabulate_dof_coordinates()
    x_dg = V_dg.tabulate_dof_coordinates()



    # retrieve all dg dofs on mesh without gap for each cell
    # and modify coordinates by gap if necessary
    num_cells = mesh_dg.topology.index_map(tdim).size_local
    for cell in range(num_cells):
        midpoint = compute_midpoints(mesh_dg, tdim, [cell])[0]
        if not midpoint[tdim - 1] > 0:
            # coordinates of corresponding dofs need to be adjusted by gap
            dofs_dg1 = V_dg.dofmap.cell_dofs(cell)
            x_dg[dofs_dg1, tdim - 1] -= gap

    indices_dg = np.zeros(x_dg.shape[0] * bs, dtype=np.int32)
    for i in range(x_cg.shape[0]):
        coordinates = x_cg[i, :]
        # find dg dofs that correspond to cg dofs for first element
        index = np.isclose(x_dg, coordinates).all(axis=1).nonzero()[0][0]
        for k in range(bs):
            indices_dg[i * bs + k] = index * bs + k
    return indices_dg
    

def create_functionspaces(ct, gap, vector=True):
    ''' This is a helper function to create the two element function spaces
        both for custom assembly and the DG formulation for
        quads, triangles, hexes and tetrahedra'''
    cell_type = to_type(ct)
    if cell_type == CellType.quadrilateral:
        x_ufl = np.array([[0, 0], [0.8, 0], [0.1, 1.3], [0.7, 1.2], [-0.1, -1.2], [0.8, -1.1]])
        x_custom = np.array([[0, 0], [0.8, 0], [0.1, 1.3], [0.7, 1.2], [0, -gap],
                             [0.8, -gap], [-0.1, -1.2 - gap], [0.8, -1.1 - gap]])
        cells_ufl = np.array([[0, 1, 2, 3], [4, 5, 0, 1]], dtype=np.int32)
        cells_custom = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    elif cell_type == CellType.triangle:
        x_ufl = np.array([[0, 0, 0], [0.8, 0, 0], [0.3, 1.3, 0.0], [0.4, -1.2, 0.0]])
        x_custom = np.array([[0, 0, 0], [0.8, 0, 0], [0.3, 1.3, 0.0], [
            0, -gap, 0], [0.8, -gap, 0], [0.4, -1.2 - gap, 0.0]])
        cells_ufl = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
        cells_custom = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    elif cell_type == CellType.tetrahedron:
        x_ufl = np.array([[0, 0, 0], [1.1, 0, 0], [0.3, 1.0, 0], [1, 1.2, 1.5], [0.8, 1.2, -1.6]])
        x_custom = np.array([[0, 0, 0], [1.1, 0, 0], [0.3, 1.0, 0], [1, 1.2, 1.5], [
            0, 0, -gap], [1.1, 0, -gap], [0.3, 1.0, -gap], [0.8, 1.2, -1.6 - gap]])
        cells_ufl = np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int32)
        cells_custom = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    elif cell_type == CellType.hexahedron:
        x_ufl = np.array([[0, 0, 0], [1.1, 0, 0], [0.1, 1, 0], [1, 1.2, 0],
                          [0, 0, 1.2], [1.0, 0, 1], [0, 1, 1], [1, 1, 1],
                          [0, 0, -1.2], [1.0, 0, -1.3], [0, 1, -1], [1, 1, -1]])
        x_custom = np.array([[0, 0, 0], [1.1, 0, 0], [0.1, 1, 0], [1, 1.2, 0],
                             [0, 0, 1.2], [1.0, 0, 1], [0, 1, 1], [1, 1, 1],
                             [0, 0, -1.2 - gap], [1.0, 0, -1.3 - gap], [0, 1, -1 - gap], [1, 1, -1 - gap],
                             [0, 0, -gap], [1.1, 0, -gap], [0.1, 1, -gap], [1, 1.2, -gap]])
        cells_ufl = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 0, 1, 2, 3]], dtype=np.int32)
        cells_custom = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]], dtype=np.int32)
    else:
        raise ValueError(f"Unsupported mesh type {ct}")
    gdim = x_ufl.shape[1]
    coord_el = element("Lagrange", cell_type.name, 1, shape=(gdim,), gdim=gdim)
    if vector:
        el_shape = (gdim,)
    else:
        el_shape = ()
    domain = ufl.Mesh(coord_el)
    mesh_ufl = create_mesh(MPI.COMM_WORLD, cells_ufl, x_ufl, domain)
    el_ufl = element("DG", cell_type.name, 1, shape=el_shape, gdim=gdim)
    V_ufl = _fem.FunctionSpace(mesh_ufl, el_ufl)
    el_custom = element("Lagrange", cell_type.name, 1, shape=el_shape, gdim=gdim)
    domain_custom = ufl.Mesh(coord_el)
    mesh_custom = create_mesh(MPI.COMM_WORLD, cells_custom, x_custom, domain_custom)
    V_custom = _fem.FunctionSpace(mesh_custom, el_custom)

    return V_ufl, V_custom


def locate_contact_facets_custom(V, gap):
    '''This function locates the contact facets for custom assembly and ensures
       that the correct facet is chosen if the gap is zero'''
    # Retrieve mesh
    mesh = V.mesh

    # locate facets
    tdim = mesh.topology.dim
    facets1 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[tdim - 1], 0))
    facets2 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[tdim - 1], -gap))

    # choose correct facet if gap is zero
    mesh.topology.create_connectivity(tdim - 1, tdim)
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    cells = [[], []]
    contact_facets1 = []
    for facet in facets1:
        cell = f_to_c.links(facet)[0]
        cell_midpoints = compute_midpoints(mesh, tdim, [cell])
        if cell_midpoints[0][tdim - 1] > 0:
            contact_facets1.append(facet)
            cells[0].append(cell)
    contact_facets2 = []
    for facet in facets2:
        cell = f_to_c.links(facet)[0]
        cell_midpoints = compute_midpoints(mesh, tdim, [cell])
        if cell_midpoints[0][tdim - 1] < -gap:
            contact_facets2.append(facet)
            cells[1].append(cell)

    return cells, [contact_facets1, contact_facets2]


def create_facet_markers(mesh, facets_cg):
    # create meshtags
    tdim = mesh.topology.dim
    val0 = np.full(len(facets_cg[0]), 0, dtype=np.int32)
    val1 = np.full(len(facets_cg[1]), 1, dtype=np.int32)
    values = np.hstack([val0, val1])
    indices = np.concatenate([facets_cg[0], facets_cg[1]])
    sorted_facets = np.argsort(indices)
    return meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])


def create_contact_data(V, u, quadrature_degree, lmbda, mu, facets_cg, search, tied=False):
    ''' This function creates the contact class and the coefficients
        passed to the assembly for the unbiased Nitsche method'''

    # Retrieve mesh
    mesh = V.mesh
    # create meshtags
    facet_marker = create_facet_markers(mesh, facets_cg)

    data = np.array([0, 1], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    surfaces = adjacencylist(data, offsets)
    # create contact class
    contact = dolfinx_contact.cpp.Contact([facet_marker._cpp_object], surfaces, [(0, 1), (1, 0)],
                                          mesh._cpp_object, quadrature_degree=quadrature_degree,
                                          search_method=search)
    contact.create_distance_map(0)
    contact.create_distance_map(1)

    # Pack material parameters mu and lambda on each contact surface
    V2 = _fem.FunctionSpace(mesh, ("DG", 0))
    lmbda2 = _fem.Function(V2)
    lmbda2.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))
    mu2 = _fem.Function(V2)
    mu2.interpolate(lambda x: np.full((1, x.shape[1]), mu))
    fric_coeff = _fem.Function(V2)
    fric_coeff.interpolate(lambda x: np.full((1, x.shape[1]), 0.1))

    # compute active entities
    integral = _fem.IntegralType.exterior_facet
    entities_0, num_local_0 = dolfinx_contact.compute_active_entities(mesh._cpp_object, facets_cg[0], integral)
    entities_0 = entities_0[:num_local_0]
    entities_1, num_local_1 = dolfinx_contact.compute_active_entities(mesh._cpp_object, facets_cg[1], integral)
    entities_1 = entities_1[:num_local_1]

    # pack coeffs mu, lambda
    material_0 = np.hstack([dolfinx_contact.cpp.pack_coefficient_quadrature(
        mu2._cpp_object, 0, entities_0),
        dolfinx_contact.cpp.pack_coefficient_quadrature(
        lmbda2._cpp_object, 0, entities_0)])
    material_1 = np.hstack([dolfinx_contact.cpp.pack_coefficient_quadrature(
        mu2._cpp_object, 0, entities_1),
        dolfinx_contact.cpp.pack_coefficient_quadrature(
        lmbda2._cpp_object, 0, entities_1)])
    friction_0 = dolfinx_contact.cpp.pack_coefficient_quadrature(
        fric_coeff._cpp_object, 0, entities_0)
    friction_1 = dolfinx_contact.cpp.pack_coefficient_quadrature(
        fric_coeff._cpp_object, 0, entities_1)

    # Pack cell diameter on each surface
    h = ufl.CellDiameter(mesh)
    surface_cells = np.unique(np.hstack([entities_0[:, 0], entities_1[:, 0]]))
    h_int = _fem.Function(V2)
    expr = _fem.Expression(h, V2.element.interpolation_points())
    h_int.interpolate(expr, surface_cells)
    h_0 = dolfinx_contact.cpp.pack_coefficient_quadrature(
        h_int._cpp_object, 0, entities_0)
    h_1 = dolfinx_contact.cpp.pack_coefficient_quadrature(
        h_int._cpp_object, 0, entities_1)

    # Pack gap
    gap_0 = contact.pack_gap(0)
    gap_1 = contact.pack_gap(1)

    # Pack test functions
    test_fn_0 = contact.pack_test_functions(0, V._cpp_object)
    test_fn_1 = contact.pack_test_functions(1, V._cpp_object)
    # pack u
    u_opp_0 = contact.pack_u_contact(0, u._cpp_object)
    u_opp_1 = contact.pack_u_contact(1, u._cpp_object)
    u_0 = dolfinx_contact.cpp.pack_coefficient_quadrature(u._cpp_object, quadrature_degree, entities_0)
    u_1 = dolfinx_contact.cpp.pack_coefficient_quadrature(u._cpp_object, quadrature_degree, entities_1)
    grad_u_0 = dolfinx_contact.cpp.pack_gradient_quadrature(u._cpp_object, quadrature_degree, entities_0)
    grad_u_1 = dolfinx_contact.cpp.pack_gradient_quadrature(u._cpp_object, quadrature_degree, entities_1)
    if tied:
        grad_test_fn_0 = contact.pack_grad_test_functions(0, V._cpp_object)
        grad_test_fn_1 = contact.pack_grad_test_functions(1, V._cpp_object)
        grad_u_opp_0 = contact.pack_grad_u_contact(0, u._cpp_object)
        grad_u_opp_1 = contact.pack_grad_u_contact(1, u._cpp_object)

        # Concatenate all coeffs
        coeff_0 = np.hstack([material_0, h_0, test_fn_0, grad_test_fn_0, u_0, grad_u_0, u_opp_0, grad_u_opp_0])
        coeff_1 = np.hstack([material_1, h_1, test_fn_1, grad_test_fn_1, u_1, grad_u_1, u_opp_1, grad_u_opp_1])
    else:
        n_0 = contact.pack_ny(0)
        n_1 = contact.pack_ny(1)

        # Concatenate all coeffs
        coeff_0 = np.hstack([material_0, friction_0, h_0, gap_0, n_0, test_fn_0, u_0, grad_u_0, u_opp_0])
        coeff_1 = np.hstack([material_1, friction_1, h_1, gap_1, n_1, test_fn_1, u_1, grad_u_1, u_opp_1])

    return contact, coeff_0, coeff_1


@pytest.mark.parametrize("ct", ["triangle", "quadrilateral", "tetrahedron", "hexahedron"])
@pytest.mark.parametrize("gap", [0.5, -0.5])
@pytest.mark.parametrize("quadrature_degree", [1, 5])
@pytest.mark.parametrize("theta", [1, 0, -1])
@pytest.mark.parametrize("formulation", ["meshtie", "frictionless", "tresca", "coulomb"])
@pytest.mark.parametrize("search", [dolfinx_contact.cpp.ContactMode.ClosestPoint,
                                    dolfinx_contact.cpp.ContactMode.Raytracing])
def test_contact_kernels(ct, gap, quadrature_degree, theta, formulation, search):

    if formulation == "meshtie" and search == dolfinx_contact.cpp.ContactMode.Raytracing:
        pytest.xfail("Raytracing and MeshTie not supported")

    # Compute lame parameters
    plane_strain = False
    E = 1e3
    nu = 0.1
    mu_func, lambda_func = lame_parameters(plane_strain)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    # Nitche parameter
    gamma = 10

    # create meshes and function spaces
    V_ufl, V_custom = create_functionspaces(ct, gap)
    mesh_ufl = V_ufl.mesh
    mesh_custom = V_custom.mesh
    tdim = mesh_ufl.topology.dim
    gdim = mesh_ufl.geometry.dim
    TOL = 1e-7
    cells_ufl_0 = locate_entities(mesh_ufl, tdim, lambda x: x[tdim - 1] > 0 - TOL)
    cells_ufl_1 = locate_entities(mesh_ufl, tdim, lambda x: x[tdim - 1] < 0 + TOL)

    def _u0(x):
        values = np.zeros((gdim, x.shape[1]))
        for i in range(tdim):
            values[i] = np.sin(x[i]) + 1
        return values

    def _u1(x):
        values = np.zeros((gdim, x.shape[1]))
        for i in range(tdim):
            values[i] = np.sin(x[i]) + 2
        return values

    def _u2(x):
        values = np.zeros((gdim, x.shape[1]))
        for i in range(tdim):
            values[i] = np.sin(x[i] + gap) + 2 if i == tdim - 1 else np.sin(x[i]) + 2
        return values

    # DG ufl 'contact'
    u0 = _fem.Function(V_ufl)
    u0.interpolate(_u0, cells_ufl_0)
    u0.interpolate(_u1, cells_ufl_1)
    v0 = ufl.TestFunction(V_ufl)
    w0 = ufl.TrialFunction(V_ufl)
    metadata = {"quadrature_degree": quadrature_degree}
    dS = ufl.Measure("dS", domain=mesh_ufl, metadata=metadata)

    n = ufl.FacetNormal(mesh_ufl)

    # Scaled Nitsche parameter
    h = ufl.CellDiameter(mesh_ufl)
    gamma_scaled = gamma * E

    # DG formulation

    if formulation == "meshtie":
        F0 = tied_dg(u0, v0, h, n, gamma_scaled, theta, sigma, dS)
        J0 = tied_dg(w0, v0, h, n, gamma_scaled, theta, sigma, dS)
        kernel_type_rhs = kt.MeshTieRhs
        kernel_type_jac = kt.MeshTieJac
    elif formulation == "frictionless":
        # Contact terms formulated using ufl consistent with https://doi.org/10.1007/s00211-018-0950-x
        F0 = DG_rhs_plus(u0, v0, h, n, gamma_scaled, theta, sigma, gap, dS)
        J0 = DG_jac_plus(u0, v0, w0, h, n, gamma_scaled, theta, sigma, gap, dS)
        kernel_type_rhs = kt.Rhs
        kernel_type_jac = kt.Jac
    elif formulation == "tresca":
        F0 = DG_rhs_tresca(u0, v0, h, n, gamma_scaled, theta, sigma, 0.1, dS, gdim)
        J0 = DG_jac_tresca(u0, v0, w0, h, n, gamma_scaled, theta, sigma, 0.1, dS, gdim)
        kernel_type_rhs = kt.TrescaRhs
        kernel_type_jac = kt.TrescaJac
    else:
        F0 = DG_rhs_coulomb(u0, v0, h, n, gamma_scaled, theta, sigma, gap, 0.1, dS, gdim)
        J0 = DG_jac_coulomb(u0, v0, w0, h, n, gamma_scaled, theta, sigma, gap, 0.1, dS, gdim)
        kernel_type_rhs = kt.CoulombRhs
        kernel_type_jac = kt.CoulombJac

    # rhs vector
    F0 = _fem.form(F0)
    b0 = _fem.petsc.create_vector(F0)
    b0.zeroEntries()
    _fem.petsc.assemble_vector(b0, F0)

    # lhs matrix
    J0 = _fem.form(J0)
    A0 = _fem.petsc.create_matrix(J0)
    A0.zeroEntries()
    _fem.petsc.assemble_matrix(A0, J0)
    A0.assemble()

    # Custom assembly
    cells, facets_cg = locate_contact_facets_custom(V_custom, gap)

    # Fem functions
    u1 = _fem.Function(V_custom)
    v1 = ufl.TestFunction(V_custom)
    w1 = ufl.TrialFunction(V_custom)

    u1.interpolate(_u0, cells[0])
    u1.interpolate(_u2, cells[1])
    u1.x.scatter_forward()

    # Dummy form for creating vector/matrix
    dx = ufl.Measure("dx", domain=mesh_custom)
    F_custom = ufl.inner(sigma(u1), epsilon(v1)) * dx
    J_custom = ufl.inner(sigma(w1), epsilon(v1)) * dx

    contact, c_0, c_1 = create_contact_data(V_custom, u1, quadrature_degree, lmbda,
                                            mu, facets_cg, search, formulation == "meshtie")

    # Generate residual data structures
    F_custom = _fem.form(F0)
    kernel_rhs = contact.generate_kernel(kernel_type_rhs, V_custom._cpp_object)
    b1 = _fem.petsc.create_vector(F_custom)

    # Generate residual data structures
    J_custom = _fem.form(J_custom)
    kernel_jac = contact.generate_kernel(kernel_type_jac, V_custom._cpp_object)
    A1 = contact.create_matrix(J_custom._cpp_object)

    # Pack constants
    consts = np.array([gamma_scaled, theta])

    # Assemble  residual
    b1.zeroEntries()
    contact.assemble_vector(b1, 0, kernel_rhs, c_0, consts, V_custom._cpp_object)
    contact.assemble_vector(b1, 1, kernel_rhs, c_1, consts, V_custom._cpp_object)

    # Assemble  jacobian
    A1.zeroEntries()
    contact.assemble_matrix(A1, 0, kernel_jac, c_0, consts, V_custom._cpp_object)
    contact.assemble_matrix(A1, 1, kernel_jac, c_1, consts, V_custom._cpp_object)
    A1.assemble()

    # Retrieve data necessary for comparison
    tdim = mesh_ufl.topology.dim
    facet_dg = locate_entities(mesh_ufl, tdim - 1, lambda x: np.isclose(x[tdim - 1], 0))
    ind_dg = compute_dof_permutations_all(V_ufl, V_custom, gap)

    # Compare rhs
    assert np.allclose(b0.array[ind_dg], b1.array)

    # create scipy matrix
    ai, aj, av = A0.getValuesCSR()
    A_sp = scipy.sparse.csr_matrix((av, aj, ai), shape=A0.getSize()).todense()
    bi, bj, bv = A1.getValuesCSR()
    B_sp = scipy.sparse.csr_matrix((bv, bj, bi), shape=A1.getSize()).todense()

    assert np.allclose(A_sp[ind_dg, :][:, ind_dg], B_sp)

    # Sanity check different formulations
    if formulation == "frictionless":
        # Contact terms formulated using ufl consistent with nitsche_ufl.py
        F2 = DG_rhs_minus(u0, v0, h, n, gamma_scaled, theta, sigma, gap, dS)

        F2 = _fem.form(F2)
        b2 = _fem.petsc.create_vector(F2)
        b2.zeroEntries()
        _fem.petsc.assemble_vector(b2, F2)
        assert np.allclose(b1.array, b2.array[ind_dg])

        # Contact terms formulated using ufl consistent with nitsche_ufl.py
        J2 = DG_jac_minus(u0, v0, w0, h, n, gamma_scaled, theta, sigma, gap, dS)
        J2 = _fem.form(J2)
        A2 = _fem.petsc.create_matrix(J2)
        A2.zeroEntries()
        _fem.petsc.assemble_matrix(A2, J2)
        A2.assemble()

        ci, cj, cv = A2.getValuesCSR()
        C_sp = scipy.sparse.csr_matrix((cv, cj, ci), shape=A2.getSize()).todense()
        assert np.allclose(C_sp[ind_dg, :][:, ind_dg], B_sp)


def poisson_dg(u0, v0, h, n, kdt, gamma, theta, dS):
    F = gamma / h('+') * ufl.inner(ufl.jump(u0), ufl.jump(v0)) * dS + \
        gamma / h('-') * ufl.inner(ufl.jump(u0), ufl.jump(v0)) * dS -\
        ufl.inner(ufl.avg(ufl.grad(u0)), n('+')) * ufl.jump(v0) * dS +\
        ufl.inner(ufl.avg(ufl.grad(u0)), n('-')) * ufl.jump(v0) * dS -\
        theta * ufl.inner(ufl.avg(ufl.grad(v0)), n('+')) * ufl.jump(u0) * dS +\
        theta * ufl.inner(ufl.avg(ufl.grad(v0)), n('-')) * ufl.jump(u0) * dS
    return 0.5 * kdt * F


@pytest.mark.parametrize("ct", ["triangle", "quadrilateral", "tetrahedron", "hexahedron"])
@pytest.mark.parametrize("gap", [0.5, -0.5])
@pytest.mark.parametrize("quadrature_degree", [1, 5])
@pytest.mark.parametrize("theta", [1, 0, -1])
def test_poisson_kernels(ct, gap, quadrature_degree, theta):

    # Nitche parameter
    gamma = 10

    # create meshes and function spaces
    V_ufl, V_custom = create_functionspaces(ct, gap, vector=False)
    mesh_ufl = V_ufl.mesh
    mesh_custom = V_custom.mesh
    tdim = mesh_ufl.topology.dim
    TOL = 1e-7
    cells_ufl_0 = locate_entities(mesh_ufl, tdim, lambda x: x[tdim - 1] > 0 - TOL)
    cells_ufl_1 = locate_entities(mesh_ufl, tdim, lambda x: x[tdim - 1] < 0 + TOL)

    def _u0(x):
        return np.sin(x[0]) + 1

    def _u1(x):
        return np.sin(x[tdim - 1]) + 2

    def _u2(x):
        return np.sin(x[tdim - 1] + gap) + 2

    # DG ufl 'contact'
    u0 = _fem.Function(V_ufl)
    u0.interpolate(_u0, cells_ufl_0)
    u0.interpolate(_u1, cells_ufl_1)
    v0 = ufl.TestFunction(V_ufl)
    w0 = ufl.TrialFunction(V_ufl)
    metadata = {"quadrature_degree": quadrature_degree}
    dS = ufl.Measure("dS", domain=mesh_ufl, metadata=metadata)

    n = ufl.FacetNormal(mesh_ufl)

    # Scaled Nitsche parameter
    h = ufl.CellDiameter(mesh_ufl)

    # DG formulation
    kdt = 5
    dx = ufl.Measure("dx", domain=mesh_ufl)
    F0 = poisson_dg(u0, v0, h, n, kdt, gamma, theta, dS)
    J0 = poisson_dg(w0, v0, h, n, kdt, gamma, theta, dS)

    # rhs vector
    F0 = _fem.form(F0)
    b0 = _fem.petsc.create_vector(F0)
    b0.zeroEntries()
    _fem.petsc.assemble_vector(b0, F0)

    # lhs matrix
    J0 = _fem.form(J0)
    A0 = _fem.petsc.create_matrix(J0)
    A0.zeroEntries()
    _fem.petsc.assemble_matrix(A0, J0)
    A0.assemble()

    # Custom assembly
    cells, facets_cg = locate_contact_facets_custom(V_custom, gap)

    # Fem functions
    u1 = _fem.Function(V_custom)
    v1 = ufl.TestFunction(V_custom)
    w1 = ufl.TrialFunction(V_custom)

    u1.interpolate(_u0, cells[0])
    u1.interpolate(_u2, cells[1])
    u1.x.scatter_forward()

    # Dummy form for creating vector/matrix
    dx = ufl.Measure("dx", domain=mesh_custom)
    F_custom = ufl.inner(ufl.grad(u1), ufl.grad(v1)) * dx
    J_custom = kdt * ufl.inner(ufl.grad(w1), ufl.grad(v1)) * dx

    # meshtie surfaces
    facet_marker = create_facet_markers(mesh_custom, facets_cg)

    data = np.array([0, 1], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    surfaces = adjacencylist(data, offsets)
    # initialise meshties
    meshties = dolfinx_contact.cpp.MeshTie([facet_marker._cpp_object], surfaces, [(0, 1), (1, 0)],
                                           mesh_custom._cpp_object, quadrature_degree=quadrature_degree)
    meshties.generate_heat_transfer_data(u1._cpp_object, kdt, gamma, theta)

    # Generate residual data structures
    F_custom = _fem.form(F0)
    b1 = _fem.petsc.create_vector(F_custom)

    # # Generate matrix
    J_custom = _fem.form(J_custom)
    A1 = meshties.create_matrix(J_custom._cpp_object)

    # Assemble  residual
    b1.zeroEntries()
    meshties.assemble_vector_heat_transfer(b1, V_custom._cpp_object)

    # Assemble  jacobian
    A1.zeroEntries()
    meshties.assemble_matrix_heat_transfer(A1, V_custom._cpp_object)
    A1.assemble()

    # Retrieve data necessary for comparison
    tdim = mesh_ufl.topology.dim
    facet_dg = locate_entities(mesh_ufl, tdim - 1, lambda x: np.isclose(x[tdim - 1], 0))
    #ind_cg, ind_dg = compute_dof_permutations(V_ufl, V_custom, gap, [facet_dg], facets_cg)
    ind_dg = compute_dof_permutations_all(V_ufl, V_custom, gap)

    # Compare rhs
    # assert np.allclose(b0.array[ind_dg], b1.array)

    # create scipy matrix
    ai, aj, av = A0.getValuesCSR()
    A_sp = scipy.sparse.csr_matrix((av, aj, ai), shape=A0.getSize()).todense()
    bi, bj, bv = A1.getValuesCSR()
    B_sp = scipy.sparse.csr_matrix((bv, bj, bi), shape=A1.getSize()).todense()

    assert np.allclose(A_sp[:, ind_dg][ind_dg, :], B_sp)
