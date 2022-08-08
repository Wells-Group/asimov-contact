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
from dolfinx.cpp.mesh import to_type
import dolfinx.fem as _fem
from dolfinx.graph import create_adjacencylist
from dolfinx.mesh import (CellType, locate_entities_boundary, locate_entities, create_mesh,
                          compute_midpoints, meshtags)
from mpi4py import MPI

import dolfinx_cuas
import dolfinx_contact
import dolfinx_contact.cpp
from dolfinx_contact.helpers import (R_minus, dR_minus, R_plus, dR_plus, epsilon, lame_parameters, sigma_func)

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


def compute_dof_permutations(V_dg, V_cg, gap, facets_dg, facets_cg):
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

    for i in range(len(facets_dg)):
        facet_dg = facets_dg[i]

        dofs_cg = []
        coordinates_cg = []
        for facet_cg in np.array(facets_cg)[:, 0]:
            # retrieve dofs and dof coordinates for mesh with gap
            cell = f_to_c_cg.links(facet_cg)[0]
            all_facets = c_to_f_cg.links(cell)
            local_index = np.argwhere(np.array(all_facets) == facet_cg)[0, 0]
            dof_layout = V_cg.dofmap.dof_layout
            local_dofs = dof_layout.entity_closure_dofs(tdim - 1, local_index)
            dofs_cg0 = V_cg.dofmap.cell_dofs(cell)[local_dofs]
            dofs_cg.append(dofs_cg0)
            coordinates_cg.append(x_cg[dofs_cg0, :])

        # retrieve all dg dofs on mesh without gap for each cell
        # and modify coordinates by gap if necessary
        cells = f_to_c_dg.links(facet_dg)
        for cell in cells:
            midpoint = compute_midpoints(mesh_dg, tdim, [cell])[0]
            if midpoint[tdim - 1] > 0:
                # coordinates of corresponding dofs are identical for both meshes
                dofs_dg0 = V_dg.dofmap.cell_dofs(cell)
                coordinates_dg0 = x_dg[dofs_dg0, :]
            else:
                # coordinates of corresponding dofs need to be adjusted by gap
                dofs_dg1 = V_dg.dofmap.cell_dofs(cell)
                coordinates_dg1 = x_dg[dofs_dg1, :]
                coordinates_dg1[:, tdim - 1] -= gap

        # create array of indices to access corresponding function values
        num_dofs_f = dofs_cg[0].size
        indices_cg = np.zeros(bs * 2 * num_dofs_f, dtype=np.int32)
        for i, dofs in enumerate(dofs_cg):
            for j, dof in enumerate(dofs):
                for k in range(bs):
                    indices_cg[i * num_dofs_f * bs + j * bs + k] = bs * dof + k
        indices_dg = np.zeros(indices_cg.size, dtype=np.int32)
        for i, dofs in enumerate(dofs_cg[0]):
            coordinates = coordinates_cg[0][i, :]
            # find dg dofs that correspond to cg dofs for first element
            dof = dofs_dg0[np.isclose(coordinates_dg0, coordinates).all(axis=1).nonzero()[0][0]]
            # create array of indices to access corresponding function values
            for k in range(bs):
                indices_dg[i * bs + k] = dof * bs + k
        for i, dofs in enumerate(dofs_cg[1]):
            coordinates = coordinates_cg[1][i, :]
            # find dg dofs that correspond to cg dofs for first element
            dof = dofs_dg1[np.isclose(coordinates_dg1, coordinates).all(axis=1).nonzero()[0][0]]
            # create array of indices to access corresponding function values
            for k in range(bs):
                indices_dg[num_dofs_f * bs + i * bs + k] = dof * bs + k

        # return indices used for comparing assembled vectors/matrices
        return indices_cg, indices_dg


def create_functionspaces(ct, gap):
    ''' This is a helper function to create the two element function spaces
        both for custom assembly and the DG formulation for
        quads, triangles, hexes and tetrahedra'''
    cell_type = to_type(ct)
    if cell_type == CellType.quadrilateral:
        x_ufl = np.array([[0, 0], [0.8, 0], [0.1, 1.3], [0.7, 1.2], [-0.1, -1.2], [0.8, -1.1]])
        x_cuas = np.array([[0, 0], [0.8, 0], [0.1, 1.3], [0.7, 1.2], [0, -gap],
                           [0.8, -gap], [-0.1, -1.2 - gap], [0.8, -1.1 - gap]])
        cells_ufl = np.array([[0, 1, 2, 3], [4, 5, 0, 1]], dtype=np.int32)
        cells_cuas = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    elif cell_type == CellType.triangle:
        x_ufl = np.array([[0, 0, 0], [0.8, 0, 0], [0.3, 1.3, 0.0], [0.4, -1.2, 0.0]])
        x_cuas = np.array([[0, 0, 0], [0.8, 0, 0], [0.3, 1.3, 0.0], [
            0, -gap, 0], [0.8, -gap, 0], [0.4, -1.2 - gap, 0.0]])
        cells_ufl = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
        cells_cuas = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    elif cell_type == CellType.tetrahedron:
        x_ufl = np.array([[0, 0, 0], [1.1, 0, 0], [0.3, 1.0, 0], [1, 1.2, 1.5], [0.8, 1.2, -1.6]])
        x_cuas = np.array([[0, 0, 0], [1.1, 0, 0], [0.3, 1.0, 0], [1, 1.2, 1.5], [
            0, 0, -gap], [1.1, 0, -gap], [0.3, 1.0, -gap], [0.8, 1.2, -1.6 - gap]])
        cells_ufl = np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int32)
        cells_cuas = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int32)
    elif cell_type == CellType.hexahedron:
        x_ufl = np.array([[0, 0, 0], [1.1, 0, 0], [0.1, 1, 0], [1, 1.2, 0],
                          [0, 0, 1.2], [1.0, 0, 1], [0, 1, 1], [1, 1, 1],
                          [0, 0, -1.2], [1.0, 0, -1.3], [0, 1, -1], [1, 1, -1]])
        x_cuas = np.array([[0, 0, 0], [1.1, 0, 0], [0.1, 1, 0], [1, 1.2, 0],
                           [0, 0, 1.2], [1.0, 0, 1], [0, 1, 1], [1, 1, 1],
                           [0, 0, -1.2 - gap], [1.0, 0, -1.3 - gap], [0, 1, -1 - gap], [1, 1, -1 - gap],
                           [0, 0, -gap], [1.1, 0, -gap], [0.1, 1, -gap], [1, 1.2, -gap]])
        cells_ufl = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 0, 1, 2, 3]], dtype=np.int32)
        cells_cuas = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]], dtype=np.int32)
    else:
        raise ValueError(f"Unsupported mesh type {ct}")
    cell = ufl.Cell(ct, geometric_dimension=x_ufl.shape[1])
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
    mesh_ufl = create_mesh(MPI.COMM_WORLD, cells_ufl, x_ufl, domain)
    el_ufl = ufl.VectorElement("DG", mesh_ufl.ufl_cell(), 1)
    V_ufl = _fem.FunctionSpace(mesh_ufl, el_ufl)
    cell_cuas = ufl.Cell(ct, geometric_dimension=x_cuas.shape[1])
    domain_cuas = ufl.Mesh(ufl.VectorElement("Lagrange", cell_cuas, 1))
    mesh_cuas = create_mesh(MPI.COMM_WORLD, cells_cuas, x_cuas, domain_cuas)
    el_cuas = ufl.VectorElement("CG", mesh_cuas.ufl_cell(), 1)
    V_cuas = _fem.FunctionSpace(mesh_cuas, el_cuas)

    return V_ufl, V_cuas


def locate_contact_facets_cuas(V, gap):
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


def create_contact_data(V, u, quadrature_degree, lmbda, mu, facets_cg, search, tied=False):
    ''' This function creates the contact class and the coefficients
        passed to the assembly for the unbiased Nitsche method'''

    # Retrieve mesh
    mesh = V.mesh
    tdim = mesh.topology.dim

    # create meshtags
    val0 = np.full(len(facets_cg[0]), 0, dtype=np.int32)
    val1 = np.full(len(facets_cg[1]), 1, dtype=np.int32)
    values = np.hstack([val0, val1])
    indices = np.concatenate([facets_cg[0], facets_cg[1]])
    sorted_facets = np.argsort(indices)
    facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])

    data = np.array([0, 1], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    surfaces = create_adjacencylist(data, offsets)
    # create contact class
    contact = dolfinx_contact.cpp.Contact([facet_marker], surfaces, [(0, 1), (1, 0)],
                                          V._cpp_object, quadrature_degree=quadrature_degree,
                                          search_method=search)
    contact.create_distance_map(0)
    contact.create_distance_map(1)

    # Pack material parameters mu and lambda on each contact surface
    V2 = _fem.FunctionSpace(mesh, ("DG", 0))
    lmbda2 = _fem.Function(V2)
    lmbda2.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))
    mu2 = _fem.Function(V2)
    mu2.interpolate(lambda x: np.full((1, x.shape[1]), mu))

    # compute active entities
    integral = _fem.IntegralType.exterior_facet
    entities_0 = dolfinx_contact.compute_active_entities(mesh, facets_cg[0], integral)
    entities_1 = dolfinx_contact.compute_active_entities(mesh, facets_cg[1], integral)

    # pack coeffs mu, lambda
    material_0 = np.hstack([dolfinx_contact.cpp.pack_coefficient_quadrature(
        mu2._cpp_object, 0, entities_0),
        dolfinx_contact.cpp.pack_coefficient_quadrature(
        lmbda2._cpp_object, 0, entities_0)])
    material_1 = np.hstack([dolfinx_contact.cpp.pack_coefficient_quadrature(
        mu2._cpp_object, 0, entities_1),
        dolfinx_contact.cpp.pack_coefficient_quadrature(
        lmbda2._cpp_object, 0, entities_1)])

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
    test_fn_0 = contact.pack_test_functions(0)
    test_fn_1 = contact.pack_test_functions(1)
    # pack u
    u_opp_0 = contact.pack_u_contact(0, u._cpp_object)
    u_opp_1 = contact.pack_u_contact(1, u._cpp_object)
    u_0 = dolfinx_cuas.pack_coefficients([u], entities_0)
    u_1 = dolfinx_cuas.pack_coefficients([u], entities_1)
    if tied:
        grad_test_fn_0 = contact.pack_grad_test_functions(0, gap_0, np.zeros(gap_0.shape))
        grad_test_fn_1 = contact.pack_grad_test_functions(1, gap_1, np.zeros(gap_1.shape))
        grad_u_opp_0 = contact.pack_grad_u_contact(0, u._cpp_object, gap_0, np.zeros(gap_0.shape))
        grad_u_opp_1 = contact.pack_grad_u_contact(1, u._cpp_object, gap_1, np.zeros(gap_1.shape))

        # Concatenate all coeffs
        coeff_0 = np.hstack([material_0, h_0, test_fn_0, grad_test_fn_0, u_0, u_opp_0, grad_u_opp_0])
        coeff_1 = np.hstack([material_1, h_1, test_fn_1, grad_test_fn_1, u_1, u_opp_1, grad_u_opp_1])
    else:
        n_0 = contact.pack_ny(0)
        n_1 = contact.pack_ny(1)

        # Concatenate all coeffs
        coeff_0 = np.hstack([material_0, h_0, gap_0, n_0, test_fn_0, u_0, u_opp_0])
        coeff_1 = np.hstack([material_1, h_1, gap_1, n_1, test_fn_1, u_1, u_opp_1])

    return contact, coeff_0, coeff_1


@pytest.mark.parametrize("ct", ["quadrilateral", "triangle", "tetrahedron", "hexahedron"])
@pytest.mark.parametrize("gap", [1e-13, -1e-13, 0.5, -0.5])
@pytest.mark.parametrize("quadrature_degree", [1, 5])
@pytest.mark.parametrize("theta", [1, 0, -1])
@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("search", [dolfinx_contact.cpp.ContactMode.ClosestPoint,
                                    dolfinx_contact.cpp.ContactMode.Raytracing])
def test_contact_kernels(ct, gap, quadrature_degree, theta, tied, search):

    if tied and search == dolfinx_contact.cpp.ContactMode.Raytracing:
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
    V_ufl, V_cuas = create_functionspaces(ct, gap)
    mesh_ufl = V_ufl.mesh
    mesh_cuas = V_cuas.mesh
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

    if tied:
        F0 = tied_dg(u0, v0, h, n, gamma_scaled, theta, sigma, dS)
        J0 = tied_dg(w0, v0, h, n, gamma_scaled, theta, sigma, dS)
        kernel_type_rhs = kt.MeshTieRhs
        kernel_type_jac = kt.MeshTieJac
    else:
        # Contact terms formulated using ufl consistent with https://doi.org/10.1007/s00211-018-0950-x
        F0 = DG_rhs_plus(u0, v0, h, n, gamma_scaled, theta, sigma, gap, dS)
        J0 = DG_jac_plus(u0, v0, w0, h, n, gamma_scaled, theta, sigma, gap, dS)
        kernel_type_rhs = kt.Rhs
        kernel_type_jac = kt.Jac

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
    cells, facets_cg = locate_contact_facets_cuas(V_cuas, gap)

    # Fem functions
    u1 = _fem.Function(V_cuas)
    v1 = ufl.TestFunction(V_cuas)
    w1 = ufl.TrialFunction(V_cuas)

    u1.interpolate(_u0, cells[0])
    u1.interpolate(_u2, cells[1])
    u1.x.scatter_forward()

    # Dummy form for creating vector/matrix
    dx = ufl.Measure("dx", domain=mesh_cuas)
    F_cuas = ufl.inner(sigma(u1), epsilon(v1)) * dx
    J_cuas = ufl.inner(sigma(w1), epsilon(v1)) * dx

    contact, c_0, c_1 = create_contact_data(V_cuas, u1, quadrature_degree, lmbda, mu, facets_cg, search, tied)

    # Generate residual data structures
    F_cuas = _fem.form(F0)
    kernel_rhs = contact.generate_kernel(kernel_type_rhs)
    b1 = _fem.petsc.create_vector(F_cuas)

    # Generate residual data structures
    J_cuas = _fem.form(J_cuas)
    kernel_jac = contact.generate_kernel(kernel_type_jac)
    A1 = contact.create_matrix(J_cuas)

    # Pack constants
    consts = np.array([gamma_scaled, theta])

    # Assemble  residual
    b1.zeroEntries()
    contact.assemble_vector(b1, 0, kernel_rhs, c_0, consts)
    contact.assemble_vector(b1, 1, kernel_rhs, c_1, consts)

    # Assemble  jacobian
    A1.zeroEntries()
    contact.assemble_matrix(A1, [], 0, kernel_jac, c_0, consts)
    contact.assemble_matrix(A1, [], 1, kernel_jac, c_1, consts)
    A1.assemble()

    # Retrieve data necessary for comparison
    tdim = mesh_ufl.topology.dim
    facet_dg = locate_entities(mesh_ufl, tdim - 1, lambda x: np.isclose(x[tdim - 1], 0))
    ind_cg, ind_dg = compute_dof_permutations(V_ufl, V_cuas, gap, [facet_dg], facets_cg)

    # Compare rhs
    assert np.allclose(b0.array[ind_dg], b1.array[ind_cg])

    # create scipy matrix
    ai, aj, av = A0.getValuesCSR()
    A_sp = scipy.sparse.csr_matrix((av, aj, ai), shape=A0.getSize()).todense()
    bi, bj, bv = A1.getValuesCSR()
    B_sp = scipy.sparse.csr_matrix((bv, bj, bi), shape=A1.getSize()).todense()

    assert np.allclose(A_sp[ind_dg, ind_dg], B_sp[ind_cg, ind_cg])

    # Sanity check different formulations
    if not tied:
        # Contact terms formulated using ufl consistent with nitsche_ufl.py
        F2 = DG_rhs_minus(u0, v0, h, n, gamma_scaled, theta, sigma, gap, dS)

        F2 = _fem.form(F2)
        b2 = _fem.petsc.create_vector(F2)
        b2.zeroEntries()
        _fem.petsc.assemble_vector(b2, F2)
        assert np.allclose(b1.array[ind_cg], b2.array[ind_dg])

        # Contact terms formulated using ufl consistent with nitsche_ufl.py
        J2 = DG_jac_minus(u0, v0, w0, h, n, gamma_scaled, theta, sigma, gap, dS)
        J2 = _fem.form(J2)
        A2 = _fem.petsc.create_matrix(J2)
        A2.zeroEntries()
        _fem.petsc.assemble_matrix(A2, J2)
        A2.assemble()

        ci, cj, cv = A2.getValuesCSR()
        C_sp = scipy.sparse.csr_matrix((cv, cj, ci), shape=A2.getSize()).todense()
        assert np.allclose(C_sp[ind_dg, ind_dg], B_sp[ind_cg, ind_cg])
