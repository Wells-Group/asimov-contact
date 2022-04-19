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
from dolfinx.mesh import (CellType, locate_entities_boundary, locate_entities, create_mesh,
                          compute_midpoints, meshtags)
from mpi4py import MPI

import dolfinx_cuas
import dolfinx_contact
import dolfinx_contact.cpp
from dolfinx_contact.helpers import (R_minus, epsilon, lame_parameters, sigma_func)

kt = dolfinx_contact.cpp.Kernel


def dR_minus(x):
    return 0.5 * (ufl.sign(x) - 1.0)


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
        cells[0].append(cell)
        cell_midpoints = compute_midpoints(mesh, tdim, [cell])
        if cell_midpoints[0][tdim - 1] > 0:
            contact_facets1.append(facet)
    contact_facets2 = []
    for facet in facets2:
        cell = f_to_c.links(facet)[0]
        cells[1].append(cell)
        cell_midpoints = compute_midpoints(mesh, tdim, [cell])
        if cell_midpoints[0][tdim - 1] < -gap:
            contact_facets2.append(facet)

    return cells, [contact_facets1, contact_facets2]


def create_contact_data(V, u, quadrature_degree, lmbda, mu, facets_cg):
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

    # create contact class
    contact = dolfinx_contact.cpp.Contact(facet_marker, [0, 1], V._cpp_object)
    contact.set_quadrature_degree(quadrature_degree)
    contact.create_distance_map(0, 1)
    contact.create_distance_map(1, 0)

    # Pack material parameters mu and lambda on each contact surface
    V2 = _fem.FunctionSpace(mesh, ("DG", 0))
    lmbda2 = _fem.Function(V2)
    lmbda2.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))
    mu2 = _fem.Function(V2)
    mu2.interpolate(lambda x: np.full((1, x.shape[1]), mu))

    # compute active entities
    integral = _fem.IntegralType.exterior_facet
    entities_0 = dolfinx_cuas.compute_active_entities(mesh, facets_cg[0], integral)
    entities_1 = dolfinx_cuas.compute_active_entities(mesh, facets_cg[1], integral)

    # pack coeffs mu, lambda
    material_0 = dolfinx_cuas.pack_coefficients([mu2, lmbda2], entities_0)
    material_1 = dolfinx_cuas.pack_coefficients([mu2, lmbda2], entities_1)

    # Pack cell diameter on each surface
    h = ufl.CellDiameter(mesh)
    surface_cells = np.unique(np.hstack([entities_0[:, 0], entities_1[:, 0]]))
    h_int = _fem.Function(V2)
    expr = _fem.Expression(h, V2.element.interpolation_points)
    h_int.interpolate(expr, surface_cells)
    h_0 = dolfinx_cuas.pack_coefficients([h_int], entities_0)
    h_1 = dolfinx_cuas.pack_coefficients([h_int], entities_1)

    # Pack gap, normals and test functions on each surface
    gap_0 = contact.pack_gap(0)
    n_0 = contact.pack_ny(0, gap_0)
    test_fn_0 = contact.pack_test_functions(0, gap_0)
    gap_1 = contact.pack_gap(1)
    n_1 = contact.pack_ny(1, gap_1)
    test_fn_1 = contact.pack_test_functions(1, gap_1)

    # Concatenate all coeffs
    coeff_0 = np.hstack([material_0, h_0, gap_0, n_0, test_fn_0])
    coeff_1 = np.hstack([material_1, h_1, gap_1, n_1, test_fn_1])

    # pack u
    u_opp_0 = contact.pack_u_contact(0, u._cpp_object, gap_0)
    u_opp_1 = contact.pack_u_contact(1, u._cpp_object, gap_1)
    u_0 = dolfinx_cuas.pack_coefficients([u], entities_0)
    u_1 = dolfinx_cuas.pack_coefficients([u], entities_1)
    c_0 = np.hstack([coeff_0, u_0, u_opp_0])
    c_1 = np.hstack([coeff_1, u_1, u_opp_1])

    return contact, c_0, c_1


@ pytest.mark.parametrize("ct", ["quadrilateral", "triangle", "tetrahedron", "hexahedron"])
@ pytest.mark.parametrize("gap", [1e-13, -1e-13, -0.5])
@ pytest.mark.parametrize("q_deg", [1, 2, 3])
def test_unbiased_rhs(ct, gap, q_deg):

    # set quadrature degree
    quadrature_degree = q_deg

    # Compute lame parameters
    plane_strain = False
    E = 1e3
    nu = 0.1
    mu_func, lambda_func = lame_parameters(plane_strain)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    # Nitche parameters and variables
    theta = 1
    gamma = 10

    # create meshes and function spaces
    V_ufl, V_cuas = create_functionspaces(ct, gap)
    mesh_ufl = V_ufl.mesh
    mesh_cuas = V_cuas.mesh
    tdim = mesh_ufl.topology.dim
    gdim = mesh_ufl.geometry.dim

    def _u0(x):
        values = np.zeros((gdim, x.shape[1]))
        for i in range(tdim):
            for j in range(x.shape[1]):
                values[i, j] = np.sin(x[i, j])
        return values

    # DG ufl 'contact'
    u0 = _fem.Function(V_ufl)
    u0.interpolate(_u0)
    v0 = ufl.TestFunction(V_ufl)
    metadata = {"quadrature_degree": quadrature_degree}
    dS = ufl.Measure("dS", domain=mesh_ufl, metadata=metadata)

    n = ufl.FacetNormal(mesh_ufl)

    # Scaled Nitsche parameter
    h = ufl.CellDiameter(mesh_ufl)
    gamma_scaled = gamma * E

    # Contact terms formulated using ufl
    F = -0.5 * 1 / (gamma_scaled / h('+')) * R_minus(ufl.dot(sigma(u0('+')) * n('+'), n('+'))
                                                     + (gamma_scaled / h('+'))
                                                     * (gap - ufl.dot(u0('-') - u0('+'), n('+'))))\
        * (theta * ufl.dot(sigma(v0('+')) * n('+'), n('-'))
           - (gamma_scaled / h('+')) * ufl.dot(v0('+') - v0('-'), n('-'))) * \
        dS

    F += -0.5 * 1 / (gamma_scaled / h('-')) * R_minus(ufl.dot(sigma(u0('-')) * n('-'), n('-'))
                                                      + (gamma_scaled / h('-'))
                                                      * (gap - ufl.dot(u0('+') - u0('-'), n('-')))) \
        * (theta * ufl.dot(sigma(v0('-')) * n('-'), n('+'))
           - (gamma_scaled / h('-')) * ufl.dot(v0('-') - v0('+'), n('+'))) * \
        dS
    F = _fem.form(F)
    b = _fem.petsc.create_vector(F)
    b.zeroEntries()
    _fem.petsc.assemble_vector(b, F)

    # Custom assembly
    cells, facets_cg = locate_contact_facets_cuas(V_cuas, gap)

    # fem functions

    def _u1(x):
        values = np.zeros((gdim, x.shape[1]))
        for i in range(tdim):
            for j in range(x.shape[1]):
                if i == tdim - 1:
                    values[i, j] = np.sin(x[i, j] + gap)
                else:
                    values[i, j] = np.sin(x[i, j])
        return values

    u1 = _fem.Function(V_cuas)
    v1 = ufl.TestFunction(V_cuas)

    u1.interpolate(_u0, cells[0])
    u1.interpolate(_u1, cells[1])

    # dummy form for creating vector/matrix
    dx = ufl.Measure("dx", domain=mesh_cuas)
    F_cuas = ufl.inner(sigma(u1), epsilon(v1)) * dx

    contact, c_0, c_1 = create_contact_data(V_cuas, u1, q_deg, lmbda, mu, facets_cg)
    # Generate residual data structures
    F_cuas = _fem.form(F)
    kernel_rhs = contact.generate_kernel(kt.Rhs)
    b2 = _fem.petsc.create_vector(F_cuas)

    # pack constants
    consts = np.array([gamma_scaled, theta])

    # assemble  residual
    b2.zeroEntries()
    contact.assemble_vector(b2, 0, kernel_rhs, c_0, consts)
    contact.assemble_vector(b2, 1, kernel_rhs, c_1, consts)

    tdim = mesh_ufl.topology.dim
    facet_dg = locate_entities(mesh_ufl, tdim - 1, lambda x: np.isclose(x[tdim - 1], 0))
    ind_cg, ind_dg = compute_dof_permutations(V_ufl, V_cuas, gap, [facet_dg], facets_cg)

    assert(np.allclose(b2.array[ind_cg], b.array[ind_dg]))


@ pytest.mark.parametrize("ct", ["quadrilateral", "triangle", "tetrahedron", "hexahedron"])
@ pytest.mark.parametrize("gap", [1e-13, -1e-13, -0.5])
@ pytest.mark.parametrize("q_deg", [1, 2, 3])
def test_unbiased_jac(ct, gap, q_deg):

    # set quadrature degree
    quadrature_degree = q_deg

    # Compute lame parameters
    plane_strain = False
    E = 1e3
    nu = 0.1
    mu_func, lambda_func = lame_parameters(plane_strain)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)
    sigma = sigma_func(mu, lmbda)

    # Nitche parameters and variables
    theta = 1
    gamma = 10

    # create meshes and function spaces
    V_ufl, V_cuas = create_functionspaces(ct, gap)
    mesh_ufl = V_ufl.mesh
    mesh_cuas = V_cuas.mesh
    tdim = mesh_ufl.topology.dim
    gdim = mesh_ufl.geometry.dim

    def _u0(x):
        values = np.zeros((gdim, x.shape[1]))
        for i in range(tdim):
            for j in range(x.shape[1]):
                values[i, j] = np.sin(x[i, j])
        return values

    # DG ufl 'contact'
    u0 = _fem.Function(V_ufl)
    u0.interpolate(_u0)

    v0 = ufl.TestFunction(V_ufl)
    w0 = ufl.TrialFunction(V_ufl)
    metadata = {"quadrature_degree": quadrature_degree}
    dS = ufl.Measure("dS", domain=mesh_ufl, metadata=metadata)

    n = ufl.FacetNormal(mesh_ufl)

    # Scaled Nitsche parameter
    h = ufl.CellDiameter(mesh_ufl)
    gamma_scaled = gamma * E

    # Contact terms formulated using ufl
    qp = ufl.dot(sigma(u0('+')) * n('+'), n('+')) + (gamma_scaled / h('+')) * (gap - ufl.dot(u0('-') - u0('+'), n('+')))
    J = -0.5 * 1 / (gamma_scaled / h('+')) * dR_minus(qp)\
        * (ufl.dot(sigma(w0('+')) * n('+'), n('-')) - (gamma_scaled / h('+')) * (ufl.dot(w0('-') - w0('+'), n('+'))))\
        * (theta * ufl.dot(sigma(v0('+')) * n('+'), n('-'))
           - (gamma_scaled / h('+')) * ufl.dot(v0('+') - v0('-'), n('-'))) * \
        dS

    qm = ufl.dot(sigma(u0('-')) * n('-'), n('-')) + (gamma_scaled / h('-')) * (gap - ufl.dot(u0('+') - u0('-'), n('-')))
    J -= 0.5 * 1 / (gamma_scaled / h('-')) * dR_minus(qm)\
        * (ufl.dot(sigma(w0('-')) * n('-'), n('+')) - (gamma_scaled / h('-')) * (ufl.dot(w0('+') - w0('-'), n('-'))))\
        * (theta * ufl.dot(sigma(v0('-')) * n('-'), n('+'))
           - (gamma_scaled / h('-')) * ufl.dot(v0('-') - v0('+'), n('+'))) * \
        dS
    J = _fem.form(J)
    A = _fem.petsc.create_matrix(J)
    A.zeroEntries()
    _fem.petsc.assemble_matrix(A, J)
    A.assemble()

    # Custom assembly
    cells, facets_cg = locate_contact_facets_cuas(V_cuas, gap)

    # fem functions
    def _u1(x):
        values = np.zeros((gdim, x.shape[1]))
        for i in range(tdim):
            for j in range(x.shape[1]):
                if i == tdim - 1:
                    values[i, j] = np.sin(x[i, j] + gap)
                else:
                    values[i, j] = np.sin(x[i, j])
        return values

    u1 = _fem.Function(V_cuas)
    v1 = ufl.TestFunction(V_cuas)
    w1 = ufl.TrialFunction(V_cuas)

    u1.interpolate(_u0, cells[0])
    u1.interpolate(_u1, cells[1])
    # dummy form for creating vector/matrix
    dx = ufl.Measure("dx", domain=mesh_cuas)
    J_cuas = ufl.inner(sigma(w1), epsilon(v1)) * dx

    contact, c_0, c_1 = create_contact_data(V_cuas, u1, q_deg, lmbda, mu, facets_cg)
    # Generate residual data structures
    J_cuas = _fem.form(J_cuas)
    kernel_jac = contact.generate_kernel(kt.Jac)
    A2 = contact.create_matrix(J_cuas)

    # pack constants
    consts = np.array([gamma_scaled, theta])

    # assemble  jacobian
    A2.zeroEntries()
    contact.assemble_matrix(A2, [], 0, kernel_jac, c_0, consts)
    contact.assemble_matrix(A2, [], 1, kernel_jac, c_1, consts)
    A2.assemble(0)

    tdim = mesh_ufl.topology.dim
    facet_dg = locate_entities(mesh_ufl, tdim - 1, lambda x: np.isclose(x[tdim - 1], 0))
    ind_cg, ind_dg = compute_dof_permutations(V_ufl, V_cuas, gap, [facet_dg], facets_cg)

    # create scipy matrix
    ai, aj, av = A.getValuesCSR()
    A_sp = scipy.sparse.csr_matrix((av, aj, ai), shape=A.getSize()).todense()
    bi, bj, bv = A2.getValuesCSR()
    B_sp = scipy.sparse.csr_matrix((bv, bj, bi), shape=A2.getSize()).todense()

    assert(np.allclose(A_sp[ind_dg, ind_dg], B_sp[ind_cg, ind_cg]))
