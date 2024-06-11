# Copyright (C) 2021-2024 Sarah Roggendorf and Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT
#
# This tests the custom assembly for the unbiased Nitsche formulation in
# a special case that can be expressed using ufl:
#
# We consider a very simple test case made up of two disconnected
# elements with a constant gap in x[tdim-1]-direction. The contact
# surfaces are made up of exactly one edge from each element that are
# perfectly aligned such that the quadrature points only differ in the
# x[tdim-1]-direction by the given gap. For comparison, we consider a DG
# function space on a mesh that is constructed by removing the gap
# between the elements and merging the edges making up the contact
# surface into one. This allows us to use DG-functions and ufl to
# formulate the contact terms in the variational form by suitably
# adjusting the deformation u and using the given constant gap.


from mpi4py import MPI

import dolfinx.fem as _fem
import numpy as np
import numpy.typing as npt
import pytest
import scipy
import ufl
from basix.ufl import element
from dolfinx.cpp.mesh import to_type
from dolfinx.graph import adjacencylist
from dolfinx.mesh import (
    CellType,
    Mesh,
    compute_midpoints,
    create_mesh,
    locate_entities,
    locate_entities_boundary,
    meshtags,
)
from dolfinx_contact.cpp import ContactMode, Kernel, MeshTie, Problem
from dolfinx_contact.general_contact.contact_problem import ContactProblem, FrictionLaw
from dolfinx_contact.helpers import (
    R_minus,
    R_plus,
    ball_projection,
    d_alpha_ball_projection,
    d_ball_projection,
    dR_minus,
    dR_plus,
    epsilon,
    lame_parameters,
    sigma_func,
    tangential_proj,
)

kt = Kernel


def DG_rhs_plus(u0, v0, h, n, gamma, theta, sigma, gap, dS):
    # This version of the ufl form agrees with the formulation in
    # https://doi.org/10.1007/s00211-018-0950-x
    def Pn_g(u, a, b):
        return ufl.dot(u(a) - u(b), -n(b)) - gap - (h(a) / gamma) * ufl.dot(sigma(u(a)) * n(a), -n(b))

    def Pn_gtheta(v, a, b):
        return ufl.dot(v(a) - v(b), -n(b)) - theta * (h(a) / gamma) * ufl.dot(sigma(v(a)) * n(a), -n(b))

    F = (
        0.5 * (gamma / h("+")) * R_plus(Pn_g(u0, "+", "-")) * Pn_gtheta(v0, "+", "-") * dS
        - 0.5
        * (h("+") / gamma)
        * theta
        * ufl.dot(sigma(u0("+")) * n("+"), -n("-"))
        * ufl.dot(sigma(v0("+")) * n("+"), -n("-"))
        * dS
    )

    F += (
        0.5 * (gamma / h("-")) * R_plus(Pn_g(u0, "-", "+")) * Pn_gtheta(v0, "-", "+") * dS
        - 0.5
        * (h("-") / gamma)
        * theta
        * ufl.dot(sigma(u0("-")) * n("-"), -n("+"))
        * ufl.dot(sigma(v0("-")) * n("-"), -n("+"))
        * dS
    )

    return F


def DG_rhs_minus(u0, v0, h, n, gamma, theta, sigma, gap, dS):
    # This version of the ufl form agrees with its one-sided equivalent
    # in nitsche_ufl.py
    def Pn_g(u, a, b):
        return ufl.dot(sigma(u(a)) * n(a), -n(b)) + (gamma / h(a)) * (gap - ufl.dot(u(a) - u(b), -n(b)))

    def Pn_gtheta(v, a, b):
        return theta * ufl.dot(sigma(v(a)) * n(a), -n(b)) - (gamma / h(a)) * ufl.dot(v(a) - v(b), -n(b))

    F = (
        0.5 * (h("+") / gamma) * R_minus(Pn_g(u0, "+", "-")) * Pn_gtheta(v0, "+", "-") * dS
        - 0.5
        * (h("+") / gamma)
        * theta
        * ufl.dot(sigma(u0("+")) * n("+"), -n("-"))
        * ufl.dot(sigma(v0("+")) * n("+"), -n("-"))
        * dS
    )

    F += (
        0.5 * (h("-") / gamma) * R_minus(Pn_g(u0, "-", "+")) * Pn_gtheta(v0, "-", "+") * dS
        - 0.5
        * (h("-") / gamma)
        * theta
        * ufl.dot(sigma(u0("-")) * n("-"), -n("+"))
        * ufl.dot(sigma(v0("-")) * n("-"), -n("+"))
        * dS
    )

    return F


def DG_jac_plus(u0, v0, w0, h, n, gamma, theta, sigma, gap, dS):
    # This version of the ufl form agrees with the formulation in
    # https://doi.org/10.1007/s00211-018-0950-x
    def Pn_g(u, a, b):
        return ufl.dot(u(a) - u(b), -n(b)) - gap - (h(a) / gamma) * ufl.dot(sigma(u(a)) * n(a), -n(b))

    def Pn_gtheta(v, a, b, t):
        return ufl.dot(v(a) - v(b), -n(b)) - t * (h(a) / gamma) * ufl.dot(sigma(v(a)) * n(a), -n(b))

    J = (
        0.5
        * (gamma / h("+"))
        * dR_plus(Pn_g(u0, "+", "-"))
        * Pn_gtheta(w0, "+", "-", 1.0)
        * Pn_gtheta(v0, "+", "-", theta)
        * dS
        - 0.5
        * (h("+") / gamma)
        * theta
        * ufl.dot(sigma(w0("+")) * n("+"), -n("-"))
        * ufl.dot(sigma(v0("+")) * n("+"), -n("-"))
        * dS
    )

    J += (
        0.5
        * (gamma / h("-"))
        * dR_plus(Pn_g(u0, "-", "+"))
        * Pn_gtheta(w0, "-", "+", 1.0)
        * Pn_gtheta(v0, "-", "+", theta)
        * dS
        - 0.5
        * (h("-") / gamma)
        * theta
        * ufl.dot(sigma(w0("-")) * n("-"), -n("+"))
        * ufl.dot(sigma(v0("-")) * n("-"), -n("+"))
        * dS
    )

    return J


def DG_jac_minus(u0, v0, w0, h, n, gamma, theta, sigma, gap, dS):
    # This version of the ufl form agrees with its one-sided equivalent
    # in nitsche_ufl.py
    def Pn_g(u, a, b):
        return ufl.dot(sigma(u(a)) * n(a), -n(b)) + (gamma / h(a)) * (gap - ufl.dot(u(a) - u(b), -n(b)))

    def Pn_gtheta(v, a, b, t):
        return t * ufl.dot(sigma(v(a)) * n(a), -n(b)) - (gamma / h(a)) * ufl.dot(v(a) - v(b), -n(b))

    J = (
        0.5
        * (h("+") / gamma)
        * dR_minus(Pn_g(u0, "+", "-"))
        * Pn_gtheta(w0, "+", "-", 1.0)
        * Pn_gtheta(v0, "+", "-", theta)
        * dS
        - 0.5
        * (h("+") / gamma)
        * theta
        * ufl.dot(sigma(w0("+")) * n("+"), -n("-"))
        * ufl.dot(sigma(v0("+")) * n("+"), -n("-"))
        * dS
    )

    J += (
        0.5
        * (h("-") / gamma)
        * dR_minus(Pn_g(u0, "-", "+"))
        * Pn_gtheta(w0, "-", "+", 1.0)
        * Pn_gtheta(v0, "-", "+", theta)
        * dS
        - 0.5
        * (h("-") / gamma)
        * theta
        * ufl.dot(sigma(w0("-")) * n("-"), -n("+"))
        * ufl.dot(sigma(v0("-")) * n("-"), -n("+"))
        * dS
    )

    return J


def DG_rhs_tresca(u0, v0, h, n, gamma, theta, sigma, fric, dS, gdim):
    """UFL version of the Tresca friction term for the unbiased Nitsche formulation."""

    def pt_g(u, a, b, c):
        return tangential_proj(u(a) - u(b) - h(a) * c * sigma(u(a)) * n(a), -n(b))

    def pt_sig(u, a, b):
        return tangential_proj(sigma(u(a)) * n(a), -n(b))

    return (
        0.5
        * gamma
        / h("+")
        * ufl.dot(
            ball_projection(pt_g(u0, "+", "-", 1.0 / gamma), fric * h("+") / gamma, gdim),
            pt_g(v0, "+", "-", theta / gamma),
        )
        * dS
        + 0.5
        * gamma
        / h("-")
        * ufl.dot(
            ball_projection(pt_g(u0, "-", "+", 1.0 / gamma), fric * h("-") / gamma, gdim),
            pt_g(v0, "-", "+", theta / gamma),
        )
        * dS
        - 0.5 * (h("+") / gamma) * theta * ufl.dot(pt_sig(u0, "+", "-"), pt_sig(v0, "+", "-")) * dS
        - 0.5 * (h("-") / gamma) * theta * ufl.dot(pt_sig(u0, "-", "+"), pt_sig(v0, "-", "+")) * dS
    )


def DG_jac_tresca(u0, v0, w0, h, n, gamma, theta, sigma, fric, dS, gdim):
    """UFL version of the Jacobian for the Tresca friction term for the unbiased Nitsche formulation."""

    def pt_g(u, a, b, c):
        return tangential_proj(u(a) - u(b) - h(a) * c * sigma(u(a)) * n(a), -n(b))

    def pt_sig(u, a, b):
        return tangential_proj(sigma(u(a)) * n(a), -n(b))

    J = (
        0.5
        * gamma
        / h("+")
        * ufl.dot(
            d_ball_projection(pt_g(u0, "+", "-", 1.0 / gamma), fric * h("+") / gamma, gdim)
            * pt_g(w0, "+", "-", 1.0 / gamma),
            pt_g(v0, "+", "-", theta / gamma),
        )
        * dS
        - 0.5 * (h("+") / gamma) * theta * ufl.dot(pt_sig(w0, "+", "-"), pt_sig(v0, "+", "-")) * dS
    )
    J += (
        0.5
        * gamma
        / h("-")
        * ufl.dot(
            d_ball_projection(pt_g(u0, "-", "+", 1.0 / gamma), fric * h("-") / gamma, gdim)
            * pt_g(w0, "-", "+", 1.0 / gamma),
            pt_g(v0, "-", "+", theta / gamma),
        )
        * dS
        - 0.5 * (h("-") / gamma) * theta * ufl.dot(pt_sig(w0, "-", "+"), pt_sig(v0, "-", "+")) * dS
    )

    return J


def DG_rhs_coulomb(u0, v0, h, n, gamma, theta, sigma, gap, fric, dS, gdim):
    """UFL version of the Coulomb friction term for the unbiased Nitsche formulation."""

    def Pn_g(u, a, b):
        return ufl.dot(u(a) - u(b), -n(b)) - gap - (h(a) / gamma) * ufl.dot(sigma(u(a)) * n(a), -n(b))

    def pt_g(u, a, b, c):
        return tangential_proj(u(a) - u(b) - h(a) * c * sigma(u(a)) * n(a), -n(b))

    def pt_sig(u, a, b):
        return tangential_proj(sigma(u(a)) * n(a), -n(b))

    Pn_u_plus = R_plus(Pn_g(u0, "+", "-"))
    Pn_u_minus = R_plus(Pn_g(u0, "-", "+"))
    return (
        0.5
        * gamma
        / h("+")
        * ufl.dot(
            ball_projection(pt_g(u0, "+", "-", 1.0 / gamma), Pn_u_plus * fric, gdim),
            pt_g(v0, "+", "-", theta / gamma),
        )
        * dS
        + 0.5
        * gamma
        / h("-")
        * ufl.dot(
            ball_projection(pt_g(u0, "-", "+", 1.0 / gamma), Pn_u_minus * fric, gdim),
            pt_g(v0, "-", "+", theta / gamma),
        )
        * dS
        - 0.5 * (h("+") / gamma) * theta * ufl.dot(pt_sig(u0, "+", "-"), pt_sig(v0, "+", "-")) * dS
        - 0.5 * (h("-") / gamma) * theta * ufl.dot(pt_sig(u0, "-", "+"), pt_sig(v0, "-", "+")) * dS
    )


def DG_jac_coulomb(u0, v0, w0, h, n, gamma, theta, sigma, gap, fric, dS, gdim):
    """UFL version of the Jacobian for the Coulomb friction term for the unbiased Nitsche formulation,"""

    def Pn_g(u, a, b):
        return ufl.dot(u(a) - u(b), -n(b)) - gap - (h(a) / gamma) * ufl.dot(sigma(u(a)) * n(a), -n(b))

    def Pn_gtheta(v, a, b, t):
        return ufl.dot(v(a) - v(b), -n(b)) - t * (h(a) / gamma) * ufl.dot(sigma(v(a)) * n(a), -n(b))

    def pt_g(u, a, b, c):
        return tangential_proj(u(a) - u(b) - h(a) * c * sigma(u(a)) * n(a), -n(b))

    def pt_sig(u, a, b):
        return tangential_proj(sigma(u(a)) * n(a), -n(b))

    Pn_u_plus = R_plus(Pn_g(u0, "+", "-"))
    Pn_u_minus = R_plus(Pn_g(u0, "-", "+"))

    J = (
        0.5
        * gamma
        / h("+")
        * ufl.dot(
            d_ball_projection(pt_g(u0, "+", "-", 1.0 / gamma), Pn_u_plus * fric, gdim)
            * pt_g(w0, "+", "-", 1.0 / gamma),
            pt_g(v0, "+", "-", theta / gamma),
        )
        * dS
        - 0.5 * (h("+") / gamma) * theta * ufl.dot(pt_sig(w0, "+", "-"), pt_sig(v0, "+", "-")) * dS
    )
    J += (
        0.5
        * gamma
        / h("-")
        * ufl.dot(
            d_ball_projection(pt_g(u0, "-", "+", 1.0 / gamma), Pn_u_minus * fric, gdim)
            * pt_g(w0, "-", "+", 1.0 / gamma),
            pt_g(v0, "-", "+", theta / gamma),
        )
        * dS
        - 0.5 * (h("-") / gamma) * theta * ufl.dot(pt_sig(w0, "-", "+"), pt_sig(v0, "-", "+")) * dS
    )

    d_alpha_plus = d_alpha_ball_projection(
        pt_g(u0, "+", "-", 1.0 / gamma),
        Pn_u_plus * fric,
        dR_plus(Pn_g(u0, "+", "-")) * fric,
        gdim,
    )
    d_alpha_minus = d_alpha_ball_projection(
        pt_g(u0, "-", "+", 1.0 / gamma),
        Pn_u_plus * fric,
        dR_plus(Pn_g(u0, "-", "+")) * fric,
        gdim,
    )
    J += (
        0.5
        * gamma
        / h("+")
        * Pn_gtheta(w0, "+", "-", 1.0)
        * ufl.dot(d_alpha_plus, pt_g(v0, "+", "-", theta / gamma))
        * dS
    )
    J += (
        0.5
        * gamma
        / h("-")
        * Pn_gtheta(w0, "-", "+", 1.0)
        * ufl.dot(d_alpha_minus, pt_g(v0, "-", "+", theta / gamma))
        * dS
    )
    return J


def compute_dof_permutations_all(V_dg, V_cg, gap):
    """The meshes used for the two different formulations are
    created independently of each other. Therefore we need to
    determine how to map the dofs from one mesh to the other in
    order to compare the results.
    """
    mesh_dg = V_dg.mesh
    mesh_cg = V_cg.mesh
    bs = V_cg.dofmap.index_map_bs
    tdim = mesh_dg.topology.dim
    mesh_dg.topology.create_connectivity(tdim - 1, tdim)
    mesh_dg.topology.create_connectivity(tdim, tdim)

    mesh_cg.topology.create_connectivity(tdim - 1, tdim)
    mesh_cg.topology.create_connectivity(tdim, tdim - 1)
    x_cg = V_cg.tabulate_dof_coordinates()
    x_dg = V_dg.tabulate_dof_coordinates()

    # retrieve all dg dofs on mesh without gap for each cell
    # and modify coordinates by gap if necessary
    num_cells = mesh_dg.topology.index_map(tdim).size_local
    for cell in range(num_cells):
        midpoint = compute_midpoints(mesh_dg, tdim, np.array([cell]))[0]
        if midpoint[tdim - 1] <= 0:
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


def create_meshes(ct: str, gap: float, xdtype: npt.DTypeLike = np.float64) -> tuple[Mesh, Mesh]:
    """Create meshes for the two different formulations.

    This is a helper function to create the two element function
    spaces both for custom assembly and the DG formulation for quads,
    triangles, hexes and tetrahedra.

    Args:
        ct: The cell-type of the mesh
        gap: Gap between the two elements in `tdim-1` direction.
        xdtype: Data type for mesh coordinates

    Note:
        The triangular grid is a flat manifold with geometrical dimension 3.

    Note:
        The gap between two elements can be negative

    Returns:
        Two meshes, (standard, custom) where the standard mesh is two elements
        glued together at the contact surface. The custom mesh consists of two elements separate
        by a distance `gap` in the `topological dimension - 1` direction.
    """
    assert MPI.COMM_WORLD.size == 1, "This test only supports running in serial"

    cell_type = to_type(ct)
    if cell_type == CellType.quadrilateral:
        x_ufl = np.array([[0, 0], [0.8, 0], [0.1, 1.3], [0.7, 1.2], [-0.1, -1.2], [0.8, -1.1]], dtype=xdtype)
        x_custom = np.array(
            [
                [0, 0],
                [0.8, 0],
                [0.1, 1.3],
                [0.7, 1.2],
                [0, -gap],
                [0.8, -gap],
                [-0.1, -1.2 - gap],
                [0.8, -1.1 - gap],
            ]
        )
        cells_ufl = np.array([[0, 1, 2, 3], [4, 5, 0, 1]], dtype=np.int64)
        cells_custom = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    elif cell_type == CellType.triangle:
        x_ufl = np.array([[0, 0, 0], [0.8, 0, 0], [0.3, 1.3, 0.0], [0.4, -1.2, 0.0]], dtype=xdtype)
        x_custom = np.array(
            [
                [0, 0, 0],
                [0.8, 0, 0],
                [0.3, 1.3, 0.0],
                [0, -gap, 0],
                [0.8, -gap, 0],
                [0.4, -1.2 - gap, 0.0],
            ],
            dtype=xdtype,
        )
        cells_ufl = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)
        cells_custom = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    elif cell_type == CellType.tetrahedron:
        x_ufl = np.array([[0, 0, 0], [1.1, 0, 0], [0.3, 1.0, 0], [1, 1.2, 1.5], [0.8, 1.2, -1.6]], dtype=xdtype)
        x_custom = np.array(
            [
                [0, 0, 0],
                [1.1, 0, 0],
                [0.3, 1.0, 0],
                [1, 1.2, 1.5],
                [0, 0, -gap],
                [1.1, 0, -gap],
                [0.3, 1.0, -gap],
                [0.8, 1.2, -1.6 - gap],
            ],
            dtype=xdtype,
        )
        cells_ufl = np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int64)
        cells_custom = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
    elif cell_type == CellType.hexahedron:
        x_ufl = np.array(
            [
                [0, 0, 0],
                [1.1, 0, 0],
                [0.1, 1, 0],
                [1, 1.2, 0],
                [0, 0, 1.2],
                [1.0, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
                [0, 0, -1.2],
                [1.0, 0, -1.3],
                [0, 1, -1],
                [1, 1, -1],
            ],
            dtype=xdtype,
        )
        x_custom = np.array(
            [
                [0, 0, 0],
                [1.1, 0, 0],
                [0.1, 1, 0],
                [1, 1.2, 0],
                [0, 0, 1.2],
                [1.0, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
                [0, 0, -1.2 - gap],
                [1.0, 0, -1.3 - gap],
                [0, 1, -1 - gap],
                [1, 1, -1 - gap],
                [0, 0, -gap],
                [1.1, 0, -gap],
                [0.1, 1, -gap],
                [1, 1.2, -gap],
            ],
            dtype=xdtype,
        )
        cells_ufl = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 0, 1, 2, 3]], dtype=np.int64)
        cells_custom = np.array([[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported mesh type {ct}")

    coord_el = element("Lagrange", cell_type.name, 1, shape=(x_ufl.shape[1],))
    mesh_ufl = create_mesh(MPI.COMM_WORLD, cells_ufl, x_ufl, e=coord_el)
    mesh_custom = create_mesh(MPI.COMM_WORLD, cells_custom, x_custom, e=coord_el)

    return mesh_ufl, mesh_custom


def locate_contact_facets_custom(V, gap):
    """This function locates the contact facets for custom assembly and ensures
    that the correct facet is chosen if the gap is zero"""
    # Retrieve mesh
    mesh = V.mesh

    # locate facets
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, tdim)
    facets1 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[tdim - 1], 0))
    facets2 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[tdim - 1], -gap))

    # choose correct facet if gap is zero
    mesh.topology.create_connectivity(tdim - 1, tdim)
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    cells = [[], []]
    contact_facets1 = []
    for facet in facets1:
        cell = f_to_c.links(facet)[0]
        cell_midpoints = compute_midpoints(mesh, tdim, np.asarray([cell], dtype=np.int32))
        if cell_midpoints[0][tdim - 1] > 0:
            contact_facets1.append(facet)
            cells[0].append(cell)
    contact_facets2 = []
    for facet in facets2:
        cell = f_to_c.links(facet)[0]
        cell_midpoints = compute_midpoints(mesh, tdim, np.asarray([cell], dtype=np.int32))
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


@pytest.mark.parametrize(
    "search",
    [
        ContactMode.Raytracing,
        ContactMode.ClosestPoint,
    ],
)
@pytest.mark.parametrize(
    "frictionlaw",
    [
        FrictionLaw.Frictionless,
        FrictionLaw.Coulomb,
        FrictionLaw.Tresca,
    ],
)
@pytest.mark.parametrize(
    "ct",
    [
        "triangle",
        "quadrilateral",
        "tetrahedron",
        "hexahedron",
    ],
)
@pytest.mark.parametrize("gap", [0.5, -0.5])
# @pytest.mark.parametrize("quadrature_degree", [1, 5])
@pytest.mark.parametrize("quadrature_degree", [1, 4])
@pytest.mark.parametrize("theta", [1, 0, -1])
def test_contact_kernels(ct, gap, quadrature_degree, theta, frictionlaw, search):
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
    mesh_ufl, mesh_custom = create_meshes(ct, gap)
    gdim = mesh_ufl.geometry.dim
    V_ufl = _fem.functionspace(mesh_ufl, ("DG", 1, (gdim,)))
    V_custom = _fem.functionspace(mesh_custom, ("Lagrange", 1, (gdim,)))
    tdim = mesh_ufl.topology.dim

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
    # Contact terms formulated using ufl consistent with https://doi.org/10.1007/s00211-018-0950-x
    F0 = DG_rhs_plus(u0, v0, h, n, gamma_scaled, theta, sigma, gap, dS)
    J0 = DG_jac_plus(u0, v0, w0, h, n, gamma_scaled, theta, sigma, gap, dS)
    if frictionlaw == FrictionLaw.Tresca:
        F0 += DG_rhs_tresca(u0, v0, h, n, gamma_scaled, theta, sigma, 0.1, dS, gdim)

        J0 += DG_jac_tresca(u0, v0, w0, h, n, gamma_scaled, theta, sigma, 0.1, dS, gdim)
    elif frictionlaw == FrictionLaw.Coulomb:
        F0 += DG_rhs_coulomb(u0, v0, h, n, gamma_scaled, theta, sigma, gap, 0.1, dS, gdim)
        J0 += DG_jac_coulomb(u0, v0, w0, h, n, gamma_scaled, theta, sigma, gap, 0.1, dS, gdim)

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
    u = _fem.Function(V_custom)

    u1.interpolate(_u0, np.array(cells[0]))
    u1.interpolate(_u2, np.array(cells[1]))
    u1.x.scatter_forward()

    # Dummy form for creating vector/matrix
    dx = ufl.Measure("dx", domain=mesh_custom)
    F_custom = ufl.inner(sigma(u1), epsilon(v1)) * dx
    J_custom = ufl.inner(sigma(w1), epsilon(v1)) * dx

    V0 = _fem.functionspace(mesh_custom, ("DG", 0))
    mu0 = _fem.Function(V0)
    lmbda0 = _fem.Function(V0)
    fric = _fem.Function(V0)
    mu0.interpolate(lambda x: np.full((1, x.shape[1]), mu))
    lmbda0.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))
    fric.interpolate(lambda x: np.full((1, x.shape[1]), 0.1))
    # create meshtags
    facet_marker = create_facet_markers(mesh_custom, facets_cg)
    data = np.array([0, 1], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    surfaces = adjacencylist(data, offsets)
    contact_problem = ContactProblem(
        [facet_marker],
        surfaces,
        [(0, 1), (1, 0)],
        mesh_custom,
        quadrature_degree,
        [search, search],
    )
    contact_problem.generate_contact_data(
        frictionlaw,
        V_custom,
        {"u": u, "du": u1, "mu": mu0, "lambda": lmbda0, "fric": fric},
        E * gamma,
        theta,
    )

    # compiler options to improve performance
    cffi_options = ["-Ofast", "-march=native"]
    jit_options = {"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]}
    # Generate residual data structures
    F_custom = _fem.form(F_custom, jit_options=jit_options)
    b1 = _fem.petsc.create_vector(F_custom)

    # Generate residual data structures
    J_custom = _fem.form(J_custom, jit_options=jit_options)
    A1 = contact_problem.create_matrix(J_custom)

    # Assemble  residual
    b1.zeroEntries()
    contact_problem.assemble_vector(b1, V_custom)

    # Assemble  jacobian
    A1.zeroEntries()
    contact_problem.assemble_matrix(A1, V_custom)
    A1.assemble()

    # Retrieve data necessary for comparison
    tdim = mesh_ufl.topology.dim
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
    if frictionlaw == FrictionLaw.Frictionless:
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
    F = (
        gamma / h("+") * ufl.inner(ufl.jump(u0), ufl.jump(v0)) * dS
        + gamma / h("-") * ufl.inner(ufl.jump(u0), ufl.jump(v0)) * dS
        - ufl.inner(ufl.avg(ufl.grad(u0)), n("+")) * ufl.jump(v0) * dS
        + ufl.inner(ufl.avg(ufl.grad(u0)), n("-")) * ufl.jump(v0) * dS
        - theta * ufl.inner(ufl.avg(ufl.grad(v0)), n("+")) * ufl.jump(u0) * dS
        + theta * ufl.inner(ufl.avg(ufl.grad(v0)), n("-")) * ufl.jump(u0) * dS
    )
    return 0.5 * kdt * F


def tied_dg(u0, v0, h, n, gamma, theta, sigma, dS):
    F = (
        gamma / h("+") * ufl.inner(ufl.jump(u0), ufl.jump(v0)) * dS
        + gamma / h("-") * ufl.inner(ufl.jump(u0), ufl.jump(v0)) * dS
        - ufl.inner(ufl.avg(sigma(u0)) * n("+"), ufl.jump(v0)) * dS
        + ufl.inner(ufl.avg(sigma(u0)) * n("-"), ufl.jump(v0)) * dS
        - theta * ufl.inner(ufl.avg(sigma(v0)) * n("+"), ufl.jump(u0)) * dS
        + theta * ufl.inner(ufl.avg(sigma(v0)) * n("-"), ufl.jump(u0)) * dS
    )
    return 0.5 * F


def tied_dg_T(u0, v0, T0, h, n, gamma, theta, sigma, sigma_T, dS):
    F = (
        gamma / h("+") * ufl.inner(ufl.jump(u0), ufl.jump(v0)) * dS
        + gamma / h("-") * ufl.inner(ufl.jump(u0), ufl.jump(v0)) * dS
        - ufl.inner(ufl.avg(sigma_T(u0, T0)) * n("+"), ufl.jump(v0)) * dS
        + ufl.inner(ufl.avg(sigma_T(u0, T0)) * n("-"), ufl.jump(v0)) * dS
        - theta * ufl.inner(ufl.avg(sigma(v0)) * n("+"), ufl.jump(u0)) * dS
        + theta * ufl.inner(ufl.avg(sigma(v0)) * n("-"), ufl.jump(u0)) * dS
    )
    return 0.5 * F


@pytest.mark.parametrize(
    "ct",
    [
        "triangle",
        "quadrilateral",
        "tetrahedron",
        "hexahedron",
    ],
)
@pytest.mark.parametrize(
    "problem",
    [
        Problem.Poisson,
        Problem.Elasticity,
        Problem.ThermoElasticity,
    ],
)
@pytest.mark.parametrize("gap", [0.5, -0.5])
@pytest.mark.parametrize("quadrature_degree", [1, 3])
@pytest.mark.parametrize("theta", [1, 0, -1])
def test_meshtie_kernels(ct, gap, quadrature_degree, theta, problem):
    # Problem parameters
    kdt = 5
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

    # create meshes and function spaces
    mesh_ufl, mesh_custom = create_meshes(ct, gap)
    gdim = mesh_ufl.geometry.dim
    tdim = mesh_ufl.topology.dim
    TOL = 1e-7
    cells_ufl_0 = locate_entities(mesh_ufl, tdim, lambda x: x[tdim - 1] > 0 - TOL)
    cells_ufl_1 = locate_entities(mesh_ufl, tdim, lambda x: x[tdim - 1] < 0 + TOL)

    if problem == Problem.Poisson:
        V_ufl = _fem.functionspace(mesh_ufl, ("DG", 1))
        V_custom = _fem.functionspace(mesh_custom, ("Lagrange", 1))

        def _u0(x):
            return np.sin(x[0]) + 1

        def _u1(x):
            return np.sin(x[tdim - 1]) + 2

        def _u2(x):
            return np.sin(x[tdim - 1] + gap) + 2
    else:
        V_ufl = _fem.functionspace(mesh_ufl, ("DG", 1, (gdim,)))
        V_custom = _fem.functionspace(mesh_custom, ("Lagrange", 1, (gdim,)))

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
    T0, Q_ufl, Q_custom = None, None, None
    metadata = {"quadrature_degree": quadrature_degree}
    dS = ufl.Measure("dS", domain=mesh_ufl, metadata=metadata)

    # Custom assembly
    cells, facets_cg = locate_contact_facets_custom(V_custom, gap)
    # Fem functions
    u1 = _fem.Function(V_custom)
    v1 = ufl.TestFunction(V_custom)
    w1 = ufl.TrialFunction(V_custom)

    u1.interpolate(_u0, np.array(cells[0]))
    u1.interpolate(_u2, np.array(cells[1]))
    u1.x.scatter_forward()

    n = ufl.FacetNormal(mesh_ufl)

    # Scaled Nitsche parameter
    h = ufl.CellDiameter(mesh_ufl)
    alpha = 0.5

    # DG formulation
    if problem == Problem.Poisson:
        F0 = poisson_dg(u0, v0, h, n, kdt, gamma, theta, dS)
        J0 = poisson_dg(w0, v0, h, n, kdt, gamma, theta, dS)
    elif problem == Problem.Elasticity:
        F0 = tied_dg(u0, v0, h, n, gamma * E, theta, sigma, dS)
        J0 = tied_dg(w0, v0, h, n, gamma * E, theta, sigma, dS)
    elif problem == Problem.ThermoElasticity:
        Q_ufl = _fem.functionspace(mesh_ufl, ("DG", 1))
        Q_custom = _fem.functionspace(mesh_custom, ("Lagrange", 1))
        T0 = _fem.Function(Q_ufl)
        T1 = _fem.Function(Q_custom)
        T0.interpolate(lambda x: np.sin(x[0]) + 1, cells_ufl_0)
        T0.interpolate(lambda x: np.sin(x[tdim - 1]) + 2, cells_ufl_1)
        T1.interpolate(lambda x: np.sin(x[0]) + 1, np.array(cells[0]))
        T1.interpolate(lambda x: np.sin(x[tdim - 1] + gap) + 2, np.array(cells[1]))

        def sigma_T(w, T):
            return sigma(w) - alpha * (3 * lmbda + 2 * mu) * T * ufl.Identity(gdim)

        F0 = tied_dg_T(u0, v0, T0, h, n, gamma * E, theta, sigma, sigma_T, dS)
        J0 = tied_dg(w0, v0, h, n, gamma * E, theta, sigma, dS)

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

    V0 = _fem.functionspace(mesh_custom, ("DG", 0))
    kdt_custom = _fem.Function(V0)
    kdt_custom.interpolate(lambda x: np.full((1, x.shape[1]), kdt))
    lmbda_custom = _fem.Function(V0)
    lmbda_custom.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))
    mu_custom = _fem.Function(V0)
    mu_custom.interpolate(lambda x: np.full((1, x.shape[1]), mu))
    alpha_custom = _fem.Function(V0)
    alpha_custom.interpolate(lambda x: np.full((1, x.shape[1]), alpha))

    if problem == Problem.Poisson:
        coeffs = {"T": u1._cpp_object, "kdt": kdt_custom._cpp_object}
    elif problem == Problem.Elasticity:
        coeffs = {
            "u": u1._cpp_object,
            "mu": mu_custom._cpp_object,
            "lambda": lmbda_custom._cpp_object,
        }
        gamma = gamma * E
    else:
        coeffs = {
            "u": u1._cpp_object,
            "T": T1._cpp_object,
            "mu": mu_custom._cpp_object,
            "lambda": lmbda_custom._cpp_object,
            "alpha": alpha_custom._cpp_object,
        }
        gamma = gamma * E

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
    meshties = MeshTie(
        [facet_marker._cpp_object],
        surfaces,
        [(0, 1), (1, 0)],
        mesh_custom._cpp_object,
        quadrature_degree=quadrature_degree,
    )
    meshties.generate_kernel_data(problem, V_custom._cpp_object, coeffs, gamma, theta)

    # Generate residual data structures
    F_custom = _fem.form(F0)
    b1 = _fem.petsc.create_vector(F_custom)

    # # Generate matrix
    J_custom = _fem.form(J_custom)
    A1 = meshties.create_matrix(J_custom._cpp_object)

    # Assemble  residual
    b1.zeroEntries()
    meshties.assemble_vector(b1, V_custom._cpp_object, problem)

    # Assemble  jacobian
    A1.zeroEntries()
    meshties.assemble_matrix(A1, V_custom._cpp_object, problem)
    A1.assemble()

    # Retrieve data necessary for comparison
    tdim = mesh_ufl.topology.dim
    ind_dg = compute_dof_permutations_all(V_ufl, V_custom, gap)

    # Compare rhs
    assert np.allclose(b0.array[ind_dg], b1.array)

    # create scipy matrix
    ai, aj, av = A0.getValuesCSR()
    A_sp = scipy.sparse.csr_matrix((av, aj, ai), shape=A0.getSize()).todense()
    bi, bj, bv = A1.getValuesCSR()
    B_sp = scipy.sparse.csr_matrix((bv, bj, bi), shape=A1.getSize()).todense()

    assert np.allclose(A_sp[:, ind_dg][ind_dg, :], B_sp)
