# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from mpi4py import MPI

import dolfinx.io.gmshio
import gmsh
import numpy as np
import pytest
from dolfinx.fem import Function, functionspace
from dolfinx.graph import adjacencylist
from dolfinx.mesh import Mesh, locate_entities_boundary, meshtags
from dolfinx_contact.cpp import Contact, ContactMode
from dolfinx_contact.meshing import (
    create_circle_plane_mesh,
    create_cylinder_cylinder_mesh,
    create_sphere_plane_mesh,
)


# @pytest.mark.xdist_group(name="group0")
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("simplex", [True, False])
@pytest.mark.parametrize("res", [0.01, 0.1])
@pytest.mark.parametrize("dim", [2, 3])
def test_copy_to_submesh(tmp_path, order, res, simplex, dim):
    """TODO."""

    gmsh.initialize()
    if dim == 3:
        if simplex:
            name = "test_copy"
            model = gmsh.model()
            model.add(name)
            model.setCurrent(name)
            model = create_sphere_plane_mesh(
                model, res=res, order=order, r=0.25, height=0.25, length=1.0, width=1.0
            )
            mesh, _, facet_marker = dolfinx.io.gmshio.model_to_mesh(
                model, MPI.COMM_WORLD, 0, gdim=3
            )
            contact_bdy_1 = 1
            contact_bdy_2 = 8
        else:
            name = "test_copy_cylinders3D"
            model = gmsh.model()
            model.add(name)
            model.setCurrent(name)
            model = create_cylinder_cylinder_mesh(model, order=order, res=10 * res, simplex=simplex)
            mesh, _, _ = dolfinx.io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=3)

            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, tdim)

            def right(x):
                return x[0] > 2.2

            def right_contact(x):
                return np.logical_and(x[0] < 2, x[0] > 1.45)

            def left_contact(x):
                return np.logical_and(x[0] > 0.25, x[0] < 1.1)

            def left(x):
                return x[0] < -0.5

            dirichlet_bdy_1 = 1
            contact_bdy_1 = 2
            contact_bdy_2 = 3
            dirichlet_bdy_2 = 4
            # Create meshtag for top and bottom markers
            dirichlet_facets_1 = locate_entities_boundary(mesh, tdim - 1, right)
            contact_facets_1 = locate_entities_boundary(mesh, tdim - 1, right_contact)
            contact_facets_2 = locate_entities_boundary(mesh, tdim - 1, left_contact)
            dirchlet_facets_2 = locate_entities_boundary(mesh, tdim - 1, left)

            val0 = np.full(len(dirichlet_facets_1), dirichlet_bdy_1, dtype=np.int32)
            val1 = np.full(len(contact_facets_1), contact_bdy_1, dtype=np.int32)
            val2 = np.full(len(contact_facets_2), contact_bdy_2, dtype=np.int32)
            val3 = np.full(len(dirchlet_facets_2), dirichlet_bdy_2, dtype=np.int32)
            indices = np.concatenate(
                [
                    dirichlet_facets_1,
                    contact_facets_1,
                    contact_facets_2,
                    dirchlet_facets_2,
                ]
            )
            values = np.hstack([val0, val1, val2, val3])
            sorted_facets = np.argsort(indices)
            facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])
    else:
        name = "test_copy_hertz2D"
        model = gmsh.model()
        model.add(name)
        model.setCurrent(name)
        model = create_circle_plane_mesh(
            model, res=res, order=order, quads=not simplex, r=0.25, height=0.25, length=1.0
        )
        mesh, _, facet_marker = dolfinx.io.gmshio.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=2)
        contact_bdy_1 = 4
        contact_bdy_2 = 9

    gmsh.finalize()

    def _test_fun(x):
        tdim = mesh.topology.dim
        vals = x[:tdim, :]
        return vals

    V = functionspace(mesh, ("Lagrange", order, (mesh.geometry.dim,)))
    contact_pairs = [(0, 1), (1, 0)]
    data = np.array([contact_bdy_1, contact_bdy_2], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    contact_surfaces = adjacencylist(data, offsets)
    search_method = [ContactMode.ClosestPoint, ContactMode.Raytracing]
    contact = Contact(
        [facet_marker._cpp_object],
        contact_surfaces,
        contact_pairs,
        mesh._cpp_object,
        quadrature_degree=3,
        search_method=search_method,
    )

    u = Function(V)
    u.interpolate(_test_fun)
    submesh_cpp = contact.submesh()
    ufl_domain = mesh.ufl_domain()
    submesh = Mesh(submesh_cpp, ufl_domain)
    V_sub = functionspace(submesh, ("Lagrange", order, (mesh.geometry.dim,)))
    u_sub = Function(V_sub)
    contact.copy_to_submesh(u._cpp_object, u_sub._cpp_object)
    u_exact = Function(V_sub)
    u_exact.interpolate(_test_fun)
    eps = np.finfo(mesh.geometry.x.dtype).eps
    np.testing.assert_allclose(u_sub.x.array[:], u_exact.x.array[:], atol=eps)
