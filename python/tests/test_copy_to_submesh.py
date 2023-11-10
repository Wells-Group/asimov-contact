# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT
import numpy as np
import pytest

from dolfinx.fem import Function, VectorFunctionSpace
from dolfinx.graph import adjacencylist
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities_boundary, meshtags, Mesh
from dolfinx_contact.meshing import (convert_mesh,
                                     create_cylinder_cylinder_mesh,
                                     create_circle_plane_mesh,
                                     create_sphere_plane_mesh)
from dolfinx_contact.cpp import Contact
from mpi4py import MPI


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("simplex", [True, False])
@pytest.mark.parametrize("res", [0.01, 0.1])
@pytest.mark.parametrize("dim", [2, 3])
def test_copy_to_submesh(order, res, simplex, dim):
    mesh_dir = "meshes"
    if dim == 3:
        if simplex:
            fname = f"{mesh_dir}/sphere3D"
            create_sphere_plane_mesh(filename=f"{fname}.msh", order=order, res=5 * res)
            convert_mesh(fname, fname, gdim=3)
            with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
                mesh = xdmf.read_mesh()
                tdim = mesh.topology.dim
                mesh.topology.create_connectivity(tdim - 1, tdim)
                facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")
            contact_bdy_1 = 1
            contact_bdy_2 = 8
        else:
            fname = f"{mesh_dir}/cylinders3D"
            create_cylinder_cylinder_mesh(fname, order=order, res=10 * res, simplex=simplex)
            with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
                mesh = xdmf.read_mesh(name="cylinder_cylinder")

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
            indices = np.concatenate([dirichlet_facets_1, contact_facets_1, contact_facets_2, dirchlet_facets_2])
            values = np.hstack([val0, val1, val2, val3])
            sorted_facets = np.argsort(indices)
            facet_marker = meshtags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])
    else:
        fname = f"{mesh_dir}/hertz2D_simplex" if simplex else f"{mesh_dir}/hertz2D_quads"
        create_circle_plane_mesh(filename=f"{fname}.msh", quads=not simplex, res=res, order=order)
        convert_mesh(fname, f"{fname}.xdmf", gdim=2)
        with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
            mesh = xdmf.read_mesh()
            tdim = mesh.topology.dim
            mesh.topology.create_connectivity(tdim - 1, tdim)
            facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")
        contact_bdy_1 = 4
        contact_bdy_2 = 9

    def _test_fun(x):
        tdim = mesh.topology.dim
        vals = x[:tdim, :]
        return vals
    V = VectorFunctionSpace(mesh, ("Lagrange", order))
    contact_pairs = [(0, 1), (1, 0)]
    data = np.array([contact_bdy_1, contact_bdy_2], dtype=np.int32)
    offsets = np.array([0, 2], dtype=np.int32)
    contact_surfaces = adjacencylist(data, offsets)
    contact = Contact([facet_marker._cpp_object], contact_surfaces, contact_pairs,
                      mesh._cpp_object, quadrature_degree=3)

    u = Function(V)
    u.interpolate(_test_fun)
    submesh_cpp = contact.submesh()
    ufl_domain = mesh.ufl_domain()
    submesh = Mesh(submesh_cpp, ufl_domain)
    V_sub = VectorFunctionSpace(submesh, ("Lagrange", order))
    u_sub = Function(V_sub)
    contact.copy_to_submesh(u._cpp_object, u_sub._cpp_object)
    u_exact = Function(V_sub)
    u_exact.interpolate(_test_fun)
    assert (np.allclose(u_sub.x.array[:], u_exact.x.array[:]))
