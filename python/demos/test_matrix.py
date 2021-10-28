# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:   MIT

import dolfinx
import dolfinx_contact
import dolfinx_contact.cpp
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import scipy.sparse
import matplotlib.pylab as plt

kt = dolfinx_contact.cpp.Kernel
q_deg = 6
x = np.array([[-0.5, 0], [0.5, 0], [0, 1], [-0.7, -0.2], [-0.2, -0.2], [-0.2, -0.9], [0.6, -0.2]])
cells = np.array([[0, 1, 2], [3, 4, 5], [4, 5, 6]])
cell = ufl.Cell("triangle", geometric_dimension=x.shape[1])
domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)
el = ufl.VectorElement("CG", mesh.ufl_cell(), 1)
V = dolfinx.FunctionSpace(mesh, el)

tdim = mesh.topology.dim
facets_0 = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], 0.0))

facets_1 = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], -0.2))

values_0 = np.full(len(facets_0), 0, dtype=np.int32)
values_1 = np.full(len(facets_1), 1, dtype=np.int32)
indices = np.concatenate([facets_0, facets_1])
values = np.concatenate([values_0, values_1])
sorted_facets = np.argsort(indices)
facet_marker = dolfinx.MeshTags(mesh, tdim - 1, indices[sorted_facets], values[sorted_facets])


contact = dolfinx_contact.cpp.Contact(facet_marker, 0, 1, V._cpp_object)
contact.set_quadrature_degree(q_deg)
contact.create_distance_map(0)
contact.create_distance_map(1)
print(facets_0)
print(facets_1)
v = ufl.TestFunction(V)
u = ufl.TrialFunction(V)
dx = ufl.Measure("dx", domain=mesh)
a = ufl.inner(u, v) * dx
a_cuas = dolfinx.fem.Form(a)
A = contact.create_matrix(a_cuas._cpp_object)
kernel = contact.generate_kernel(0, kt.Jac)
print("kernel generated")
contact.assemble_matrix(A, [], 0, kernel, [[]], [])
contact.assemble_matrix(A, [], 1, kernel, [[]], [])
gap = contact.pack_gap(0)
print("gap_packed")
test_fn = contact.pack_test_functions(0, gap)
print("test functions packed")
gap1 = contact.pack_gap(1)
test_fn1 = contact.pack_test_functions(1, gap1)

print(test_fn)
print(test_fn1)


A.assemble()
# Create scipy CSR matrices
ai, aj, av = A.getValuesCSR()
A_sp = scipy.sparse.csr_matrix((av, aj, ai), shape=A.getSize())
plt.spy(A_sp)
plt.savefig("test.pdf")


def f(x):
    values = np.zeros((1, x.shape[1]))
    for i in range(x.shape[1]):
        values[0, i] = np.max(np.abs(x[:, i]))
    return values


# Interpolate some function and pack
u_D = dolfinx.Function(V)

nu = 0.5
E = 20


def _u_ex(x):
    values = np.zeros((mesh.geometry.dim, x.shape[1]))
    values[0] = (nu + 1) / E * x[1]**4
    values[1] = (nu + 1) / E * x[0]**4
    return values


u_D.interpolate(_u_ex)
print(contact.pack_u_contact(0, u_D._cpp_object, gap))
print(contact.pack_u_contact(1, u_D._cpp_object, gap1))
# nnz form only: 23
# 1st insert 32 = 23 + 9 (3x3)
# 2nd insert 41 = 32 + 9
# Final:44
