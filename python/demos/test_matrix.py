# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:   MIT

import dolfinx
import dolfinx_contact
import dolfinx_contact.cpp
import dolfinx_cuas
import dolfinx_cuas.cpp
import numpy as np
import ufl
from mpi4py import MPI
# from petsc4py import PETSc
import scipy.sparse
import matplotlib.pylab as plt

kt = dolfinx_contact.cpp.Kernel
it = dolfinx.cpp.fem.IntegralType

q_deg = 5
x = np.array([[-0.5, 0], [0.5, 0], [0, 1], [-0.7, -0.2], [-0.2, -0.2], [-0.2, -0.9], [0.6, -0.2]])
cells = np.array([[0, 1, 2], [3, 4, 5], [4, 5, 6]])
cell = ufl.Cell("triangle", geometric_dimension=x.shape[1])
domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)
el = ufl.VectorElement("CG", mesh.ufl_cell(), 1)
V = dolfinx.FunctionSpace(mesh, el)

tdim = mesh.topology.dim
gdim = mesh.geometry.dim
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

v = ufl.TestFunction(V)
u = ufl.TrialFunction(V)
dx = ufl.Measure("dx", domain=mesh)
a = ufl.inner(u, v) * dx
a_cuas = dolfinx.fem.Form(a)
A = contact.create_matrix(a_cuas._cpp_object)
kernel = contact.generate_kernel(kt.Jac)
gap = contact.pack_gap(0)
test_fn = contact.pack_test_functions(0, gap)
gap1 = contact.pack_gap(1)
test_fn1 = contact.pack_test_functions(1, gap1)

# Interpolate some function and pack


def f(x):
    values = np.zeros((1, x.shape[1]))
    for i in range(x.shape[1]):
        values[0, i] = np.max(np.abs(x[:, i]))
    return values


V2 = dolfinx.FunctionSpace(mesh, ("DG", 0))
mufunc = dolfinx.Function(V2)
mufunc.interpolate(f)
nu = 0.5
E = 20


def _u_ex(x):
    values = np.zeros((mesh.geometry.dim, x.shape[1]))
    values[0] = (nu + 1) / E * x[1]**4
    values[1] = (nu + 1) / E * x[0]**4
    return values


u_D = dolfinx.Function(V)
u_D.interpolate(_u_ex)
consts = [1, 1]
coeff_0 = contact.pack_coefficient_dofs(0, mufunc._cpp_object)
# dummy mu, lmbda, h
h_facets = dolfinx_contact.cpp.pack_circumradius_facet(mesh, np.sort(facets_0))

coeff_0 = np.hstack([coeff_0, coeff_0, h_facets])
coeff_0 = np.hstack([coeff_0, gap, test_fn])

coeff_0_test = contact.pack_coeffs_const(0, mufunc._cpp_object, mufunc._cpp_object)
print("packing works")
print(np.allclose(coeff_0, coeff_0_test))
# dummy u
coeff_0 = np.hstack([coeff_0, contact.pack_u_contact(0, u_D._cpp_object, gap)])
# dummy u opposite
coeff_0 = np.hstack([coeff_0, contact.pack_coefficient_dofs(0, u_D._cpp_object)])


coeff_1 = contact.pack_coefficient_dofs(1, mufunc._cpp_object)
# dummy mu, lmbda, h
h_facets_1 = dolfinx_contact.cpp.pack_circumradius_facet(mesh, np.sort(facets_1))
coeff_1 = np.hstack([coeff_1, coeff_1, h_facets_1])
coeff_1 = np.hstack([coeff_1, gap1, test_fn1])
# dummy u
coeff_1 = np.hstack([coeff_1, contact.pack_coefficient_dofs(1, u_D._cpp_object)])
# dummy u opposite
coeff_1 = np.hstack([coeff_1, contact.pack_u_contact(1, u_D._cpp_object, gap)])
# contact.assemble_matrix(A, [], 0, kernel, coeff_0, consts)
contact.assemble_matrix(A, [], 1, kernel, coeff_1, consts)

A.assemble()
# Create scipy CSR matrices
ai, aj, av = A.getValuesCSR()
A_sp = scipy.sparse.csr_matrix((av, aj, ai), shape=A.getSize())
plt.spy(A_sp)
plt.savefig("test.pdf")
q_rule = dolfinx_cuas.cpp.QuadratureRule(mesh.topology.cell_type, q_deg, mesh.topology.dim - 1, "default")

h_cells = dolfinx_contact.cpp.facet_to_cell_data(mesh, np.sort(facets_1), h_facets_1, 1)
coeffs = dolfinx_cuas.cpp.pack_coefficients([mufunc._cpp_object, mufunc._cpp_object])
g_vec_c = dolfinx_contact.cpp.facet_to_cell_data(
    mesh, np.sort(facets_1), gap1, gdim * q_rule.weights(0).size)

kernel_one_sided = dolfinx_contact.cpp.generate_contact_kernel(V._cpp_object, kt.Jac, q_rule,
                                                               [u_D._cpp_object, mufunc._cpp_object, mufunc._cpp_object],
                                                               False)
B = dolfinx.fem.create_matrix(a_cuas)
u_packed = dolfinx_cuas.cpp.pack_coefficients([u_D._cpp_object])

c = np.hstack([u_packed, coeffs, h_cells, g_vec_c])
dolfinx_cuas.assemble_matrix(B, V, facets_1, kernel_one_sided, c, consts, it.exterior_facet)
B.assemble()
bi, bj, bv = B.getValuesCSR()
B_sp = scipy.sparse.csr_matrix((bv, bj, bi), shape=B.getSize())

print(np.max(np.abs(A_sp.toarray() - B_sp.toarray())))
print(np.allclose(A_sp.toarray(), B_sp.toarray()))


L = ufl.inner(u_D, v) * dx
L_cuas = dolfinx.fem.Form(L)
b1 = dolfinx.fem.create_vector(L_cuas)
kernel_rhs = contact.generate_kernel(kt.Rhs)
contact.assemble_vector(b1, 1, kernel_rhs, coeff_1, consts)


kernel_one_sided_rhs = dolfinx_contact.cpp.generate_contact_kernel(V._cpp_object, kt.Rhs, q_rule,
                                                                   [u_D._cpp_object, mufunc._cpp_object, mufunc._cpp_object],
                                                                   False)
b2 = dolfinx.fem.create_vector(L_cuas)

dolfinx_cuas.assemble_vector(b2, V, np.sort(facets_1), kernel_one_sided_rhs, c, consts, it.exterior_facet)
print(np.max(np.abs(b1.array - b2.array)))
print(np.allclose(b1.array, b2.array))
