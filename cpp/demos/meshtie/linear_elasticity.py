# Copyright (C) 2023 Sarah Roggendorf
#
# This file is part of DOLFINx_Contact
# SPDX-License-Identifier:    MIT
#
# UFL input for linear elasticity part of meshtie demo
# ====================================================

from basix.ufl import element
from ufl import (Coefficient, ds, dx, functionspace, grad, Identity,
                 inner, Mesh, sym, TestFunction, tr, TrialFunction)


# tags for boundaries (see mesh file)
neumann_bdy = 7
contact_bdy_1 = 6  # top contact interface
contact_bdy_2 = 13  # bottom contact interface

e = element("Lagrange", "tetrahedron", 1, shape=(3,))
e0 = element("Discontinuous Lagrange", "tetrahedron", 0)
coord_element = element("Lagrange", "tetrahedron", 1, shape=(3,))
mesh = Mesh(coord_element)
V = functionspace(mesh, e)
V0 = functionspace(mesh, e0)
v = TestFunction(V)
w = TrialFunction(V)

lmbda = Coefficient(V0)
mu = Coefficient(V0)


def epsilon(z):
    return sym(grad(z))


def sigma(z):
    return (2.0 * mu * epsilon(z) + lmbda * tr(epsilon(z)) * Identity(len(z)))


# Bilinear form
J = inner(sigma(w), epsilon(v)) * dx

f = Coefficient(V)
t = Coefficient(V)

# Linear form
F = inner(f, v) * dx
F += inner(t, v) * ds(neumann_bdy)

forms = [F, J]
