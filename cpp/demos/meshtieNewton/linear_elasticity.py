# Copyright (C) 2023 Sarah Roggendorf
#
# This file is part of DOLFINx_Contact
# SPDX-License-Identifier:    MIT
#
# UFL input for linear elasticity part of meshtie demo
# ====================================================

from basix.ufl import element
from ufl import (
    Coefficient,
    derivative,
    dx,
    FunctionSpace,
    grad,
    Identity,
    inner,
    Mesh,
    sym,
    TestFunction,
    tr,
    TrialFunction,
)


e = element("Lagrange", "tetrahedron", 1, shape=(3,))
e0 = element("DG", "tetrahedron", 0)
coord_element = element("Lagrange", "tetrahedron", 1, shape=(3,))
mesh = Mesh(coord_element)
V = FunctionSpace(mesh, e)
V0 = FunctionSpace(mesh, e0)
v = TestFunction(V)
w = TrialFunction(V)

lmbda = Coefficient(V0)
mu = Coefficient(V0)


def epsilon(z):
    return sym(grad(z))


def sigma(z):
    return 2.0 * mu * epsilon(z) + lmbda * tr(epsilon(z)) * Identity(len(z))


f = Coefficient(V)
t = Coefficient(V)
u = Coefficient(V)

# Linear form
F = inner(sigma(u), epsilon(v)) * dx

# Bilinear form
J = derivative(F, u, w)

forms = [F, J]
