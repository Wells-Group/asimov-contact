# Copyright (C) 2023 Sarah Roggendorf
#
# This file is part of DOLFINx_Contact
# SPDX-License-Identifier:    MIT
#
# UFL input for thermo elasticity part of meshtie demo
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
    rhs,
    lhs,
    Mesh,
    sym,
    TestFunction,
    tr,
    TrialFunction,
)

# Mesh
coord_element = element("Lagrange", "tetrahedron", 1, shape=(3,))
mesh = Mesh(coord_element)

# DG space for material parameters
e0 = element("DG", "tetrahedron", 0)
V0 = FunctionSpace(mesh, e0)

# Thermal problem

# Functionsspace, functions, coefficients
e_therm = element("Lagrange", "tetrahedron", 1, shape=())
Q = FunctionSpace(mesh, e_therm)
q = TrialFunction(Q)
r = TestFunction(Q)
kdt = Coefficient(V0)
T0 = Coefficient(Q)

# bi-linear and linear form
therm = (q - T0) * r * dx + kdt * inner(grad(q), grad(r)) * dx
a_therm, L_therm = lhs(therm), rhs(therm)

# Elastic problem

# Functionspace, functions, coefficients
e_disp = element("Lagrange", "tetrahedron", 1, shape=(3,))
V = FunctionSpace(mesh, e_disp)

v = TestFunction(V)
w = TrialFunction(V)

# Material parameters
lmbda = Coefficient(V0)
mu = Coefficient(V0)
alpha = Coefficient(V0)
u = Coefficient(V)


# Stress tensor definition
def epsilon(z):
    return sym(grad(z))


def sigma(z, T):
    return 2.0 * mu * epsilon(z) + (
        lmbda * tr(epsilon(z)) - alpha * (3 * lmbda + 2 * mu) * T
    ) * Identity(len(z))


# Linear form
F = inner(sigma(u, T0), epsilon(v)) * dx

# Bilinear form
J = derivative(F, u, w)

forms = [a_therm, L_therm, F, J]
