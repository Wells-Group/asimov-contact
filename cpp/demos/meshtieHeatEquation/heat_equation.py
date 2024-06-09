# Copyright (C) 2023 Sarah Roggendorf
#
# This file is part of DOLFINx_Contact
# SPDX-License-Identifier:    MIT
#
# UFL input for heat equation with implicit Euler
# ====================================================

from basix.ufl import element
from ufl import (
    Coefficient,
    dx,
    FunctionSpace,
    grad,
    lhs,
    rhs,
    inner,
    Mesh,
    TestFunction,
    TrialFunction,
)

# Mesh and elements
e = element("Lagrange", "tetrahedron", 1, shape=())
e0 = element("DG", "tetrahedron", 0)
coord_element = element("Lagrange", "tetrahedron", 1, shape=(3,))
mesh = Mesh(coord_element)

# Function spaces, functions, coefficients
Q = FunctionSpace(mesh, e)
V0 = FunctionSpace(mesh, e0)
r = TestFunction(Q)
q = TrialFunction(Q)
T0 = Coefficient(Q)

# Problem parameter
kdt = Coefficient(V0)


# Bilinear form and linear form
therm = (q - T0) * r * dx + kdt * inner(grad(q), grad(r)) * dx
a_therm, L_therm = lhs(therm), rhs(therm)

forms = [L_therm, a_therm]
