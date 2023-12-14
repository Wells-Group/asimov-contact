# Copyright (C) 2023 Sarah Roggendorf
#
# This file is part of DOLFINx_Contact
# SPDX-License-Identifier:    MIT
#
# UFL input for linear elasticity part of meshtie demo
# ====================================================

from basix.ufl import element
from ufl import (Coefficient, dx, FunctionSpace, grad, lhs, rhs,
                 inner, Mesh, TestFunction, TrialFunction)


e = element("Lagrange", "tetrahedron", 1, shape=())
e0 = element("DG", "tetrahedron", 0)
coord_element = element("Lagrange", "tetrahedron", 1, shape=(3,))
mesh = Mesh(coord_element)
Q = FunctionSpace(mesh, e)
V0 = FunctionSpace(mesh, e0)
v = TestFunction(Q)
w = TrialFunction(Q)
T0 = Coefficient(Q)

kdt = Coefficient(V0)


# Bilinear form
therm = (w - T0) * v *dx + kdt * inner(grad(w), grad(v)) * dx
a_therm, L_therm = lhs(therm), rhs(therm)

forms = [L_therm, a_therm]
