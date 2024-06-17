#
# testing of large number of meshties
# written by Neeraj Cherukunnath
# UFL input for linear elasticity
# two dirichlet faces, domains with CFload 
# 
import basix
from basix.ufl import element
from ufl import (Coefficient, ds, dx, FunctionSpace, grad, Identity,
                 inner, Mesh, sym, TestFunction, tr, TrialFunction,
                 SpatialCoordinate, as_vector)


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
    return (2.0 * mu * epsilon(z) + lmbda * tr(epsilon(z)) * Identity(len(z)))

# CFload with x-axis of rotation
x = SpatialCoordinate(mesh)
rad = as_vector((0.0 , x[1], x[2]))
rho = 0.442900E-08 #density
omega = 1500.0 #rotational speed
accel = omega*omega*rad

# Bilinear form
J = inner(sigma(w), epsilon(v)) * dx

# Linear form
F = rho*inner(accel,v)*dx

forms = [F, J]
