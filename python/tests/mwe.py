import basix
from dolfinx.fem import (Expression, Function, FunctionSpace)
from dolfinx.mesh import (CellType, create_unit_square, to_string)
from mpi4py import MPI
from ufl import grad

import dolfinx_contact.cpp

N = 15
mesh = create_unit_square(MPI.COMM_WORLD, N, N, cell_type=CellType.triangle)
V = FunctionSpace(mesh, ('DG', 0))
v = Function(V)
v.interpolate(lambda x: x[0] < 0.5 + x[1])

ct = mesh.topology.cell_type
# Use prepare quadrature points and geometry for eval
quadrature_points, _ = basix.make_quadrature(
    basix.QuadratureType.Default, basix.cell.string_to_type(to_string(ct)), 1)
expr = Expression(grad(v), quadrature_points)
