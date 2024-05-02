# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT
#
from mpi4py import MPI

import numpy as np

from basix.ufl import element
from dolfinx import default_scalar_type
from dolfinx.fem import (Constant, dirichletbc, Function, form,
                         functionspace, locate_dofs_topological)
from dolfinx.fem.petsc import set_bc, apply_lifting, assemble_matrix, assemble_vector, create_vector
from dolfinx.graph import adjacencylist
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_mesh
from petsc4py.PETSc import InsertMode, ScatterMode  # type: ignore
from ufl import (derivative, grad, Identity, inner, Mesh, Measure,
                 replace, sym, TrialFunction, TestFunction, tr)

from dolfinx_contact.helpers import rigid_motions_nullspace_subdomains
from dolfinx_contact.newton_solver import NewtonSolver
from dolfinx_contact.general_contact.contact_problem import ContactProblem, FrictionLaw
from dolfinx_contact.cpp import ContactMode

# read mesh from file
fname = "cont-blocks_sk24_fnx"
with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
    cell_type, cell_degree = xdmf.read_cell_type(name="volume markers")
    topo = xdmf.read_topology_data(name="volume markers")
    x = xdmf.read_geometry_data(name="geometry")
    domain = Mesh(element("Lagrange", cell_type.name,
                  cell_degree, shape=(x.shape[1],)))
    mesh = create_mesh(MPI.COMM_WORLD, topo, x, domain)
    tdim = mesh.topology.dim
    domain_marker = xdmf.read_meshtags(mesh, name="volume markers")
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facet_marker = xdmf.read_meshtags(mesh, name="facet markers")

# tags for boundaries (see mesh file)
dirichlet_bdy_1 = 8  # top face
dirichlet_bdy_2 = 2  # bottom face

contact_bdy_1 = 12  # top contact interface
contact_bdy_2 = 6  # bottom contact interface

# measures
dx = Measure("dx", domain=mesh, subdomain_data=domain_marker)
ds = Measure("ds", domain=mesh, subdomain_data=facet_marker)


# Elasticity problem

# Function space
gdim = mesh.topology.dim
V = functionspace(mesh, ("Lagrange", 1, (gdim,)))

# Function, TestFunction
u = Function(V)
du = Function(V)
v = TestFunction(V)
w = TrialFunction(V)

# Compute lame parameters
E = 1e4
nu = 0.2
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
V0 = functionspace(mesh, ("DG", 0))
mu_dg = Function(V0)
lmbda_dg = Function(V0)

mu_dg.interpolate(lambda x: np.full((1, x.shape[1]), mu))
lmbda_dg.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))

# Create variational form without contact contributions


def epsilon(v):
    return sym(grad(v))


def sigma(v):
    return (2.0 * mu * epsilon(v) + lmbda * tr(epsilon(v)) * Identity(len(v)))


F = inner(sigma(u), epsilon(v)) * dx

F = replace(F, {u: u + du})
J = derivative(F, du, w)

# compiler options to improve performance
cffi_options = ["-Ofast", "-march=native"]
jit_options = {"cffi_extra_compile_args": cffi_options,
               "cffi_libraries": ["m"]}
# compiled forms for rhs and tangen system
F_compiled = form(F, jit_options=jit_options)
J_compiled = form(J, jit_options=jit_options)

# boundary conditions
g = Constant(mesh, default_scalar_type((0, 0, 0)))     # zero Dirichlet
dofs_g = locate_dofs_topological(
    V, tdim - 1, facet_marker.find(dirichlet_bdy_2))
d = Constant(mesh, default_scalar_type((0, -0.2, 0)))  # vertical displacement
dofs_d = locate_dofs_topological(
    V, tdim - 1, facet_marker.find(dirichlet_bdy_1))
bcs = [dirichletbc(d, dofs_d, V), dirichletbc(g, dofs_g, V)]

# Nitsche parameters
gamma = 10 * E
theta = 1  # 1 - symmetric

# contact surface data
# stored in adjacency list to allow for using multiple meshtags to mark
# contact surfaces. In this case only one meshtag is used, hence offsets has length 2 and
# the second value in offsets is 2 (=2 tags in first and only meshtag).
# The surface with tags [contact_bdy_1, contact_bdy_2] both can be found in this meshtag
data = np.array([contact_bdy_1, contact_bdy_2], dtype=np.int32)
offsets = np.array([0, 2], dtype=np.int32)
surfaces = adjacencylist(data, offsets)
# For unbiased computation the contact detection is performed in both directions
contact_pairs = [(0, 1), (1, 0)]
search_mode = [ContactMode.ClosestPoint for _ in range(len(contact_pairs))]


# Solver options
ksp_tol = 1e-10
newton_tol = 1e-7

# non-linear solver options
newton_options = {"relaxation_parameter": 1.0,
                  "atol": newton_tol,
                  "rtol": newton_tol,
                  "convergence_criterion": "residual",
                  "max_it": 200,
                  "error_on_nonconvergence": True}

# linear solver options
petsc_options = {
    "matptap_via": "scalable",
    "ksp_type": "cg",
    "ksp_rtol": ksp_tol,
    "ksp_atol": ksp_tol,
    "pc_type": "gamg",
    "pc_mg_levels": 3,
    "pc_mg_cycles": 1,   # 1 is v, 2 is w
    "mg_levels_ksp_type": "chebyshev",
    "mg_levels_pc_type": "jacobi",
    "pc_gamg_type": "agg",
    "pc_gamg_coarse_eq_limit": 100,
    "pc_gamg_agg_nsmooths": 1,
    "pc_gamg_threshold": 1e-3,
    "pc_gamg_square_graph": 2,
    "pc_gamg_reuse_interpolation": False,
    "ksp_norm_type": "unpreconditioned"
}

# create contact solver
contact_problem = ContactProblem([facet_marker], surfaces, contact_pairs, mesh, 5, search_mode)
contact_problem.generate_contact_data(FrictionLaw.Frictionless, V, {"u": u, "du": du, "mu": mu_dg,
                                      "lambda": lmbda_dg}, gamma, theta)
# create vector and matrix
a_mat = contact_problem.create_matrix(J_compiled)
b = create_vector(F_compiled)


# define functions for newton solver
def compute_coefficients(x, coeffs):
    du.x.scatter_forward()
    contact_problem.update_contact_data(du)


def compute_residual(x, b, coeffs):
    b.zeroEntries()
    b.ghostUpdate(addv=InsertMode.INSERT,
                  mode=ScatterMode.FORWARD)
    contact_problem.assemble_vector(b, V)
    assemble_vector(b, F_compiled)
    apply_lifting(b, [J_compiled], bcs=[bcs], x0=[x], scale=-1.0)
    b.ghostUpdate(addv=InsertMode.ADD, mode=ScatterMode.REVERSE)
    set_bc(b, bcs, x, -1.0)


def compute_jacobian_matrix(x, a_mat, coeffs):
    a_mat.zeroEntries()
    contact_problem.assemble_matrix(a_mat, V)
    assemble_matrix(a_mat, J_compiled, bcs=bcs)
    a_mat.assemble()


# Set up snes solver for nonlinear solver
newton_solver = NewtonSolver(mesh.comm, a_mat, b, contact_problem.coeffs)
# Set matrix-vector computations
newton_solver.set_residual(compute_residual)
newton_solver.set_jacobian(compute_jacobian_matrix)
newton_solver.set_coefficients(compute_coefficients)

# Set rigid motion nullspace
null_space = rigid_motions_nullspace_subdomains(V, domain_marker, np.unique(
    domain_marker.values), num_domains=len(np.unique(domain_marker.values)))
newton_solver.A.setNearNullSpace(null_space)

# Set Newton solver options
newton_solver.set_newton_options(newton_options)

# Set Krylov solver options
newton_solver.set_krylov_options(petsc_options)

n, converged = newton_solver.solve(du, write_solution=True)
u.x.array[:] += du.x.array[:]
u.x.scatter_forward()

# write resulting diplacements to file
with XDMFFile(mesh.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    u.name = "u"
    xdmf.write_function(u)
