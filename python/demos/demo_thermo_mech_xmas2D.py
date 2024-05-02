# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT
from mpi4py import MPI

import numpy as np

import dolfinx.fem as _fem
import ufl
from dolfinx import default_scalar_type, io, log
from dolfinx.common import timed, Timer
from dolfinx.fem import (Function, FunctionSpace)
from dolfinx.fem.petsc import LinearProblem, assemble_vector, assemble_matrix, create_vector
from dolfinx.graph import adjacencylist
from dolfinx.io import XDMFFile
from dolfinx_contact.helpers import epsilon, lame_parameters, sigma_func, weak_dirichlet
from dolfinx_contact.meshing import convert_mesh, create_christmas_tree_mesh
from dolfinx_contact.newton_solver import NewtonSolver
from dolfinx_contact.parallel_mesh_ghosting import create_contact_mesh
from dolfinx_contact.general_contact.contact_problem import ContactProblem, FrictionLaw
from dolfinx_contact.cpp import ContactMode
from petsc4py.PETSc import InsertMode, ScatterMode  # type: ignore

fname = "meshes/xmas_2D"
create_christmas_tree_mesh(filename=fname, res=0.2)
convert_mesh(fname, fname, gdim=2)
with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    tdim = mesh.topology.dim
    domain_marker = xdmf.read_meshtags(mesh, name="cell_marker")
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facet_marker = xdmf.read_meshtags(mesh, name="facet_marker")

contact_bdy_1 = 5
contact_bdy_2 = 6
if mesh.comm.size > 1:
    mesh, facet_marker, domain_marker = create_contact_mesh(
        mesh, facet_marker, domain_marker, [contact_bdy_1, contact_bdy_2])

# measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)


# Thermal problem
Q = _fem.functionspace(mesh, ("Lagrange", 1))
q, r = ufl.TrialFunction(Q), ufl.TestFunction(Q)
T0 = _fem.Function(Q)
kdt = 0.1
tau = 1.0
therm = (q - T0) * r * dx + kdt * ufl.inner(ufl.grad(tau * q + (1 - tau) * T0), ufl.grad(r)) * dx

a_therm, L_therm = ufl.lhs(therm), ufl.rhs(therm)

T0.x.array[:] = 0
dofs = _fem.locate_dofs_topological(Q, entity_dim=tdim - 1, entities=facet_marker.find(3))
Tbc = _fem.dirichletbc(value=default_scalar_type((1.0)), dofs=dofs, V=Q)
dofs2 = _fem.locate_dofs_topological(Q, entity_dim=tdim - 1, entities=facet_marker.find(4))
Tbc2 = _fem.dirichletbc(value=default_scalar_type((0.0)), dofs=dofs2, V=Q)
Tproblem = LinearProblem(a_therm, L_therm, bcs=[Tbc, Tbc2], petsc_options={
    "ksp_type": "preonly", "pc_type": "lu"}, u=T0)


# Elasticity problem
V = _fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim, )))
g = _fem.Constant(mesh, default_scalar_type((0, 0)))     # zero Dirichlet
t = _fem.Constant(mesh, default_scalar_type((0.2, 0.5)))  # traction
f = _fem.Constant(mesh, default_scalar_type((1.0, 0.5)))  # body force


# contact surface data
contact_pairs = [(0, 1), (1, 0)]
data = np.array([contact_bdy_1, contact_bdy_2], dtype=np.int32)
offsets = np.array([0, 2], dtype=np.int32)
surfaces = adjacencylist(data, offsets)


# Function, TestFunction, TrialFunction and measures
u = _fem.Function(V)
du = _fem.Function(V)
w = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Compute lame parameters
E = 1e4
nu = 0.2
mu_func, lambda_func = lame_parameters(True)
mu = mu_func(E, nu)
lmbda = lambda_func(E, nu)
V0 = FunctionSpace(mesh, ("DG", 0))
mu_dg = Function(V0)
lmbda_dg = Function(V0)
mu_dg.interpolate(lambda x: np.full((1, x.shape[1]), mu))
lmbda_dg.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))


def eps(w):
    return ufl.sym(ufl.grad(w))


alpha = 0.1


def sigma(w, T):
    return (lmbda * ufl.tr(eps(w)) - alpha * (3 * lmbda + 2 * mu) * T) * ufl.Identity(tdim) + 2.0 * mu * eps(w)


# Create variational form without contact contributions
F = ufl.inner(sigma(u, T0), epsilon(v)) * dx


# Apply weak Dirichlet boundary conditions using Nitsche's method
gamma = 10
theta = 1
sigma_u = sigma_func(mu, lmbda)
F = weak_dirichlet(F, u, g, sigma_u, E * gamma, theta, ds(4))
F = weak_dirichlet(F, u, g, sigma_u, E * gamma, theta, ds(3))

F = ufl.replace(F, {u: u + du})

J = ufl.derivative(F, du, w)

# compiler options to improve performance
cffi_options = ["-Ofast", "-march=native"]
jit_options = {"cffi_extra_compile_args": cffi_options,
               "cffi_libraries": ["m"]}
# compiled forms for rhs and tangen system
F_compiled = _fem.form(F, jit_options=jit_options)
J_compiled = _fem.form(J, jit_options=jit_options)

# Solver options
ksp_tol = 1e-10
newton_tol = 1e-6
newton_options = {"relaxation_parameter": 1.0,
                  "atol": newton_tol,
                  "rtol": newton_tol,
                  "convergence_criterion": "residual",
                  "max_it": 50,
                  "error_on_nonconvergence": False}

# In order to use an LU solver for debugging purposes on small scale problems
# use the following PETSc options: {"ksp_type": "preonly", "pc_type": "lu"}
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


# Solve contact problem using Nitsche's method
problem_parameters = {"gamma": np.float64((1 - alpha) * E * gamma), "theta": np.float64(theta), "fric": np.float64(0.4)}
log.set_log_level(log.LogLevel.WARNING)
size = mesh.comm.size
outname = f"results/xmas_{tdim}D_{size}"
u.name = 'displacement'
T0.name = 'temperature'

search_mode = [ContactMode.ClosestPoint for _ in range(len(contact_pairs))]
contact_problem = ContactProblem([facet_marker], surfaces, contact_pairs, mesh, 5, search_mode)
contact_problem.generate_contact_data(FrictionLaw.Frictionless, V, {"u": u, "du": du, "mu": mu_dg,
                                                                    "lambda": lmbda_dg}, E * gamma, theta)
# define functions for newton solver


def compute_coefficients(x, coeffs):
    du.x.scatter_forward()
    contact_problem.update_contact_data(du)


@timed("~Contact: Assemble residual")
def compute_residual(x, b, coeffs):
    b.zeroEntries()
    b.ghostUpdate(addv=InsertMode.INSERT,
                  mode=ScatterMode.FORWARD)
    with Timer("~~Contact: Contact contributions (in assemble vector)"):
        contact_problem.assemble_vector(b, V)
    with Timer("~~Contact: Standard contributions (in assemble vector)"):
        assemble_vector(b, F_compiled)
    b.ghostUpdate(addv=InsertMode.ADD,
                  mode=ScatterMode.REVERSE)


@timed("~Contact: Assemble matrix")
def compute_jacobian_matrix(x, a_mat, coeffs):
    a_mat.zeroEntries()
    with Timer("~~Contact: Contact contributions (in assemble matrix)"):
        contact_problem.assemble_matrix(a_mat, V)
    with Timer("~~Contact: Standard contributions (in assemble matrix)"):
        assemble_matrix(a_mat, J_compiled)
    a_mat.assemble()


# create vector and matrix
a_mat = contact_problem.create_matrix(J_compiled)
b = create_vector(F_compiled)


# Set up snes solver for nonlinear solver
newton_solver = NewtonSolver(mesh.comm, a_mat, b, contact_problem.coeffs)
# Set matrix-vector computations
newton_solver.set_residual(compute_residual)
newton_solver.set_jacobian(compute_jacobian_matrix)
newton_solver.set_coefficients(compute_coefficients)

# Set rigid motion nullspace
null_space = rigid_motions_nullspace_subdomains(V, domain_marker, np.unique(
    domain_marker.values), num_domains=2)
newton_solver.A.setNearNullSpace(null_space)

# Set Newton solver options
newton_solver.set_newton_options(newton_options)

# Set Krylov solver options
newton_solver.set_krylov_options(petsc_options)


# initialise vtx write
W = FunctionSpace(mesh, ("DG", 1))
sigma_vm_h = Function(W)
sigma_dev = sigma(u, T0) - (1 / 3) * \
    ufl.tr(sigma(u, T0)) * ufl.Identity(len(u))
sigma_vm = ufl.sqrt((3 / 2) * ufl.inner(sigma_dev, sigma_dev))
gdim = mesh.geometry.dim
W2 = _fem.functionspace(mesh, ("Discontinuous Lagrange", 1, (gdim, )))
u_dg = Function(W2)
u_dg.interpolate(u)
T_dg = Function(W)
T_dg.interpolate(T0)
sigma_vm_h.name = "vonMises"
u_dg.name = "displacement"
T_dg.name = "temperature"
vtx = io.VTXWriter(mesh.comm, "results/xmas_disp.bp", [u_dg, T_dg, sigma_vm_h], "bp4")
vtx.write(0)
for i in range(50):
    Tproblem.solve()

    n, converged = newton_solver.solve(du)
    du.x.scatter_forward()
    u.x.array[:] += du.x.array[:]
    contact_problem.update_contact_detection(u)
    a_mat = contact_problem.create_matrix(J_compiled)
    a_mat.setNearNullSpace(null_space)
    newton_solver.set_petsc_matrix(a_mat)
    # take a fraction of du as initial guess
    # this is to ensure non-singular matrices in the case of no Dirichlet boundary
    du.x.array[:] = 0.1 * du.x.array[:]
    contact_problem.update_contact_data(du)
    sigma_vm_expr = _fem.Expression(sigma_vm, W.element.interpolation_points())
    sigma_vm_h.interpolate(sigma_vm_expr)
    u_dg.interpolate(u)
    T_dg.interpolate(T0)
    vtx.write(i + 1)

vtx.close()
