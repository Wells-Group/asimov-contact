# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse

import numpy as np
import ufl
from dolfinx import default_scalar_type, io, log
from dolfinx.common import timed, Timer, TimingType, list_timings
from dolfinx.fem import (Constant, form, Function, functionspace, locate_dofs_topological, dirichletbc)
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, LinearProblem
from dolfinx.graph import adjacencylist
from dolfinx.io import XDMFFile
from dolfinx.mesh import GhostMode
from dolfinx_contact.helpers import (epsilon, lame_parameters, sigma_func,
                                     weak_dirichlet, rigid_motions_nullspace_subdomains)
from dolfinx_contact.newton_solver import NewtonSolver
from dolfinx_contact.parallel_mesh_ghosting import create_contact_mesh
from dolfinx_contact.unbiased.contact_problem import ContactProblem, FrictionLaw
from dolfinx_contact.cpp import ContactMode
from mpi4py import MPI
from petsc4py.PETSc import InsertMode, ScatterMode  # type: ignore

desc = "Thermal expansion leading to contact"
parser = argparse.ArgumentParser(description=desc,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--time_steps", default=50, type=np.int32, dest="time_steps",
                    help="Number of pseudo time steps")


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
    "pc_gamg_reuse_interpolation": False
}

args = parser.parse_args()
steps = args.time_steps

timer = Timer("~Contact: - all")
mesh_dir = "meshes"
fname = f"{mesh_dir}/box-key"
WARNING = log.LogLevel.WARNING
log.set_log_level(WARNING)

with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(ghost_mode=GhostMode.none)
    domain_marker = xdmf.read_meshtags(mesh, "cell_marker")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facet_marker = xdmf.read_meshtags(mesh, "facet_marker")

# markers for different parts of the boundary
contact_bdy_1 = 6
contact_bdy_2 = 7
dirichlet_bdy1 = 3
dirichlet_bdy2 = 4

# Call function that repartitions mesh for parallel computation
if mesh.comm.size > 1:
    with Timer("~Contact: Add ghosts"):
        mesh, facet_marker, domain_marker = create_contact_mesh(
            mesh, facet_marker, domain_marker, [contact_bdy_1, contact_bdy_2])

# measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=domain_marker)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)


# Thermal problem
Q = functionspace(mesh, ("CG", 1))
q, r = ufl.TrialFunction(Q), ufl.TestFunction(Q)
T0 = Function(Q)
kdt = 0.1
tau = 1.0
therm = (q - T0) * r * dx + kdt * ufl.inner(ufl.grad(tau * q + (1 - tau) * T0), ufl.grad(r)) * dx

a_therm, L_therm = ufl.lhs(therm), ufl.rhs(therm)

T0.x.array[:] = 0.0
dofs = locate_dofs_topological(Q, entity_dim=tdim - 1, entities=facet_marker.find(dirichlet_bdy1))
Tbc = dirichletbc(value=default_scalar_type((1.0)), dofs=dofs, V=Q)
dofs2 = locate_dofs_topological(Q, entity_dim=tdim - 1, entities=facet_marker.find(dirichlet_bdy2))
Tbc2 = dirichletbc(value=default_scalar_type((0.0)), dofs=dofs2, V=Q)
Tproblem = LinearProblem(a_therm, L_therm, bcs=[Tbc, Tbc2], petsc_options=petsc_options, u=T0)


# Elasticity problem
gdim = mesh.geometry.dim
V = functionspace(mesh, ("Lagrange", 1, (gdim, )))
g = Constant(mesh, default_scalar_type((0, 0, 0)))     # zero Dirichlet


# contact surface data
contact_pairs = [(0, 1), (1, 0)]
data = np.array([contact_bdy_1, contact_bdy_2], dtype=np.int32)
offsets = np.array([0, 2], dtype=np.int32)
surfaces = adjacencylist(data, offsets)


# Function, TestFunction, TrialFunction and measures
u = Function(V)
du = Function(V)
w = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Compute lame parameters
E = 1e4
nu = 0.2
mu_func, lambda_func = lame_parameters(True)
mu = mu_func(E, nu)
lmbda = lambda_func(E, nu)
V0 = functionspace(mesh, ("DG", 0))
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
F = weak_dirichlet(F, u, g, sigma_u, E * gamma, theta, ds(dirichlet_bdy1))
F = weak_dirichlet(F, u, g, sigma_u, E * gamma, theta, ds(dirichlet_bdy2))

F = ufl.replace(F, {u: u + du})

J = ufl.derivative(F, du, w)

# compiler options to improve performance
cffi_options = ["-Ofast", "-march=native"]
jit_options = {"cffi_extra_compile_args": cffi_options,
               "cffi_libraries": ["m"]}
# compiled forms for rhs and tangen system
F_compiled = form(F, jit_options=jit_options)
J_compiled = form(J, jit_options=jit_options)


# Solve contact problem using Nitsche's method
log.set_log_level(log.LogLevel.WARNING)
size = mesh.comm.size
u.name = 'displacement'
T0.name = 'temperature'


search_mode = [ContactMode.ClosestPoint for i in range(len(contact_pairs))]
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
def compute_jacobian_matrix(x, A, coeffs):
    A.zeroEntries()
    with Timer("~~Contact: Contact contributions (in assemble matrix)"):
        contact_problem.assemble_matrix(A, V)
    with Timer("~~Contact: Standard contributions (in assemble matrix)"):
        assemble_matrix(A, J_compiled)
    A.assemble()


# create vector and matrix
A = contact_problem.create_matrix(J_compiled)
b = create_vector(F_compiled)


# Set up snes solver for nonlinear solver
newton_solver = NewtonSolver(mesh.comm, A, b, contact_problem.coeffs)
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
vtx = io.VTXWriter(mesh.comm, "results/box_key_thermo_mech.bp", [u, T0], "bp4")
vtx.write(0)
for i in range(steps):
    with Timer("~Contact: Solve thermal"):
        Tproblem.solve()
    with Timer("~Contact: Solve contact"):

        n, converged = newton_solver.solve(du)
        du.x.scatter_forward()
        u.x.array[:] += du.x.array[:]
        contact_problem.update_contact_detection(u)
        A = contact_problem.create_matrix(J_compiled)
        A.setNearNullSpace(null_space)
        newton_solver.set_petsc_matrix(A)
        # take a fraction of du as initial guess
        # this is to ensure non-singular matrices in the case of no Dirichlet boundary
        du.x.array[:] = 0.1 * du.x.array[:]
        contact_problem.update_contact_data(du)
    vtx.write(i + 1)
vtx.close()
timer.stop()
list_timings(MPI.COMM_WORLD, [TimingType.wall])
