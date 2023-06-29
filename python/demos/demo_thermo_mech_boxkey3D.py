# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import argparse

import dolfinx.fem as _fem
import numpy as np
import ufl
from dolfinx import io, log
from dolfinx.common import Timer, TimingType, list_timings
from dolfinx.graph import create_adjacencylist
from dolfinx.io import XDMFFile
from dolfinx.mesh import GhostMode
from dolfinx_contact.helpers import (epsilon, lame_parameters, sigma_func,
                                     weak_dirichlet)
from dolfinx_contact.parallel_mesh_ghosting import create_contact_mesh
from dolfinx_contact.unbiased.contact_problem import create_contact_solver
from mpi4py import MPI
from petsc4py import PETSc as _PETSc

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
Q = _fem.FunctionSpace(mesh, ("CG", 1))
q, r = ufl.TrialFunction(Q), ufl.TestFunction(Q)
T0 = _fem.Function(Q)
kdt = 0.1
tau = 1.0
therm = (q - T0) * r * dx + kdt * ufl.inner(ufl.grad(tau * q + (1 - tau) * T0), ufl.grad(r)) * dx

a_therm, L_therm = ufl.lhs(therm), ufl.rhs(therm)

T0.x.set(0.0)
dofs = _fem.locate_dofs_topological(Q, entity_dim=tdim - 1, entities=facet_marker.find(dirichlet_bdy1))
Tbc = _fem.dirichletbc(value=_PETSc.ScalarType((1.0)), dofs=dofs, V=Q)
dofs2 = _fem.locate_dofs_topological(Q, entity_dim=tdim - 1, entities=facet_marker.find(dirichlet_bdy2))
Tbc2 = _fem.dirichletbc(value=_PETSc.ScalarType((0.0)), dofs=dofs2, V=Q)
Tproblem = _fem.petsc.LinearProblem(a_therm, L_therm, bcs=[Tbc, Tbc2], petsc_options=petsc_options, u=T0)


# Elasticity problem
V = _fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
g = _fem.Constant(mesh, _PETSc.ScalarType((0, 0, 0)))     # zero Dirichlet


# contact surface data
contact_pairs = [(0, 1), (1, 0)]
data = np.array([contact_bdy_1, contact_bdy_2], dtype=np.int32)
offsets = np.array([0, 2], dtype=np.int32)
surfaces = create_adjacencylist(data, offsets)


# Function, TestFunction, TrialFunction and measures
u = _fem.Function(V)
v = ufl.TestFunction(V)

# Compute lame parameters
E = 1e4
nu = 0.2
mu_func, lambda_func = lame_parameters(True)
mu = mu_func(E, nu)
lmbda = lambda_func(E, nu)


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


# Solve contact problem using Nitsche's method
problem_parameters = {"gamma": E * gamma, "theta": theta, "mu": mu, "lambda": lmbda}
log.set_log_level(log.LogLevel.WARNING)
size = mesh.comm.size
outname = f"results/xmas_{tdim}D_{size}"
u.name = 'displacement'
T0.name = 'temperature'

cffi_options = ["-Ofast", "-march=native"]
jit_options = {"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]}
contact_problem = create_contact_solver(ufl_form=F, u=u, markers=[domain_marker, facet_marker],
                                        contact_data=(surfaces, contact_pairs),
                                        bcs=(np.empty(shape=(2, 0), dtype=np.int32), []),
                                        problem_parameters=problem_parameters,
                                        raytracing=False,
                                        newton_options=newton_options,
                                        petsc_options=petsc_options,
                                        jit_options=jit_options,
                                        quadrature_degree=5,
                                        search_radius=np.float64(0.5))

# initialise vtx write
vtx = io.VTXWriter(mesh.comm, "results/box_key_thermo_mech.bp", [contact_problem.u, T0])
vtx.write(0)
for i in range(steps):
    with Timer("~Contact: Solve thermal"):
        Tproblem.solve()
    with Timer("~Contact: Solve contact"):
        for j in range(len(contact_pairs)):
            contact_problem.contact.create_distance_map(j)

        n = contact_problem.solve()
        contact_problem.du.x.scatter_forward()
        contact_problem.u.x.array[:] += contact_problem.du.x.array[:]
        contact_problem.contact.update_submesh_geometry(u._cpp_object)
        # take a fraction of du as initial guess
        # this is to ensure non-singular matrices in the case of no Dirichlet boundary
        contact_problem.du.x.array[:] = 0.1 * contact_problem.du.x.array[:]
    vtx.write(i + 1)
vtx.close()
timer.stop()
list_timings(MPI.COMM_WORLD, [TimingType.wall])
