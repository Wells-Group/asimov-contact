# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT
import dolfinx.fem as _fem
import numpy as np
import ufl
from dolfinx import io, log
from dolfinx.fem import (Function, FunctionSpace)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.graph import create_adjacencylist
from dolfinx.io import XDMFFile
from dolfinx_contact.helpers import (epsilon, lame_parameters, sigma_func,
                                     weak_dirichlet)
from dolfinx_contact.meshing import convert_mesh, create_christmas_tree_mesh
from dolfinx_contact.parallel_mesh_ghosting import create_contact_mesh
from dolfinx_contact.unbiased.contact_problem import create_contact_solver
from dolfinx_contact.cpp import ContactMode
from mpi4py import MPI
from petsc4py import PETSc as _PETSc

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
Q = _fem.FunctionSpace(mesh, ("CG", 1))
q, r = ufl.TrialFunction(Q), ufl.TestFunction(Q)
T0 = _fem.Function(Q)
kdt = 0.1
tau = 1.0
therm = (q - T0) * r * dx + kdt * ufl.inner(ufl.grad(tau * q + (1 - tau) * T0), ufl.grad(r)) * dx

a_therm, L_therm = ufl.lhs(therm), ufl.rhs(therm)

T0.x.array[:] = 0
dofs = _fem.locate_dofs_topological(Q, entity_dim=tdim - 1, entities=facet_marker.find(3))
Tbc = _fem.dirichletbc(value=_PETSc.ScalarType((1.0)), dofs=dofs, V=Q)
dofs2 = _fem.locate_dofs_topological(Q, entity_dim=tdim - 1, entities=facet_marker.find(4))
Tbc2 = _fem.dirichletbc(value=_PETSc.ScalarType((0.0)), dofs=dofs2, V=Q)
Tproblem = LinearProblem(a_therm, L_therm, bcs=[Tbc, Tbc2], petsc_options={
    "ksp_type": "preonly", "pc_type": "lu"}, u=T0)


# Elasticity problem
V = _fem.VectorFunctionSpace(mesh, ("Lagrange", 1))
g = _fem.Constant(mesh, _PETSc.ScalarType((0, 0)))     # zero Dirichlet
t = _fem.Constant(mesh, _PETSc.ScalarType((0.2, 0.5)))  # traction
f = _fem.Constant(mesh, _PETSc.ScalarType((1.0, 0.5)))  # body force


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


# Solve contact problem using Nitsche's method
problem_parameters = {"gamma": np.float64(E * gamma), "theta": np.float64(theta)}
log.set_log_level(log.LogLevel.WARNING)
size = mesh.comm.size
outname = f"results/xmas_{tdim}D_{size}"
u.name = 'displacement'
T0.name = 'temperature'

cffi_options = ["-Ofast", "-march=native"]
jit_options = {"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]}
search_mode = [ContactMode.ClosestPoint for i in range(len(contact_pairs))]
contact_problem = create_contact_solver(ufl_form=F, u=u, mu=mu_dg, lmbda=lmbda_dg,
                                        markers=[domain_marker, facet_marker],
                                        contact_data=(surfaces, contact_pairs),
                                        bcs=[],
                                        problem_parameters=problem_parameters,
                                        search_method=search_mode,
                                        newton_options=newton_options,
                                        petsc_options=petsc_options,
                                        jit_options=jit_options,
                                        quadrature_degree=5,
                                        search_radius=np.float64(0.5))

# initialise vtx write
vtx_therm = io.VTXWriter(mesh.comm, "results/xmas_disp.bp", [contact_problem.u, T0])
vtx_mech = io.VTXWriter(mesh.comm, "results/xmas_temp.bp", [T0])
vtx_therm.write(0)
vtx_mech.write(0)
for i in range(50):
    Tproblem.solve()
    for j in range(len(contact_pairs)):
        contact_problem.contact.create_distance_map(j)

    n = contact_problem.solve()
    contact_problem.du.x.scatter_forward()
    contact_problem.u.x.array[:] += contact_problem.du.x.array[:]
    contact_problem.contact.update_submesh_geometry(u._cpp_object)
    # take a fraction of du as initial guess
    # this is to ensure non-singular matrices in the case of no Dirichlet boundary
    contact_problem.du.x.array[:] = 0.1 * contact_problem.du.x.array[:]
    vtx_therm.write(i + 1)
    vtx_mech.write(i + 1)


vtx_therm.close()
vtx_mech.close()
