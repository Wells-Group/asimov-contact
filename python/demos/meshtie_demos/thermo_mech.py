# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import numpy as np

from dolfinx import default_scalar_type
from dolfinx.io import VTXWriter
from dolfinx.fem import Constant, dirichletbc, form, functionspace, Function, locate_dofs_topological
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc, create_matrix, create_vector
from dolfinx.mesh import (CellType, GhostMode, create_rectangle, locate_entities_boundary)
from dolfinx_contact.helpers import epsilon, lame_parameters, rigid_motions_nullspace
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (grad, inner, sym, tr, Identity, lhs, rhs, Measure, TrialFunction, TestFunction)

L = 2.0
H = 2.0
nx = 20
ny = 20

mesh = create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([L, H])], [nx, ny],
                        CellType.triangle, ghost_mode=GhostMode.none)
tdim = mesh.topology.dim
gdim = mesh.geometry.dim

# compiler options to improve performance
cffi_options = ["-Ofast", "-march=native"]
jit_options = {"cffi_extra_compile_args": cffi_options,
               "cffi_libraries": ["m"]}

# Linear solver options
ksp_tol = 1e-10
petsc_options = {
    "ksp_type": "cg",
    "ksp_rtol": ksp_tol,
    "ksp_atol": ksp_tol,
    "pc_type": "gamg",
    "mg_levels_ksp_type": "chebyshev",
    "mg_levels_pc_type": "jacobi",
    "pc_gamg_type": "agg",
    "pc_gamg_coarse_eq_limit": 100,
    "pc_gamg_agg_nsmooths": 1,
    "pc_gamg_threshold": 1e-3,
    "pc_gamg_square_graph": 2,
    "ksp_norm_type": "unpreconditioned"
}

# measures
dx = Measure("dx", domain=mesh)
ds = Measure("ds", domain=mesh)


# Thermal problem ufl (implicit Euler)
Q = functionspace(mesh, ("CG", 1))
q, r = TrialFunction(Q), TestFunction(Q)
T0 = Function(Q)
kdt = 0.1

therm = (q - T0) * r * dx + kdt * inner(grad(q), grad(r)) * dx
a_therm, L_therm = lhs(therm), rhs(therm)
T0.x.array[:] = 0

dirichlet_facets = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[0], 0))
dofs = locate_dofs_topological(Q, tdim - 1, dirichlet_facets)
Tbc = dirichletbc(value=default_scalar_type((1.0)), dofs=dofs, V=Q)

# Create matrix and vector
a_therm = form(a_therm, jit_options=jit_options)
L_therm = form(L_therm, jit_options=jit_options)
mat_therm = create_matrix(a_therm)
vec_therm = create_vector(L_therm)

# Thermal problem: functions for updating matrix and vector
def assemble_mat_therm(A):
    A.zeroEntries()
    assemble_matrix(A, a_therm, bcs=[Tbc])
    A.assemble()


def assemble_vec_therm(b):
    b.zeroEntries()
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
    assemble_vector(b, L_therm)

    # Apply boundary condition and scatter reverse
    apply_lifting(b, [a_therm], bcs=[[Tbc]], scale=1.0)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
    set_bc(b, [Tbc])

# Thermal problem: create linear solver
ksp_therm = PETSc.KSP().create(mesh.comm)
prefix_therm = "Solver_thermal_"
ksp_therm.setOptionsPrefix(prefix_therm)
opts = PETSc.Options()  # type: ignore
opts.prefixPush(ksp_therm.getOptionsPrefix())
for key in petsc_options:
    opts[key] = petsc_options[key]
opts.prefixPop()
ksp_therm.setFromOptions()
ksp_therm.setOperators(mat_therm)
ksp_therm.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))


# Elasticity problem
V = functionspace(mesh, ("Lagrange", 1, (gdim, )))
v, w = TestFunction(V), TrialFunction(V)
u = Function(V)

# Compute lame parameters
E = 1e3
nu = 0.1
mu_func, lambda_func = lame_parameters(False)
V2 = functionspace(mesh, ("DG", 0))
lmbda = Function(V2)
lmbda.interpolate(lambda x: np.full((1, x.shape[1]), lambda_func(E, nu)))
mu = Function(V2)
mu.interpolate(lambda x: np.full((1, x.shape[1]), mu_func(E, nu)))

def eps(w):
    return sym(grad(w))


alpha = 0.1


def sigma(w, T):
    return (lmbda * tr(eps(w)) - alpha * (3 * lmbda + 2 * mu) * T) * Identity(tdim) + 2.0 * mu * eps(w)


# Elasticity problem: ufl
F = inner(sigma(w, T0), epsilon(v)) * dx
J, F = lhs(F), rhs(F)

# boundary conditions
g = Constant(mesh, default_scalar_type((0.0, 0.0)))
dofs_e = locate_dofs_topological(V, tdim - 1, dirichlet_facets)
bcs = [dirichletbc(g, dofs_e, V)]

# matrix and vector
F = form(F, jit_options=jit_options)
J = form(J, jit_options=jit_options)
A = create_matrix(J)
b = create_vector(F)

# Set rigid motion nullspace
null_space = rigid_motions_nullspace(V)
A.setNearNullSpace(null_space)


# Elasticity problem: functions for updating matrix and vector
def assemble_mat_el(A):
    A.zeroEntries()
    assemble_matrix(A, J, bcs=bcs)
    A.assemble()


def assemble_vec_el(b):
    b.zeroEntries()
    b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
    assemble_vector(b, F)

    # Apply boundary condition and scatter reverse
    apply_lifting(b, [J], bcs=[bcs], scale=1.0)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore
    set_bc(b, bcs)

# Elasticity problem: create linear solver
ksp_el = PETSc.KSP().create(mesh.comm)
prefix_el = "Solver_elasticity_"
ksp_el.setOptionsPrefix(prefix_el)
opts = PETSc.Options()  # type: ignore
opts.prefixPush(ksp_el.getOptionsPrefix())
for key in petsc_options:
    opts[key] = petsc_options[key]
opts.prefixPop()
ksp_el.setFromOptions()
ksp_el.setOperators(A)
ksp_el.setMonitor(lambda _, its, rnorm: print(f"Iteration: {its}, rel. residual: {rnorm}"))

time_steps = 50
T0.name = 'temperature'
u.name = 'displacement'
vtx = VTXWriter(mesh.comm, "results/thermo_mech.bp", [u, T0], "bp4")
vtx.write(0)

for i in range(time_steps):
    assemble_mat_therm(mat_therm)
    assemble_vec_therm(vec_therm)
    ksp_therm.solve(vec_therm, T0.vector)
    assemble_mat_el(A)
    assemble_vec_el(b)
    ksp_el.solve(b, u.vector)
    vtx.write(i + 1)
vtx.close()

