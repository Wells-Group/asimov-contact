# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT
#
import numpy as np

from basix.ufl import element
from dolfinx.fem import (Constant, dirichletbc, Function,
                         functionspace, locate_dofs_topological)
from dolfinx.graph import adjacencylist
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_mesh
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import grad, Identity, inner, Mesh, Measure, TestFunction, tr, sym

from dolfinx_contact.unbiased.contact_problem import create_contact_solver

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
v = TestFunction(V)

# Compute lame parameters
E = 1e4
nu = 0.2
mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

# Create variational form without contact contributions


def epsilon(v):
    return sym(grad(v))


def sigma(v):
    return (2.0 * mu * epsilon(v) + lmbda * tr(epsilon(v)) * Identity(len(v)))


F = inner(sigma(u), epsilon(v)) * dx

# Nitsche parameters
gamma = 10
theta = 1  # 1 - symmetric
problem_parameters = {"gamma": np.float64(E * gamma), "theta": np.float64(theta),
                      "mu": np.float64(mu), "lambda": np.float64(lmbda)}

# boundary conditions
g = Constant(mesh, ScalarType((0, 0, 0)))     # zero Dirichlet
dofs_g = locate_dofs_topological(
    V, tdim - 1, facet_marker.find(dirichlet_bdy_2))
d = Constant(mesh, ScalarType((0, 0, -0.2)))  # vertical displacement
dofs_d = locate_dofs_topological(
    V, tdim - 1, facet_marker.find(dirichlet_bdy_1))
bcs = [dirichletbc(d, dofs_d, V), dirichletbc(g, dofs_g, V)]

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


# Solver options
ksp_tol = 1e-10
newton_tol = 1e-7

# non-linear solver options
newton_options = {"relaxation_parameter": 1,
                  "atol": newton_tol,
                  "rtol": newton_tol,
                  "convergence_criterion": "residual",
                  "max_it": 50,
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

# compiler options to improve performance
cffi_options = ["-Ofast", "-march=native"]
jit_options = {"cffi_extra_compile_args": cffi_options,
               "cffi_libraries": ["m"]}


# create contact solver
contact_problem = create_contact_solver(ufl_form=F, u=u, markers=[domain_marker, facet_marker],
                                        contact_data=(surfaces, contact_pairs),
                                        bcs=bcs,
                                        problem_parameters=problem_parameters,
                                        raytracing=False,
                                        newton_options=newton_options,
                                        petsc_options=petsc_options,
                                        jit_options=jit_options,
                                        quadrature_degree=5,
                                        search_radius=np.float64(0.5))

# Perform contact detection
for j in range(len(contact_pairs)):
    contact_problem.contact.create_distance_map(j)

# solve non-linear problem
n = contact_problem.solve()

# update displacement according to computed increment
contact_problem.du.x.scatter_forward()  # only relevant in parallel
contact_problem.u.x.array[:] += contact_problem.du.x.array[:]


# write resulting diplacements to file
with XDMFFile(mesh.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    contact_problem.u.name = "u"
    xdmf.write_function(contact_problem.u)
