# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT
#
import numpy as np

from basix.ufl import element
from dolfinx import default_scalar_type
from dolfinx.fem import (Constant, dirichletbc, DirichletBC, form, Function,
                         FunctionSpace,
                         functionspace, locate_dofs_topological)
from dolfinx.fem.forms import Form
from dolfinx.fem.petsc import assemble_vector, apply_lifting, set_bc, assemble_matrix
from dolfinx.graph import adjacencylist
from dolfinx.io import XDMFFile
from dolfinx import log
from dolfinx.la import vector, create_petsc_vector_wrap
from dolfinx.mesh import create_mesh
from dolfinx.cpp.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (derivative, grad, Identity, inner, Mesh,
                 Measure, TestFunction, TrialFunction, tr, sym)

from dolfinx_contact.cpp import ContactMode, MeshTie
from dolfinx_contact.helpers import rigid_motions_nullspace_subdomains


class MeshTieProblem:
    __slots__ = ["_l", "_j", "_bcs", "_meshties", "_b", "_b_petsc", "_matA", "_u",
                 "_lmbda", "_mu", "_gamma", "_theta"]

    def __init__(self, L: Form, J: Form, bcs: list[DirichletBC], meshties: MeshTie, subdomains,
                 u: Function, lmbda: Function, mu: Function, gamma: np.float64, theta: np.float64):
        """
        Create a MeshTie problem

        Args:
            L:          The form describing the residual
            J:          The form describing the jacobian
            bcs:        The boundary conditions
            meshties:   The MeshTie class describing the tied surfaces
            subdomains: The domain marker labelling the individual components
            u:          The displacement function
            lmbda:      The lame parameter lambda
            mu:         The lame parameter mu
            gamma:      The Nitsche parameter
            theta:      The parameter selecting version of Nitsche (1 - symmetric, -1
                        anti-symmetric, 0 - penalty-like)
        """
        # Initialise class from input
        self._l = L
        self._j = J
        self._bcs = bcs
        self._meshties = meshties
        self._matA = self._meshties.create_matrix(self._j._cpp_object)
        self._u = u
        self._lmbda = lmbda
        self._mu = mu
        self._gamma = gamma
        self._theta = theta

        # Create PETSc rhs vector
        self._b = vector(L.function_spaces[0].dofmap.index_map, L.function_spaces[0].dofmap.index_map_bs)
        self._b_petsc = create_petsc_vector_wrap(self._b)

        # Initialise the input data for integration kernels
        self._meshties.generate_meshtie_data_matrix_only(lmbda._cpp_object, mu._cpp_object, gamma, theta)

        # Build near null space preventing rigid body motion of individual components
        tags = np.unique(subdomains.values)
        ns = rigid_motions_nullspace_subdomains(u.function_space, subdomains, tags, len(tags))
        self._matA.setNearNullSpace(ns)

    def F(self, x, b):
        """Function for computing the residual vector.

        Args:
           x: The vector containing the latest solution.
           b: The residual vector.

        """
        # Avoid long log output coming from the custom kernel
        log.set_log_level(log.LogLevel.OFF)

        # Generate input data for custom kernel.
        self._meshties.generate_meshtie_data(
            self._u._cpp_object, self._lmbda._cpp_object, self._mu._cpp_object, gamma, theta)

        # Assemble residual vector
        self._b_petsc.zeroEntries()
        self._b_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self._meshties.assemble_vector(self._b_petsc)  # custom kernel
        assemble_vector(self._b_petsc, self._l)  # standard kernels

        # Apply boundary condition
        apply_lifting(self._b_petsc, [self._j], bcs=[self._bcs], x0=[x], scale=-1.0)
        self._b_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self._b_petsc, self._bcs, x, -1.0)

        # Restore log level info for monitoring Newton solver
        log.set_log_level(log.LogLevel.INFO)

    def J(self, x, A):
        """Function for computing the Jacobian matrix.

        Args:
           x: The vector containing the latest solution.
           A: The matrix to assemble into.

        """
        log.set_log_level(log.LogLevel.OFF)
        self._matA.zeroEntries()
        self._meshties.assemble_matrix(self._matA)
        assemble_matrix(self._matA, self._j, self._bcs)
        self._matA.assemble()
        log.set_log_level(log.LogLevel.INFO)

    def form(self,
             x: PETSc.Vec  # type: ignore
             ) -> None:
        """This function is called before the residual or Jacobian is
        computed in the NewtonSolver. Used to update ghost values.

        Args:
           x: The vector containing the latest solution

        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore


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
dirichlet_bdry_1 = 8  # top face
dirichlet_bdry_2 = 2  # bottom face

contact_bdry_1 = 12  # top contact interface
contact_bdry_2 = 6  # bottom contact interface

# measures
dx = Measure("dx", domain=mesh, subdomain_data=domain_marker)
ds = Measure("ds", domain=mesh, subdomain_data=facet_marker)


# Elasticity problem

# Function space
gdim = mesh.topology.dim
V = functionspace(mesh, ("Lagrange", 1, (gdim,)))

# Function, test and trial functions
u = Function(V)
v = TestFunction(V)
w = TrialFunction(V)

# Compute lame parameters
E = 1e4
nu = 0.2
mu_val = E / (2 * (1 + nu))
lmbda_val = E * nu / ((1 + nu) * (1 - 2 * nu))

# DG functions for material parameters
V0 = FunctionSpace(mesh, ("DG", 0))
mu = Function(V0)
lmbda = Function(V0)
mu.interpolate(lambda x: np.full((1, x.shape[1]), mu_val))
lmbda.interpolate(lambda x: np.full((1, x.shape[1]), lmbda_val))

# Create variational form without contact contributions


def epsilon(v):
    return sym(grad(v))


def sigma(v):
    return (2.0 * mu * epsilon(v) + lmbda * tr(epsilon(v)) * Identity(len(v)))


F = inner(sigma(u), epsilon(v)) * dx

# Nitsche parameters
gamma = 10 * E
theta = 1  # 1 - symmetric


# boundary conditions
g = Constant(mesh, default_scalar_type((0, 0, 0)))     # zero Dirichlet
dofs_g = locate_dofs_topological(
    V, tdim - 1, facet_marker.find(dirichlet_bdry_2))
d = Constant(mesh, default_scalar_type((0, -0.2, 0)))  # vertical displacement
dofs_d = locate_dofs_topological(
    V, tdim - 1, facet_marker.find(dirichlet_bdry_1))
bcs = [dirichletbc(d, dofs_d, V), dirichletbc(g, dofs_g, V)]

# contact surface data
# stored in adjacency list to allow for using multiple meshtags to mark
# contact surfaces. In this case only one meshtag is used, hence offsets has length 2 and
# the second value in offsets is 2 (=2 tags in first and only meshtag).
# The surface with tags [contact_bdry_1, contact_bdry_2] both can be found in this meshtag
data = np.array([contact_bdry_1, contact_bdry_2], dtype=np.int32)
offsets = np.array([0, 2], dtype=np.int32)
surfaces = adjacencylist(data, offsets)
# For unbiased computation the contact detection is performed in both directions
contact_pairs = [(0, 1), (1, 0)]

# compiler options to improve performance
cffi_options = ["-Ofast", "-march=native"]
jit_options = {"cffi_extra_compile_args": cffi_options,
               "cffi_libraries": ["m"]}

# Derive form for ufl part of Jacobian
J = derivative(F, u, w)

# compiled forms for rhs and tangent system
F_compiled = form(F, jit_options=jit_options)
J_compiled = form(J, jit_options=jit_options)

search_mode = [ContactMode.ClosestPoint, ContactMode.ClosestPoint]

# Initialise MeshTie class and generate MeshTie problem
meshties = MeshTie([facet_marker._cpp_object], surfaces, contact_pairs, V._cpp_object, quadrature_degree=5)
problem = MeshTieProblem(F_compiled, J_compiled, bcs, meshties, domain_marker,
                         u, lmbda, mu, np.float64(gamma), np.float64(theta))

# Set up Newton sovler
newton_solver = NewtonSolver(mesh.comm)

# Set matrix-vector computations
newton_solver.setF(problem.F, problem._b_petsc)
newton_solver.setJ(problem.J, problem._matA)
newton_solver.set_form(problem.form)

# Set Newton options
newton_solver.rtol = 1e-7
newton_solver.atol = 1e-7


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


# Set Krylov solver options
ksp = newton_solver.krylov_solver
opts = PETSc.Options()  # type: ignore
option_prefix = ksp.getOptionsPrefix()
opts.prefixPush(option_prefix)
for k, v in petsc_options.items():
    opts[k] = v
opts.prefixPop()
ksp.setFromOptions()

# Start Newton solver with loglevel info for monitoring
newton_solver.report = True
log.set_log_level(log.LogLevel.INFO)
newton_solver.solve(u.vector)
u.x.scatter_forward()
