# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
from dolfinx import default_scalar_type, log
from dolfinx.cpp.nls.petsc import NewtonSolver
from dolfinx.fem import (
    Constant,
    DirichletBC,
    Form,
    Function,
    dirichletbc,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_vector,
    set_bc,
)
from dolfinx.graph import adjacencylist
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.la import create_petsc_vector_wrap, vector
from dolfinx.mesh import locate_entities_boundary
from dolfinx_contact.cpp import MeshTie, Problem
from dolfinx_contact.helpers import (
    epsilon,
    lame_parameters,
    rigid_motions_nullspace_subdomains,
)
from dolfinx_contact.meshing import create_split_box_2D, horizontal_sine
from dolfinx_contact.parallel_mesh_ghosting import create_contact_mesh
from ufl import (
    Identity,
    Measure,
    TestFunction,
    TrialFunction,
    derivative,
    grad,
    inner,
    lhs,
    rhs,
    sym,
    tr,
)


class ThermoElasticProblem:
    __slots__ = [
        "_l",
        "_j",
        "_bcs",
        "_meshties",
        "_b",
        "_b_petsc",
        "_mat_a",
        "_u",
        "_T",
        "_lmbda",
        "_mu",
        "_gamma",
        "_theta",
        "_alpha",
    ]

    def __init__(
        self,
        a: Form,
        j: Form,
        bcs: list[DirichletBC],
        meshties: MeshTie,
        subdomains,
        u: Function,
        T: Function,
        lmbda: Function,
        mu: Function,
        alpha: Function,
        gamma: np.float64,
        theta: np.float64,
        num_domains: int = 2,
    ):
        """
        Create a MeshTie problem

        Args:
            l:          The form describing the residual
            j:          The form describing the jacobian
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
        self._l = a
        self._j = j
        self._bcs = bcs
        self._meshties = meshties
        self._mat_a = self._meshties.create_matrix(self._j._cpp_object)
        self._u = u
        self._T = T
        self._lmbda = lmbda
        self._mu = mu
        self._gamma = gamma
        self._theta = theta
        self._alpha = alpha

        # Create PETSc rhs vector
        self._b = vector(
            a.function_spaces[0].dofmap.index_map,
            a.function_spaces[0].dofmap.index_map_bs,
        )
        self._b_petsc = create_petsc_vector_wrap(self._b)

        # Initialise the input data for integration kernels
        self._meshties.generate_kernel_data(
            Problem.ThermoElasticity,
            a.function_spaces[0],
            {
                "lambda": lmbda._cpp_object,
                "mu": mu._cpp_object,
                "alpha": alpha._cpp_object,
            },
            gamma,
            theta,
        )

        # Build near null space preventing rigid body motion of individual components
        tags = np.unique(subdomains.values)
        ns = rigid_motions_nullspace_subdomains(u.function_space, subdomains, tags, num_domains)
        self._mat_a.setNearNullSpace(ns)

    def f(self, x, _b):
        """Function for computing the residual vector.

        Args:
           x: The vector containing the latest solution.
           b: The residual vector.

        """
        # Avoid long log output coming from the custom kernel
        log.set_log_level(log.LogLevel.OFF)

        # Generate input data for custom kernel.
        self._meshties.update_kernel_data(
            {"T": self._T._cpp_object, "u": self._u._cpp_object},
            Problem.ThermoElasticity,
        )

        # Assemble residual vector
        self._b_petsc.zeroEntries()
        self._b_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self._meshties.assemble_vector(
            self._b_petsc, self._l.function_spaces[0], Problem.ThermoElasticity
        )  # custom kernel
        assemble_vector(self._b_petsc, self._l)  # standard kernels

        # Apply boundary condition
        apply_lifting(self._b_petsc, [self._j], bcs=[self._bcs], x0=[x], scale=-1.0)
        self._b_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self._b_petsc, self._bcs, x, -1.0)

        # Restore log level info for monitoring Newton solver
        log.set_log_level(log.LogLevel.INFO)

    def j(self, _x, _matrix):
        """Function for computing the Jacobian matrix.

        Args:
           x: The vector containing the latest solution.
           A: The matrix to assemble into.

        """
        log.set_log_level(log.LogLevel.OFF)
        self._mat_a.zeroEntries()
        self._meshties.assemble_matrix(self._mat_a, self._j.function_spaces[0], Problem.ThermoElasticity)
        assemble_matrix(self._mat_a, self._j, self._bcs)
        self._mat_a.assemble()
        log.set_log_level(log.LogLevel.INFO)

    def form(
        self,
        x: PETSc.Vec,  # type: ignore
    ) -> None:
        """This function is called before the residual or Jacobian is
        computed in the NewtonSolver. Used to update ghost values.

        Args:
           x: The vector containing the latest solution

        """
        x.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,  # type: ignore
            mode=PETSc.ScatterMode.FORWARD,  # type: ignore
        )


L = 2.0
H = 2.0
nx = 20
ny = 20
# parameter for surface approximation
num_segments = (
    8 * np.ceil(5.0 / 1.2).astype(np.int32),
    8 * np.ceil(5.0 / (1.2 * 0.7)).astype(np.int32),
)
fname = "./meshes/split_box"
create_split_box_2D(
    fname,
    res=0.1,
    L=1.0,
    H=1.0,
    domain_1=[0, 1, 5, 4],
    domain_2=[4, 5, 2, 3],
    x0=[0, 0.5],
    x1=[1.0, 0.7],
    curve_fun=horizontal_sine,
    num_segments=num_segments,
    quads=False,
)

# read in mesh and markers
with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(name="Grid")
tdim = mesh.topology.dim
gdim = mesh.geometry.dim
mesh.topology.create_connectivity(tdim - 1, 0)
mesh.topology.create_connectivity(tdim - 1, tdim)
with XDMFFile(MPI.COMM_WORLD, f"{fname}.xdmf", "r") as xdmf:
    domain_marker = xdmf.read_meshtags(mesh, name="domain_marker")
    facet_marker = xdmf.read_meshtags(mesh, name="contact_facets")

if mesh.comm.size > 1:
    mesh, facet_marker, domain_marker = create_contact_mesh(mesh, facet_marker, domain_marker, [3, 4, 7, 8])

# compiler options to improve performance
cffi_options = ["-Ofast", "-march=native"]
jit_options = {"cffi_extra_compile_args": cffi_options, "cffi_libraries": ["m"]}

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
    "ksp_norm_type": "unpreconditioned",
}

# measures
dx = Measure("dx", domain=mesh)
ds = Measure("ds", domain=mesh)


# Thermal problem ufl (implicit Euler)
Q = functionspace(mesh, ("CG", 1))
q, r = TrialFunction(Q), TestFunction(Q)
T0 = Function(Q)
kdt = 0.01
Q0 = functionspace(mesh, ("DG", 0))
kdt_custom = Function(Q0)
kdt_custom.interpolate(lambda x: np.full((1, x.shape[1]), kdt))

therm = (q - T0) * r * dx + kdt * inner(grad(q), grad(r)) * dx
a_therm, L_therm = lhs(therm), rhs(therm)
T0.x.array[:] = 0

dirichlet_facets = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], 0))
dofs = locate_dofs_topological(Q, tdim - 1, dirichlet_facets)
Tbc = dirichletbc(value=default_scalar_type((1.0)), dofs=dofs, V=Q)

# surface data for Nitsche
gamma = 10
theta = 1
contact = [(0, 2), (0, 3), (1, 2), (1, 3), (2, 0), (2, 1), (3, 0), (3, 1)]
data = np.array([3, 4, 7, 8], dtype=np.int32)
offsets = np.array([0, 4], dtype=np.int32)
surfaces = adjacencylist(data, offsets)

# initialise meshties
meshties = MeshTie([facet_marker._cpp_object], surfaces, contact, mesh._cpp_object, quadrature_degree=3)
meshties.generate_kernel_data(
    Problem.Poisson,
    Q._cpp_object,
    {"T": T0._cpp_object, "kdt": kdt_custom._cpp_object},
    gamma,
    theta,
)

# Create matrix and vector
a_therm = form(a_therm, jit_options=jit_options)
L_therm = form(L_therm, jit_options=jit_options)
mat_therm = meshties.create_matrix(a_therm._cpp_object)
vec_therm = create_vector(L_therm)


# Thermal problem: functions for updating matrix and vector
def assemble_mat_therm(A):
    A.zeroEntries()
    meshties.assemble_matrix(A, Q._cpp_object, Problem.Poisson)
    assemble_matrix(A, a_therm, bcs=[Tbc])
    A.assemble()


def assemble_vec_therm(b):
    b.zeroEntries()
    b.ghostUpdate(
        addv=PETSc.InsertMode.INSERT,  # type: ignore
        mode=PETSc.ScatterMode.FORWARD,
    )  # type: ignore
    assemble_vector(b, L_therm)

    # Apply boundary condition and scatter reverse
    apply_lifting(b, [a_therm], bcs=[[Tbc]], scale=1.0)
    b.ghostUpdate(
        addv=PETSc.InsertMode.ADD,  # type: ignore
        mode=PETSc.ScatterMode.REVERSE,
    )  # type: ignore
    set_bc(b, [Tbc])


# Thermal problem: create linear solver
ksp_therm = PETSc.KSP().create(mesh.comm)  # type: ignore
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
V = functionspace(mesh, ("Lagrange", 1, (gdim,)))
v, w = TestFunction(V), TrialFunction(V)
u = Function(V)

# Compute lame parameters
E = 1e3
nu = 0.1
alpha = 1.0
mu_func, lambda_func = lame_parameters(False)
V2 = functionspace(mesh, ("DG", 0))
lmbda = Function(V2)
lmbda.interpolate(lambda x: np.full((1, x.shape[1]), lambda_func(E, nu)))
mu = Function(V2)
mu.interpolate(lambda x: np.full((1, x.shape[1]), mu_func(E, nu)))
alpha_c = Function(V2)
alpha_c.interpolate(lambda x: np.full((1, x.shape[1]), alpha))


def eps(w):
    return sym(grad(w))


def sigma(w, T):
    return (lmbda * tr(eps(w)) - alpha * (3 * lmbda + 2 * mu) * T) * Identity(tdim) + 2.0 * mu * eps(w)


# Elasticity problem: ufl
F = inner(sigma(u, T0), epsilon(v)) * dx
J = derivative(F, u, w)
# boundary conditions
dirichlet_facets2 = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], 1))
g = Constant(mesh, default_scalar_type((0.0, 0.0)))
dofs_e = locate_dofs_topological(V, tdim - 1, dirichlet_facets)
dofs_e2 = locate_dofs_topological(V, tdim - 1, dirichlet_facets2)
bcs = [dirichletbc(g, dofs_e2, V)]

# matrix and vector
F = form(F, jit_options=jit_options)
J = form(J, jit_options=jit_options)

elastic_problem = ThermoElasticProblem(
    F,
    J,
    bcs,
    meshties,
    domain_marker,
    u,
    T0,
    lmbda,
    mu,
    alpha_c,
    np.float64(gamma * E),
    np.float64(theta),
)

# Set up Newton sovler
newton_solver = NewtonSolver(mesh.comm)

# Set matrix-vector computations
newton_solver.setF(elastic_problem.f, elastic_problem._b_petsc)
newton_solver.setJ(elastic_problem.j, elastic_problem._mat_a)
newton_solver.set_form(elastic_problem.form)

# Set Newton options
newton_solver.rtol = 1e-7
newton_solver.atol = 1e-7
newton_solver.report = True
log.set_log_level(log.LogLevel.INFO)

# Set Krylov solver options
ksp = newton_solver.krylov_solver
opts = PETSc.Options()  # type: ignore
option_prefix = ksp.getOptionsPrefix()
opts.prefixPush(option_prefix)
for k, v in petsc_options.items():
    opts[k] = v
opts.prefixPop()
ksp.setFromOptions()

time_steps = 40
T0.name = "temperature"
u.name = "displacement"
vtx = VTXWriter(mesh.comm, "results/thermo_mech.bp", [u, T0], "bp4")
vtx.write(0)

for i in range(time_steps):
    assemble_mat_therm(mat_therm)
    assemble_vec_therm(vec_therm)
    ksp_therm.solve(vec_therm, T0.x.petsc_vec)
    T0.x.scatter_forward()
    newton_solver.solve(u.x.petsc_vec)
    u.x.scatter_forward()
    vtx.write(i + 1)
vtx.close()
