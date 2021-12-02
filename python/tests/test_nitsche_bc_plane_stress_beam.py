# Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import os
import pytest
import numpy as np
import ufl
from dolfinx.fem import (Function, NonlinearProblem,
                         VectorFunctionSpace, LinearProblem, assemble_scalar)
from dolfinx.nls import NewtonSolver
from dolfinx.generation import RectangleMesh
from dolfinx.mesh import (CellType, GhostMode, MeshTags,
                          locate_entities_boundary)
from dolfinx.io import XDMFFile
from dolfinx_contact.helpers import epsilon, lame_parameters, sigma_func
from mpi4py import MPI


def solve_manufactured(nx: int, ny: int, theta: float, gamma: float,
                       strain: bool, linear_solver: bool, L: float = 10):
    """
    Solve the manufactured problem
    u = [(nu + 1) / E * x[1]**4, (nu + 1) / E * x[0]**4]
    on the domain [0, -1]x[L, 1]
    where u solves the linear elasticity equations (plane stress/plane strain) with
    (strong/Nitsche) Dirichlet condition at (0,y) and Neumann conditions everywhere else.
    """

    mesh = RectangleMesh(MPI.COMM_WORLD, [np.array([0, -1, 0]), np.array([L, 1, 0])],
                         [nx, ny], CellType.triangle, GhostMode.none)

    def left(x):
        return np.isclose(x[0], 0)

    # Identify facets on left boundary
    tdim = mesh.topology.dim
    left_marker = 1
    left_facets = locate_entities_boundary(mesh, tdim - 1, left)
    left_values = np.full(len(left_facets), left_marker, dtype=np.int32)

    # Sort values to work in parallel
    sorted = np.argsort(left_facets)
    facet_marker = MeshTags(mesh, tdim - 1, left_facets[sorted], left_values[sorted])

    V = VectorFunctionSpace(mesh, ("CG", 1))
    n = ufl.FacetNormal(mesh)
    E = 1500
    nu = 0.25
    h = ufl.Circumradius(mesh)

    mu_func, lambda_func = lame_parameters(strain)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)

    # Problem specific body force and traction
    # https://doi.org/10.4208/aamm.2014.m548 (Chapter 5.1)
    x = ufl.SpatialCoordinate(mesh)
    u_ex = (nu + 1) / E * ufl.as_vector((x[1]**4, x[0]**4))

    # Use UFL to derive source and traction
    sigma = sigma_func(mu, lmbda)
    f = -ufl.div(sigma(u_ex))
    g = ufl.dot(sigma(u_ex), n)

    # Interpolate exact solution into approximation space
    u_D = Function(V)

    def _u_ex(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[0] = (nu + 1) / E * x[1]**4
        values[1] = (nu + 1) / E * x[0]**4
        return values

    u_D.interpolate(_u_ex)
    u_D.name = "u_exact"
    u_D.x.scatter_forward()

    u = ufl.TrialFunction(V) if linear_solver else Function(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
    F = ufl.inner(sigma(u), epsilon(v)) * dx - ufl.inner(f, v) * dx - ufl.inner(g, v) * ds

    # Nitsche for Dirichlet, theta-scheme.
    # https://doi.org/10.1016/j.cma.2018.05.024
    n = ufl.FacetNormal(mesh)
    F += -ufl.inner(sigma(u) * n, v) * ds(left_marker)\
        - theta * ufl.inner(sigma(v) * n, u - u_D) * ds(left_marker)\
        + gamma / h * ufl.inner(u - u_D, v) * ds(left_marker)
    bcs = []

    if linear_solver:
        problem = LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=bcs,
                                petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        u = problem.solve()
        u.name = "uh"
    else:
        # Create nonlinear problem and Newton solver
        problem = NonlinearProblem(F, u, bcs)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)

        # Set Newton solver options
        solver.atol = 1e-6
        solver.rtol = 1e-6
        solver.convergence_criterion = "incremental"

        # Solve non-linear problem
        n, converged = solver.solve(u)
        assert (converged)
        print(f"Number of interations: {n:d}")
    u.x.scatter_forward()

    os.system("mkdir -p results")

    with XDMFFile(mesh.comm, f"results/u_manufactured_{nx}_{ny}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)

    # Error computation:
    error = (u - u_D)**2 * dx
    E_L2 = np.sqrt(mesh.comm.allreduce(assemble_scalar(error), op=MPI.SUM))
    if mesh.comm.rank == 0:
        print(f"{nx} {ny}: L2-error={E_L2:.2e}")

    return max(L / nx, 2 / ny), E_L2


@pytest.mark.parametrize("theta", [-1, 1])
@pytest.mark.parametrize("strain", [True, False])
@pytest.mark.parametrize("gamma", [10, 100, 100])
@pytest.mark.parametrize("linear_solver", [True, False])
def test_nitsche_dirichlet(theta, gamma, strain, linear_solver):
    """
    Manufatured solution test for the linear elasticity equation using Nitsche-Dirichlet 
    boundary conditions.

    Parameters
    ----------
    theta
        Theta parameter for Nitsche, 1 symmetric, -1 skew symmetric, 0 Penalty-like
    gamma
        Coercivity/Stabilization parameter for Nitsche condition
    strain
        Use plane strain formulation. If false use plane stress
    linear_solver
        Use linear solver if true, else use Newton
    """

    num_refs = 4
    Nx = np.asarray([10 * 2**i for i in range(num_refs)], dtype=np.int32)
    Ny = np.asarray([5 * 2**i for i in range(num_refs)], dtype=np.int32)
    errors = np.zeros(len(Nx))
    hs = np.zeros(len(Nx))
    for i, (nx, ny) in enumerate(zip(Nx, Ny)):
        h, E = solve_manufactured(nx, ny, theta, gamma, strain, linear_solver)
        errors[i] = E
        hs[i] = h
    rates = np.log(errors[:-1] / errors[1:]) / np.log(hs[:-1] / hs[1:])
    print(errors)
    print(f"Convergence rate: {rates}")
    assert(np.isclose(rates[-1], 2, atol=0.05))
