import argparse
import os

import dolfinx
import dolfinx.io
import numpy as np
import ufl
from dolfinx.cpp.mesh import CellType, GhostMode
from dolfinx.fem import NonlinearProblem
from mpi4py import MPI

from helpers import epsilon, lame_parameters, sigma_func


def solve_manufactured(nx, ny, theta, gamma, nitsche, strain, linear_solver, L=10):
    """
    Solve the manufactured problem 
    u = [(nu + 1) / E * x[1]**4, (nu + 1) / E * x[0]**4]
    on the domain [0, -1]x[L, 1]
    where u solves the linear elasticity equations (plane stress/plane strain) with
    (strong/Nitsche) Dirichlet condition at (0,y) and Neumann conditions everywhere else.
    """

    mesh = dolfinx.RectangleMesh(MPI.COMM_WORLD,
                                 [np.array([0, -1, 0]),
                                  np.array([L, 1, 0])], [nx, ny],
                                 CellType.triangle, GhostMode.none)

    def left(x):
        return np.isclose(x[0], 0)

    # Identify facets on left boundary
    tdim = mesh.topology.dim
    left_marker = 1
    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, left)
    left_values = np.full(len(left_facets), left_marker, dtype=np.int32)

    # Sort values to work in parallel
    sorted = np.argsort(left_facets)
    facet_marker = dolfinx.MeshTags(mesh, tdim - 1, left_facets[sorted], left_values[sorted])

    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))
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
    u_D = dolfinx.Function(V)

    def _u_ex(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[0] = (nu + 1) / E * x[1]**4
        values[1] = (nu + 1) / E * x[0]**4
        return values

    u_D.interpolate(_u_ex)
    u_D.name = "u_exact"
    u_D.x.scatter_forward()

    u = ufl.TrialFunction(V) if linear_solver else dolfinx.Function(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
    F = ufl.inner(sigma(u), epsilon(v)) * dx - \
        ufl.inner(f, v) * dx - ufl.inner(g, v) * ds

    # Nitsche for Dirichlet, theta-scheme.
    # https://doi.org/10.1016/j.cma.2018.05.024
    if nitsche:
        n = ufl.FacetNormal(mesh)
        F += -ufl.inner(sigma(u) * n, v) * ds(left_marker)\
            - theta * ufl.inner(sigma(v) * n, u - u_D) * ds(left_marker)\
            + gamma / h * ufl.inner(u - u_D, v) * ds(left_marker)
        bcs = []
    else:
        bc = dolfinx.DirichletBC(u_D, dolfinx.fem.locate_dofs_topological(
            V, mesh.topology.dim - 1, left_facets))
        bcs = [bc]

    if linear_solver:
        problem = dolfinx.fem.LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=bcs,
                                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        u = problem.solve()
        u.name = "uh"
    else:
        # Create nonlinear problem and Newton solver
        problem = NonlinearProblem(F, u, bcs)
        solver = dolfinx.NewtonSolver(MPI.COMM_WORLD, problem)

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

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"results/u_manufactured_{nx}_{ny}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)

    # Error computation:
    error = (u - u_D)**2 * dx
    E_L2 = np.sqrt(dolfinx.fem.assemble_scalar(error))
    print(f"{nx} {ny}: Nitsche={nitsche}, L2-error={E_L2:.2e}")

    return max(L / nx, 2 / ny), E_L2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Manufatured solution test for the linear elasticity equation"
                                     + " using Nitsche-Dirichlet boundary conditions")
    parser.add_argument("--theta", default=1, type=np.float64, dest="theta",
                        help="Theta parameter for Nitsche, 1 symmetric, -1 skew symmetric, 0 Penalty-like")
    parser.add_argument("--gamma", default=1000, type=np.float64, dest="gamma",
                        help="Coercivity/Stabilization parameter for Nitsche condition")
    _dirichlet = parser.add_mutually_exclusive_group(required=False)
    _dirichlet.add_argument('--dirichlet', dest='dirichlet', action='store_true',
                            help="Use strong Dirichlet formulation", default=False)
    _strain = parser.add_mutually_exclusive_group(required=False)
    _strain.add_argument('--strain', dest='plane_strain', action='store_true',
                         help="Use plane strain formulation", default=False)
    _solve = parser.add_mutually_exclusive_group(required=False)
    _solve.add_argument('--linear', dest='linear_solver', action='store_true',
                        help="Use linear solver", default=False)

    args = parser.parse_args()
    theta = args.theta
    gamma = args.gamma
    strain = args.plane_strain
    nitsche = not args.dirichlet
    linear_solver = args.linear_solver
    Nx = np.asarray([5 * 2**i for i in range(1, 6)], dtype=np.int32)
    Ny = np.asarray([2**i for i in range(1, 6)], dtype=np.int32)
    errors = np.zeros(len(Nx))
    hs = np.zeros(len(Nx))
    for i, (nx, ny) in enumerate(zip(Nx, Ny)):
        h, E = solve_manufactured(
            nx, ny, theta, gamma, nitsche, strain, linear_solver)
        errors[i] = E
        hs[i] = h
    print(
        f"Convergence rate: {np.log(errors[:-1]/errors[1:])/np.log(hs[:-1]/hs[1:])}")
