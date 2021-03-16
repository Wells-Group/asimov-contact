import argparse
import os

import dolfinx
import dolfinx.fem
import dolfinx.io
import dolfinx.log
import dolfinx.mesh
import numpy as np
import ufl
from dolfinx.cpp.mesh import CellType
from mpi4py import MPI
from petsc4py import PETSc
from ufl.tensors import as_vector

from helpers import NonlinearPDEProblem, lame_parameters, epsilon, sigma_func


def solve_manufactured(nx, ny, theta, gamma, nitsche, strain, linear_solver):
    L = 10
    mesh = dolfinx.RectangleMesh(
        MPI.COMM_WORLD,
        [np.array([0, -1, 0]), np.array([L, 1, 0])], [nx, ny],
        CellType.triangle, dolfinx.cpp.mesh.GhostMode.none)

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

    def _u_ex(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[0] = (nu + 1) / E * x[1]**4
        values[1] = (nu + 1) / E * x[0]**4
        return values

    # Interpolate exact solution into approximation space
    u_D = dolfinx.Function(V)
    u_D.interpolate(_u_ex)
    u_D.name = "u_exact"
    u_D.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    # Body force for example 5.2
    # E * ufl.pi**2 * (ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
    #  -ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])))

    # Problem specific body force and traction
    # from DOI: 10.4208/aamm.2014.m548 (Chapter 5.1)
    x = ufl.SpatialCoordinate(mesh)
    u_ex = (nu + 1) / E * ufl.as_vector((x[1]**4, x[0]**4))

    # Use UFL to derive source and traction
    sigma = sigma_func(mu, lmbda)
    f = -ufl.div(sigma(u_ex))
    g = ufl.dot(sigma(u_ex), n)

    u = ufl.TrialFunction(V) if linear_solver else dolfinx.Function(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
    F = ufl.inner(sigma(u), epsilon(v)) * dx - ufl.inner(f, v) * dx - ufl.inner(g, v) * ds

    # Nitsche for Dirichlet, theta-scheme.
    # https://www.sciencedirect.com/science/article/pii/S004578251830269X
    if nitsche:
        n = ufl.FacetNormal(mesh)
        F += -ufl.inner(sigma(u) * n, v) * ds(left_marker)\
            - theta * ufl.inner(sigma(v) * n, u - u_D) * ds(left_marker)\
            + gamma / h * ufl.inner(u - u_D, v) * ds(left_marker)
        bcs = []
    else:
        bc = dolfinx.DirichletBC(u_D, dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, left_facets))
        bcs = [bc]

    if linear_solver:
        problem = dolfinx.fem.LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=bcs,
                                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        u = problem.solve()
        u.name = "uh"
    else:
        # Create nonlinear problem
        problem = NonlinearPDEProblem(F, u, bcs)
        # Create Newton solver
        solver = dolfinx.cpp.nls.NewtonSolver(MPI.COMM_WORLD)

        # Set Newton solver options
        solver.atol = 1e-6
        solver.rtol = 1e-6
        solver.convergence_criterion = "incremental"

        # Set non-linear problem for Newton solver
        solver.setF(problem.F, problem.vector)
        solver.setJ(problem.J, problem.matrix)
        solver.set_form(problem.form)

        # Solve non-linear problem
        n, converged = solver.solve(u.vector)
        assert (converged)
        print(f"Number of interations: {n:d}")
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

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
    parser = argparse.ArgumentParser()
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
        h, E = solve_manufactured(nx, ny, theta, gamma, nitsche, strain, linear_solver)
        errors[i] = E
        hs[i] = h
    print(f"Convergence rate: {np.log(errors[:-1]/errors[1:])/np.log(hs[:-1]/hs[1:])}")
