import argparse
import os
from typing import List

import dolfinx
import dolfinx.fem
import dolfinx.geometry
import dolfinx.io
import dolfinx.log
import dolfinx.mesh
import numpy as np
import ufl
from dolfinx.cpp.mesh import CellType
from mpi4py import MPI
from petsc4py import PETSc

from helpers import NonlinearPDEProblem, epsilon, lame_parameters, sigma_func


def solve_euler_bernoulli(nx, ny, theta, gamma, linear_solver, plane_strain, nitsche):
    L = 47
    H = 2.73
    mesh = dolfinx.RectangleMesh(
        MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([L, H, 0])], [nx, ny],
        CellType.triangle, dolfinx.cpp.mesh.GhostMode.none)

    def left(x):
        return np.isclose(x[0], 0)

    def top(x):
        return np.isclose(x[1], H)

    tdim = mesh.topology.dim
    left_marker = int(1)
    top_marker = int(2)
    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, left)
    top_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, top)
    left_values = np.full(len(left_facets), left_marker, dtype=np.int32)
    top_values = np.full(len(top_facets), top_marker, dtype=np.int32)
    indices = np.concatenate([left_facets, top_facets])
    values = np.hstack([left_values, top_values])
    # Sort values to work in parallel
    sorted = np.argsort(indices)
    facet_marker = dolfinx.MeshTags(mesh, tdim - 1, indices[sorted], values[sorted])

    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))
    E = 1e5
    nu = 0.3
    h = 2 * ufl.Circumradius(mesh)
    mu_func, lambda_func = lame_parameters(plane_strain)
    mu = mu_func(E, nu)
    lmbda = lambda_func(E, nu)

    rho_g = 1e-2
    f = ufl.as_vector((0, 0))
    g = ufl.as_vector((0, -rho_g))

    sigma = sigma_func(mu, lmbda)

    u = ufl.TrialFunction(V) if linear_solver else dolfinx.Function(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
    F = ufl.inner(sigma(u), epsilon(v)) * dx - ufl.inner(f, v) * dx - ufl.inner(g, v) * ds(top_marker)
    if nitsche:
        # Nitsche for Dirichlet, theta-scheme.
        # https://www.sciencedirect.com/science/article/pii/S004578251830269X
        u_D = ufl.as_vector((0, 0))
        n = ufl.FacetNormal(mesh)
        F += -ufl.inner(sigma(u) * n, v) * ds(left_marker)\
            - theta * ufl.inner(sigma(v) * n, u - u_D) * ds(left_marker)\
            + gamma / h * ufl.inner(u - u_D, v) * ds(left_marker)
        bcs = []
    else:
        # Strong Dirichlet enforcement via lifting
        u_bc = dolfinx.Function(V)
        with u_bc.vector.localForm() as loc:
            loc.set(0)
        bc = dolfinx.DirichletBC(u_bc, dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, left_facets))
        bcs = [bc]

    # Solve as linear problem
    if linear_solver:
        problem = dolfinx.fem.LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=bcs,
                                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        u = problem.solve()
    else:
        # Create nonlinear problem
        problem = NonlinearPDEProblem(F, u, bcs)

        # Create Newton solver
        solver = dolfinx.cpp.nls.NewtonSolver(MPI.COMM_WORLD)

        # Set Newton solver options
        solver.atol = 1e-6
        solver.rtol = 1e-6
        solver.convergence_criterion = "incremental"

        # # Set non-linear problem for Newton solver
        solver.setF(problem.F, problem.vector)
        solver.setJ(problem.J, problem.matrix)
        solver.set_form(problem.form)

        # Solve non-linear problem
        n, converged = solver.solve(u.vector)
        assert (converged)
        print(f"Number of interations: {n:d}")
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                         mode=PETSc.ScatterMode.FORWARD)

    os.system("mkdir -p results")
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"results/u_euler_bernoulli_{nx}_{ny}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)

    # Evaluate deflection at L, H/2
    cells = []
    points_on_proc = []
    point = np.array([L, H / 2, 0])
    bb_tree = dolfinx.geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    cell_candidates = dolfinx.geometry.compute_collisions_point(bb_tree, point)
    # Choose one of the cells that contains the point
    cell = dolfinx.geometry.select_colliding_cells(mesh, cell_candidates, point, 1)
    # Only use evaluate for points on current processor
    if len(cell) == 1:
        points_on_proc.append(point)
        cells.append(cell[0])
    if len(points_on_proc) > 0:
        u_at_point = u.eval(points_on_proc, cells)

        # Exact solution of Euler Bernoulli:
        # https://en.wikipedia.org/wiki/Euler%E2%80%93Bernoulli_beam_theory#Cantilever_beams
        # Second order moment of area for rectangular beam
        I = H**3 / (12)
        uz = (rho_g * L**4) / (8 * E * I)
        print(f"-----{nx}x{ny}--Nitsche:{nitsche} (gamma: {gamma})----\n", f"Maximal deflection: {u_at_point[mesh.geometry.dim-1]}\n",
              f"Theoretical deflection: {-uz}\n", f"Error: {100*abs((u_at_point[mesh.geometry.dim-1]+uz)/uz)} %", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", default=1, type=np.float64, dest="theta",
                        help="Theta parameter for Nitsche, 1 symmetric, -1 skew symmetric, 0 Penalty-like")
    parser.add_argument("--gamma", default=1000, type=np.float64, dest="gamma",
                        help="Coercivity/Stabilization parameter for Nitsche condition")
    _solve = parser.add_mutually_exclusive_group(required=False)
    _solve.add_argument('--linear', dest='linear_solver', action='store_true',
                        help="Use linear solver", default=False)
    _strain = parser.add_mutually_exclusive_group(required=False)
    _strain.add_argument('--strain', dest='plane_strain', action='store_true',
                         help="Use plane strain formulation", default=False)
    _dirichlet = parser.add_mutually_exclusive_group(required=False)
    _dirichlet.add_argument('--dirichlet', dest='dirichlet', action='store_true',
                            help="Use strong Dirichlet formulation", default=False)

    args = parser.parse_args()
    theta = args.theta
    gamma = args.gamma
    plane_strain = args.plane_strain
    linear_solver = args.linear_solver
    nitsche = not args.dirichlet
    Nx = np.asarray([5 * 2**i for i in range(1, 8)], dtype=np.int32)
    Ny = np.asarray([2**i for i in range(1, 8)], dtype=np.int32)
    for (nx, ny) in zip(Nx, Ny):
        solve_euler_bernoulli(nx, ny, theta, gamma, linear_solver, plane_strain, nitsche)
