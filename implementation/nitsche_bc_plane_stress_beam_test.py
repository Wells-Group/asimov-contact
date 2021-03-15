from typing import List
import dolfinx
import dolfinx.fem
import dolfinx.io
import dolfinx.log
import dolfinx.mesh
import numpy as np
import os
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.cpp.mesh import CellType
import argparse


class NonlinearPDEProblem:
    """Nonlinear problem class for solving the non-linear problem
    F(u, v) = 0 for all v in V
    """
    def __init__(self, F: ufl.form.Form, u: dolfinx.Function,
                 bcs: List[dolfinx.DirichletBC]):
        """
        Input:
        - F: The PDE residual F(u, v)
        - u: The unknown
        - bcs: List of Dirichlet boundary conditions
        This class set up structures for solving the non-linear problem using Newton's method,
        dF/du(u) du = -F(u)
        """
        V = u.function_space
        du = ufl.TrialFunction(V)
        self.L = F
        # Create the Jacobian matrix, dF/du
        self.a = ufl.derivative(F, u, du)
        self.bcs = bcs

        # Create matrix and vector to be used for assembly
        # of the non-linear problem
        self.matrix = dolfinx.fem.create_matrix(self.a)
        self.vector = dolfinx.fem.create_vector(self.L)

    def form(self, x: PETSc.Vec):
        """
        This function is called before the residual or Jacobian is computed. This is usually used to update ghost values.
        Input:
           x: The vector containing the latest solution
        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)

    def F(self, x: PETSc.Vec, b: PETSc.Vec):
        """Assemble the residual F into the vector b.
        Input:
           x: The vector containing the latest solution
           b: Vector to assemble the residual into
        """
        # Reset the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
        dolfinx.fem.assemble_vector(b, self.L)
        # Apply boundary condition
        dolfinx.fem.apply_lifting(b, [self.a], [self.bcs], [x], -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(b, self.bcs, x, -1.0)

    def J(self, x: PETSc.Vec, A: PETSc.Mat):
        """Assemble the Jacobian matrix.
        Input:
          - x: The vector containing the latest solution
          - A: The matrix to assemble the Jacobian into
        """
        A.zeroEntries()
        dolfinx.fem.assemble_matrix(A, self.a, self.bcs)
        A.assemble()


def solve_manufactured(nx, ny, theta, gamma):
    L = 10

    mesh = dolfinx.RectangleMesh(
        MPI.COMM_WORLD,
        [np.array([0, -1, 0]), np.array([L, 1, 0])], [nx, ny],
        CellType.triangle, dolfinx.cpp.mesh.GhostMode.none)

    def left(x):
        return np.isclose(x[0], 0)

    def right(x):
        return np.isclose(x[0], L)

    tdim = mesh.topology.dim
    left_marker = 1
    right_marker = 2
    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, left)
    right_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, right)
    left_values = np.full(len(left_facets), left_marker, dtype=np.int32)
    right_values = np.full(len(right_facets), right_marker, dtype=np.int32)
    indices = np.concatenate([left_facets, right_facets])
    values = np.hstack([left_values, right_values])
    # Need to sort indices and values before meshtags
    facet_marker = dolfinx.MeshTags(mesh, tdim - 1, indices, values)

    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))
    n = ufl.FacetNormal(mesh)
    E = 1500
    nu = 0.25
    h = ufl.Circumradius(mesh)
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

    # Problem specific body force and traction
    # from DOI: 10.4208/aamm.2014.m548 (Chapter 5.1)
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.as_vector((-6 * x[1]**2, 6 * x[0]**2))
    g = ufl.as_vector((0, 2000 + 2 * x[1]**3))

    # Body force for example 5.2
    # E * ufl.pi**2 * (ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
    #  -ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])))

    def epsilon(v):
        return ufl.sym(ufl.grad(v))

    def sigma(v):
        return (2.0 * mu * epsilon(v) +
                lmbda * ufl.tr(epsilon(v)) * ufl.Identity(len(v)))

    u = ufl.TrialFunction(V)
    # u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
    F = ufl.inner(sigma(u), epsilon(v)) * dx - ufl.inner(
        f, v) * dx - ufl.inner(g, v) * ds(right_marker)

    # # Nitsche for Dirichlet, theta-scheme.
    # https://www.sciencedirect.com/science/article/pii/S004578251830269X
    u_D = ufl.as_vector((0, 0))
    n_facet = ufl.FacetNormal(mesh)
    F += -ufl.inner(sigma(u) * n_facet, v) * ds(left_marker)\
        - theta * ufl.inner(sigma(v) * n_facet, u - u_D) * ds(left_marker)\
             + gamma / h * ufl.inner(u - u_D, v) * ds(left_marker)
    # Create nonlinear problem
    # problem = NonlinearPDEProblem(F, u, [])
    # Create Newton solver
    # solver = dolfinx.cpp.nls.NewtonSolver(MPI.COMM_WORLD)
    problem = dolfinx.fem.LinearProblem(ufl.lhs(F), ufl.rhs(F), bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u = problem.solve()


    # Set Newton solver options
    # solver.atol = 1e-6
    # solver.rtol = 1e-6
    # solver.convergence_criterion = "incremental"

    # # Set non-linear problem for Newton solver
    # solver.setF(problem.F, problem.vector)
    # solver.setJ(problem.J, problem.matrix)
    # solver.set_form(problem.form)

    # Solve non-linear problem
    # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    # n, converged = solver.solve(u.vector)
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                         mode=PETSc.ScatterMode.FORWARD)
    # assert (converged)
    # print(f"Number of interations: {n:d}")
    os.system("mkdir -p results")
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"results/u_{nx}_{ny}.xdmf",
                             "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)
        xdmf.write_meshtags(facet_marker)
    # Error computaion:
    u_ex = (nu + 1) / E * ufl.as_vector((x[1]**4, x[0]**4))
    error = (u - u_ex)**2 * ufl.dx
    E_L2 = np.sqrt(dolfinx.fem.assemble_scalar(error))
    print(f"{nx} {ny}: L2-error: {E_L2:.2e}")
    return E_L2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--theta",
        default=1,
        type=np.float64,
        dest="theta",
        help=
        "Theta parameter for Nitsche, 1 symmetric, -1 skew symmetric, 0 Penalty-like"
    )
    parser.add_argument(
        "--gamma",
        default=1000,
        type=np.float64,
        dest="gamma",
        help="Coercivity/Stabilization parameter for Nitsche condition")

    args = parser.parse_args()
    theta = args.theta
    gamma = args.gamma
    Nx = np.asarray([5 * 2**i for i in range(3, 4)], dtype=np.int32)
    Ny = np.asarray([2**i for i in range(3, 4)], dtype=np.int32)
    for (nx, ny) in zip(Nx, Ny):
        solve_manufactured(nx, ny, theta, gamma)
        break