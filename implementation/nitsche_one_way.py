from typing import List
from IPython import embed
import dolfinx
import dolfinx.fem
import dolfinx.io
import dolfinx.log
import dolfinx.mesh
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc


def nitsche_one_way(mesh = None, strain=True, square = False, refinement = 0):

    if square:
        def top(x):
            return np.isclose(x[1], 1)

        def bottom(x):
            return np.isclose(x[1], 0)

        mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 50, 50)
        tdim = mesh.topology.dim
        top_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, top)
        bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, bottom)
        top_values = np.full(len(top_facets), 1, dtype=np.int32)
        bottom_values = np.full(len(bottom_facets), 2, dtype=np.int32)
        indices = np.concatenate([top_facets, bottom_facets])
        values = np.hstack([top_values, bottom_values])
        facet_marker = dolfinx.MeshTags(mesh, tdim - 1, indices, values)

    else:
        if mesh is None:
            with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "disk.xdmf", "r") as xdmf:
                mesh = xdmf.read_mesh(name="Grid")

        def upper_part(x):
            return x[1] > 0.9

        def lower_part(x):
            return x[1] < 0.25
        tdim = mesh.topology.dim
        top_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, upper_part)
        bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, lower_part)
        top_values = np.full(len(top_facets), 1, dtype=np.int32)
        bottom_values = np.full(len(bottom_facets), 2, dtype=np.int32)
        indices = np.concatenate([top_facets, bottom_facets])
        values = np.hstack([top_values, bottom_values])
        facet_marker = dolfinx.MeshTags(mesh, tdim - 1, indices, values)

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "results/mf.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_marker)


    def R_minus(x):
        abs_x = abs(x)
        return 0.5 * (x - abs_x)


    def ball_projection(x, s):
        dim = x.geometric_dimension()
        abs_x = ufl.sqrt(sum([x[i]**2 for i in range(dim)]))
        return ufl.conditional(ufl.le(abs_x, s), x, s * x / abs_x)


    theta = 0
    s = 0  # 100e5
    V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))
    n = ufl.FacetNormal(mesh)
    E = 1e3
    nu = 0.1
    h = ufl.Circumradius(mesh)
    gamma = 10 / h
    mu = E / (2 * (1 + nu))
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))


    # Mimicking the plane y=-g
    x = ufl.SpatialCoordinate(mesh)
    g = 0.0
    gap = g


    def epsilon(v):
        return ufl.sym(ufl.grad(v))


    def sigma(v):
        return (2.0 * mu * epsilon(v)
                + lmbda * ufl.tr(epsilon(v)) * ufl.Identity(len(v)))


    def sigma_n(v):
        return ufl.dot(sigma(v) * n, n)


    def tangential_proj(u):
        """
        See for instance:
        https://link.springer.com/content/pdf/10.1023/A:1022235512626.pdf
        """
        return (ufl.Identity(u.ufl_shape[0]) - ufl.outer(n, n)) * u


    u = dolfinx.Function(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_marker)
    F = ufl.inner(sigma(u), epsilon(v)) * dx - ufl.inner(dolfinx.Constant(mesh, (0, 0)), v) * dx

    # Nitsche for contact (with Friction). 
    # NOTE: Differs from unitlateral contact even in the case of s=0!
    # F -= theta / gamma * sigma_n(u) * sigma_n(v) * ds(2)
    # F += 1 / gamma * R_minus(sigma_n(u) + gamma * (gap - ufl.dot(u, n))) * \
    #     (theta * sigma_n(v) - gamma * ufl.dot(v, n)) * ds(2)
    # F -= theta / gamma * ufl.dot(tangential_proj(u), tangential_proj(v)) * ds(2)
    # F += 1 / gamma * ufl.dot(ball_projection(tangential_proj(u) - gamma * tangential_proj(u), s),
    #                         theta * tangential_proj(v) - gamma * tangential_proj(v)) * ds(2)
    #P(v) = theta*sigma_n(v)-gamma*v
    # P = lambda (v, theta): theta*sigma_n - gamma()
    F -= theta / gamma * sigma_n(u) * sigma_n(v) * ds(2)
    F += 1 / gamma * R_minus(sigma_n(u) - gamma * (ufl.dot(u, n) - g))* (theta * sigma_n(v) - gamma * ufl.dot(v, n)) * ds(2)

    # Nitsche for Dirichlet, another theta-scheme.
    # https://www.sciencedirect.com/science/article/pii/S004578251830269X
    # Ultimately, it might make sense to use the same theta as for contact. But we keep things separate for now.
    # u_D = ufl.as_vector((0, -0.1))
    # n_facet = ufl.FacetNormal(mesh)
    # gamma_2 = 1000
    # theta_2 = 1  # 1 symmetric, -1 skew symmetric
    # F += -ufl.inner(sigma(u) * n_facet, v) * ds(1)\
    #     - theta_2 * ufl.inner(sigma(v) * n_facet, u - u_D) * ds(1) + gamma_2 / h * ufl.inner(u - u_D, v) * ds(1)

    # Dirichlet boundary conditions
    def _u_D(x):
        values = np.zeros((mesh.geometry.dim, x.shape[1]))
        values[0] = 0
        values[1] = -0.1
        return values
    u_D = dolfinx.Function(V)
    u_D.interpolate(_u_D)
    u_D.name = "u_D"
    u_D.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    bc = dolfinx.DirichletBC(u_D, dolfinx.fem.locate_dofs_topological(V, tdim - 1, top_facets))
    bcs = [bc] 

    class NonlinearPDEProblem:
        """Nonlinear problem class for solving the non-linear problem
        F(u, v) = 0 for all v in V
        """

        def __init__(self, F: ufl.form.Form, u: dolfinx.Function, bcs: List[dolfinx.DirichletBC]):
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
            x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

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
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
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
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    n, converged = solver.solve(u.vector)
    u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    assert(converged)
    print(f"Number of interations: {n:d}")

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"results/u_nitsche_{refinement}.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(u)

    return u
