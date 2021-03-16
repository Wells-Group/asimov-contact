import dolfinx
import ufl
from typing import List
from petsc4py import PETSc

__all__ = ["NonlinearPDEProblem", "lame_parameters", "epsilon", "sigma_func"]


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


def lame_parameters(plane_strain=False):
    """
    Returns the Lame parameters for plane stress or plane strain.
    Return type is lambda functions
    """
    def mu(E, nu): return E / (2 * (1 + nu))
    if plane_strain:
        def lmbda(E, nu): return E * nu / ((1 + nu) * (1 - 2 * nu))
        return mu, lmbda
    else:
        def lmbda(E, nu): return E * nu / ((1 + nu) * (1 - nu))
        return mu, lmbda


def epsilon(v):
    return ufl.sym(ufl.grad(v))


def sigma_func(mu, lmbda):
    return lambda v: (2.0 * mu * epsilon(v) + lmbda * ufl.tr(epsilon(v)) * ufl.Identity(len(v)))
