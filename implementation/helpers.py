from typing import List

import dolfinx
import numpy
import ufl
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


class NonlinearPDE_SNESProblem:
    def __init__(self, F, u, bc):
        V = u.function_space
        du = ufl.TrialFunction(V)

        self.L = F
        self.a = ufl.derivative(F, u, du)
        self.a_comp = dolfinx.fem.Form(self.a)
        self.bc = bc
        self._F, self._J = None, None
        self.u = u

    def F(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)
        dolfinx.fem.assemble_vector(F, self.L)
        dolfinx.fem.apply_lifting(F, [self.a], [[self.bc]], [x], -1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(F, [self.bc], x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        dolfinx.fem.assemble_matrix(J, self.a, [self.bc])
        J.assemble()


def convert_mesh(filename, cell_type):
    """
    Given the filename of a msh file, read data and convert to XDMF file containing cells of given cell type
    """
    try:
        import meshio
    except ImportError:
        print("Meshio and h5py must be installed to convert meshes."
              + " Please run `pip3 install --no-binary=h5py h5py meshio`")
    from mpi4py import MPI
    if MPI.COMM_WORLD.rank == 0:
        mesh = meshio.read(f"{filename}.msh")
        cells = mesh.get_cells_type(cell_type)
        data = numpy.hstack([mesh.cell_data_dict["gmsh:physical"][key]
                            for key in mesh.cell_data_dict["gmsh:physical"].keys() if key == cell_type])
        out_mesh = meshio.Mesh(points=mesh.points[:, :2], cells={cell_type: cells}, cell_data={"name_to_read": [data]})
        meshio.write(f"{filename}.xdmf", out_mesh)
