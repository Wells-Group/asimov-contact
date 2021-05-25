from typing import List

import dolfinx
import dolfinx.la
import numpy
import ufl
from contextlib import ExitStack
from petsc4py import PETSc

__all__ = ["lame_parameters", "epsilon", "sigma_func", "convert_mesh"]


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


def convert_mesh(filename, cell_type, prune_z=False):
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
        pts = mesh.points[:, :2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=pts, cells={cell_type: cells}, cell_data={"name_to_read": [data]})
        meshio.write(f"{filename}.xdmf", out_mesh)


def rigid_motions_nullspace(V):
    """Function to build nullspace for 2D/3D elasticity"""

    # Get geometric dim
    gdim = V.mesh.geometry.dim
    assert gdim == 2 or gdim == 3

    # Set dimension of nullspace
    dim = 3 if gdim == 2 else 6

    # Create list of vectors for null space
    nullspace_basis = [dolfinx.cpp.la.create_vector(V.dofmap.index_map, V.dofmap.index_map_bs) for i in range(dim)]

    with ExitStack() as stack:
        vec_local = [stack.enter_context(x.localForm()) for x in nullspace_basis]
        basis = [numpy.asarray(x) for x in vec_local]

        dofs = [V.sub(i).dofmap.list.array for i in range(gdim)]

        # Build translational null space basis
        for i in range(gdim):
            basis[i][dofs[i]] = 1.0

        # Build rotational null space basis
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.array
        x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
        if gdim == 2:
            basis[2][dofs[0]] = -x1
            basis[2][dofs[1]] = x0
        elif gdim == 3:
            basis[3][dofs[0]] = -x1
            basis[3][dofs[1]] = x0

            basis[4][dofs[0]] = x2
            basis[4][dofs[2]] = -x0
            basis[5][dofs[2]] = x1
            basis[5][dofs[1]] = -x2

    basis = dolfinx.la.VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    _x = [basis[i] for i in range(dim)]
    nsp = PETSc.NullSpace().create(vectors=_x)
    return nsp
