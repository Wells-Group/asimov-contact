# Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT


import dolfinx.la as _la
import dolfinx.fem as _fem
import numpy
import ufl
from contextlib import ExitStack
from petsc4py import PETSc

__all__ = ["lame_parameters", "epsilon", "sigma_func", "convert_mesh"]


def lame_parameters(plane_strain: bool = False):
    """
    Returns the Lame parameters for plane stress or plane strain.
    Return type is lambda functions
    """
    def mu(E, nu):
        return E / (2 * (1 + nu))

    if plane_strain:
        def lmbda(E, nu):
            return E * nu / ((1 + nu) * (1 - 2 * nu))
        return mu, lmbda
    else:
        def lmbda(E, nu):
            return E * nu / ((1 + nu) * (1 - nu))
        return mu, lmbda


def epsilon(v):
    return ufl.sym(ufl.grad(v))


def sigma_func(mu, lmbda):
    return lambda v: (2.0 * mu * epsilon(v) + lmbda * ufl.tr(epsilon(v)) * ufl.Identity(len(v)))


# R_minus(x) returns x if negative zero else
def R_minus(x):
    abs_x = abs(x)
    return 0.5 * (x - abs_x)


# ball projection
def ball_projection(x, s):
    dim = x.geometric_dimension()
    abs_x = ufl.sqrt(sum([x[i]**2 for i in range(dim)]))
    return ufl.conditional(ufl.le(abs_x, s), x, s * x / abs_x)


def tangential_proj(u, n):
    """
    See for instance:
    https://doi.org/10.1023/A:1022235512626
    """
    return (ufl.Identity(u.ufl_shape[0]) - ufl.outer(n, n)) * u


class NonlinearPDE_SNESProblem:
    def __init__(self, F, u, bc):
        V = u.function_space
        du = ufl.TrialFunction(V)

        self.L = F
        self.a = ufl.derivative(F, u, du)
        self.a_comp = _fem.Form(self.a)
        self.bc = bc
        self._F, self._J = None, None
        self.u = u

    def F(self, snes, x, F):
        """Assemble residual vector."""
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with F.localForm() as f_local:
            f_local.set(0.0)
        _fem.assemble_vector(F, self.L)
        _fem.apply_lifting(F, [self.a], [[self.bc]], [x], -1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        _fem.set_bc(F, [self.bc], x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        J.zeroEntries()
        _fem.assemble_matrix(J, self.a, [self.bc])
        J.assemble()


def convert_mesh(filename: str, cell_type: str, prune_z: bool = False, ext: str = None, cell_data: str = "gmsh:physical"):
    """
    Given the filename of a msh file, read data and convert to XDMF file containing cells of given cell type
    """
    try:
        import meshio
    except ImportError:
        print("Meshio and h5py must be installed to convert meshes."
              + " Please run `pip3 install --no-binary=h5py h5py meshio`")
        exit(1)
    from mpi4py import MPI
    if MPI.COMM_WORLD.rank == 0:
        mesh = meshio.read(f"{filename}.msh")
        cells = mesh.get_cells_type(cell_type)
        data = mesh.get_cell_data(cell_data, cell_type)
        pts = mesh.points[:, :2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=pts, cells={cell_type: cells}, cell_data={"name_to_read": [data]})
        if ext is None:
            ext = ""
        else:
            ext = "_" + ext
        meshio.write(f"{filename}{ext}.xdmf", out_mesh)


def rigid_motions_nullspace(V: _fem.FunctionSpace):
    """
    Function to build nullspace for 2D/3D elasticity.

    Parameters:
    ===========
    V
        The function space
    """
    _x = _fem.Function(V)
    # Get geometric dim
    gdim = V.mesh.geometry.dim
    assert gdim == 2 or gdim == 3

    # Set dimension of nullspace
    dim = 3 if gdim == 2 else 6

    # Create list of vectors for null space
    nullspace_basis = [_x.vector.copy() for i in range(dim)]

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

    _la.orthonormalize(nullspace_basis)
    assert _la.is_orthonormal(nullspace_basis)
    return PETSc.NullSpace().create(vectors=nullspace_basis)
