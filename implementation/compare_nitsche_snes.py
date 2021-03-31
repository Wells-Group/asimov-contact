import dolfinx
import dolfinx.fem
import dolfinx.io
import dolfinx.log
import dolfinx.mesh
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from snes_disk_against_plane import snes_solver
from nitsche_one_way import nitsche_one_way

disk = True
top_value = 1
bottom_value = 2
if disk:
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "disk.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    def top(x):
        return x[1] > 0.5

    def bottom(x):
        return x[1] < 0.1
else:
    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 50, 50)

    def top(x):
        return np.isclose(x[1], 1)

    def bottom(x):
        return np.isclose(x[1], 0)

physical_parameters = {"E": 1e3, "nu": 0.1, "strain": False}
vertical_displacement = -0.05
e_abs = []
e_rel = []
for i in range(1, 2):
    if i > 0:
        mesh.topology.create_entities(mesh.topology.dim - 1)
        mesh = dolfinx.mesh.refine(mesh)
    tdim = mesh.topology.dim
    top_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, top)
    bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, bottom)
    top_values = np.full(len(top_facets), top_value, dtype=np.int32)
    bottom_values = np.full(len(bottom_facets), bottom_value, dtype=np.int32)
    indices = np.concatenate([top_facets, bottom_facets])
    values = np.hstack([top_values, bottom_values])
    facet_marker = dolfinx.MeshTags(mesh, tdim - 1, indices, values)
    mesh_data = (facet_marker, top_value, bottom_value)

    u1 = nitsche_one_way(mesh=mesh, mesh_data=mesh_data, physical_parameters=physical_parameters,
                         vertical_displacement=vertical_displacement, refinement=i)
    u2 = snes_solver(mesh=mesh, mesh_data=mesh_data, physical_parameters=physical_parameters,
                     vertical_displacement=vertical_displacement, refinement=i)

    V = u1.function_space
    dx = ufl.Measure("dx", domain=mesh)

    error = (u1 - u2)**2 * dx
    E_L2 = np.sqrt(dolfinx.fem.assemble_scalar(error))
    u2_norm = u2**2 * dx
    u2_L2 = np.sqrt(dolfinx.fem.assemble_scalar(u2_norm))
    print(f"abs. L2-error={E_L2:.2e}")
    print(f"rel. L2-error={E_L2/u2_L2:.2e}")
    e_abs.append(E_L2)
    e_rel.append(E_L2 / u2_L2)

print(e_abs)
print(e_rel)
