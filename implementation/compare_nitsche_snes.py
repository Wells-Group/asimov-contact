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

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "disk.xdmf", "r") as xdmf:
                mesh = xdmf.read_mesh(name="Grid")

e_abs =[]
e_rel = []
for i in range(1):
    if i >0:
        mesh = dolfinx.mesh.refine(mesh)

    u1 = nitsche_one_way(mesh = mesh, refinement = i)
    u2 = snes_solver(mesh = mesh, refinement = i)

    V  = u1.function_space
    dx = ufl.Measure("dx", domain=mesh)

    error = (u1 - u2)**2 * dx
    E_L2 = np.sqrt(dolfinx.fem.assemble_scalar(error))
    u2_norm = u2**2*dx
    u2_L2 = np.sqrt(dolfinx.fem.assemble_scalar(u2_norm))
    print(f"abs. L2-error={E_L2:.2e}")
    print(f"rel. L2-error={E_L2/u2_L2:.2e}")
    e_abs.append(E_L2)
    e_rel.append(E_L2/u2_L2)

print(e_abs)
print(e_rel)