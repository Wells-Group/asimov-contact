# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT

from mpi4py import MPI
from pathlib import Path
import dolfinx.io


def convert_mesh(filename: str|Path, outname:str|Path, gdim: int = 3):
    """
    Read a GMSH mesh (msh format) and convert it to XDMF with both cell
    and facet tags in a single file.

    Parameters
    ==========
    filename
        Name of input file
    outname
        Name of output file
    gdim
        The geometrical dimension of the mesh
    """
    fname = Path(filename)
    oname = Path(outname)

    if MPI.COMM_WORLD.rank == 0:
        mesh, ct, ft = dolfinx.io.gmshio.read_from_msh(fname.with_suffix(".msh").absolute().as_posix(), MPI.COMM_SELF, 0, gdim=gdim)
        ct.name = "cell_marker"
        ft.name = "facet_marker"
        with dolfinx.io.XDMFFile(mesh.comm, oname.with_suffix(".xdmf"), "w") as xdmf:
            xdmf.write_mesh(mesh)
            xdmf.write_meshtags(ct, mesh.geometry)
            mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
            xdmf.write_meshtags(ft, mesh.geometry)
    MPI.COMM_WORLD.Barrier()
