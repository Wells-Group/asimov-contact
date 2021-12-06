# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT

from mpi4py import MPI
import meshio


def convert_mesh(filename: str, outname: str, cell_type: str, prune_z: bool = False, cell_data: str = "gmsh:physical"):
    """
    Read a GMSH mesh (msh format) and convert it to XDMF with only cells of input cell-type.
    Name of output file will be the same as input file (.msh->.xdmf/.h5)

    Parameters
    ==========
    filename
        Name of input file
    outname
        Name of output file
    cell_type
        The cell type
    prune_z
        Sets mesh geometrical dimension to 2 if True, else gdim=3
    cell_data
        Key to cell data dictionary in msh file
    """
    fname = filename.split(".msh")[0]
    oname = outname.split(".xdmf")[0]

    if MPI.COMM_WORLD.rank == 0:
        mesh = meshio.read(f"{fname}.msh")
        cells = mesh.get_cells_type(cell_type)
        data = mesh.get_cell_data(cell_data, cell_type)
        pts = mesh.points[:, :2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=pts, cells={cell_type: cells}, cell_data={"name_to_read": [data]})
        meshio.write(f"{oname}.xdmf", out_mesh)
    MPI.COMM_WORLD.Barrier()
