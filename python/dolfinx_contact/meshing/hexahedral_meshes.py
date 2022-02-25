# Copyright (C) 2022 Jørgen S. Dokken and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import gmsh
import numpy as np
from dolfinx.io import (XDMFFile, cell_perm_gmsh,
                        extract_gmsh_geometry,
                        extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh)
from dolfinx.mesh import CellType, create_mesh

from mpi4py import MPI

__all__ = ["create_hexahedral_mesh"]


def create_hexahedral_mesh(filename: str, order: int = 1):
    if MPI.COMM_WORLD.rank == 0:
        gmsh.initialize()
        model = gmsh.model()

        # Generate a mesh with 2nd-order hexahedral cells using gmsh
        model.add("Hexahedral mesh")
        model.setCurrent("Hexahedral mesh")
        # Recombine tetrahedrons to hexahedrons
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.1)

        circle = model.occ.addDisk(0, 0, 0, 2, 2)
        circle2 = model.occ.addDisk(3.5, 0, 0, 1, 1)
        fuse = gmsh.model.occ.fuse([(2, circle)], [(2, circle2)])
        extruded_geometry = model.occ.extrude(
            fuse[0], 0, 0, 0.5, numElements=[5], recombine=True)
        model.occ.synchronize()

        model.mesh.generate(3)
        model.mesh.setOrder(order)
        volume_entities = []
        for entity in extruded_geometry:
            if entity[0] == 3:
                volume_entities.append(entity[1])
        model.addPhysicalGroup(3, volume_entities, tag=1)
        model.setPhysicalName(3, 1, "Mesh volume")

        # Sort mesh nodes according to their index in gmsh
        x = extract_gmsh_geometry(model, model.getCurrent())

        # Broadcast cell type data and geometric dimension
        gmsh_cell_id = MPI.COMM_WORLD.bcast(
            model.mesh.getElementType("hexahedron", order), root=0)

        # Get mesh data for dim (0, tdim) for all physical entities
        topologies = extract_gmsh_topology_and_markers(model, model.getCurrent())
        cells = topologies[gmsh_cell_id]["topology"]

        num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
        gmsh.finalize()
    else:
        gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
        num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
        cells, x = np.empty([0, num_nodes]), np.empty([0, 3])

    # Permute the mesh topology from GMSH ordering to DOLFINx ordering
    domain = ufl_mesh_from_gmsh(gmsh_cell_id, 3)
    gmsh_hex = cell_perm_gmsh(CellType.hexahedron, ((1 + order)**3))
    cells = cells[:, gmsh_hex]

    msh = create_mesh(MPI.COMM_WORLD, cells, x, domain)
    msh.name = "hex_d2"

    # Permute also entities which are tagged
    with XDMFFile(MPI.COMM_WORLD, f"{filename}.xdmf", "w") as file:
        file.write_mesh(msh)
