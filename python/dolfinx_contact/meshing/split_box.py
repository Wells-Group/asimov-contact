# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import numpy as np
import gmsh
from mpi4py import MPI

from dolfinx.graph import create_adjacencylist
from dolfinx.io import (XDMFFile, cell_perm_gmsh, distribute_entity_data,
                        extract_gmsh_geometry,
                        extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh)
from dolfinx.mesh import CellType, create_mesh, meshtags_from_entities


def create_split_box_2D(filename: str, res=0.8):
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", res)
        L1 = 3
        H = 1
        model = gmsh.model()
        model.add("left")
        model.setCurrent("left")
        # Create box
        p0 = gmsh.model.occ.addPoint(0, 0, 0)
        p1 = gmsh.model.occ.addPoint(L1, 0, 0)
        p2 = gmsh.model.occ.addPoint(L1, H, 0)
        p3 = gmsh.model.occ.addPoint(0, H, 0)
        ps = [p0, p1, p2, p3]
        lines = [gmsh.model.occ.addLine(ps[i - 1], ps[i]) for i in range(len(ps))]
        curve = gmsh.model.occ.addCurveLoop(lines)
        surface1 = gmsh.model.occ.addPlaneSurface([curve])
        model.occ.synchronize()
        model.addPhysicalGroup(2, [surface1], tag=1)
        model.addPhysicalGroup(1, lines[0:2] + lines[3:], tag=3)
        model.addPhysicalGroup(1, [lines[2]], tag=4)
        model.mesh.generate(2)
        # Sort mesh nodes according to their index in gmsh
        x = extract_gmsh_geometry(model, model_name="left")

        # Broadcast cell type data and geometric dimension
        gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("triangle", 1), root=0)

        # Get mesh data for dim (0, tdim) for all physical entities
        topologies = extract_gmsh_topology_and_markers(model, "left")
        cells = topologies[gmsh_cell_id]["topology"]
        cell_data = topologies[gmsh_cell_id]["cell_data"]
        num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)
        gmsh_facet_id = model.mesh.getElementType("line", 1)
        marked_facets = topologies[gmsh_facet_id]["topology"].astype(np.int64)
        facet_values = topologies[gmsh_facet_id]["cell_data"].astype(np.int32)
        model.add("right")
        model.setCurrent("right")
        # Create box
        L2 = 2
        p0 = gmsh.model.occ.addPoint(L1, 0, 0)
        p1 = gmsh.model.occ.addPoint(L1 + L2, 0, 0)
        p2 = gmsh.model.occ.addPoint(L1 + L2, H, 0)
        p3 = gmsh.model.occ.addPoint(L1, H, 0)
        ps = [p0, p1, p2, p3]
        lines2 = [gmsh.model.occ.addLine(ps[i - 1], ps[i]) for i in range(len(ps))]
        curve2 = gmsh.model.occ.addCurveLoop(lines2)
        surface2 = gmsh.model.occ.addPlaneSurface([curve2])
        model.occ.synchronize()
        model.addPhysicalGroup(2, [surface2], tag=2)
        model.addPhysicalGroup(1, lines2[1:], tag=5)
        model.addPhysicalGroup(1, [lines2[0]], tag=6)
        model.mesh.generate(2)
        # Sort mesh nodes according to their index in gmsh
        x2 = extract_gmsh_geometry(model, model_name="right")

        # Broadcast cell type data and geometric dimension
        gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("triangle", 1), root=0)

        # Get mesh data for dim (0, tdim) for all physical entities
        topologies2 = extract_gmsh_topology_and_markers(model, "right")
        cells2 = topologies2[gmsh_cell_id]["topology"]
        cell_data2 = topologies2[gmsh_cell_id]["cell_data"]
        marked_facets2 = topologies2[gmsh_facet_id]["topology"].astype(np.int64)
        facet_values2 = topologies2[gmsh_facet_id]["cell_data"].astype(np.int32)
        marked_facets = np.vstack([marked_facets, marked_facets2 + x.shape[0]])
        facet_values = np.hstack([facet_values, facet_values2])
        cell_data = np.hstack([cell_data, cell_data2])
        cells = np.vstack([cells, cells2 + x.shape[0]])
        x = np.vstack([x, x2])

    else:
        gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
        num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
        cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
        #marked_facets, facet_values = np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int32)

    msh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh_from_gmsh(gmsh_cell_id, 2))
    msh.name = "Grid"
    entities, values = distribute_entity_data(msh, 1, marked_facets, facet_values)
    msh.topology.create_connectivity(1, 0)
    mt = meshtags_from_entities(msh, 1, create_adjacencylist(entities), values)
    mt.name = "contact_facets"
    entities, values = distribute_entity_data(msh, 2, cells.astype(np.int64), cell_data.astype(np.int32))
    mt_domain = meshtags_from_entities(msh, 2, create_adjacencylist(entities), values)
    mt_domain.name = "domain_marker"
    gmsh.finalize()
    with XDMFFile(MPI.COMM_WORLD, f"{filename}.xdmf", "w") as file:
        file.write_mesh(msh)
        msh.topology.create_connectivity(1, 2)
        file.write_meshtags(mt_domain)
        file.write_meshtags(mt)
