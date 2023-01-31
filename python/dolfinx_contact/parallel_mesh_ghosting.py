# Copyright (C) 2022 Chris N. Richardson and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from dolfinx.mesh import create_mesh, meshtags
import dolfinx
from dolfinx.cpp.mesh import entities_to_geometry, cell_num_vertices, cell_entity_type, to_type
import numpy as np

__all__ = ["create_contact_mesh"]


def create_contact_mesh(mesh, fmarker, dmarker, tags):

    tdim = mesh.topology.dim
    num_cell_vertices = cell_num_vertices(mesh.topology.cell_type)
    facet_type = cell_entity_type(to_type(str(mesh.ufl_cell())), tdim - 1, 0)
    num_facet_vertices = cell_num_vertices(facet_type)

    # Get cells attached to marked facets
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mesh.topology.create_connectivity(tdim, 0)
    fc = mesh.topology.connectivity(tdim - 1, tdim)
    fv = mesh.topology.connectivity(tdim - 1, 0)
    cv = mesh.topology.connectivity(tdim, 0)

    facets = np.hstack([fmarker.find(tag) for tag in tags])
    cells_to_ghost = np.unique([fc.links(f)[0] for f in facets])
    ncells = mesh.topology.index_map(tdim).size_local

    # Convert marked facets to list of (global) vertices for each facet
    fv_indices = [sorted(mesh.topology.index_map(0).local_to_global(fv.links(f))) for f in fmarker.indices]
    cv_indices = [sorted(mesh.topology.index_map(0).local_to_global(cv.links(c))) for c in dmarker.indices]

    # Copy facets and markers to all processes
    global_fmarkers = np.concatenate(fv_indices)
    all_indices = mesh.comm.allgather(global_fmarkers)
    all_indices = np.concatenate(all_indices).reshape(-1, num_facet_vertices)
    all_values = np.concatenate(mesh.comm.allgather(fmarker.values))

    global_dmarkers = np.concatenate(cv_indices)
    all_cell_indices = mesh.comm.allgather(global_dmarkers)
    all_cell_indices = np.concatenate(all_cell_indices).reshape(-1, num_cell_vertices)
    all_cell_values = np.concatenate(mesh.comm.allgather(dmarker.values))

    def partitioner(comm, n, m, topo):
        rank = comm.Get_rank()
        other_ranks = [i for i in range(comm.Get_size()) if i != rank]

        dests = []
        offsets = [0]
        for c in range(ncells):
            dests.append(rank)
            if c in cells_to_ghost:
                dests.extend(other_ranks)  # Ghost to other processes
            offsets.append(len(dests))
        return dolfinx.cpp.graph.AdjacencyList_int32(dests, offsets)

    # Convert topology to global indexing, and restrict to non-ghost cells
    topo = mesh.topology.connectivity(tdim, 0).array
    topo = mesh.topology.index_map(0).local_to_global(topo).reshape((-1, num_cell_vertices))
    topo = topo[:ncells, :]

    # Cut off any ghost vertices
    num_vertices = mesh.topology.index_map(0).size_local
    gdim = mesh.geometry.dim
    x = mesh.geometry.x[:num_vertices, :gdim]
    domain = mesh.ufl_domain()
    new_mesh = create_mesh(mesh.comm, topo, x, domain, partitioner)

    # Remap vertices back to input indexing
    # This is rather messy, we need to map vertices to geometric nodes
    # then back to original index
    global_remap = np.array(new_mesh.geometry.input_global_indices, dtype=np.int32)
    nv = new_mesh.topology.index_map(0).size_local + new_mesh.topology.index_map(0).num_ghosts
    vert_to_geom = entities_to_geometry(new_mesh._cpp_object, 0, np.arange(nv, dtype=np.int32), False).flatten()
    rmap = np.vectorize(lambda idx: global_remap[vert_to_geom[idx]])

    # Recreate facets
    new_mesh.topology.create_entities(tdim - 1)
    new_mesh.topology.create_connectivity(tdim - 1, tdim)
    new_mesh.topology.create_connectivity(tdim, 0)

    # Create a list of all facet-vertices (original global index)
    fv = new_mesh.topology.connectivity(tdim - 1, 0)
    fv_indices = rmap(fv.array).reshape((-1, num_facet_vertices))
    fv_indices = np.sort(fv_indices, axis=1)

    # Search for marked facets in list of all facets
    new_fmarkers = []
    for idx, val in zip(all_indices, all_values):
        f = np.nonzero(np.all(fv_indices == idx, axis=1))[0]
        if len(f) > 0:
            assert len(f) == 1
            f = f[0]
            new_fmarkers += [[f, val]]

    # Sort new markers into order and make unique
    new_fmarkers = np.array(sorted(new_fmarkers), dtype=np.int32)
    new_fmarkers = np.unique(new_fmarkers, axis=0)

    new_fmarker = meshtags(new_mesh, tdim - 1, new_fmarkers[:, 0],
                           new_fmarkers[:, 1])

    # Create a list of all cell-vertices (original global index)
    cv = new_mesh.topology.connectivity(tdim, 0)
    cv_indices = rmap(cv.array).reshape((-1, num_cell_vertices))
    cv_indices = np.sort(cv_indices, axis=1)

    # Search for marked cells in list of all cells
    new_cmarkers = []
    for idx, val in zip(all_cell_indices, all_cell_values):
        c = np.nonzero(np.all(cv_indices == idx, axis=1))[0]
        if len(c) > 0:
            assert len(c) == 1
            c = c[0]
            new_cmarkers += [[c, val]]

    # Sort new markers into order and make unique
    new_cmarkers = np.array(sorted(new_cmarkers), dtype=np.int32)
    new_cmarkers = np.unique(new_cmarkers, axis=0)

    new_dmarker = meshtags(new_mesh, tdim, new_cmarkers[:, 0],
                           new_cmarkers[:, 1])
    return new_mesh, new_fmarker, new_dmarker
