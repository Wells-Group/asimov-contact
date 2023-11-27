# Copyright (C) 2022 Chris N. Richardson and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from dolfinx import log
from dolfinx.mesh import create_mesh, meshtags
from dolfinx.common import Timer
import dolfinx
from dolfinx.cpp.mesh import entities_to_geometry, cell_num_vertices, cell_entity_type, to_type
import numpy as np
from dolfinx_contact.cpp import compute_ghost_cell_destinations, lex_match

__all__ = ["create_contact_mesh"]


def create_contact_mesh(mesh, fmarker, dmarker, tags, R=0.2):

    log.log(log.LogLevel.WARNING, "Create Contact Mesh")
    tdim = mesh.topology.dim
    num_cell_vertices = cell_num_vertices(mesh.topology.cell_types[0])
    facet_type = cell_entity_type(to_type(str(mesh.ufl_cell())), tdim - 1, 0)
    num_facet_vertices = cell_num_vertices(facet_type)

    # Get cells attached to marked facets
    mesh.topology.create_connectivity(tdim - 1, tdim)
    mesh.topology.create_connectivity(tdim, 0)
    fc = mesh.topology.connectivity(tdim - 1, tdim)
    fv = mesh.topology.connectivity(tdim - 1, 0)
    cv = mesh.topology.connectivity(tdim, 0)

    # Extract facet markers with given tags
    marker_subset_i = [i for i, (idx, k) in enumerate(zip(fmarker.indices, fmarker.values)) if k in tags]
    marker_subset = fmarker.indices[marker_subset_i]
    # marker_subset_val = fmarker.values[marker_subset_i]
    # facets = np.hstack([fmarker.find(tag) for tag in tags])

    log.log(log.LogLevel.WARNING, "Compute cell destinations")
    # Find destinations for the cells attached to the tag-marked facets
    with Timer("~Contact: Add ghosts: Compute cell destinations"):
        cell_dests = compute_ghost_cell_destinations(mesh._cpp_object, marker_subset, R)
    log.log(log.LogLevel.WARNING, "cells to ghost")
    timer = Timer("~Contact: Add ghosts: cells to ghost")
    cells_to_ghost = [fc.links(f)[0] for f in marker_subset]
    cell_to_dests = {}
    for i, c in enumerate(cells_to_ghost):
        cell_to_dests[c] = cell_dests.links(i)

    ncells = mesh.topology.index_map(tdim).size_local

    # Convert marked facets to list of (global) vertices for each facet
    fv_indices = [sorted(mesh.topology.index_map(0).local_to_global(fv.links(f))) for f in fmarker.indices]
    cv_indices = [sorted(mesh.topology.index_map(0).local_to_global(cv.links(c))) for c in dmarker.indices]

    timer.stop()
    log.log(log.LogLevel.WARNING, "Copy markers to other processes")
    timer = Timer("~Contact: Add ghosts: Copy markers to other processes")
    # Copy facets and markers to all processes
    if len(fv_indices) > 0:
        global_fmarkers = np.concatenate(fv_indices)
    else:
        global_fmarkers = []
    all_indices = [f for f in mesh.comm.allgather(global_fmarkers) if len(f) > 0]
    all_indices = np.concatenate(all_indices).reshape(-1, num_facet_vertices)

    all_values = np.concatenate([v for v in mesh.comm.allgather(fmarker.values) if len(v) > 0])
    assert len(all_values) == all_indices.shape[0]

    global_dmarkers = np.concatenate(cv_indices)
    all_cell_indices = mesh.comm.allgather(global_dmarkers)
    all_cell_indices = np.concatenate(all_cell_indices).reshape(-1, num_cell_vertices)
    all_cell_values = np.concatenate(mesh.comm.allgather(dmarker.values))

    def partitioner(comm, n, m, topo):
        rank = comm.Get_rank()
        dests = []
        offsets = [0]
        for c in range(ncells):
            dests.append(rank)
            if c in cell_to_dests:
                dests.extend(cell_to_dests[c])  # Ghost to other processes
            offsets.append(len(dests))
        return dolfinx.cpp.graph.AdjacencyList_int32(dests, offsets)

    # Convert topology to global indexing, and restrict to non-ghost cells
    topo = mesh.topology.connectivity(tdim, 0).array
    topo = np.asarray(
        mesh.topology.index_map(0).local_to_global(topo), dtype=np.int64).reshape(
            (-1, num_cell_vertices))
    topo = topo[:ncells, :]

    # Cut off any ghost vertices
    num_vertices = mesh.topology.index_map(0).size_local
    gdim = mesh.geometry.dim
    x = mesh.geometry.x[:num_vertices, :gdim]
    domain = mesh.ufl_domain()
    timer.stop()
    log.log(log.LogLevel.WARNING, "Repartition")
    with Timer("~Contact: Add ghosts: Repartition"):
        new_mesh = create_mesh(mesh.comm, topo, x, domain, partitioner)

    log.log(log.LogLevel.WARNING, "Remap markers on new mesh")
    timer = Timer("~Contact: Add ghosts: Remap markers on new mesh")
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

    timer.stop()
    log.log(log.LogLevel.WARNING, "Lex match facet markers")

    timer = Timer("~Contact: Add ghosts: Lex match facet markers")
    new_fmarkers = lex_match(fv_indices.shape[1], list(fv_indices.flatten()),
                             list(all_indices.flatten()), list(all_values))

    # Sort new markers into order and make unique
    new_fmarkers = np.array(new_fmarkers, dtype=np.int32)

    if new_fmarkers.shape[0] == 0:
        new_fmarkers = np.zeros((0, 2), dtype=np.int32)

    new_fmarker = meshtags(new_mesh, tdim - 1, new_fmarkers[:, 0],
                           new_fmarkers[:, 1])

    # Create a list of all cell-vertices (original global index)
    cv = new_mesh.topology.connectivity(tdim, 0)
    cv_indices = rmap(cv.array).reshape((-1, num_cell_vertices))
    cv_indices = np.sort(cv_indices, axis=1)
    timer.stop()

    # Search for marked cells in list of all cells
    log.log(log.LogLevel.WARNING, "Lex match cell markers")
    timer = Timer("~Contact: Add ghosts: Lex match cell markers")
    new_cmarkers = lex_match(cv_indices.shape[1], list(cv_indices.flatten()),
                             list(all_cell_indices.flatten()), list(all_cell_values))

    # Sort new markers into order and make unique
    new_cmarkers = np.array(new_cmarkers, dtype=np.int32)

    new_dmarker = meshtags(new_mesh, tdim, new_cmarkers[:, 0],
                           new_cmarkers[:, 1])

    timer.stop()
    return new_mesh, new_fmarker, new_dmarker
