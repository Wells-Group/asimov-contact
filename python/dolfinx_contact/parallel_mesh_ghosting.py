# Copyright (C) 2022 Chris N. Richardson and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from dolfinx import log
from dolfinx.mesh import create_mesh, meshtags
import dolfinx
from dolfinx.cpp.mesh import entities_to_geometry, cell_num_vertices, cell_entity_type, to_type
import numpy as np
# import numba

__all__ = ["create_contact_mesh", "point_cloud_pairs", "compute_ghost_cell_destinations"]


# @numba.njit
def point_cloud_pairs(x, r):
    """Find all neighbors of each point which are within a radius r."""

    # Get sort-order in ascending x-value, and reverse permutation
    x_fwd = np.argsort(x[:, 0])
    x_rev = np.empty_like(x_fwd)
    for i, fi in enumerate(x_fwd):
        x_rev[fi] = i

    npoints = len(x_fwd)
    x_near = [[int(0) for k in range(0)] for j in range(0)]  # weird stuff for numba
    for i in range(npoints):
        xni = [int(0) for j in range(0)]  # empty list of int for numba
        # Nearest neighbor with greater x-value
        idx = x_rev[i] + 1
        while idx < npoints:
            dx = x[x_fwd[idx], 0] - x[i, 0]
            if dx > r:
                break
            dr = np.linalg.norm(x[x_fwd[idx], :] - x[i, :])
            if dr < r:
                xni += [x_fwd[idx]]
            idx += 1
        # Nearest neighbor with smaller x-value
        idx = x_rev[i] - 1
        while idx > 0:
            dx = x[i, 0] - x[x_fwd[idx], 0]
            if dx > r:
                break
            dr = np.linalg.norm(x[x_fwd[idx], :] - x[i, :])
            if dr < r:
                xni += [x_fwd[idx]]
            idx -= 1
        x_near += [xni]

    return x_near


def compute_ghost_cell_destinations(mesh, marker_subset, R):
    """For each marked facet, given by indices in "marker_subset", get the list of processes which
    the attached cell should be sent to, for ghosting. Neighbouring facets within distance "R"."""

    # 1. Get midpoints of all facets on interfaces
    tdim = mesh.topology.dim
    x = mesh.geometry.x
    facet_to_geom = entities_to_geometry(mesh._cpp_object, tdim - 1, marker_subset, False)
    x_facet = np.array([sum([x[i] for i in idx]) / len(idx) for idx in facet_to_geom])

    # 2. Send midpoints to process zero
    comm = mesh.comm
    x_all = comm.gather(x_facet, root=0)
    scatter_back = []
    if comm.rank == 0:
        offsets = np.cumsum([0] + [w.shape[0] for w in x_all])
        x_all_flat = np.concatenate(x_all)

        # Find all pairs of facets within radius R
        x_near = point_cloud_pairs(x_all_flat, R)

        # Find which process the neighboring facet came from
        i = 0
        procs = [[] for p in range(len(x_all))]
        for p in range(len(x_all)):
            for j in range(x_all[p].shape[0]):
                pr = set()
                for n in x_near[i]:
                    # Find which process this facet came from
                    q = np.searchsorted(offsets, n, side='right') - 1
                    # Add to the sendback list, if not the same process
                    if q != p:
                        pr.add(q)
                procs[p] += [list(pr)]
                i += 1

        # Pack up to return to sending processes
        for i, q in enumerate(procs):
            off = np.cumsum([0] + [len(w) for w in q])
            flat_q = sum(q, [])
            scatter_back += [[len(off)] + list(off) + flat_q]

    d = comm.scatter(scatter_back, root=0)
    # Unpack received data to get additional destinations for each facet/cell
    n = d[0] + 1
    offsets = d[1:n]
    cell_dests = [d[n + offsets[j]:n + offsets[j + 1]] for j in range(n - 2)]
    assert len(cell_dests) == len(marker_subset)
    return cell_dests


def create_contact_mesh(mesh, fmarker, dmarker, tags, R=0.2):

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

    # Extract facet markers with given tags
    marker_subset_i = [i for i, (idx, k) in enumerate(zip(fmarker.indices, fmarker.values)) if k in tags]
    marker_subset = fmarker.indices[marker_subset_i]
    # marker_subset_val = fmarker.values[marker_subset_i]
    # facets = np.hstack([fmarker.find(tag) for tag in tags])

    # Find destinations for the cells attached to the tag-marked facets
    cell_dests = compute_ghost_cell_destinations(mesh, marker_subset, R)
    cells_to_ghost = [fc.links(f)[0] for f in marker_subset]
    assert len(cell_dests) == len(cells_to_ghost)
    cell_to_dests = {c: d for c, d in zip(cells_to_ghost, cell_dests)}

    ncells = mesh.topology.index_map(tdim).size_local

    # Convert marked facets to list of (global) vertices for each facet
    fv_indices = [sorted(mesh.topology.index_map(0).local_to_global(fv.links(f))) for f in fmarker.indices]
    cv_indices = [sorted(mesh.topology.index_map(0).local_to_global(cv.links(c))) for c in dmarker.indices]

    log.log(log.LogLevel.WARNING, "Copy markers to other processes")
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
    topo = mesh.topology.index_map(0).local_to_global(topo).reshape((-1, num_cell_vertices))
    topo = topo[:ncells, :]

    # Cut off any ghost vertices
    num_vertices = mesh.topology.index_map(0).size_local
    gdim = mesh.geometry.dim
    x = mesh.geometry.x[:num_vertices, :gdim]
    domain = mesh.ufl_domain()
    log.log(log.LogLevel.WARNING, "Repartition")
    new_mesh = create_mesh(mesh.comm, topo, x, domain, partitioner)

    log.log(log.LogLevel.WARNING, "Remap markers on new mesh")
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

    def lex_match(local_indices, in_indices, in_values):
        lx_loc = np.lexsort(np.flip(local_indices, axis=1).T)
        lx_in = np.lexsort(np.flip(in_indices, axis=1).T)

        new_markers = []
        i = 0
        j = 0
        while i < len(lx_in) and j < len(lx_loc):
            a = in_indices[lx_in[i]]
            b = local_indices[lx_loc[j]]
            idx = np.where((a > b) != (a < b))[0]
            if len(idx) == 0:
                new_markers += [[lx_loc[j], in_values[lx_in[i]]]]
                i += 1
                j += 1
            else:
                idx = idx[0]
                if b[idx] > a[idx]:
                    i += 1
                elif a[idx] > b[idx]:
                    j += 1
        return new_markers

    log.log(log.LogLevel.WARNING, "Lex match facet markers")
    new_fmarkers = lex_match(fv_indices, all_indices, all_values)

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
    log.log(log.LogLevel.WARNING, "Lex match cell markers")
    new_cmarkers = lex_match(cv_indices, all_cell_indices, all_cell_values)

    # Sort new markers into order and make unique
    new_cmarkers = np.array(sorted(new_cmarkers), dtype=np.int32)
    new_cmarkers = np.unique(new_cmarkers, axis=0)

    new_dmarker = meshtags(new_mesh, tdim, new_cmarkers[:, 0],
                           new_cmarkers[:, 1])
    return new_mesh, new_fmarker, new_dmarker
