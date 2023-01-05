from mpi4py import MPI
from dolfinx.io import XDMFFile, VTKFile
from dolfinx.mesh import create_mesh, meshtags
import dolfinx
from dolfinx.cpp.mesh import entities_to_geometry, cell_num_vertices, cell_entity_type
import numpy as np
import numba


@numba.njit
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
    the attached cell should be sent to, for ghosting. Any neighbouring facets within distance "R" should be ghosted."""
    # 1. Get midpoints of all facets on interfaces
    tdim = mesh.topology.dim
    x = mesh.geometry.x
    facet_to_geom = entities_to_geometry(mesh, tdim - 1, marker_subset, False)
    x_facet = np.array([sum([x[i] for i in idx]) / len(idx) for idx in facet_to_geom])

    # 2. Send midpoints to process zero
    comm = mesh.comm
    x_all = comm.gather(x_facet, root=0)
    scatter_back = []
    if comm.rank == 0:
        offsets = np.cumsum([0] + [w.shape[0] for w in x_all])
        x_all_joined = np.concatenate(x_all)

        # Find all pairs of facets within radius R
        x_near = point_cloud_pairs(x_all_joined, R)

        # Find which process the neighboring facet came from
        i = 0
        sender = np.concatenate([np.ones(arr.shape[0], dtype=int)*p for p, arr in enumerate(x_all)])
        procs = [[] for p in range(len(x_all))]
        for p, xn in zip(sender, x_near):
            pr = set()
            for n in xn:
                # Find which process this facet came from
                q = np.searchsorted(offsets, n, side='right') - 1
                # Add to the sendback list, if not the same process
                if q != p:
                    pr.add(q)
            procs[p] += [list(pr)]

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


xdmf = XDMFFile(MPI.COMM_WORLD, 'xmas_tree.xdmf', 'r')
mesh = xdmf.read_mesh()
tdim = mesh.topology.dim
num_cell_vertices = cell_num_vertices(mesh.topology.cell_type)
facet_type = cell_entity_type(mesh.topology.cell_type, tdim - 1, 0)
num_facet_vertices = cell_num_vertices(facet_type)
mesh.topology.create_entities(tdim - 1)
marker = xdmf.read_meshtags(mesh, 'facet_marker')

# Get cells attached to marked facets
mesh.topology.create_connectivity(tdim - 1, tdim)
fc = mesh.topology.connectivity(tdim - 1, tdim)
fv = mesh.topology.connectivity(tdim - 1, 0)
ncells = mesh.topology.index_map(tdim).size_local

# Get subset of markers on desired interfaces
contact_keys = (5, 6)
marker_subset_i = [i for i, (idx, k) in enumerate(zip(marker.indices, marker.values)) if k in contact_keys]
marker_subset = marker.indices[marker_subset_i]
marker_subset_val = marker.values[marker_subset_i]

cell_dests = compute_ghost_cell_destinations(mesh, marker_subset, 0.1)

cells_to_ghost = [fc.links(f)[0] for f in marker_subset]
assert len(cell_dests) == len(cells_to_ghost)

# TODO: deal with duplicates here?
cell_to_dests = {c: d for c, d in zip(cells_to_ghost, cell_dests)}

# Collect up locally occuring markers

# Convert marked facets to list of (global) vertices for each facet
all_indices = [sorted(mesh.topology.index_map(0).local_to_global(fv.links(f))) for f in marker.indices]
all_values = list(marker.values)

# Send any 'extras' over to processes that need them
send_buffer = [[] for p in range(mesh.comm.size)]
for i, f in enumerate(marker_subset):
    val = marker_subset_val[i]
    for dest in cell_dests[i]:
        send_buffer[dest] += sorted(mesh.topology.index_map(0).local_to_global(fv.links(f))) + [val]

# Append received markers
recv_buffer = np.array(sum(mesh.comm.alltoall(send_buffer), [])).reshape((-1, num_facet_vertices + 1))
for row in recv_buffer:
    all_indices += [list(row[:-1])]
    all_values += [row[-1]]


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
x = mesh.geometry.x[:num_vertices, :]
domain = mesh.ufl_domain()
new_mesh = create_mesh(mesh.comm, topo, x, domain, partitioner)

# Remap vertices back to input indexing
# This is rather messy, we need to map vertices to geometric nodes
# then back to original index
global_remap = np.array(new_mesh.geometry.input_global_indices, dtype=np.int32)
nv = new_mesh.topology.index_map(0).size_local + new_mesh.topology.index_map(0).num_ghosts
vert_to_geom = entities_to_geometry(new_mesh, 0, np.arange(nv, dtype=np.int32), False).flatten()
rmap = np.vectorize(lambda idx: global_remap[vert_to_geom[idx]])

# Recreate facets
new_mesh.topology.create_entities(tdim - 1)
new_mesh.topology.create_connectivity(tdim - 1, tdim)

# Create a list of all facet-vertices (original global index)
fv = new_mesh.topology.connectivity(tdim - 1, 0)
fv_indices = rmap(fv.array).reshape((-1, tdim))
fv_indices = np.sort(fv_indices, axis=1)

# Search for marked facets in list of all facets
new_markers = []
for idx, val in zip(all_indices, all_values):
    f = np.nonzero(np.all(fv_indices == idx, axis=1))[0]
    if len(f) > 0:
        assert len(f) == 1
        f = f[0]
        new_markers += [[f, val]]

# Sort new markers into order and make unique
new_markers = np.array(sorted(new_markers), dtype=np.int32)
new_markers = np.unique(new_markers, axis=0)

new_meshtag = meshtags(new_mesh, tdim - 1, new_markers[:, 0],
                       new_markers[:, 1])


# Write out
new_xdmf = XDMFFile(MPI.COMM_WORLD, "output.xdmf", "w")
new_xdmf.write_mesh(new_mesh)
new_xdmf.write_meshtags(new_meshtag)

import matplotlib.pyplot as plt
tri = np.array([vert_to_geom[t] for t in new_mesh.topology.connectivity(tdim, 0).array]).reshape(-1, tdim + 1)

# Set colours for owned and ghost cells
cols = 0.75 * np.ones(tri.shape[0])
cols[:-new_mesh.topology.index_map(tdim).num_ghosts] = 1.0

x = new_mesh.geometry.x
plt.tripcolor(x[:,0], x[:, 1], tri, cols, linewidth=0.2, color='k', cmap='gray', vmin=0, vmax=1)
plt.gca().set_aspect('equal')

# Plot facets by marker color
facet_geom = entities_to_geometry(new_mesh, tdim - 1, new_meshtag.indices, False)
lc = {5:'red', 6:'green', 4:'blue', 3:'yellow'}
for k, v in zip(facet_geom, new_meshtag.values):
    pt = np.array([x[idx] for idx in k])
    plt.plot(pt[:, 0], pt[:, 1], color=lc[v])

plt.show()
