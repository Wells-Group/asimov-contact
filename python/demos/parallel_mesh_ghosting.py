from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_mesh, meshtags
import dolfinx
import numpy as np

xdmf = XDMFFile(MPI.COMM_WORLD, 'box_2D.xdmf', 'r')
mesh = xdmf.read_mesh()
tdim = mesh.topology.dim
mesh.topology.create_entities(tdim - 1)
marker = xdmf.read_meshtags(mesh, 'facet_marker')

# Get cells attached to marked facets
mesh.topology.create_connectivity(tdim - 1, tdim)
fc = mesh.topology.connectivity(tdim - 1, tdim)
fv = mesh.topology.connectivity(tdim - 1, 0)
cells_to_ghost = [fc.links(f)[0] for f in marker.indices]
ncells = mesh.topology.index_map(tdim).size_local

# Convert marked facets to list of (global) vertices for each facet
fv_indices = [sorted(mesh.topology.index_map(0).local_to_global(fv.links(f))) for f in marker.indices]

# Copy facets and markers to all processes
global_markers = sum(fv_indices, [])
all_indices = mesh.comm.allgather(global_markers)
all_indices = np.array(sum(all_indices, [])).reshape(-1, tdim)
all_values = np.array(sum(mesh.comm.allgather(list(marker.values)), []))


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
topo = mesh.topology.index_map(0).local_to_global(topo).reshape((-1, tdim + 1))
topo = topo[:ncells, :]

# Cut off any ghost vertices
num_vertices = mesh.topology.index_map(0).size_local
x = mesh.geometry.x[:num_vertices, :]
domain = mesh.ufl_domain()
new_mesh = create_mesh(mesh.comm, topo, x, domain, partitioner)

# Remap vertices back to input indexing
global_remap = new_mesh.geometry.input_global_indices
rmap = np.vectorize(lambda idx: global_remap[idx])

# Recreate facets
new_mesh.topology.create_entities(tdim - 1)
new_mesh.topology.create_connectivity(tdim - 1, tdim)

# Create a list of all facet-vertices (original global index)
fv = new_mesh.topology.connectivity(tdim - 1, 0)
fv_indices = rmap(fv.array).reshape((-1, 2))
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
