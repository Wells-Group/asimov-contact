# Copyright (C) 2022 Chris N. Richardson and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from mpi4py import MPI
from dolfinx.io import XDMFFile
from dolfinx.mesh import create_mesh, meshtags
import dolfinx
import numpy as np

__all__ = ["create_contact_mesh"]


def create_contact_mesh(fname, facet_marker, domain_marker, tags):
    xdmf = XDMFFile(MPI.COMM_WORLD, f'{fname}.xdmf', 'r')
    mesh = xdmf.read_mesh()
    tdim = mesh.topology.dim
    mesh.topology.create_entities(tdim - 1)
    marker = xdmf.read_meshtags(mesh, facet_marker)
    dmarker = xdmf.read_meshtags(mesh, domain_marker)

    # Get cells attached to marked facets
    mesh.topology.create_connectivity(tdim - 1, tdim)
    fc = mesh.topology.connectivity(tdim - 1, tdim)
    facets = np.hstack([marker.find(tag) for tag in tags])

    cells = np.unique([fc.links(f)[0] for f in facets])
    ncells = mesh.topology.index_map(tdim).size_local

    def partitioner(comm, n, m, topo):
        rank = comm.Get_rank()
        other_ranks = [i for i in range(comm.Get_size()) if i != rank]

        dests = []
        offsets = [0]
        for c in range(ncells):
            dests.append(rank)
            if c in cells:
                dests.extend(other_ranks)  # Ghost to other processes
            offsets.append(len(dests))
        return dolfinx.cpp.graph.AdjacencyList_int32(dests, offsets)

    # Convert topology to global indexing, and restrict to non-ghost cells
    topo = mesh.topology.connectivity(tdim, 0).array
    topo = mesh.topology.index_map(0).local_to_global(topo).reshape((-1, 3))
    topo = topo[:ncells, :]

    # Cut off any ghost vertices
    num_vertices = mesh.topology.index_map(0).size_local
    gdim = mesh.geometry.dim
    x = mesh.geometry.x[:num_vertices, :gdim]
    domain = mesh.ufl_domain()
    new_mesh = create_mesh(mesh.comm, topo, x, domain, partitioner)
    new_xdmf = XDMFFile(MPI.COMM_WORLD, "output.xdmf", "w")
    new_xdmf.write_mesh(new_mesh)

    return new_mesh
