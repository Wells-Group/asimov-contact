from dolfinx.mesh import create_mesh, meshtags
from dolfinx.io import XDMFFile
import basix
import ufl
from mpi4py import MPI
import sys
import numpy as np

fn = sys.argv[1]
f = open(fn, 'r')
data = f.readlines()
f.close()

gindex = 1
tindex = 1
geometry = []
cells = []
marker0 = []
for line in data:
    b = line.split()
    if len(b) == 0 or b[0][0] == '$':
        continue
    if b[0] == 'GRID*':
        i = int(b[1])
        assert i == gindex
        gindex += 1
        x = float(b[2])
        y = float(b[3])
        geometry += [[x, y]]
    if b[0] == 'CTRIA6':
        j = int(b[1])
        assert j == tindex
        tindex += 1
        w = [int(k)-1 for k in b[3:]]
        cells += [w]
        marker0 += [int(b[2])]


x = np.array(geometry)
cells = np.array(cells)
perm = [0, 1, 2, 4, 5, 3]
cells = cells[:, perm].copy()

cell = ufl.Cell("triangle", geometric_dimension=2)
element = basix.ufl.element(basix.ElementFamily.P, cell.cellname(), 2,
                            basix.LagrangeVariant.equispaced, shape=(2,),
                            gdim=2, dtype=np.float64)
domain = ufl.Mesh(element)
mesh = create_mesh(MPI.COMM_WORLD, cells, x, domain)

ncell = mesh.topology.index_map(2).size_local
marker = np.zeros(ncell, dtype=np.int32)
perm = mesh.topology.original_cell_index
for i in range(ncell):
    marker[i] = marker0[perm[i]]

entities = np.arange(ncell, dtype=np.int32)
tags = meshtags(mesh, 2, entities, marker)
tags.name = "volume"

# make surface markers
topo = mesh.topology
topo.create_entities(1)
topo.create_connectivity(1, 2)
f_to_c = topo.connectivity(1, 2)
facet_indices = []
facet_markers = []
for f in range(topo.index_map(1).size_local):
    q = f_to_c.links(f)
    if len(q) == 1:
        print(f, marker[q[0]])
        facet_indices += [f]
        facet_markers += [marker[q[0]]]

facet_indices = np.array(facet_indices, dtype=np.int32)
facet_markers = np.array(facet_markers, dtype=np.int32)
facet_meshtags = meshtags(mesh, 1, facet_indices, facet_markers)
facet_meshtags.name = "facets"


fn = fn.split(".")
xdmf = XDMFFile(mesh.comm, fn[0]+".xdmf", "w", XDMFFile.Encoding.ASCII)
xdmf.write_mesh(mesh)
xdmf.write_meshtags(tags, mesh.geometry)
xdmf.write_meshtags(facet_meshtags, mesh.geometry)
xdmf.close()
