import numpy as np
import gmsh
from mpi4py import MPI

def add_bars(idx, r, n):
    for j in range(n):
        gmsh.model.occ.addCylinder(5.0 + j, 0.0, -2.0, 0.0, 0.0, 4.0, r,
                                   idx + j * 2)
        gmsh.model.occ.addCylinder(5.0 + j, -2.0, 0.0, 0.0, 4.0, 0.0, r,
                                   idx + j * 2 + 1)

gmsh.initialize()

gmsh.model.add("box-key")

d = 0.05
gmsh.model.occ.addBox(0, -1+d, -1+d, 10, 2-2*d, 2-2*d, 0)
gmsh.model.occ.addBox(0, -1, -1, 10, 2, 2, 1)
d = 0.4
gmsh.model.occ.addBox(0, -1-d, -1-d, 10, 2+2*d, 2+2*d, 2)
gmsh.model.occ.cut([(3, 2)], [(3, 1)], 3)

idx = 4
n = 2
r = 0.4
add_bars(idx, r, n)
tag_outer = gmsh.model.occ.cut([(3, 3)], [(3, i) for i in range(idx, idx + 2*n)])
tag_outer = tag_outer[0][0][1]

idx += 2*n
d = 0.05
add_bars(idx, r - d, n)
tag_inner = gmsh.model.occ.fuse([(3, 0)], [(3, i) for i in range(idx, idx + 2*n)])
tag_inner = tag_inner[0][0][1]

gmsh.model.occ.synchronize()
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(3, [tag_inner], 1)
gmsh.model.setPhysicalName(3, 1, "InnerVolume")

gmsh.model.addPhysicalGroup(3, [tag_outer], 2)
gmsh.model.setPhysicalName(3, 2, "OuterVolume")

inner_surfaces = gmsh.model.getAdjacencies(dim=3, tag=tag_inner)[1]
outer_surfaces = gmsh.model.getAdjacencies(dim=3, tag=tag_outer)[1]
print('inner = ', tag_inner, len(inner_surfaces))
print('outer = ', tag_outer, len(outer_surfaces))


for s in inner_surfaces:
    com = gmsh.model.occ.getCenterOfMass(2, s)
    if np.isclose(com[0], 10.0):
        marker = 13
        gmsh.model.addPhysicalGroup(2, [s], marker)
        gmsh.model.setPhysicalName(2, marker, "InnerTop")
    if np.isclose(com[0], 0.0):
        marker = 3
        gmsh.model.addPhysicalGroup(2, [s], marker)
        gmsh.model.setPhysicalName(2, marker, "InnerBot")

for s in outer_surfaces:
    com = gmsh.model.occ.getCenterOfMass(2, s)
    if np.isclose(com[0], 10.0):
        marker = 4
        gmsh.model.addPhysicalGroup(2, [s], marker)
        gmsh.model.setPhysicalName(2, marker, "OuterTop")
    if np.isclose(com[0], 0.0):
        marker = 12
        gmsh.model.addPhysicalGroup(2, [s], marker)
        gmsh.model.setPhysicalName(2, marker, "OuterBot")

inner_marker = 6
gmsh.model.addPhysicalGroup(2, inner_surfaces, inner_marker)
gmsh.model.setPhysicalName(2, inner_marker, "Inner")

outer_marker = 7
gmsh.model.addPhysicalGroup(2, outer_surfaces, outer_marker)
gmsh.model.setPhysicalName(2, outer_marker, "Outer")

gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.1)

gmsh.model.mesh.generate(3)
gmsh.write("box-key.msh")

gmsh.finalize()
