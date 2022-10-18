# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import gmsh
from mpi4py import MPI
from dolfinx_contact.meshing import convert_mesh


def create_box_mesh(filename: str, quads: bool = False, res=0.1, order: int = 1):
    """
    Create two boxes, one slightly skewed
    """
    L = 0.5
    H = 0.5
    disp = -0.6
    delta = 0.0

    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", res)

        # Create box
        p0 = gmsh.model.occ.addPoint(-delta, 0, 0)
        p1 = gmsh.model.occ.addPoint(L - delta, 0, 0)
        p2 = gmsh.model.occ.addPoint(L - delta, H, 0)
        p3 = gmsh.model.occ.addPoint(-delta, H, 0)
        ps = [p0, p1, p2, p3]
        lines = [gmsh.model.occ.addLine(ps[i - 1], ps[i]) for i in range(len(ps))]
        curve = gmsh.model.occ.addCurveLoop(lines)
        surface = gmsh.model.occ.addPlaneSurface([curve])

        # Create box
        p4 = gmsh.model.occ.addPoint(0, 0 + disp, 0)
        p5 = gmsh.model.occ.addPoint(L, 0 + disp, 0)
        p6 = gmsh.model.occ.addPoint(L, H + disp, 0)
        p7 = gmsh.model.occ.addPoint(0, H + disp, 0)
        ps2 = [p4, p5, p6, p7]
        lines2 = [gmsh.model.occ.addLine(ps2[i - 1], ps2[i]) for i in range(len(ps2))]
        curve2 = gmsh.model.occ.addCurveLoop(lines2)
        surface2 = gmsh.model.occ.addPlaneSurface([curve2])

        gmsh.model.occ.synchronize()
        # Set mesh sizes on the points from the surface we are extruding
        top_nodes = gmsh.model.getBoundary([(2, surface)], recursive=True, oriented=False)
        gmsh.model.occ.mesh.setSize(top_nodes, res)
        bottom_nodes = gmsh.model.getBoundary([(2, surface2)], recursive=True, oriented=False)
        gmsh.model.occ.mesh.setSize(bottom_nodes, 2 * res)

        # Synchronize and create physical tags
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface], tag=3)
        bndry = gmsh.model.getBoundary([(2, surface)], oriented=False)
        gmsh.model.addPhysicalGroup(1, [b[1] for b in bndry], tag=1)

        gmsh.model.addPhysicalGroup(2, [surface2], tag=4)
        bndry2 = gmsh.model.getBoundary([(2, surface2)], oriented=False)
        gmsh.model.addPhysicalGroup(1, [b[1] for b in bndry2], tag=2)

        if quads:
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
            gmsh.option.setNumber("Mesh.RecombineAll", 2)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)

        # gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(filename)
    MPI.COMM_WORLD.Barrier()

    gmsh.finalize()


fname = "test_mesh"
create_box_mesh(filename=f"{fname}.msh", quads=False, res=4.0,
                order=1)
convert_mesh(fname, f"{fname}.xdmf", gdim=2)
