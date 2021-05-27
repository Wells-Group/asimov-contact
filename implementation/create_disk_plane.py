import argparse
import warnings

import gmsh
from mpi4py import MPI
import numpy as np
from helpers import convert_mesh

warnings.filterwarnings("ignore")


def create_circle_plane_mesh(filename: str):
    center = [0.5, 0.5, 0]
    r = 0.3
    angle = np.pi/4
    L = 1
    H = 0.1

    gmsh.initialize()
    # Create circular mesh (divided into 4 segments)
    c = gmsh.model.occ.addPoint(center[0], center[1], center[2])
    a1 = gmsh.model.occ.addPoint(
        center[0]+r*np.cos(-angle), center[1]+r*np.sin(-angle), center[2])
    a2 = gmsh.model.occ.addPoint(center[0]+r*np.cos(np.pi+angle),
                                 center[1]+r*np.sin(np.pi+angle), center[2])
    a3 = gmsh.model.occ.addPoint(
        center[0]+r*np.cos(np.pi/2-angle), center[1]+r*np.sin(np.pi/2-angle), 0)
    a4 = gmsh.model.occ.addPoint(
        center[0]+r*np.cos(np.pi/2+angle), center[1]+r*np.sin(np.pi/2+angle), 0)
    c_points = [a1, a2, a3, a4]
    arcs = [gmsh.model.occ.addCircleArc(
        c_points[i-1], c, c_points[i]) for i in range(len(c_points))]
    curve = gmsh.model.occ.addCurveLoop(arcs)
    gmsh.model.occ.synchronize()
    surface = gmsh.model.occ.addPlaneSurface([curve])
    # Create box
    p0 = gmsh.model.occ.addPoint(0, 0, 0)
    p1 = gmsh.model.occ.addPoint(L, 0, 0)
    p2 = gmsh.model.occ.addPoint(L, H, 0)
    p3 = gmsh.model.occ.addPoint(0, H, 0)
    ps = [p0, p1, p2, p3]
    lines = [gmsh.model.occ.addLine(ps[i-1], ps[i]) for i in range(len(ps))]
    curve2 = gmsh.model.occ.addCurveLoop(lines)
    surface2 = gmsh.model.occ.addPlaneSurface([curve, curve2])

    # Synchronize and create physical tags
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [surface], 1)
    gmsh.model.addPhysicalGroup(2, [surface2], 2)

    gmsh.model.addPhysicalGroup(1, [arcs[1]], 3)
    gmsh.model.addPhysicalGroup(1, [arcs[3]], 4)
    gmsh.model.addPhysicalGroup(1, [lines[3]])
    gmsh.model.mesh.generate(2)
    # gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(filename)

    gmsh.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='GMSH Python API script for creating a 2D disk mesh ')
    parser.add_argument("--name", default="disk", type=str, dest="name",
                        help="Name of file to write the mesh to (without filetype extension)")
    args = parser.parse_args()

    create_circle_plane_mesh(filename=f"{args.name}.msh")
    convert_mesh(args.name, "triangle", prune_z=True)
    convert_mesh(f"{args.name}", "line", ext="facets", prune_z=True)
