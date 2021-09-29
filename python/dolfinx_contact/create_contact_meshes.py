# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT

import argparse
import warnings

import gmsh
# from mpi4py import MPI
import numpy as np
from dolfinx_contact.helpers import convert_mesh

warnings.filterwarnings("ignore")


def create_circle_plane_mesh(filename: str):
    center = [0.5, 0.5, 0]
    r = 0.3
    angle = np.pi / 4
    L = 1
    H = 0.1

    gmsh.initialize()
    # Create circular mesh (divided into 4 segments)
    c = gmsh.model.occ.addPoint(center[0], center[1], center[2])
    # Add 4 points on circle (clockwise, starting in top left)
    angles = [angle, -angle, np.pi + angle, np.pi - angle]
    c_points = [gmsh.model.occ.addPoint(center[0] + r * np.cos(angle), center[1]
                                        + r * np.sin(angle), center[2]) for angle in angles]
    arcs = [gmsh.model.occ.addCircleArc(
        c_points[i - 1], c, c_points[i]) for i in range(len(c_points))]
    curve = gmsh.model.occ.addCurveLoop(arcs)
    gmsh.model.occ.synchronize()
    surface = gmsh.model.occ.addPlaneSurface([curve])
    # Create box
    p0 = gmsh.model.occ.addPoint(0, 0, 0)
    p1 = gmsh.model.occ.addPoint(L, 0, 0)
    p2 = gmsh.model.occ.addPoint(L, H, 0)
    p3 = gmsh.model.occ.addPoint(0, H, 0)
    ps = [p0, p1, p2, p3]
    lines = [gmsh.model.occ.addLine(ps[i - 1], ps[i]) for i in range(len(ps))]
    curve2 = gmsh.model.occ.addCurveLoop(lines)
    surface2 = gmsh.model.occ.addPlaneSurface([curve, curve2])

    # Synchronize and create physical tags
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [surface])
    bndry = gmsh.model.getBoundary([(2, surface)])
    [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry]

    gmsh.model.addPhysicalGroup(2, [surface2], 2)
    bndry2 = gmsh.model.getBoundary([(2, surface2)])
    [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry2]

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "NodesList", [c])

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", 0.01)
    gmsh.model.mesh.field.setNumber(2, "LcMax", 0.01)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.3)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.6)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.model.mesh.generate(2)
    # gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(filename)

    gmsh.finalize()


def create_circle_circle_mesh(filename: str):
    center = [0.5, 0.5, 0]
    r = 0.3
    angle = np.pi / 4

    gmsh.initialize()
    # Create circular mesh (divided into 4 segments)
    c = gmsh.model.occ.addPoint(center[0], center[1], center[2])
    # Add 4 points on circle (clockwise, starting in top left)
    angles = [angle, -angle, np.pi + angle, np.pi - angle]
    c_points = [gmsh.model.occ.addPoint(center[0] + r * np.cos(angle), center[1]
                                        + r * np.sin(angle), center[2]) for angle in angles]
    arcs = [gmsh.model.occ.addCircleArc(
        c_points[i - 1], c, c_points[i]) for i in range(len(c_points))]
    curve = gmsh.model.occ.addCurveLoop(arcs)
    gmsh.model.occ.synchronize()
    surface = gmsh.model.occ.addPlaneSurface([curve])
    # Create 2nd circular mesh (divided into 4 segments)
    center2 = [0.5, -0.5, 0]
    c2 = gmsh.model.occ.addPoint(center2[0], center2[1], center2[2])
    # Add 4 points on circle (clockwise, starting in top left)
    c_points2 = [gmsh.model.occ.addPoint(center2[0] + 2 * r * np.cos(angle), center2[1]
                                         + 2 * r * np.sin(angle), center2[2]) for angle in angles]
    arcs2 = [gmsh.model.occ.addCircleArc(
        c_points2[i - 1], c2, c_points2[i]) for i in range(len(c_points2))]
    curve2 = gmsh.model.occ.addCurveLoop(arcs2)
    gmsh.model.occ.synchronize()
    surface2 = gmsh.model.occ.addPlaneSurface([curve, curve2])
    # # Create box
    # p0 = gmsh.model.occ.addPoint(0, 0, 0)
    # p1 = gmsh.model.occ.addPoint(L, 0, 0)
    # p2 = gmsh.model.occ.addPoint(L, H, 0)
    # p3 = gmsh.model.occ.addPoint(0, H, 0)
    # ps = [p0, p1, p2, p3]
    # lines = [gmsh.model.occ.addLine(ps[i - 1], ps[i]) for i in range(len(ps))]
    # curve2 = gmsh.model.occ.addCurveLoop(lines)
    # surface2 = gmsh.model.occ.addPlaneSurface([curve, curve2])

    # Synchronize and create physical tags
    gmsh.model.occ.addPoint(0.5, 0.2, 0, tag=17)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [surface])
    bndry = gmsh.model.getBoundary([(2, surface)])
    [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry]

    gmsh.model.addPhysicalGroup(2, [surface2], 2)
    bndry2 = gmsh.model.getBoundary([(2, surface2)])
    [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry2]

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "NodesList", [17])

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", 0.005)
    gmsh.model.mesh.field.setNumber(2, "LcMax", 0.015)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.3)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 0.6)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.model.mesh.generate(2)
    # gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(filename)

    gmsh.finalize()


def create_sphere_plane_mesh(filename: str):
    center = [0.0, 0.0, 0.0]
    r = 0.3
    angle = np.pi / 8
    # L = 1
    # B = 1
    gap = 0.05
    H = 0.05
    theta = 0  # np.pi / 10
    LcMin = 0.05 * r
    LcMax = 0.2 * r
    gmsh.initialize()
    # Create sphere composed of of two volumes
    sphere_bottom = gmsh.model.occ.addSphere(center[0], center[1], center[2], r, angle1=-np.pi / 2, angle2=-angle)
    p0 = gmsh.model.occ.addPoint(center[0], center[1], center[2] - r)
    sphere_top = gmsh.model.occ.addSphere(center[0], center[1], center[2], r, angle1=-angle, angle2=np.pi / 2)
    out_vol_tags, _ = gmsh.model.occ.fragment([(3, sphere_bottom)], [(3, sphere_top)])

    # Add bottom box
    box = gmsh.model.occ.add_box(center[0] - r, center[1] - r, center[2] - r - gap - H, 3 * r, 3 * r, H)
    # Rotate after marking boundaries
    gmsh.model.occ.rotate([(3, box)], center[0], center[1], center[2]
                          - r - 3 * gap, 1, 0, 0, theta)
    # Synchronize and create physical tags
    gmsh.model.occ.synchronize()

    sphere_boundary = gmsh.model.getBoundary(out_vol_tags)
    for boundary_tag in sphere_boundary:
        gmsh.model.addPhysicalGroup(boundary_tag[0], boundary_tag[1:2])
    box_boundary = gmsh.model.getBoundary([(3, box)])
    for boundary_tag in box_boundary:
        gmsh.model.addPhysicalGroup(boundary_tag[0], boundary_tag[1:2])

    p_v = [v_tag[1] for v_tag in out_vol_tags]
    gmsh.model.addPhysicalGroup(3, p_v)
    gmsh.model.addPhysicalGroup(3, [box])

    gmsh.model.occ.synchronize()
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "NodesList", [p0])

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", LcMin)
    gmsh.model.mesh.field.setNumber(2, "LcMax", LcMax)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5 * r)
    gmsh.model.mesh.field.setNumber(2, "DistMax", r)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.model.mesh.generate(3)
    # gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.write(filename)

    gmsh.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='GMSH Python API script for creating a 2D/3D mesh for a circle/sphere and a rectangle/box')
    parser.add_argument("--name", default="disk", type=str, dest="name",
                        help="Name of file to write the mesh to (without filetype extension)")
    _3D = parser.add_mutually_exclusive_group(required=False)
    _3D.add_argument('--3D', dest='threed', action='store_true',
                     help="Use 3D mesh", default=False)

    args = parser.parse_args()
    if args.threed:
        create_sphere_plane_mesh(filename=f"{args.name}.msh")
        convert_mesh(args.name, "tetra")
        convert_mesh(f"{args.name}", "triangle", ext="facets")
    else:
        create_circle_plane_mesh(filename=f"{args.name}.msh")
        convert_mesh(args.name, "triangle", prune_z=True)
        convert_mesh(f"{args.name}", "line", ext="facets", prune_z=True)
