# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT

import gmsh
import numpy as np
from mpi4py import MPI

__all__ = ["create_circle_plane_mesh", "create_circle_circle_mesh", "create_box_mesh_2D",
           "create_box_mesh_3D", "create_sphere_plane_mesh", "create_sphere_sphere_mesh"]


def create_circle_plane_mesh(filename: str, quads: bool = False):
    """
    Create a circular mesh, with center at (0.5,0.5,0) with radius 3 and a box [0,1]x[0,0.1]
    """
    center = [0.5, 0.5, 0]
    r = 0.3
    angle = np.pi / 4
    L = 1
    H = 0.1
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
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
        bndry = gmsh.model.getBoundary([(2, surface)], oriented=False)
        [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry]

        gmsh.model.addPhysicalGroup(2, [surface2], 2)
        bndry2 = gmsh.model.getBoundary([(2, surface2)], oriented=False)
        [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry2]

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "NodesList", [c])

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", 0.01)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 0.01)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.3)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.6)
        if quads:
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
            gmsh.option.setNumber("Mesh.RecombineAll", 2)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.model.mesh.generate(2)
        # gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(filename)
    MPI.COMM_WORLD.Barrier()
    gmsh.finalize()


def create_circle_circle_mesh(filename: str, quads: bool = False):
    """
    Create two circular meshes, with radii 0.3 and 0.6 with centers (0.5,0.5) and (0.5, -0.5)
    """
    center = [0.5, 0.5, 0]
    r = 0.3
    angle = np.pi / 4

    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
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

        # Synchronize and create physical tags
        gmsh.model.occ.addPoint(0.5, 0.2, 0, tag=17)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface])
        bndry = gmsh.model.getBoundary([(2, surface)], oriented=False)
        [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry]

        gmsh.model.addPhysicalGroup(2, [surface2], 2)
        bndry2 = gmsh.model.getBoundary([(2, surface2)], oriented=False)
        [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry2]

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "NodesList", [17])

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", 0.005)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 0.015)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.3)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.6)
        if quads:
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
            gmsh.option.setNumber("Mesh.RecombineAll", 2)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.model.mesh.generate(2)
        # gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(filename)
    MPI.COMM_WORLD.Barrier()
    gmsh.finalize()


def create_box_mesh_2D(filename: str, quads: bool = False):
    """
    Create two boxes, one slightly skewed
    """
    L = 0.5
    H = 0.5
    disp = -0.6
    delta = 0.1

    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:

        # Create box
        p0 = gmsh.model.occ.addPoint(-delta, 0, 0)
        p1 = gmsh.model.occ.addPoint(L - delta, delta, 0)
        p2 = gmsh.model.occ.addPoint(L - delta, H + delta, 0)
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
        res = 0.1
        # Set mesh sizes on the points from the surface we are extruding
        top_nodes = gmsh.model.getBoundary([(2, surface)], recursive=True, oriented=False)
        gmsh.model.occ.mesh.setSize(top_nodes, res)
        bottom_nodes = gmsh.model.getBoundary([(2, surface2)], recursive=True, oriented=False)
        gmsh.model.occ.mesh.setSize(bottom_nodes, 2 * res)

        # Synchronize and create physical tags
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface])
        bndry = gmsh.model.getBoundary([(2, surface)], oriented=False)
        [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry]

        gmsh.model.addPhysicalGroup(2, [surface2], 2)
        bndry2 = gmsh.model.getBoundary([(2, surface2)], oriented=False)
        [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry2]
        if quads:
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
            gmsh.option.setNumber("Mesh.RecombineAll", 2)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.generate(2)
        # gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(filename)
    MPI.COMM_WORLD.Barrier()

    gmsh.finalize()


def create_box_mesh_3D(filename: str):
    """
    Create two boxes lying directly over eachother with a gap in between"""
    L = 0.5
    H = 0.5
    W = 0.5

    disp = -0.6
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        # Create box
        box = gmsh.model.occ.add_box(0, 0, 0, L, H, W)
        box2 = gmsh.model.occ.add_box(0, 0, disp, L, H, W)
        gmsh.model.occ.synchronize()
        res = 0.1
        # Set mesh sizes on the points from the surface we are extruding
        # top_nodes = gmsh.model.getBoundary([(2, box)], recursive=True)
        # gmsh.model.occ.mesh.setSize(top_nodes, res)
        # bottom_nodes = gmsh.model.getBoundary([(2, box2)], recursive=True)
        # gmsh.model.occ.mesh.setSize(bottom_nodes, 2 * res)
        gmsh.model.mesh.field.add("Box", 1)
        gmsh.model.mesh.field.setNumber(1, "VIn", res / 5.)
        gmsh.model.mesh.field.setNumber(1, "VOut", res)
        gmsh.model.mesh.field.setNumber(1, "XMin", 0)
        gmsh.model.mesh.field.setNumber(1, "XMax", L)
        gmsh.model.mesh.field.setNumber(1, "YMin", 0)
        gmsh.model.mesh.field.setNumber(1, "YMax", H)
        gmsh.model.mesh.field.setNumber(1, "ZMin", 0)
        gmsh.model.mesh.field.setNumber(1, "ZMax", W)

        gmsh.model.mesh.field.setAsBackgroundMesh(1)

        # Synchronize and create physical tags
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [box])
        bndry = gmsh.model.getBoundary([(2, box)], oriented=False)
        [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry]

        gmsh.model.addPhysicalGroup(3, [box2])
        bndry2 = gmsh.model.getBoundary([(2, box2)], oriented=False)
        [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry2]

        gmsh.model.mesh.generate(3)
        # gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(filename)
    MPI.COMM_WORLD.Barrier()
    gmsh.finalize()


def create_sphere_plane_mesh(filename: str):
    """
    Create a 3D sphere with center (0,0,0), r=0.3
    with a box at [-0.3, 0.6] x [-0.3, 0.6] x [ -0.1, -0.5]
    """
    center = [0.0, 0.0, 0.0]
    r = 0.3
    angle = np.pi / 8
    gap = 0.05
    H = 0.05
    theta = 0  # np.pi / 10
    LcMin = 0.05 * r
    LcMax = 0.2 * r
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
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

        sphere_boundary = gmsh.model.getBoundary(out_vol_tags, oriented=False)
        for boundary_tag in sphere_boundary:
            gmsh.model.addPhysicalGroup(boundary_tag[0], boundary_tag[1:2])
        box_boundary = gmsh.model.getBoundary([(3, box)], oriented=False)
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
    MPI.COMM_WORLD.Barrier()
    gmsh.finalize()


def create_sphere_sphere_mesh(filename: str):
    """
    Create a 3D mesh consisting of two spheres with radii 0.3 and 0.6 and
    centers (0.5,0.5,0.5) and (0.5,0.5,-0.5)
    """
    center = [0.5, 0.5, 0.5]
    r = 0.3
    angle = np.pi / 8
    LcMin = 0.05 * r
    LcMax = 0.2 * r
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        # Create sphere 1 composed of of two volumes
        sphere_bottom = gmsh.model.occ.addSphere(center[0], center[1], center[2], r, angle1=-np.pi / 2, angle2=-angle)
        p0 = gmsh.model.occ.addPoint(center[0], center[1], center[2] - r)
        sphere_top = gmsh.model.occ.addSphere(center[0], center[1], center[2], r, angle1=-angle, angle2=np.pi / 2)
        out_vol_tags, _ = gmsh.model.occ.fragment([(3, sphere_bottom)], [(3, sphere_top)])

        # Create sphere 2 composed of of two volumes
        sphere_bottom2 = gmsh.model.occ.addSphere(
            center[0], center[1], -center[2], 2 * r, angle1=-np.pi / 2, angle2=-angle)
        p1 = gmsh.model.occ.addPoint(center[0], center[1], -center[2] - 2 * r)
        sphere_top2 = gmsh.model.occ.addSphere(center[0], center[1], -center[2], 2 * r, angle1=-angle, angle2=np.pi / 2)
        out_vol_tags2, _ = gmsh.model.occ.fragment([(3, sphere_bottom2)], [(3, sphere_top2)])

        # Synchronize and create physical tags
        gmsh.model.occ.synchronize()

        sphere_boundary = gmsh.model.getBoundary(out_vol_tags, oriented=False)
        for boundary_tag in sphere_boundary:
            gmsh.model.addPhysicalGroup(boundary_tag[0], boundary_tag[1:2])
        sphere_boundary2 = gmsh.model.getBoundary(out_vol_tags2, oriented=False)
        for boundary_tag in sphere_boundary2:
            gmsh.model.addPhysicalGroup(boundary_tag[0], boundary_tag[1:2])

        p_v = [v_tag[1] for v_tag in out_vol_tags]
        p_v2 = [v_tag[1] for v_tag in out_vol_tags2]
        gmsh.model.addPhysicalGroup(3, p_v)
        gmsh.model.addPhysicalGroup(3, p_v2)

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "NodesList", [p0, p1])

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
    MPI.COMM_WORLD.Barrier()
    gmsh.finalize()
