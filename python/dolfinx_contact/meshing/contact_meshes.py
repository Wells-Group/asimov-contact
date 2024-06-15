# Copyright (C) 2023 Jørgen S. Dokken and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import typing
from pathlib import Path

from mpi4py import MPI

import gmsh
import numpy as np

__all__ = [
    "create_circle_plane_mesh",
    "create_circle_circle_mesh",
    "create_box_mesh_3D",
    "create_gmsh_box_mesh_2D",
    "create_sphere_plane_mesh",
    "create_sphere_sphere_mesh",
    "create_cylinder_cylinder_mesh",
    "create_2d_rectangle_split",
    "create_quarter_disks_mesh",
    "sliding_wedges",
]


def create_circle_plane_mesh(
    model,
    quads: bool = False,
    res=0.1,
    order: int = 1,
    r: float = 0.25,
    height: float = 0.25,
    length: float = 1.0,
    gap: float = 0.01,
    comm: MPI.Comm = MPI.COMM_WORLD,
    rank: int = 0,
):
    """Create a circular mesh, with center at (0.0,0.0,0) with radius r
    and a box [-length/2, length/2]x[-height-gap-r,-gap-r]
    """
    center = [0, 0, 0]
    if comm.rank == 0:
        # Create circular mesh (divided into 4 segments)
        c = model.occ.addPoint(center[0], center[1], center[2])
        contact_pt = model.occ.addPoint(center[0], center[1] - r, center[2])
        left = model.occ.addPoint(-r, 0, 0)
        right = model.occ.addPoint(r, 0, 0)
        angle = np.pi / 3
        top_left = model.occ.addPoint(-r * np.cos(angle), r * np.sin(angle), 0)
        top_right = model.occ.addPoint(r * np.cos(angle), r * np.sin(angle), 0)

        arcs = [
            model.occ.addCircleArc(left, c, top_left),
            model.occ.addCircleArc(top_left, c, top_right),
            model.occ.addCircleArc(top_right, c, right),
            model.occ.addCircleArc(right, c, left),
        ]
        curve = model.occ.addCurveLoop(arcs)
        model.occ.synchronize()

        surface = model.occ.addPlaneSurface([curve])
        # Create box
        p0 = model.occ.addPoint(-length / 2, -height - r - gap, 0)
        p1 = model.occ.addPoint(length / 2, -height - r - gap, 0)
        p2 = model.occ.addPoint(length / 2, -r - gap, 0)
        p3 = model.occ.addPoint(-length / 2, -r - gap, 0)
        ps = [p0, p1, p2, p3]
        lines = [model.occ.addLine(ps[i - 1], ps[i]) for i in range(len(ps))]
        curve2 = model.occ.addCurveLoop(lines)
        surface2 = model.occ.addPlaneSurface([curve2])

        # Synchronize and create physical tags
        model.occ.synchronize()
        model.addPhysicalGroup(2, [surface], tag=1)

        model.addPhysicalGroup(2, [surface2], tag=2)
        bndry2 = model.getBoundary([(2, surface2)], oriented=False)
        [model.addPhysicalGroup(b[0], [b[1]]) for b in bndry2]
        [model.addPhysicalGroup(1, [arc]) for arc in arcs]

        model.mesh.field.add("Distance", 1)
        model.mesh.field.setNumbers(1, "NodesList", [contact_pt])
        model.mesh.field.add("Threshold", 2)
        model.mesh.field.setNumber(2, "IField", 1)
        model.mesh.field.setNumber(2, "LcMin", 0.5 * res)
        model.mesh.field.setNumber(2, "LcMax", 2 * res)
        model.mesh.field.setNumber(2, "DistMin", r / 2)
        model.mesh.field.setNumber(2, "DistMax", r)
        if quads:
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
            gmsh.option.setNumber("Mesh.RecombineAll", 2)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        model.mesh.field.setAsBackgroundMesh(2)

        model.mesh.generate(2)
        model.mesh.setOrder(order)
        return model
    else:
        return None


def create_halfdisk_plane_mesh(
    filename: typing.Union[str, Path],
    res=0.1,
    order: int = 1,
    quads=False,
    r=0.25,
    height=0.25,
    length=1.0,
    gap=0.01,
):
    """Create a halfdisk, with center at (0.0,0.0,0), radius r and  y<=0.0
    and a box [-length/2, length/2]x[-height-gap-r,-gap-r]
    """
    center = [0, 0, 0]
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        # Create circular mesh (divided into 4 segments)
        c = gmsh.model.occ.addPoint(center[0], center[1], center[2])
        pt_refine = gmsh.model.occ.addPoint(0.0, -r, 0.0)
        left = gmsh.model.occ.addPoint(-r, 0, 0)
        right = gmsh.model.occ.addPoint(r, 0, 0)
        arc = gmsh.model.occ.addCircleArc(right, c, left)
        line = gmsh.model.occ.addLine(left, right)

        curve = gmsh.model.occ.addCurveLoop([arc, line])
        gmsh.model.occ.synchronize()
        surface = gmsh.model.occ.addPlaneSurface([curve])
        # Create boxpy
        p0 = gmsh.model.occ.addPoint(-length / 2, -r - height - gap, 0)
        p1 = gmsh.model.occ.addPoint(length / 2, -r - height - gap, 0)
        p2 = gmsh.model.occ.addPoint(length / 2, -r - gap, 0)
        p3 = gmsh.model.occ.addPoint(-length / 2, -r - gap, 0)
        ps = [p0, p1, p2, p3]
        lines = [gmsh.model.occ.addLine(ps[i - 1], ps[i]) for i in range(len(ps))]
        curve2 = gmsh.model.occ.addCurveLoop(lines)
        surface2 = gmsh.model.occ.addPlaneSurface([curve2])

        # Synchronize and create physical tags
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface], tag=1)

        gmsh.model.addPhysicalGroup(2, [surface2], tag=2)
        bndry2 = gmsh.model.getBoundary([(2, surface2)], oriented=False)
        [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry2]
        gmsh.model.addPhysicalGroup(1, [arc])
        gmsh.model.addPhysicalGroup(1, [line])

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "NodesList", [pt_refine])

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", 0.5 * res)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 2 * res)
        gmsh.model.mesh.field.setNumber(2, "DistMin", r / 2)
        gmsh.model.mesh.field.setNumber(2, "DistMax", r)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
        if quads:
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
            gmsh.option.setNumber("Mesh.RecombineAll", 2)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)

        # gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(str(filename))
    MPI.COMM_WORLD.Barrier()
    gmsh.finalize()


def create_quarter_disks_mesh(
    filename: typing.Union[str, Path], res=0.1, order: int = 1, quads=False, r=0.25, gap=0.01
):
    """Create a quarter disk, with center at (0.0,0.0,0), radius r and  y<=0.0, x>=0
    and a a second quarter disk with center (0.0, -2r - gap, 0.0), radius r and y>= -3r-gap, x>=0
    """
    center = [0, 0, 0]
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        # Create first quarter disk
        c = gmsh.model.occ.addPoint(center[0], center[1], center[2])
        bottom1 = gmsh.model.occ.addPoint(0.0, -r, 0.0)
        top_right = gmsh.model.occ.addPoint(r, 0, 0)
        top_left = gmsh.model.occ.addPoint(-r, 0, 0)
        angle = np.pi / 6
        right1 = gmsh.model.occ.addPoint(r * np.sin(angle), -r * np.cos(angle), 0)
        left1 = gmsh.model.occ.addPoint(-r * np.sin(angle), -r * np.cos(angle), 0)
        arcs1 = []
        arcs1.append(gmsh.model.occ.addCircleArc(top_left, c, left1))
        arcs1.append(gmsh.model.occ.addCircleArc(left1, c, bottom1))
        arcs1.append(gmsh.model.occ.addCircleArc(bottom1, c, right1))
        arcs1.append(gmsh.model.occ.addCircleArc(right1, c, top_right))
        line1 = gmsh.model.occ.addLine(top_right, c)
        line2 = gmsh.model.occ.addLine(c, bottom1)
        line3 = gmsh.model.occ.addLine(top_left, c)
        curve = gmsh.model.occ.addCurveLoop([arcs1[2], arcs1[3], line1, line2])
        curve2 = gmsh.model.occ.addCurveLoop([-line2, -line3, arcs1[0], arcs1[1]])
        gmsh.model.occ.synchronize()
        surface = gmsh.model.occ.addPlaneSurface([curve])
        surface2 = gmsh.model.occ.addPlaneSurface([curve2])

        # Create second quarter disk
        c2 = gmsh.model.occ.addPoint(center[0], center[1] - 2 * r - gap, center[2])
        bottom_right = gmsh.model.occ.addPoint(r, -2 * r - gap, 0.0)
        bottom_left = gmsh.model.occ.addPoint(-r, -2 * r - gap, 0.0)
        top2 = gmsh.model.occ.addPoint(0, -r - gap, 0)
        right2 = gmsh.model.occ.addPoint(r * np.sin(angle), r * np.cos(angle) - 2 * r - gap, 0)
        left2 = gmsh.model.occ.addPoint(-r * np.sin(angle), r * np.cos(angle) - 2 * r - gap, 0)
        arcs2 = []
        arcs2.append(gmsh.model.occ.addCircleArc(bottom_left, c2, left2))
        arcs2.append(gmsh.model.occ.addCircleArc(left2, c2, top2))
        arcs2.append(gmsh.model.occ.addCircleArc(top2, c2, right2))
        arcs2.append(gmsh.model.occ.addCircleArc(right2, c2, bottom_right))
        line3 = gmsh.model.occ.addLine(top2, c2)
        line4 = gmsh.model.occ.addLine(c2, bottom_right)
        line5 = gmsh.model.occ.addLine(bottom_left, c2)
        curve3 = gmsh.model.occ.addCurveLoop([arcs2[2], arcs2[3], -line4, -line3])
        curve4 = gmsh.model.occ.addCurveLoop([arcs2[0], arcs2[1], line3, -line5])
        surface3 = gmsh.model.occ.addPlaneSurface([curve3])
        surface4 = gmsh.model.occ.addPlaneSurface([curve4])

        # Synchronize and create physical tags
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface, surface2], tag=1)

        gmsh.model.addPhysicalGroup(2, [surface3, surface4], tag=2)

        bndry1 = gmsh.model.getBoundary([(2, surface), (2, surface2)], oriented=False)
        [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry1]
        bndry2 = gmsh.model.getBoundary([(2, surface3), (2, surface4)], oriented=False)
        [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry2]

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "NodesList", [bottom1])

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", 0.5 * res)
        gmsh.model.mesh.field.setNumber(2, "LcMax", 2 * res)
        gmsh.model.mesh.field.setNumber(2, "DistMin", r / 2)
        gmsh.model.mesh.field.setNumber(2, "DistMax", r)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
        if quads:
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
            gmsh.option.setNumber("Mesh.RecombineAll", 2)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)

        # gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(str(filename))
    MPI.COMM_WORLD.Barrier()
    gmsh.finalize()


def sliding_wedges(
    filename: typing.Union[str, Path],
    quads: bool = False,
    res: float = 0.1,
    order: int = 1,
    angle=np.pi / 12,
):
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        bl = gmsh.model.occ.addPoint(0, 0, 0)
        br = gmsh.model.occ.addPoint(9, 0, 0)
        tl = gmsh.model.occ.addPoint(3, 3 + 9 * np.tan(angle), 0)
        tr = gmsh.model.occ.addPoint(6, 3 + 9 * np.tan(angle), 0)
        cl = gmsh.model.occ.addPoint(0, 2, 0)
        cr = gmsh.model.occ.addPoint(9, 2 + 9 * np.tan(angle), 0)
        ctl = gmsh.model.occ.addPoint(3, 2 + 3 * np.tan(angle), 0)
        ctr = gmsh.model.occ.addPoint(6, 2 + 6 * np.tan(angle), 0)
        cbl = gmsh.model.occ.addPoint(3 + 1.0 * res, 2 + (3 + 1.0 * res) * np.tan(angle), 0)
        cbr = gmsh.model.occ.addPoint(6 + 0.5 * res, 2 + (6 + 0.5 * res) * np.tan(angle), 0)

        lb1 = gmsh.model.occ.addLine(bl, br)
        lb2 = gmsh.model.occ.addLine(br, cr)
        lb3 = gmsh.model.occ.addLine(cr, cbr)
        lb4 = gmsh.model.occ.addLine(cbr, cbl)
        lb5 = gmsh.model.occ.addLine(cbl, cl)
        lb6 = gmsh.model.occ.addLine(cl, bl)

        curve1 = gmsh.model.occ.addCurveLoop([lb1, lb2, lb3, lb4, lb5, lb6])

        lt1 = gmsh.model.occ.addLine(ctl, ctr)
        lt2 = gmsh.model.occ.addLine(ctr, tr)
        lt3 = gmsh.model.occ.addLine(tr, tl)
        lt4 = gmsh.model.occ.addLine(tl, ctl)

        curve2 = gmsh.model.occ.addCurveLoop([lt1, lt2, lt3, lt4])

        surface1 = gmsh.model.occ.addPlaneSurface([curve1])
        gmsh.model.occ.synchronize()
        surface2 = gmsh.model.occ.addPlaneSurface([curve2])

        gmsh.model.occ.synchronize()

        gmsh.model.addPhysicalGroup(2, [surface2], tag=1)
        gmsh.model.addPhysicalGroup(2, [surface1], tag=2)

        bndry1 = gmsh.model.getBoundary([(2, surface1)], oriented=False)
        gmsh.model.addPhysicalGroup(1, [bndry1[0][1]])
        gmsh.model.addPhysicalGroup(1, [bndry1[1][1]])
        gmsh.model.addPhysicalGroup(1, [bndry1[2][1], bndry1[3][1], bndry1[4][1]])
        gmsh.model.addPhysicalGroup(1, [bndry1[5][1]])
        bndry2 = gmsh.model.getBoundary([(2, surface2)], oriented=False)
        [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry2]

        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "NodesList", [ctl, ctr])

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", 0.5 * res)
        gmsh.model.mesh.field.setNumber(2, "LcMax", res)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.3)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 1.0)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        if quads:
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
            gmsh.option.setNumber("Mesh.RecombineAll", 2)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)

        # gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(str(filename))
    MPI.COMM_WORLD.Barrier()
    gmsh.finalize()


def create_circle_circle_mesh(
    model, quads: bool = False, res: float = 0.1, order: int = 1, comm: MPI.Comm = MPI.COMM_WORLD, rank: int = 0
) -> typing.Optional[gmsh.model]:
    """Create two circular meshes, with radii 0.3 and 0.6 with centers (0.5,0.5) and (0.5, -0.5)"""
    center = [0.5, 0.5, 0]
    r = 0.3
    angle = np.pi / 4

    if comm.rank == 0:
        # Create circular mesh (divided into 4 segments)
        c = model.occ.addPoint(center[0], center[1], center[2])
        # Add 4 points on circle (clockwise, starting in top left)
        angles = [angle, -angle, np.pi + angle, np.pi - angle]
        c_points = [
            model.occ.addPoint(center[0] + r * np.cos(angle), center[1] + r * np.sin(angle), center[2])
            for angle in angles
        ]
        arcs = [model.occ.addCircleArc(c_points[i - 1], c, c_points[i]) for i in range(len(c_points))]
        curve = model.occ.addCurveLoop(arcs)
        model.occ.synchronize()
        surface = model.occ.addPlaneSurface([curve])
        # Create 2nd circular mesh (divided into 4 segments)
        center2 = [0.5, -0.5, 0]
        c2 = model.occ.addPoint(center2[0], center2[1], center2[2])
        # Add 4 points on circle (clockwise, starting in top left)
        c_points2 = [
            model.occ.addPoint(center2[0] + 2 * r * np.cos(angle), center2[1] + 2 * r * np.sin(angle), center2[2])
            for angle in angles
        ]
        arcs2 = [model.occ.addCircleArc(c_points2[i - 1], c2, c_points2[i]) for i in range(len(c_points2))]
        curve2 = model.occ.addCurveLoop(arcs2)
        model.occ.synchronize()
        surface2 = model.occ.addPlaneSurface([curve2])

        # Synchronize and create physical tags
        model.occ.addPoint(0.5, 0.2, 0, tag=17)
        model.occ.synchronize()
        model.addPhysicalGroup(2, [surface], tag=1)
        bndry = model.getBoundary([(2, surface)], oriented=False)
        [model.addPhysicalGroup(b[0], [b[1]]) for b in bndry]

        model.addPhysicalGroup(2, [surface2], tag=2)
        bndry2 = model.getBoundary([(2, surface2)], oriented=False)
        [model.addPhysicalGroup(b[0], [b[1]]) for b in bndry2]

        model.mesh.field.add("Distance", 1)
        model.mesh.field.setNumbers(1, "NodesList", [17])

        model.mesh.field.add("Threshold", 2)
        model.mesh.field.setNumber(2, "IField", 1)
        model.mesh.field.setNumber(2, "LcMin", res)
        model.mesh.field.setNumber(2, "LcMax", 3 * res)
        model.mesh.field.setNumber(2, "DistMin", 0.3)
        model.mesh.field.setNumber(2, "DistMax", 0.6)
        if quads:
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
            gmsh.option.setNumber("Mesh.RecombineAll", 2)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        model.mesh.field.setAsBackgroundMesh(2)
        model.mesh.generate(2)
        model.mesh.setOrder(order)

        return model
    else:
        return None


def create_gmsh_box_mesh_2D(
    model, quads: bool = False, res: float = 0.1, order: int = 1, comm: MPI.Comm = MPI.COMM_WORLD, rank: int = 0
) -> typing.Optional[gmsh.model]:
    """Create a Gmsh mode/mesh of two boxes, one slightly skewed.

    Args:
        quads:
        res:
        order:
        comm: Communicator
        rank: MPI rank to create model on

    Return:
        A Gmsh model on ``rank``, ``None`` on other ranks.
    """
    length = 0.5
    height = 0.5
    disp = -0.6
    delta = 0.1

    if comm.rank == rank:
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", res)

        # Create box
        p0 = model.occ.addPoint(-delta, 0, 0)
        p1 = model.occ.addPoint(length - delta, delta, 0)
        p2 = model.occ.addPoint(length - delta, height + delta, 0)
        p3 = model.occ.addPoint(-delta, height, 0)
        ps = [p0, p1, p2, p3]
        lines = [model.occ.addLine(ps[i - 1], ps[i]) for i in range(len(ps))]
        curve = model.occ.addCurveLoop(lines)
        surface = model.occ.addPlaneSurface([curve])

        # Create box
        p4 = model.occ.addPoint(0, 0 + disp, 0)
        p5 = model.occ.addPoint(length, 0 + disp, 0)
        p6 = model.occ.addPoint(length, height + disp, 0)
        p7 = model.occ.addPoint(0, height + disp, 0)
        ps2 = [p4, p5, p6, p7]
        lines2 = [model.occ.addLine(ps2[i - 1], ps2[i]) for i in range(len(ps2))]
        curve2 = model.occ.addCurveLoop(lines2)
        surface2 = model.occ.addPlaneSurface([curve2])

        model.occ.synchronize()

        # Set mesh sizes on the points from the surface we are extruding
        top_nodes = model.getBoundary([(2, surface)], recursive=True, oriented=False)
        model.occ.mesh.setSize(top_nodes, res)
        bottom_nodes = model.getBoundary([(2, surface2)], recursive=True, oriented=False)
        model.occ.mesh.setSize(bottom_nodes, 2 * res)

        # Synchronize and create physical tags
        model.occ.synchronize()
        model.addPhysicalGroup(2, [surface], tag=1)
        bndry = model.getBoundary([(2, surface)], oriented=False)
        [model.addPhysicalGroup(b[0], [b[1]]) for b in bndry]

        model.addPhysicalGroup(2, [surface2], tag=2)
        bndry2 = model.getBoundary([(2, surface2)], oriented=False)
        [model.addPhysicalGroup(b[0], [b[1]]) for b in bndry2]
        if quads:
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
            gmsh.option.setNumber("Mesh.RecombineAll", 2)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        model.mesh.generate(2)
        model.mesh.setOrder(order)
    else:
        model = None

    return model


def create_box_mesh_3D(
    model,
    simplex: bool = True,
    order: int = 1,
    res: float = 0.1,
    gap: float = 0.1,
    width: float = 0.5,
    offset: float = 0.2,
    comm: MPI.Comm = MPI.COMM_WORLD,
    rank: int = 0,
):
    """Create two boxes lying directly over each other with a gap in between"""
    length = 0.5
    height = 0.5

    disp = -width - gap
    if comm.rank == rank:
        # Create box
        if simplex:
            model.occ.add_box(0, 0 + offset, 0, length, height, width)
            model.occ.add_box(0, 0, disp, length, height, width)
            model.occ.synchronize()
        else:
            square1 = model.occ.add_rectangle(0, 0 + offset, 0, length, height)
            square2 = model.occ.add_rectangle(0, 0, disp, length, height)
            model.occ.extrude([(2, square1)], 0, 0, width, numElements=[20], recombine=True)
            model.occ.extrude([(2, square2)], 0, 0, width, numElements=[15], recombine=True)
            model.occ.synchronize()
        volumes = model.getEntities(3)

        model.mesh.field.add("Box", 1)
        model.mesh.field.setNumber(1, "VIn", res / 5.0)
        model.mesh.field.setNumber(1, "VOut", res)
        model.mesh.field.setNumber(1, "XMin", 0)
        model.mesh.field.setNumber(1, "XMax", length)
        model.mesh.field.setNumber(1, "YMin", 0)
        model.mesh.field.setNumber(1, "YMax", height)
        model.mesh.field.setNumber(1, "ZMin", 0)
        model.mesh.field.setNumber(1, "ZMax", width)

        model.mesh.field.setAsBackgroundMesh(1)

        # Synchronize and create physical tags
        model.occ.synchronize()
        model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], tag=1)
        bndry = model.getBoundary([(3, volumes[0][1])], oriented=False)
        [model.addPhysicalGroup(b[0], [b[1]]) for b in bndry]

        model.addPhysicalGroup(3, [volumes[1][1]], tag=2)
        bndry2 = model.getBoundary([(3, volumes[1][1])], oriented=False)
        [model.addPhysicalGroup(b[0], [b[1]]) for b in bndry2]
        if not simplex:
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            gmsh.option.setNumber("Mesh.RecombineAll", 2)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        model.mesh.generate(3)
        model.mesh.setOrder(order)
        # gmsh.option.setNumber("Mesh.SaveAll", 1)

        return model
    else:
        return None


def create_sphere_plane_mesh(
    model,
    order: int = 1,
    res=0.05,
    r=0.25,
    height=0.25,
    length=1.0,
    width=1.0,
    gap=0.0,
    comm: MPI.Comm = MPI.COMM_WORLD,
    rank: int = 0,
):
    """Create a 3D sphere with center (0,0,0) an radius r
    with a box at [-length/2, length/2] x [ -width/2, width/2] x [-gap-height-r, -gap-r]
    """
    center = [0.0, 0.0, 0.0]
    angle = 0
    lc_min = res
    lc_max = 2 * res
    if comm.rank == rank:
        # Create sphere composed of of two volumes
        sphere_bottom = model.occ.addSphere(center[0], center[1], center[2], r, angle1=-np.pi / 2, angle2=-angle)
        p0 = model.occ.addPoint(center[0], center[1], center[2] - r)
        sphere_top = model.occ.addSphere(center[0], center[1], center[2], r, angle1=-angle, angle2=np.pi / 2)
        out_vol_tags, _ = model.occ.fragment([(3, sphere_bottom)], [(3, sphere_top)])

        # Add bottom box
        box = model.occ.add_box(-length / 2, -width / 2, -height - r - gap, length, width, height)

        # Synchronize and create physical tags
        model.occ.synchronize()

        sphere_boundary = model.getBoundary(out_vol_tags, oriented=False)
        for boundary_tag in sphere_boundary:
            model.addPhysicalGroup(boundary_tag[0], boundary_tag[1:2])
        box_boundary = gmsh.model.getBoundary([(3, box)], oriented=False)
        for boundary_tag in box_boundary:
            model.addPhysicalGroup(boundary_tag[0], boundary_tag[1:2])

        p_v = [v_tag[1] for v_tag in out_vol_tags]
        model.addPhysicalGroup(3, p_v, tag=1)
        model.addPhysicalGroup(3, [box], tag=2)

        model.occ.synchronize()
        model.mesh.field.add("Distance", 1)
        model.mesh.field.setNumbers(1, "NodesList", [p0])

        model.mesh.field.add("Threshold", 2)
        model.mesh.field.setNumber(2, "IField", 1)
        model.mesh.field.setNumber(2, "LcMin", lc_min)
        model.mesh.field.setNumber(2, "LcMax", lc_max)
        model.mesh.field.setNumber(2, "DistMin", 0.5 * r)
        model.mesh.field.setNumber(2, "DistMax", r)
        model.mesh.field.setAsBackgroundMesh(2)

        model.mesh.generate(3)
        model.mesh.setOrder(order)

        return model
    else:
        return None


def create_sphere_sphere_mesh(filename: typing.Union[str, Path], order: int = 1):
    """Create a 3D mesh consisting of two spheres with radii 0.3 and 0.6 and
    centers (0.5,0.5,0.5) and (0.5,0.5,-0.5)
    """
    center = [0.5, 0.5, 0.5]
    r = 0.3
    angle = np.pi / 8
    lc_min = 0.05 * r
    lc_max = 0.2 * r
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        # Create sphere 1 composed of of two volumes
        sphere_bottom = gmsh.model.occ.addSphere(center[0], center[1], center[2], r, angle1=-np.pi / 2, angle2=-angle)
        p0 = gmsh.model.occ.addPoint(center[0], center[1], center[2] - r)
        sphere_top = gmsh.model.occ.addSphere(center[0], center[1], center[2], r, angle1=-angle, angle2=np.pi / 2)
        out_vol_tags, _ = gmsh.model.occ.fragment([(3, sphere_bottom)], [(3, sphere_top)])

        # Create sphere 2 composed of of two volumes
        sphere_bottom2 = gmsh.model.occ.addSphere(
            center[0], center[1], -center[2], 2 * r, angle1=-np.pi / 2, angle2=-angle
        )
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
        gmsh.model.mesh.field.setNumber(2, "LcMin", lc_min)
        gmsh.model.mesh.field.setNumber(2, "LcMax", lc_max)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5 * r)
        gmsh.model.mesh.field.setNumber(2, "DistMax", r)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(order)

        # gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(str(filename))
    MPI.COMM_WORLD.Barrier()
    gmsh.finalize()


def create_cylinder_cylinder_mesh(
    model, order: int = 1, res=0.25, simplex: bool = False, comm: MPI.Comm = MPI.COMM_WORLD, rank: int = 0
):
    """Generate a mesh with 2nd-order hexahedral cells using gmsh."""
    if comm.rank == rank:
        # Recombine tetrahedrons to hexahedrons
        if not simplex:
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
            gmsh.option.setNumber("Mesh.RecombineAll", 2)
        # gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 0.1)
        center1 = (0, 0, 0.5)
        r1 = 0.8
        l1 = 1
        Nl1 = int(1 / res)
        circle = model.occ.addDisk(*center1, r1, r1)
        model.occ.rotate([(2, circle)], 0, 0, 0, 1, 0, 0, np.pi / 2)
        model.occ.extrude([(2, circle)], 0, l1, 0, numElements=[Nl1], recombine=not simplex)

        center2 = (2, 0, -0.5)
        r2 = 0.5
        l2 = 1
        Nl2 = int(1 / res)
        circle2 = model.occ.addDisk(*center2, r2, r2)
        model.occ.extrude([(2, circle2)], 0, 0, l2, numElements=[Nl2], recombine=not simplex)

        model.mesh.field.add("Box", 1)
        model.mesh.field.setNumber(1, "VIn", res)
        model.mesh.field.setNumber(1, "VOut", res)
        model.mesh.field.setNumber(1, "XMin", center1[0] - l1)
        model.mesh.field.setNumber(1, "XMax", center1[0] + l1)
        model.mesh.field.setNumber(1, "YMin", center1[1] - 2 * r1)
        model.mesh.field.setNumber(1, "YMax", center1[1] + 2 * r1)
        model.mesh.field.setNumber(1, "ZMin", center1[2] - 2 * r1)
        model.mesh.field.setNumber(1, "ZMax", center1[2] + 2 * r1)

        model.mesh.field.setAsBackgroundMesh(1)
        model.occ.synchronize()
        for i, entity in enumerate(model.getEntities(3)):
            model.addPhysicalGroup(3, [entity[1]], tag=i)

        model.setPhysicalName(3, 1, "Mesh volume")
        model.mesh.generate(3)
        model.mesh.setOrder(order)

        return model
    else:
        return None


def create_2d_rectangle_split(filename: typing.Union[str, Path], quads: bool = False, res=0.1, order: int = 1, gap=0.2):
    """Create rectangle split into two domains"""
    length = 0.5
    height = 0.5

    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", res)

        # Create box
        p0 = gmsh.model.occ.addPoint(0, 0, 0)
        p1 = gmsh.model.occ.addPoint(length, 0, 0)
        p2 = gmsh.model.occ.addPoint(length, height, 0)
        p3 = gmsh.model.occ.addPoint(0, height, 0)
        ps = [p0, p1, p2, p3]
        lines = [gmsh.model.occ.addLine(ps[i - 1], ps[i]) for i in range(len(ps))]
        curve = gmsh.model.occ.addCurveLoop(lines)
        surface = gmsh.model.occ.addPlaneSurface([curve])

        # Create box
        p4 = gmsh.model.occ.addPoint(length + gap, 0, 0)
        p5 = gmsh.model.occ.addPoint(2 * length + gap, 0, 0)
        p6 = gmsh.model.occ.addPoint(2 * length + gap, height, 0)
        p7 = gmsh.model.occ.addPoint(length + gap, height, 0)
        ps2 = [p4, p5, p6, p7]
        lines2 = [gmsh.model.occ.addLine(ps2[i - 1], ps2[i]) for i in range(len(ps2))]
        curve2 = gmsh.model.occ.addCurveLoop(lines2)
        surface2 = gmsh.model.occ.addPlaneSurface([curve2])

        p8 = gmsh.model.occ.addPoint(2 * length + gap - res / 10, 0.5 * height, 0)
        p9 = gmsh.model.occ.addPoint(2 * length + gap - res / 5, 0.5 * height, 0)

        gmsh.model.occ.synchronize()
        # Set mesh sizes on the points from the surface we are extruding
        top_nodes = gmsh.model.getBoundary([(2, surface)], recursive=True, oriented=False)
        gmsh.model.occ.mesh.setSize(top_nodes, 1.2 * res)
        bottom_nodes = gmsh.model.getBoundary([(2, surface2)], recursive=True, oriented=False)
        gmsh.model.occ.mesh.setSize(bottom_nodes, res)
        # Synchronize and create physical tags
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface], tag=1)
        bndry = gmsh.model.getBoundary([(2, surface)], oriented=False)
        [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry]

        gmsh.model.addPhysicalGroup(2, [surface2], tag=2)
        bndry2 = gmsh.model.getBoundary([(2, surface2)], oriented=False)
        [gmsh.model.addPhysicalGroup(b[0], [b[1]]) for b in bndry2]
        gmsh.model.mesh.embed(0, [p8, p9], 2, 2)
        if quads:
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
            gmsh.option.setNumber("Mesh.RecombineAll", 2)
            gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)

        # gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(str(filename))
    MPI.COMM_WORLD.Barrier()

    gmsh.finalize()


def create_halfsphere_box_mesh(
    filename: typing.Union[str, Path],
    order: int = 1,
    res=0.05,
    r=0.25,
    height=0.25,
    length=1.0,
    width=1.0,
    gap=0.0,
):
    """Create a 3D half-sphere with center (0,0,0), radius r and z<=0.0
    with a box at [-length/2, length/2] x [ -width/2, width/2] x [-gap-height-r, -gap-r].
    """
    center = [0.0, 0.0, 0.0]
    angle = 0
    lc_min = res
    lc_max = 2 * res
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        # Create sphere composed of of two volumes
        sphere_bottom = gmsh.model.occ.addSphere(center[0], center[1], center[2], r, angle1=-np.pi / 2, angle2=-angle)
        p0 = gmsh.model.occ.addPoint(center[0], center[1], center[2] - r)

        # Add bottom box
        box = gmsh.model.occ.add_box(-length / 2, -width / 2, -height - r - gap, length, width, height)

        # Synchronize and create physical tags
        gmsh.model.occ.synchronize()

        sphere_boundary = gmsh.model.getBoundary([(3, sphere_bottom)], oriented=False)
        for boundary_tag in sphere_boundary:
            gmsh.model.addPhysicalGroup(boundary_tag[0], boundary_tag[1:2])
        box_boundary = gmsh.model.getBoundary([(3, box)], oriented=False)
        for boundary_tag in box_boundary:
            gmsh.model.addPhysicalGroup(boundary_tag[0], boundary_tag[1:2])

        gmsh.model.addPhysicalGroup(3, [sphere_bottom], tag=1)
        gmsh.model.addPhysicalGroup(3, [box], tag=2)

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "NodesList", [p0])

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", lc_min)
        gmsh.model.mesh.field.setNumber(2, "LcMax", lc_max)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5 * r)
        gmsh.model.mesh.field.setNumber(2, "DistMax", r)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(order)

        # gmsh.option.setNumber("Mesh.SaveAll", 1)
        gmsh.write(str(filename))
    MPI.COMM_WORLD.Barrier()
    gmsh.finalize()
