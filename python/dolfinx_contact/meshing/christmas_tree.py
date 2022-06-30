# Copyright (C) 2022 Chris Richardson and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import numpy as np
import gmsh
from mpi4py import MPI

__all__ = ["create_christmas_tree_mesh"]


def create_closed_curve(model, curve, z=0.0):
    xv, yv = curve
    ps = []
    for i, q in enumerate(zip(xv, yv)):
        point = model.occ.addPoint(q[0], q[1], z)
        ps.append(point)
    lines = [model.occ.addLine(ps[i - 1], ps[i]) for i in range(len(ps))]
    gmsh_curve = model.occ.addCurveLoop(lines)
    return ps, lines, gmsh_curve


def jagged_curve(npoints, t, r0, r1, xmax):
    xvals = []
    yvals = []

    dx = xmax / npoints

    xvals = [0.0]
    yvals = [0.0]

    while (xvals[-1] < xmax):
        ylast = yvals[-1]
        xlast = xvals[-1]
        xp = dx
        ds = 4 * dx
        while (ds > 2 * dx):
            x = xlast + xp
            y = r0(x) * np.arctan(t * np.sin(np.pi * x)
                                  / (1 - t * np.cos(np.pi * x))) / t + r1(x) * x

            ds = np.sqrt((y - ylast)**2 + (x - xlast)**2)
            xp *= 0.8

        xvals += [x]
        yvals += [y]

    return np.array(xvals), np.array(yvals)


def create_christmas_tree_mesh_3D(filename: str, res=0.2, split=1, n1=101, n2=51):
    offset = [0.0, 0.0, 1.0]
    # TODO: Some of the curve parameters should probably be input
    nc = n1
    x, y = jagged_curve(nc, -0.95, lambda x: 0.8 * x / 5.0, lambda x: 0.6, 4.0)
    nc = n2
    # Fill in bottom
    yb = np.linspace(y[-1], -y[-1], nc)
    xb = np.ones_like(yb) * x[-1]

    # Flip and concatenate
    xv = np.concatenate((x[1:], xb[1:-1], np.flip(x[1:])))
    yv = np.concatenate((y[1:], yb[1:-1], np.flip(-y[1:])))

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if MPI.COMM_WORLD.rank == 0:
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", res)
        model = gmsh.model()
        model.add("xmas")
        model.setCurrent("xmas")
        ps1, lines1, curve1 = create_closed_curve(model, (xv, yv))
        ps2, lines2, curve2 = create_closed_curve(model, (xv + offset[0], yv + offset[1]), z=offset[2])
        lines_z1 = []
        for i in range(len(ps1)):
            lines_z1.append(model.occ.addLine(ps1[i], ps2[i]))
        surfaces1 = []
        for i in range(len(ps1)):
            curve = model.occ.addCurveLoop([lines1[i], lines_z1[i], -lines2[i], -lines_z1[i - 1]])
            surfaces1.append(model.occ.addPlaneSurface([curve]))
        surface1 = model.occ.addPlaneSurface([curve1])
        surface2 = model.occ.addPlaneSurface([curve2])
        sloop1 = model.occ.addSurfaceLoop([surface1] + surfaces1 + [surface2])
        volume1 = model.occ.addVolume([sloop1])
        model.occ.synchronize()
        model.addPhysicalGroup(3, [volume1], tag=1)
        tree_inner = surfaces1[len(x) + len(xb) - 2:] + surfaces1[:len(x) - 1]
        tree_bottom = surfaces1[len(x) - 1:len(x) + len(xb) - 2]
        num_surfaces = len(tree_inner) // split
        for i in range(split - 1):
            model.addPhysicalGroup(2, tree_inner[i * num_surfaces:(i + 1) * num_surfaces], tag=5 + i + 1)
        model.addPhysicalGroup(2, tree_inner[(split - 1) * num_surfaces:], tag=5 + split)
        model.addPhysicalGroup(2, tree_bottom, tag=3)
        model.addPhysicalGroup(2, [surface1, surface2], tag=5)

        dx = 0.01
        dy = 0.1
        x += dx
        y += dy

        xf = np.linspace(-1, x[-1], nc)
        yf = np.ones_like(xf) * y[-1] * 1.5

        yg = np.linspace(yf[0], -yf[0], nc)[1:-1]
        xg = np.ones_like(yg) * -1

        yh = np.linspace(y[-1], y[-1] * 1.5, 8)[1:-1]
        xh = np.ones_like(yh) * x[-1]

        xv = np.concatenate((xh, np.flip(x), x, xh, np.flip(xf), xg, xf))
        yv = np.concatenate((-np.flip(yh), np.flip(-y), y, yh, yf, yg, -yf))

        ps3, lines3, curve3 = create_closed_curve(model, (xv, yv))
        ps4, lines4, curve4 = create_closed_curve(model, (xv + offset[0], yv + offset[1]), z=offset[2])
        lines_z2 = []
        for i in range(len(ps3)):
            lines_z2.append(model.occ.addLine(ps3[i], ps4[i]))
        surfaces2 = []
        for i in range(len(ps3)):
            curve = model.occ.addCurveLoop([lines3[i], lines_z2[i], -lines4[i], -lines_z2[i - 1]])
            surfaces2.append(model.occ.addPlaneSurface([curve]))
        surface3 = model.occ.addPlaneSurface([curve3])
        surface4 = model.occ.addPlaneSurface([curve4])
        sloop2 = model.occ.addSurfaceLoop([surface3] + surfaces2 + [surface4])
        volume2 = model.occ.addVolume([sloop2])
        model.occ.synchronize()
        model.addPhysicalGroup(3, [volume2], tag=2)
        tree_outer = surfaces2[len(xh) + 1:len(xh) + 2 * len(x)]
        box = surfaces2[:len(xh)] + surfaces2[1 + len(xh) + 2 * len(x):]
        model.addPhysicalGroup(2, box, tag=4)
        for i in range(split):
            if i == 0:
                start = 0
            else:
                start = i * num_surfaces + 1
            if i == split - 1:
                end = len(tree_outer)
            else:
                end = (i + 1) * num_surfaces + 1
            model.addPhysicalGroup(2, tree_outer[start:end], tag=5 + split + i + 1)
        model.mesh.field.setAsBackgroundMesh(2)
        model.mesh.generate(3)
        model.mesh.optimize("Netgen")
        gmsh.write(f"{filename}.msh")

    MPI.COMM_WORLD.Barrier()
    gmsh.finalize()


def create_christmas_tree_mesh(filename: str, res=0.2, split=1):
    # TODO: Some of the curve parameters should probably be input
    nc = 101
    x, y = jagged_curve(nc, -0.95, lambda x: 0.8 * x / 5.0, lambda x: 0.6, 8.0)
    nc = 51
    # Fill in bottom
    yb = np.linspace(y[-1], -y[-1], nc)
    xb = np.ones_like(yb) * x[-1]

    # Flip and concatenate
    xv = np.concatenate((x[1:], xb[1:-1], np.flip(x[1:])))
    yv = np.concatenate((y[1:], yb[1:-1], np.flip(-y[1:])))

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if MPI.COMM_WORLD.rank == 0:
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", res)
        model = gmsh.model()
        model.add("xmas")
        model.setCurrent("xmas")
        _, lines1, curve1 = create_closed_curve(model, (xv, yv))
        surface = model.occ.addPlaneSurface([curve1])
        model.occ.synchronize()
        model.addPhysicalGroup(2, [surface], tag=1)
        tree_inner = lines1[len(x) + len(xb) - 2:] + lines1[:len(x) - 1]
        tree_bottom = lines1[len(x) - 1:len(x) + len(xb) - 2]
        num_lines = len(tree_inner) // split
        for i in range(split - 1):
            model.addPhysicalGroup(1, tree_inner[i * num_lines:(i + 1) * num_lines], tag=4 + i + 1)
        model.addPhysicalGroup(1, tree_inner[(split - 1) * num_lines:], tag=4 + split)
        model.addPhysicalGroup(1, tree_bottom, tag=3)

        dx = 0.01
        dy = 0.1
        x += dx
        y += dy

        xf = np.linspace(-1, x[-1], nc)
        yf = np.ones_like(xf) * y[-1] * 1.2

        yg = np.linspace(yf[0], -yf[0], nc)[1:-1]
        xg = np.ones_like(yg) * -1

        yh = np.linspace(y[-1], y[-1] * 1.2, 8)[1:-1]
        xh = np.ones_like(yh) * x[-1]

        xv = np.concatenate((xh, np.flip(x), x, xh, np.flip(xf), xg, xf))
        yv = np.concatenate((-np.flip(yh), np.flip(-y), y, yh, yf, yg, -yf))

        _, lines2, curve2 = create_closed_curve(model, (xv, yv))
        surface2 = model.occ.addPlaneSurface([curve1, curve2])
        model.occ.synchronize()
        tree_outer = lines2[len(xh) + 1:len(xh) + 2 * len(x)]
        box = lines2[:len(xh) + 1] + lines2[len(xh) + 2 * len(x):]

        model.addPhysicalGroup(2, [surface2], tag=2)
        model.addPhysicalGroup(1, box, tag=4)
        for i in range(split):
            if i == 0:
                start = 0
            else:
                start = i * num_lines + 1
            if i == split - 1:
                end = len(tree_outer)
            else:
                end = (i + 1) * num_lines + 1
            model.addPhysicalGroup(1, tree_outer[start:end], tag=4 + split + i + 1)
        model.mesh.field.setAsBackgroundMesh(2)
        model.mesh.generate(2)
        gmsh.write(f"{filename}.msh")

    MPI.COMM_WORLD.Barrier()
    gmsh.finalize()
