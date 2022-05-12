# Copyright (C) 2022 Chris Richardson and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import numpy as np
import gmsh
from mpi4py import MPI


__all__ = ["create_christmas_tree_mesh"]


def create_closed_curve(curve):
    xv, yv = curve
    ps = []
    for i, q in enumerate(zip(xv, yv)):
        point = gmsh.model.occ.addPoint(q[0], q[1], 0.0)
        ps.append(point)
    lines = [gmsh.model.occ.addLine(ps[i - 1], ps[i]) for i in range(len(ps))]
    gmsh_curve = gmsh.model.occ.addCurveLoop(lines)
    return lines, gmsh_curve


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
        while (ds > 3 * dx):
            x = xlast + xp
            y = r0(x) * np.arctan(t * np.sin(np.pi * x)
                                  / (1 - t * np.cos(np.pi * x))) / t + r1(x) * x

            ds = np.sqrt((y - ylast)**2 + (x - xlast)**2)
            print('ds=', ds / dx)
            xp *= 0.8

        xvals += [x]
        yvals += [y]

    return np.array(xvals), np.array(yvals)


def create_christmas_tree_mesh(filename: str, quads: bool = False, res=0.2):
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
    if MPI.COMM_WORLD.rank == 0:
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", res)

        lines1, curve1 = create_closed_curve((xv, yv))
        surface = gmsh.model.occ.addPlaneSurface([curve1])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface])
        gmsh.model.addPhysicalGroup(1, lines1[:len(x) - 1] + lines1[len(x) + len(xb) - 2:])
        gmsh.model.addPhysicalGroup(1, lines1[len(x) - 1:len(x) + len(xb) - 2])

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

        lines2, curve2 = create_closed_curve((xv, yv))
        surface2 = gmsh.model.occ.addPlaneSurface([curve1, curve2])
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [surface2])
        gmsh.model.addPhysicalGroup(1, lines2[:len(xh) + 1] + lines2[len(xh) + 2 * len(x):])
        gmsh.model.addPhysicalGroup(1, lines2[len(xh) + 1:len(xh) + 2 * len(x) + 1])

        if quads:
            gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
            gmsh.option.setNumber("Mesh.RecombineAll", 2)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
        gmsh.model.mesh.generate(2)
        gmsh.write(filename)
    MPI.COMM_WORLD.Barrier()
    gmsh.finalize()
