# Copyright (C) 2022 Chris Richardson and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from scipy.misc import face
import numpy as np
import gmsh
from mpi4py import MPI
from dolfinx.graph import create_adjacencylist
from dolfinx.io import (XDMFFile, cell_perm_gmsh, distribute_entity_data,
                        extract_gmsh_geometry,
                        extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh)
from dolfinx.mesh import CellType, create_mesh, meshtags_from_entities

__all__ = ["create_christmas_tree_mesh"]


def create_closed_curve(model, curve):
    xv, yv = curve
    ps = []
    for i, q in enumerate(zip(xv, yv)):
        point = model.occ.addPoint(q[0], q[1], 0.0)
        ps.append(point)
    lines = [model.occ.addLine(ps[i - 1], ps[i]) for i in range(len(ps))]
    gmsh_curve = model.occ.addCurveLoop(lines)
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
        while (ds > 2 * dx):
            x = xlast + xp
            y = r0(x) * np.arctan(t * np.sin(np.pi * x)
                                  / (1 - t * np.cos(np.pi * x))) / t + r1(x) * x

            ds = np.sqrt((y - ylast)**2 + (x - xlast)**2)
            xp *= 0.8

        xvals += [x]
        yvals += [y]

    return np.array(xvals), np.array(yvals)


def create_christmas_tree_mesh(filename: str, quads: bool = False, res=0.2, split=1, padding=1):
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
        model = gmsh.model()
        model.add("xmas")
        model.setCurrent("xmas")
        lines1, curve1 = create_closed_curve(model, (xv, yv))
        surface = model.occ.addPlaneSurface([curve1])
        model.occ.synchronize()
        model.addPhysicalGroup(2, [surface], tag=1)
        tree_inner = lines1[len(x) + len(xb) - 2:] + lines1[:len(x) - 1]
        tree_bottom = lines1[len(x) - 1:len(x) + len(xb) - 2]
        num_lines = len(tree_inner) // split
        for i in range(split - 1):
            model.addPhysicalGroup(1, tree_inner[i * num_lines:(i + 1) * num_lines], tag=4 + i + 1)
            start = max(0, i * (num_lines) - padding)
            end = min((i + 1) * num_lines + padding, len(tree_inner))
            model.addPhysicalGroup(1, tree_inner[start:end], tag=4 + 2 * split + i + 1)
        model.addPhysicalGroup(1, tree_inner[(split - 1) * num_lines:], tag=4 + split)
        model.addPhysicalGroup(1, tree_inner[max(0, (split - 1) * num_lines - padding):], tag=4 + 3 * split)
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

        lines2, curve2 = create_closed_curve(model, (xv, yv))
        surface2 = model.occ.addPlaneSurface([curve1, curve2])
        model.occ.synchronize()
        tree_outer = lines2[len(xh) + 1:len(xh) + 2 * len(x)]
        box = lines2[:len(xh) + 1] + lines2[len(xh) + 2 * len(x):]
        print(tree_outer)
        print(box)
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
            start = max(0, start - padding)
            end = min(end + padding, len(tree_outer))
            model.addPhysicalGroup(1, tree_outer[start:end], tag=4 + 3 * split + i + 1)
            print(tree_outer[start:end])

        model.mesh.field.setAsBackgroundMesh(2)
        model.mesh.generate(2)
        # Sort mesh nodes according to their index in gmsh
        x = extract_gmsh_geometry(model, model_name="xmas")

        # Broadcast cell type data and geometric dimension
        gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("triangle", 1), root=0)

        # Get mesh data for dim (0, tdim) for all physical entities
        topologies = extract_gmsh_topology_and_markers(model, "xmas")
        cells = topologies[gmsh_cell_id]["topology"].astype(np.int64)
        cell_values = topologies[gmsh_cell_id]["cell_data"].astype(np.int32)
        num_nodes = MPI.COMM_WORLD.bcast(cells.shape[1], root=0)

        gmsh_facet_id = model.mesh.getElementType("line", 1)
        marked_facets = topologies[gmsh_facet_id]["topology"].astype(np.int64)
        facet_values = topologies[gmsh_facet_id]["cell_data"].astype(np.int32)
        # gmsh.write(filename)
    else:
        gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
        num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
        cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
        cell_values = np.empty((0,), dtype=np.int32)
        marked_facets, facet_values = np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int32)

    msh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh_from_gmsh(gmsh_cell_id, 3))
    msh.name = "xmas"

    msh.topology.create_connectivity(1, 0)

    padded_mts = []
    for i in range(2 * split):
        entities, values = distribute_entity_data(
            msh, 1, marked_facets[facet_values == 4 + 2 * split + i + 1], facet_values[facet_values == 4 + 2 * split + i + 1])
        mt = meshtags_from_entities(msh, 1, create_adjacencylist(entities), np.int32(values))
        mt.name = f"padded_{i}"
        padded_mts.append(mt)

    entities, values = distribute_entity_data(
        msh, 1, marked_facets[facet_values <= 4 + 2 * split], facet_values[facet_values <= 4 + 2 * split])

    bc_mt = meshtags_from_entities(msh, 1, create_adjacencylist(entities), np.int32(values))
    bc_mt.name = "disjoint"

    entities, values = distribute_entity_data(msh, 2, cells, cell_values)
    domain_mt = meshtags_from_entities(msh, 2, create_adjacencylist(entities), np.int32(values))
    domain_mt.name = "domain_marker"
    with XDMFFile(MPI.COMM_WORLD, f"{filename}.xdmf", "w") as file:
        file.write_mesh(msh)
        msh.topology.create_connectivity(1, 2)
        file.write_meshtags(bc_mt)
        file.write_meshtags(domain_mt)
        for mt in padded_mts:
            file.write_meshtags(mt)

    # for i, mt2 in enumerate(padded_mts):
    #     with XDMFFile(MPI.COMM_WORLD, f"test_mt_{i}.xdmf", "w") as file:
    #         file.write_mesh(msh)
    #         file.write_meshtags(mt2)
    # with XDMFFile(MPI.COMM_WORLD, "test_mt.xdmf", "r") as xdmf:
    #     mesh = xdmf.read_mesh(name="xmas")
    #     tdim = mesh.topology.dim
    #     mesh.topology.create_connectivity(tdim - 1, 0)
    #     mesh.topology.create_connectivity(tdim - 1, tdim)
    #     mt = xdmf.read_meshtags(mesh, name="disjoint")
        # mt2 = xdmf.read_meshtags(mesh, name="extra")
        # facets1 = mt.indices[mt.values == 5]
        # facets2 = mt2.indices[mt2.values == 12]
        # print(facets1, facets2)
    MPI.COMM_WORLD.Barrier()
    gmsh.finalize()
