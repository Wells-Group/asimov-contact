# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

import numpy as np
import gmsh
from mpi4py import MPI

from dolfinx.graph import create_adjacencylist
from dolfinx.io.gmshio import (cell_perm_array,
                               extract_geometry,
                               extract_topology_and_markers, ufl_mesh)
from dolfinx.io import XDMFFile, distribute_entity_data
from dolfinx.mesh import CellType, create_mesh, meshtags_from_entities


def vertical_line(t, x0, x1):
    points = []
    for tt in t:
        points.append([x0[0], x0[1] + tt * (x1[1] - x0[1])])
    return points


def horizontal_line(t, x0, x1):
    points = []
    for tt in t:
        points.append([x0[0] + tt * (x1[0] - x0[0]), x0[1] + tt * (x1[1] - x0[1])])
    return points


def horizontal_sine(t, x0, x1):
    points = []
    for tt in t:
        points.append([x0[0] + tt * (x1[0] - x0[0]), x0[1] + tt * (x1[1] - x0[1]) + 0.1 * np.sin(8 * np.pi * tt)])
    return points


def get_surface_points(domain, points, line_pts, split=1):
    pts = [points[node] for node in domain]
    i0 = np.argwhere(np.array(domain, dtype=np.int32) == 4)[0, 0]
    i1 = np.argwhere(np.array(domain, dtype=np.int32) == 5)[0, 0]
    num_pts = len(pts)
    if i0 == i1 - 1:
        if i0 == 0:
            pts = np.vstack([line_pts[:], pts[i1 + 1:]])
        elif i1 == num_pts - 1:
            pts = np.vstack([line_pts[:], pts[:i0]])
        else:
            pts = np.vstack([line_pts[:], pts[i1 + 1:], pts[:i0]])
    elif i1 == i0 - 1:
        if i1 == 0:
            pts = np.vstack([list(reversed(line_pts))[:], pts[i0 + 1:]])
        elif i0 == num_pts - 1:
            pts = np.vstack([list(reversed(line_pts))[:], pts[:i1]])
        else:
            pts = np.vstack([list(reversed(line_pts))[:], pts[i0 + 1:], pts[:i1]])
    elif i0 == 0 and i1 == num_pts - 1:
        pts = np.vstack([list(reversed(line_pts)), pts[1:-1]])
    elif i1 == 0 and num_pts - 1:
        pts = np.vstack([line_pts, pts[1:-1]])
    else:
        raise RuntimeError("Invalid domains")

    return pts


def retrieve_mesh_data(model, name, gmsh_cell_id, gmsh_facet_id):
    x = extract_geometry(model, name=name)
    topologies = extract_topology_and_markers(model, name)
    cells = topologies[gmsh_cell_id]["topology"]
    cell_data = topologies[gmsh_cell_id]["cell_data"]
    marked_facets = topologies[gmsh_facet_id]["topology"].astype(np.int64)
    facet_values = topologies[gmsh_facet_id]["cell_data"].astype(np.int32)
    return x, cells, cell_data, marked_facets, facet_values


def create_dolfinx_mesh(filename, x, cells, cell_data, gmsh_cell_id, marked_facets, facet_values, tdim):
    msh = create_mesh(MPI.COMM_WORLD, cells, x, ufl_mesh(gmsh_cell_id, 3))
    msh.name = "Grid"
    entities, values = distribute_entity_data(msh, tdim - 1, marked_facets, facet_values)
    msh.topology.create_connectivity(tdim - 1, 0)
    mt = meshtags_from_entities(msh, tdim - 1, create_adjacencylist(entities), values)
    mt.name = "contact_facets"
    msh.topology.create_connectivity(tdim, 0)
    entities, values = distribute_entity_data(msh, tdim, cells.astype(np.int64), cell_data.astype(np.int32))
    mt_domain = meshtags_from_entities(msh, tdim, create_adjacencylist(entities), values)
    mt_domain.name = "domain_marker"
    gmsh.finalize()
    with XDMFFile(MPI.COMM_WORLD, f"{filename}.xdmf", "w") as file:
        file.write_mesh(msh)
        msh.topology.create_connectivity(tdim - 1, tdim)
        file.write_meshtags(mt_domain)
        file.write_meshtags(mt)


def create_surface_mesh(domain, points, line_pts, model, tags):
    pts = get_surface_points(domain, points, line_pts)
    ps = []
    for point in pts:
        ps.append(gmsh.model.occ.addPoint(point[0], point[1], 0))

    lines = [gmsh.model.occ.addLine(ps[i - 1], ps[i]) for i in range(len(ps))]
    curve = gmsh.model.occ.addCurveLoop(lines)
    surface = gmsh.model.occ.addPlaneSurface([curve])
    model.occ.synchronize()
    model.addPhysicalGroup(2, [surface], tag=tags[0])
    model.addPhysicalGroup(1, [lines[0]] + lines[len(line_pts):], tag=tags[1])
    model.addPhysicalGroup(1, lines[1:len(line_pts)], tag=tags[2])
    model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")


def create_unsplit_box_2d(H=1.0, L=5.0, res=0.1, x0=[0.0, 0.5], x1=[5.0, 0.7], quads=False,
                          filename="box_2D", num_segments=10, curve_fun=horizontal_sine):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if quads:
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 2)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    if MPI.COMM_WORLD.rank == 0:
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", res)
        model = gmsh.model()
        model.add("box")
        model.setCurrent("box")
        t = np.linspace(0, 1, num_segments + 1)
        line_pts = curve_fun(t, x0, x1)
        pts = get_surface_points([2, 3, 4, 5], [[0.0, 0.0], [L, 0.0], [L, H], [0.0, H], x0, x1], line_pts)
        ps1 = []
        for point in pts:
            ps1.append(gmsh.model.occ.addPoint(point[0], point[1], 0))
        p1 = model.occ.addPoint(0.0, 0.0, 0.0)
        p2 = model.occ.addPoint(L, 0.0, 0.0)
        p3 = ps1[-3]
        p4 = ps1[0]
        ps2 = [p3, p2, p1, p4]

        lines1 = [model.occ.addLine(ps1[i - 1], ps1[i]) for i in range(len(ps1))]
        lines2 = [model.occ.addLine(ps2[i - 1], ps2[i]) for i in range(1, len(ps2))]
        curve1 = model.occ.addCurveLoop(lines1)
        curve2 = model.occ.addCurveLoop(lines1[1:-2] + lines2)
        surface1 = model.occ.addPlaneSurface([curve1])
        surface2 = model.occ.addPlaneSurface([curve2])
        model.occ.synchronize()
        model.addPhysicalGroup(2, [surface1, surface2], tag=1)
        model.addPhysicalGroup(1, [lines1[0]] + lines1[-2:] + lines2, tag=2)
        model.mesh.generate(2)
        model.mesh.optimize("Netgen")

        if quads:
            gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("quadrangle", 1), root=0)
        else:
            gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("triangle", 1), root=0)

        gmsh_facet_id = model.mesh.getElementType("line", 1)
        x, cells, cell_data, marked_facets, facet_values = retrieve_mesh_data(
            model, "box", gmsh_cell_id, gmsh_facet_id)
    else:
        gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
        num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
        cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
        marked_facets, facet_values = np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int32)

    if quads:
        gmsh_quad4 = cell_perm_array(CellType.quadrilateral, 4)
        cells = cells[:, gmsh_quad4]
    create_dolfinx_mesh(filename, x[:, :2], cells, cell_data, gmsh_cell_id, marked_facets, facet_values, 2)


def create_unsplit_box_3d(L=5.0, H=1.0, W=1.0, res=0.1, fname="box_3D", hex=False,
                          curve_fun=horizontal_sine, num_segments=10, x0=[0.0, 0.5], x1=[5.0, 0.7]):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if hex:
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 2)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    if MPI.COMM_WORLD.rank == 0:
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", res)
        model = gmsh.model()
        model.add("box")
        model.setCurrent("box")

        t = np.linspace(0, 1, num_segments + 1)
        line_pts = curve_fun(t, x0, x1)
        pts = get_surface_points([2, 3, 4, 5], [[0.0, 0.0], [L, 0.0], [L, H], [0.0, H], x0, x1], line_pts)
        ps1 = []
        ps3 = []
        for point in pts:
            ps1.append(gmsh.model.occ.addPoint(point[0], point[1], 0))
            ps3.append(gmsh.model.occ.addPoint(point[0], point[1], W))
        p1 = model.occ.addPoint(0.0, 0.0, 0.0)
        p2 = model.occ.addPoint(L, 0.0, 0.0)
        p3 = ps1[-3]
        p4 = ps1[0]
        ps2 = [p3, p2, p1, p4]
        p1 = model.occ.addPoint(0.0, 0.0, W)
        p2 = model.occ.addPoint(L, 0.0, W)
        p3 = ps3[-3]
        p4 = ps3[0]
        ps4 = [p3, p2, p1, p4]

        lines1 = [model.occ.addLine(ps1[i - 1], ps1[i]) for i in range(len(ps1))]
        lines2 = [model.occ.addLine(ps2[i - 1], ps2[i]) for i in range(1, len(ps2))]
        curve1 = model.occ.addCurveLoop(lines1)
        curve2 = model.occ.addCurveLoop(lines1[1:-2] + lines2)
        surface1 = model.occ.addPlaneSurface([curve1])
        surface2 = model.occ.addPlaneSurface([curve2])
        if not hex:
            lines3 = [model.occ.addLine(ps3[i - 1], ps3[i]) for i in range(len(ps3))]
            lines4 = [model.occ.addLine(ps4[i - 1], ps4[i]) for i in range(1, len(ps4))]
            curve3 = model.occ.addCurveLoop(lines3)
            curve4 = model.occ.addCurveLoop(lines3[1:-2] + lines4)
            surface3 = model.occ.addPlaneSurface([curve3])
            surface4 = model.occ.addPlaneSurface([curve4])
            lines5 = [model.occ.addLine(ps1[i], ps3[i]) for i in range(len(ps1))]
            curves1 = []
            for i in range(len(lines1)):
                curves1.append(model.occ.addCurveLoop([lines1[i], lines5[i], -lines3[i], -lines5[i - 1]]))

            curves2 = []
            lines6 = [lines5[-3]]
            lines6.append(model.occ.addLine(ps2[1], ps4[1]))
            lines6.append(model.occ.addLine(ps2[2], ps4[2]))
            lines6.append(lines5[0])
            for i in range(len(lines2)):
                curves2.append(model.occ.addCurveLoop([lines2[i], lines6[i + 1], -lines4[i], -lines6[i]]))
            surfaces1 = [model.occ.addPlaneSurface([curve]) for curve in curves1]
            surfaces2 = [model.occ.addPlaneSurface([curve]) for curve in curves2]
            sloop1 = model.occ.addSurfaceLoop([surface1] + surfaces1 + [surface3])
            sloop2 = model.occ.addSurfaceLoop([surface2] + surfaces1[1:-2] + surfaces2 + [surface4])
            vol1 = model.occ.addVolume([sloop1])
            vol2 = model.occ.addVolume([sloop2])
            model.occ.synchronize()
            out_vol_tags, _ = model.occ.fragment([(3, vol1)], [(3, vol2)])
            model.occ.synchronize()
            p_v = [v_tag[1] for v_tag in out_vol_tags]
            model.addPhysicalGroup(3, p_v, tag=1)
            model.addPhysicalGroup(2, [surface1, surface2, surface3, surface4,
                                   surfaces1[0], surfaces1[-1]] + surfaces2, tag=2)
            model.addPhysicalGroup(2, surfaces1[1:-2], tag=3)
        else:
            square = model.occ.add_rectangle(0, 0, 0, L, H)
            model.occ.extrude([(2, square)], 0, 0, W, numElements=[np.ceil(1. / res)], recombine=True)
            model.occ.synchronize()
            volumes = model.getEntities(3)
            model.occ.synchronize()
            model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], tag=1)
            bndry = model.getBoundary([(3, volumes[0][1])], oriented=False)
            model.addPhysicalGroup(2, [b[1] for b in bndry], tag=2)
        model.mesh.generate(3)

        if hex:
            gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("hexahedron", 1), root=0)
            gmsh_facet_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("quadrangle", 1), root=0)
        else:
            gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("tetrahedron", 1), root=0)
            gmsh_facet_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("triangle", 1), root=0)
        x, cells, cell_data, marked_facets, facet_values = retrieve_mesh_data(
            model, "box", gmsh_cell_id, gmsh_facet_id)
    else:
        gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
        num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
        cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
        marked_facets, facet_values = np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int32)

    if hex:
        gmsh_hex8 = cell_perm_array(CellType.hexahedron, 8)
        cells = cells[:, gmsh_hex8]
        gmsh_quad4 = cell_perm_array(CellType.quadrilateral, 4)
        marked_facets = marked_facets[:, gmsh_quad4]
    create_dolfinx_mesh(fname, x, cells, cell_data, gmsh_cell_id, marked_facets, facet_values, 3)


def create_tet_mesh(domain, points, line_pts, model, tags, z):
    pts = get_surface_points(domain, points, line_pts)
    ps1 = []
    ps2 = []
    for point in pts:
        ps1.append(gmsh.model.occ.addPoint(point[0], point[1], 0))
        ps2.append(gmsh.model.occ.addPoint(point[0], point[1], z))

    lines1 = [gmsh.model.occ.addLine(ps1[i - 1], ps1[i]) for i in range(len(ps1))]
    curve1 = gmsh.model.occ.addCurveLoop(lines1)
    surface1 = gmsh.model.occ.addPlaneSurface([curve1])

    lines2 = [gmsh.model.occ.addLine(ps2[i - 1], ps2[i]) for i in range(len(ps2))]
    curve2 = gmsh.model.occ.addCurveLoop(lines2)
    surface2 = gmsh.model.occ.addPlaneSurface([curve2])

    lines_z = [gmsh.model.occ.addLine(ps1[i], ps2[i]) for i in range(len(ps2))]

    surfaces = []
    for i in range(len(pts)):
        curve = gmsh.model.occ.addCurveLoop([lines1[i], lines_z[i], -lines2[i], -lines_z[i - 1]])
        surfaces.append(gmsh.model.occ.addPlaneSurface([curve]))

    sloop = gmsh.model.occ.addSurfaceLoop([surface1] + surfaces + [surface2])
    volume = gmsh.model.occ.addVolume([sloop])
    model.occ.synchronize()

    model.addPhysicalGroup(2, [surfaces[i] for i in range(1, len(surfaces) - 2)], tag=tags[2])
    model.addPhysicalGroup(2, [surface1, surface2, surfaces[0], surfaces[-2], surfaces[-1]], tag=tags[1])
    model.addPhysicalGroup(3, [volume], tag=tags[0])
    model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")


def create_hex_mesh(domain, points, line_pts, model, tags, z, res):
    pts = get_surface_points(domain, points, line_pts)
    ps = []
    for point in pts:
        ps.append(gmsh.model.occ.addPoint(point[0], point[1], 0))

    lines = [gmsh.model.occ.addLine(ps[i - 1], ps[i]) for i in range(len(ps))]
    curve = gmsh.model.occ.addCurveLoop(lines)
    surface = gmsh.model.occ.addPlaneSurface([curve])

    model.occ.extrude([(2, surface)], 0, 0, z, numElements=[np.ceil(5 * z / res)], recombine=True)
    model.occ.synchronize()
    volumes = model.getEntities(3)
    surfaces = model.getEntities(2)

    model.addPhysicalGroup(2, [surfaces[i][1] for i in range(2, len(surfaces) - 3)], tag=tags[2])
    model.addPhysicalGroup(2, [surfaces[i][1]
                           for i in [0, 1, len(surfaces) - 3, len(surfaces) - 2, len(surfaces) - 1]], tag=tags[1])
    model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], tag=tags[0])

    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 2)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")


def create_split_box_2D(filename: str, res=0.8, L=5.0, H=1.0, domain_1=[0, 4, 5, 3],
                        domain_2=[4, 1, 2, 5], x0=[2.5, 0.0], x1=[2.5, 1.0], curve_fun=vertical_line,
                        num_segments=(1, 2), quads=False):
    points = [[0.0, 0.0], [L, 0.0], [L, H], [0.0, H], x0, x1]
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if quads:
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 2)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    if MPI.COMM_WORLD.rank == 0:
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", res)
        model = gmsh.model()
        model.add("first")
        model.setCurrent("first")
        # Create box
        t = np.linspace(0, 1, num_segments[0] + 1)
        line_pts = curve_fun(t, x0, x1)
        tags = [1, 3, 4]
        create_surface_mesh(domain_1, points, line_pts, model, tags)

        # Broadcast cell type data and geometric dimension
        if quads:
            gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("quadrangle", 1), root=0)
        else:
            gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("triangle", 1), root=0)

        # Get mesh data for dim (0, tdim) for all physical entities
        gmsh_facet_id = model.mesh.getElementType("line", 1)
        x, cells, cell_data, marked_facets, facet_values = retrieve_mesh_data(
            model, "first", gmsh_cell_id, gmsh_facet_id)
        model.add("second")
        model.setCurrent("second")
        # Create box
        t = np.linspace(0, 1, num_segments[1] + 1)
        line_pts = curve_fun(t, x0, x1)
        tags = [2, 5, 6]
        create_surface_mesh(domain_2, points, line_pts, model, tags)

        # Get mesh data for dim (0, tdim) for all physical entities
        x2, cells2, cell_data2, marked_facets2, facet_values2 = retrieve_mesh_data(
            model, "second", gmsh_cell_id, gmsh_facet_id)

        # combine mesh data
        marked_facets = np.vstack([marked_facets, marked_facets2 + x.shape[0]])
        facet_values = np.hstack([facet_values, facet_values2])
        cell_data = np.hstack([cell_data, cell_data2])
        cells = np.vstack([cells, cells2 + x.shape[0]])
        x = np.vstack([x, x2])

    else:
        gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
        num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
        cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
        marked_facets, facet_values = np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int32)

    if quads:
        gmsh_quad4 = cell_perm_array(CellType.quadrilateral, 4)
        cells = cells[:, gmsh_quad4]
    create_dolfinx_mesh(filename, x[:, :2], cells, cell_data, gmsh_cell_id, marked_facets, facet_values, 2)


def create_split_box_3D(filename: str, res=0.8, L=5.0, H=1.0, W=1.0, domain_1=[0, 4, 5, 3],
                        domain_2=[4, 1, 2, 5], x0=[2.5, 0.0], x1=[2.5, 1.0], curve_fun=vertical_line,
                        num_segments=(1, 2), hex=False):
    points = [[0.0, 0.0], [L, 0.0], [L, H], [0.0, H], x0, x1]
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if MPI.COMM_WORLD.rank == 0:
        gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", res)
        model = gmsh.model()
        model.add("first")
        model.setCurrent("first")
        # Create box
        t = np.linspace(0, 1, num_segments[0] + 1)
        line_pts = curve_fun(t, x0, x1)
        tags = [1, 3, 4]
        if hex:
            create_hex_mesh(domain_1, points, line_pts, model, tags, W, res)
            gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("hexahedron", 1), root=0)
            gmsh_facet_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("quadrangle", 1), root=0)
        else:
            create_tet_mesh(domain_1, points, line_pts, model, tags, W)
            gmsh_cell_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("tetrahedron", 1), root=0)
            gmsh_facet_id = MPI.COMM_WORLD.bcast(model.mesh.getElementType("triangle", 1), root=0)
        x, cells, cell_data, marked_facets, facet_values = retrieve_mesh_data(
            model, "first", gmsh_cell_id, gmsh_facet_id)

        model.add("second")
        model.setCurrent("second")
        t = np.linspace(0, 1, num_segments[1] + 1)
        line_pts = curve_fun(t, x0, x1)
        tags = [2, 5, 6]
        if hex:
            create_hex_mesh(domain_2, points, line_pts, model, tags, W, 0.8 * res)

        else:
            create_tet_mesh(domain_2, points, line_pts, model, tags, W)
        # Create box

        # Get mesh data for dim (0, tdim) for all physical entities
        x2, cells2, cell_data2, marked_facets2, facet_values2 = retrieve_mesh_data(
            model, "second", gmsh_cell_id, gmsh_facet_id)

        # combine mesh data
        marked_facets = np.vstack([marked_facets, marked_facets2 + x.shape[0]])
        facet_values = np.hstack([facet_values, facet_values2])
        cell_data = np.hstack([cell_data, cell_data2])
        cells = np.vstack([cells, cells2 + x.shape[0]])
        x = np.vstack([x, x2])

    else:
        gmsh_cell_id = MPI.COMM_WORLD.bcast(None, root=0)
        num_nodes = MPI.COMM_WORLD.bcast(None, root=0)
        cells, x = np.empty([0, num_nodes]), np.empty([0, 3])
        marked_facets, facet_values = np.empty((0, 3), dtype=np.int64), np.empty((0,), dtype=np.int32)
    if hex:
        gmsh_hex8 = cell_perm_array(CellType.hexahedron, 8)
        cells = cells[:, gmsh_hex8]
        gmsh_quad4 = cell_perm_array(CellType.quadrilateral, 4)
        marked_facets = marked_facets[:, gmsh_quad4]
    create_dolfinx_mesh(filename, x, cells, cell_data, gmsh_cell_id, marked_facets, facet_values, 3)


create_unsplit_box_3d()
