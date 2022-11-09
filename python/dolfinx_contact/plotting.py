# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT
import dolfinx
from matplotlib import pyplot as plt


def plot_facet_3D(ax, facet_geometry, mesh_geometry):
    coords = mesh_geometry[facet_geometry][0]
    if coords.shape[0] == 4:
        ax.plot3D(coords[0:2, 0], coords[0:2, 1],
                  coords[0:2, 2], color="black")
        ax.plot3D([coords[1, 0], coords[3, 0]], [coords[1, 1], coords[3, 1]],
                  [coords[1, 2], coords[3, 2]], color="black")
        ax.plot3D([coords[0, 0], coords[2, 0]], [coords[0, 1], coords[2, 1]],
                  [coords[0, 2], coords[2, 2]], color="black")
        ax.plot3D([coords[2, 0], coords[3, 0]], [coords[2, 1], coords[3, 1]],
                  [coords[2, 2], coords[3, 2]], color="black")
    if coords.shape[0] == 3:
        ax.plot3D([coords[0, 0], coords[1, 0]], [coords[0, 1], coords[1, 1]],
                  [coords[0, 2], coords[1, 2]], color="black")
        ax.plot3D([coords[1, 0], coords[2, 0]], [coords[1, 1], coords[2, 1]],
                  [coords[1, 2], coords[2, 2]], color="black")
        ax.plot3D([coords[2, 0], coords[0, 0]], [coords[2, 1], coords[0, 1]],
                  [coords[2, 2], coords[0, 2]], color="black")

# Visualise the gap. For debugging. Works in 2D only


def plot_gap(mesh, contact, tag, gap, facets, facets_opp, step):
    gdim = mesh.geometry.dim
    fdim = mesh.topology.dim - 1
    mesh_geometry = mesh.geometry.x

    # Draw facets on opposite surface
    if True:
        plt.figure(dpi=600)
        for facet in facets_opp:
            facet_geometry = dolfinx.cpp.mesh.entities_to_geometry(mesh, fdim, [facet], False)
            coords = mesh_geometry[facet_geometry][0]
            plt.plot(coords[:, 0], coords[:, 1], color="black")
        num_facets = len(facets)
        # min_x = 1
        # max_x = 0
        for i in range(num_facets):
            facet = facets[i]
            facet_geometry = dolfinx.cpp.mesh.entities_to_geometry(mesh, fdim, [facet], False)
            coords = mesh_geometry[facet_geometry][0]
            plt.plot(coords[:, 0], coords[:, 1], color="black")
            qp = contact.qp_phys(tag, i)
            num_qp = qp.shape[0]
            for q in range(num_qp):
                g = gap[i, q * gdim:(q + 1) * gdim]
                x = [qp[q, 0], qp[q, 0] + g[0]]
                y = [qp[q, 1], qp[q, 1] + g[1]]
                # max_x = max(x[0], x[1], max_x)
                # min_x = min(x[0], x[1], min_x)
                plt.plot(x, y)
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.xlim(6.5, 8.5)
        # plt.ylim(2.5, 6)
        rank = mesh.comm.rank
        plt.savefig(f"gap_{tag}_{rank}_{step}.png")
    # else:
    #     plt.figure()
    #     ax = plt.axes(projection='3d')
    #     for facet in facets_opp:
    #         facet_geometry = dolfinx.cpp.mesh.entities_to_geometry(mesh, fdim, [facet], False)
    #         plot_facet_3D(ax, facet_geometry, mesh_geometry)

    #     num_facets = len(facets)
    #     # min_x = 1
    #     # max_x = 0
    #     for i in range(num_facets):
    #         qp = contact.qp_phys(tag, i)
    #         num_qp = qp.shape[0]
    #         for q in range(num_qp):
    #             g = gap[i, q * gdim:(q + 1) * gdim]
    #             x = [qp[q, 0], qp[q, 0] + g[0]]
    #             y = [qp[q, 1], qp[q, 1] + g[1]]
    #             z = [qp[q, 2], qp[q, 2] + g[2]]
    #             ax.plot3D(x, y, z)

    #     for facet in facets:
    #         facet_geometry = dolfinx.cpp.mesh.entities_to_geometry(mesh, fdim, [facet], False)
    #         plot_facet_3D(ax, facet_geometry, mesh_geometry)
    #     rank = mesh.comm.rank
    #     plt.savefig(f"gap_{tag}_{rank}_{step}.png")
