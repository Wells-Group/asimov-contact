# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT
import dolfinx
from matplotlib import pyplot as plt


# Visualise the gap. For debugging. Works in 2D only
def plot_gap(mesh, contact, tag, gap, facets, facets_opp):
    gdim = mesh.geometry.dim
    fdim = mesh.topology.dim - 1
    mesh_geometry = mesh.geometry.x

    # Draw facets on opposite surface
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
    plt.savefig(f"gap_{tag}_{rank}.png")
