# Copyright (C) 2021 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT
import dolfinx
from matplotlib import pyplot as plt
import numpy as np


# Visualise the gap. For debugging. Works in 2D only
def plot_gap(mesh, contact, gaps, entities, num_pairs):
    gdim = mesh.geometry.dim
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh_geometry = mesh.geometry.x

    for i in range(num_pairs):
        facet_map = contact.facet_map(i)
        c_to_f = mesh.topology.connectivity(tdim, tdim - 1)
        num_facets = entities[i].shape[0]
        facet_origin = np.zeros(num_facets, dtype=np.int32)
        for j in range(num_facets):
            cell = entities[i][j, 0]
            f_index = entities[i][j, 1]
            facet_origin[j] = c_to_f.links(cell)[f_index]
        facets_opp = facet_map.array
        facets_opp = facets_opp[facets_opp >= 0]

        # Draw facets on opposite surface
        plt.figure(dpi=600)
        for facet in facets_opp:
            facet_geometry = dolfinx.cpp.mesh.entities_to_geometry(mesh._cpp_object, fdim, [facet], False)
            coords = mesh_geometry[facet_geometry][0]
            plt.plot(coords[:, 0], coords[:, 1], color="black")
        min_x = 1
        max_x = 0
        for j in range(num_facets):
            facet = facet_origin[j]
            facet_geometry = dolfinx.cpp.mesh.entities_to_geometry(mesh._cpp_object, fdim, [facet], False)
            coords = mesh_geometry[facet_geometry][0]
            plt.plot(coords[:, 0], coords[:, 1], color="black")
            qp = contact.qp_phys(i, j)
            num_qp = qp.shape[0]
            for q in range(num_qp):
                g = gaps[i][j, q * gdim:(q + 1) * gdim]
                x = [qp[q, 0], qp[q, 0] + g[0]]
                y = [qp[q, 1], qp[q, 1] + g[1]]
                max_x = max(x[0], x[1], max_x)
                min_x = min(x[0], x[1], min_x)
                plt.plot(x, y)
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.xlim(min_x, max_x)
        rank = mesh.comm.rank
        plt.savefig(f"gap_{i}_{rank}.png")
