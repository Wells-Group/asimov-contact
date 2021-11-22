# Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT
import dolfinx
from matplotlib import pyplot as plt


# Visualise the gap. For debugging. Works in 2D only
def plot_gap(mesh, contact, tag, gap):
    gdim = mesh.geometry.dim
    fdim = mesh.topology.dim - 1
    mesh_geometry = mesh.geometry.x
    if tag == 0:
        facets = contact.facet_0()
        facets_opp = contact.facet_1()
    else:
        facets = contact.facet_1()
        facets_opp = contact.facet_0()

    # Draw facets on opposite surface
    plt.figure()
    for facet in facets_opp:
        facet_geometry = dolfinx.cpp.mesh.entities_to_geometry(mesh, fdim, [facet], False)
        coords = mesh_geometry[facet_geometry][0]
        plt.plot(coords[:, 0], coords[:, 1], color="black")
    num_facets = len(facets)
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
            plt.plot(x, y)
    plt.savefig(f"gap_{tag}.svg")
