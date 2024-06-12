# Copyright (C) 2021 Jørgen S. Dokken
#
# SPDX-License-Identifier:    MIT

import typing
import warnings
from pathlib import Path

from mpi4py import MPI

import gmsh

warnings.filterwarnings("ignore")

__all__ = ["create_disk_mesh", "create_sphere_mesh"]


def create_disk_mesh(LcMin=0.005, LcMax=0.015, filename: typing.Union[str, Path] = "disk.msh"):
    """
    Create a disk mesh centered at (0.5, 0.5) with radius 0.5.
    Mesh is finer at (0.5,0) using LcMin, and gradually decreasing to LcMax
    """
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        gmsh.model.occ.addDisk(0.5, 0.5, 0, 0.5, 0.5)
        gmsh.model.occ.addPoint(0.5, 0, 0, tag=5)
        gmsh.model.occ.synchronize()
        domains = gmsh.model.getEntities(dim=2)
        domain_marker = 11
        gmsh.model.addPhysicalGroup(domains[0][0], [domains[0][1]], domain_marker)

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "NodesList", [5])

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", LcMin)
        gmsh.model.mesh.field.setNumber(2, "LcMax", LcMax)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.2)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.5)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.model.mesh.generate(2)

        gmsh.write(str(filename))

    gmsh.finalize()


def create_sphere_mesh(LcMin=0.025, LcMax=0.1, filename: typing.Union[str, Path] = "disk.msh"):
    """Create a sphere mesh centered at (0.5, 0.5, 0.5) with radius 0.5.

    Mesh is finer at (0.5, 0.5, 0) using LcMin, and gradually decreasing
    to LcMax.
    """
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        gmsh.model.occ.addSphere(0.5, 0.5, 0.5, 0.5)
        gmsh.model.occ.addPoint(0.5, 0.5, 0, tag=19)
        gmsh.model.occ.synchronize()
        domains = gmsh.model.getEntities(dim=3)
        domain_marker = 11
        gmsh.model.addPhysicalGroup(domains[0][0], [domains[0][1]], domain_marker)

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "NodesList", [19])

        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "IField", 1)
        gmsh.model.mesh.field.setNumber(2, "LcMin", LcMin)
        gmsh.model.mesh.field.setNumber(2, "LcMax", LcMax)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.3)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 0.6)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        gmsh.model.mesh.generate(3)
        gmsh.write(str(filename))

    gmsh.finalize()
