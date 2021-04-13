
from helpers import convert_mesh
from mpi4py import MPI
import gmsh
import warnings
warnings.filterwarnings("ignore")


def create_disk_mesh(LcMin=0.005, LcMax=0.015, filename="disk.msh"):
    """
    Create a disk mesh centered at (0.5, 0.5) with radius 0.5.
    Mesh is finer at (0.5,0) using LcMin, and gradually decreasing to LcMax
    """
    gmsh.initialize()
    if MPI.COMM_WORLD.rank == 0:
        disk = gmsh.model.occ.addDisk(0.5, 0.5, 0, 0.5, 0.5)
        p = gmsh.model.occ.addPoint(0.5, 0, 0, tag=5)
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

        gmsh.write("disk.msh")

    gmsh.finalize()


if __name__ == "__main__":
    filename = "disk"
    create_disk_mesh(filename=f"{filename}.msh")
    convert_mesh(filename, "triangle")
