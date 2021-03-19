from IPython import embed
import meshio
import numpy
import gmsh
import warnings
warnings.filterwarnings("ignore")
gmsh.initialize()

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
gmsh.model.mesh.field.setNumber(2, "LcMin", 0.005)
gmsh.model.mesh.field.setNumber(2, "LcMax", 0.015)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.2)
gmsh.model.mesh.field.setNumber(2, "DistMax", 0.5)
gmsh.model.mesh.field.setAsBackgroundMesh(2)

gmsh.model.mesh.generate(2)

gmsh.write("disk.msh")


def create_mesh(mesh, cell_type):
    cells = numpy.vstack([cell.data for cell in mesh.cells if cell.type == cell_type])
    data = numpy.hstack([mesh.cell_data_dict["gmsh:physical"][key]
                         for key in mesh.cell_data_dict["gmsh:physical"].keys() if key == cell_type])
    mesh = meshio.Mesh(points=mesh.points[:, :2], cells={cell_type: cells}, cell_data={"name_to_read": [data]})
    return mesh


msh = meshio.read("disk.msh")
meshio.write("disk.xdmf", create_mesh(msh, "triangle"))
