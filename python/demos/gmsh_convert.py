import meshio
import numpy

mesh = meshio.read("xmas.msh")
mesh.remove_lower_dimensional_cells()
cells = {'triangle': numpy.concatenate((mesh.cells[0].data, mesh.cells[1].data))}
points = mesh.points
mesh = meshio.Mesh(points=points, cells=cells)
meshio.write('xmas.xdmf', mesh)
