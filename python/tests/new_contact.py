import basix
from IPython import embed
import dolfinx
import dolfinx.io
import dolfinx_cuas.cpp
import dolfinx_contact.cpp as contact
from mpi4py import MPI
import numpy as np
import ufl

points = np.array([[0, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0],
                   [-0.1, -1, 0],
                   [0.7, -1, 0],
                   [0.3, -2, 0],
                   [1, 1, 0]], dtype=np.float64)
cells = np.array([[0, 1, 2], [1, 2, 6], [5, 3, 4]], dtype=np.int32)
cell = ufl.Cell("triangle", geometric_dimension=points.shape[1])
domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, 1))
mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, points, domain)
tdim = mesh.topology.dim
facets_0 = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], 0))
facets_02 = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[0], 0))
facets_1 = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], -1))
facets = np.hstack([facets_0, facets_02, facets_1])
print(facets_0, facets_02, facets_1)
values = np.hstack([np.full(len(facets_0), 1, dtype=np.int32),
                    np.full(len(facets_02), 1, dtype=np.int32),
                    np.full(len(facets_1), 2, dtype=np.int32)])
indices = np.argsort(facets)
mt = dolfinx.MeshTags(mesh, tdim - 1, facets[indices], values[indices])

ci = contact.ContactInterface(mt, 1, 2)
q_rule = dolfinx_cuas.cpp.QuadratureRule(mesh.topology.cell_type, 2, mesh.topology.dim - 1)

family = "Lagrange"
ct = dolfinx.cpp.mesh.to_string(mesh.topology.cell_type)
degree = 1
el = basix.create_element(basix.finite_element.string_to_family(
    family, ct), basix.cell.string_to_type(ct), degree, basix.LagrangeVariant.equispaced)

cell_map = ci.create_cell_map(q_rule, el)
# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "test_mesh.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_meshtags(mt)

embed()
# embed()
