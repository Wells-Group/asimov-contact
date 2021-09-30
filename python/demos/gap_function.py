
import sys
import argparse

import dolfinx
import dolfinx.geometry
import dolfinx.io
import gmsh
import numpy as np
from dolfinx_cuas.contact import facet_master_puppet_relation
from dolfinx.cpp.io import perm_gmsh
from dolfinx.cpp.mesh import to_type
import dolfinx_cuas.cpp as cuas
from dolfinx.io import (extract_gmsh_geometry,
                        extract_gmsh_topology_and_markers, ufl_mesh_from_gmsh)
from dolfinx.mesh import create_mesh
from mpi4py import MPI

# Initialize gmsh
gmsh.initialize()


def mesh2D(res_min=0.1, res_max=0.25):
    gdim = 2
    r = 0.4
    c_x, c_y = 2.5, 0.5

    # Create rectangular mesh
    rect_tag = gmsh.model.occ.addRectangle(0, -1, 0, 5, 1)
    # Create circular mesh
    circ_tag = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [rect_tag], rect_tag)
    gmsh.model.addPhysicalGroup(2, [circ_tag], circ_tag)

    # View gmsh output
    # gmsh.option.setNumber("General.Terminal", 1)

    # Create finer mesh on upper geometry
    gmsh.model.mesh.field.add("Distance", 1)
    circle_arc = gmsh.model.getBoundary([(2, circ_tag)])
    gmsh.model.mesh.field.setNumbers(1, "EdgesList", [e[1] for e in circle_arc])
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(2, "LcMax", res_max)
    gmsh.model.mesh.field.setNumber(2, "DistMin", r / 4)
    gmsh.model.mesh.field.setNumber(2, "DistMax", r / 1.5)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.model.mesh.generate(dim=gdim)
    return gdim, (c_x, c_y), r


def mesh3D(res_min=0.1, res_max=0.25):
    gdim = 3
    r = 0.4
    c_x, c_y, c_z = 1.5, 1.5, 0.5

    # Create box mesh
    box_tag = gmsh.model.occ.addBox(0, 0, -1, 2.5, 2.5, 1)
    # Create sphere
    sphere_tag = gmsh.model.occ.addSphere(c_x, c_y, c_z, r)
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [box_tag], box_tag)
    gmsh.model.addPhysicalGroup(3, [sphere_tag], sphere_tag)
    # Create finer mesh on upper geometry
    gmsh.model.mesh.field.add("Distance", 1)
    sphere_surface = gmsh.model.getBoundary([(3, sphere_tag)])
    gmsh.model.mesh.field.setNumbers(1, "FacesList", [e[1] for e in sphere_surface])
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(2, "LcMax", res_max)
    gmsh.model.mesh.field.setNumber(2, "DistMin", r / 4)
    gmsh.model.mesh.field.setNumber(2, "DistMax", r / 1.5)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    gmsh.model.mesh.generate(dim=gdim)
    return gdim, (c_x, c_y, c_z), r


parser = argparse.ArgumentParser()
solver_parser = parser.add_mutually_exclusive_group(required=False)
solver_parser.add_argument('--3D', dest='is_3D', default=False, action='store_true',
                           help="2D or 3D simulation (default 3D=False)")
args = parser.parse_args()

thismodule = sys.modules[__name__]
is_3D = None
for key in vars(args):
    setattr(thismodule, key, getattr(args, key))

if is_3D:
    gdim, c, r = mesh3D()
    c_x, c_y, c_z = c
else:
    gdim, c, r = mesh2D()
    c_x, c_y = c


# Convert mesh to dolfin-X
cell_information = {}
topologies, cell_dimensions, x = None, None, None
if MPI.COMM_WORLD.rank == 0:
    # Get mesh geometry
    x = extract_gmsh_geometry(gmsh.model)

    # Get mesh topology for each element
    topologies = extract_gmsh_topology_and_markers(gmsh.model)

    # Get information about each cell type from the msh files
    num_cell_types = len(topologies.keys())
    cell_dimensions = np.zeros(num_cell_types, dtype=np.int32)
    for i, element in enumerate(topologies.keys()):
        properties = gmsh.model.mesh.getElementProperties(element)
        name, dim, order, num_nodes, local_coords, _ = properties
        cell_information[i] = {"id": element, "dim": dim,
                               "num_nodes": num_nodes}
        cell_dimensions[i] = dim

gmsh.finalize()
if MPI.COMM_WORLD.rank == 0:
    # Sort elements by ascending dimension
    perm_sort = np.argsort(cell_dimensions)

    # Broadcast cell type data and geometric dimension
    cell_id = cell_information[perm_sort[-1]]["id"]
    tdim = cell_information[perm_sort[-1]]["dim"]
    num_nodes = cell_information[perm_sort[-1]]["num_nodes"]
    cell_id, num_nodes = MPI.COMM_WORLD.bcast([cell_id, num_nodes], root=0)
    cells = topologies[cell_id]["topology"]
    cell_values = topologies[cell_id]["cell_data"]

else:
    cell_id, num_nodes = MPI.COMM_WORLD.bcast([None, None], root=0)
    cells, x = np.empty([0, num_nodes]), np.empty([0, gdim])
    cell_values = np.empty((0,))


# Create distributed mesh
ufl_domain = ufl_mesh_from_gmsh(cell_id, gdim)
gmsh_cell_perm = perm_gmsh(to_type(str(ufl_domain.ufl_cell())), num_nodes)
cells = cells[:, gmsh_cell_perm]
mesh = create_mesh(MPI.COMM_WORLD, cells, x[:, :gdim], ufl_domain)
tdim = mesh.topology.dim
fdim = tdim - 1


def curved_contact(x):
    if is_3D:
        return np.logical_and(np.isclose((x[0] - c_x)**2 + (x[1] - c_y)**2 + (x[2] - c_z)**2, r**2),
                              x[2] < c_z - 0.1 * r)

    else:
        # Curved contact area (Parts of upper circle)
        return np.logical_and(np.isclose((x[0] - c_x)**2 + (x[1] - c_y)**2, r**2), x[1] < c_y - 0.1 * r)


def master_obstacle(x):
    # Parts of facets on top of rectangle
    if is_3D:
        return np.logical_and(np.logical_and(np.isclose(x[2], 0), (x[0] - c_x)**2 + (x[1] - c_y)**2 < (2 * r)**2),
                              np.abs(x[1] - c_y) < 2 * r)
    else:
        return np.logical_and(np.isclose(x[1], 0), np.abs(x[0] - c_x) < 2 * r)


# Locate facets on boundary of circle
circ_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, curved_contact)
sorted = np.argsort(circ_facets)
circ_facets = circ_facets[sorted]

# Locate facets on boundary of rectangle
rect_facets = dolfinx.mesh.locate_entities_boundary(mesh, fdim, master_obstacle)
sorted = np.argsort(rect_facets)
rect_facets = rect_facets[sorted]

values = np.hstack([np.full(circ_facets.size, 1), np.full(rect_facets.size, 2)])
indices = np.hstack([circ_facets, rect_facets])
values = np.asarray(values, dtype=np.int32)
sorted = np.argsort(indices)
mt = dolfinx.MeshTags(mesh, mesh.topology.dim - 1, indices[sorted], values[sorted])
V = dolfinx.VectorFunctionSpace(mesh, ("CG", 1))
contact = cuas.contact.Contact(mt, 1, 2, V._cpp_object)
contact.create_distance_map(1)
circ_to_rect = facet_master_puppet_relation(mesh, rect_facets, circ_facets, quadrature_degree=2)

# print(f"With quadrature eval: {circ_to_rect}")
circ_to_rect = facet_master_puppet_relation(mesh, rect_facets, circ_facets)
# print(f"Without quadrature eval: {circ_to_rect}")

# Write contact facets to file (NOTE: Indicies has to be sorted)
indices = np.asarray(np.hstack([np.asarray(list(circ_to_rect.keys())), np.hstack(
    [circ_to_rect[key] for key in circ_to_rect.keys()])]), dtype=np.int32)
arg_sort = np.argsort(indices)
# values = np.hstack([np.full(len(circ_facets), 2, dtype=np.int32), np.full(len(rect_facets), 1, dtype=np.int32)])
values = indices
ft = dolfinx.MeshTags(mesh, fdim, indices[arg_sort], indices[arg_sort])
ft.name = "Contact facets"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mt.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ft)
