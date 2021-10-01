import dolfinx
import basix
import numpy as np

__all__ = ["facet_master_puppet_relation"]

_dolfinx_to_basix_celltype = {dolfinx.cpp.mesh.CellType.interval: basix.CellType.interval,
                              dolfinx.cpp.mesh.CellType.triangle: basix.CellType.triangle,
                              dolfinx.cpp.mesh.CellType.quadrilateral: basix.CellType.quadrilateral,
                              dolfinx.cpp.mesh.CellType.hexahedron: basix.CellType.hexahedron,
                              dolfinx.cpp.mesh.CellType.tetrahedron: basix.CellType.tetrahedron}


def facet_master_puppet_relation(mesh, puppet_facets, candidate_facets, quadrature_degree=None):
    """
    For a set of facets, find which of the candidate facets are closest to each of them.

    Parameters
    ----------
    mesh
        The mesh
    puppet_facets
        List of facets (local to process) that we are finding the closest facet to.
    candidate_facets
        List of facets (local to process) that are the possible closest facets
    quadrature_degree
        Integer (default: None) indicating if the search should include points on the
        facet (quadrature points of some order). If None: search is only done at vertices
    """

    # Mesh info
    gdim = mesh.geometry.dim
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh_geometry = mesh.geometry.x
    x_dofmap = mesh.geometry.dofmap
    cell_type = mesh.topology.cell_type
    cmap = mesh.geometry.cmap
    degree = mesh.ufl_domain().ufl_coordinate_element().degree()

    # Create midpoint tree as compute_closest_entity will be called many times
    master_bbox = dolfinx.cpp.geometry.BoundingBoxTree(mesh, fdim, candidate_facets)
    master_midpoint_tree = dolfinx.cpp.geometry.create_midpoint_tree(mesh, fdim, candidate_facets)

    # Connectivity to evaluate at vertices
    mesh.topology.create_connectivity(fdim, 0)
    f_to_v = mesh.topology.connectivity(fdim, 0)

    # Connectivity to evaluate at quadrature points
    mesh.topology.create_connectivity(fdim, tdim)
    f_to_c = mesh.topology.connectivity(fdim, tdim)
    mesh.topology.create_connectivity(tdim, fdim)
    c_to_f = mesh.topology.connectivity(tdim, fdim)

    # Create reference topology and geometry
    basix_cell = _dolfinx_to_basix_celltype[cell_type]
    facet_topology = basix.topology(basix_cell)[fdim]
    ref_geom = basix.geometry(basix_cell)

    # Create facet quadrature points
    if quadrature_degree is not None:
        # FIXME: Does not work for prism meshes
        basix_facet = _dolfinx_to_basix_celltype[dolfinx.cpp.mesh.cell_entity_type(cell_type, fdim, 0)]
        quadrature_points, _ = basix.make_quadrature("default", basix_facet, quadrature_degree)

        # Tabulate basis functions at quadrature points
        # FIXME: Does not work for prism meshes
        surface_cell_type = dolfinx.cpp.mesh.cell_entity_type(mesh.topology.cell_type, mesh.topology.dim - 1, 0)
        surface_str = dolfinx.cpp.mesh.to_string(surface_cell_type)

        # Push forward quadrature points on reference facet to reference cell
        surface_element = basix.create_element(basix.finite_element.string_to_family("Lagrange", surface_str),
                                               basix.cell.string_to_type(surface_str),
                                               degree, basix.LagrangeVariant.equispaced)
        c_tab = surface_element.tabulate_x(0, quadrature_points)
        phi_s = c_tab[0, :, :, 0]  # Assuming value_size == 1 for coordinate element
        q_cell = {}
        for i, facet in enumerate(facet_topology):
            coords = ref_geom[facet]
            q_cell[i] = phi_s @ coords

    puppet_to_master = {}
    print("dist python")
    for facet in puppet_facets:
        if quadrature_degree is None:
            # For each vertex on facet, find closest entity on the other interface
            vertices = f_to_v.links(facet)
            vertex_x = dolfinx.cpp.mesh.entities_to_geometry(mesh, 0, vertices, False)
            m_facets = []
            for geometry_index in vertex_x:
                point = mesh_geometry[geometry_index].reshape(3,)
                print("point")
                print(point)
                # Find initial search radius
                potential_facet, R_init = dolfinx.geometry.compute_closest_entity(master_midpoint_tree, point, mesh)
                # Find mesh entity
                master_facet, R = dolfinx.geometry.compute_closest_entity(master_bbox, point, mesh, R=R_init)
                m_facets.append(master_facet)
            puppet_to_master[facet] = np.unique(m_facets)
        else:
            # For each facet, find the facet closest to each quadratue point
            o_facets = []

            # First, find local index of facet in cell
            cells = f_to_c.links(facet)
            assert(len(cells) == 1)
            cell = cells[0]
            x_dofs = x_dofmap.links(cell)
            facets = c_to_f.links(cell)
            local_index = np.argwhere(facets == facet)[0, 0]

            # Second, push forward reference quadrature (cell) to physical cell
            coordinate_dofs = mesh_geometry[x_dofs, : gdim]
            x = cmap.push_forward(q_cell[local_index], coordinate_dofs)

            quadrature_padded = np.zeros((x.shape[0], 3))
            quadrature_padded[:, :gdim] = x
            for point in quadrature_padded:
                # Find initial search radius
                potential_facet, R_init = dolfinx.geometry.compute_closest_entity(master_midpoint_tree, point, mesh)

                # Find mesh entity
                master_facet, R = dolfinx.geometry.compute_closest_entity(master_bbox, point, mesh, R=R_init)

                # Compute distance from quadrature point to closest facet
                master_facet_geometry = dolfinx.cpp.mesh.entities_to_geometry(mesh, fdim, [master_facet], False)
                master_coords = mesh_geometry[master_facet_geometry][0]
                # print(master_coords.shape)
                # print(point.shape)
                dist_vec = dolfinx.geometry.compute_distance_gjk(master_coords, point)
                print(dist_vec)
                o_facets.append(master_facet)

            puppet_to_master[facet] = np.unique(o_facets)
    return puppet_to_master
