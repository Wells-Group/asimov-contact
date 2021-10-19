// Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx_cuas/QuadratureRule.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace
{
/// Sort pair of integers by its first index
bool sort_by_first(const std::pair<std::int32_t, int>& a,
                   const std::pair<std::int32_t, int>& b)
{
  return (a.first < b.first);
}

/// Tabulate coordinate element basis functions at quadrature points on each
/// facet of the reference cell
/// @param[in] cmap The coordinate map
/// @param[in] q_rule The quadrature rule
/// @param[out] phi_facets The coordinate element basis functions at the
/// quadrature points
std::vector<xt::xtensor<double, 2>>
tabulate_cmap_at_facets(const dolfinx::fem::CoordinateElement& cmap,
                        dolfinx_cuas::QuadratureRule& q_rule)
{
  if (q_rule.dim() != cmap.topological_dimension() - 1)
    throw std::runtime_error("Coordinate and quadrature rule not matching");

  const std::vector<xt::xarray<double>>& q_points = q_rule.points_ref();
  std::vector<xt::xtensor<double, 2>> phi_facets;
  phi_facets.reserve(q_points.size());
  for (std::size_t i = 0; i < q_points.size(); i++)
  {
    auto facet_points = q_points[i];
    auto phi_i = cmap.tabulate(0, facet_points);
    xt::xtensor<double, 2> phi({phi_i.shape(1), phi_i.shape(2)});
    phi.assign(xt::view(phi_i, 0, xt::all(), xt::all(), 0));
    phi_facets.push_back(phi);
  }
  return phi_facets;
}

/// Push forward coordinates on a set of facets for a given quadrature rule
/// @param[in] quadrature_surface The surface to evaluate at every quadrature
/// point
/// @param[in] q_rule The quadrature rule
/// @param[in] mesh The mesh containing the surface
/// @param[out] (q_phys, num_q_points) The quadrature points pushed forward for
/// all facets in the surface. The num_q_points is an Adjacency list where each
/// node indicates the cell index, and the links indicate how many quadrature
/// points there are on each facet of the cell. The offset of this adjacency
/// list matches the input quadrature surface.
/// @note The surfaces are represented with a tuple (cell_index,
/// local_entitiy_index) where the cell index is local to the process,
/// the local_entity index is local to the cell
std::pair<xt::xtensor<double, 2>, dolfinx::graph::AdjacencyList<std::int32_t>>
push_forward_at_facets(
    std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
        quadrature_surface,
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
    dolfinx_cuas::QuadratureRule& q_rule)
{
  auto mesh_geometry = mesh->geometry().x();
  auto x_dofmap = mesh->geometry().dofmap();
  const int gdim = mesh->geometry().dim();
  // Tabulate coordinate element at quadrature points for each facet of the
  // reference cell
  const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();
  std::vector<xt::xtensor<double, 2>> phi_facets
      = tabulate_cmap_at_facets(cmap, q_rule);

  // Compute how many quadrature points there is per facet (per unique surface
  // cell)
  const std::vector<xt::xarray<double>>& q_points = q_rule.points_ref();
  std::vector<std::int32_t> num_quadrature_points;
  for (std::size_t i = 0; i < quadrature_surface->num_nodes(); i++)
  {
    auto facets = quadrature_surface->links(i);
    for (std::size_t j = 0; j < facets.size(); j++)
    {
      const std::int32_t num_facet_points = q_points[facets[j]].shape(0);
      num_quadrature_points.push_back(num_facet_points);
    }
  }
  dolfinx::graph::AdjacencyList<std::int32_t> surface_to_num_qp(
      num_quadrature_points, quadrature_surface->offsets());
  const std::vector<std::int32_t>& num_qps = surface_to_num_qp.array();
  const std::size_t total_qps
      = std::accumulate(num_qps.begin(), num_qps.end(), std::int64_t(0));
  xt::xtensor<double, 2> qp_phys({total_qps, static_cast<std::size_t>(gdim)});
  const std::vector<std::int32_t>& surface_offsets
      = quadrature_surface->offsets();
  std::size_t insert_offset = 0;
  for (std::size_t i = 0; i < quadrature_surface->num_nodes(); i++)
  {
    auto facets = quadrature_surface->links(i);

    /// Get input position in total array
    auto num_qps = surface_to_num_qp.links(i);
    std::vector<std::int32_t> num_qps_offset(num_qps.size() + 1);
    num_qps_offset[0] = 0;
    std::inclusive_scan(num_qps.begin(), num_qps.end(),
                        num_qps_offset.begin() + 1);

    // Extract geometry dofs
    auto x_dofs = x_dofmap.links(i);
    auto coordinate_dofs
        = xt::view(mesh_geometry, xt::keep(x_dofs), xt::range(0, gdim));

    for (std::size_t j = 0; j < facets.size(); j++)
    {
      const std::int32_t index = facets[j];
      xt::xtensor<double, 2> qp_facet({static_cast<std::size_t>(num_qps[j]),
                                       static_cast<std::size_t>(gdim)});
      cmap.push_forward(qp_facet, coordinate_dofs, phi_facets[index]);
      xt::view(qp_phys,
               xt::range(insert_offset + num_qps_offset[j],
                         insert_offset + num_qps_offset[j + 1]),
               xt::all())
          = qp_facet;
    }
    insert_offset += num_qps_offset.back();
  }
  return {qp_phys, surface_to_num_qp};
};

/// Compute the closest entity to a point
// void compute_gap_function(
//     std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
//         quadrature_surface,
//     std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
//         candidate_surface,
//     std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
//     dolfinx_cuas::QuadratureRule q_rule)
// {
//   auto mesh_geometry = mesh->geometry().x();
//   auto x_dofmap = mesh->geometry().dofmap();
//   const int gdim = mesh->geometry().dim();
//   // Tabulate coordinate element at quadrature points for each facet of the
//   // reference cell
//   const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();
//   std::vector<xt::xtensor<double, 2>> phi_facets
//       = tabulate_cmap_at_facets(cmap, q_rule);

//   // Find maximum number of quadrature points per facet
//   std::size_t max_qp = 1;
//   const std::vector<xt::xarray<double>>& q_points = q_rule.points_ref();
//   for (std::size_t i = 0; i < q_points.size(); i++)
//   {
//     auto facet_points = q_points[i];
//     max_qp = std::max(max_qp, facet_points.shape(0));
//   }
// }

/// Given to surfaces, compute the closest cell on the second surface to
/// every quadrature point on the first surface.
/// @param[in] quadrature_surface The surface to evaluate at every
/// quadrature point
/// @param[in] candidate_surface The surface of candidate facets
/// @param[in] mesh The mesh containing both surfaces
/// @param[in] q_points The quadrature points for each facet on the reference
/// element
/// @param[in] qp_phys_facets The quadrature points for each facet in the
/// quadrature surface
/// @param[out] A map from each cell index (local to process) to the closest
/// cell (local to process) on the other interface
/// @note The surfaces are represented with a tuple (cell_index,
/// local_entitiy_index) where the cell index is local to the process, the
/// local_entity index is local to the cell
// dolfinx::graph::AdjacencyList<std::int32_t> compute_closest_cells(
//     std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
//         quadrature_surface,
//     std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
//         candidate_surface,
//     std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
//     const std::vector<xt::xarray<double>>& q_points,
//     const std::vector<std::vector<xt::xtensor<double, 2>>>& qp_phys_facets)
// {
//   auto mesh_geometry = mesh->geometry().x();
//   const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
//       = mesh->geometry().dofmap();
//   const int gdim = mesh->geometry().dim();

//   // Push forward quadrature points at all input facets
//   const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();

//   // Find maximum number of quadrature points per facet
//   std::size_t max_qp = 1;
//   for (std::size_t i = 0; i < q_points.size(); i++)
//   {
//     auto facet_points = q_points[i];
//     max_qp = std::max(max_qp, facet_points.shape(0));
//   }

//   // Create midpoint tree as compute_closest_entity will be called many
//   // times
//   std::vector<std::int32_t> candidate_cells;
//   // There largest number of cells is if there is facet per cell
//   candidate_cells.reserve(candidate_surface->offsets().back());
//   for (std::size_t i = 0; i <= candidate_surface->num_nodes(); i++)
//     if (candidate_surface->num_links(i) > 0)
//       candidate_cells.push_back(i);
//   const int tdim = mesh->topology().dim();
//   dolfinx::geometry::BoundingBoxTree candidate_bbox(*mesh, tdim,
//                                                     candidate_cells);
//   auto candidate_midpoint_tree
//       = dolfinx::geometry::create_midpoint_tree(*mesh, tdim,
//       candidate_cells);

//   // Compute number of unique cells on candidate facet
//   int unique_cells = 0;
//   for (std::size_t i = 0; i <= quadrature_surface->num_nodes(); i++)
//     if (quadrature_surface->num_links(i) > 0)
//       unique_cells++;

//   // For each quadrature cell, find which cell is closest
//   // on the other surface at every quadrature point
//   std::vector<std::int32_t> closest_cells;
//   closest_cells.reserve(max_qp * quadrature_surface->offsets().back());
//   std::vector<std::int32_t> cell_offsets(1, 0);
//   cell_offsets.reserve(unique_cells);

//   std::array<double, 3> point;
//   point.fill(0);
//   xt::xtensor_fixed<double, xt::xshape<1, 3>> _p;
//   // for (std::size_t i = 0; i < quadrature_surface->num_nodes(); i++)
//   // {
//   //   auto facets = quadrature_surface->links(i);
//   //   for (std::size_t j = 0; j < facets.size(); j++)
//   //   {
//   //     const int index = facets[j];
//   //     auto facet_points = q_points[index];
//   //     const int num_q_points = facet_points.shape(0);
//   //     const xt::xtensor<double, 2>& qp_phys = qp_phys_facets[i][j];
//   //     // Find closest facet on the other side for each quadrature point
//   //     for (std::int32_t k = 0; k < num_q_points; k++)
//   //     {
//   //       for (int l = 0; l < gdim; l++)
//   //         point[k] = qp_phys(k, l);
//   //       // Do initial search to get radius R0 to search per entity
//   //       auto [cell0, R0] = dolfinx::geometry::compute_closest_entity(
//   //           candidate_midpoint_tree, point, *mesh);
//   //       // Find closest facet
//   //       auto [cell, R] = dolfinx::geometry::compute_closest_entity(
//   //           candidate_bbox, point, *mesh, R0);

//   //       // Compute gap function
//   //       auto dofs = x_dofmap.links(cell);
//   //       xt::xtensor<double, 2> nodes({dofs.size(), 3});
//   //       for (std::size_t i = 0; i < dofs.size(); ++i)
//   //         for (std::size_t j = 0; j < 3; ++j)
//   //           nodes(i, j) = mesh_geometry(dofs[i], j);
//   //       std::copy(point.begin(), point.end(), _p.data());
//   //       dolfinx::geometry::compute_distance_gjk(_p, nodes);

//   // // Map facet to cell
//   // closest_cells.push_back(cell);
//   //   }
//   // }
//   // cell_offsets.push_back(closest_cells.size());
//   // }
//   return std::move(
//       dolfinx::graph::AdjacencyList<std::int32_t>(closest_cells,
//       cell_offsets));
// };

} // namespace

namespace dolfinx_contact
{
class ContactInterface
{
public:
  /// Constructor
  /// @param[in] marker The meshtags defining the contact surfaces
  /// @param[in] surface_0 Value of the meshtag marking the first surface
  /// @param[in] surface_1 Value of the meshtag marking the second surface
  /// @param[in] q_rule The quadrature rule to use
  ContactInterface(
      std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> marker,
      int surface_0, int surface_1, dolfinx_cuas::QuadratureRule q_rule)
      : _q_rule(q_rule), _mesh(marker->mesh()), _surface_0(), _surface_1(),
        _cell_map_0(), _cell_map_1()
  {
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh = marker->mesh();
    const int tdim = mesh->topology().dim();
    mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
    mesh->topology_mutable().create_connectivity(tdim, tdim - 1);
    const dolfinx::mesh::Topology& topology = mesh->topology();
    auto f_to_c = mesh->topology().connectivity(tdim - 1, tdim);
    assert(f_to_c);
    auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
    assert(c_to_f);
    std::vector<std::int32_t> facets_0 = marker->find(surface_0);
    std::vector<std::int32_t> facets_1 = marker->find(surface_1);

    // Helper function to convert facets indices to an adjacency list for
    // cells with local indices in each link
    auto get_cell_indices = [c_to_f, f_to_c](std::vector<std::int32_t> facets)
    {
      // Count how many facets each cell has in the list
      std::vector<std::int32_t> counter(c_to_f->num_nodes(), 0);
      for (auto facet : facets)
      {
        auto cells = f_to_c->links(facet);
        assert(cells.size() == 1);
        counter[cells[0]]++;
      }

      // Create offset
      const int num_cells = c_to_f->num_nodes();
      std::vector<std::int32_t> offset(num_cells + 1);
      offset[0] = 0;
      std::partial_sum(counter.begin(), counter.end(),
                       std::next(offset.begin()));
      std::fill(counter.begin(), counter.end(), 0);
      std::vector<std::int32_t> data(offset.back());
      for (auto facet : facets)
      {
        auto cells = f_to_c->links(facet);
        const std::int32_t cell = cells[0];

        // Find local facet index
        auto local_facets = c_to_f->links(cell);
        const auto it
            = std::find(local_facets.begin(), local_facets.end(), facet);
        assert(it != local_facets.end());
        const int facet_index = std::distance(local_facets.begin(), it);
        data[offset[cell] + counter[cell]++] = facet_index;
      }
      return std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          data, offset);
    };
    _surface_0 = get_cell_indices(facets_0);
    _surface_1 = get_cell_indices(facets_1);
    push_forward_at_facets(_surface_0, _mesh, _q_rule);
  }

  // /// Compute gap function for candidate surface at every quadrature point
  // std::vector<std::vector<std::array<double, 3>>> compute_gap_function()
  // {
  //   // Push quadrature points forward to facets
  //   std::vector<std::vector<xt::xtensor<double, 2>>> qp_phys_facets
  //       = push_forward_at_facets(_surface_0, _mesh, _q_rule);
  //   // Compute cell->cell map
  //   _cell_map_0 =
  //   std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
  //       compute_closest_cells(_surface_0, _surface_1, _mesh,
  //                             _q_rule.points_ref(), qp_phys_facets));

  //   // Compute gap function
  //   std::vector<std::vector<std::array<double, 3>>> gap;
  //     = compute_gap_function(_mesh, _cell_map_0, qp_phys_facets);

  // return gap;
  //};

  /// Create interface map

private:
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _surface_0;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _surface_1;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _cell_map_0;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _cell_map_1;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _facet_map_0;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _facet_map_1;

  dolfinx_cuas::QuadratureRule _q_rule;
  std::shared_ptr<const dolfinx::mesh::Mesh> _mesh;
};

} // namespace dolfinx_contact