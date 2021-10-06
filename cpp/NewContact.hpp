// Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx_cuas/QuadratureRule.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>

namespace
{
    // Sort pair of integers by its first index
    bool sort_by_first(const std::pair<std::int32_t, int> &a,
                       const std::pair<std::int32_t, int> &b)
    {
        return (a.first < b.first);
    }

}

namespace dolfinx_contact
{
    class ContactInterface
    {
    public:
        /// Constructor
        /// @param[in] marker The meshtags defining the contact surfaces
        /// @param[in] surface_0 Value of the meshtag marking the first surface
        /// @param[in] surface_1 Value of the meshtag marking the second surface
        ContactInterface(std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> marker, int surface_0,
                         int surface_1)
            : _mesh(marker->mesh()), _surface_0(), _surface_1()
        {
            std::shared_ptr<const dolfinx::mesh::Mesh> mesh = marker->mesh();
            const int tdim = mesh->topology().dim();
            mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
            mesh->topology_mutable().create_connectivity(tdim, tdim - 1);
            const dolfinx::mesh::Topology &topology = mesh->topology();
            auto f_to_c = mesh->topology().connectivity(tdim - 1, tdim);
            assert(f_to_c);
            auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
            assert(c_to_f);
            std::vector<std::int32_t> facets_0 = marker->find(surface_0);
            std::vector<std::int32_t> facets_1 = marker->find(surface_1);

            // Helper function to convert facets indices to an adjacency list for cells with local indices in each link
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
                std::cout << xt::adapt(counter) << " " << xt::adapt(facets) << "\n";
                // Create offset
                const int num_cells = c_to_f->num_nodes();
                std::vector<std::int32_t> offset(num_cells + 1);
                offset[0] = 0;
                std::partial_sum(counter.begin(), counter.end(), std::next(offset.begin()));
                std::fill(counter.begin(), counter.end(), 0);
                std::vector<std::int32_t> data(offset.back());
                for (auto facet : facets)
                {
                    auto cells = f_to_c->links(facet);
                    const std::int32_t cell = cells[0];

                    // Find local facet index
                    auto local_facets = c_to_f->links(cell);
                    const auto it = std::find(local_facets.begin(), local_facets.end(), facet);
                    assert(it != local_facets.end());
                    const int facet_index = std::distance(local_facets.begin(), it);
                    data[offset[cell] + counter[cell]++] = facet_index;
                }
                std::cout << xt::adapt(data) << xt::adapt(offset) << "\n";
                return std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(data, offset);
            };
            _surface_0 = get_cell_indices(facets_0);
            _surface_1 = get_cell_indices(facets_1);
        }

        /// Compute closest points on other surface 1 for a given quadrature-rule
        // void compute_projection_surface_0(dolfinx_cuas::QuadratureRule &q_rule, const basix::FiniteElement &element)
        // {
        //     auto mesh_geometry = _mesh->geometry().x();
        //     auto x_dofmap = _mesh->geometry().dofmap();
        //     const int gdim = _mesh->geometry().dim();
        //     const std::vector<xt::xarray<double>> &q_points = q_rule.points_ref();
        //     std::vector<std::vector<xt::xarray<double>>> _q_physical;

        //     // Tabulate coordinate element at quadrature points for each facet
        //     const dolfinx::fem::CoordinateElement &cmap = _mesh->geometry().cmap();
        //     std::vector<xt::xtensor<double, 2>> phi_facets;
        //     phi_facets.reserve(q_points.size());
        //     for (std::size_t i = 0; i < q_points.size(); i++)
        //     {
        //         auto facet_points = q_points[i];
        //         auto phi_i = cmap.tabulate(0, facet_points);
        //         xt::xtensor<double, 2> phi({phi_i.shape(1), phi_i.shape(2)});
        //         phi.assign(xt::view(phi_i, 0, xt::all(), xt::all(), 0));
        //         phi_facets.push_back(phi);
        //     }

        //     // Create midpoint tree as compute_closest_entity will be called many times
        //     std::vector<std::int32_t> candidate_facets;
        //     candidate_facets.reserve(_facets_1.size());
        //     for (std::size_t i = 0; i <= _facets_1.size(); i++)
        //         candidate_facets.push_back(_facets_1[i].first);
        //     const int fdim = _mesh->topology().dim() - 1;
        //     dolfinx::geometry::BoundingBoxTree candidate_bbox(*_mesh, fdim, candidate_facets);
        //     auto candidate_midpoint_tree = dolfinx::geometry::create_midpoint_tree(*_mesh, fdim, (candidate_facets));

        //     // Compute connectivity to map facets to cells
        //     _mesh->topology_mutable().create_connectivity(fdim, fdim + 1);
        //     auto f_to_c = _mesh->topology().connectivity(fdim, fdim + 1);
        //     assert(f_to_c);

        //     // Compute Adjacency list where each row contains the closest cell on the other interface for the
        //     // set of quadrature points
        //     std::vector<std::int32_t> cell_indices;
        //     cell_indices.reserve(_facets_0.size());
        //     std::vector<std::int32_t> offset(1);
        //     offset.reserve(_facets_0.size());

        //     std::array<double, 3> point;
        //     point.fill(0);
        //     for (std::size_t i = 0; i < _facets_0.size(); i++)
        //     {
        //         auto [cell, index] = _facets_0[i];
        //         auto facet_points = q_points[index];
        //         const int num_q_points = facet_points.shape(0);
        //         // Extract geometry dofs
        //         auto x_dofs = x_dofmap.links(cell);
        //         auto coordinate_dofs = xt::view(mesh_geometry, xt::keep(x_dofs), xt::range(0, gdim));

        //         // Push quadrature rule forward to reference
        //         std::array<std::size_t, 2>
        //             shape = {num_q_points, gdim};
        //         xt::xtensor<double, 2> q_phys(shape);
        //         cmap.push_forward(q_phys, coordinate_dofs, phi_facets[index]);

        //         // Find closest facet on the other side for each quadrature point
        //         for (std::int32_t j = 0; j < num_q_points; j++)
        //         {
        //             for (int k = 0; k < gdim; k++)
        //                 point[k] = q_phys(j, k);
        //             // Do initial search to get radius R0 to search per entity
        //             auto [facet0, R0] = dolfinx::geometry::compute_closest_entity(candidate_midpoint_tree, point, *_mesh);
        //             // Find closest facet
        //             auto [facet, R] = dolfinx::geometry::compute_closest_entity(
        //                 candidate_bbox, point, *_mesh, R0);
        //             // Map facet to cell
        //             auto cells = f_to_c->links(facet);
        //             assert(cells.size() == 1);
        //             cell_indices.push_back(cells[0]);
        //         }
        //         offset.push_back(cell_indices.size());
        //     }
        //     std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> surface_map = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(cell_indices, offset);
        // }

        std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> &
        surface_0()
        {
            return _surface_0;
        }

        std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> &
        surface_1()
        {
            return _surface_1;
        }

    private:
        std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _surface_0;
        std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _surface_1;

        std::shared_ptr<const dolfinx::mesh::Mesh> _mesh;
    };

} // namespace dolfinx_cuas