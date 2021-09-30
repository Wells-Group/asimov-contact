// Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx_cuas/math.hpp>
#include <basix/cell.h>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx_cuas/utils.h>
#include <iostream>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtl/xspan.hpp>

namespace dolfinx_contact
{
  class Contact
  {
  public:
    /// Constructor
    /// @param[in] marker The meshtags defining the contact surfaces
    /// @param[in] surface_0 Value of the meshtag marking the first surface
    /// @param[in] surface_1 Value of the meshtag marking the first surface
    Contact(std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> marker, int surface_0,
            int surface_1, std::shared_ptr<dolfinx::fem::FunctionSpace> V)
        : _marker(marker), _surface_0(surface_0), _surface_1(surface_1), _V(V)
    {
      _facet_0 = marker->find(_surface_0);
      _facet_1 = marker->find(_surface_1);
    }

    // Return Adjacency list of closest facet on surface_1 for every quadrature point in _qp_phys_0
    // (quadrature points on every facet of surface_0)
    const std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> map_0_to_1() const
    {
      return _map_0_to_1;
    }
    // Return Adjacency list of closest facet on surface_0 for every quadrature point in _qp_phys_1
    // (quadrature points on every facet of surface_1)
    const std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> map_1_to_0() const
    {
      return _map_1_to_0;
    }
    const std::vector<int32_t> &facet_0() const { return _facet_0; }
    const std::vector<int32_t> &facet_1() const { return _facet_1; }
    // Return meshtag value for surface_0
    const int surface_0() const { return _surface_0; }
    // Return mestag value for suface_1
    const int surface_1() const { return _surface_1; }
    // return quadrature degree
    const int quadrature_degree() const { return _quadrature_degree; }
    void set_quadrature_degree(int deg) { _quadrature_degree = deg; }

    // Return meshtags
    std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> meshtags() const { return _marker; }

    /// Tabulate the basis function at the quadrature points _qp_ref_facet
    /// creates and fills _phi_ref_facets
    xt::xtensor<double, 3> tabulate_on_ref_cell(basix::FiniteElement element)
    {

      // Create _phi_ref_facets
      std::uint32_t num_facets = _qp_ref_facet.shape(0);
      std::uint32_t num_quadrature_pts = _qp_ref_facet.shape(1);
      std::uint32_t num_local_dofs = element.dim();
      xt::xtensor<double, 3> phi({num_facets, num_quadrature_pts, num_local_dofs});

      // Tabulate basis functions at quadrature points _qp_ref_facet for each facet of the reference
      // cell. Fill _phi_ref_facets
      for (int i = 0; i < num_facets; ++i)
      {
        auto phi_i = xt::view(phi, i, xt::all(), xt::all());
        auto q_facet = xt::view(_qp_ref_facet, i, xt::all(), xt::all());
        auto cell_tab = element.tabulate(0, q_facet);
        phi_i = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
      }
      return phi;
    }

    /// Compute push forward of quadrature points _qp_ref_facet to the physical facet for
    /// each facet in _facet_"origin_meshtag" Creates and fills _qp_phys_"origin_meshtag"
    /// @param[in] origin_meshtag flag to choose between surface_0 and  surface_1
    void create_q_phys(int origin_meshtag)
    {
      // Mesh info
      auto mesh = _marker->mesh();
      auto mesh_geometry = mesh->geometry().x();
      auto cmap = mesh->geometry().cmap();
      auto x_dofmap = mesh->geometry().dofmap();
      const int gdim = mesh->geometry().dim(); // geometrical dimension
      const int tdim = mesh->topology().dim(); // topological dimension
      const int fdim = tdim - 1;               // topological dimension of facet

      // Connectivity to evaluate at quadrature points
      mesh->topology_mutable().create_connectivity(fdim, tdim);
      auto f_to_c = mesh->topology().connectivity(fdim, tdim);
      mesh->topology_mutable().create_connectivity(tdim, fdim);
      auto c_to_f = mesh->topology().connectivity(tdim, fdim);

      std::vector<int32_t> *puppet_facets; // pointer to point to _facet_"origin_meshtag"
      xt::xtensor<double, 3> *q_phys_pt;   // pointer to point to _qp_phys_"origin_meshtag"
      if (origin_meshtag == 0)
      {
        puppet_facets = &_facet_0;
        // create _qp_phys_0
        _qp_phys_0 = xt::xtensor<double, 3>(
            {(*puppet_facets).size(), _qp_ref_facet.shape(1), _qp_ref_facet.shape(2)});
        q_phys_pt = &_qp_phys_0;
      }
      else
      {
        puppet_facets = &_facet_1;
        // create _qp_phys_1
        _qp_phys_1 = xt::xtensor<double, 3>(
            {(*puppet_facets).size(), _qp_ref_facet.shape(1), _qp_ref_facet.shape(2)});
        q_phys_pt = &_qp_phys_1;
      }

      // push forward of quadrature points _qp_ref_facet to physical facet for each facet in
      // _facet_"origin_meshtag"
      xt::xtensor<double, 2> q_phys({_qp_ref_facet.shape(1), _qp_ref_facet.shape(2)});
      for (int i = 0; i < (*puppet_facets).size(); ++i)
      {
        int facet = (*puppet_facets)[i]; // extract facet
        auto cells = f_to_c->links(facet);
        // since the facet is on the boundary it should only link to one cell
        assert(cells.size() == 1);
        auto cell = cells[0]; // extract cell

        // find local index of facet
        auto facets = c_to_f->links(cell);
        auto local_facet = std::find(facets.begin(), facets.end(), facet);
        const std::int32_t local_index = std::distance(facets.data(), local_facet);

        // extract local dofs
        auto x_dofs = x_dofmap.links(cell);
        auto coordinate_dofs = xt::view(mesh_geometry, xt::keep(x_dofs), xt::range(0, gdim));
        auto phi_facet = xt::view(_phi_ref_facets, local_index, xt::all(), xt::all());
        xt::xtensor<double, 2> q_phys({_qp_ref_facet.shape(1), _qp_ref_facet.shape(2)});

        // push forward of quadrature points _qp_ref_facet to the physical facet
        cmap.push_forward(q_phys, coordinate_dofs, phi_facet);
        xt::view(*q_phys_pt, i, xt::all(), xt::all()) = q_phys;
      }
    }

    /// Compute closest candidate_facet for each quadrature point in _qp_phys_"origin_meshtag"
    /// This is saved as an adjacency list _map_0_to_1 or _map_1_to_0
    void create_distance_map(int origin_meshtag)
    {
      // Create _qp_ref_facet (quadrature points on reference facet)
      auto facet_quadrature = dolfinx_cuas::create_reference_facet_qp(_marker->mesh(), _quadrature_degree);
      _qp_ref_facet = facet_quadrature.first;
      _qw_ref_facet = facet_quadrature.second;

      // Tabulate basis function on reference cell (_phi_ref_facets)// Create coordinate element
      // FIXME: For higher order geometry need basix element public in mesh
      // auto degree = mesh->geometry().cmap()._element->degree;
      int degree = 1;
      auto dolfinx_cell = _marker->mesh()->topology().cell_type();
      auto coordinate_element = basix::create_element(
          basix::element::family::P, dolfinx::mesh::cell_type_to_basix_type(dolfinx_cell), degree,
          basix::element::lagrange_variant::equispaced);
      _phi_ref_facets = tabulate_on_ref_cell(coordinate_element);

      // Compute quadrature points on physical facet _qp_phys_"origin_meshtag"
      create_q_phys(origin_meshtag);

      // Mesh info
      auto mesh = _marker->mesh();
      const int gdim = mesh->geometry().dim();
      const int tdim = mesh->topology().dim();
      const int fdim = tdim - 1;

      std::array<double, 3> point;
      point[2] = 0;

      // assign puppet_ and candidate_facets
      std::vector<int32_t> *candidate_facets;
      std::vector<int32_t> *puppet_facets;
      xt::xtensor<double, 3> *q_phys_pt;
      if (origin_meshtag == 0)
      {
        puppet_facets = &_facet_0;
        candidate_facets = &_facet_1;
        q_phys_pt = &_qp_phys_0;
      }
      else
      {
        puppet_facets = &_facet_1;
        candidate_facets = &_facet_0;
        q_phys_pt = &_qp_phys_1;
      }
      // Create midpoint tree as compute_closest_entity will be called many times
      dolfinx::geometry::BoundingBoxTree master_bbox(*mesh, fdim, (*candidate_facets));
      auto master_midpoint_tree = dolfinx::geometry::create_midpoint_tree(*mesh, fdim, (*candidate_facets));

      std::vector<std::int32_t> data; // will contain closest candidate facet
      std::vector<std::int32_t> offset(1);
      offset[0] = 0;
      for (int i = 0; i < (*puppet_facets).size(); ++i)
      {
        for (int j = 0; j < (*q_phys_pt).shape(1); ++j)
        {
          for (int k = 0; k < gdim; ++k)
            point[k] = (*q_phys_pt)(i, j, k);
          // Find initial search radius R = intermediate_result.second
          std::pair<int, double> intermediate_result = dolfinx::geometry::compute_closest_entity(master_midpoint_tree, point, *mesh);
          // Find closest facet to point
          std::pair<int, double> search_result = dolfinx::geometry::compute_closest_entity(
              master_bbox, point, *mesh, intermediate_result.second);
          data.push_back(search_result.first);
        }
        offset.push_back(data.size());
      }

      // save as an adjacency list _map_0_to_1 or _map_1_to_0
      if (origin_meshtag == 0)
        _map_0_to_1 = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(data, offset);
      else
        _map_1_to_0 = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(data, offset);
    }

    /// Compute and pack the gap function for each quadrature point the set of facets.
    /// For a set of facets; go through the quadrature points on each facet find the closest facet on
    /// the other surface and compute the distance vector
    /// @param[in] orgin_meshtag - surface on which to integrate
    /// @param[out] c - gap packed on facets. c[i*cstride +  gdim * k+ j] contains the jth component
    /// of the Gap on the ith facet at kth quadrature point
    std::pair<std::vector<PetscScalar>, int> pack_gap(int origin_meshtag)
    {
      // Mesh info
      auto mesh = _marker->mesh();             // mesh
      const int gdim = mesh->geometry().dim(); // geometrical dimension
      const int tdim = mesh->topology().dim();
      const int fdim = tdim - 1;
      const xt::xtensor<double, 2> &mesh_geometry = mesh->geometry().x();
      std::vector<int32_t> *puppet_facets;
      std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> map;
      xt::xtensor<double, 3> *q_phys_pt;

      // Select which side of the contact interface to loop from and get the correct map
      if (origin_meshtag == 0)
      {
        puppet_facets = &_facet_0;
        map = _map_0_to_1;
        q_phys_pt = &_qp_phys_0;
      }
      else
      {
        puppet_facets = &_facet_1;
        map = _map_1_to_0;
        q_phys_pt = &_qp_phys_1;
      }
      const std::int32_t num_facets = (*puppet_facets).size();
      const std::int32_t num_q_point = _qp_ref_facet.shape(1);

      // Pack gap function for each quadrature point on each facet
      std::vector<PetscScalar> c(num_facets * num_q_point * gdim, 0.0);
      const int cstride = num_q_point * gdim;
      xt::xtensor<double, 2> point = {{0, 0, 0}};

      for (int i = 0; i < num_facets; ++i)
      {
        auto master_facets = map->links(i);
        auto master_facet_geometry = dolfinx::mesh::entities_to_geometry(*mesh, fdim, master_facets, false);
        int offset = i * cstride;
        for (int j = 0; j < map->num_links(i); ++j)
        {
          // Get quadrature points in physical space for the ith facet, jth quadrature point
          for (int k = 0; k < gdim; k++)
            point(0, k) = (*q_phys_pt)(i, j, k);

          // Get the coordinates of the geometry on the other interface, and compute the distance of
          // the convex hull created by the points
          auto master_facet = xt::view(master_facet_geometry, j, xt::all());
          auto master_coords = xt::view(mesh_geometry, xt::keep(master_facet), xt::all());
          auto dist_vec = dolfinx::geometry::compute_distance_gjk(master_coords, point);

          // Add distance vector to coefficient array
          for (int k = 0; k < gdim; k++)
            c[offset + j * gdim + k] -= dist_vec(k);
        }
      }
      return {std::move(c), cstride};
    }

    /// Pack gap with rigid surface defined by x[gdim-1] = -g.
    /// g_vec = zeros(gdim), g_vec[gdim-1] = -g
    /// Gap = x - g_vec
    /// @param[in] orgin_meshtag - surface on which to integrate
    /// @param[in] g - defines location of plane
    /// @param[out] c - gap packed on facets. c[i, gdim * k+ j] contains the jth component of the Gap
    /// on the ith facet at kth quadrature point
    std::pair<std::vector<PetscScalar>, int> pack_gap_plane(int origin_meshtag, double g)
    {
      // Mesh info
      auto mesh = _marker->mesh();             // mesh
      const int gdim = mesh->geometry().dim(); // geometrical dimension
      const int tdim = mesh->topology().dim();
      const int fdim = tdim - 1;
      auto mesh_geometry = mesh->geometry().x();
      // Create _qp_ref_facet (quadrature points on reference facet)
      auto facet_quadrature = dolfinx_cuas::create_reference_facet_qp(_marker->mesh(), _quadrature_degree);
      _qp_ref_facet = facet_quadrature.first;
      _qw_ref_facet = facet_quadrature.second;

      // Tabulate basis function on reference cell (_phi_ref_facets)// Create coordinate element
      // FIXME: For higher order geometry need basix element public in mesh
      // auto degree = mesh->geometry().cmap()._element->degree;
      int degree = 1;
      auto dolfinx_cell = _marker->mesh()->topology().cell_type();
      auto coordinate_element = basix::create_element(
          basix::element::family::P, dolfinx::mesh::cell_type_to_basix_type(dolfinx_cell), degree,
          basix::element::lagrange_variant::equispaced);

      _phi_ref_facets = tabulate_on_ref_cell(coordinate_element);
      // Compute quadrature points on physical facet _qp_phys_"origin_meshtag"
      create_q_phys(origin_meshtag);
      std::vector<int32_t> *puppet_facets;
      xt::xtensor<double, 3> *q_phys_pt;
      if (origin_meshtag == 0)
      {
        puppet_facets = &_facet_0;
        q_phys_pt = &_qp_phys_0;
      }
      else
      {
        puppet_facets = &_facet_1;
        q_phys_pt = &_qp_phys_1;
      }
      int32_t num_facets = (*puppet_facets).size();
      int32_t num_q_point = _qp_ref_facet.shape(1);
      std::vector<PetscScalar> c(num_facets * num_q_point * gdim, 0.0);
      const int cstride = num_q_point * gdim;
      for (int i = 0; i < num_facets; i++)
      {
        int offset = i * cstride;
        for (int k = 0; k < num_q_point; k++)
        {
          for (int j = 0; j < gdim; j++)
          {
            c[offset + k * gdim + j] += (*q_phys_pt)(i, k, j);
          }
          c[offset + (k + 1) * gdim - 1] += g;
        }
      }
      return {std::move(c), cstride};
    }

  private:
    int _quadrature_degree = 3;
    std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> _marker;
    int _surface_0;                                  // meshtag value for surface 0
    int _surface_1;                                  // meshtag value for surface 1
    std::shared_ptr<dolfinx::fem::FunctionSpace> _V; // Function space

    // Adjacency list of closest facet on surface_1 for every quadrature point in _qp_phys_0
    // (quadrature points on every facet of surface_0)
    std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _map_0_to_1;
    // Adjacency list of closest facet on surface_0 for every quadrature point in _qp_phys_1
    // (quadrature points on every facet of surface_1)
    std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _map_1_to_0;
    // quadrature points on physical facet for each facet on surface 0
    xt::xtensor<double, 3> _qp_phys_0;
    // quadrature points on physical facet for each facet on surface 1
    xt::xtensor<double, 3> _qp_phys_1;
    // quadrature points on reference facet
    xt::xtensor<double, 3> _qp_ref_facet;
    // quadrature weights
    std::vector<double> _qw_ref_facet;
    // quadrature points on facets of reference cell
    xt::xtensor<double, 3> _phi_ref_facets;
    // facets in surface 0
    std::vector<int32_t> _facet_0;
    // facets in surface 1
    std::vector<int32_t> _facet_1;
    // normals on surface 0 in order of facets in _facet_0
    xt::xtensor<double, 2> _normals_0;
    // normals on surface 1 in order of facets in _facet_1
    xt::xtensor<double, 2> _normals_1;
  };
} // namespace dolfinx_contact
