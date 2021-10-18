// Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/cell.h>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/dolfin_la.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx_cuas/QuadratureRule.hpp>
#include <dolfinx_cuas/math.hpp>
#include <dolfinx_cuas/utils.hpp>
#include <iostream>
#include <xtensor/xindex_view.hpp>
#include <xtl/xspan.hpp>

using contact_kernel_fn = std::function<void(
    std::vector<std::vector<PetscScalar>>&, const double*, const double*,
    const double*, const int*, const std::uint8_t*, const std::int32_t)>;
namespace dolfinx_contact
{
class Contact
{
public:
  /// Constructor
  /// @param[in] marker The meshtags defining the contact surfaces
  /// @param[in] surface_0 Value of the meshtag marking the first surface
  /// @param[in] surface_1 Value of the meshtag marking the first surface
  Contact(std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> marker,
          int surface_0, int surface_1,
          std::shared_ptr<dolfinx::fem::FunctionSpace> V)
      : _marker(marker), _surface_0(surface_0), _surface_1(surface_1), _V(V)
  {
    _facet_0 = marker->find(_surface_0);
    _facet_1 = marker->find(_surface_1);
  }

  // Return Adjacency list of closest facet on surface_1 for every quadrature
  // point in _qp_phys_0 (quadrature points on every facet of surface_0)
  const std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
  map_0_to_1() const
  {
    return _map_0_to_1;
  }
  // Return Adjacency list of closest facet on surface_0 for every quadrature
  // point in _qp_phys_1 (quadrature points on every facet of surface_1)
  const std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
  map_1_to_0() const
  {
    return _map_1_to_0;
  }
  const std::vector<int32_t>& facet_0() const { return _facet_0; }
  const std::vector<int32_t>& facet_1() const { return _facet_1; }
  // Return meshtag value for surface_0
  const int surface_0() const { return _surface_0; }
  // Return mestag value for suface_1
  const int surface_1() const { return _surface_1; }
  // return quadrature degree
  const int quadrature_degree() const { return _quadrature_degree; }
  void set_quadrature_degree(int deg) { _quadrature_degree = deg; }

  // quadrature points on physical facet for each facet on surface 0
  std::vector<xt::xtensor<double, 2>> qp_phys_0() { return _qp_phys_0; }
  // quadrature points on physical facet for each facet on surface 1
  std::vector<xt::xtensor<double, 2>> qp_phys_1() { return _qp_phys_1; }

  // Return meshtags
  std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> meshtags() const
  {
    return _marker;
  }

  Mat create_matrix(const dolfinx::fem::Form<PetscScalar>& a, std::string type)
  {

    // Build standard sparsity pattern
    dolfinx::la::SparsityPattern pattern
        = dolfinx::fem::create_sparsity_pattern(a);

    // facet to cell connectivity
    auto mesh = _marker->mesh();
    auto dofmap = a.function_spaces().at(0)->dofmap();
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim(); // topological dimension
    const int fdim = tdim - 1;               // topological dimension of facet

    mesh->topology_mutable().create_connectivity(fdim, tdim);
    auto f_to_c = mesh->topology().connectivity(fdim, tdim);

    for (std::int32_t i = 0; i < _facet_0.size(); i++)
    {
      auto facet = _facet_0[i];
      auto cell = f_to_c->links(facet)[0];
      auto cell_dofs = dofmap->cell_dofs(cell);
      std::int32_t num_links = _map_0_to_1->num_links(i);
      auto links = _map_0_to_1->links(i);
      std::vector<std::int32_t> linked_dofs;
      for (std::int32_t j = 0; j < num_links; j++)
      {
        auto linked_facet = links[j];
        auto linked_cell = f_to_c->links(linked_facet)[0];
        auto linked_cell_dofs = dofmap->cell_dofs(linked_cell);
        for (std::int32_t k = 0; k < linked_cell_dofs.size(); k++)
        {
          linked_dofs.push_back(linked_cell_dofs[k]);
        }
      }
      // Remove duplicates
      std::sort(linked_dofs.begin(), linked_dofs.end());
      linked_dofs.erase(std::unique(linked_dofs.begin(), linked_dofs.end()),
                        linked_dofs.end());

      pattern.insert(cell_dofs, linked_dofs);
      pattern.insert(linked_dofs, cell_dofs);
    }

    for (std::int32_t i = 0; i < _facet_1.size(); i++)
    {
      auto facet = _facet_1[i];
      auto cell = f_to_c->links(facet)[0];
      auto cell_dofs = dofmap->cell_dofs(cell);
      std::int32_t num_links = _map_1_to_0->num_links(i);
      auto links = _map_1_to_0->links(i);
      std::vector<std::int32_t> linked_dofs;
      for (std::int32_t j = 0; j < num_links; j++)
      {
        auto linked_facet = links[j];
        auto linked_cell = f_to_c->links(linked_facet)[0];
        auto linked_cell_dofs = dofmap->cell_dofs(linked_cell);
        for (std::int32_t k = 0; k < linked_cell_dofs.size(); k++)
        {
          linked_dofs.push_back(linked_cell_dofs[k]);
        }
      }

      pattern.insert(cell_dofs, linked_dofs);
      pattern.insert(linked_dofs, cell_dofs);
    }

    // Finalise communication
    pattern.assemble();
    std::cout << "nnz " << pattern.num_nonzeros() << "\n";

    return dolfinx::la::create_petsc_matrix(a.mesh()->mpi_comm(), pattern,
                                            type);
  }

  /// Assemble matrix over exterior facets
  /// Provides easier interface to dolfinx::fem::impl::assemble_exterior_facets
  /// @param[in] mat_set the function for setting the values in the matrix
  /// @param[in] V the function space
  /// @param[in] active_facets list of indices (local to process) of facets to
  /// be integrated over
  /// @param[in] kernel the custom integration kernel
  /// @param[in] coeffs coefficients used in the variational form packed on
  /// facets
  /// @param[in] cstride Number of coefficients per facet
  /// @param[in] constants used in the variational form
  void assemble_matrix(
      const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                              const std::int32_t*, const PetscScalar*)>&
          mat_set,
      const std::vector<
          std::shared_ptr<const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
      int origin_meshtag, contact_kernel_fn& kernel,
      const xtl::span<const PetscScalar> coeffs, int cstride,
      const xtl::span<const PetscScalar>& constants)
  {
    auto mesh = _marker->mesh();
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim(); // topological dimension
    const int fdim = tdim - 1;               // topological dimension of facet

    mesh->topology_mutable().create_connectivity(fdim, tdim);
    auto f_to_c = mesh->topology().connectivity(fdim, tdim);
    mesh->topology_mutable().create_connectivity(tdim, fdim);
    auto c_to_f = mesh->topology().connectivity(tdim, fdim);

    // Prepare cell geometry
    const graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh->geometry().dofmap();

    // FIXME: Add proper interface for num coordinate dofs
    const std::size_t num_dofs_g = x_dofmap.num_links(0);
    const xt::xtensor<double, 2>& x_g = mesh->geometry().x();

    // Extract function space data (assuming same test and trial space)
    std::shared_ptr<const dolfinx::fem::DofMap> dofmap = _V->dofmap();
    const std::int32_t ndofs_cell = dofmap->cell_dofs(0).size();
    const dolfinx::graph::AdjacencyList<std::int32_t>& dofs = dofmap->list();
    const int bs = dofmap->bs();

    // FIXME: Need to reconsider facet permutations for jump integrals
    std::uint8_t perm = 0;
    // Select which side of the contact interface to loop from and get the
    // correct map
    std::vector<int32_t>* active_facets;
    std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> map;
    std::size_t max_links = 0;
    if (origin_meshtag == 0)
    {
      active_facets = &_facet_0;
      map = _map_0_to_1;
      max_links = _max_links_0;
    }
    else
    {
      active_facets = &_facet_1;
      map = _map_1_to_0;
      max_links = _max_links_1;
    }

    // Data structures used in assembly
    std::vector<double> coordinate_dofs(3 * num_dofs_g);
    std::vector<std::int32_t> dmapjoint;
    std::vector<std::vector<PetscScalar>> Ae_vec(
        2 * max_links + 1,
        std::vector<PetscScalar>(bs * ndofs_cell * bs * ndofs_cell));
    for (int i = 0; i < (*active_facets).size(); i++)
    {
      int facet = (*active_facets)[i]; // extract facet
      auto cells = f_to_c->links(facet);
      // since the facet is on the boundary it should only link to one cell
      assert(cells.size() == 1);
      auto cell = cells[0]; // extract cell
      // Get cell coordinates/geometry
      auto x_dofs = x_dofmap.links(cell);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                    std::next(coordinate_dofs.begin(), 3 * i));
      }
      // find local index of facet
      auto facets = c_to_f->links(cell);
      auto local_facet = std::find(facets.begin(), facets.end(), facet);
      const std::int32_t local_index
          = std::distance(facets.data(), local_facet);
      std::vector<std::int32_t> linked_cells;
      std::int32_t num_links = map->num_links(i);
      auto links = map->links(i);
      for (std::int32_t j = 0; j < num_links; j++)
      {
        auto linked_facet = links[j];
        linked_cells.push_back(f_to_c->links(linked_facet)[0]);
      }
      // Remove duplicates
      std::sort(linked_cells.begin(), linked_cells.end());
      linked_cells.erase(std::unique(linked_cells.begin(), linked_cells.end()),
                         linked_cells.end());
      const int num_linked_cells = linked_cells.size();

      std::fill(Ae_vec[0].begin(), Ae_vec[0].end(), 0);
      for (std::int32_t j = 0; j < num_linked_cells; j++)
      {
        std::fill(Ae_vec[2 * j + 1].begin(), Ae_vec[2 * j + 1].end(), 0);
        std::fill(Ae_vec[2 * (j + 1)].begin(), Ae_vec[2 * (j + 1)].end(), 0);
      }

      kernel(Ae_vec, coeffs.data() + i * cstride, constants.data(),
             coordinate_dofs.data(), &local_index, &perm, num_linked_cells);

      auto dmap_cell = dofmap->cell_dofs(cell);
      mat_set(dmap_cell.size(), dmap_cell.data(), dmap_cell.size(),
              dmap_cell.data(), Ae_vec[0].data());

      for (std::int32_t j = 0; j < num_linked_cells; j++)
      {
        auto dmap_linked = dofmap->cell_dofs(linked_cells[j]);
        mat_set(dmap_cell.size(), dmap_cell.data(), dmap_linked.size(),
                dmap_linked.data(), Ae_vec[2 * j + 1].data());
        mat_set(dmap_linked.size(), dmap_linked.data(), dmap_cell.size(),
                dmap_cell.data(), Ae_vec[2 * (j + 1)].data());
      }
    }
  }

  contact_kernel_fn generate_kernel()
  {
    // mesh data
    auto mesh = _marker->mesh();
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim(); // topological dimension
    const int fdim = tdim - 1;
    // Extract function space data (assuming same test and trial space)
    std::shared_ptr<const dolfinx::fem::DofMap> dofmap = _V->dofmap();
    const std::int32_t ndofs_cell = dofmap->cell_dofs(0).size();
    const int bs = dofmap->bs();

    contact_kernel_fn unbiased_jac
        = [ndofs_cell, bs](std::vector<std::vector<PetscScalar>>& A,
                           const double* c, const double* w,
                           const double* coordinate_dofs,
                           const int* entity_local_index,
                           const std::uint8_t* quadrature_permutation,
                           const std::int32_t num_links)
    {
      // Fill contributions of facet with itself
      for (int j = 0; j < ndofs_cell; j++)
      {
        for (int l = 0; l < bs; l++)
        {
          for (int i = 0; i < ndofs_cell; i++)
          {
            for (int b = 0; b < bs; b++)
            {
              A[0][(b + i * bs) * ndofs_cell * bs + l + j * bs] += 1;
            }
          }
        }
      }
      // Fill contributions of facet with contact facet
      for (int k = 0; k < num_links; k++)
      {
        for (int j = 0; j < ndofs_cell; j++)
        {
          for (int l = 0; l < bs; l++)
          {
            for (int i = 0; i < ndofs_cell; i++)
            {
              for (int b = 0; b < bs; b++)
              {
                A[2 * k + 1][(b + i * bs) * ndofs_cell * bs + l + j * bs] += 1;
                A[2 * (k + 1)][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                    += 1;
              }
            }
          }
        }
      }
    };
    return unbiased_jac;
  }
  /// Tabulate the basis function at the quadrature points _qp_ref_facet
  /// creates and fills _phi_ref_facets
  std::vector<xt::xtensor<double, 2>>
  tabulate_on_ref_cell(basix::FiniteElement element)
  {

    // Create _phi_ref_facets
    std::uint32_t num_facets = _qp_ref_facet.size();
    std::uint32_t num_local_dofs = element.dim();
    std::vector<xt::xtensor<double, 2>> phi;
    phi.reserve(num_facets);

    // Tabulate basis functions at quadrature points _qp_ref_facet for each
    // facet of the reference cell. Fill _phi_ref_facets
    for (int i = 0; i < num_facets; ++i)
    {
      auto cell_tab = element.tabulate(0, _qp_ref_facet[i]);
      const xt::xtensor<double, 2> _phi_i
          = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
      phi.push_back(_phi_i);
    }
    return phi;
  }

  /// Compute push forward of quadrature points _qp_ref_facet to the physical
  /// facet for each facet in _facet_"origin_meshtag" Creates and fills
  /// _qp_phys_"origin_meshtag"
  /// @param[in] origin_meshtag flag to choose between surface_0 and surface_1
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

    std::vector<std::int32_t>* puppet_facets;
    std::vector<xt::xtensor<double, 2>>* q_phys_pt;
    if (origin_meshtag == 0)
    {
      puppet_facets = &_facet_0;
      _qp_phys_0.reserve(_facet_0.size());
      q_phys_pt = &_qp_phys_0;
    }
    else
    {
      puppet_facets = &_facet_1;
      _qp_phys_1.reserve(_facet_1.size());
      q_phys_pt = &_qp_phys_1;
    }
    q_phys_pt->clear();
    // push forward of quadrature points _qp_ref_facet to physical facet for
    // each facet in _facet_"origin_meshtag"
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
      const std::int32_t local_index
          = std::distance(facets.data(), local_facet);

      // extract local dofs
      auto x_dofs = x_dofmap.links(cell);
      auto coordinate_dofs
          = xt::view(mesh_geometry, xt::keep(x_dofs), xt::range(0, gdim));
      xt::xtensor<double, 2> q_phys({_qp_ref_facet[local_index].shape(0),
                                     _qp_ref_facet[local_index].shape(1)});

      // push forward of quadrature points _qp_ref_facet to the physical facet
      cmap.push_forward(q_phys, coordinate_dofs, _phi_ref_facets[local_index]);
      q_phys_pt->push_back(q_phys);
    }
  }
  /// Compute maximum number of links
  /// I think this should actually be part of create_distance_map
  /// which should be easier after the rewrite of contact
  /// It is therefore called inside create_distance_map
  void max_links(int origin_meshtag)
  {
    // facet to cell connectivity
    auto mesh = _marker->mesh();
    auto dofmap = _V->dofmap();
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim(); // topological dimension
    const int fdim = tdim - 1;               // topological dimension of facet

    mesh->topology_mutable().create_connectivity(fdim, tdim);
    auto f_to_c = mesh->topology().connectivity(fdim, tdim);

    std::size_t max_links = 0;
    // Select which side of the contact interface to loop from and get the
    // correct map
    std::vector<int32_t>* active_facets;
    std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> map;
    if (origin_meshtag == 0)
    {
      active_facets = &_facet_0;
      map = _map_0_to_1;
    }
    else
    {
      active_facets = &_facet_1;
      map = _map_1_to_0;
    }

    for (std::int32_t i = 0; i < (*active_facets).size(); i++)
    {
      auto facet = (*active_facets)[i];
      auto cell = f_to_c->links(facet)[0];
      auto cell_dofs = dofmap->cell_dofs(cell);
      std::int32_t num_links = map->num_links(i);
      auto links = map->links(i);
      std::vector<std::int32_t> linked_dofs;
      for (std::int32_t j = 0; j < num_links; j++)
      {
        auto linked_facet = links[j];
        auto linked_cell = f_to_c->links(linked_facet)[0];
        auto linked_cell_dofs = dofmap->cell_dofs(linked_cell);
        for (std::int32_t k = 0; k < linked_cell_dofs.size(); k++)
        {
          linked_dofs.push_back(linked_cell_dofs[k]);
        }
      }
      // Remove duplicates
      std::sort(linked_dofs.begin(), linked_dofs.end());
      linked_dofs.erase(std::unique(linked_dofs.begin(), linked_dofs.end()),
                        linked_dofs.end());
      max_links = std::max(max_links, linked_dofs.size());
    }
    if (origin_meshtag == 0)
      _max_links_0 = max_links;
    else
      _max_links_1 = max_links;
  }
  /// Compute closest candidate_facet for each quadrature point in
  /// _qp_phys_"origin_meshtag" This is saved as an adjacency list _map_0_to_1
  /// or _map_1_to_0
  void create_distance_map(int origin_meshtag)
  {

    // Mesh info
    auto mesh = _marker->mesh();
    const int gdim = mesh->geometry().dim();
    const int tdim = mesh->topology().dim();
    const int fdim = tdim - 1;

    // Create _qp_ref_facet (quadrature points on reference facet)
    dolfinx_cuas::QuadratureRule q_rule(mesh->topology().cell_type(),
                                        _quadrature_degree, fdim);
    _qp_ref_facet = q_rule.points();

    // Tabulate basis function on reference cell (_phi_ref_facets)// Create
    // coordinate element
    // FIXME: For higher order geometry need basix element public in mesh
    // auto degree = mesh->geometry().cmap()._element->degree;
    int degree = 1;
    auto dolfinx_cell = mesh->topology().cell_type();
    auto coordinate_element = basix::create_element(
        basix::element::family::P,
        dolfinx::mesh::cell_type_to_basix_type(dolfinx_cell), degree,
        basix::element::lagrange_variant::equispaced);
    _phi_ref_facets = tabulate_on_ref_cell(coordinate_element);

    // Compute quadrature points on physical facet _qp_phys_"origin_meshtag"
    create_q_phys(origin_meshtag);

    std::array<double, 3> point;
    point[2] = 0;

    // assign puppet_ and candidate_facets
    std::vector<int32_t>* candidate_facets;
    std::vector<int32_t>* puppet_facets;
    std::vector<xt::xtensor<double, 2>>* q_phys_pt;
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
    // Create midpoint tree as compute_closest_entity will be called many
    // times
    dolfinx::geometry::BoundingBoxTree master_bbox(*mesh, fdim,
                                                   (*candidate_facets));
    auto master_midpoint_tree = dolfinx::geometry::create_midpoint_tree(
        *mesh, fdim, (*candidate_facets));

    std::vector<std::int32_t> data; // will contain closest candidate facet
    std::vector<std::int32_t> offset(1);
    offset[0] = 0;
    for (int i = 0; i < (*puppet_facets).size(); ++i)
    {
      // FIXME: This does not work for prism meshes
      for (int j = 0; j < (*q_phys_pt)[0].shape(0); ++j)
      {
        for (int k = 0; k < gdim; ++k)
          point[k] = (*q_phys_pt)[i](j, k);
        // Find initial search radius R = intermediate_result.second
        std::pair<int, double> intermediate_result
            = dolfinx::geometry::compute_closest_entity(master_midpoint_tree,
                                                        point, *mesh);
        // Find closest facet to point
        std::pair<int, double> search_result
            = dolfinx::geometry::compute_closest_entity(
                master_bbox, point, *mesh, intermediate_result.second);
        data.push_back(search_result.first);
      }
      offset.push_back(data.size());
    }

    // save as an adjacency list _map_0_to_1 or _map_1_to_0
    if (origin_meshtag == 0)
      _map_0_to_1
          = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
              data, offset);
    else
      _map_1_to_0
          = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
              data, offset);
    max_links(origin_meshtag);
  }

  /// Compute and pack the gap function for each quadrature point the set of
  /// facets. For a set of facets; go through the quadrature points on each
  /// facet find the closest facet on the other surface and compute the
  /// distance vector
  /// @param[in] orgin_meshtag - surface on which to integrate
  /// @param[out] c - gap packed on facets. c[i*cstride +  gdim * k+ j]
  /// contains the jth component of the Gap on the ith facet at kth quadrature
  /// point
  std::pair<std::vector<PetscScalar>, int> pack_gap(int origin_meshtag)
  {
    // Mesh info
    auto mesh = _marker->mesh();             // mesh
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim();
    const int fdim = tdim - 1;
    const xt::xtensor<double, 2>& mesh_geometry = mesh->geometry().x();
    std::vector<int32_t>* puppet_facets;
    std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> map;
    std::vector<xt::xtensor<double, 2>>* q_phys_pt;

    // Select which side of the contact interface to loop from and get the
    // correct map
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
    const std::int32_t num_q_point = _qp_ref_facet[0].shape(1);

    // Pack gap function for each quadrature point on each facet
    std::vector<PetscScalar> c(num_facets * num_q_point * gdim, 0.0);
    const int cstride = num_q_point * gdim;
    xt::xtensor<double, 2> point = {{0, 0, 0}};

    for (int i = 0; i < num_facets; ++i)
    {
      auto master_facets = map->links(i);
      auto master_facet_geometry = dolfinx::mesh::entities_to_geometry(
          *mesh, fdim, master_facets, false);
      int offset = i * cstride;
      for (int j = 0; j < map->num_links(i); ++j)
      {
        // Get quadrature points in physical space for the ith facet, jth
        // quadrature point
        for (int k = 0; k < gdim; k++)
          point(0, k) = (*q_phys_pt)[i](j, k);

        // Get the coordinates of the geometry on the other interface, and
        // compute the distance of the convex hull created by the points
        auto master_facet = xt::view(master_facet_geometry, j, xt::all());
        auto master_coords
            = xt::view(mesh_geometry, xt::keep(master_facet), xt::all());
        auto dist_vec
            = dolfinx::geometry::compute_distance_gjk(master_coords, point);

        // Add distance vector to coefficient array
        for (int k = 0; k < gdim; k++)
          c[offset + j * gdim + k] -= dist_vec(k);
      }
    }
    return {std::move(c), cstride};
  }

  /// Compute test functions on opposite surface at quadrature points of
  /// facets
  /// @param[in] orgin_meshtag - surface on which to integrate
  /// @param[out] c - test functions packed on facets.
  std::pair<std::vector<PetscScalar>, int>
  pack_test_functions(int origin_meshtag)
  {
    // Mesh info
    auto mesh = _marker->mesh();             // mesh
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim();
    const int fdim = tdim - 1;
    const xt::xtensor<double, 2>& mesh_geometry = mesh->geometry().x();
    std::vector<int32_t>* puppet_facets;
    std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> map;
    std::vector<xt::xtensor<double, 2>>* q_phys_pt;
    std::size_t max_links = 0;

    // Select which side of the contact interface to loop from and get the
    // correct map
    if (origin_meshtag == 0)
    {
      puppet_facets = &_facet_0;
      map = _map_0_to_1;
      q_phys_pt = &_qp_phys_0;
      max_links = _max_links_0;
    }
    else
    {
      puppet_facets = &_facet_1;
      map = _map_1_to_0;
      q_phys_pt = &_qp_phys_1;
      max_links = _max_links_1;
    }
    const std::int32_t num_facets = (*puppet_facets).size();
    const std::int32_t num_q_point = _qp_ref_facet[0].shape(1);
    const std::int32_t ndofs = _V->dofmap()->cell_dofs(0).size();
    std::vector<PetscScalar> c(num_facets * num_q_point * max_links * ndofs,
                               0.0);
    const int cstride = num_q_point * max_links * ndofs;

    return {std::move(c), cstride};
  }

  /// Pack gap with rigid surface defined by x[gdim-1] = -g.
  /// g_vec = zeros(gdim), g_vec[gdim-1] = -g
  /// Gap = x - g_vec
  /// @param[in] orgin_meshtag - surface on which to integrate
  /// @param[in] g - defines location of plane
  /// @param[out] c - gap packed on facets. c[i, gdim * k+ j] contains the
  /// jth component of the Gap on the ith facet at kth quadrature point
  std::pair<std::vector<PetscScalar>, int> pack_gap_plane(int origin_meshtag,
                                                          double g)
  {
    // Mesh info
    auto mesh = _marker->mesh();             // mesh
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim();
    const int fdim = tdim - 1;
    auto mesh_geometry = mesh->geometry().x();
    // Create _qp_ref_facet (quadrature points on reference facet)
    dolfinx_cuas::QuadratureRule facet_quadrature(
        _marker->mesh()->topology().cell_type(), _quadrature_degree, fdim);
    _qp_ref_facet = facet_quadrature.points();
    _qw_ref_facet = facet_quadrature.weights();

    // Tabulate basis function on reference cell (_phi_ref_facets)// Create
    // coordinate element
    // FIXME: For higher order geometry need basix element public in mesh
    // auto degree = mesh->geometry().cmap()._element->degree;
    int degree = 1;
    auto dolfinx_cell = _marker->mesh()->topology().cell_type();
    auto coordinate_element = basix::create_element(
        basix::element::family::P,
        dolfinx::mesh::cell_type_to_basix_type(dolfinx_cell), degree,
        basix::element::lagrange_variant::equispaced);

    _phi_ref_facets = tabulate_on_ref_cell(coordinate_element);
    // Compute quadrature points on physical facet _qp_phys_"origin_meshtag"
    create_q_phys(origin_meshtag);
    std::vector<int32_t>* puppet_facets;
    std::vector<xt::xtensor<double, 2>>* q_phys_pt;
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
    // FIXME: This does not work for prism meshes
    int32_t num_q_point = _qp_ref_facet[0].shape(0);
    std::vector<PetscScalar> c(num_facets * num_q_point * gdim, 0.0);
    const int cstride = num_q_point * gdim;
    for (int i = 0; i < num_facets; i++)
    {
      int offset = i * cstride;
      for (int k = 0; k < num_q_point; k++)
      {
        for (int j = 0; j < gdim; j++)
        {
          c[offset + k * gdim + j] += (*q_phys_pt)[i](k, j);
        }
        c[offset + (k + 1) * gdim - 1] += g;
      }
    }
    return {std::move(c), cstride};
  }

private:
  int _quadrature_degree = 3;
  std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> _marker;
  int _surface_0; // meshtag value for surface 0
  int _surface_1; // meshtag value for surface 1
  std::shared_ptr<dolfinx::fem::FunctionSpace> _V; // Function space

  // Adjacency list of closest facet on surface_1 for every quadrature point
  // in _qp_phys_0 (quadrature points on every facet of surface_0)
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _map_0_to_1;
  // Adjacency list of closest facet on surface_0 for every quadrature point
  // in _qp_phys_1 (quadrature points on every facet of surface_1)
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _map_1_to_0;
  // quadrature points on physical facet for each facet on surface 0
  std::vector<xt::xtensor<double, 2>> _qp_phys_0;
  // quadrature points on physical facet for each facet on surface 1
  std::vector<xt::xtensor<double, 2>> _qp_phys_1;
  // quadrature points on reference facet
  std::vector<xt::xarray<double>> _qp_ref_facet;
  // quadrature weights
  std::vector<std::vector<double>> _qw_ref_facet;
  // quadrature points on facets of reference cell
  std::vector<xt::xtensor<double, 2>> _phi_ref_facets;
  // facets in surface 0
  std::vector<int32_t> _facet_0;
  // facets in surface 1
  std::vector<int32_t> _facet_1;
  // normals on surface 0 in order of facets in _facet_0
  xt::xtensor<double, 2> _normals_0;
  // normals on surface 1 in order of facets in _facet_1
  xt::xtensor<double, 2> _normals_1;
  // maximum number of cells linked to a cell on surface_0
  std::size_t _max_links_0 = 0;
  // maximum number of cells linked to a cell on surface_1
  std::size_t _max_links_1 = 0;
};
} // namespace dolfinx_contact
