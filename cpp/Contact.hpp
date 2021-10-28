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
#include <dolfinx/common/sort.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/dolfin_la.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx_contact/utils.hpp>
#include <dolfinx_cuas/QuadratureRule.hpp>
#include <dolfinx_cuas/math.hpp>
#include <dolfinx_cuas/utils.hpp>
#include <iostream>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtl/xspan.hpp>

using contact_kernel_fn = std::function<void(
    std::vector<std::vector<PetscScalar>>&, const double*, const double*,
    const double*, const int*, const std::uint8_t*, const std::int32_t)>;
namespace dolfinx_contact
{
enum Kernel
{
  Rhs,
  Jac
};
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
    auto mesh = _marker->mesh();
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim(); // topological dimension
    const int fdim = tdim - 1;               // topological dimension of facet
    const dolfinx::mesh::Topology& topology = mesh->topology();
    auto f_to_c = mesh->topology().connectivity(tdim - 1, tdim);
    assert(f_to_c);
    auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
    assert(c_to_f);
    _facet_0 = marker->find(surface_0);
    _facet_1 = marker->find(surface_1);

    auto get_cell_indices = [c_to_f, f_to_c](std::vector<std::int32_t> facets)
    {
      int num_facets = facets.size();
      xt::xtensor<std::int32_t, 2> cell_facet_pairs
          = xt::zeros<std::int32_t>({num_facets, 2});
      for (int i = 0; i < num_facets; i++)
      {
        std::int32_t facet = facets[i];
        auto cells = f_to_c->links(facet);
        const std::int32_t cell = cells[0];
        // Find local facet index
        auto local_facets = c_to_f->links(cell);
        const auto it
            = std::find(local_facets.begin(), local_facets.end(), facet);
        assert(it != local_facets.end());
        const int facet_index = std::distance(local_facets.begin(), it);
        cell_facet_pairs(i, 0) = cell;
        cell_facet_pairs(i, 1) = facet_index;
      }
      return cell_facet_pairs;
    };

    _cell_facet_pairs_0 = get_cell_indices(_facet_0);
    _cell_facet_pairs_1 = get_cell_indices(_facet_1);
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

    // assumes same number of quadrature points on each facet
    const std::int32_t num_qp = _map_0_to_1.shape(1);
    for (std::int32_t i = 0; i < _cell_facet_pairs_0.shape(0); i++)
    {
      auto cell = _cell_facet_pairs_0(i, 0);
      auto cell_dofs = dofmap->cell_dofs(cell);
      std::vector<std::int32_t> linked_dofs;
      for (std::int32_t j = 0; j < num_qp; j++)
      {
        auto linked_cell = _map_0_to_1(i, j, 0);
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

    for (std::int32_t i = 0; i < _cell_facet_pairs_1.shape(0); i++)
    {
      auto cell = _cell_facet_pairs_1(i, 0);
      auto cell_dofs = dofmap->cell_dofs(cell);
      std::vector<std::int32_t> linked_dofs;
      for (std::int32_t j = 0; j < num_qp; j++)
      {
        auto linked_cell = _map_1_to_0(i, j, 0);
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
    xt::xtensor<std::int32_t, 2>* active_facets;
    xt::xtensor<std::int32_t, 3>* map;
    std::size_t max_links = 0;
    if (origin_meshtag == 0)
    {
      active_facets = &_cell_facet_pairs_0;
      map = &_map_0_to_1;
      max_links = _max_links_0;
    }
    else
    {
      active_facets = &_cell_facet_pairs_1;
      map = &_map_1_to_0;
      max_links = _max_links_1;
    }

    // Data structures used in assembly
    std::vector<double> coordinate_dofs(3 * num_dofs_g);
    std::vector<std::int32_t> dmapjoint;
    std::vector<std::vector<PetscScalar>> Ae_vec(
        2 * max_links + 1,
        std::vector<PetscScalar>(bs * ndofs_cell * bs * ndofs_cell));
    for (int i = 0; i < (*active_facets).shape(0); i++)
    {
      auto cell = (*active_facets)(i, 0);
      // Get cell coordinates/geometry
      auto x_dofs = x_dofmap.links(cell);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                    std::next(coordinate_dofs.begin(), 3 * i));
      }
      const std::int32_t local_index = (*active_facets)(i, 1);
      std::vector<std::int32_t> linked_cells;
      std::int32_t num_qp = (*map).shape(1);
      for (std::int32_t j = 0; j < num_qp; j++)
      {
        linked_cells.push_back((*map)(i, j, 0));
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

  contact_kernel_fn generate_kernel(int origin_meshtag,
                                    dolfinx_contact::Kernel type)
  {
    // mesh data
    auto mesh = _marker->mesh();
    const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
    const std::size_t tdim = mesh->topology().dim(); // topological dimension
    const std::size_t fdim = tdim - 1;
    // Extract function space data (assuming same test and trial space)
    std::shared_ptr<const dolfinx::fem::DofMap> dofmap = _V->dofmap();
    const std::size_t ndofs_cell = dofmap->cell_dofs(0).size();
    const std::size_t bs = dofmap->bs();

    // Select which side of the contact interface to loop from and get the
    // correct map
    xt::xtensor<std::int32_t, 2>* active_facets;
    xt::xtensor<std::int32_t, 3>* map;
    std::size_t max_links = 0;
    if (origin_meshtag == 0)
    {
      active_facets = &_cell_facet_pairs_0;
      map = &_map_0_to_1;
      max_links = _max_links_0;
    }
    else
    {
      active_facets = &_cell_facet_pairs_1;
      map = &_map_1_to_0;
      max_links = _max_links_1;
    }

    const std::size_t num_q_points = _qp_ref_facet[0].shape(0);

    // coefficient offsets
    // expecting coefficients in following order:
    // mu, lmbda, h, gap, test_fn, u, u_opposite
    // packed at quadrature points
    // mu, lmbda, h scalar
    // gap vector valued gdim
    // test_fn, u, u_opposite vector valued bs (should be bs = gdim)
    const std::size_t num_coeffs = 7;
    std::vector<std::size_t> cstrides
        = {num_q_points,
           num_q_points,
           num_q_points,
           num_q_points * gdim,
           num_q_points * ndofs_cell * bs * max_links,
           num_q_points * bs,
           num_q_points * bs};

    contact_kernel_fn unbiased_jac
        = [ndofs_cell, bs, cstrides](std::vector<std::vector<PetscScalar>>& A,
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
    switch (type)
    {
    case dolfinx_contact::Kernel::Jac:
      return unbiased_jac;
    default:
      throw std::runtime_error("Unrecognized kernel");
    }
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
    xt::xtensor<std::int32_t, 2>* puppet_facets;
    std::vector<xt::xtensor<double, 2>>* q_phys_pt;
    if (origin_meshtag == 0)
    {
      puppet_facets = &_cell_facet_pairs_0;
      _qp_phys_0.reserve(_cell_facet_pairs_0.shape(0));
      q_phys_pt = &_qp_phys_0;
    }
    else
    {
      puppet_facets = &_cell_facet_pairs_1;
      _qp_phys_1.reserve(_cell_facet_pairs_1.shape(0));
      q_phys_pt = &_qp_phys_1;
    }
    q_phys_pt->clear();
    // push forward of quadrature points _qp_ref_facet to physical facet for
    // each facet in _facet_"origin_meshtag"
    for (int i = 0; i < (*puppet_facets).shape(0); ++i)
    {
      auto cell = (*puppet_facets)(i, 0); // extract cell
      const std::int32_t local_index = (*puppet_facets)(i, 1);

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

    std::size_t max_links = 0;
    // Select which side of the contact interface to loop from and get the
    // correct map
    xt::xtensor<std::int32_t, 2>* active_facets;
    xt::xtensor<std::int32_t, 3>* map;
    if (origin_meshtag == 0)
    {
      active_facets = &_cell_facet_pairs_0;
      map = &_map_0_to_1;
    }
    else
    {
      active_facets = &_cell_facet_pairs_1;
      map = &_map_1_to_0;
    }

    std::int32_t num_qp = (*map).shape(1);
    for (std::int32_t i = 0; i < (*active_facets).shape(0); i++)
    {
      auto cell = (*active_facets)(i, 0);
      auto cell_dofs = dofmap->cell_dofs(cell);
      std::vector<std::int32_t> linked_cells;
      for (std::int32_t j = 0; j < num_qp; j++)
      {
        linked_cells.push_back((*map)(i, j, 0));
      }
      // Remove duplicates
      std::sort(linked_cells.begin(), linked_cells.end());
      linked_cells.erase(std::unique(linked_cells.begin(), linked_cells.end()),
                         linked_cells.end());
      max_links = std::max(max_links, linked_cells.size());
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

    mesh->topology_mutable().create_connectivity(fdim, tdim);
    auto f_to_c = mesh->topology().connectivity(fdim, tdim);
    mesh->topology_mutable().create_connectivity(tdim, fdim);
    auto c_to_f = mesh->topology().connectivity(tdim, fdim);
    auto get_cell_indices = [c_to_f, f_to_c](std::vector<std::int32_t> facets)
    {
      int num_facets = facets.size();
      xt::xtensor<std::int32_t, 2> cell_facet_pairs
          = xt::zeros<std::int32_t>({num_facets, 2});
      for (int i = 0; i < num_facets; i++)
      {
        std::int32_t facet = facets[i];
        auto cells = f_to_c->links(facet);
        const std::int32_t cell = cells[0];
        // Find local facet index
        auto local_facets = c_to_f->links(cell);
        const auto it
            = std::find(local_facets.begin(), local_facets.end(), facet);
        assert(it != local_facets.end());
        const int facet_index = std::distance(local_facets.begin(), it);
        cell_facet_pairs(i, 0) = cell;
        cell_facet_pairs(i, 1) = facet_index;
      }
      return cell_facet_pairs;
    };
    const int num_qp = (*q_phys_pt)[0].shape(0);
    const int num_facets = (*puppet_facets).size();
    std::vector<std::int32_t> data(
        num_qp); // will contain closest candidate facet
    xt::xtensor<std::int32_t, 3> map
        = xt::zeros<std::int32_t>({num_facets, num_qp, 2});
    xt::xtensor<std::int32_t, 2> old_map
        = xt::zeros<std::int32_t>({num_facets, num_qp});
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
        data[j] = search_result.first;
        old_map(i, j) = search_result.first;
      }

      xt::view(map, i, xt::all(), xt::all()) = get_cell_indices(data);
    }

    // save as an adjacency list _map_0_to_1 or _map_1_to_0
    if (origin_meshtag == 0)
    {
      _map_0_to_1 = map;
      _map_0_to_1_facet = old_map;
    }
    else
    {
      _map_1_to_0 = map;
      _map_1_to_0_facet = old_map;
    }
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
    xt::xtensor<std::int32_t, 2>* map;
    std::vector<xt::xtensor<double, 2>>* q_phys_pt;

    // Select which side of the contact interface to loop from and get the
    // correct map
    if (origin_meshtag == 0)
    {
      puppet_facets = &_facet_0;
      map = &_map_0_to_1_facet;
      q_phys_pt = &_qp_phys_0;
    }
    else
    {
      puppet_facets = &_facet_1;
      map = &_map_1_to_0_facet;
      q_phys_pt = &_qp_phys_1;
    }
    const std::int32_t num_facets = (*puppet_facets).size();
    const std::int32_t num_q_point = _qp_ref_facet[0].shape(0);

    // Pack gap function for each quadrature point on each facet
    std::vector<PetscScalar> c(num_facets * num_q_point * gdim, 0.0);
    const int cstride = num_q_point * gdim;
    xt::xtensor<double, 2> point = {{0, 0, 0}};

    for (int i = 0; i < num_facets; ++i)
    {
      auto master_facets = xt::view((*map), i, xt::all());
      auto master_facet_geometry = dolfinx::mesh::entities_to_geometry(
          *mesh, fdim, master_facets, false);
      int offset = i * cstride;
      for (int j = 0; j < (*map).shape(1); ++j)
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
  /// @param[in] - gap packed on facets per quadrature point
  /// @param[out] c - test functions packed on facets.
  std::pair<std::vector<PetscScalar>, int>
  pack_test_functions(int origin_meshtag,
                      const xtl::span<const PetscScalar> gap)
  {
    // Mesh info
    auto mesh = _marker->mesh();             // mesh
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim();
    const int fdim = tdim - 1;
    auto cmap = mesh->geometry().cmap();
    auto x_dofmap = mesh->geometry().dofmap();
    const xt::xtensor<double, 2>& mesh_geometry = mesh->geometry().x();
    xt::xtensor<std::int32_t, 3>* map;
    std::vector<xt::xtensor<double, 2>>* q_phys_pt;
    xt::xtensor<std::int32_t, 2>* puppet_facets;
    std::size_t max_links = 0;
    auto element = _V->element();
    const std::uint32_t bs = element->block_size();
    mesh->topology_mutable().create_entity_permutations();

    const std::vector<std::uint32_t> permutation_info
        = mesh->topology().get_cell_permutation_info();

    // Select which side of the contact interface to loop from and get the
    // correct map
    if (origin_meshtag == 0)
    {
      map = &_map_0_to_1;
      q_phys_pt = &_qp_phys_0;
      max_links = _max_links_0;
      puppet_facets = &_cell_facet_pairs_0;
    }
    else
    {
      map = &_map_1_to_0;
      q_phys_pt = &_qp_phys_1;
      max_links = _max_links_1;
      puppet_facets = &_cell_facet_pairs_1;
    }
    const std::int32_t num_facets = (*map).shape(0);
    const std::int32_t num_q_points = _qp_ref_facet[0].shape(0);
    const std::int32_t ndofs = _V->dofmap()->cell_dofs(0).size();
    std::vector<PetscScalar> c(
        num_facets * num_q_points * max_links * ndofs * bs, 0.0);
    const int cstride = num_q_points * max_links * ndofs * bs;
    std::vector<std::int32_t> perm(num_q_points);
    std::vector<std::int32_t> sorted_cells(num_q_points);
    xt::xtensor<double, 2> q_points
        = xt::zeros<double>({std::size_t(num_q_points), std::size_t(gdim)});
    xt::xtensor<double, 2> dphi;
    xt::xtensor<double, 3> J = xt::zeros<double>(
        {std::size_t(1), std::size_t(gdim), std::size_t(tdim)});
    xt::xtensor<double, 3> K = xt::zeros<double>(
        {std::size_t(1), std::size_t(tdim), std::size_t(gdim)});
    xt::xtensor<double, 1> detJ = xt::zeros<double>({1});
    xt::xtensor<double, 4> phi(cmap.tabulate_shape(1, 1));
    for (int i = 0; i < num_facets; i++)
    {
      auto cell = (*puppet_facets)(i, 0); // extract cell

      for (int j = 0; j < num_q_points; j++)
      {
        sorted_cells[j] = (*map)(i, j, 0);
        // Is there a better way to do this???
        for (int k = 0; k < gdim; k++)
        {
          q_points(j, k) = (*q_phys_pt)[i](j, k)
                           - gap[i * gdim * num_q_points + j * gdim + k];
        }
      }
      std::iota(perm.begin(), perm.end(), 0);
      dolfinx::argsort_radix<std::int32_t>(sorted_cells, perm);
      std::sort(sorted_cells.begin(), sorted_cells.end());
      auto it = sorted_cells.begin();
      int link = 0;
      while (it != sorted_cells.end())
      {
        auto upper = std::upper_bound(it, sorted_cells.end(), *it);
        int num_indices
            = upper - sorted_cells.begin() - (it - sorted_cells.begin());
        std::vector<std::int32_t> indices(num_indices, 0);
        for (int k = it - sorted_cells.begin();
             k < upper - sorted_cells.begin(); k++)
        {
          int l = it - sorted_cells.begin();
          indices[k - l] = perm[k];
        }
        int linked_cell = *it;
        // extract local dofs
        auto x_dofs = x_dofmap.links(linked_cell);
        const xt::xtensor<double, 2> coordinate_dofs
            = xt::view(mesh_geometry, xt::keep(x_dofs), xt::range(0, gdim));
        auto qp = xt::view(q_points, xt::keep(indices), xt::all());
        auto test_fn = dolfinx_contact::get_basis_functions(
            J, K, detJ, qp, coordinate_dofs, linked_cell,
            permutation_info[linked_cell], element, cmap);

        std::cout << "get basis functions works \n";
        for (std::size_t k = 0; k < ndofs; k++)
        {
          for (std::size_t q = 0; q < test_fn.shape(0); ++q)
          {
            for (std::size_t l = 0; l < bs; l++)
            {
              c[i * cstride + link * ndofs * bs * num_q_points
                + k * bs * num_q_points + indices[q] * bs + l]
                  = test_fn(q, k * bs + l, l);
            }
          }
        }

        it = upper;
        link += 1;
      }
    }

    return {std::move(c), cstride};
  }

  /// Compute function on opposite surface at quadrature points of
  /// facets
  /// @param[in] orgin_meshtag - surface on which to integrate
  /// @param[in] - gap packed on facets per quadrature point
  /// @param[out] c - test functions packed on facets.
  std::pair<std::vector<PetscScalar>, int>
  pack_u_contact(int origin_meshtag,
                 std::shared_ptr<dolfinx::fem::Function<PetscScalar>> u,
                 const xtl::span<const PetscScalar> gap)
  {

    // Mesh info
    auto mesh = _marker->mesh();                     // mesh
    const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
    xt::xtensor<std::int32_t, 3>* map;
    std::vector<xt::xtensor<double, 2>>* q_phys_pt;
    const std::size_t bs = _V->element()->block_size();

    // Select which side of the contact interface to loop from and get the
    // correct map
    if (origin_meshtag == 0)
    {
      map = &_map_0_to_1;
      q_phys_pt = &_qp_phys_0;
    }
    else
    {
      map = &_map_1_to_0;
      q_phys_pt = &_qp_phys_1;
    }
    const std::size_t num_facets = (*map).shape(0);
    const std::size_t num_q_points = _qp_ref_facet[0].shape(0);
    std::vector<PetscScalar> c(num_facets * num_q_points * bs, 0.0);
    const int cstride = num_q_points * bs;

    xt::xtensor<double, 2> points = xt::zeros<double>({num_q_points, gdim});
    std::vector<std::int32_t> cells(num_q_points, 0);
    xt::xtensor<PetscScalar, 2> vals
        = xt::zeros<PetscScalar>({num_q_points, bs});
    for (std::size_t i = 0; i < num_facets; ++i)
    {
      for (std::size_t q = 0; q < num_q_points; ++q)
      {
        for (std::size_t j = 0; j < gdim; ++j)
        {
          points(q, j) = (*q_phys_pt)[i](q, j)
                         - gap[i * gdim * num_q_points + q * gdim + j];
          cells[q] = (*map)(i, q, 0);
        }
      }
      vals.fill(0);
      u->eval(points, cells, vals);
      for (std::size_t q = 0; q < num_q_points; ++q)
      {
        for (std::size_t j = 0; j < bs; ++j)
        {
          c[i * cstride + q * bs + j] = vals(q, j);
        }
      }
    }

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
    xt::xtensor<int32_t, 2>* puppet_facets;
    std::vector<xt::xtensor<double, 2>>* q_phys_pt;
    if (origin_meshtag == 0)
    {
      puppet_facets = &_cell_facet_pairs_0;
      q_phys_pt = &_qp_phys_0;
    }
    else
    {
      puppet_facets = &_cell_facet_pairs_1;
      q_phys_pt = &_qp_phys_1;
    }
    int32_t num_facets = (*puppet_facets).shape(0);
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
  xt::xtensor<std::int32_t, 3> _map_0_to_1;
  // Adjacency list of closest facet on surface_0 for every quadrature point
  // in _qp_phys_1 (quadrature points on every facet of surface_1)
  xt::xtensor<std::int32_t, 3> _map_1_to_0;

  // should be made redundant
  xt::xtensor<std::int32_t, 2> _map_0_to_1_facet;
  xt::xtensor<std::int32_t, 2> _map_1_to_0_facet;
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
  // cell facets pairs in surface 0
  // this duplicates _facet_0 but with (cell, local_facet) instead of facet
  // index only this duplication should be resolved with upcoming changes
  xt::xtensor<std::int32_t, 2> _cell_facet_pairs_0;
  // cell facets pairs in surface 1
  // this duplicates _facet_1 but with (cell, local_facet) instead of facet
  // index only this duplication should be resolved with upcoming changes
  xt::xtensor<std::int32_t, 2> _cell_facet_pairs_1;
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
