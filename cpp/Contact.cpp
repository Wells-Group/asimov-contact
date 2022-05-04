// Copyright (C) 2021-2022 Sarah Roggendorf and Jørgen S. Dokken
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#include "Contact.h"
#include "utils.h"
#include <dolfinx/common/log.h>
#include <dolfinx_cuas/utils.hpp>
using namespace dolfinx_contact;

namespace
{

/// Given a set of facets on the submesh, find all cells on the oposite surface
/// of the parent mesh that is linked.
/// @param[in, out] linked_cells List of unique cells on the parent mesh
/// (sorted)
/// @param[in] submesh_facets List of facets on the submesh
/// @param[in] sub_to_parent Map from each facet of on the submesh (local to
/// process) to the tuple (submesh_cell_index, local_facet_index)
/// @param[in] parent_cells Map from submesh cell (local to process) to parent
/// mesh cell (local to process)
void compute_linked_cells(
    std::vector<std::int32_t>& linked_cells,
    const tcb::span<const std::int32_t>& submesh_facets,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>&
        sub_to_parent,
    const tcb::span<const std::int32_t>& parent_cells)
{
  linked_cells.resize(submesh_facets.size());
  std::transform(submesh_facets.cbegin(), submesh_facets.cend(),
                 linked_cells.begin(),
                 [&sub_to_parent, &parent_cells](const auto facet)
                 {
                   // Extract (cell, facet) pair from submesh
                   auto facet_pair = sub_to_parent->links(facet);
                   assert(facet_pair.size() == 2);
                   return parent_cells[facet_pair[0]];
                 });

  // Remove duplicates
  dolfinx::radix_sort(xtl::span<std::int32_t>(linked_cells));
  linked_cells.erase(std::unique(linked_cells.begin(), linked_cells.end()),
                     linked_cells.end());
}

} // namespace

dolfinx_contact::Contact::Contact(
    std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> marker,
    const std::array<int, 2>& surfaces,
    std::shared_ptr<dolfinx::fem::FunctionSpace> V)
    : _marker(marker), _surfaces(surfaces), _V(V)
{
  auto mesh = _marker->mesh();
  const int tdim = mesh->topology().dim(); // topological dimension
  const int fdim = tdim - 1;               // topological dimension of facet
  const dolfinx::mesh::Topology& topology = mesh->topology();
  auto f_to_c = topology.connectivity(fdim, tdim);
  assert(f_to_c);
  auto c_to_f = topology.connectivity(tdim, fdim);
  assert(c_to_f);
  for (std::size_t s = 0; s < _surfaces.size(); ++s)
  {
    auto facets = _marker->find(_surfaces[s]);
    std::variant<std::vector<std::int32_t>,
                 std::vector<std::pair<std::int32_t, int>>,
                 std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>>
        pairs = dolfinx_cuas::compute_active_entities(
            mesh, facets, dolfinx::fem::IntegralType::exterior_facet);

    _cell_facet_pairs[s]
        = std::get<std::vector<std::pair<std::int32_t, int>>>(pairs);

    _submeshes[s] = dolfinx_contact::SubMesh(mesh, _cell_facet_pairs[s]);
  }
}
//------------------------------------------------------------------------------------------------
std::size_t dolfinx_contact::Contact::coefficients_size()
{
  // mesh data
  auto mesh = _marker->mesh();
  const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = _V->dofmap();
  const std::size_t ndofs_cell = dofmap->cell_dofs(0).size();
  const std::size_t bs = dofmap->bs();

  // NOTE: Assuming same number of quadrature points on each cell
  const std::size_t num_q_points = _qp_ref_facet[0].shape(0);
  const std::size_t max_links
      = *std::max_element(_max_links.begin(), _max_links.end());

  return 3 + num_q_points * (2 * gdim + ndofs_cell * bs * max_links + bs)
         + ndofs_cell * bs;
}

Mat dolfinx_contact::Contact::create_petsc_matrix(
    const dolfinx::fem::Form<PetscScalar>& a, const std::string& type)
{

  // Build standard sparsity pattern
  dolfinx::la::SparsityPattern pattern
      = dolfinx::fem::create_sparsity_pattern(a);

  auto dofmap = a.function_spaces().at(0)->dofmap();

  // Temporary array to hold dofs for sparsity pattern
  std::vector<std::int32_t> linked_dofs;

  // Loop over each contact interface, and create sparsity pattern for the
  // dofs on the opposite surface
  for (int s = 0; s < 2; ++s)
  {
    auto facet_map = _submeshes[_opposites[s]].facet_map();
    auto parent_cells = _submeshes[_opposites[s]].parent_cells();
    for (int i = 0; i < (int)_cell_facet_pairs[s].size(); i++)
    {
      auto cell = _cell_facet_pairs[s][i].first;
      auto cell_dofs = dofmap->cell_dofs(cell);

      linked_dofs.clear();
      for (auto link : _facet_maps[s]->links(i))
      {
        auto linked_sub_cell = facet_map->links(link)[0];
        auto linked_cell = parent_cells[linked_sub_cell];
        auto linked_cell_dofs = dofmap->cell_dofs(linked_cell);
        for (auto dof : linked_cell_dofs)
          linked_dofs.push_back(dof);
      }

      // Remove duplicates
      dolfinx::radix_sort(xtl::span<std::int32_t>(linked_dofs));
      linked_dofs.erase(std::unique(linked_dofs.begin(), linked_dofs.end()),
                        linked_dofs.end());

      pattern.insert(cell_dofs, linked_dofs);
      pattern.insert(linked_dofs, cell_dofs);
    }
  }
  // Finalise communication
  pattern.assemble();

  return dolfinx::la::petsc::create_matrix(a.mesh()->comm(), pattern, type);
}
//------------------------------------------------------------------------------------------------
std::pair<std::vector<PetscScalar>, int>
dolfinx_contact::Contact::pack_ny(int origin_meshtag,
                                  const xtl::span<const PetscScalar> gap)
{

  // Get information from candidate mesh

  // Get mesh and submesh
  const dolfinx_contact::SubMesh& submesh
      = _submeshes[_opposites[origin_meshtag]];
  auto candidate_mesh = submesh.mesh(); // mesh

  // Geometrical info
  const dolfinx::mesh::Geometry& geometry = candidate_mesh->geometry();
  const int gdim = geometry.dim(); // geometrical dimension
  const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
  auto x_dofmap = geometry.dofmap();
  xtl::span<const double> mesh_geometry = geometry.x();

  // Topological info
  const int tdim = candidate_mesh->topology().dim();
  auto cell_type = dolfinx::mesh::cell_type_to_basix_type(
      candidate_mesh->topology().cell_type());

  // Get facet normals on reference cell
  const xt::xtensor<double, 2> reference_normals
      = basix::cell::facet_outward_normals(cell_type);

  // Select which side of the contact interface to loop from and get the
  // correct map
  auto facet_map = submesh.facet_map();
  auto map = _facet_maps[origin_meshtag];
  auto qp_phys = _qp_phys[origin_meshtag];

  const std::size_t num_facets = _cell_facet_pairs[origin_meshtag].size();
  const std::size_t num_q_points = _qp_ref_facet[0].shape(0);

  // Needed for pull_back in get_facet_normals
  xt::xtensor<double, 2> J
      = xt::zeros<double>({std::size_t(gdim), std::size_t(tdim)});
  xt::xtensor<double, 2> K
      = xt::zeros<double>({std::size_t(tdim), std::size_t(gdim)});

  const std::size_t num_dofs_g = cmap.dim();
  xt::xtensor<double, 2> coordinate_dofs
      = xt::zeros<double>({num_dofs_g, std::size_t(gdim)});
  std::array<double, 3> normal = {0, 0, 0};
  std::array<double, 3> point = {0, 0, 0}; // To store Pi(x)

  // Loop over quadrature points
  const auto cstride = (int)num_q_points * gdim;
  std::vector<PetscScalar> normals(num_facets * num_q_points * gdim, 0.0);

  for (std::size_t i = 0; i < num_facets; ++i)
  {
    const xt::xtensor<double, 2>& qp_i = qp_phys[i];
    auto links = map->links((int)i);
    assert(links.size() == num_q_points);
    for (std::size_t q = 0; q < num_q_points; ++q)
    {

      // Extract linked cell and facet at quadrature point q
      auto linked_pair = facet_map->links(links[q]);
      std::int32_t linked_cell = linked_pair[0];
      // Compute Pi(x) from x, and gap = Pi(x) - x
      auto qp_iq = xt::row(qp_i, q);
      for (int k = 0; k < gdim; ++k)
        point[k] = qp_iq[k] + gap[i * gdim * num_q_points + q * gdim + k];

      // Extract local dofs
      auto x_dofs = x_dofmap.links(linked_cell);
      assert(num_dofs_g == (std::size_t)x_dofmap.num_links(linked_cell));

      for (std::size_t j = 0; j < x_dofs.size(); ++j)
      {
        std::copy_n(std::next(mesh_geometry.begin(), 3 * x_dofs[j]), gdim,
                    std::next(coordinate_dofs.begin(), j * gdim));
      }

      // Compute outward unit normal in point = Pi(x)
      // Note: in the affine case potential gains can be made
      //       if the cells are sorted like in pack_test_functions
      assert(linked_cell >= 0);
      normal = dolfinx_contact::push_forward_facet_normal(
          J, K, point, coordinate_dofs, linked_pair[1], cmap,
          reference_normals);
      // Copy normal into c
      const std::size_t offset = i * cstride + q * gdim;
      for (int l = 0; l < gdim; ++l)
        normals[offset + l] = normal[l];
    }
  }
  return {std::move(normals), cstride};
}
//------------------------------------------------------------------------------------------------
void dolfinx_contact::Contact::assemble_matrix(
    mat_set_fn& mat_set,
    [[maybe_unused]] const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
    int origin_meshtag, const dolfinx_contact::kernel_fn<PetscScalar>& kernel,
    const xtl::span<const PetscScalar> coeffs, int cstride,
    const xtl::span<const PetscScalar>& constants)
{
  auto mesh = _marker->mesh();
  assert(mesh);

  // Extract geometry data
  const dolfinx::mesh::Geometry& geometry = mesh->geometry();
  const int gdim = geometry.dim();
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = geometry.dofmap();
  xtl::span<const double> x_g = geometry.x();
  const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
  const std::size_t num_dofs_g = cmap.dim();

  std::shared_ptr<const dolfinx::fem::FiniteElement> element = _V->element();
  if (element->needs_dof_transformations())
  {
    throw std::invalid_argument(
        "Function-space requiring dof-transformations is not supported.");
  }

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = _V->dofmap();
  const std::size_t ndofs_cell = dofmap->cell_dofs(0).size();
  const int bs = dofmap->bs();

  std::size_t max_links = std::max(_max_links[0], _max_links[1]);
  if (max_links == 0)
  {
    LOG(WARNING)
        << "No links between interfaces, compute_linked_cell will be skipped";
  }

  auto active_facets = _cell_facet_pairs[origin_meshtag];
  auto map = _facet_maps[origin_meshtag];
  auto facet_map = _submeshes[_opposites[origin_meshtag]].facet_map();
  auto parent_cells = _submeshes[_opposites[origin_meshtag]].parent_cells();
  // Data structures used in assembly
  std::vector<double> coordinate_dofs(3 * num_dofs_g);
  std::vector<std::vector<PetscScalar>> Aes(
      3 * max_links + 1,
      std::vector<PetscScalar>(bs * ndofs_cell * bs * ndofs_cell));
  std::vector<std::int32_t> linked_cells;
  for (std::size_t i = 0; i < active_facets.size(); i++)
  {
    [[maybe_unused]] auto [cell, local_index] = active_facets[i];
    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(cell);
    for (std::size_t j = 0; j < x_dofs.size(); ++j)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[j]), gdim,
                  std::next(coordinate_dofs.begin(), j * 3));
    }

    if (max_links > 0)
    {
      // Compute the unique set of cells linked to the current facet
      compute_linked_cells(linked_cells, map->links((int)i), facet_map,
                           parent_cells);
    }
    // Fill initial local element matrices with zeros prior to assembly
    const std::size_t num_linked_cells = linked_cells.size();
    std::fill(Aes[0].begin(), Aes[0].end(), 0);
    for (std::size_t j = 0; j < num_linked_cells; j++)
    {
      std::fill(Aes[3 * j + 1].begin(), Aes[3 * j + 1].end(), 0);
      std::fill(Aes[3 * j + 2].begin(), Aes[3 * j + 2].end(), 0);
      std::fill(Aes[3 * j + 3].begin(), Aes[3 * j + 3].end(), 0);
    }

    kernel(Aes, coeffs.data() + i * cstride, constants.data(),
           coordinate_dofs.data(), local_index, num_linked_cells);

    // FIXME: We would have to handle possible Dirichlet conditions here, if we
    // think that we can have a case with contact and Dirichlet
    auto dmap_cell = dofmap->cell_dofs(cell);
    mat_set(dmap_cell, dmap_cell, Aes[0]);

    for (std::size_t j = 0; j < num_linked_cells; j++)
    {
      auto dmap_linked = dofmap->cell_dofs(linked_cells[j]);
      mat_set(dmap_cell, dmap_linked, Aes[3 * j + 1]);
      mat_set(dmap_linked, dmap_cell, Aes[3 * j + 2]);
      mat_set(dmap_linked, dmap_linked, Aes[3 * j + 3]);
    }
  }
}
//------------------------------------------------------------------------------------------------

void dolfinx_contact::Contact::assemble_vector(
    xtl::span<PetscScalar> b, int origin_meshtag,
    const dolfinx_contact::kernel_fn<PetscScalar>& kernel,
    const xtl::span<const PetscScalar>& coeffs, int cstride,
    const xtl::span<const PetscScalar>& constants)
{
  /// Check that we support the function space
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = _V->element();
  if (const bool needs_dof_transformations
      = element->needs_dof_transformations();
      needs_dof_transformations)
  {
    throw std::invalid_argument(
        "Function-space requiring dof-transformations is not supported.");
  }

  // Extract mesh
  auto mesh = _marker->mesh();
  assert(mesh);
  const dolfinx::mesh::Geometry& geometry = mesh->geometry();
  const int gdim = geometry.dim(); // geometrical dimension

  // Prepare cell geometry
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = geometry.dofmap();
  xtl::span<const double> x_g = geometry.x();

  const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
  const std::size_t num_dofs_g = cmap.dim();

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = _V->dofmap();
  const std::size_t ndofs_cell = dofmap->cell_dofs(0).size();
  const int bs = dofmap->bs();

  // Select which side of the contact interface to loop from and get the
  // correct map
  auto active_facets = _cell_facet_pairs[origin_meshtag];
  auto map = _facet_maps[origin_meshtag];
  auto facet_map = _submeshes[_opposites[origin_meshtag]].facet_map();
  auto parent_cells = _submeshes[_opposites[origin_meshtag]].parent_cells();
  std::size_t max_links = std::max(_max_links[0], _max_links[1]);
  if (max_links == 0)
  {
    LOG(WARNING)
        << "No links between interfaces, compute_linked_cell will be skipped";
  }
  // Data structures used in assembly
  std::vector<double> coordinate_dofs(3 * num_dofs_g);
  std::vector<std::vector<PetscScalar>> bes(
      max_links + 1, std::vector<PetscScalar>(bs * ndofs_cell));

  // Tempoary array to hold cell links
  std::vector<std::int32_t> linked_cells;
  for (std::size_t i = 0; i < active_facets.size(); i++)
  {
    [[maybe_unused]] auto [cell, local_index] = active_facets[i];

    // Get cell coordinates/geometry
    auto x_dofs = x_dofmap.links(cell);
    for (std::size_t j = 0; j < x_dofs.size(); ++j)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[j]), gdim,
                  std::next(coordinate_dofs.begin(), j * 3));
    }

    // Compute the unique set of cells linked to the current facet
    if (max_links > 0)
    {
      compute_linked_cells(linked_cells, map->links((int)i), facet_map,
                           parent_cells);
    }

    // Using integer loop here to reduce number of zeroed vectors
    const std::size_t num_linked_cells = linked_cells.size();
    std::fill(bes[0].begin(), bes[0].end(), 0);
    for (std::size_t j = 0; j < num_linked_cells; j++)
      std::fill(bes[j + 1].begin(), bes[j + 1].end(), 0);
    kernel(bes, coeffs.data() + i * cstride, constants.data(),
           coordinate_dofs.data(), local_index, num_linked_cells);

    // Add element vector to global vector
    auto dofs_cell = dofmap->cell_dofs(cell);
    for (std::size_t j = 0; j < ndofs_cell; ++j)
      for (int k = 0; k < bs; ++k)
        b[bs * dofs_cell[j] + k] += bes[0][bs * j + k];
    for (std::size_t l = 0; l < num_linked_cells; ++l)
    {
      auto dofs_linked = dofmap->cell_dofs(linked_cells[l]);
      for (std::size_t j = 0; j < ndofs_cell; ++j)
        for (int k = 0; k < bs; ++k)
          b[bs * dofs_linked[j] + k] += bes[l + 1][bs * j + k];
    }
  }
}
