// Copyright (C) 2021-2022 Sarah Roggendorf and Jørgen S. Dokken
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#include "Contact.h"
#include "error_handling.h"
#include "utils.h"
#include <dolfinx/common/log.h>
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
    const std::vector<std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>>&
        markers,
    std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>> surfaces,
    const std::vector<std::array<int, 2>>& contact_pairs,
    std::shared_ptr<dolfinx::fem::FunctionSpace> V, const int q_deg)
    : _surfaces(surfaces->array()), _contact_pairs(contact_pairs), _V(V)
{
  std::size_t num_surfaces = surfaces->array().size();
  assert(_V);
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = _V->mesh();
  const int tdim = mesh->topology().dim(); // topological dimension
  const int fdim = tdim - 1;               // topological dimension of facet
  const dolfinx::mesh::Topology& topology = mesh->topology();
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> f_to_c
      = topology.connectivity(fdim, tdim);
  assert(f_to_c);
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> c_to_f
      = topology.connectivity(tdim, fdim);
  assert(c_to_f);
  // used to store list of (cell, facet) for each surface
  _cell_facet_pairs.resize(num_surfaces);
  // used to store submesh for each surface
  _submeshes.resize(num_surfaces);
  // used to store map from puppet to candidate surface for each contact pair
  _facet_maps.resize(contact_pairs.size());
  // store physical quadrature points for each surface
  _qp_phys.resize(num_surfaces);
  // store max number of links for each puppet surface
  _max_links.resize(contact_pairs.size());
  for (std::size_t s = 0; s < markers.size(); ++s)
  {
    std::shared_ptr<dolfinx::mesh::MeshTags<int>> marker = markers[s];
    tcb::span<const int> links = surfaces->links(int(s));
    for (std::size_t i = 0; i < links.size(); ++i)
    {
      std::vector<std::int32_t> facets = marker->find(links[i]);
      int index = surfaces->offsets()[s] + int(i);
      _cell_facet_pairs[index] = dolfinx_contact::compute_active_entities(
          mesh, facets, dolfinx::fem::IntegralType::exterior_facet);
      _submeshes[index]
          = dolfinx_contact::SubMesh(mesh, _cell_facet_pairs[index]);
    }
  }
  _quadrature_rule = std::make_shared<QuadratureRule>(
      topology.cell_type(), q_deg, fdim, basix::quadrature::type::Default);
}
//------------------------------------------------------------------------------------------------
std::size_t dolfinx_contact::Contact::coefficients_size()
{
  // mesh data
  assert(_V);
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = _V->mesh();
  const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = _V->dofmap();
  const std::size_t ndofs_cell = dofmap->cell_dofs(0).size();
  const std::size_t bs = dofmap->bs();

  // NOTE: Assuming same number of quadrature points on each cell
  dolfinx_contact::error::check_cell_type(mesh->topology().cell_type());

  const std::size_t num_q_points
      = _quadrature_rule->offset()[1] - _quadrature_rule->offset()[0];
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

  std::shared_ptr<const dolfinx::fem::DofMap> dofmap
      = a.function_spaces().at(0)->dofmap();

  // Temporary array to hold dofs for sparsity pattern
  std::vector<std::int32_t> linked_dofs;

  // Loop over each contact interface, and create sparsity pattern for the
  // dofs on the opposite surface
  for (std::size_t k = 0; k < _contact_pairs.size(); ++k)
  {
    const std::array<int, 2>& contact_pair = _contact_pairs[k];
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
        = _submeshes[contact_pair[1]].facet_map();
    const std::vector<std::int32_t>& parent_cells
        = _submeshes[contact_pair[1]].parent_cells();
    for (int i = 0; i < (int)_cell_facet_pairs[contact_pair[0]].size(); i += 2)
    {
      std::int32_t cell = _cell_facet_pairs[contact_pair[0]][i];
      tcb::span<const int> cell_dofs = dofmap->cell_dofs(cell);

      linked_dofs.clear();
      for (auto link : _facet_maps[k]->links(i / 2))
      {
        const int linked_sub_cell = facet_map->links(link)[0];
        const std::int32_t linked_cell = parent_cells[linked_sub_cell];
        tcb::span<const int> linked_cell_dofs = dofmap->cell_dofs(linked_cell);
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
void dolfinx_contact::Contact::create_distance_map(int pair)
{
  // Get quadrature mesh info
  int puppet_mt = _contact_pairs[pair][0];
  const std::vector<std::int32_t>& puppet_facets = _cell_facet_pairs[puppet_mt];
  std::shared_ptr<const dolfinx::mesh::Mesh> puppet_mesh
      = _submeshes[puppet_mt].mesh();
  std::vector<std::int32_t> quadrature_facets(puppet_facets.size());
  {
    const int tdim = puppet_mesh->topology().dim();
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> c_to_f
        = puppet_mesh->topology().connectivity(tdim, tdim - 1);
    assert(c_to_f);
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> cell_map
        = _submeshes[puppet_mt].cell_map();

    for (std::size_t i = 0; i < puppet_facets.size(); i += 2)
    {
      auto sub_cells = cell_map->links(puppet_facets[i]);
      assert(!sub_cells.empty());
      quadrature_facets[i] = sub_cells.front();
      quadrature_facets[i + 1] = puppet_facets[i + 1];
    }
  }
  int candidate_mt = _contact_pairs[pair][1];
  const std::vector<std::int32_t>& candidate_facets
      = _cell_facet_pairs[candidate_mt];
  std::vector<std::int32_t> submesh_facets(candidate_facets.size());
  std::shared_ptr<const dolfinx::mesh::Mesh> candidate_mesh
      = _submeshes[candidate_mt].mesh();
  {
    const int tdim = candidate_mesh->topology().dim();
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> c_to_f
        = candidate_mesh->topology().connectivity(tdim, tdim - 1);
    assert(c_to_f);
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> cell_map
        = _submeshes[candidate_mt].cell_map();

    for (std::size_t i = 0; i < candidate_facets.size(); i += 2)
    {
      auto submesh_cell = cell_map->links(candidate_facets[i]);
      assert(!submesh_cell.empty());
      submesh_facets[i] = submesh_cell.front();
      submesh_facets[i + 1] = candidate_facets[i + 1];
    }
  }

  // Compute facet map
  _facet_maps[pair]
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          dolfinx_contact::compute_distance_map(*puppet_mesh, quadrature_facets,
                                                *candidate_mesh, submesh_facets,
                                                *_quadrature_rule));

  // NOTE: More data that should be updated inside this code
  const dolfinx::fem::CoordinateElement& cmap
      = candidate_mesh->geometry().cmap();
  _phi_ref_facets = tabulate(cmap, _quadrature_rule);

  // NOTE: This function should be moved somwhere else, or return the actual
  // points such that we compuld send them in to compute_distance_map.
  // Compute quadrature points on physical facet _qp_phys_"origin_meshtag"
  create_q_phys(puppet_mt);

  // Update maximum number of connected cells
  max_links(pair);
}
//------------------------------------------------------------------------------------------------
std::pair<std::vector<PetscScalar>, int>
dolfinx_contact::Contact::pack_ny(int pair,
                                  const xtl::span<const PetscScalar> gap)
{
  const std::array<int, 2>& contact_pair = _contact_pairs[pair];
  // Get information from candidate mesh

  // Get mesh and submesh
  const dolfinx_contact::SubMesh& submesh = _submeshes[contact_pair[1]];
  std::shared_ptr<const dolfinx::mesh::Mesh> candidate_mesh
      = submesh.mesh(); // mesh

  // Geometrical info
  const dolfinx::mesh::Geometry& geometry = candidate_mesh->geometry();
  const int gdim = geometry.dim(); // geometrical dimension
  const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
  const dolfinx::graph::AdjacencyList<int>& x_dofmap = geometry.dofmap();
  xtl::span<const double> mesh_geometry = geometry.x();

  // Topological info
  const int tdim = candidate_mesh->topology().dim();
  basix::cell::type cell_type = dolfinx::mesh::cell_type_to_basix_type(
      candidate_mesh->topology().cell_type());

  // Get facet normals on reference cell
  const xt::xtensor<double, 2> reference_normals
      = basix::cell::facet_outward_normals(cell_type);

  // Select which side of the contact interface to loop from and get the
  // correct map
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
      = submesh.facet_map();
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> map
      = _facet_maps[pair];
  const std::vector<xt::xtensor<double, 2>>& qp_phys
      = _qp_phys[contact_pair[0]];

  const std::size_t num_facets = _cell_facet_pairs[contact_pair[0]].size() / 2;
  const std::size_t num_q_points
      = _quadrature_rule->offset()[1] - _quadrature_rule->offset()[0];

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
  const int cstride = (int)num_q_points * gdim;
  std::vector<PetscScalar> normals(num_facets * num_q_points * gdim, 0.0);

  for (std::size_t i = 0; i < num_facets; ++i)
  {
    const xt::xtensor<double, 2>& qp_i = qp_phys[i];
    const tcb::span<const int> links = map->links((int)i);
    assert(links.size() == num_q_points);
    for (std::size_t q = 0; q < num_q_points; ++q)
    {

      // Extract linked cell and facet at quadrature point q
      const tcb::span<const int> linked_pair = facet_map->links(links[q]);
      std::int32_t linked_cell = linked_pair[0];
      // Compute Pi(x) from x, and gap = Pi(x) - x
      auto qp_iq = xt::row(qp_i, q);
      for (int k = 0; k < gdim; ++k)
        point[k] = qp_iq[k] + gap[i * gdim * num_q_points + q * gdim + k];

      // Extract local dofs
      const tcb::span<const int> x_dofs = x_dofmap.links(linked_cell);
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
    int pair, const dolfinx_contact::kernel_fn<PetscScalar>& kernel,
    const xtl::span<const PetscScalar> coeffs, int cstride,
    const xtl::span<const PetscScalar>& constants)
{
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = _V->mesh();
  assert(mesh);

  // Extract geometry data
  const dolfinx::mesh::Geometry& geometry = mesh->geometry();
  const int gdim = geometry.dim();
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = geometry.dofmap();
  xtl::span<const double> x_g = geometry.x();
  const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
  const std::size_t num_dofs_g = cmap.dim();

  if (_V->element()->needs_dof_transformations())
  {
    throw std::invalid_argument(
        "Function-space requiring dof-transformations is not supported.");
  }

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = _V->dofmap();
  const std::size_t ndofs_cell = dofmap->cell_dofs(0).size();
  const int bs = dofmap->bs();
  const std::size_t max_links
      = *std::max_element(_max_links.begin(), _max_links.end());
  if (max_links == 0)
  {
    LOG(WARNING)
        << "No links between interfaces, compute_linked_cell will be skipped";
  }

  const std::array<int, 2>& contact_pair = _contact_pairs[pair];
  const std::vector<std::int32_t>& active_facets
      = _cell_facet_pairs[contact_pair[0]];
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> map
      = _facet_maps[pair];
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
      = _submeshes[contact_pair[1]].facet_map();
  const std::vector<std::int32_t>& parent_cells
      = _submeshes[_contact_pairs[pair][1]].parent_cells();
  // Data structures used in assembly
  std::vector<double> coordinate_dofs(3 * num_dofs_g);
  std::vector<std::vector<PetscScalar>> Aes(
      3 * max_links + 1,
      std::vector<PetscScalar>(bs * ndofs_cell * bs * ndofs_cell));
  std::vector<std::int32_t> linked_cells;
  for (std::size_t i = 0; i < active_facets.size(); i += 2)
  {
    // Get cell coordinates/geometry
    assert(active_facets[i] < x_dofmap.num_nodes());
    const tcb::span<const int> x_dofs = x_dofmap.links(active_facets[i]);
    for (std::size_t j = 0; j < x_dofs.size(); ++j)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[j]), gdim,
                  std::next(coordinate_dofs.begin(), j * 3));
    }

    if (max_links > 0)
    {
      // Compute the unique set of cells linked to the current facet
      compute_linked_cells(linked_cells, map->links((int)i / 2), facet_map,
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

    kernel(Aes, coeffs.data() + i / 2 * cstride, constants.data(),
           coordinate_dofs.data(), active_facets[i + 1], num_linked_cells);

    // FIXME: We would have to handle possible Dirichlet conditions here, if we
    // think that we can have a case with contact and Dirichlet
    auto dmap_cell = dofmap->cell_dofs(active_facets[i]);
    mat_set(dmap_cell, dmap_cell, Aes[0]);

    for (std::size_t j = 0; j < num_linked_cells; j++)
    {
      auto dmap_linked = dofmap->cell_dofs(linked_cells[j]);
      assert(!dmap_linked.empty());
      mat_set(dmap_cell, dmap_linked, Aes[3 * j + 1]);
      mat_set(dmap_linked, dmap_cell, Aes[3 * j + 2]);
      mat_set(dmap_linked, dmap_linked, Aes[3 * j + 3]);
    }
  }
}
//------------------------------------------------------------------------------------------------

void dolfinx_contact::Contact::assemble_vector(
    xtl::span<PetscScalar> b, int pair,
    const dolfinx_contact::kernel_fn<PetscScalar>& kernel,
    const xtl::span<const PetscScalar>& coeffs, int cstride,
    const xtl::span<const PetscScalar>& constants)
{
  /// Check that we support the function space
  if (_V->element()->needs_dof_transformations())
  {
    throw std::invalid_argument(
        "Function-space requiring dof-transformations is not supported.");
  }

  // Extract mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = _V->mesh();
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
  const std::array<int, 2>& contact_pair = _contact_pairs[pair];
  const std::vector<std::int32_t>& active_facets
      = _cell_facet_pairs[contact_pair[0]];
  const dolfinx_contact::SubMesh& submesh = _submeshes[contact_pair[1]];
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> map
      = _facet_maps[pair];
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
      = submesh.facet_map();
  std::vector<std::int32_t> parent_cells = submesh.parent_cells();
  const std::size_t max_links
      = *std::max_element(_max_links.begin(), _max_links.end());
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
  for (std::size_t i = 0; i < active_facets.size(); i += 2)
  {

    // Get cell coordinates/geometry
    const tcb::span<const int> x_dofs = x_dofmap.links(active_facets[i]);
    for (std::size_t j = 0; j < x_dofs.size(); ++j)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[j]), gdim,
                  std::next(coordinate_dofs.begin(), j * 3));
    }

    // Compute the unique set of cells linked to the current facet
    if (max_links > 0)
    {
      compute_linked_cells(linked_cells, map->links((int)i / 2), facet_map,
                           parent_cells);
    }

    // Using integer loop here to reduce number of zeroed vectors
    const std::size_t num_linked_cells = linked_cells.size();
    std::fill(bes[0].begin(), bes[0].end(), 0);
    for (std::size_t j = 0; j < num_linked_cells; j++)
      std::fill(bes[j + 1].begin(), bes[j + 1].end(), 0);
    kernel(bes, coeffs.data() + i / 2 * cstride, constants.data(),
           coordinate_dofs.data(), active_facets[i + 1], num_linked_cells);

    // Add element vector to global vector
    const tcb::span<const int> dofs_cell = dofmap->cell_dofs(active_facets[i]);
    for (std::size_t j = 0; j < ndofs_cell; ++j)
      for (int k = 0; k < bs; ++k)
        b[bs * dofs_cell[j] + k] += bes[0][bs * j + k];
    for (std::size_t l = 0; l < num_linked_cells; ++l)
    {
      const tcb::span<const int> dofs_linked
          = dofmap->cell_dofs(linked_cells[l]);
      for (std::size_t j = 0; j < ndofs_cell; ++j)
        for (int k = 0; k < bs; ++k)
          b[bs * dofs_linked[j] + k] += bes[l + 1][bs * j + k];
    }
  }
}
//-----------------------------------------------------------------------------------------------
void dolfinx_contact::Contact::update_submesh_geometry(
    dolfinx::fem::Function<PetscScalar>& u) const
{

  for (auto submesh : _submeshes)
    submesh.update_geometry(u);
}
