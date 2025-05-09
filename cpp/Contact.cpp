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
dolfinx::graph::AdjacencyList<std::int32_t> create_cell_facet_pairs(
    const dolfinx::mesh::Mesh<double>& mesh,
    const std::vector<
        std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>>>& markers,
    const dolfinx::graph::AdjacencyList<std::int32_t>& surfaces)
{
  std::vector<std::int32_t> all_facet_pairs;
  std::vector<std::int32_t> offsets;
  offsets.push_back(0);

  // Loop over surfaces
  for (std::size_t s = 0; s < markers.size(); ++s)
  {
    std::shared_ptr<const dolfinx::mesh::MeshTags<int>> marker = markers[s];

    // For each surface, loop of the tag values to be extracted from
    // marker
    std::span<const int> links = surfaces.links(s);
    for (auto tag : links)
    {
      // Get the facets ids with tag links[i]
      std::vector<std::int32_t> facets = marker->find(tag);

      // Compute the (cell, local index) pairs for each facet
      std::vector<std::int32_t> cell_facet_pairs = compute_active_entities(
          mesh, facets, dolfinx::fem::IntegralType::exterior_facet);

      // Add to list of  (cell, local index) pairs
      all_facet_pairs.insert(all_facet_pairs.end(),
                             std::begin(cell_facet_pairs),
                             std::end(cell_facet_pairs));

      // Add to offset
      offsets.push_back(offsets.back() + cell_facet_pairs.size());
    }
  }

  return dolfinx::graph::AdjacencyList<std::int32_t>(std::move(all_facet_pairs),
                                                     std::move(offsets));
}

/// Tabulate the coordinate element basis functions at quadrature
/// points.
///
/// @param[in] cmap The coordinate element
/// @param[in] q_rule The quadrature rule
std::pair<std::vector<double>, std::array<std::size_t, 4>>
tabulate(const dolfinx::fem::CoordinateElement<double>& cmap,
         const dolfinx_contact::QuadratureRule& q_rule)
{

  // Create quadrature points on reference facet
  const std::vector<double>& q_weights = q_rule.weights();
  const std::vector<double>& q_points = q_rule.points();
  assert(q_weights.size() == (std::size_t)q_rule.offset().back());
  // Tabulate Coordinate element (first derivative to compute Jacobian)
  std::array<std::size_t, 4> cmap_shape
      = cmap.tabulate_shape(0, q_weights.size());
  std::vector<double> cmap_basis(
      std::reduce(cmap_shape.begin(), cmap_shape.end(), 1, std::multiplies{}));
  cmap.tabulate(0, q_points, {q_weights.size(), q_rule.tdim()}, cmap_basis);
  return {cmap_basis, cmap_shape};
}

/// Given a set of facets on the submesh, find all cells on the opposite surface
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
    std::span<const std::int32_t> submesh_facets,
    const dolfinx::graph::AdjacencyList<std::int32_t>& sub_to_parent,
    std::span<const std::int32_t> parent_cells)
{
  linked_cells.resize(0);
  linked_cells.reserve(submesh_facets.size());
  std::for_each(submesh_facets.begin(), submesh_facets.end(),
                [&sub_to_parent, &parent_cells, &linked_cells](const auto facet)
                {
                  // Remove facets with negative index
                  if (facet >= 0)
                  {
                    // Extract (cell, facet) pair from submesh
                    auto facet_pair = sub_to_parent.links(facet);
                    assert(facet_pair.size() == 2);
                    linked_cells.push_back(parent_cells[facet_pair[0]]);
                  }
                });

  // Remove duplicates
  dolfinx::radix_sort(linked_cells);
  linked_cells.erase(std::unique(linked_cells.begin(), linked_cells.end()),
                     linked_cells.end());
}

} // namespace

//----------------------------------------------------------------------------
Contact::Contact(const dolfinx::graph::AdjacencyList<std::int32_t>& surfaces,
                 const std::vector<std::array<int, 2>>& contact_pairs,
                 std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh,
                 const std::vector<ContactMode>& mode, int q_deg)
    : _quadrature_rule(mesh->topology()->cell_type(), q_deg,
                       mesh->topology()->dim() - 1,
                       basix::quadrature::type::Default),
      _contact_pairs(contact_pairs), _mesh(mesh),
      _facet_maps(contact_pairs.size()),
      _reference_contact_points(contact_pairs.size()),
      _reference_contact_shape(contact_pairs.size()),
      _qp_phys(surfaces.array().size()), _max_links(contact_pairs.size()),
      _cell_facet_pairs(surfaces), _submesh(*mesh, surfaces.array()),
      _mode(mode)
{
  assert(_mesh);
  auto topology = mesh->topology();
  std::int32_t num_cells = topology->index_map(topology->dim())->size_local();
  std::vector<std::size_t> _num_local_facets;
  for (int s = 0; s < _cell_facet_pairs.num_nodes(); ++s)
  {
    auto facets = _cell_facet_pairs.links(s);

    // Count the number of owned (cells, local facet index pairs). List
    // is sorted by cell index, so we could backwards from the end of
    // the list for efficiency.
    _local_facets.push_back(0);
    for (std::int32_t i = (std::int32_t)facets.size() - 2; i >= 0; i -= 2)
    {
      if (facets[i] < num_cells)
      {
        _local_facets.back() = i / 2 + 1;
        break;
      }
    }
  }
}
//----------------------------------------------------------------------------
Contact::Contact(
    const std::vector<
        std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>>>& markers,
    const dolfinx::graph::AdjacencyList<std::int32_t>& surfaces,
    const std::vector<std::array<int, 2>>& contact_pairs,
    std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh,
    const std::vector<ContactMode>& mode, int q_deg)
    : Contact(create_cell_facet_pairs(*mesh, markers, surfaces), contact_pairs,
              mesh, mode, q_deg)
{
  if (markers.size() != (std::size_t)surfaces.num_nodes())
    throw std::runtime_error("maker and surfaces have different sizes.");
}
//----------------------------------------------------------------------------
std::pair<std::vector<double>, std::array<std::size_t, 3>>
Contact::qp_phys(int surface) const
{
  std::size_t num_facets = _local_facets[surface];
  std::size_t num_q_points
      = _quadrature_rule.offset()[1] - _quadrature_rule.offset()[0];
  std::size_t gdim = _mesh->geometry().dim();
  std::array<std::size_t, 3> shape = {num_facets, num_q_points, gdim};
  return {_qp_phys[surface], shape};
}
//-----------------------------------------------------------------------------
std::size_t
Contact::coefficients_size(bool meshtie,
                           const dolfinx::fem::FunctionSpace<double>& V) const
{
  // mesh data
  std::size_t gdim = _mesh->geometry().dim(); // geometrical dimension

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V.dofmap();
  std::size_t ndofs_cell = dofmap->cell_dofs(0).size();
  std::size_t bs = dofmap->bs();

  // NOTE: Assuming same number of quadrature points on each cell
  error::check_cell_type(_mesh->topology()->cell_type());

  std::size_t num_q_points
      = _quadrature_rule.offset()[1] - _quadrature_rule.offset()[0];
  std::size_t max_links
      = *std::max_element(_max_links.begin(), _max_links.end());

  if (meshtie)
  {

    // Coefficient offsets
    // Expecting coefficients in following order:
    // mu, lmbda, h,test_fn, grad(test_fn), u, grad(u), u_opposite,
    // grad(u_opposite)
    std::array<std::size_t, 9> cstrides
        = {1,
           1,
           1,
           num_q_points * ndofs_cell * bs * max_links,
           num_q_points * ndofs_cell * bs * max_links,
           num_q_points * gdim,
           num_q_points * gdim * gdim,
           num_q_points * bs,
           num_q_points * gdim * bs};
    return std::accumulate(cstrides.cbegin(), cstrides.cend(), 0);
  }
  else
  {
    // Coefficient offsets
    // Expecting coefficients in the following order
    // mu, lmbda, h, gap, friction_coeff, normals, test_fns, u, grad(u),
    // u_opposite,
    std::array<std::size_t, 9> cstrides
        = {4,
           num_q_points * bs,
           num_q_points * bs,
           num_q_points * ndofs_cell * bs * max_links,
           num_q_points * gdim,
           num_q_points * gdim * gdim,
           num_q_points * bs,
           num_q_points * gdim};
    return std::accumulate(cstrides.cbegin(), cstrides.cend(), 0);
  }
}
//------------------------------------------------------------------------------------------------
Mat Contact::create_petsc_matrix(const dolfinx::fem::Form<PetscScalar>& a,
                                 std::string type) const
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
    std::array<int, 2> contact_pair = _contact_pairs[k];
    std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
        = _submesh.facet_map();
    assert(facet_map);
    std::span<const std::int32_t> parent_cells = _submesh.parent_cells();

    auto cell_facet_pairs = _cell_facet_pairs.links(contact_pair.front());
    for (std::size_t i = 0; i < 2 * _local_facets[contact_pair.front()]; i += 2)
    {
      std::int32_t cell = cell_facet_pairs[i];
      std::span<const int> cell_dofs = dofmap->cell_dofs(cell);

      linked_dofs.clear();
      for (auto link : _facet_maps[k]->links(i / 2))
      {
        if (link < 0)
          continue;
        const int linked_sub_cell = facet_map->links(link).front();
        const std::int32_t linked_cell = parent_cells[linked_sub_cell];
        for (auto dof : dofmap->cell_dofs(linked_cell))
          linked_dofs.push_back(dof);
      }

      // Remove duplicates
      dolfinx::radix_sort(std::span<std::int32_t>(linked_dofs));
      linked_dofs.erase(std::unique(linked_dofs.begin(), linked_dofs.end()),
                        linked_dofs.end());

      pattern.insert(cell_dofs, linked_dofs);
      pattern.insert(linked_dofs, cell_dofs);
    }
  }
  // Finalise communication
  pattern.finalize();

  return dolfinx::la::petsc::create_matrix(a.mesh()->comm(), pattern, type);
}
//------------------------------------------------------------------------------------------------
void Contact::create_distance_map(int pair)
{
  // Get quadrature mesh info
  auto [quadrature_mt, candidate_mt] = _contact_pairs[pair];
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> quadrature_mesh
      = _submesh.mesh();
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> candidate_mesh
      = _submesh.mesh();
  std::size_t num_facets = _local_facets[quadrature_mt];

  // Get (cell, local_facet_index) tuples on quadrature submesh
  std::vector<std::int32_t> quadrature_facets = _submesh.get_submesh_tuples(
      _cell_facet_pairs.links(quadrature_mt).subspan(0, 2 * num_facets));

  // Get (cell, local_facet_index) tuples on candidate submesh
  std::vector<std::int32_t> submesh_facets
      = _submesh.get_submesh_tuples(_cell_facet_pairs.links(candidate_mt));

  // Compute facet map
  [[maybe_unused]] auto [adj, reference_x, shape] = compute_distance_map(
      *quadrature_mesh, quadrature_facets, *candidate_mesh, submesh_facets,
      _quadrature_rule, _mode[pair], _radius);

  _facet_maps[pair]
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(adj);

  _reference_contact_points[pair] = reference_x;
  _reference_contact_shape[pair] = shape;

  // NOTE: More data that should be updated inside this code
  const dolfinx::fem::CoordinateElement<double>& cmap
      = candidate_mesh->geometry().cmap();
  std::tie(_reference_basis, _reference_shape)
      = tabulate(cmap, _quadrature_rule);

  // NOTE: This function should be moved somwhere else, or return the
  // actual points such that we could send them in to
  // compute_distance_map. Compute quadrature points on physical facet
  // _qp_phys_"origin_meshtag"
  create_q_phys(quadrature_mt);

  // Update maximum number of connected cells
  _max_links[pair] = _quadrature_rule.num_points(0);
}
//----------------------------------------------------------------------------
std::pair<std::vector<PetscScalar>, int> Contact::pack_nx(int pair) const
{
  auto [quadrature_mt, candidate_mt] = _contact_pairs[pair];
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> quadrature_mesh
      = _submesh.mesh();
  assert(quadrature_mesh);

  std::size_t num_facets = _local_facets[quadrature_mt];

  // Get (cell, local_facet_index) tuples on quadrature submesh
  std::vector<std::int32_t> quadrature_facets = _submesh.get_submesh_tuples(
      _cell_facet_pairs.links(quadrature_mt).subspan(0, 2 * num_facets));

  // Get information about submesh geometry and topology
  const dolfinx::mesh::Geometry<double>& geometry = quadrature_mesh->geometry();
  int gdim = geometry.dim();
  std::span<const double> x_g = geometry.x();
  auto x_dofmap = geometry.dofmap();
  const dolfinx::fem::CoordinateElement<double>& cmap = geometry.cmap();
  std::size_t num_dofs_g = cmap.dim();
  auto topology = quadrature_mesh->topology();
  int tdim = topology->dim();

  // num quadrature pints
  error::check_cell_type(quadrature_mesh->topology()->cell_type());
  std::size_t num_q_points = _quadrature_rule.num_points(0);
  std::vector<PetscScalar> normals(num_facets * num_q_points * gdim, 0.0);
  int cstride = num_q_points * gdim;

  // return if no facets on process
  if (num_facets == 0)
    return {std::move(normals), cstride};

  // Get all quadrature points
  const std::vector<double>& q_points = _quadrature_rule.points();
  assert(_quadrature_rule.tdim() == (std::size_t)tdim);
  std::array<std::size_t, 2> shape
      = {q_points.size() / tdim, (std::size_t)tdim};

  // Tabulate first derivatives basis functions at all reference points
  std::array<std::size_t, 4> basis_shape = cmap.tabulate_shape(1, shape[0]);
  assert(basis_shape.back() == 1);
  std::vector<double> cmap_basisb(std::reduce(
      basis_shape.cbegin(), basis_shape.cend(), 1, std::multiplies{}));
  cmap.tabulate(1, q_points, shape, cmap_basisb);

  // Get facet normals on reference cell
  basix::cell::type cell_type
      = dolfinx::mesh::cell_type_to_basix_type(topology->cell_type());
  auto [facet_normalsb, n_shape]
      = basix::cell::facet_outward_normals<double>(cell_type);
  mdspan_t<const double, 2> facet_normals(facet_normalsb.data(), n_shape);

  // Working memory for loop
  std::vector<double> coordinate_dofsb(num_dofs_g * gdim);
  mdspan_t<const double, 2> coordinate_dofs(coordinate_dofsb.data(), num_dofs_g,
                                            gdim);
  std::array<double, 9> Jb;
  std::array<double, 9> Kb;
  mdspan_t<double, 2> J(Jb.data(), gdim, tdim);
  mdspan_t<double, 2> K(Kb.data(), tdim, gdim);
  mdspan_t<double, 4> full_basis(cmap_basisb.data(), basis_shape);

  // Loop over quadrature points
  for (std::size_t i = 0; i < quadrature_facets.size(); i += 2)
  {
    // Copy coordinate dofs of candidate cell
    // Get cell geometry (coordinate dofs)
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, quadrature_facets[i],
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    assert(x_dofs.size() == num_dofs_g);
    for (std::size_t j = 0; j < num_dofs_g; ++j)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[j]), gdim,
                  std::next(coordinate_dofsb.begin(), j * gdim));
    }

    for (std::size_t q = 0; q < num_q_points; ++q)
    {
      auto dphi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          full_basis, std::pair{1, tdim + 1},
          quadrature_facets[i + 1] * num_q_points + q,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

      // Compute Jacobian and Jacobian inverse for Piola mapping of normal
      std::fill(Jb.begin(), Jb.end(), 0);
      dolfinx::fem::CoordinateElement<double>::compute_jacobian(
          dphi, coordinate_dofs, J);
      std::fill(Kb.begin(), Kb.end(), 0);
      dolfinx::fem::CoordinateElement<double>::compute_jacobian_inverse(J, K);

      // Push forward normal using covariant Piola
      physical_facet_normal(
          std::span(normals.data() + i / 2 * cstride + q * gdim, gdim), K,
          MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              facet_normals, quadrature_facets[i + 1],
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent));
    }
  }

  return {std::move(normals), cstride};
}
//------------------------------------------------------------------------------------------------
dolfinx_contact::kernel_fn<PetscScalar>
Contact::generate_kernel(Kernel type,
                         const dolfinx::fem::FunctionSpace<double>& V) const
{
  std::size_t max_links
      = *std::max_element(_max_links.begin(), _max_links.end());
  return generate_contact_kernel(type, V, _quadrature_rule, max_links);
}
//----------------------------------------------------------------------------
void Contact::create_q_phys(int origin_meshtag)
{
  // Get information depending on surface

  std::vector<std::int32_t> submesh_facets
      = _submesh.get_submesh_tuples(_cell_facet_pairs.links(origin_meshtag));
  auto mesh_sub = _submesh.mesh();
  std::size_t gdim = mesh_sub->geometry().dim();
  const std::vector<size_t>& qp_offsets = _quadrature_rule.offset();
  _qp_phys[origin_meshtag].resize((qp_offsets[1] - qp_offsets[0])
                                  * (submesh_facets.size() / 2) * gdim);
  compute_physical_points(
      *mesh_sub, submesh_facets, qp_offsets,
      mdspan_t<const double, 4>(_reference_basis.data(), _reference_shape),
      _qp_phys[origin_meshtag]);
}
//----------------------------------------------------------------------------
void dolfinx_contact::Contact::max_links(int pair)
{
  std::size_t max_links = 0;

  // Select which side of the contact interface to loop from and get the
  // correct map
  std::array<int, 2> contact_pair = _contact_pairs[pair];
  std::span<const std::int32_t> active_facets
      = _cell_facet_pairs.links(contact_pair.front());
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> map
      = _facet_maps[pair];
  assert(map);
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
      = _submesh.facet_map();
  assert(facet_map);
  for (std::size_t i = 0; i < active_facets.size(); i += 2)
  {
    std::vector<std::int32_t> linked_cells;
    for (auto link : map->links(i / 2))
    {
      if (link >= 0)
      {
        auto facet_pair = facet_map->links(link);
        linked_cells.push_back(facet_pair.front());
      }
    }

    // Remove duplicates
    std::sort(linked_cells.begin(), linked_cells.end());
    linked_cells.erase(std::unique(linked_cells.begin(), linked_cells.end()),
                       linked_cells.end());
    max_links = std::max(max_links, linked_cells.size());
  }

  _max_links[pair] = max_links;
}
//------------------------------------------------------------------------------------------------
std::pair<std::vector<PetscScalar>, int> Contact::pack_gap(int pair) const
{
  // FIXME: This function should take in the quadrature points
  // (push_forward_quadrature) of the relevant facet, and the reference
  // points on the other surface (output of distance map)
  auto [quadrature_mt, candidate_mt] = _contact_pairs[pair];

  // Retrieve submeshes
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> quadrature_mesh
      = _submesh.mesh();
  assert(quadrature_mesh);
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> candidate_mesh
      = _submesh.mesh();
  assert(candidate_mesh);

  // Determine coefficient size and allocate coefficient memory
  std::size_t num_facets = _local_facets[quadrature_mt];

  // NOTE: Assumes same number of quadrature points on all facets
  error::check_cell_type(candidate_mesh->topology()->cell_type());
  std::size_t num_q_point
      = _quadrature_rule.offset()[1] - _quadrature_rule.offset()[0];
  const dolfinx::mesh::Geometry<double>& geometry = candidate_mesh->geometry();
  int gdim = geometry.dim();

  // Pack gap function for each quadrature point on each facet
  std::vector<PetscScalar> c(num_facets * num_q_point * gdim, 0.0);
  int cstride = num_q_point * gdim;

  // return if no facets on process
  if (num_facets == 0)
    return {std::move(c), cstride};

  // Get (cell, local_facet_index) tuples on quadrature submesh
  std::vector<std::int32_t> quadrature_facets = _submesh.get_submesh_tuples(
      _cell_facet_pairs.links(quadrature_mt).subspan(0, 2 * num_facets));

  // Get (cell, local_facet_index) tuples on candidate submesh
  std::vector<std::int32_t> candidate_facets
      = _submesh.get_submesh_tuples(_cell_facet_pairs.links(candidate_mt));

  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      candidate_map = _facet_maps[pair];
  const std::vector<double>& reference_x = _reference_contact_points[pair];
  std::array<std::size_t, 2> shape = _reference_contact_shape[pair];

  int q_gdim = quadrature_mesh->geometry().dim();
  mdspan_t<const double, 3> qp_span(_qp_phys[quadrature_mt].data(), num_facets,
                                    num_q_point, q_gdim);

  // Get information about submesh geometry and topology
  std::span<const double> x_g = geometry.x();
  auto x_dofmap = geometry.dofmap();
  const dolfinx::fem::CoordinateElement<double>& cmap = geometry.cmap();
  std::size_t num_dofs_g = cmap.dim();
  auto topology = candidate_mesh->topology();
  int tdim = topology->dim();

  std::vector<double> coordinate_dofsb(num_dofs_g * gdim);
  mdspan_t<const double, 2> coordinate_dofs(coordinate_dofsb.data(), num_dofs_g,
                                            gdim);
  std::array<double, 3> coordb;
  mdspan_t<double, 2> coord(coordb.data(), 1, gdim);

  auto f_to_c = candidate_mesh->topology()->connectivity(tdim - 1, tdim);
  if (!f_to_c)
  {
    throw std::runtime_error("Missing facet to cell connectivity on "
                             "candidate submesh");
  }

  std::array<std::size_t, 4> basis_shape = cmap.tabulate_shape(0, shape[0]);
  assert(basis_shape.back() == 1);
  std::vector<double> cmap_basis(std::reduce(
      basis_shape.begin(), basis_shape.end(), 1, std::multiplies{}));
  cmap.tabulate(0, reference_x, shape, cmap_basis);

  mdspan_t<const double, 4> full_basis(cmap_basis.data(), basis_shape);
  for (std::size_t i = 0; i < num_facets; ++i)
  {
    int offset = i * cstride;
    auto facets = candidate_map->links(i);
    assert(facets.size() == num_q_point);
    for (std::size_t q = 0; q < num_q_point; ++q)
    {
      // Skip negative facet indices (No facet on opposite surface has
      // been found)
      if (facets[q] < 0)
        continue;

      auto candidate_cells = f_to_c->links(facets[q]);
      assert(candidate_cells.size() == 1);

      // Copy coordinate dofs of candidate cell
      // Get cell geometry (coordinate dofs)
      auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, candidate_cells.front(),
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      assert(x_dofs.size() == num_dofs_g);
      for (std::size_t j = 0; j < num_dofs_g; ++j)
      {
        std::copy_n(std::next(x_g.begin(), 3 * x_dofs[j]), gdim,
                    std::next(coordinate_dofsb.begin(), j * gdim));
      }

      auto basis_q = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          full_basis, 0,
          std::pair{i * num_q_point + q, i * num_q_point + q + 1},
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

      dolfinx::fem::CoordinateElement<double>::push_forward(
          coord, coordinate_dofs, basis_q);
      for (int k = 0; k < gdim; k++)
        c[offset + q * gdim + k] = coordb[k] - qp_span(i, q, k);
    }
  }

  return {c, cstride};
}
//------------------------------------------------------------------------------------------------
std::pair<std::vector<PetscScalar>, int>
Contact::pack_test_functions(int pair,
                             const dolfinx::fem::FunctionSpace<double>& V) const
{
  auto [quadrature_mt, candidate_mt] = _contact_pairs[pair];

  // Get mesh info for candidate side
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> candidate_mesh
      = _submesh.mesh();
  assert(candidate_mesh);
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> quadrature_mesh
      = _submesh.mesh();
  assert(quadrature_mesh);

  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> sub_to_parent
      = _submesh.facet_map();
  std::span<const std::int32_t> parent_cells = _submesh.parent_cells();

  // Get (cell, local_facet_index) tuples on quadrature submesh
  std::size_t num_facets = _local_facets[quadrature_mt];

  // Get (cell, local_facet_index) tuples on quadrature submesh
  std::vector<std::int32_t> quadrature_facets = _submesh.get_submesh_tuples(
      _cell_facet_pairs.links(quadrature_mt).subspan(0, 2 * num_facets));

  // Get (cell, local_facet_index) tuples on candidate submesh
  std::vector<std::int32_t> candidate_facets
      = _submesh.get_submesh_tuples(_cell_facet_pairs.links(candidate_mt));

  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      candidate_map = _facet_maps[pair];
  const std::vector<double>& reference_x = _reference_contact_points[pair];
  std::array<std::size_t, 2> shape = _reference_contact_shape[pair];

  // Compute values of basis functions for all y = Pi(x) in qp
  auto V_sub = std::make_shared<dolfinx::fem::FunctionSpace<double>>(
      _submesh.create_functionspace(V));

  std::shared_ptr<const dolfinx::fem::FiniteElement<double>> element
      = V_sub->element();
  std::array<std::size_t, 4> b_shape
      = element->basix_element().tabulate_shape(0, shape[0]);
  if (b_shape.back() > 1)
    throw std::invalid_argument("pack_test_functions assumes values size 1");

  // FIXME: Aim to remove this as it depends on the state of the contact
  // algorithm
  std::size_t max_links
      = *std::max_element(_max_links.begin(), _max_links.end());
  std::size_t num_q_points
      = _quadrature_rule.offset()[1] - _quadrature_rule.offset()[0];
  std::size_t bs = element->block_size();
  int cstride = num_q_points * max_links * b_shape[2] * bs;
  std::vector<PetscScalar> cb(
      num_facets * max_links * num_q_points * b_shape[2] * bs, 0.0);
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      PetscScalar, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 5>>
      c(cb.data(), num_facets, max_links, b_shape[2], num_q_points, bs);

  // return if no facets on process
  if (num_facets == 0)
    return {std::move(cb), cstride};

  std::vector<double> basis_valuesb(
      std::reduce(b_shape.cbegin(), b_shape.cend(), 1, std::multiplies{}));
  element->tabulate(basis_valuesb, reference_x, shape, 0);
  mdspan_t<const double, 4> basis_values(basis_valuesb.data(), b_shape);

  // Need to apply push forward and dof transformations to test
  // functions
  assert(b_shape.front() == 1 and b_shape.back() == 1);
  if (const basix::FiniteElement<double>& b_el = element->basix_element();
      element->needs_dof_transformations()
      or b_el.map_type() != basix::maps::type::identity)
  {
    // If we want to do this we need to apply transformation and push
    // forward
    throw std::runtime_error(
        "Packing basis (test) functions of space that uses "
        "non-indentity maps is not supported");
  }

  // Convert facet index on candidate mesh into cell index
  auto topology = candidate_mesh->topology();
  int tdim = topology->dim();
  auto f_to_c = topology->connectivity(tdim - 1, tdim);
  assert(f_to_c);
  const std::vector<std::int32_t>& facets = candidate_map->array();
  error::check_cell_type(topology->cell_type());
  assert(num_facets * num_q_points == facets.size());
  std::vector<std::int32_t> cells(facets.size(), -1);
  for (std::size_t i = 0; i < cells.size(); ++i)
  {
    if (facets[i] >= 0)
    {
      // Extract (cell, facet) pair from submesh
      auto facet_pair = sub_to_parent->links(facets[i]);
      assert(facet_pair.size() == 2);
      cells[i] = parent_cells[facet_pair[0]];
    }
    else
      cells[i] = -1;
  }

  std::vector<std::int32_t> perm(num_q_points);
  for (std::size_t i = 0; i < c.extent(0); ++i)
  {
    std::span<const std::int32_t> f_cells(cells.data() + i * num_q_points,
                                          num_q_points);
    // As radix sort is broken for negative numbers, we need to filter them out                                          
    std::vector<std::int32_t> positive_cells;
    std::vector<std::int32_t> positive_to_old;
    positive_cells.reserve(f_cells.size());
    positive_to_old.reserve(f_cells.size());
    for (std::size_t j = 0; j< f_cells.size(); ++j)
    {
      if (f_cells[j] >= 0)
      {
        positive_cells.push_back(f_cells[j]);
        positive_to_old.push_back(j);
      }
    } 
    auto [unique_cells, offsets] = sort_cells(positive_cells, std::span(perm.data(), positive_cells.size()));
    std::int32_t link = 0;
    for (std::size_t j = 0; j < unique_cells.size(); ++j)
    {
      assert(unique_cells[j] >= 0);
      auto indices
          = std::span(perm.data() + offsets[j], offsets[j + 1] - offsets[j]);
      assert(positive_cells.size() >= (std::size_t)offsets[j + 1]);
      for (std::size_t k = 0; k < c.extent(2); ++k)
      {
        for (std::size_t q = 0; q < indices.size(); ++q)
        {
          for (std::size_t l = 0; l < c.extent(4); ++l)
          {
            c(i, link, k, positive_to_old[indices[q]], l)
                = basis_values(0, i * num_q_points + indices[q], k, 0);
          }
        }
      }
      link += 1;
    }
  }

  return {std::move(cb), cstride};
}
//------------------------------------------------------------------------------------------------
void Contact::crop_invalid_points(std::size_t pair, std::span<const double> gap,
                                  std::span<const double> n_y, double tol)
{
  auto [quadrature_mt, candidate_mt] = _contact_pairs[pair];
  std::size_t num_facets = _local_facets[quadrature_mt];
  std::size_t num_q_points
      = _quadrature_rule.offset()[1] - _quadrature_rule.offset()[0];
  std::size_t gdim = _mesh->geometry().dim();
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      candidate_map = _facet_maps[pair];
  std::vector<std::int32_t> offsets(candidate_map->offsets());
  std::vector<std::int32_t> data(candidate_map->array());
  for (std::size_t f = 0; f < num_facets; ++f)
  {
    for (std::size_t q = 0; q < num_q_points; ++q)
    {
      double dot = 0;
      double norm = 0;
      for (std::size_t i = 0; i < gdim; ++i)
      {
        std::size_t index = f * num_q_points * gdim + q * gdim + i;
        dot += gap[index] * n_y[index];
        norm += gap[index] * gap[index];
      }

      norm = std::sqrt(norm);
      if (norm > tol)
        data[offsets[f] + q] = -1;
      else if (norm > 1e-7)
      {
        dot = std::abs(dot) / norm;
        if (dot < 0.7)
          data[offsets[f] + q] = -1;
      }
    }
  }

  _facet_maps[pair]
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(data,
                                                                      offsets);
}
//------------------------------------------------------------------------------------------------
std::pair<std::vector<PetscScalar>, int>
Contact::pack_u_contact(int pair,
                        const dolfinx::fem::Function<PetscScalar>& u) const
{
  dolfinx::common::Timer t("Pack contact u");
  auto [quadrature_mt, candidate_mt] = _contact_pairs[pair];

  // Get mesh info for candidate side
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> candidate_mesh
      = _submesh.mesh();
  assert(candidate_mesh);
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> quadrature_mesh
      = _submesh.mesh();
  assert(quadrature_mesh);

  const std::size_t num_facets = _local_facets[quadrature_mt];

  // copy u onto submesh
  auto V_sub = std::make_shared<dolfinx::fem::FunctionSpace<double>>(
      _submesh.create_functionspace(*u.function_space()));
  std::shared_ptr<const dolfinx::fem::FiniteElement<double>> element
      = V_sub->element();
  auto topology = candidate_mesh->topology();
  error::check_cell_type(topology->cell_type());
  std::size_t bs_element = element->block_size();
  dolfinx::fem::Function<PetscScalar> u_sub(V_sub);
  std::shared_ptr<const dolfinx::fem::DofMap> sub_dofmap = V_sub->dofmap();
  assert(sub_dofmap);
  int bs_dof = sub_dofmap->bs();
  _submesh.copy_function(u, u_sub);

  // Output vector
  std::size_t num_q_points
      = _quadrature_rule.offset()[1] - _quadrature_rule.offset()[0];
  std::vector<PetscScalar> c(num_facets * num_q_points * bs_element, 0.0);
  int cstride = num_q_points * bs_element;

  // return if no facets on process
  if (num_facets == 0)
  {
    t.stop();
    return {std::move(c), cstride};
  }

  // Get (cell, local_facet_index) tuples on quadrature submesh
  std::vector<std::int32_t> quadrature_facets = _submesh.get_submesh_tuples(
      _cell_facet_pairs.links(quadrature_mt).subspan(0, 2 * num_facets));

  // Get (cell, local_facet_index) tuples on candidate submesh
  std::vector<std::int32_t> candidate_facets
      = _submesh.get_submesh_tuples(_cell_facet_pairs.links(candidate_mt));

  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      candidate_map = _facet_maps[pair];
  const std::vector<double>& reference_x = _reference_contact_points[pair];
  std::array<std::size_t, 2> shape = _reference_contact_shape[pair];

  // Compute values of basis functions for all y = Pi(x) in qp
  std::array<std::size_t, 4> b_shape
      = element->basix_element().tabulate_shape(0, shape[0]);
  if (b_shape.back() > 1)
    throw std::invalid_argument("pack_test_functions assumes values size 1");
  std::vector<double> basis_valuesb(
      std::reduce(b_shape.cbegin(), b_shape.cend(), 1, std::multiplies{}));
  element->tabulate(basis_valuesb, reference_x, shape, 0);

  // Need to apply push forward and dof transformations to test functions
  assert((b_shape.front() == 1) and (b_shape.back() == 1));
  if (const basix::FiniteElement<double>& b_el = element->basix_element();
      element->needs_dof_transformations()
      or b_el.map_type() != basix::maps::type::identity)
  {
    // If we want to do this we need to apply transformation and push
    // forward
    throw std::runtime_error(
        "Packing u on opposite surface functions of space that uses "
        "non-indentity maps is not supported");
  }

  mdspan_t<const double, 4> basis_values(basis_valuesb.data(), b_shape);
  std::span<const PetscScalar> u_coeffs = u_sub.x()->array();

  // Get cell index on sub-mesh
  int tdim = topology->dim();
  auto f_to_c = topology->connectivity(tdim - 1, tdim);
  assert(f_to_c);
  const std::vector<std::int32_t>& facets = candidate_map->array();
  std::vector<std::int32_t> cells(facets.size(), -1);
  for (std::size_t i = 0; i < cells.size(); ++i)
  {
    if (facets[i] < 0)
      continue;
    auto f_cells = f_to_c->links(facets[i]);
    assert(f_cells.size() == 1);
    cells[i] = f_cells.front();
  }

  // Create work vector for expansion coefficients
  std::size_t num_basis_functions = b_shape[2];
  std::size_t value_size = b_shape[3];
  std::vector<PetscScalar> coefficients(num_basis_functions * bs_element);
  for (std::size_t i = 0; i < num_facets; ++i)
  {
    for (std::size_t q = 0; q < num_q_points; ++q)
    {
      // Get degrees of freedom for current cell
      if (facets[i * num_q_points + q] < 0)
        continue;

      auto dofs = sub_dofmap->cell_dofs(cells[i * num_q_points + q]);
      for (std::size_t j = 0; j < dofs.size(); ++j)
        for (int k = 0; k < bs_dof; ++k)
          coefficients[bs_dof * j + k] = u_coeffs[bs_dof * dofs[j] + k];

      // Compute expansion
      for (std::size_t k = 0; k < bs_element; ++k)
      {
        for (std::size_t l = 0; l < num_basis_functions; ++l)
        {
          for (std::size_t m = 0; m < value_size; ++m)
          {
            c[cstride * i + q * bs_element + k]
                += coefficients[bs_element * l + k]
                   * basis_values(0, num_q_points * i + q, l, m);
          }
        }
      }
    }
  }

  t.stop();
  return {std::move(c), cstride};
}
//------------------------------------------------------------------------------------------------
std::pair<std::vector<PetscScalar>, int> Contact::pack_gap_plane(int pair,
                                                                 double g)
{
  int quadrature_mt = _contact_pairs[pair][0];
  int gdim = _mesh->geometry().dim(); // geometrical dimension

  // Tabulate basis function on reference cell (_phi_ref_facets)
  const dolfinx::fem::CoordinateElement<double>& cmap
      = _mesh->geometry().cmap();
  std::tie(_reference_basis, _reference_shape)
      = tabulate(cmap, _quadrature_rule);

  // Compute quadrature points on physical facet
  // _qp_phys_"quadrature_mt"
  create_q_phys(quadrature_mt);
  std::size_t num_facets = _local_facets[quadrature_mt];

  // FIXME: This does not work for prism meshes
  std::size_t num_q_point
      = _quadrature_rule.offset()[1] - _quadrature_rule.offset()[0];
  mdspan_t<const double, 3> qp_span(_qp_phys[quadrature_mt].data(), num_facets,
                                    num_q_point, gdim);
  std::vector<PetscScalar> c(num_facets * num_q_point * gdim, 0.0);
  int cstride = num_q_point * gdim;
  for (std::size_t i = 0; i < num_facets; i++)
  {
    int offset = i * cstride;
    for (std::size_t k = 0; k < num_q_point; k++)
      c[offset + (k + 1) * gdim - 1] = g - qp_span(i, k, gdim - 1);
  }

  return {std::move(c), cstride};
}
//----------------------------------------------------------------------------
std::pair<std::vector<PetscScalar>, int>
dolfinx_contact::Contact::pack_ny(int pair) const
{
  // FIXME: This function should take in the quadrature points
  // (push_forward_quadrature) of the relevant facet, and the reference
  // points on the other surface (output of distance map)
  auto [quadrature_mt, candidate_mt] = _contact_pairs[pair];

  // Get mesh info for candidate side
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> candidate_mesh
      = _submesh.mesh();
  assert(candidate_mesh);
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> quadrature_mesh
      = _submesh.mesh();
  assert(quadrature_mesh);

  std::size_t num_facets = _local_facets[quadrature_mt];

  // create coefficient vector
  const dolfinx::mesh::Geometry<double>& geometry = candidate_mesh->geometry();
  int gdim = geometry.dim();
  std::size_t num_q_points = _quadrature_rule.num_points(0);
  std::vector<PetscScalar> normals(num_facets * num_q_points * gdim, 0.0);
  int cstride = num_q_points * gdim;

  // return if no facets on process
  if (num_facets == 0)
    return {std::move(normals), cstride};

  // Get (cell, local_facet_index) tuples on quadrature submesh
  std::vector<std::int32_t> quadrature_facets = _submesh.get_submesh_tuples(
      _cell_facet_pairs.links(quadrature_mt).subspan(0, 2 * num_facets));

  // Get (cell, local_facet_index) tuples on candidate submesh
  std::vector<std::int32_t> candidate_facets
      = _submesh.get_submesh_tuples(_cell_facet_pairs.links(candidate_mt));

  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
      candidate_map = _facet_maps[pair];
  const std::vector<double>& reference_x = _reference_contact_points[pair];
  std::array<std::size_t, 2> shape = _reference_contact_shape[pair];

  // Get information about submesh geometry and topology
  std::span<const double> x_g = geometry.x();
  auto x_dofmap = geometry.dofmap();
  const dolfinx::fem::CoordinateElement<double>& cmap = geometry.cmap();
  std::size_t num_dofs_g = cmap.dim();
  auto topology = candidate_mesh->topology();
  int tdim = topology->dim();

  // Tabulate first derivatives basis functions at all reference points
  std::array<std::size_t, 4> basis_shape = cmap.tabulate_shape(1, shape[0]);
  assert(basis_shape.back() == 1);
  std::vector<double> cmap_basisb(std::reduce(
      basis_shape.cbegin(), basis_shape.cend(), 1, std::multiplies{}));
  cmap.tabulate(1, reference_x, shape, cmap_basisb);

  // Loop over quadrature points
  error::check_cell_type(candidate_mesh->topology()->cell_type());

  auto f_to_c = candidate_mesh->topology()->connectivity(tdim - 1, tdim);
  if (!f_to_c)
  {
    throw std::runtime_error("Missing facet to cell connectivity on "
                             "candidate submesh");
  }

  auto c_to_f = candidate_mesh->topology()->connectivity(tdim, tdim - 1);
  if (!c_to_f)
  {
    throw std::runtime_error("Missing cell to facet connectivity on "
                             "candidate submesh");
  }

  // Get facet normals on reference cell
  basix::cell::type cell_type = dolfinx::mesh::cell_type_to_basix_type(
      candidate_mesh->topology()->cell_type());
  auto [facet_normalsb, n_shape]
      = basix::cell::facet_outward_normals<double>(cell_type);
  mdspan_t<const double, 2> facet_normals(facet_normalsb.data(), n_shape);

  // Working memory for loop
  std::vector<double> coordinate_dofsb(num_dofs_g * gdim);
  mdspan_t<const double, 2> coordinate_dofs(coordinate_dofsb.data(), num_dofs_g,
                                            gdim);
  std::array<double, 9> Jb;
  std::array<double, 9> Kb;
  mdspan_t<double, 2> J(Jb.data(), gdim, tdim);
  mdspan_t<double, 2> K(Kb.data(), tdim, gdim);
  mdspan_t<double, 4> full_basis(cmap_basisb.data(), basis_shape);
  for (std::size_t i = 0; i < num_facets; ++i)
  {
    auto facets = candidate_map->links(i);
    assert(facets.size() == num_q_points);
    for (std::size_t q = 0; q < num_q_points; ++q)
    {
      // Skip computation if quadrature point does not have a matching
      // facet on the other side
      if (facets[q] < 0)
        continue;

      auto candidate_cells = f_to_c->links(facets[q]);
      assert(candidate_cells.size() == 1);
      assert(candidate_cells.front() >= 0);

      // Get local facet index of candidate facet
      auto local_facets = c_to_f->links(candidate_cells.front());
      auto it = std::find(local_facets.begin(), local_facets.end(), facets[q]);
      int local_idx = std::distance(local_facets.begin(), it);

      // Copy coordinate dofs of candidate cell
      // Get cell geometry (coordinate dofs)
      auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, candidate_cells.front(),
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      assert(x_dofs.size() == num_dofs_g);
      for (std::size_t j = 0; j < num_dofs_g; ++j)
      {
        std::copy_n(std::next(x_g.begin(), 3 * x_dofs[j]), gdim,
                    std::next(coordinate_dofsb.begin(), j * gdim));
      }

      auto dphi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          full_basis, std::pair{1, tdim + 1}, i * num_q_points + q,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

      // Compute Jacobian and Jacobian inverse for Piola mapping of normal
      std::fill(Jb.begin(), Jb.end(), 0);
      dolfinx::fem::CoordinateElement<double>::compute_jacobian(
          dphi, coordinate_dofs, J);
      std::fill(Kb.begin(), Kb.end(), 0);
      dolfinx::fem::CoordinateElement<double>::compute_jacobian_inverse(J, K);

      // Push forward normal using covariant Piola
      physical_facet_normal(
          std::span(normals.data() + i * cstride + q * gdim, gdim), K,
          MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              facet_normals, local_idx,
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent));
    }
  }

  return {std::move(normals), cstride};
}
//----------------------------------------------------------------------------
void Contact::assemble_matrix(mat_set_fn& mat_set, int pair,
                              const kernel_fn<PetscScalar>& kernel,
                              std::span<const PetscScalar> coeffs, int cstride,
                              std::span<const PetscScalar> constants,
                              const dolfinx::fem::FunctionSpace<double>& V)
{
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh = V.mesh();
  assert(mesh);

  // Extract geometry data
  const dolfinx::mesh::Geometry<double>& geometry = mesh->geometry();
  int gdim = geometry.dim();
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = geometry.dofmap();
  std::span<const double> x_g = geometry.x();
  const dolfinx::fem::CoordinateElement<double>& cmap = geometry.cmap();
  std::size_t num_dofs_g = cmap.dim();
  if (V.element()->needs_dof_transformations())
  {
    throw std::invalid_argument(
        "Function-space requiring dof-transformations is not supported.");
  }

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V.dofmap();
  std::size_t ndofs_cell = dofmap->cell_dofs(0).size();
  int bs = dofmap->bs();
  std::size_t max_links
      = *std::max_element(_max_links.begin(), _max_links.end());
  if (max_links == 0)
  {
    spdlog::warn(
        "No links between interfaces, compute_linked_cell will be skipped");
  }

  std::array<int, 2> contact_pair = _contact_pairs[pair];
  std::span<const std::int32_t> active_facets
      = _cell_facet_pairs.links(contact_pair.front());
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> map
      = _facet_maps[pair];
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
      = _submesh.facet_map();
  assert(facet_map);

  std::span<const std::int32_t> parent_cells = _submesh.parent_cells();

  // Data structures used in assembly
  std::vector<double> coordinate_dofs(3 * num_dofs_g);
  std::vector<std::vector<PetscScalar>> Aes(
      3 * max_links + 1,
      std::vector<PetscScalar>(bs * ndofs_cell * bs * ndofs_cell));
  std::vector<std::int32_t> linked_cells;
  for (std::size_t i = 0; i < 2 * _local_facets[contact_pair.front()]; i += 2)
  {
    // Get cell coordinates/geometry
    assert(std::size_t(active_facets[i]) < x_dofmap.extent(0));
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, active_facets[i],
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t j = 0; j < x_dofs.size(); ++j)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[j]), gdim,
                  std::next(coordinate_dofs.begin(), j * 3));
    }

    // Compute what quadrature points to integrate over (which ones has
    // corresponding facets on other surface)
    std::vector<std::int32_t> q_indices;
    if (max_links > 0)
    {
      assert(map);
      auto connected_facets = map->links(i / 2);
      q_indices.reserve(connected_facets.size());

      // NOTE: Should probably be pre-computed
      for (std::size_t j = 0; j < connected_facets.size(); ++j)
        if (connected_facets[j] >= 0)
          q_indices.push_back(j);

      // Compute the unique set of cells linked to the current facet
      compute_linked_cells(linked_cells, connected_facets, *facet_map,
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

    kernel(Aes, std::span(coeffs.data() + i / 2 * cstride, cstride),
           constants.data(), coordinate_dofs.data(), active_facets[i + 1],
           num_linked_cells, q_indices);

    // FIXME: We would have to handle possible Dirichlet conditions
    // here, if we think that we can have a case with contact and
    // Dirichlet
    auto dmap_cell = dofmap->cell_dofs(active_facets[i]);
    mat_set(dmap_cell, dmap_cell, Aes[0]);
    for (std::size_t j = 0; j < num_linked_cells; j++)
    {
      if (linked_cells[j] < 0)
        continue;
      auto dmap_linked = dofmap->cell_dofs(linked_cells[j]);
      assert(!dmap_linked.empty());
      mat_set(dmap_cell, dmap_linked, Aes[3 * j + 1]);
      mat_set(dmap_linked, dmap_cell, Aes[3 * j + 2]);
      mat_set(dmap_linked, dmap_linked, Aes[3 * j + 3]);
    }
  }
}
//----------------------------------------------------------------------------
void Contact::assemble_vector(std::span<PetscScalar> b, int pair,
                              const kernel_fn<PetscScalar>& kernel,
                              std::span<const PetscScalar> coeffs, int cstride,
                              std::span<const PetscScalar> constants,
                              const dolfinx::fem::FunctionSpace<double>& V)
{
  /// Check that we support the function space
  if (V.element()->needs_dof_transformations())
  {
    throw std::invalid_argument(
        "Function-space requiring dof-transformations is not supported.");
  }

  // Extract mesh
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh = V.mesh();
  assert(mesh);
  const dolfinx::mesh::Geometry<double>& geometry = mesh->geometry();
  int gdim = geometry.dim(); // geometrical dimension

  // Prepare cell geometry
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = geometry.dofmap();
  std::span<const double> x_g = geometry.x();

  const dolfinx::fem::CoordinateElement<double>& cmap = geometry.cmap();
  std::size_t num_dofs_g = cmap.dim();

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V.dofmap();
  std::size_t ndofs_cell = dofmap->cell_dofs(0).size();
  int bs = dofmap->bs();

  // Select which side of the contact interface to loop from and get the
  // correct map
  std::array<int, 2> contact_pair = _contact_pairs[pair];
  std::span<const std::int32_t> active_facets
      = _cell_facet_pairs.links(contact_pair.front());
  std::size_t local_size = 2 * _local_facets[contact_pair.front()];
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> map
      = _facet_maps[pair];
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
      = _submesh.facet_map();
  assert(facet_map);
  std::span<const std::int32_t> parent_cells = _submesh.parent_cells();
  std::size_t max_links
      = *std::max_element(_max_links.begin(), _max_links.end());
  if (max_links == 0)
  {
    spdlog::warn(
        "No links between interfaces, compute_linked_cell will be skipped");
  }

  // Data structures used in assembly
  std::vector<double> coordinate_dofs(3 * num_dofs_g);
  std::vector<std::vector<PetscScalar>> bes(
      max_links + 1, std::vector<PetscScalar>(bs * ndofs_cell));

  // Tempoary array to hold cell links
  std::vector<std::int32_t> linked_cells;
  for (std::size_t i = 0; i < local_size; i += 2)
  {
    // Get cell coordinates/geometry
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, active_facets[i],
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t j = 0; j < x_dofs.size(); ++j)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[j]), gdim,
                  std::next(coordinate_dofs.begin(), j * 3));
    }

    // Compute what quadrature points to integrate over (which ones has
    // corresponding facets on other surface)
    std::vector<std::int32_t> q_indices;

    // Compute the unique set of cells linked to the current facet
    if (max_links > 0)
    {
      assert(map);
      auto connected_facets = map->links(i / 2);
      q_indices.reserve(connected_facets.size());

      // NOTE: Should probably be pre-computed
      for (std::size_t j = 0; j < connected_facets.size(); ++j)
        if (connected_facets[j] >= 0)
          q_indices.push_back(j);

      compute_linked_cells(linked_cells, connected_facets, *facet_map,
                           parent_cells);
    }

    // Using integer loop here to reduce number of zeroed vectors
    std::size_t num_linked_cells = linked_cells.size();
    std::fill(bes[0].begin(), bes[0].end(), 0);
    for (std::size_t j = 0; j < num_linked_cells; j++)
      std::fill(bes[j + 1].begin(), bes[j + 1].end(), 0);

    kernel(bes, std::span(coeffs.data() + i / 2 * cstride, cstride),
           constants.data(), coordinate_dofs.data(), active_facets[i + 1],
           num_linked_cells, q_indices);

    // Add element vector to global vector
    std::span<const int> dofs_cell = dofmap->cell_dofs(active_facets[i]);
    for (std::size_t j = 0; j < ndofs_cell; ++j)
      for (int k = 0; k < bs; ++k)
        b[bs * dofs_cell[j] + k] += bes[0][bs * j + k];

    for (std::size_t l = 0; l < num_linked_cells; ++l)
    {
      std::span<const int> dofs_linked = dofmap->cell_dofs(linked_cells[l]);
      for (std::size_t j = 0; j < ndofs_cell; ++j)
        for (int k = 0; k < bs; ++k)
          b[bs * dofs_linked[j] + k] += bes[l + 1][bs * j + k];
    }
  }
}
//-----------------------------------------------------------------------------------------------
std::pair<std::vector<PetscScalar>, int> Contact::pack_grad_test_functions(
    int pair, const dolfinx::fem::FunctionSpace<double>& V) const
{
  auto [quadrature_mt, candidate_mt] = _contact_pairs[pair];

  // Mesh info
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh = V.mesh(); // mesh
  assert(mesh);
  std::size_t gdim = mesh->geometry().dim();
  std::size_t tdim = mesh->topology()->dim();
  std::span<const std::int32_t> parent_cells = _submesh.parent_cells();
  std::shared_ptr<const fem::FiniteElement<double>> element = V.element();
  assert(element);
  int bs_element = element->block_size();
  std::size_t ndofs = element->space_dimension() / bs_element;

  // Select which side of the contact interface to loop from and get the
  // correct map
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> map
      = _facet_maps[pair];
  std::size_t num_facets = _local_facets[quadrature_mt];
  std::size_t num_q_points
      = _quadrature_rule.offset()[1] - _quadrature_rule.offset()[0];
  std::vector<double> q_points(std::size_t(num_q_points) * std::size_t(gdim));
  mdspan_t<const double, 3> qp_span(_qp_phys[quadrature_mt].data(), num_facets,
                                    num_q_points, gdim);

  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
      = _submesh.facet_map();
  std::size_t max_links
      = *std::max_element(_max_links.begin(), _max_links.end());

  std::vector<std::int32_t> perm(num_q_points);
  std::vector<std::int32_t> linked_cells(num_q_points);

  // Create output vector
  std::vector<PetscScalar> c(
      num_facets * num_q_points * max_links * ndofs * gdim, 0.0);
  int cstride = num_q_points * max_links * ndofs * gdim;

  // return if no facets on process
  if (num_facets == 0)
    return {std::move(c), cstride};

  // temporary data structure used inside loop
  std::vector<std::int32_t> cells(max_links, -1);
  // Loop over all facets
  const std::vector<double>& reference_x = _reference_contact_points[pair];
  for (std::size_t i = 0; i < num_facets; i++)
  {
    std::span<const int> links = map->links(i);
    assert(links.size() == num_q_points);
    for (std::size_t j = 0; j < num_q_points; j++)
    {
      if (links[j] < 0)
        linked_cells[j] = -1;
      else
      {
        std::span<const int> linked_pair = facet_map->links(links[j]);
        assert(!linked_pair.empty());
        linked_cells[j] = linked_pair.front();
      }
    }

    // Sort linked cells
    // As radix sort is broken for negative numbers, we need to filter them out                                          
    std::vector<std::int32_t> positive_cells;
    std::vector<std::int32_t> positive_to_old;
    positive_cells.reserve(linked_cells.size());
    positive_to_old.reserve(linked_cells.size());
    for (std::size_t j = 0; j< linked_cells.size(); ++j)
    {
      if (linked_cells[j] >= 0)
      {
        positive_cells.push_back(linked_cells[j]);
        positive_to_old.push_back(j);
      }
    } 
    auto [unique_cells, offsets]
        = sort_cells(positive_cells,
                     std::span(perm.data(), positive_cells.size()));

    // Loop over sorted array of unique cells
    std::int32_t link = 0;
    for (std::size_t j = 0; j < unique_cells.size(); ++j)
    {
      if (unique_cells[j] < 0)
        continue;

      std::int32_t linked_cell = parent_cells[unique_cells[j]];

      // Extract indices of all occurances of cell in the unsorted cell
      // array
      auto indices
          = std::span(perm.data() + offsets[j], offsets[j + 1] - offsets[j]);

      // Extract local dofs
      assert(std::size_t(linked_cell) < mesh->geometry().dofmap().extent(0));
      std::vector<double> x_c(indices.size() * tdim);
      for (std::size_t l = 0; l < indices.size(); l++)
      {
        std::int32_t ind = positive_to_old[indices[l]];
        std::copy_n(std::next(reference_x.begin(),
                              num_q_points * tdim * i + ind * tdim),
                    tdim, std::next(x_c.begin(), l * tdim));
      }

      // Compute values of basis functions for all y = Pi(x) in qp
      std::array<std::size_t, 4> b_shape
          = evaluate_basis_shape(V, indices.size(), 1);
      if (b_shape[3] != 1)
      {
        throw std::invalid_argument(
            "pack_grad_test_functions assumes values size 1");
      }

      std::vector<double> basis_valuesb(
          std::reduce(b_shape.cbegin(), b_shape.cend(), 1, std::multiplies{}));
      cells.resize(indices.size());
      std::fill(cells.begin(), cells.end(), linked_cell);
      evaluate_basis_functions(V, x_c, cells, basis_valuesb, 1);
      mdspan_t<const double, 4> basis_values(basis_valuesb.data(), b_shape);

      // Insert basis function values into c
      for (std::size_t k = 0; k < ndofs; k++)
      {
        for (std::size_t q = 0; q < indices.size(); ++q)
        {
          for (std::size_t l = 0; l < gdim; l++)
          {
            c[i * cstride + link * ndofs * gdim * num_q_points
              + k * gdim * num_q_points + indices[q] * gdim + l]
                = basis_values(l + 1, q, k, 0);
          }
        }
      }
      link += 1;
    }
  }

  return {std::move(c), cstride};
}
//-----------------------------------------------------------------------------------------------
std::pair<std::vector<PetscScalar>, int>
Contact::pack_grad_u_contact(int pair,
                             const dolfinx::fem::Function<PetscScalar>& u) const
{
  auto [quadrature_mt, candidate_mt] = _contact_pairs[pair];

  // Mesh inf
  std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V
      = u.function_space();
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh = V->mesh();
  std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
  std::span<const std::int32_t> parent_cells = _submesh.parent_cells();
  std::size_t bs_element = V->element()->block_size();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  assert(dofmap);
  int bs_dof = dofmap->bs();

  // Select which side of the contact interface to loop from and get the
  // correct map
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> map
      = _facet_maps[pair];
  std::size_t num_facets = _local_facets[quadrature_mt];
  std::size_t num_q_points
      = _quadrature_rule.offset()[1] - _quadrature_rule.offset()[0];

  // Output vector
  int cstride = num_q_points * bs_element * gdim;
  std::vector<PetscScalar> c(num_facets * cstride, 0.0);

  // return if no facets on process
  if (num_facets == 0)
    return {std::move(c), cstride};

  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
      = _submesh.facet_map();
  assert(facet_map);

  // NOTE: Assuming same number of quadrature points on each cell
  error::check_cell_type(mesh->topology()->cell_type());
  std::vector<double> points(num_facets * num_q_points * gdim);
  mdspan_t<double, 3> pts(points.data(), num_facets, num_q_points, gdim);
  std::vector<std::int32_t> cells(num_facets * num_q_points, -1);
  for (std::size_t i = 0; i < num_facets; ++i)
  {
    auto links = map->links(i);
    assert(links.size() == num_q_points);
    for (std::size_t q = 0; q < num_q_points; ++q)
    {
      if (links[q] < 0)
        continue;

      std::size_t row = i * num_q_points;
      auto linked_pair = facet_map->links(links[q]);
      cells[row + q] = parent_cells[linked_pair.front()];
    }
  }

  const std::vector<double>& reference_x = _reference_contact_points[pair];

  std::array<std::size_t, 4> b_shape
      = evaluate_basis_shape(*V, num_facets * num_q_points, 1);
  std::vector<double> basis_values(
      std::reduce(b_shape.begin(), b_shape.end(), 1, std::multiplies{}));
  std::fill(basis_values.begin(), basis_values.end(), 0);
  evaluate_basis_functions(*V, reference_x, cells, basis_values, 1);

  std::span<const PetscScalar> u_coeffs = u.x()->array();

  // Create work vector for expansion coefficients

  std::size_t num_basis_functions = b_shape[2];
  std::size_t value_size = b_shape[3];
  mdspan_t<double, 4> bvals(basis_values.data(), b_shape[0], b_shape[1],
                            b_shape[2], b_shape[3]);
  std::vector<PetscScalar> coefficients(num_basis_functions * bs_element);
  for (std::size_t i = 0; i < num_facets; ++i)
  {
    auto links = map->links(i);
    for (std::size_t q = 0; q < num_q_points; ++q)
    {
      if (links[q] < 0)
        continue;

      // Get degrees of freedom for current cell
      auto dofs = dofmap->cell_dofs(cells[i * num_q_points + q]);
      for (std::size_t j = 0; j < dofs.size(); ++j)
        for (int k = 0; k < bs_dof; ++k)
          coefficients[bs_dof * j + k] = u_coeffs[bs_dof * dofs[j] + k];

      // Compute expansion
      for (std::size_t k = 0; k < bs_element; ++k)
      {
        for (std::size_t j = 0; j < gdim; ++j)
        {
          for (std::size_t l = 0; l < num_basis_functions; ++l)
          {
            for (std::size_t m = 0; m < value_size; ++m)
            {
              c[cstride * i + q * bs_element * gdim + k * gdim + j]
                  += coefficients[bs_element * l + k]
                     * bvals(j + 1, num_q_points * i + q, l, m);
            }
          }
        }
      }
    }
  }

  return {std::move(c), cstride};
}
//-----------------------------------------------------------------------------------------------
void Contact::update_submesh_geometry(
    const dolfinx::fem::Function<PetscScalar>& u)
{
  _submesh.update_geometry(u);
}
//-----------------------------------------------------------------------------------------------
std::size_t Contact::num_q_points() const
{
  return _quadrature_rule.offset()[1] - _quadrature_rule.offset()[0];
}
//-----------------------------------------------------------------------------------------------
