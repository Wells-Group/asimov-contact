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
    const std::span<const std::int32_t>& submesh_facets,
    const std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>&
        sub_to_parent,
    const std::span<const std::int32_t>& parent_cells)
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
                    auto facet_pair = sub_to_parent->links(facet);
                    assert(facet_pair.size() == 2);
                    linked_cells.push_back(parent_cells[facet_pair[0]]);
                  }
                });

  // Remove duplicates
  dolfinx::radix_sort(std::span<std::int32_t>(linked_cells));
  linked_cells.erase(std::unique(linked_cells.begin(), linked_cells.end()),
                     linked_cells.end());
}

} // namespace

dolfinx_contact::Contact::Contact(
    const std::vector<std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>>&
        markers,
    std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>> surfaces,
    const std::vector<std::array<int, 2>>& contact_pairs,
    std::shared_ptr<dolfinx::fem::FunctionSpace> V, const int q_deg,
    ContactMode mode)
    : _surfaces(surfaces->array()), _contact_pairs(contact_pairs), _V(V),
      _mode(mode)
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
    std::span<const int> links = surfaces->links(int(s));
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
std::pair<std::vector<double>, std::array<std::size_t, 3>>
dolfinx_contact::Contact::qp_phys(int surface)
{
  const std::size_t num_facets = _cell_facet_pairs[surface].size() / 2;
  const std::size_t num_q_points
      = _quadrature_rule->offset()[1] - _quadrature_rule->offset()[0];
  const std::size_t gdim = _V->mesh()->geometry().dim();
  std::array<std::size_t, 3> shape = {num_facets, num_q_points, gdim};
  return {_qp_phys[surface], shape};
}
//------------------------------------------------------------------------------------------------
std::size_t dolfinx_contact::Contact::coefficients_size(bool meshtie)
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

  if (meshtie)
  {

    // Coefficient offsets
    // Expecting coefficients in following order:
    // mu, lmbda, h,test_fn, grad(test_fn), u, u_opposite,
    // grad(u_opposite)
    std::array<std::size_t, 8> cstrides
        = {1,
           1,
           1,
           num_q_points * ndofs_cell * bs * max_links,
           num_q_points * ndofs_cell * bs * max_links,
           ndofs_cell * bs,
           num_q_points * bs,
           num_q_points * gdim * bs};
    return std::accumulate(cstrides.cbegin(), cstrides.cend(), 0);
  }
  else
  {
    // Coefficient offsets
    // Expecting coefficients in the following order
    // mu, lmbda, h, gap, normals, test_fns, u, u_opposite,
    std::array<std::size_t, 8> cstrides
        = {1,
           1,
           1,
           num_q_points * bs,
           num_q_points * bs,
           num_q_points * ndofs_cell * bs * max_links,
           ndofs_cell * bs,
           num_q_points * bs};
    return std::accumulate(cstrides.cbegin(), cstrides.cend(), 0);
  };
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
        = _submeshes[contact_pair.back()].facet_map();
    assert(facet_map);
    std::span<const std::int32_t> parent_cells
        = _submeshes[contact_pair.back()].parent_cells();
    for (int i = 0; i < (int)_cell_facet_pairs[contact_pair.front()].size();
         i += 2)
    {
      std::int32_t cell = _cell_facet_pairs[contact_pair.front()][i];
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
  pattern.assemble();

  return dolfinx::la::petsc::create_matrix(a.mesh()->comm(), pattern, type);
}
//------------------------------------------------------------------------------------------------
void dolfinx_contact::Contact::create_distance_map(int pair)
{
  // Get quadrature mesh info
  auto [puppet_mt, candidate_mt] = _contact_pairs[pair];
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
  [[maybe_unused]] auto [adj, reference_x, shape]
      = dolfinx_contact::compute_distance_map(*puppet_mesh, quadrature_facets,
                                              *candidate_mesh, submesh_facets,
                                              *_quadrature_rule, _mode);

  _facet_maps[pair]
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(adj);

  // NOTE: More data that should be updated inside this code
  const dolfinx::fem::CoordinateElement& cmap
      = candidate_mesh->geometry().cmap();
  std::tie(_reference_basis, _reference_shape)
      = impl::tabulate(cmap, _quadrature_rule);

  // NOTE: This function should be moved somwhere else, or return the actual
  // points such that we compuld send them in to compute_distance_map.
  // Compute quadrature points on physical facet _qp_phys_"origin_meshtag"
  create_q_phys(puppet_mt);

  // Update maximum number of connected cells
  max_links(pair);
}
std::pair<std::vector<PetscScalar>, int>
dolfinx_contact::Contact::pack_nx(int pair)
{
  auto [quadrature_mt, candidate_mt] = _contact_pairs[pair];
  const std::shared_ptr<const dolfinx::mesh::Mesh>& quadrature_mesh
      = _submeshes[quadrature_mt].mesh();
  assert(quadrature_mesh);

  // Get (cell, local_facet_index) tuples on quadrature submesh
  const std::vector<std::int32_t> quadrature_facets
      = _submeshes[quadrature_mt].get_submesh_tuples(
          _cell_facet_pairs[quadrature_mt]);

  // Get information about submesh geometry and topology
  const dolfinx::mesh::Geometry& geometry = quadrature_mesh->geometry();
  const int gdim = geometry.dim();
  std::span<const double> x_g = geometry.x();
  auto x_dofmap = geometry.dofmap();
  const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
  const std::size_t num_dofs_g = cmap.dim();
  const dolfinx::mesh::Topology& topology = quadrature_mesh->topology();
  const int tdim = topology.dim();

  // Get all quadrature points
  const std::vector<double>& q_points = _quadrature_rule->points();
  assert(_quadrature_rule->tdim() == (std::size_t)tdim);
  const std::array<std::size_t, 2> shape
      = {q_points.size() / tdim, (std::size_t)tdim};

  // Tabulate first derivatives basis functions at all reference points
  const std::array<std::size_t, 4> basis_shape
      = cmap.tabulate_shape(1, shape[0]);
  assert(basis_shape.back() == 1);
  std::vector<double> cmap_basisb(std::reduce(
      basis_shape.cbegin(), basis_shape.cend(), 1, std::multiplies{}));
  cmap.tabulate(1, q_points, shape, cmap_basisb);

  // Loop over quadrature points
  error::check_cell_type(quadrature_mesh->topology().cell_type());
  const std::size_t num_facets = quadrature_facets.size() / 2;
  const std::size_t num_q_points = _quadrature_rule->num_points(0);

  // Get facet normals on reference cell
  basix::cell::type cell_type
      = dolfinx::mesh::cell_type_to_basix_type(topology.cell_type());
  auto [facet_normalsb, n_shape]
      = basix::cell::facet_outward_normals(cell_type);
  cmdspan2_t facet_normals(facet_normalsb.data(), n_shape);

  // Working memory for loop
  std::vector<double> coordinate_dofsb(num_dofs_g * gdim);
  cmdspan2_t coordinate_dofs(coordinate_dofsb.data(), num_dofs_g, gdim);
  std::array<double, 9> Jb;
  std::array<double, 9> Kb;
  mdspan2_t J(Jb.data(), gdim, tdim);
  mdspan2_t K(Kb.data(), tdim, gdim);
  mdspan4_t full_basis(cmap_basisb.data(), basis_shape);

  std::vector<PetscScalar> normals(num_facets * num_q_points * gdim, 0.0);
  const int cstride = (int)num_q_points * gdim;
  for (std::size_t i = 0; i < quadrature_facets.size(); i += 2)
  {

    // Copy coordinate dofs of candidate cell
    // Get cell geometry (coordinate dofs)
    auto x_dofs = x_dofmap.links(quadrature_facets[i]);
    assert(x_dofs.size() == num_dofs_g);
    for (std::size_t j = 0; j < num_dofs_g; ++j)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[j]), gdim,
                  std::next(coordinate_dofsb.begin(), j * gdim));
    }
    for (std::size_t q = 0; q < num_q_points; ++q)
    {
      auto dphi = stdex::submdspan(full_basis, std::pair{1, tdim + 1},
                                   quadrature_facets[i + 1] * num_q_points + q,
                                   stdex::full_extent, 0);

      // Compute Jacobian and Jacobian inverse for Piola mapping of normal
      std::fill(Jb.begin(), Jb.end(), 0);
      dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs,
                                                        J);
      std::fill(Kb.begin(), Kb.end(), 0);
      dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);

      // Push forward normal using covariant Piola
      physical_facet_normal(
          std::span(std::next(normals.begin(), i / 2 * cstride + q * gdim),
                    gdim),
          K,
          stdex::submdspan(facet_normals, quadrature_facets[i + 1],
                           stdex::full_extent));
    }
  }
  return {std::move(normals), cstride};
}
//------------------------------------------------------------------------------------------------
dolfinx_contact::kernel_fn<PetscScalar>
dolfinx_contact::Contact::generate_kernel(Kernel type)
{

  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = _V->mesh();
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
  const std::size_t bs = _V->dofmap()->bs();
  // FIXME: This will not work for prism meshes
  const std::vector<std::size_t>& qp_offsets = _quadrature_rule->offset();
  const std::size_t num_q_points = qp_offsets[1] - qp_offsets[0];
  const std::size_t max_links
      = *std::max_element(_max_links.begin(), _max_links.end());
  const std::size_t ndofs_cell = _V->dofmap()->element_dof_layout().num_dofs();

  // Coefficient offsets
  // Expecting coefficients in following order:
  // mu, lmbda, h, gap, normals, test_fn, u, u_opposite
  std::vector<std::size_t> cstrides
      = {1,
         1,
         1,
         num_q_points * gdim,
         num_q_points * gdim,
         num_q_points * ndofs_cell * bs * max_links,
         ndofs_cell * bs,
         num_q_points * bs};

  auto kd = dolfinx_contact::KernelData(_V, _quadrature_rule, cstrides);

  /// @brief Assemble kernel for RHS of unbiased contact problem
  ///
  /// Assemble of the residual of the unbiased contact problem into vector
  /// `b`.
  /// @param[in,out] b The vector to assemble the residual into
  /// @param[in] c The coefficients used in kernel. Assumed to be
  /// ordered as mu, lmbda, h, gap, normals, test_fn, u, u_opposite.
  /// @param[in] w The constants used in kernel. Assumed to be ordered as
  /// `gamma`, `theta`.
  /// @param[in] coordinate_dofs The physical coordinates of cell. Assumed to
  /// be padded to 3D, (shape (num_nodes, 3)).
  /// @param[in] facet_index Local facet index (relative to cell)
  /// @param[in] num_links How many cells from opposite surface are connected
  /// with the cell.
  /// @param[in] q_indices The quadrature points to loop over
  kernel_fn<PetscScalar> unbiased_rhs =
      [kd, gdim, ndofs_cell,
       bs](std::vector<std::vector<PetscScalar>>& b,
           std::span<const PetscScalar> c, const PetscScalar* w,
           const double* coordinate_dofs, const int facet_index,
           const std::size_t num_links, std::span<const std::int32_t> q_indices)

  {
    // Retrieve some data from kd
    const std::uint32_t tdim = kd.tdim();

    // NOTE: DOLFINx has 3D input coordinate dofs
    cmdspan2_t coord(coordinate_dofs, kd.num_coordinate_dofs(), 3);

    // Create data structures for jacobians
    // We allocate more memory than required, but its better for the compiler
    std::array<double, 9> Jb;
    mdspan2_t J(Jb.data(), gdim, tdim);
    std::array<double, 9> Kb;
    mdspan2_t K(Kb.data(), tdim, gdim);
    std::array<double, 6> J_totb;
    mdspan2_t J_tot(J_totb.data(), gdim, tdim - 1);
    double detJ = 0;
    std::array<double, 18> detJ_scratch;

    // Normal vector on physical facet at a single quadrature point
    std::array<double, 3> n_phys;

    // Pre-compute jacobians and normals for affine meshes
    if (kd.affine())
    {
      detJ = kd.compute_first_facet_jacobian(facet_index, J, K, J_tot,
                                             detJ_scratch, coord);
      physical_facet_normal(std::span(n_phys.data(), gdim), K,
                            stdex::submdspan(kd.facet_normals(), facet_index,
                                             stdex::full_extent));
    }

    // Extract constants used inside quadrature loop
    double gamma = c[2] / w[0];     // h/gamma
    double gamma_inv = w[0] / c[2]; // gamma/h
    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];
    // Extract reference to the tabulated basis function
    s_cmdspan2_t phi = kd.phi();
    s_cmdspan3_t dphi = kd.dphi();

    // Extract reference to quadrature weights for the local facet

    auto weights = kd.weights(facet_index);

    // Temporary data structures used inside quadrature loop
    std::array<double, 3> n_surf = {0, 0, 0};
    std::vector<double> epsnb(ndofs_cell * gdim, 0);
    mdspan2_t epsn(epsnb.data(), ndofs_cell, gdim);
    std::vector<double> trb(ndofs_cell * gdim, 0);
    mdspan2_t tr(trb.data(), ndofs_cell, gdim);

    // Loop over quadrature points
    const std::size_t q_start = kd.qp_offsets(facet_index);
    const std::size_t q_end = kd.qp_offsets(facet_index + 1);
    const std::size_t num_points = q_end - q_start;
    for (auto q : q_indices)
    {
      const std::size_t q_pos = q_start + q;

      // Update Jacobian and physical normal
      detJ = kd.update_jacobian(q, facet_index, detJ, J, K, J_tot, detJ_scratch,
                                coord);
      kd.update_normal(std::span(n_phys.data(), gdim), K, facet_index);
      double n_dot = 0;
      double gap = 0;
      // For ray tracing the gap is given by n * (Pi(x) -x)
      // where n = n_x
      // For closest point n = -n_y
      for (std::size_t i = 0; i < gdim; i++)
      {
        n_surf[i] = -c[kd.offsets(4) + q * gdim + i];
        n_dot += n_phys[i] * n_surf[i];
        gap += c[kd.offsets(3) + q * gdim + i] * n_surf[i];
      }

      compute_normal_strain_basis(epsn, tr, K, dphi, n_surf,
                                  std::span(n_phys.data(), gdim), q_pos);
      // compute tr(eps(u)), epsn at q
      double tr_u = 0;
      double epsn_u = 0;
      double jump_un = 0;
      for (std::size_t i = 0; i < ndofs_cell; i++)
      {
        std::size_t block_index = kd.offsets(6) + i * bs;
        for (std::size_t j = 0; j < bs; j++)
        {
          PetscScalar coeff = c[block_index + j];
          tr_u += coeff * tr(i, j);
          epsn_u += coeff * epsn(i, j);
          jump_un += coeff * phi(q_pos, i) * n_surf[j];
        }
      }
      std::size_t offset_u_opp = kd.offsets(7) + q * bs;
      for (std::size_t j = 0; j < bs; ++j)
        jump_un += -c[offset_u_opp + j] * n_surf[j];
      double sign_u = lmbda * tr_u * n_dot + mu * epsn_u;
      const double w0 = weights[q] * detJ;

      double Pn_u = R_plus((jump_un - gap) - gamma * sign_u) * w0;
      // Fill contributions of facet with itself
      for (std::size_t i = 0; i < ndofs_cell; i++)
      {
        for (std::size_t n = 0; n < bs; n++)
        {
          double v_dot_nsurf = n_surf[n] * phi(q_pos, i);
          double sign_v = (lmbda * tr(i, n) * n_dot + mu * epsn(i, n));
          // This is (1./gamma)*Pn_v to avoid the product gamma*(1./gamma)
          double Pn_v = gamma_inv * v_dot_nsurf - theta * sign_v;
          b[0][n + i * bs] += 0.5 * Pn_u * Pn_v;

          // entries corresponding to v on the other surface
          for (std::size_t k = 0; k < num_links; k++)
          {
            std::size_t index = kd.offsets(5) + k * num_points * ndofs_cell * bs
                                + i * num_points * bs + q * bs + n;
            double v_n_opp = c[index] * n_surf[n];

            b[k + 1][n + i * bs] -= 0.5 * gamma_inv * v_n_opp * Pn_u;
          }
        }
      }
    }
  };

  /// @brief Assemble kernel for Jacobian (LHS) of unbiased contact
  /// problem
  ///
  /// Assemble of the residual of the unbiased contact problem into matrix
  /// `A`.
  /// @param[in,out] A The matrix to assemble the Jacobian into
  /// @param[in] c The coefficients used in kernel. Assumed to be
  /// ordered as mu, lmbda, h, gap, normals, test_fn, u, u_opposite.
  /// @param[in] w The constants used in kernel. Assumed to be ordered as
  /// `gamma`, `theta`.
  /// @param[in] coordinate_dofs The physical coordinates of cell. Assumed
  /// to be padded to 3D, (shape (num_nodes, 3)).
  /// @param[in] facet_index Local facet index (relative to cell)
  /// @param[in] num_links How many cells from opposite surface are connected
  /// with the cell.
  /// @param[in] q_indices The quadrature points to loop over
  kernel_fn<PetscScalar> unbiased_jac =
      [kd, gdim, ndofs_cell, bs](
          std::vector<std::vector<PetscScalar>>& A, std::span<const double> c,
          const double* w, const double* coordinate_dofs, const int facet_index,
          const std::size_t num_links, std::span<const std::int32_t> q_indices)
  {
    // Retrieve some data from kd
    const std::uint32_t tdim = kd.tdim();

    // NOTE: DOLFINx has 3D input coordinate dofs
    cmdspan2_t coord(coordinate_dofs, kd.num_coordinate_dofs(), 3);

    // Create data structures for jacobians
    // We allocate more memory than required, but its better for the compiler
    std::array<double, 9> Jb;
    mdspan2_t J(Jb.data(), gdim, tdim);
    std::array<double, 9> Kb;
    mdspan2_t K(Kb.data(), tdim, gdim);
    std::array<double, 6> J_totb;
    mdspan2_t J_tot(J_totb.data(), gdim, tdim - 1);
    double detJ;
    std::array<double, 18> detJ_scratch;

    // Normal vector on physical facet at a single quadrature point
    std::array<double, 3> n_phys;

    // Pre-compute jacobians and normals for affine meshes
    if (kd.affine())
    {
      detJ = kd.compute_first_facet_jacobian(facet_index, J, K, J_tot,
                                             detJ_scratch, coord);
      physical_facet_normal(std::span(n_phys.data(), gdim), K,
                            stdex::submdspan(kd.facet_normals(), facet_index,
                                             stdex::full_extent));
    }

    // Extract scaled gamma (h/gamma) and its inverse
    double gamma = c[2] / w[0];
    double gamma_inv = w[0] / c[2];

    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];

    cmdspan3_t dphi = kd.dphi();
    cmdspan2_t phi = kd.phi();
    std::array<std::size_t, 2> q_offset
        = {kd.qp_offsets(facet_index), kd.qp_offsets(facet_index + 1)};
    const std::size_t num_points = q_offset.back() - q_offset.front();
    std::span<const double> weights = kd.weights(facet_index);
    std::array<double, 3> n_surf = {0, 0, 0};
    std::vector<double> epsnb(ndofs_cell * gdim);
    mdspan2_t epsn(epsnb.data(), ndofs_cell, gdim);
    std::vector<double> trb(ndofs_cell * gdim);
    mdspan2_t tr(trb.data(), ndofs_cell, gdim);

    // Loop over quadrature points
    for (auto q : q_indices)
    {
      const std::size_t q_pos = q_offset.front() + q;
      // Update Jacobian and physical normal
      detJ = kd.update_jacobian(q, facet_index, detJ, J, K, J_tot, detJ_scratch,
                                coord);
      kd.update_normal(std::span(n_phys.data(), gdim), K, facet_index);

      double n_dot = 0;
      double gap = 0;
      // The gap is given by n * (Pi(x) -x)
      // For raytracing n = n_x
      // For closest point n = -n_y
      for (std::size_t i = 0; i < gdim; i++)
      {
        n_surf[i] = -c[kd.offsets(4) + q * gdim + i];
        n_dot += n_phys[i] * n_surf[i];
        gap += c[kd.offsets(3) + q * gdim + i] * n_surf[i];
      }

      compute_normal_strain_basis(epsn, tr, K, dphi, n_surf,
                                  std::span(n_phys.data(), gdim), q_pos);

      // compute tr(eps(u)), epsn at q
      double tr_u = 0;
      double epsn_u = 0;
      double jump_un = 0;

      for (std::size_t i = 0; i < ndofs_cell; i++)
      {
        std::size_t block_index = kd.offsets(6) + i * bs;
        for (std::size_t j = 0; j < bs; j++)
        {
          tr_u += c[block_index + j] * tr(i, j);
          epsn_u += c[block_index + j] * epsn(i, j);
          jump_un += c[block_index + j] * phi(q_pos, i) * n_surf[j];
        }
      }
      std::size_t offset_u_opp = kd.offsets(7) + q * bs;
      for (std::size_t j = 0; j < bs; ++j)
        jump_un += -c[offset_u_opp + j] * n_surf[j];
      double sign_u = lmbda * tr_u * n_dot + mu * epsn_u;
      double Pn_u = dR_plus((jump_un - gap) - gamma * sign_u);

      // Fill contributions of facet with itself
      const double w0 = weights[q] * detJ;
      for (std::size_t j = 0; j < ndofs_cell; j++)
      {
        for (std::size_t l = 0; l < bs; l++)
        {
          double sign_du = (lmbda * tr(j, l) * n_dot + mu * epsn(j, l));
          double Pn_du
              = (phi(q_pos, j) * n_surf[l] - gamma * sign_du) * Pn_u * w0;

          sign_du *= w0;
          for (std::size_t i = 0; i < ndofs_cell; i++)
          {
            for (std::size_t b = 0; b < bs; b++)
            {
              double v_dot_nsurf = n_surf[b] * phi(q_pos, i);
              double sign_v = (lmbda * tr(i, b) * n_dot + mu * epsn(i, b));
              double Pn_v = gamma_inv * v_dot_nsurf - theta * sign_v;
              A[0][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                  += 0.5 * Pn_du * Pn_v;

              // entries corresponding to u and v on the other surface
              for (std::size_t k = 0; k < num_links; k++)
              {
                std::size_t index = kd.offsets(5)
                                    + k * num_points * ndofs_cell * bs
                                    + j * num_points * bs + q * bs + l;
                double du_n_opp = c[index] * n_surf[l];

                du_n_opp *= w0 * Pn_u;
                index = kd.offsets(5) + k * num_points * ndofs_cell * bs
                        + i * num_points * bs + q * bs + b;
                double v_n_opp = c[index] * n_surf[b];
                A[3 * k + 1][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    -= 0.5 * du_n_opp * Pn_v;
                A[3 * k + 2][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    -= 0.5 * gamma_inv * Pn_du * v_n_opp;
                A[3 * k + 3][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    += 0.5 * gamma_inv * du_n_opp * v_n_opp;
              }
            }
          }
        }
      }
    }
  };
  switch (type)
  {
  case Kernel::Rhs:
    return unbiased_rhs;
  case Kernel::Jac:
    return unbiased_jac;
  case Kernel::MeshTieRhs:
  {

    return generate_meshtie_kernel(type, _V, _quadrature_rule, max_links);
  }
  case Kernel::MeshTieJac:
  {
    return generate_meshtie_kernel(type, _V, _quadrature_rule, max_links);
  }
  default:
    throw std::invalid_argument("Unrecognized kernel");
  }
}
//------------------------------------------------------------------------------------------------
std::pair<std::vector<PetscScalar>, int>
dolfinx_contact::Contact::pack_ny(int pair)
{
  // FIXME: This function should take in the quadrature points
  // (push_forward_quadrature) of the relevant facet, and the reference points
  // on the other surface (output of distance map)
  auto [quadrature_mt, candidate_mt] = _contact_pairs[pair];

  // Get mesh info for candidate side
  const std::shared_ptr<const dolfinx::mesh::Mesh>& candidate_mesh
      = _submeshes[candidate_mt].mesh();
  assert(candidate_mesh);
  const std::shared_ptr<const dolfinx::mesh::Mesh>& quadrature_mesh
      = _submeshes[quadrature_mt].mesh();
  assert(quadrature_mesh);

  // Get (cell, local_facet_index) tuples on quadrature submesh
  const std::vector<std::int32_t> quadrature_facets
      = _submeshes[quadrature_mt].get_submesh_tuples(
          _cell_facet_pairs[quadrature_mt]);

  // Get (cell, local_facet_index) tuples on candidate submesh
  const std::vector<std::int32_t> candidate_facets
      = _submeshes[candidate_mt].get_submesh_tuples(
          _cell_facet_pairs[candidate_mt]);

  auto [candidate_map, reference_x, shape]
      = dolfinx_contact::compute_distance_map(
          *quadrature_mesh, quadrature_facets, *candidate_mesh,
          candidate_facets, *_quadrature_rule, _mode);

  // Get information about submesh geometry and topology
  const dolfinx::mesh::Geometry& geometry = candidate_mesh->geometry();
  const int gdim = geometry.dim();
  std::span<const double> x_g = geometry.x();
  auto x_dofmap = geometry.dofmap();
  const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
  const std::size_t num_dofs_g = cmap.dim();
  const dolfinx::mesh::Topology& topology = candidate_mesh->topology();
  const int tdim = topology.dim();

  // Tabulate first derivatives basis functions at all reference points
  const std::array<std::size_t, 4> basis_shape
      = cmap.tabulate_shape(1, shape[0]);
  assert(basis_shape.back() == 1);
  std::vector<double> cmap_basisb(std::reduce(
      basis_shape.cbegin(), basis_shape.cend(), 1, std::multiplies{}));
  cmap.tabulate(1, reference_x, shape, cmap_basisb);

  // Loop over quadrature points
  error::check_cell_type(candidate_mesh->topology().cell_type());
  const int num_facets = candidate_map.num_nodes();
  const std::size_t num_q_points
      = num_facets == 0 ? 0 : candidate_map.num_links(0);

  std::vector<PetscScalar> normals(num_facets * num_q_points * gdim, 0.0);
  const int cstride = (int)num_q_points * gdim;

  auto f_to_c = candidate_mesh->topology().connectivity(tdim - 1, tdim);
  if (!f_to_c)
  {
    throw std::runtime_error("Missing facet to cell connectivity on "
                             "candidate submesh");
  }
  auto c_to_f = candidate_mesh->topology().connectivity(tdim, tdim - 1);
  if (!c_to_f)
  {
    throw std::runtime_error("Missing cell to facet connectivity on "
                             "candidate submesh");
  }

  // Get facet normals on reference cell
  basix::cell::type cell_type = dolfinx::mesh::cell_type_to_basix_type(
      candidate_mesh->topology().cell_type());
  auto [facet_normalsb, n_shape]
      = basix::cell::facet_outward_normals(cell_type);
  cmdspan2_t facet_normals(facet_normalsb.data(), n_shape);

  // Working memory for loop
  std::vector<double> coordinate_dofsb(num_dofs_g * gdim);
  cmdspan2_t coordinate_dofs(coordinate_dofsb.data(), num_dofs_g, gdim);
  std::array<double, 9> Jb;
  std::array<double, 9> Kb;
  mdspan2_t J(Jb.data(), gdim, tdim);
  mdspan2_t K(Kb.data(), tdim, gdim);
  mdspan4_t full_basis(cmap_basisb.data(), basis_shape);
  for (int i = 0; i < num_facets; ++i)
  {
    auto facets = candidate_map.links(i);
    assert(facets.size() == num_q_points);
    for (std::size_t q = 0; q < num_q_points; ++q)
    {
      // Skip computation if quadrature point does not have a matching facet on
      // the other side
      if (facets[q] < 0)
        continue;

      auto candidate_cells = f_to_c->links(facets[q]);
      assert(candidate_cells.size() == 1);
      assert(candidate_cells.front() >= 0);

      // Get local facet index of candidate facet
      auto local_facets = c_to_f->links(candidate_cells.front());
      auto it = std::find(local_facets.begin(), local_facets.end(), facets[q]);
      const int local_idx = std::distance(local_facets.begin(), it);

      // Copy coordinate dofs of candidate cell
      // Get cell geometry (coordinate dofs)
      auto x_dofs = x_dofmap.links(candidate_cells.front());
      assert(x_dofs.size() == num_dofs_g);
      for (std::size_t j = 0; j < num_dofs_g; ++j)
      {
        std::copy_n(std::next(x_g.begin(), 3 * x_dofs[j]), gdim,
                    std::next(coordinate_dofsb.begin(), j * gdim));
      }
      auto dphi = stdex::submdspan(full_basis, std::pair{1, tdim + 1},
                                   i * num_q_points + q, stdex::full_extent, 0);
      // Compute Jacobian and Jacobian inverse for Piola mapping of normal
      std::fill(Jb.begin(), Jb.end(), 0);
      dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs,
                                                        J);
      std::fill(Kb.begin(), Kb.end(), 0);
      dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);

      // Push forward normal using covariant Piola
      physical_facet_normal(
          std::span(std::next(normals.begin(), i * cstride + q * gdim), gdim),
          K, stdex::submdspan(facet_normals, local_idx, stdex::full_extent));
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
    const std::span<const PetscScalar> coeffs, int cstride,
    const std::span<const PetscScalar>& constants)
{
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = _V->mesh();
  assert(mesh);

  // Extract geometry data
  const dolfinx::mesh::Geometry& geometry = mesh->geometry();
  const int gdim = geometry.dim();
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = geometry.dofmap();
  std::span<const double> x_g = geometry.x();
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
      = _cell_facet_pairs[contact_pair.front()];
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> map
      = _facet_maps[pair];
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
      = _submeshes[contact_pair.back()].facet_map();
  assert(facet_map);

  std::span<const std::int32_t> parent_cells
      = _submeshes[contact_pair.back()].parent_cells();
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
    const std::span<const int> x_dofs = x_dofmap.links(active_facets[i]);
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
      auto connected_facets = map->links((int)i / 2);
      q_indices.reserve(connected_facets.size());
      // NOTE: Should probably be pre-computed
      for (std::size_t j = 0; j < connected_facets.size(); ++j)
        if (connected_facets[j] >= 0)
          q_indices.push_back(j);

      // Compute the unique set of cells linked to the current facet
      compute_linked_cells(linked_cells, connected_facets, facet_map,
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

    // FIXME: We would have to handle possible Dirichlet conditions here, if
    // we think that we can have a case with contact and Dirichlet
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
//------------------------------------------------------------------------------------------------

void dolfinx_contact::Contact::assemble_vector(
    std::span<PetscScalar> b, int pair,
    const dolfinx_contact::kernel_fn<PetscScalar>& kernel,
    const std::span<const PetscScalar>& coeffs, int cstride,
    const std::span<const PetscScalar>& constants)
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
  std::span<const double> x_g = geometry.x();

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
      = _cell_facet_pairs[contact_pair.front()];
  const dolfinx_contact::SubMesh& submesh = _submeshes[contact_pair.back()];
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> map
      = _facet_maps[pair];
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
      = submesh.facet_map();
  assert(facet_map);
  std::span<const std::int32_t> parent_cells = submesh.parent_cells();
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
    const std::span<const int> x_dofs = x_dofmap.links(active_facets[i]);
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
      auto connected_facets = map->links((int)i / 2);
      q_indices.reserve(connected_facets.size());

      // NOTE: Should probably be pre-computed
      for (std::size_t j = 0; j < connected_facets.size(); ++j)
        if (connected_facets[j] >= 0)
          q_indices.push_back(j);

      compute_linked_cells(linked_cells, connected_facets, facet_map,
                           parent_cells);
    }

    // Using integer loop here to reduce number of zeroed vectors
    const std::size_t num_linked_cells = linked_cells.size();
    std::fill(bes[0].begin(), bes[0].end(), 0);
    for (std::size_t j = 0; j < num_linked_cells; j++)
      std::fill(bes[j + 1].begin(), bes[j + 1].end(), 0);

    kernel(bes, std::span(coeffs.data() + i / 2 * cstride, cstride),
           constants.data(), coordinate_dofs.data(), active_facets[i + 1],
           num_linked_cells, q_indices);

    // Add element vector to global vector
    const std::span<const int> dofs_cell = dofmap->cell_dofs(active_facets[i]);
    for (std::size_t j = 0; j < ndofs_cell; ++j)
      for (int k = 0; k < bs; ++k)
        b[bs * dofs_cell[j] + k] += bes[0][bs * j + k];
    for (std::size_t l = 0; l < num_linked_cells; ++l)
    {
      const std::span<const int> dofs_linked
          = dofmap->cell_dofs(linked_cells[l]);
      for (std::size_t j = 0; j < ndofs_cell; ++j)
        for (int k = 0; k < bs; ++k)
          b[bs * dofs_linked[j] + k] += bes[l + 1][bs * j + k];
    }
  }
}
//-----------------------------------------------------------------------------------------------
std::pair<std::vector<PetscScalar>, int>
dolfinx_contact::Contact::pack_grad_test_functions(
    int pair, const std::span<const PetscScalar>& gap,
    const std::span<const PetscScalar>& u_packed)
{
  auto [puppet_mt, candidate_mt] = _contact_pairs[pair];
  // Mesh info
  const dolfinx_contact::SubMesh& submesh = _submeshes[candidate_mt];
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = _V->mesh(); // mesh
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim();
  std::span<const std::int32_t> parent_cells = submesh.parent_cells();
  std::shared_ptr<const fem::FiniteElement> element = _V->element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t ndofs
      = (std::size_t)element->space_dimension() / bs_element;

  // Select which side of the contact interface to loop from and get the
  // correct map
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> map
      = _facet_maps[pair];
  const std::vector<std::int32_t>& puppet_facets = _cell_facet_pairs[puppet_mt];
  const std::size_t num_facets = puppet_facets.size() / 2;
  const std::size_t num_q_points
      = _quadrature_rule->offset()[1] - _quadrature_rule->offset()[0];
  std::vector<double> q_points(std::size_t(num_q_points) * std::size_t(gdim));
  mdspan3_t qp_span(_qp_phys[puppet_mt].data(), num_facets, num_q_points, gdim);

  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
      = _submeshes[candidate_mt].facet_map();
  const std::size_t max_links
      = *std::max_element(_max_links.begin(), _max_links.end());

  std::vector<std::int32_t> perm(num_q_points);
  std::vector<std::int32_t> linked_cells(num_q_points);

  // Create output vector
  std::vector<PetscScalar> c(
      num_facets * num_q_points * max_links * ndofs * gdim, 0.0);
  const auto cstride = int(num_q_points * max_links * ndofs * gdim);

  // temporary data structure used inside loop
  std::vector<std::int32_t> cells(max_links, -1);
  // Loop over all facets
  for (std::size_t i = 0; i < num_facets; i++)
  {
    const std::span<const int> links = map->links((int)i);
    assert(links.size() == num_q_points);
    for (std::size_t j = 0; j < num_q_points; j++)
    {
      const std::span<const int> linked_pair = facet_map->links(links[j]);
      assert(!linked_pair.empty());
      linked_cells[j] = linked_pair.front();
    }
    // Sort linked cells
    const auto [unique_cells, offsets] = dolfinx_contact::sort_cells(
        std::span(linked_cells.data(), linked_cells.size()),
        std::span(perm.data(), perm.size()));

    // Loop over sorted array of unique cells
    for (std::size_t j = 0; j < unique_cells.size(); ++j)
    {

      std::int32_t linked_cell = parent_cells[unique_cells[j]];
      // Extract indices of all occurances of cell in the unsorted cell
      // array
      auto indices
          = std::span(perm.data() + offsets[j], offsets[j + 1] - offsets[j]);
      // Extract local dofs
      assert(linked_cell < mesh->geometry().dofmap().num_nodes());
      auto qp = std::span(q_points.data(), indices.size() * gdim);
      mdspan2_t qp_j(qp.data(), indices.size(), gdim);
      // Compute Pi(x) form points x and gap funtion Pi(x) - x
      for (std::size_t l = 0; l < indices.size(); l++)
      {
        std::int32_t ind = indices[l];
        const std::size_t row = i * num_q_points;
        for (std::size_t k = 0; k < gdim; k++)
          qp_j(l, k) = qp_span(i, ind, k) + gap[row * gdim + ind * gdim + k]
                       - u_packed[row * gdim + ind * gdim + k];
      }

      // Compute values of basis functions for all y = Pi(x) in qp
      std::array<std::size_t, 4> b_shape
          = evaluate_basis_shape(*_V, indices.size(), 1);
      if (b_shape[3] != 1)
        throw std::invalid_argument(
            "pack_grad_test_functions assumes values size 1");
      std::vector<double> basis_valuesb(
          std::reduce(b_shape.cbegin(), b_shape.cend(), 1, std::multiplies{}));
      cells.resize(indices.size());
      std::fill(cells.begin(), cells.end(), linked_cell);
      evaluate_basis_functions(*_V, qp, cells, basis_valuesb, 1);
      cmdspan4_t basis_values(basis_valuesb.data(), b_shape);
      // Insert basis function values into c
      for (std::size_t k = 0; k < ndofs; k++)
        for (std::size_t q = 0; q < indices.size(); ++q)
          for (std::size_t l = 0; l < gdim; l++)
          {
            c[i * cstride + j * ndofs * gdim * num_q_points
              + k * gdim * num_q_points + indices[q] * gdim + l]
                = basis_values(l + 1, q, k, 0);
          }
    }
  }

  return {std::move(c), cstride};
}
//-----------------------------------------------------------------------------------------------
std::pair<std::vector<PetscScalar>, int>
dolfinx_contact::Contact::pack_grad_u_contact(
    int pair, std::shared_ptr<dolfinx::fem::Function<PetscScalar>> u,
    const std::span<const PetscScalar> gap,
    const std::span<const PetscScalar> u_packed)
{
  auto [puppet_mt, candidate_mt] = _contact_pairs[pair];

  // Mesh info
  const dolfinx_contact::SubMesh& submesh = _submeshes[candidate_mt];
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = _V->mesh();
  const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
  std::span<const std::int32_t> parent_cells = submesh.parent_cells();
  const std::size_t bs_element = _V->element()->block_size();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = _V->dofmap();
  assert(dofmap);
  const int bs_dof = dofmap->bs();
  // Select which side of the contact interface to loop from and get the
  // correct map
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> map
      = _facet_maps[pair];
  const std::size_t num_facets = _cell_facet_pairs[puppet_mt].size() / 2;
  const std::size_t num_q_points
      = _quadrature_rule->offset()[1] - _quadrature_rule->offset()[0];
  mdspan3_t qp_span(_qp_phys[puppet_mt].data(), num_facets, num_q_points, gdim);
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
      = submesh.facet_map();
  assert(facet_map);

  // NOTE: Assuming same number of quadrature points on each cell
  dolfinx_contact::error::check_cell_type(mesh->topology().cell_type());
  std::vector<double> points(num_facets * num_q_points * gdim);
  mdspan3_t pts(points.data(), num_facets, num_q_points, gdim);
  std::vector<std::int32_t> cells(num_facets * num_q_points, -1);
  for (std::size_t i = 0; i < num_facets; ++i)
  {
    auto links = map->links((int)i);
    assert(links.size() == num_q_points);
    for (std::size_t q = 0; q < num_q_points; ++q)
    {
      if (links[q] < 0)
        continue;
      const std::size_t row = i * num_q_points;
      auto linked_pair = facet_map->links(links[q]);
      cells[row + q] = parent_cells[linked_pair.front()];
      for (std::size_t j = 0; j < gdim; ++j)
      {
        pts(i, q, j) = qp_span(i, q, j) + gap[row * gdim + q * gdim + j]
                       - u_packed[row * gdim + q * gdim + j];
      }
    }
  }
  std::array<std::size_t, 4> b_shape
      = evaluate_basis_shape(*_V, num_facets * num_q_points, 1);
  std::vector<double> basis_values(
      std::reduce(b_shape.begin(), b_shape.end(), 1, std::multiplies{}));
  std::fill(basis_values.begin(), basis_values.end(), 0);
  evaluate_basis_functions(*u->function_space(), points, cells, basis_values,
                           1);

  const std::span<const PetscScalar>& u_coeffs = u->x()->array();
  // Output vector
  const auto cstride = int(num_q_points * bs_element * gdim);
  std::vector<PetscScalar> c(num_facets * cstride, 0.0);

  // Create work vector for expansion coefficients

  const std::size_t num_basis_functions = b_shape[2];
  const std::size_t value_size = b_shape[3];
  mdspan4_t bvals(basis_values.data(), b_shape[0], b_shape[1], b_shape[2],
                  b_shape[3]);
  std::vector<PetscScalar> coefficients(num_basis_functions * bs_element);
  for (std::size_t i = 0; i < num_facets; ++i)
  {
    for (std::size_t q = 0; q < num_q_points; ++q)
    {
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
void dolfinx_contact::Contact::update_submesh_geometry(
    dolfinx::fem::Function<PetscScalar>& u) const
{

  for (auto submesh : _submeshes)
    submesh.update_geometry(u);
}
