// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#include "SubMesh.h"
#include "utils.h"

using namespace dolfinx_contact;

//----------------------------------------------------------------------------
SubMesh::SubMesh(const dolfinx::mesh::Mesh<double>& mesh,
                 std::span<const std::int32_t> cell_facet_pairs)
{
  const int tdim = mesh.topology()->dim();

  // Copy cell indices
  std::vector<std::int32_t> cells(cell_facet_pairs.size() / 2);
  for (std::size_t f = 0; f < cell_facet_pairs.size() / 2; ++f)
    cells[f] = cell_facet_pairs[2 * f];

  // sort cells and remove duplicates
  dolfinx::radix_sort<std::int32_t>(std::span(cells.data(), cells.size()));
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());

  // save sorted cell vector as _parent_cells

  // call dolfinx::mesh::create_submesh and save ouput to member
  // variables
  auto [submesh, cell_map, vertex_map, x_dof_map]
      = dolfinx::mesh::create_submesh(mesh, tdim, cells);
  _parent_cells = cell_map;

  _mesh = std::make_shared<dolfinx::mesh::Mesh<double>>(submesh);
  _submesh_to_mesh_vertex_map = vertex_map;
  _submesh_to_mesh_x_dof_map = x_dof_map;

  // create/retrieve connectivities on submesh
  _mesh->topology_mutable()->create_connectivity(tdim - 1, tdim);
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> f_to_c
      = _mesh->topology()->connectivity(tdim - 1, tdim);
  assert(f_to_c);
  _mesh->topology_mutable()->create_connectivity(tdim, tdim - 1);
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> c_to_f
      = _mesh->topology()->connectivity(tdim, tdim - 1);
  assert(c_to_f);

  // create adjacency list mapping cells on parent mesh to cells on
  // submesh if cell is not contained in submesh it will have no links
  // in adjacency list if it is contained, it has exactly one link, the
  // submesh cell index

  // get number of cells on process
  std::shared_ptr<const dolfinx::common::IndexMap> map_c
      = mesh.topology()->index_map(tdim);
  const std::int32_t num_cells = map_c->size_local() + map_c->num_ghosts();

  // mark which cells are in cells, i.e. which cells are in the submesh
  std::vector<std::int32_t> marked_cells(num_cells, 0);
  for (auto cell : cells)
    marked_cells[cell] = 1;

  {
    // Create offsets
    std::vector<int32_t> offsets(num_cells + 1, 0);
    std::partial_sum(marked_cells.begin(), marked_cells.end(),
                     offsets.begin() + 1);
    // fill data array
    std::vector<std::int32_t> data(offsets.back());
    for (std::size_t c = 0; c < cells.size(); ++c)
      data[offsets[cells[c]]] = (std::int32_t)c;

    // create adjacency list
    _mesh_to_submesh_cell_map
        = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
            std::move(data), std::move(offsets));
  }

  // Create facet to (cell, local_facet) map for exterior facet from the
  // original input
  {
    // Retrieve number of facets on process
    std::shared_ptr<const dolfinx::common::IndexMap> map_f
        = _mesh->topology()->index_map(tdim - 1);
    int num_facets = map_f->size_local() + map_f->num_ghosts();

    // mark which facets are in any of the facet lists

    std::vector<std::int32_t> marked_facets(num_facets, 0);
    for (std::size_t i = 0; i < cell_facet_pairs.size(); i += 2)
    {
      // get submesh cell from parent cell
      auto sub_cells = _mesh_to_submesh_cell_map->links(cell_facet_pairs[i]);
      assert(!sub_cells.empty());

      // cell facet index the same for both meshes: use c_to_f to get
      // submesh facet index
      auto facets = c_to_f->links(sub_cells.front());
      assert((std::size_t)cell_facet_pairs[i + 1] < facets.size());
      std::int32_t submesh_facet = facets[cell_facet_pairs[i + 1]];
      marked_facets[submesh_facet] = 2;
    }

    // Create offsets
    std::vector<int32_t> offsets(num_facets + 1, 0);
    std::partial_sum(marked_facets.begin(), marked_facets.end(),
                     offsets.begin() + 1);

    std::vector<std::int32_t> data(offsets.back());
    for (std::size_t i = 0; i < cell_facet_pairs.size(); i += 2)
    {
      // get submesh cell from parent cell
      auto sub_cells = _mesh_to_submesh_cell_map->links(cell_facet_pairs[i]);
      assert(!sub_cells.empty());

      // cell facet index the same for both meshes: use c_to_f to get
      // submesh facet index
      auto facets = c_to_f->links(sub_cells.front());
      assert((std::size_t)cell_facet_pairs[i + 1] < facets.size());
      std::int32_t submesh_facet = facets[cell_facet_pairs[i + 1]];
      data[offsets[submesh_facet]] = sub_cells.front();
      data[offsets[submesh_facet] + 1] = cell_facet_pairs[i + 1];
    }

    // create adjacency list
    _facets_to_cells
        = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
            std::move(data), std::move(offsets));
  }
}
//----------------------------------------------------------------------------
dolfinx::fem::FunctionSpace<double> SubMesh::create_functionspace(
    const dolfinx::fem::FunctionSpace<double>& V_parent) const
{
  // get element and element_dof_layout from parent mesh
  std::shared_ptr<const dolfinx::fem::FiniteElement<double>> element
      = V_parent.element();
  const dolfinx::fem::ElementDofLayout& element_dof_layout
      = V_parent.dofmap()->element_dof_layout();

  // use parent mesh data and submesh comm/topology to create new dofmap
  std::function<void(std::span<std::int32_t>, std::uint32_t)> unpermute_dofs;
  if (element->needs_dof_permutations())
    unpermute_dofs = element->dof_permutation_fn(true, true);

  auto dofmap = std::make_shared<dolfinx::fem::DofMap>(
      dolfinx::fem::create_dofmap(_mesh->comm(), element_dof_layout,
                                  *_mesh->topology(), unpermute_dofs, nullptr));

  // create and return function space
  std::span<const std::size_t> vs = V_parent.value_shape();
  std::vector _value_shape(vs.data(), vs.data() + vs.size());

  return dolfinx::fem::FunctionSpace(_mesh, element, dofmap, _value_shape);
}
//----------------------------------------------------------------------------
void SubMesh::copy_function(const dolfinx::fem::Function<PetscScalar>& u_parent,
                            dolfinx::fem::Function<PetscScalar>& u_sub) const
{
  // retrieve function space on submesh
  std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V_sub
      = u_sub.function_space();

  // get dofmaps for both function spaces
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap_sub = V_sub->dofmap();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap_parent
      = u_parent.function_space()->dofmap();

  // Assume tdim is the same for both
  const int tdim = _mesh->topology()->dim();

  // get number of submesh cells on proces
  std::shared_ptr<const dolfinx::common::IndexMap> cell_map
      = _mesh->topology()->index_map(tdim);
  assert(cell_map);
  std::int32_t num_cells = cell_map->size_local() + cell_map->num_ghosts();

  // get block size, assume they are the same for both function spaces
  const int bs = dofmap_sub->bs();
  assert(bs == dofmap_parent->bs());

  // retrieve value array
  std::span<PetscScalar> u_sub_data = u_sub.x()->mutable_array();
  std::span<const PetscScalar> u_data = u_parent.x()->array();

  // copy data from u into u_sub
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    std::span<const int> dofs_sub = dofmap_sub->cell_dofs(c);
    std::span<const int> dofs_parent
        = dofmap_parent->cell_dofs(_parent_cells[c]);
    assert(dofs_sub.size() == dofs_parent.size());
    for (std::size_t i = 0; i < dofs_sub.size(); ++i)
      for (int j = 0; j < bs; ++j)
        u_sub_data[bs * dofs_sub[i] + j] = u_data[bs * dofs_parent[i] + j];
  }
}
//----------------------------------------------------------------------------
void SubMesh::update_geometry(const dolfinx::fem::Function<PetscScalar>& u)
{
  // Recover original geometry from parent mesh
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> parent_mesh
      = u.function_space()->mesh();
  std::span<double> sub_geometry = _mesh->geometry().x();
  std::span<const double> parent_geometry = parent_mesh->geometry().x();
  std::size_t num_x_dofs = sub_geometry.size() / 3;
  for (std::size_t i = 0; i < num_x_dofs; ++i)
  {
    std::copy_n(
        std::next(parent_geometry.begin(), 3 * _submesh_to_mesh_x_dof_map[i]),
        3, std::next(sub_geometry.begin(), 3 * i));
  }

  // use u to update geometry
  std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V_parent
      = u.function_space();
  auto V_sub = std::make_shared<dolfinx::fem::FunctionSpace<double>>(
      create_functionspace(*V_parent));
  auto u_sub = dolfinx::fem::Function<PetscScalar>(V_sub);
  copy_function(u, u_sub);
  dolfinx_contact::update_geometry(u_sub, *_mesh);
}
//----------------------------------------------------------------------------
std::vector<std::int32_t>
SubMesh::get_submesh_tuples(std::span<const std::int32_t> facets) const
{
  assert(_mesh);

  // Map (cell, facet) tuples from parent to sub mesh
  std::vector<std::int32_t> submesh_facets(facets.size());

  int tdim = _mesh->topology()->dim();
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> c_to_f
      = _mesh->topology()->connectivity(tdim, tdim - 1);
  assert(c_to_f);

  for (std::size_t i = 0; i < facets.size(); i += 2)
  {
    auto submesh_cells = _mesh_to_submesh_cell_map->links(facets[i]);
    assert(!submesh_cells.empty());
    assert(submesh_cells.size() == 1);
    submesh_facets[i] = submesh_cells.front();
    submesh_facets[i + 1] = facets[i + 1];
  }

  return submesh_facets;
}
//----------------------------------------------------------------------------
