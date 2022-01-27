// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#include "SubMesh.h"
using namespace dolfinx_contact;

dolfinx_contact::SubMesh::SubMesh(
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
    std::vector<std::pair<std::int32_t, int>>& cell_facet_pairs)
{
  const int tdim = mesh->topology().dim(); // topological dimension

  std::vector<std::int32_t> cells(cell_facet_pairs.size());
  for (std::size_t f = 0; f < cell_facet_pairs.size(); ++f)
    cells[f] = cell_facet_pairs[f].first;
  std::sort(cells.begin(), cells.end());
  cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
  _parent_cells = cells;
  auto submesh_data = dolfinx::mesh::create_submesh(
      *mesh, tdim, xtl::span(cells.data(), cells.size()));
  _mesh = std::make_shared<dolfinx::mesh::Mesh>(std::get<0>(submesh_data));
  _submesh_to_mesh_vertex_map = std::get<1>(submesh_data);
  _submesh_to_mesh_x_dof_map = std::get<2>(submesh_data);
  _mesh->topology().create_connectivity(tdim - 1, tdim);
  auto f_to_c = _mesh->topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  _mesh->topology().create_connectivity(tdim, tdim - 1);
  auto c_to_f = _mesh->topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);
  auto map_c = mesh->topology().index_map(tdim);
  const int num_cells = map_c->size_local() + map_c->num_ghosts();
  std::vector<std::int32_t> marked_cells(num_cells, 0);
  // mark which cells are in cells
  for (auto cell : cells)
    marked_cells[cell] = 1;
  // Create offsets
  std::vector<int32_t> offsets(num_cells + 1, 0);
  std::partial_sum(marked_cells.begin(), marked_cells.end(),
                   offsets.begin() + 1);
  std::vector<std::int32_t> data(offsets[num_cells]);
  for (std::int32_t c = 0; c < cells.size(); ++c)
  {
    data[offsets[cells[c]]] = c;
  }

  _mesh_to_submesh_cell_map
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          std::move(data), std::move(offsets));

  // Create facet to (cell, local_facet) map
  // Retrieve number of facets on process
  auto map_f = _mesh->topology().index_map(tdim - 1);
  const int num_facets = map_f->size_local() + map_f->num_ghosts();

  std::vector<std::int32_t> marked_facets(num_facets, 0);
  std::vector<std::int32_t> facets(cell_facet_pairs.size(), 0);

  // mark which facets are in any of the facet lists
  for (size_t i = 0; i < cell_facet_pairs.size(); ++i)
  {
    auto facet_pair = cell_facet_pairs[i];
    auto cell = _mesh_to_submesh_cell_map->links(facet_pair.first)[0];
    std::int32_t facet = c_to_f->links(cell)[facet_pair.second];
    marked_facets[facet] += 2;
    facets[i] = facet;
  }
  // Create offsets
  std::vector<int32_t> offsets2(num_facets + 1, 0);
  std::partial_sum(marked_facets.begin(), marked_facets.end(),
                   offsets2.begin() + 1);

  std::vector<std::int32_t> data2(offsets2[num_facets]);
  for (std::size_t i = 0; i < cell_facet_pairs.size(); ++i)
  {
    auto facet_pair = cell_facet_pairs[i];
    auto cell = _mesh_to_submesh_cell_map->links(facet_pair.first)[0];
    std::int32_t facet = facets[i];
    data2[offsets2[facet]] = cell;
    data2[offsets2[facet] + 1] = facet_pair.second;
  }
  _facets_to_cells
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          std::move(data2), std::move(offsets2));
}

//------------------------------------------------------------------------------------------------
dolfinx::fem::FunctionSpace dolfinx_contact::SubMesh::create_functionspace(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V_parent)
{
  auto element = V_parent->element();
  auto element_dof_layout = V_parent->dofmap()->element_dof_layout();
  auto dofmap = std::make_shared<dolfinx::fem::DofMap>(
      dolfinx::fem::create_dofmap(_mesh->comm(), element_dof_layout,
                                  _mesh->topology(), nullptr, *element));
  return dolfinx::fem::FunctionSpace(_mesh, element, dofmap);
}

//-----------------------------------------------------------------------------------------------
dolfinx::fem::Function<PetscScalar> dolfinx_contact::SubMesh::copy_function(
    dolfinx::fem::Function<PetscScalar>& u,
    std::shared_ptr<dolfinx::fem::FunctionSpace> V_sub)
{
  auto u_sub = dolfinx::fem::Function<PetscScalar>(V_sub);
  auto dofmap_sub = V_sub->dofmap();
  auto dofmap_parent = u.function_space()->dofmap();
  const int tdim = _mesh->topology().dim();
  auto cell_map = _mesh->topology().index_map(tdim);
  assert(cell_map);
  const std::int32_t num_cells
      = cell_map->size_local() + cell_map->num_ghosts();
  const int bs = dofmap_sub->bs();
  assert(bs == dofmap_parent->bs());
  auto u_sub_data = u_sub.x()->mutable_array();
  const auto& u_data = u.x()->array();

  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    auto dofs_sub = dofmap_sub->cell_dofs(c);
    auto dofs_parent = dofmap_parent->cell_dofs(_parent_cells[c]);
    assert(dofs_sub.size() == dofs_parent.size());
    for (std::size_t i = 0; i < dofs_sub.size(); ++i)
      for (int j = 0; j < bs; ++j)
      {
        u_sub_data[bs * dofs_sub[i] + j] = u_data[bs * dofs_parent[i] + j];
      }
  }

  return u_sub;
}