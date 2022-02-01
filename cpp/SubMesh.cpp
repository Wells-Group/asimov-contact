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

  // create sorted vector of unique cells adjacent to the input facets
  std::vector<std::int32_t> cells(cell_facet_pairs.size());
  for (std::size_t f = 0; f < cell_facet_pairs.size(); ++f)
    cells[f] = cell_facet_pairs[f].first;              // retrieve cells
  dolfinx::radix_sort<std::int32_t>(xtl::span(cells)); // sort cells
  cells.erase(std::unique(cells.begin(), cells.end()),
              cells.end()); // remove duplicates

  // save sorted cell vector as _parent_cells
  _parent_cells = cells;

  // call doflinx::mesh::create_submesh and save ouput to member variables
  auto submesh_data = dolfinx::mesh::create_submesh(
      *mesh, tdim, xtl::span(cells.data(), cells.size()));
  _mesh = std::make_shared<dolfinx::mesh::Mesh>(std::get<0>(submesh_data));
  _submesh_to_mesh_vertex_map = std::get<1>(submesh_data);
  _submesh_to_mesh_x_dof_map = std::get<2>(submesh_data);

  // create/retrieve connectivities on submesh
  _mesh->topology().create_connectivity(tdim - 1, tdim);
  auto f_to_c = _mesh->topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  _mesh->topology().create_connectivity(tdim, tdim - 1);
  auto c_to_f = _mesh->topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);

  // create adjacency list mapping cells on parent mesh to cells on submesh
  // if cell is not contained in submesh it will have no links in adjacency list
  // if it is contained, it has exactly one link, the submesh cell index

  // get number of cells on process
  auto map_c = mesh->topology().index_map(tdim);
  const int num_cells = map_c->size_local() + map_c->num_ghosts();

  // mark which cells are in cells, i.e. which cells are in the submesh
  std::vector<std::int32_t> marked_cells(num_cells, 0);
  for (auto cell : cells)
    marked_cells[cell] = 1;
  // Create offsets
  std::vector<int32_t> offsets(num_cells + 1, 0);
  std::partial_sum(marked_cells.begin(), marked_cells.end(),
                   offsets.begin() + 1);
  // fill data array
  std::vector<std::int32_t> data(offsets.back());
  for (std::size_t c = 0; c < cells.size(); ++c)
  {
    data[offsets[cells[c]]] = (std::int32_t)c;
  }

  // create adjacency list
  _mesh_to_submesh_cell_map
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          std::move(data), std::move(offsets));

  // Create facet to (cell, local_facet) map for exterior facet from the
  // original input

  // Retrieve number of facets on process
  auto map_f = _mesh->topology().index_map(tdim - 1);
  const int num_facets = map_f->size_local() + map_f->num_ghosts();

  // mark which facets are in any of the facet lists
  std::vector<std::int32_t> marked_facets(num_facets, 0);
  for (std::size_t i = 0; i < cell_facet_pairs.size(); ++i)
  {
    auto facet_pair = cell_facet_pairs[i];
    // get submesh cell from parent cell
    auto cell = _mesh_to_submesh_cell_map->links(facet_pair.first)[0];
    // cell facet index the same for both meshes: use c_to_f to get submesh
    // facet index
    std::int32_t facet = c_to_f->links(cell)[facet_pair.second];
    marked_facets[facet] = 2;
  }
  // Create offsets
  std::vector<int32_t> offsets2(num_facets + 1, 0);
  std::partial_sum(marked_facets.begin(), marked_facets.end(),
                   offsets2.begin() + 1);

  // fill data
  std::vector<std::int32_t> data2(offsets2.back());
  for (std::size_t i = 0; i < cell_facet_pairs.size(); ++i)
  {
    auto facet_pair = cell_facet_pairs[i];
    auto cell = _mesh_to_submesh_cell_map->links(facet_pair.first)[0];
    std::int32_t facet = c_to_f->links(cell)[facet_pair.second];
    data2[offsets2[facet]] = cell;
    data2[offsets2[facet] + 1] = facet_pair.second;
  }

  // create adjacency list
  _facets_to_cells
      = std::make_shared<dolfinx::graph::AdjacencyList<std::int32_t>>(
          std::move(data2), std::move(offsets2));
}

//------------------------------------------------------------------------------------------------
dolfinx::fem::FunctionSpace dolfinx_contact::SubMesh::create_functionspace(
    std::shared_ptr<dolfinx::fem::FunctionSpace> V_parent)
{
  // get element and element_dof_layout from parent mesh
  auto element = V_parent->element();
  auto element_dof_layout = V_parent->dofmap()->element_dof_layout();
  // use parent mesh data and submesh comm/topology to create new dofmap
  auto dofmap = std::make_shared<dolfinx::fem::DofMap>(
      dolfinx::fem::create_dofmap(_mesh->comm(), element_dof_layout,
                                  _mesh->topology(), nullptr, *element));
  // create and return function space
  return dolfinx::fem::FunctionSpace(_mesh, element, dofmap);
}

//-----------------------------------------------------------------------------------------------
void dolfinx_contact::SubMesh::copy_function(
    dolfinx::fem::Function<PetscScalar>& u_parent,
    dolfinx::fem::Function<PetscScalar>& u_sub)
{
  // retrieve function space on submesh
  auto V_sub = u_sub.function_space();
  // get dofmaps for both function spaces
  auto dofmap_sub = V_sub->dofmap();
  auto dofmap_parent = u_parent.function_space()->dofmap();
  // Assume tdim is the same for both
  const int tdim = _mesh->topology().dim();
  // get number of submesh cells on proces
  auto cell_map = _mesh->topology().index_map(tdim);
  assert(cell_map);
  const std::int32_t num_cells
      = cell_map->size_local() + cell_map->num_ghosts();
  // get block size, assume they are the same for both function spaces
  const int bs = dofmap_sub->bs();
  assert(bs == dofmap_parent->bs());

  // retrieve value array
  auto u_sub_data = u_sub.x()->mutable_array();
  const auto& u_data = u_parent.x()->array();

  // copy data from u into u_sub
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
}