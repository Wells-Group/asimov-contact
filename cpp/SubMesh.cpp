// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#include "SubMesh.h"
#include "utils.h"

using namespace dolfinx_contact;

dolfinx_contact::SubMesh::SubMesh(
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
    xtl::span<const std::int32_t> cell_facet_pairs)
{
  const int tdim = mesh->topology().dim(); // topological dimension

  // create sorted vector of unique cells adjacent to the input facets
  std::vector<std::int32_t> cells(cell_facet_pairs.size() / 2);
  for (std::size_t f = 0; f < cell_facet_pairs.size(); f += 2)
    cells[f / 2] = cell_facet_pairs[f]; // retrieve cells
  dolfinx::radix_sort<std::int32_t>(
      std::span(cells.data(), cells.size())); // sort cells
  cells.erase(std::unique(cells.begin(), cells.end()),
              cells.end()); // remove duplicates

  // save sorted cell vector as _parent_cells

  // call dolfinx::mesh::create_submesh and save ouput to member variables
  auto [submesh, cell_map, vertex_map, x_dof_map]
      = dolfinx::mesh::create_submesh(*mesh, tdim,
                                      std::span(cells.data(), cells.size()));
  _parent_cells = cell_map;

  _mesh = std::make_shared<dolfinx::mesh::Mesh>(submesh);
  _submesh_to_mesh_vertex_map = vertex_map;
  _submesh_to_mesh_x_dof_map = x_dof_map;

  // create/retrieve connectivities on submesh
  _mesh->topology_mutable().create_connectivity(tdim - 1, tdim);
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> f_to_c
      = _mesh->topology().connectivity(tdim - 1, tdim);
  assert(f_to_c);
  _mesh->topology_mutable().create_connectivity(tdim, tdim - 1);
  std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> c_to_f
      = _mesh->topology().connectivity(tdim, tdim - 1);
  assert(c_to_f);

  // create adjacency list mapping cells on parent mesh to cells on submesh
  // if cell is not contained in submesh it will have no links in adjacency list
  // if it is contained, it has exactly one link, the submesh cell index

  // get number of cells on process
  std::shared_ptr<const dolfinx::common::IndexMap> map_c
      = mesh->topology().index_map(tdim);
  const int num_cells = map_c->size_local() + map_c->num_ghosts();

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
        = _mesh->topology().index_map(tdim - 1);
    const int num_facets = map_f->size_local() + map_f->num_ghosts();

    // mark which facets are in any of the facet lists

    std::vector<std::int32_t> marked_facets(num_facets, 0);
    for (std::size_t i = 0; i < cell_facet_pairs.size(); i += 2)
    {
      // get submesh cell from parent cell
      auto sub_cells = _mesh_to_submesh_cell_map->links(cell_facet_pairs[i]);
      assert(!sub_cells.empty());
      // cell facet index the same for both meshes: use c_to_f to
      // get submesh facet index
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
      // cell facet index the same for both meshes: use c_to_f to
      // get submesh facet index
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

//------------------------------------------------------------------------------------------------
dolfinx::fem::FunctionSpace dolfinx_contact::SubMesh::create_functionspace(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V_parent) const
{
  // get element and element_dof_layout from parent mesh
  std::shared_ptr<const dolfinx::fem::FiniteElement> element
      = V_parent->element();
  const dolfinx::fem::ElementDofLayout& element_dof_layout
      = V_parent->dofmap()->element_dof_layout();
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
  std::shared_ptr<const dolfinx::fem::FunctionSpace> V_sub
      = u_sub.function_space();
  // get dofmaps for both function spaces
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap_sub = V_sub->dofmap();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap_parent
      = u_parent.function_space()->dofmap();
  // Assume tdim is the same for both
  const int tdim = _mesh->topology().dim();
  // get number of submesh cells on proces
  std::shared_ptr<const dolfinx::common::IndexMap> cell_map
      = _mesh->topology().index_map(tdim);
  assert(cell_map);
  const std::int32_t num_cells
      = cell_map->size_local() + cell_map->num_ghosts();
  // get block size, assume they are the same for both function spaces
  const int bs = dofmap_sub->bs();
  assert(bs == dofmap_parent->bs());

  // retrieve value array
  tcb::span<PetscScalar> u_sub_data = u_sub.x()->mutable_array();
  tcb::span<const PetscScalar> u_data = u_parent.x()->array();

  // copy data from u into u_sub
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    const tcb::span<const int> dofs_sub = dofmap_sub->cell_dofs(c);
    const tcb::span<const int> dofs_parent
        = dofmap_parent->cell_dofs(_parent_cells[c]);
    assert(dofs_sub.size() == dofs_parent.size());
    for (std::size_t i = 0; i < dofs_sub.size(); ++i)
      for (int j = 0; j < bs; ++j)
      {
        u_sub_data[bs * dofs_sub[i] + j] = u_data[bs * dofs_parent[i] + j];
      }
  }
}

//-----------------------------------------------------------------------------------------------
void dolfinx_contact::SubMesh::update_geometry(
    dolfinx::fem::Function<PetscScalar>& u)
{
  // Recover original geometry from parent mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> parent_mesh
      = u.function_space()->mesh();
  tcb::span<double> sub_geometry = _mesh->geometry().x();
  tcb::span<const double> parent_geometry = parent_mesh->geometry().x();
  std::size_t num_x_dofs = sub_geometry.size() / 3;
  for (std::size_t i = 0; i < num_x_dofs; ++i)
  {
    dolfinx::common::impl::copy_N<3>(
        std::next(parent_geometry.begin(), 3 * _submesh_to_mesh_x_dof_map[i]),
        std::next(sub_geometry.begin(), 3 * i));
  }
  // use u to update geometry
  std::shared_ptr<const dolfinx::fem::FunctionSpace> V_parent
      = u.function_space();
  auto V_sub = std::make_shared<dolfinx::fem::FunctionSpace>(
      create_functionspace(V_parent));
  auto u_sub = dolfinx::fem::Function<PetscScalar>(V_sub);
  copy_function(u, u_sub);
  dolfinx_contact::update_geometry(u_sub, _mesh);
}