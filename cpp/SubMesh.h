// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#pragma once
#include <dolfinx.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>

namespace dolfinx_contact
{
class SubMesh
{
public:
  // empty constructor
  SubMesh() {}

  // submesh constructor
  // constructs a submesh consisting of the cells adjacent to a given set of
  // exterior facets
  /// @param[in] mesh - the parent mesh
  /// @param[in] facets - vector of pairs (cell, facet) of exterior facets,
  /// where cell is the index of the cell local to the process and facet is the
  /// facet index within the cell
  SubMesh(std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
          std::vector<std::pair<std::int32_t, int>>& facets);

  // Return mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh() const { return _mesh; }

  // Return adjacency list mapping from parent mesh cell to submesh cell
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
  cell_map() const
  {
    return _mesh_to_submesh_cell_map;
  }

  // Return adjacency list mapping from submesh facet corresponding to facets
  // from the original input list to pair (cell, facet)
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
  facet_map() const
  {
    return _facets_to_cells;
  }

  // Return parent cells: parent_cells()[i] is the cell in the parent mesh for
  // ith cell in submesh
  std::vector<std::int32_t> parent_cells() const { return _parent_cells; }

  // Create FunctionSpace on submesh that is identical with a given
  // FunctionSpace on the parent mesh but restricted to submesh
  /// @param[in] V_parent - the function space on the the parent mesh
  /// @return the function space on the submesh
  dolfinx::fem::FunctionSpace
  create_functionspace(std::shared_ptr<dolfinx::fem::FunctionSpace> V_parent);

  // Copy of a function on the parent mesh/ in the parent function space
  // to submesh/ function space on submesh
  ///@param[in] u_parent - function to be copied
  ///@param[in, out] u_sub - function into which the function values are to be copied
  void copy_function(dolfinx::fem::Function<PetscScalar>& u_parent,
                     dolfinx::fem::Function<PetscScalar>& u_sub);

private:
  // the submesh mesh
  std::shared_ptr<dolfinx::mesh::Mesh> _mesh;

  // submesh to mesh vertex map returned by dolfinx::mesh::create_submesh
  std::vector<std::int32_t> _submesh_to_mesh_vertex_map;

  // submesh to mesh x_dof map returned by dolfinx::mesh::create_submesh
  std::vector<std::int32_t> _submesh_to_mesh_x_dof_map;

  // adjacency list mapping from submesh facet corresponding to facets
  // from the original input list to pair (cell, facet)
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
      _mesh_to_submesh_cell_map;

  // parent_cells()[i] is the cell in the parent mesh for
  // ith cell in submesh
  std::vector<std::int32_t> _parent_cells;

  // adjacency list mapping from submesh facet corresponding to facets
  // from the original input list to pair (cell, facet)
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _facets_to_cells;
};
} // namespace dolfinx_contact