// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#pragma once
#include <dolfinx.h>
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
  SubMesh() {}
  SubMesh(std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
          std::vector<std::pair<std::int32_t, int>>& facets);

  /// Return mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh() const { return _mesh; }

  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
  cell_map() const
  {
    return _mesh_to_submesh_cell_map;
  }
  std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
  facet_map() const
  {
    return _facets_to_cells;
  }
  std::vector<std::int32_t> parent_cells() const { return _parent_cells; }

  dolfinx::fem::FunctionSpace
  create_functionspace(std::shared_ptr<dolfinx::fem::FunctionSpace> V_parent);

  dolfinx::fem::Function<PetscScalar>
  copy_function(dolfinx::fem::Function<PetscScalar>& u,
                std::shared_ptr<dolfinx::fem::FunctionSpace> V_sub);

private:
  std::shared_ptr<dolfinx::mesh::Mesh> _mesh;
  std::vector<std::int32_t> _submesh_to_mesh_vertex_map;
  std::vector<std::int32_t> _submesh_to_mesh_x_dof_map;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>>
      _mesh_to_submesh_cell_map;
  std::vector<std::int32_t> _parent_cells;
  std::shared_ptr<dolfinx::graph::AdjacencyList<std::int32_t>> _facets_to_cells;
};
} // namespace dolfinx_contact