// Copyright (C) 2023 Chris N. Richardson
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <span>

namespace dolfinx_contact
{
  /// Compute destinations
  dolfinx::graph::AdjacencyList<std::int32_t>
    compute_ghost_cell_destinations(const dolfinx::mesh::Mesh<double>& mesh,
                                    std::span<const std::int32_t> marker_subset, double R);

  /// @brief Creates a new mesh with additional ghost cells
  /// @param mesh input mesh
  /// @param fmarker facet markers
  /// @param cmarker cell/domain markers
  /// @param tags
  /// @param R search radius
  /// @return new mesh and markers
  std::tuple<dolfinx::mesh::Mesh<double>, dolfinx::mesh::MeshTags<std::int32_t>>
  create_contact_mesh(dolfinx::mesh::Mesh<double>& mesh,
                      const dolfinx::mesh::MeshTags<std::int32_t>& fmarker,
                      const dolfinx::mesh::MeshTags<std::int32_t>& cmarker,
                      const std::vector<std::int32_t>& tags, double R = 0.2);

  /// @brief Lexical matching of input markers with local entities.
  ///
  /// From the input markers in (in_indices, in_values) find any matching
  /// entities in local_indices, and copy the values across. Entities are
  /// represented by their vertex indices (dim vertices per entity).
  /// @param dim Number of vertices per entity
  /// @param local_indices Local entites as vertex indices, flattened
  /// @param in_indices Input entities as vertex indices, flattened
  /// @param in_values Values at input entities
  /// @return Values at local entities
  std::vector<std::pair<int, int>>
  lex_match(int dim, const std::vector<std::int64_t>& local_indices,
            const std::vector<std::int64_t>& in_indices,
            const std::vector<std::int32_t>& in_values);
}
