// Copyright (C) 2023 Chris N. Richardson
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <span>

namespace dolfinx_contact
{
  /// Compute destinations
  dolfinx::graph::AdjacencyList<std::int32_t>
    compute_ghost_cell_destinations(const dolfinx::mesh::Mesh<double>& mesh,
                                    std::span<const std::int32_t> marker_subset, double R);

  /// @brief Lexical matching of input markers with local entities
  /// From the input markers in (in_indices, in_values) find any matching
  /// entities in local_indices, and copy the values across. Entities are
  /// represented by their vertex indices (dim vertices per entity).
  /// @param dim Number of vertices per entity
  /// @param local_indices Local entites as vertex indices, flattened
  /// @param in_indices Input entities as vertex indices, flattened
  /// @param in_values Values at input entities
  /// @return Values at local entities
  std::vector<std::pair<int, int>>
  lex_match(int dim, const std::vector<std::int32_t>& local_indices,
            const std::vector<std::int32_t>& in_indices,
            const std::vector<std::int32_t>& in_values);
}
