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
}
