// Copyright (C) 2023 Chris N. Richardson
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <dolfinx/graph/AdjacencyList.h>
#include <vector>

namespace dolfinx_contact
{
/// @brief Compute near neighbours in list of points
/// @param x List of points in 3D flattened row major
/// @param r Search distance
/// @return For each point, the list of other points within radius r.
dolfinx::graph::AdjacencyList<std::int32_t>
point_cloud_pairs(std::span<const double> x, double r);
} // namespace dolfinx_contact
