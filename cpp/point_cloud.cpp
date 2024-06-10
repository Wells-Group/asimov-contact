// Copyright (C) 2023 Chris N. Richardson
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "point_cloud.h"
#include <algorithm>
#include <array>
#include <dolfinx/graph/AdjacencyList.h>
#include <vector>

/// Find all neighbors of each point which are within a radius r.
dolfinx::graph::AdjacencyList<std::int32_t>
dolfinx_contact::point_cloud_pairs(std::span<const double> x, double r)
{

  assert(x.size() % 3 == 0);
  const int npoints = x.size() / 3;

  // Get argsort of x[:, 0]
  std::vector<int> x_fwd(npoints), x_rev(npoints);
  std::iota(x_fwd.begin(), x_fwd.end(), 0);
  std::sort(x_fwd.begin(), x_fwd.end(),
            [x](int a, int b) { return x[a * 3] < x[b * 3]; });
  for (int i = 0; i < npoints; ++i)
    x_rev[x_fwd[i]] = i;

  std::vector<int> x_near;
  std::vector<int> offsets = {0};

  for (int i = 0; i < npoints; ++i)
  {
    int idx = x_rev[i] + 1;
    while (idx < npoints)
    {
      const double* xj = x.data() + x_fwd[idx] * 3;
      const double* xi = x.data() + i * 3;
      double dx = xj[0] - xi[0];
      if (dx > r)
        break;
      const double dy = (xj[1] - xi[1]);
      const double dz = (xj[2] - xi[2]);
      const double dr = dx * dx + dy * dy + dz * dz;
      if (dr < r * r)
        x_near.push_back(x_fwd[idx]);
      ++idx;
    }
    idx = x_rev[i] - 1;
    while (idx > 0)
    {
      const double* xj = x.data() + x_fwd[idx] * 3;
      const double* xi = x.data() + i * 3;
      double dx = xi[0] - xj[0];
      if (dx > r)
        break;
      const double dy = (xj[1] - xi[1]);
      const double dz = (xj[2] - xi[2]);
      const double dr = dx * dx + dy * dy + dz * dz;
      if (dr < r * r)
        x_near.push_back(x_fwd[idx]);
      --idx;
    }
    offsets.push_back(x_near.size());
  }

  return dolfinx::graph::AdjacencyList<std::int32_t>(x_near, offsets);
}
