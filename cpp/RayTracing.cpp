
// Copyright (C) 2022 Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "RayTracing.h"
//------------------------------------------------------------------------------------------------
std::tuple<int, std::int32_t, xt::xtensor<double, 2>>
dolfinx_contact::raytracing(
    const dolfinx::mesh::Mesh& mesh, const xt::xtensor<double, 1>& point,
    const xt::xtensor<double, 1>& normal,
    const std::vector<std::pair<std::int32_t, int>>& cells, const int max_iter,
    const double tol)
{
  const int tdim = mesh.topology().dim();
  const int gdim = mesh.geometry().dim();

  assert((std::size_t)gdim == point.shape(0));
  assert(normal.shape(0) == (std::size_t)gdim);
  std::tuple<int, std::int32_t, xt::xtensor<double, 2>> output;
  if (tdim == 2)
  {
    if (gdim == 2)
    {
      auto [status, cell_idx, coords] = dolfinx_contact::compute_ray<2, 2>(
          mesh, point, normal, cells, max_iter, tol);
      output = std::make_tuple(status, cell_idx, coords);
    }
    else if (gdim == 3)
    {
      auto [status, cell_idx, coords] = dolfinx_contact::compute_ray<2, 3>(
          mesh, point, normal, cells, max_iter, tol);
      output = std::make_tuple(status, cell_idx, coords);
    }
  }
  else if (tdim == 3)
  {
    auto [status, cell_idx, coords] = dolfinx_contact::compute_ray<3, 3>(
        mesh, point, normal, cells, max_iter, tol);
    output = std::make_tuple(status, cell_idx, coords);
  }
  return output;
}

//------------------------------------------------------------------------------------------------
