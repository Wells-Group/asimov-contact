
// Copyright (C) 2022 Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "RayTracing.h"

//------------------------------------------------------------------------------------------------
std::tuple<int, std::int32_t, std::vector<double>, std::vector<double>>
dolfinx_contact::raytracing(const dolfinx::mesh::Mesh<double>& mesh,
                            std::span<const double> point,
                            std::span<const double> normal,
                            std::span<const std::int32_t> cells, int max_iter,
                            double tol)
{
  const int tdim = mesh.topology()->dim();
  const int gdim = mesh.geometry().dim();

  assert((std::size_t)gdim == point.size());
  assert((std::size_t)gdim == normal.size());
  std::tuple<int, std::int32_t, std::vector<double>, std::vector<double>>
      output;
  if (tdim == 2)
  {
    if (gdim == 2)
    {
      auto [status, cell_idx, x, X] = compute_ray<2, 2>(
          mesh, std::span<const double, 2>(point.data(), 2),
          std::span<const double, 2>(normal.data(), 2), cells, max_iter, tol);
      std::vector<double> x_out(x.begin(), x.end());
      std::vector<double> X_out(X.begin(), X.end());
      output = std::make_tuple(status, cell_idx, x_out, X_out);
    }
    else if (gdim == 3)
    {
      auto [status, cell_idx, x, X] = compute_ray<2, 3>(
          mesh, std::span<const double, 3>(point.data(), 3),
          std::span<const double, 3>(normal.data(), 3), cells, max_iter, tol);
      std::vector<double> x_out(x.begin(), x.end());
      std::vector<double> X_out(X.begin(), X.end());
      output = std::make_tuple(status, cell_idx, x_out, X_out);
    }
  }
  else if (tdim == 3)
  {
    auto [status, cell_idx, x, X] = compute_ray<3, 3>(
        mesh, std::span<const double, 3>(point.data(), 3),
        std::span<const double, 3>(normal.data(), 3), cells, max_iter, tol);
    std::vector<double> x_out(x.begin(), x.end());
    std::vector<double> X_out(X.begin(), X.end());
    output = std::make_tuple(status, cell_idx, x_out, X_out);
  }
  return output;
}

//------------------------------------------------------------------------------------------------
