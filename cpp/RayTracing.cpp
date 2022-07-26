
// Copyright (C) 2022 Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "RayTracing.h"
//------------------------------------------------------------------------------------------------
std::tuple<int, std::int32_t, xt::xtensor<double, 1>, xt::xtensor<double, 1>>
dolfinx_contact::raytracing(const dolfinx::mesh::Mesh& mesh,
                            const xt::xtensor<double, 1>& point,
                            const xt::xtensor<double, 1>& normal,
                            xtl::span<const std::int32_t> cells,
                            const int max_iter, const double tol)
{
  const int tdim = mesh.topology().dim();
  const int gdim = mesh.geometry().dim();

  assert((std::size_t)gdim == point.shape(0));
  assert(normal.shape(0) == (std::size_t)gdim);
  std::tuple<int, std::int32_t, xt::xtensor<double, 1>, xt::xtensor<double, 1>>
      output;
  if (tdim == 2)
  {
    if (gdim == 2)
    {
      auto [status, cell_idx, x, X] = dolfinx_contact::compute_ray<2, 2>(
          mesh, std::span<const double, 2>(point.data(), 2),
          std::span<const double, 2>(normal.data(), 2), cells, max_iter, tol);
      xt::xtensor<double, 1> xt_x = xt::zeros<double>({(std::size_t)gdim});
      xt::xtensor<double, 1> xt_X = xt::zeros<double>({(std::size_t)tdim});
      std::copy(x.begin(), x.end(), xt_x.begin());
      std::copy(X.begin(), X.end(), xt_X.begin());
      output = std::make_tuple(status, cell_idx, xt_x, xt_X);
    }
    else if (gdim == 3)
    {
      auto [status, cell_idx, x, X] = dolfinx_contact::compute_ray<2, 3>(
          mesh, std::span<const double, 3>(point.data(), 3),
          std::span<const double, 3>(normal.data(), 3), cells, max_iter, tol);
      xt::xtensor<double, 1> xt_x = xt::zeros<double>({(std::size_t)gdim});
      xt::xtensor<double, 1> xt_X = xt::zeros<double>({(std::size_t)tdim});
      std::copy(x.begin(), x.end(), xt_x.begin());
      std::copy(X.begin(), X.end(), xt_X.begin());
      output = std::make_tuple(status, cell_idx, xt_x, xt_X);
    }
  }
  else if (tdim == 3)
  {
    auto [status, cell_idx, x, X] = dolfinx_contact::compute_ray<3, 3>(
        mesh, std::span<const double, 3>(point.data(), 3),
        std::span<const double, 3>(normal.data(), 3), cells, max_iter, tol);
    xt::xtensor<double, 1> xt_x = xt::zeros<double>({(std::size_t)gdim});
    xt::xtensor<double, 1> xt_X = xt::zeros<double>({(std::size_t)tdim});
    std::copy(x.begin(), x.end(), xt_x.begin());
    std::copy(X.begin(), X.end(), xt_X.begin());
    output = std::make_tuple(status, cell_idx, xt_x, xt_X);
  }
  return output;
}

//------------------------------------------------------------------------------------------------
