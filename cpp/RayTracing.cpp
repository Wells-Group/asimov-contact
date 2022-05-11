
// Copyright (C) 2022 Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "RayTracing.h"

std::tuple<int, std::array<double, 3>, std::array<double, 3>>
dolfinx_contact::compute_3D_ray(const dolfinx::mesh::Mesh& mesh,
                                const std::array<double, 3>& point,
                                const std::array<double, 3>& t1,
                                const std::array<double, 3>& t2, int cell,
                                int facet_index, const int max_iter,
                                const double tol)
{
  int status = -1;
  std::array<double, 3> x_k;
  std::array<double, 3> X_k;

  std::tuple<int, std::array<double, 3>, std::array<double, 3>> output
      = std::make_tuple(status, x_k, X_k);
  return output;
};