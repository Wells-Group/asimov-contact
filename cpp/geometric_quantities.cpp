// Copyright (C) 2021-2022 Sarah Roggendorf and Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "geometric_quantities.h"
#include <xtensor/xmath.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xtensor.hpp>
using namespace dolfinx_contact;
std::array<double, 3> dolfinx_contact::push_forward_facet_normal(
    xt::xtensor<double, 2>& J, xt::xtensor<double, 2>& K,
    const std::array<double, 3>& x,
    const xt::xtensor<double, 2>& coordinate_dofs,
    const std::size_t facet_index, const dolfinx::fem::CoordinateElement& cmap,
    const xt::xtensor<double, 2>& reference_normals)
{
  assert(J.shape(0) == K.shape(1));
  assert(K.shape(0) == J.shape(1));

  // Shapes needed for computing the Jacobian inverse
  const size_t tdim = K.shape(0);
  const size_t gdim = K.shape(1);

  // Data structures for computing J inverse
  xt::xtensor<double, 4> phi(cmap.tabulate_shape(1, 1));
  xt::xtensor<double, 2> dphi({tdim, cmap.tabulate_shape(1, 1)[2]});
  xt::xtensor<double, 2> X({1, tdim});

  // Compute Jacobian inverse
  if (cmap.is_affine())
  {
    // Affine Jacobian can be computed at any point in the cell (0,0,0) in
    // the reference cell
    std::fill(X.begin(), X.end(), 0);
    cmap.tabulate(1, X, phi);
    dphi = xt::view(phi, xt::range(1, tdim + 1), 0, xt::all(), 0);
    std::fill(J.begin(), J.end(), 0);
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs, J);
    std::fill(K.begin(), K.end(), 0);
    dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
  }
  else
  {
    // Copy x into gdim restriction
    xt::xtensor<double, 2> _x({1, gdim});
    std::transform(x.cbegin(), std::next(x.cbegin() + gdim), _x.begin(),
                   [](auto xi) { return xi; });

    // For non-affine geometries we have to compute the point in the reference
    // cell, which is a nonlinear operation. Internally cmap uses a
    // Newton-solver to get the reference coordinates X
    cmap.pull_back_nonaffine(X, _x, coordinate_dofs);
    cmap.tabulate(1, X, phi);

    dphi = xt::view(phi, xt::range(1, tdim + 1), 0, xt::all(), 0);
    std::fill(J.begin(), J.end(), 0);
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs, J);
    std::fill(K.begin(), K.end(), 0);
    dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
  }
  // Push forward reference facet normal
  std::array<double, 3> normal = {0, 0, 0};
  physical_facet_normal(xtl::span(normal.data(), gdim), K,
                        xt::row(reference_normals, facet_index));
  return normal;
}

//-----------------------------------------------------------------------------
double dolfinx_contact::compute_circumradius(
    const dolfinx::mesh::Mesh& mesh, double detJ,
    const xt::xtensor<double, 2>& coordinate_dofs)
{
  const dolfinx::mesh::CellType cell_type = mesh.topology().cell_type();
  const int gdim = mesh.geometry().dim();

  switch (cell_type)
  {
  case dolfinx::mesh::CellType::triangle:
  {
    // Formula for circumradius of a triangle with sides with length a, b, c
    // is R = a b c / (4 A) where A is the area of the triangle
    const double ref_area = basix::cell::volume(basix::cell::type::triangle);
    double area = ref_area * std::abs(detJ);

    // Compute the lenghts of each side of the cell
    std::array<double, 3> sides
        = {0, 0, 0}; // Array to hold lenghts of sides of triangle
    for (int i = 0; i < gdim; i++)
    {
      sides[0] += std::pow(coordinate_dofs(0, i) - coordinate_dofs(1, i), 2);
      sides[1] += std::pow(coordinate_dofs(1, i) - coordinate_dofs(2, i), 2);
      sides[2] += std::pow(coordinate_dofs(2, i) - coordinate_dofs(0, i), 2);
    }
    std::for_each(sides.begin(), sides.end(),
                  [](double& side) { side = std::sqrt(side); });

    return sides[0] * sides[1] * sides[2] / (4 * area);
  }
  case dolfinx::mesh::CellType::tetrahedron:
  {
    // Formula for circunradius of a tetrahedron with volume V.
    // Given three edges meeting at a vertex with length a, b, c,
    // and opposite edges with corresponding length A, B, C we have that the
    // circumradius
    // R = sqrt((aA + bB + cC)(aA+bB-cC)(aA-bB+cC)(-aA +bB+cC))/24V
    const double ref_volume
        = basix::cell::volume(basix::cell::type::tetrahedron);
    double cellvolume = std::abs(detJ) * ref_volume;

    // Edges ordered as a, b, c, A, B, C
    std::array<double, 6> edges = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < gdim; i++)
    {
      // Accummulate a^2, b^2, c^2
      edges[0] += std::pow(coordinate_dofs(0, i) - coordinate_dofs(1, i), 2);
      edges[1] += std::pow(coordinate_dofs(0, i) - coordinate_dofs(2, i), 2);
      edges[2] += std::pow(coordinate_dofs(0, i) - coordinate_dofs(3, i), 2);

      // Accumulate A^2, B^2, C^2
      edges[3] += std::pow(coordinate_dofs(2, i) - coordinate_dofs(3, i), 2);
      edges[4] += std::pow(coordinate_dofs(1, i) - coordinate_dofs(3, i), 2);
      edges[5] += std::pow(coordinate_dofs(1, i) - coordinate_dofs(2, i), 2);
    }
    // Compute length of each edge
    std::for_each(edges.begin(), edges.end(),
                  [](double& edge) { edge = std::sqrt(edge); });

    // Compute temporary variables
    const double aA = edges[0] * edges[3];
    const double bB = edges[1] * edges[4];
    const double cC = edges[2] * edges[5];

    // Compute circumradius
    double h = std::sqrt((aA + bB + cC) * (aA + bB - cC) * (aA - bB + cC)
                         * (-aA + bB + cC))
               / (24 * cellvolume);
    return h;
  }
  default:
    throw std::invalid_argument("Unsupported cell_type "
                                + dolfinx::mesh::to_string(cell_type));
  }
}