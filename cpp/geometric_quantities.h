// Copyright (C) 2021 Sarah Roggendorf and Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once
#include "QuadratureRule.h"
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/mesh/Mesh.h>

namespace dolfinx_contact
{

//-----------------------------------------------------------------------------
/// @brief Compute facet normal on physical cell
///
/// Computes the outward unit normal fora a point x located on
/// the surface of the physical cell. Each point x has a corresponding facet
/// index, relating to which local facet the point belongs to.
/// @note: The normal is computed using a covariant Piola transform
/// @param[in, out] work_array: Work array to avoid dynamic memory allocation,
/// use: `dolfinx_contact::allocate_pull_back_nonaffine` to get sufficent
/// memory.
/// @param[in] x: The point on the facet of the physical cell, size gdim
/// @param[in] gdim: The geometrical dimension of the cell
/// @param[in] tdim: The topogical dimension of the cell
/// @param[in] coordinate_dofs: Geometry coordinates of cell
/// @param[in] facet_index: Local facet index
/// @param[in] cmap: The coordinate element
/// @param[in] reference_normals: The facet normals on the reference cell
std::array<double, 3> push_forward_facet_normal(
    std::span<double> work_array, std::span<const double> x, std::size_t gdim,
    std::size_t tdim, cmdspan2_t coordinate_dofs, const std::size_t facet_index,
    const dolfinx::fem::CoordinateElement<double>& cmap, cmdspan2_t reference_normals);

/// @brief Allocate memory for pull-back on a non affine cell for a single
/// point.
/// @param[in] cmap The coordinate element
/// @param[in] gdim The geometrical dimension
/// @param[in] tdim The topological dimension
/// @returns Vector of sufficient size
std::vector<double>
allocate_pull_back_nonaffine(const dolfinx::fem::CoordinateElement<double>& cmap,
                             int gdim, int tdim);

/// @brief Pull back a single point of a non-affine cell to the reference
/// cell.
///
/// Given a single cell and a point, pull it back to the reference
/// element. To compute this pull back Newton's method is employed.

/// @param[in, out] X The point on the reference cell
/// @param[in, out] work_array Work array of at least size
/// 2*gdim*tdim+(tdim+1)*num_basis_functions + 9
/// @param[in] x The physical point
/// @param[in] cmap The coordinate element
/// @param[in] cell_geometry The cell geometry
/// @param[in] tol The tolerance for the Newton solver
/// @param[in] max_it The maximum number of Newton iterations
void pull_back_nonaffine(std::span<double> X, std::span<double> work_array,
                         std::span<const double> x,
                         const dolfinx::fem::CoordinateElement<double>& cmap,
                         cmdspan2_t cell_geometry, double tol = 1e-8,
                         const int max_it = 10);

/// Compute circumradius for a cell with given coordinates and determinant
/// of Jacobian
/// @param[in] mesh The mesh
/// @param[in] detJ The determinant of the Jacobian of the mapping to the
/// reference cell
/// @param[in] coordinate_dofs The cell geometry
/// @returns The circumradius of the cell
double compute_circumradius(const dolfinx::mesh::Mesh<double>& mesh,
                            double detJ, cmdspan2_t coordinate_dofs);

/// @brief Push forward facet normal
///
/// Compute normal of physical facet using a normalized covariant Piola
/// transform n_phys = J^{-T} n_ref / ||J^{-T} n_ref||
/// See: DOI: 10.1137/08073901X
/// @param[in, out] physical_normal The physical normal
/// @param[in] K The inverse of the Jacobian
/// @param[in] reference_normal The reference normal
template <class E, class F, class G>
void physical_facet_normal(E&& physical_normal, F&& K, G&& reference_normal)
{
  assert(physical_normal.size() == K.extent(1));
  const std::size_t tdim = K.extent(0);
  const std::size_t gdim = K.extent(1);
  std::fill(physical_normal.begin(), physical_normal.end(), 0);
  for (std::size_t i = 0; i < gdim; i++)
    for (std::size_t j = 0; j < tdim; j++)
      physical_normal[i] += K(j, i) * reference_normal[j];

  // Normalize vector
  double norm = 0;
  std::for_each(physical_normal.begin(), physical_normal.end(),
                [&norm](auto ni) { norm += std::pow(ni, 2); });
  norm = std::sqrt(norm);
  std::for_each(physical_normal.begin(), physical_normal.end(),
                [norm](auto& ni) { ni = ni / norm; });
}

//-----------------------------------------------------------------------------

} // namespace dolfinx_contact
