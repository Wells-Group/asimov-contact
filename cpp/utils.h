// Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <basix/cell.h>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx_cuas/QuadratureRule.hpp>
#include <xtensor/xtensor.hpp>
namespace dolfinx_contact
{

/// This function computes the pull back for a set of points x on a cell
/// described by coordinate_dofs as well as the corresponding Jacobian, their
/// inverses and their determinants
/// @param[in, out] J: Jacobians of transformation from reference element to
/// physical element. Shape = (num_points, tdim, gdim). Computed at each point
/// in x
/// @param[in, out] K: inverse of J at each point.
/// @param[in, out] detJ: determinant of J at each  point
/// @param[in] x: points on physical element
/// @param[in ,out] X: pull pack of x (points on reference element)
/// @param[in] coordinate_dofs: geometry coordinates of cell
/// @param[in] cmap: the coordinate element
//-----------------------------------------------------------------------------
void pull_back(xt::xtensor<double, 3>& J, xt::xtensor<double, 3>& K,
               xt::xtensor<double, 1>& detJ, const xt::xtensor<double, 2>& x,
               xt::xtensor<double, 2>& X,
               const xt::xtensor<double, 2>& coordinate_dofs,
               const dolfinx::fem::CoordinateElement& cmap);

//-----------------------------------------------------------------------------
/// This function computes the basis function values on a given cell at a
/// given set of points
/// @param[in, out] J: Jacobians of transformation from reference element to
/// physical element. Shape = (num_points, tdim, gdim). Computed at each point
/// in x
/// @param[in, out] K: inverse of J at each point.
/// @param[in, out] detJ: determinant of J at each  point
/// @param[in] x: points on physical element
/// @param[in] coordinate_dofs: geometry coordinates of cell
/// @param[in] index: the index of the cell (local to process)
/// @param[in] perm: permutation infor for cell
/// @param[in] element: the corresponding finite element
/// @param[in] cmap: the coordinate element
xt::xtensor<double, 3>
get_basis_functions(xt::xtensor<double, 3>& J, xt::xtensor<double, 3>& K,
                    xt::xtensor<double, 1>& detJ,
                    const xt::xtensor<double, 2>& x,
                    const xt::xtensor<double, 2>& coordinate_dofs,
                    const std::int32_t index, const std::int32_t perm,
                    std::shared_ptr<const dolfinx::fem::FiniteElement> element,
                    const dolfinx::fem::CoordinateElement& cmap);

/// @param[in] cells: the cells to be sorted
/// @param[in, out] perm: the permutation for the sorted cells
/// @param[out] pair(unique_cells, offsets): unique_cells is a vector of
/// sorted cells with all duplicates deleted, offsets contains the start and
/// end for each unique value in the sorted vector with all duplicates
// Example: cells = [5, 7, 6, 5]
//          unique_cells = [5, 6, 7]
//          offsets = [0, 2, 3, 4]
//          perm = [0, 3, 2, 1]
// Then given a cell and its index ("i") in unique_cells, one can recover the
// indices for its occurance in cells with perm[k], where
// offsets[i]<=k<offsets[i+1]. In the example if i = 0, then perm[k] = 0 or
// perm[k] = 3.
std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
sort_cells(const xtl::span<const std::int32_t>& cells,
           const xtl::span<std::int32_t>& perm);

/// @param[in] u: dolfinx function on function space base on basix element
/// @param[in] mesh: mesh to be updated
/// Adds perturbation u to mesh
void update_geometry(const dolfinx::fem::Function<PetscScalar>& u,
                     std::shared_ptr<dolfinx::mesh::Mesh> mesh);

/// Compute the positive restriction of a double, i.e. f(x)= x if x>0 else 0
double R_plus(double x);

/// Compute the derivative of the positive restriction (i.e.) the step function.
/// @note Evaluates to 0 at x=0
double dR_plus(double x);

/// Get shape of in,out variable for filling basis functions in for
/// evaluate_basis_functions
std::array<std::size_t, 3>
evaulate_basis_shape(const dolfinx::fem::FunctionSpace& V,
                     const std::size_t num_points);

/// Get basis values (not unrolled for block size) for a set of points and
/// corresponding cells.
/// @param[in] V The function space
/// @param[in] x The coordinates of the points. It has shape
/// (num_points, 3).
/// @param[in] cells An array of cell indices. cells[i] is the index
/// of the cell that contains the point x(i). Negative cell indices
/// can be passed, and the corresponding point will be ignored.
/// @param[in,out] basis_values The values at the points. Values are not
/// computed for points with a negative cell index. This argument must be passed
/// with the correct size (num_points, number_of_dofs, value_size).
void evaluate_basis_functions(const dolfinx::fem::FunctionSpace& V,
                              const xt::xtensor<double, 2>& x,
                              const xtl::span<const std::int32_t>& cells,
                              xt::xtensor<double, 3>& basis_values);

/// Compute physical normal
/// @param[in] n_ref facet normal on reference element
/// @param[in] K inverse Jacobian
/// @param[out] n_phys facet normal on physical element
void compute_normal(const xt::xtensor<double, 1>& n_ref,
                    const xt::xtensor<double, 2>& K,
                    xt::xarray<double>& n_phys);

/// Compute jacobians on a given facet
/// @param[in] q - index of quadrature points
/// @param[in] dphi - derivatives of coordinate basis tabulated for quardrature
/// points
/// @param[in] coords - the coordinates of the facet
/// @param[in] J_f - the Jacobian between reference facet and reference cell
/// @param[out] J - Jacboian between reference cell and physical cell
/// @param[out] K - inverse of J
/// @param[out] J_tot - J_f*J
/// @return absolute value of determinant of J_tot
double compute_facet_jacobians(int q, const xt::xtensor<double, 3>& dphi,
                               const xt::xtensor<double, 2>& coords,
                               const xt::xtensor<double, 2> J_f,
                               xt::xtensor<double, 2>& J,
                               xt::xtensor<double, 2>& K,
                               xt::xtensor<double, 2>& J_tot);

} // namespace dolfinx_contact