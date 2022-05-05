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
#include <xtensor/xtensor.hpp>

namespace dolfinx_contact
{
// NOTE: this function should change signature to T * ,..... , num_links,
// num_dofs_per_link
template <typename T>
using kernel_fn
    = std::function<void(std::vector<std::vector<T>>&, const T*, const T*,
                         const double*, const int, const std::size_t)>;

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

/// Compute the negative restriction of a double, i.e. f(x)= x if x<0 else 0
double R_minus(double x);

/// Compute the derivative of the positive restriction (i.e.) the step function.
/// @note Evaluates to 0 at x=0
double dR_plus(double x);

/// Compute the derivative of the negative restriction (i.e.) the step function.
/// @note Evaluates to 0 at x=0
double dR_minus(double x);

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
/// @param[in,out] n_phys facet normal on physical element
void compute_normal(const xt::xtensor<double, 1>& n_ref,
                    const xt::xtensor<double, 2>& K,
                    xt::xarray<double>& n_phys);

/// Compute the following jacobians on a given facet:
/// J: physical cell -> reference cell (and its inverse)
/// J_tot: physical facet -> reference facet
/// @param[in] q - index of quadrature points
/// @param[in,out] J - Jacboian between reference cell and physical cell
/// @param[in,out] K - inverse of J
/// @param[in,out] J_tot - J_f*J
/// @param[in] J_f - the Jacobian between reference facet and reference cell
/// @param[in] dphi - derivatives of coordinate basis tabulated for quardrature
/// points
/// @param[in] coords - the coordinates of the facet
/// @return absolute value of determinant of J_tot
double compute_facet_jacobians(std::size_t q, xt::xtensor<double, 2>& J,
                               xt::xtensor<double, 2>& K,
                               xt::xtensor<double, 2>& J_tot,
                               const xt::xtensor<double, 2>& J_f,
                               const xt::xtensor<double, 3>& dphi,
                               const xt::xtensor<double, 2>& coords);

/// @brief Convenience function to update Jacobians
///
/// For affine geometries, the input determinant is returned.
/// For non-affine geometries, the Jacobian, it's inverse and the total Jacobian
/// (J*J_f) is computed.
/// @param[in] cmap The coordinate element
std::function<double(
    std::size_t, double, xt::xtensor<double, 2>&, xt::xtensor<double, 2>&,
    xt::xtensor<double, 2>&, const xt::xtensor<double, 2>&,
    const xt::xtensor<double, 3>&, const xt::xtensor<double, 2>&)>
get_update_jacobian_dependencies(const dolfinx::fem::CoordinateElement& cmap);

/// @brief Convenience function to update facet normals
///
/// For affine geometries, a do nothing function is returned.
/// For non-affine geometries, a function updating the physical facet normal is
/// returned.
/// @param[in] cmap The coordinate element
std::function<void(xt::xtensor<double, 1>&, const xt::xtensor<double, 2>&,
                   const xt::xtensor<double, 2>&, std::size_t)>
get_update_normal(const dolfinx::fem::CoordinateElement& cmap);

/// @brief Convert local entity indices to integration entities
///
/// Compute the active entities in DOLFINx format for a given integral type over
/// a set of entities If the integral type is cell, return the input, if it is
/// exterior facets, return a list of pairs (cell, local_facet_index), and if it
/// is interior facets, return a list of tuples (cell_0, local_facet_index_0,
/// cell_1, local_facet_index_1) for each entity.
///
/// @param[in] mesh The mesh
/// @param[in] entities List of mesh entities
/// @param[in] integral The type of integral
std::variant<std::vector<std::int32_t>,
             std::vector<std::pair<std::int32_t, int>>,
             std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>>
compute_active_entities(std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
                        tcb::span<const std::int32_t> entities,
                        dolfinx::fem::IntegralType integral);

} // namespace dolfinx_contact