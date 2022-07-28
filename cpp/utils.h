// Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "QuadratureRule.h"
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
enum class Kernel
{
  Rhs,
  Jac,
  MeshTieRhs,
  MeshTieJac
};
// NOTE: this function should change signature to T * ,..... , num_links,
// num_dofs_per_link
template <typename T>
using kernel_fn
    = std::function<void(std::vector<std::vector<T>>&, xtl::span<const T>,
                         const T*, const double*, const int,
                         const std::size_t)>;

/// This function computes the pull back for a set of points x on a cell
/// described by coordinate_dofs as well as the corresponding Jacobian, their
/// inverses and their determinants
/// @param[in, out] J: Jacobians of transformation from reference element to
/// physical element. Shape = (num_points, tdim, gdim). Computed at each point
/// in x
/// @param[in, out] K: inverse of J at each point.
/// @param[in, out] detJ: determinant of J at each  point
/// @param[in ,out] X: pull pack of x (points on reference element)
/// @param[in] x: points on physical element
/// @param[in] coordinate_dofs: geometry coordinates of cell
/// @param[in] cmap: the coordinate element
//-----------------------------------------------------------------------------
void pull_back(mdspan3_t J, mdspan3_t K, std::span<double> detJ,
               std::span<double> X, cmdspan2_t x, cmdspan2_t coordinate_dofs,
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
sort_cells(const std::span<const std::int32_t>& cells,
           const std::span<std::int32_t>& perm);

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
std::array<std::size_t, 4>
evaluate_basis_shape(const dolfinx::fem::FunctionSpace& V,
                     const std::size_t num_points,
                     const std::size_t num_derivatives);

/// Get basis values (not unrolled for block size) for a set of points and
/// corresponding cells.
/// @param[in] V The function space
/// @param[in] x The coordinates of the points. It has shape
/// (num_points, gdim). Flattened row major
/// @param[in] cells An array of cell indices. cells[i] is the index
/// of the cell that contains the point x(i). Negative cell indices
/// can be passed, and the corresponding point will be ignored.
/// @param[in,out] basis_values The values at the points. Values are not
/// computed for points with a negative cell index. This argument must be passed
/// with the correct size (num_points, number_of_dofs, value_size). The basis
/// values are flattened row-major.
/// @param[in] number_of_derivatives FIXME: Add docs
void evaluate_basis_functions(const dolfinx::fem::FunctionSpace& V,
                              std::span<const double> x,
                              std::span<const std::int32_t> cells,
                              std::span<double> basis_values,
                              std::size_t num_derivatives);

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
/// @param[in,out] J - Jacobian between reference cell and physical cell
/// @param[in,out] K - inverse of J
/// @param[in,out] J_tot - J_f*J
/// @param[in,out] detJ_scratch Working memory for Jacobian computation. Has to
/// be at least 2*gdim*tdim.
/// @param[in] J_f - the Jacobian between reference facet and reference cell
/// @param[in] dphi - derivatives of coordinate basis tabulated for quardrature
/// points
/// @param[in] coords - the coordinates of the facet
/// @return absolute value of determinant of J_tot
double compute_facet_jacobians(std::size_t q, mdspan2_t J, mdspan2_t K,
                               mdspan2_t J_tot, std::span<double> detJ_scratch,
                               cmdspan2_t J_f, cmdspan3_t dphi,
                               cmdspan2_t coords);

/// @brief Convenience function to update Jacobians
///
/// For affine geometries, the input determinant is returned.
/// For non-affine geometries, the Jacobian, it's inverse and the total Jacobian
/// (J*J_f) is computed.
/// @param[in] cmap The coordinate element
std::function<double(std::size_t, double, mdspan2_t, mdspan2_t, mdspan2_t,
                     std::span<double>, cmdspan2_t, cmdspan3_t, cmdspan2_t)>
get_update_jacobian_dependencies(const dolfinx::fem::CoordinateElement& cmap);

/// @brief Convenience function to update facet normals
///
/// For affine geometries, a do nothing function is returned.
/// For non-affine geometries, a function updating the physical facet normal is
/// returned.
/// @param[in] cmap The coordinate element
std::function<void(std::span<double>, cmdspan2_t, cmdspan2_t,
                   const std::size_t)>
get_update_normal(const dolfinx::fem::CoordinateElement& cmap);

/// @brief Convert local entity indices to integration entities
///
/// Compute the active entities in DOLFINx format for a given integral type over
/// a set of entities If the integral type is cell, return the input, if it is
/// exterior facets, return a list of pairs (cell, local_facet_index), and if it
/// is interior facets, return a list of tuples (cell_0, local_facet_index_0,
/// cell_1, local_facet_index_1) for each entity.
/// @param[in] mesh The mesh
/// @param[in] entities List of mesh entities
/// @param[in] integral The type of integral
std::vector<std::int32_t>
compute_active_entities(std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
                        std::span<const std::int32_t> entities,
                        dolfinx::fem::IntegralType integral);

/// @brief Compute the geometry dof indices for a set of entities
///
/// For a set of entities, compute the geometry closure dofs of the entity.
///
/// @param[in] mesh The mesh
/// @param[in] dim The dimension of the entities
/// @param[in] entities List of mesh entities
/// @returns An adjacency list where the i-th link corresponds to the
/// closure dofs of the i-th input entity
dolfinx::graph::AdjacencyList<std::int32_t>
entities_to_geometry_dofs(const mesh::Mesh& mesh, int dim,
                          const std::span<const std::int32_t>& entity_list);

/// @brief find candidate facets within a given radius of puppet facets
///
/// Given a list of puppet facets and a list of candidate facets return
/// only those candidate facet within the given radius
///
/// @param[in] mesh The mesh
/// @param[in] puppet_facets Puppet facets
/// @param[in] candidate_facets Candidate facets
/// @param[in] radius The search radius
/// @return candidate facets within radius of puppet facets
std::vector<std::int32_t> find_candidate_surface_segment(
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
    const std::vector<std::int32_t>& puppet_facets,
    const std::vector<std::int32_t>& candidate_facets, const double radius);

/// @brief compute physical points on set of facets
///
/// Given a list of facets and the basis functions evaluated at set of points on
/// reference facets compute physical points
///
/// @param[in] mesh The mesh
/// @param[in] facets The list of facets as (cell, local_facet). The data is
/// flattened row-major
/// @param[in] offsets for accessing the basis_values for local_facet
/// @param[in] phi Basis functions evaluated at desired set of point osn
/// reference facet
/// @param[in, out] qp_phys Vector to store the quadrature points. Shape
/// (num_facets, num_q_points_per_facet, gdim). Flattened row-major
void compute_physical_points(const dolfinx::mesh::Mesh& mesh,
                             std::span<const std::int32_t> facets,
                             std::span<const std::size_t> offsets,
                             cmdspan4_t phi, std::span<double> qp_phys);

/// Compute the closest entity at every quadrature point on a subset of facets
/// on one mesh, to a subset of facets on the other mesh.
/// @param[in] quadrature_mesh The mesh to compute quadrature points on
/// @param[in] quadrature_facets The facets to compute quadrature points on,
/// defined as (cell, local_facet_index). Flattened row-major.
/// @param[in] candidate_mesh The mesh with the facets we want to compute the
/// distance to
/// @param[in] candidate_facets The facets on candidate_mesh,defined as (cell,
/// local_facet_index). Flattened row-major.
/// @returns An adjacency list for each input facet in quadrature facets, where
/// the links indicate which facet on the other mesh is closest for each
/// quadrature point.
dolfinx::graph::AdjacencyList<std::int32_t>
compute_distance_map(const dolfinx::mesh::Mesh& quadrature_mesh,
                     std::span<const std::int32_t> quadrature_facets,
                     const dolfinx::mesh::Mesh& candidate_mesh,
                     std::span<const std::int32_t> candidate_facets,
                     const QuadratureRule& q_rule);

} // namespace dolfinx_contact