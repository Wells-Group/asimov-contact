
// Copyright (C) 2022 Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "QuadratureRule.h"
#include "error_handling.h"
#include <dolfinx/common/utils.h>
#include <dolfinx/mesh/Mesh.h>

namespace dolfinx_contact
{
namespace stdex = std::experimental;
template <std::size_t A, std::size_t B>
using AB_span
    = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<double,
                                             stdex::extents<std::size_t, A, B>>;

namespace impl
{
/// Normalize a set of vectors by its length
/// @tparam The number of vectors
/// @tparam The length of the vectors
template <std::size_t A, std::size_t B>
void normalize(AB_span<A, B>& vectors)
{
  for (std::size_t i = 0; i < A; ++i)
  {
    double norm = 0;
    for (std::size_t j = 0; j < B; ++j)
      norm += vectors(i, j) * vectors(i, j);
    norm = std::sqrt(norm);
    for (std::size_t j = 0; j < B; ++j)
      vectors(i, j) = vectors(i, j) / norm;
  }
}

/// Compute two normalized tangets of a normal vector
///
/// @param[in] n the normal
/// @tparam gdim The dimension of the normal
/// @returns The tangent(s)
template <std::size_t gdim>
void compute_tangents(std::span<const double, gdim> n,
                      AB_span<gdim - 1, gdim> tangents)
{

  // Compute local maximum and create iteration array
  auto max_el = std::max_element(n.begin(), n.end(), [](double a, double b)
                                 { return std::norm(a) < std::norm(b); });
  auto max_pos = std::distance(n.begin(), max_el);
  std::size_t c = 0;
  std::vector<std::size_t> indices(gdim - 1, 0);
  for (std::size_t i = 0; i < gdim; ++i)
    if (i != (std::size_t)max_pos)
      indices[c++] = i;

  /// Compute first tangent
  for (std::size_t i = 0; i < gdim; ++i)
    tangents(0, i) = 1;
  tangents(0, max_pos) = 0;
  for (std::size_t i = 0; i < gdim - 1; ++i)
    tangents(0, max_pos) -= n[indices[i]] / n[max_pos];

  /// Compute second tangent by cross product
  if constexpr (gdim == 3)
  {
    tangents(1, 0) = tangents(0, 1) * n[2] - tangents(0, 2) * n[1];
    tangents(1, 1) = tangents(0, 2) * n[0] - tangents(0, 0) * n[2];
    tangents(1, 2) = tangents(0, 0) * n[1] - tangents(0, 1) * n[0];
  }
  normalize<gdim - 1, gdim>(tangents);
};

} // namespace impl

template <std::size_t tdim, std::size_t gdim>
class NewtonStorage
{
public:
  /// Creates storage for Newton solver with the following entries (listed in
  /// order of appearance in the work-array)
  /// The data-structures the Newton step requires is:
  /// dxi The Jacobian of the reference mapping, shape (tdim, tdim-1)
  /// X_k Solution on reference domain, size: tdim
  /// x_k Solution in phyiscal space, size: gdim
  /// xi_k Reference parameters, size: tdim-1
  /// dxi_k Gradient of reference parameter, size: tdim-1
  /// J Jacobian of cell basis, shape (gdim, tdim)
  /// dGk_tmp Temporary variable to invert Jacobian of Newton solver LHS, shape
  /// (gdim, tdim-1)
  /// dGk Newton solver LHS Jacobian, shape (gdim-1, tdim-1)
  /// dGk_inv Inverse of Newton solver Jacobian LHS, shape (tdim-1, gdim-1)
  /// Gk Resiudal (RHS) of Newton solver, size: gdim-1
  /// tangents Tangents of the ray, shape (gdim-1, gdim)
  /// point Point of origin of ray, size gdim
  NewtonStorage()
  {
    std::array<std::size_t, 12> distribution = {tdim * (tdim - 1),
                                                tdim,
                                                gdim,
                                                tdim - 1,
                                                tdim - 1,
                                                gdim * tdim,
                                                gdim * (tdim - 1),
                                                (gdim - 1) * (tdim - 1),
                                                (tdim - 1) * (gdim - 1),
                                                gdim - 1,
                                                (gdim - 1) * gdim,
                                                gdim};
    _offsets[0] = 0;
    std::partial_sum(distribution.cbegin(), distribution.cend(),
                     std::next(_offsets.begin()));
    _work_array = std::vector<double>(_offsets.back());
  }

  /// Return the Jacobian of the reference mapping, shape (tdim, tdim-1)
  AB_span<tdim, tdim - 1> dxi()
  {
    return AB_span<tdim, tdim - 1>(_work_array.data() + _offsets[0]);
  }

  /// Return the solution on the reference domain
  std::span<double, tdim> X_k()
  {
    return std::span<double, tdim>(_work_array.data() + _offsets[1], tdim);
  }

  /// Return the solution in physical space
  std::span<double, gdim> x_k()
  {
    return std::span<double, gdim>(_work_array.data() + _offsets[2], gdim);
  }

  /// Return the reference parameters
  std::span<double, tdim - 1> xi_k()
  {
    return std::span<double, tdim - 1>(_work_array.data() + _offsets[3],
                                       tdim - 1);
  }

  ///  Return the gradient of reference parameter
  std::span<double, tdim - 1> dxi_k()
  {
    return std::span<double, tdim - 1>(_work_array.data() + _offsets[4],
                                       tdim - 1);
  }

  /// Return the Jacobian of cell basis
  AB_span<gdim, tdim> J()
  {
    return AB_span<gdim, tdim>(_work_array.data() + _offsets[5]);
  }

  /// Return temporary variable to invert Jacobian of Newton solver LHS
  AB_span<gdim, tdim - 1> dGk_tmp()
  {
    return AB_span<gdim, tdim - 1>(_work_array.data() + _offsets[6]);
  }

  /// Return Newton solver LHS Jacobian
  AB_span<gdim - 1, tdim - 1> dGk()
  {
    return AB_span<gdim - 1, tdim - 1>(_work_array.data() + _offsets[7]);
  }

  /// Return inverse of Newton solver Jacobian LHS
  AB_span<tdim - 1, gdim - 1> dGk_inv()
  {
    return AB_span<tdim - 1, gdim - 1>(_work_array.data() + _offsets[8]);
  }

  /// Return resiudal (RHS) of Newton solver
  std::span<double, gdim - 1> Gk()
  {
    return std::span<double, gdim - 1>(_work_array.data() + _offsets[9],
                                       gdim - 1);
  }

  /// Return the tangents of the ray
  AB_span<gdim - 1, gdim> tangents()
  {
    return AB_span<gdim - 1, gdim>(_work_array.data() + _offsets[10]);
  }

  /// Return resiudal (RHS) of Newton solver
  std::span<double, gdim> point()
  {
    return std::span<double, gdim>(_work_array.data() + _offsets[11], gdim);
  }

private:
  std::vector<double> _work_array;
  std::array<std::size_t, 13> _offsets;
};

/// @brief Compute the solution to the ray tracing problem for a single
/// cell.
///
/// The implementation solves dot(\Phi(\xi, \eta)-p, t_i)=0, i=1,..,,
/// tdim-1 where \Phi(\xi,\eta) is the parameterized mapping from the
/// reference facet to the physical facet, p the point of origin of the
/// ray, and t_i is the ith tangents defining the ray. For more details,
/// see DOI: 10.1016/j.compstruc.2015.02.027 (eq 14).
///
/// @note The problem is solved using Newton's method
///
/// @tparam tdim The topological dimension of the cell
/// @tparam gdim The geometrical dimension of the cell
///
/// @param[in,out] storage Structure holding all memory required for the
/// newton iteration.
/// @note It is expected that the variables tangents, point, xi is
/// filled with appropriate input values
/// @note All other variables of the class is updated.
/// @param[in, out] basis_values Work_array for basis evaluation. Should
/// have the length given by `cmap.tabulate_shape(1,1)`
/// @param[in] max_iter Maximum number of iterations for the Newton
/// solver
/// @param[in] tol The tolerance for termination of the Newton solver
/// @param[in] cmap The coordinate element
/// @param[in] cell_type The cell type of the mesh
/// @param[in] coordinate_dofs The cell geometry, shape (num_dofs_g,
/// gdim). Flattened row-major
/// @param[in] reference_map Function mapping from reference parameters
/// (xi, eta) to the physical element
template <std::size_t tdim, std::size_t gdim>
int raytracing_cell(
    NewtonStorage<tdim, gdim>& storage, std::span<double> basis_values,
    std::array<std::size_t, 4> basis_shape, int max_iter, double tol,
    const dolfinx::fem::CoordinateElement<double>& cmap,
    dolfinx::mesh::CellType cell_type, std::span<const double> coordinate_dofs,
    const std::function<void(std::span<const double, tdim - 1>,
                             std::span<double, tdim>)>& reference_map)
{
  if constexpr ((gdim != 2) and (gdim != 3))
    throw std::invalid_argument("The geometrical dimension has to be 2 or 3");

  int status = -1;
  auto x_k = storage.x_k();
  std::fill(x_k.begin(), x_k.end(), 0);

  // Set initial guess for Newton-iteration (midpoint of facet)
  auto xi_k = storage.xi_k();
  if constexpr (tdim == 3)
  {
    xi_k[0] = 0.5;
    xi_k[1] = 0.25;
  }
  else if constexpr (tdim == 2)
    xi_k[0] = 0.5;
  else
    throw std::invalid_argument("The topological dimension has to be 2 or 3");

  auto X_k = storage.X_k();
  auto dGk = storage.dGk();
  auto dGk_inv = storage.dGk_inv();
  auto Gk = storage.Gk();
  auto point = storage.point();
  auto tangents = storage.tangents();
  auto J = storage.J();
  auto dGk_tmp = storage.dGk_tmp();
  auto dxi = storage.dxi();
  auto dxi_k = storage.dxi_k();
  assert(std::size_t(std::reduce(basis_shape.cbegin(), basis_shape.cend(), 1,
                                 std::multiplies{}))
         == basis_values.size());
  cmdspan4_t basis(basis_values.data(), basis_shape);
  auto dphi = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      basis, std::pair{1, tdim + 1}, 0,
      MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  cmdspan2_t coords(coordinate_dofs.data(), cmap.dim(), gdim);
  mdspan2_t _xk(x_k.data(), 1, gdim);
  for (int k = 0; k < max_iter; ++k)
  {
    // Evaluate reference coordinate at current iteration
    reference_map(xi_k, X_k);

    // Tabulate coordinate element basis function
    cmap.tabulate(1, X_k, {1, tdim}, basis_values);

    // Push forward reference coordinate
    dolfinx::fem::CoordinateElement<double>::push_forward(
        _xk, coords,
        MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            basis, 0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0));

    // Compute residual at current iteration
    std::fill(Gk.begin(), Gk.end(), 0);
    for (std::size_t i = 0; i < gdim; ++i)
      for (std::size_t j = 0; j < gdim - 1; ++j)
        Gk[j] += (x_k[i] - point[i]) * tangents(j, i);

    // Compute Jacobian
    for (std::size_t i = 0; i < gdim; ++i)
      for (std::size_t j = 0; j < tdim; ++j)
        J(i, j) = 0;
    dolfinx::fem::CoordinateElement<double>::compute_jacobian(dphi, coords, J);

    /// Compute dGk/dxi
    for (std::size_t i = 0; i < gdim; ++i)
      for (std::size_t j = 0; j < tdim - 1; ++j)
        dGk_tmp(i, j) = 0;
    dolfinx::math::dot(J, dxi, dGk_tmp);

    for (std::size_t i = 0; i < gdim - 1; ++i)
      for (std::size_t j = 0; j < tdim - 1; ++j)
        dGk(i, j) = 0;

    for (std::size_t i = 0; i < gdim - 1; ++i)
      for (std::size_t j = 0; j < tdim - 1; ++j)
        for (std::size_t l = 0; l < gdim; ++l)
          dGk(i, j) += dGk_tmp(l, j) * tangents(i, l);

    // Compute determinant of dGk/dxi to determine if invertible
    double det_dGk;
    if constexpr ((gdim != tdim) and (gdim == 3) and (tdim == 2))
    {
      // If non-square matrix compute det(A) = sqrt(det(A^T A))
      double ATA = dGk(0, 0) * dGk(0, 0) + dGk(1, 0) * dGk(1, 0);
      det_dGk = std::sqrt(ATA);
    }
    else
      det_dGk = dolfinx::math::det(dGk);

    // Terminate if dGk/dxi is not invertible
    if (std::abs(det_dGk) < tol)
    {
      status = -2;
      break;
    }

    // Invert dGk/dxi
    if constexpr (gdim == tdim)
      dolfinx::math::inv(dGk, dGk_inv);
    else
      dolfinx::math::pinv(dGk, dGk_inv);

    // Compute dxi
    std::fill(dxi_k.begin(), dxi_k.end(), 0);
    for (std::size_t i = 0; i < tdim - 1; ++i)
      for (std::size_t j = 0; j < gdim - 1; ++j)
        dxi_k[i] += dGk_inv(i, j) * Gk[j];

    // Check for convergence
    double norm_dxi = 0;
    for (std::size_t i = 0; i < tdim - 1; i++)
      norm_dxi += dxi_k[i] * dxi_k[i];
    if (norm_dxi < tol * tol)
    {
      status = 1;
      break;
    }

    // Update xi
    std::transform(xi_k.begin(), xi_k.end(), dxi_k.begin(), xi_k.begin(),
                   [](auto x, auto y) { return x - y; });
  }
  // Check if converged  parameters are valid
  switch (cell_type)
  {
  case dolfinx::mesh::CellType::tetrahedron:
    if ((xi_k[0] < -tol) or (xi_k[0] > 1 + tol) or (xi_k[1] < -tol)
        or (xi_k[1] > 1 - xi_k[0] + tol))
    {
      status = -3;
    }
    break;
  case dolfinx::mesh::CellType::hexahedron:
    if ((xi_k[0] < -tol) or (xi_k[0] > 1 + tol) or (xi_k[1] < -tol)
        or (xi_k[1] > 1 + tol))
    {
      status = -3;
    }
    break;
  case dolfinx::mesh::CellType::triangle:
    if ((xi_k[0] < -tol) or (xi_k[0] > 1 + tol))
      status = -3;
    break;
  case dolfinx::mesh::CellType::quadrilateral:
    if ((xi_k[0] < -tol) or (xi_k[0] > 1 + tol))
      status = -3;
    break;
  default:
    throw std::invalid_argument("Unsupported cell type");
  }
  return status;
}

/// @brief Compute the first intersection between a ray and a set of
/// facets in the mesh templated for the topological dimension.
///
/// @param[in] mesh The mesh
/// @param[in] point The point of origin for the ray
/// @param[in] tangents The tangents of the ray. Each row corresponds to
/// a tangent.
/// @param[in] cells List of tuples (cell, facet), where the cell index
/// is local to process and the facet index is local to the cell. Data
/// is flattened row-major.
/// @param[in] max_iter The maximum number of iterations to use for
/// Newton's method
/// @param[in] tol The tolerance for convergence in Newton's method
/// @returns A quadruplet (status, cell_idx, point, reference_point),
/// where x is the convergence status, cell_idx is which entry in the
/// input list the ray goes through and point, reference_point is the
/// collision point in global and reference coordinates respectively.
/// @note The convergence status is >0 if converging, -1 if the facet is
/// if the maximum number of iterations are reached, -2 if the facet is
/// parallel with the tangent, -3 if the Newton solver finds a solution
/// outside the element.
/// @tparam tdim The topological dimension of the cell
template <std::size_t tdim, std::size_t gdim>
std::tuple<int, std::int32_t, std::array<double, gdim>,
           std::array<double, tdim>>
compute_ray(const dolfinx::mesh::Mesh<double>& mesh,
            std::span<const double, gdim> point,
            std::span<const double, gdim> normal,
            std::span<const std::int32_t> cells, const int max_iter = 25,
            const double tol = 1e-8)
{
  int status = -1;
  dolfinx::mesh::CellType cell_type = mesh.topology()->cell_type();
  if ((mesh.topology()->dim() != tdim) or (mesh.geometry().dim() != gdim))
    throw std::invalid_argument("Invalid topological or geometrical dimension");

  const dolfinx::fem::CoordinateElement<double>& cmap = mesh.geometry().cmap();

  // Get cell coordinates/geometry
  const dolfinx::mesh::Geometry<double>& geometry = mesh.geometry();
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      const std::int32_t,
      MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
      x_dofmap = geometry.dofmap();
  std::span<const double> x_g = geometry.x();
  const std::size_t num_dofs_g = cmap.dim();
  std::vector<double> coordinate_dofs(num_dofs_g * gdim);

  // Temporary variables
  const std::array<std::size_t, 4> basis_shape = cmap.tabulate_shape(1, 1);
  std::vector<double> basis_values(std::reduce(
      basis_shape.cbegin(), basis_shape.cend(), 1, std::multiplies{}));

  std::size_t cell_idx = -1;
  auto allocated_memory = NewtonStorage<tdim, gdim>();
  auto tangents = allocated_memory.tangents();
  impl::compute_tangents<gdim>(normal, tangents);
  std::span<double, gdim> m_point = allocated_memory.point();
  auto dxi = allocated_memory.dxi();
  std::copy_n(point.begin(), gdim, m_point.begin());

  // Check for parameterization and jacobian parameterization
  error::check_cell_type(cell_type);
  basix::cell::type basix_cell
      = dolfinx::mesh::cell_type_to_basix_type(cell_type);
  assert(dolfinx::mesh::cell_dim(cell_type) == tdim);

  // Get facet jacobians from Basix
  auto [ref_jac, jac_shape] = basix::cell::facet_jacobians<double>(basix_cell);
  assert(tdim == jac_shape[1]);
  assert(tdim - 1 == jac_shape[2]);
  mdspan_t<const double, 3> facet_jacobians(ref_jac.data(), jac_shape);

  // Get basix geometry information
  std::pair<std::vector<double>, std::array<std::size_t, 2>> bgeometry
      = basix::cell::geometry<double>(basix_cell);
  auto xb = bgeometry.first;
  auto x_shape = bgeometry.second;
  const std::vector<std::vector<int>> facets
      = basix::cell::topology(basix_cell)[tdim - 1];

  for (std::size_t c = 0; c < cells.size(); c += 2)
  {

    // Get cell geometry
    auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        x_dofmap, cells[c], MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t j = 0; j < x_dofs.size(); ++j)
    {
      std::copy_n(std::next(x_g.begin(), 3 * x_dofs[j]), gdim,
                  std::next(coordinate_dofs.begin(), gdim * j));
    }

    // Assign Jacobian of reference mapping
    for (std::size_t i = 0; i < tdim; ++i)
      for (std::size_t j = 0; j < tdim - 1; ++j)
        dxi(i, j) = facet_jacobians(cells[c + 1], i, j);

    dolfinx::common::Timer tfc("~~get parameterization");

    // Get parameterization map
    std::function<void(std::span<const double, tdim - 1>,
                       std::span<double, tdim>)>
        reference_map
        = [&xb, &x_shape, &facets, facet_index = cells[c + 1]](
              std::span<const double, tdim - 1> xi, std::span<double, tdim> X)
    {
      const std::vector<int>& facet = facets[facet_index];
      dolfinx_contact::cmdspan2_t x(xb.data(), x_shape);
      for (std::size_t i = 0; i < tdim; ++i)
      {
        X[i] = x(facet.front(), i);
        for (std::size_t j = 0; j < tdim - 1; ++j)
          X[i] += (x(facet[j + 1], i) - x(facet.front(), i)) * xi[j];
      }
    };
    tfc.stop();

    status = raytracing_cell<tdim, gdim>(
        allocated_memory, basis_values, basis_shape, max_iter, tol, cmap,
        cell_type, coordinate_dofs, reference_map);
    if (status > 0)
    {
      cell_idx = c / 2;
      break;
    }
  }

  if (status < 0)
    spdlog::warn("No ray through the facets have been found");

  std::array<double, gdim> x;
  std::array<double, tdim> X;
  auto x_fin = allocated_memory.x_k();
  auto X_fin = allocated_memory.X_k();
  std::copy_n(x_fin.begin(), gdim, x.begin());
  std::copy_n(X_fin.begin(), tdim, X.begin());
  std::tuple<int, std::int32_t, std::array<double, gdim>,
             std::array<double, tdim>>
      output = std::make_tuple(status, cell_idx, x, X);
  return output;
}

/// @brief Compute the first intersection between a ray and a set of
/// facets in the mesh.
///
/// @param[in] mesh The mesh
/// @param[in] point The point of origin for the ray
/// @param[in] normal The vector defining the direction of the ray
/// @param[in] cells List of tuples (cell, facet), where the cell index
/// is local to process and the facet index is local to the cell cell.
/// Data is flattened row-major.
/// @param[in] max_iter The maximum number of iterations to use for
/// Newton's method
/// @param[in] tol The tolerance for convergence in Newton's method
/// @returns A quadruplet (status, cell_idx, x, X), where status is the
/// convergence status, cell_idx is which entry in the input list the
/// ray goes through and x, X is the collision point in global and
/// reference coordinates, respectively.
/// @note The convergence status is >0 if converging, -1 if the facet is
/// if the maximum number of iterations are reached, -2 if the facet is
/// parallel with the tangent, -3 if the Newton solver finds a solution
/// outside the element.
std::tuple<int, std::int32_t, std::vector<double>, std::vector<double>>
raytracing(const dolfinx::mesh::Mesh<double>& mesh,
           std::span<const double> point, std::span<const double> normal,
           std::span<const std::int32_t> cells, const int max_iter = 25,
           const double tol = 1e-8);

} // namespace dolfinx_contact
