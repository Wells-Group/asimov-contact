// Copyright (C) 2022 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "QuadratureRule.h"
#include "error_handling.h"
#include "utils.h"
#include <dolfinx.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>

namespace dolfinx_contact
{
/// @brief  Typedef
using jac_fn = std::function<double(
    double, mdspan_t<double, 2>, mdspan_t<double, 2>, mdspan_t<double, 2>,
    std::span<double>, mdspan_t<const double, 2>,
    mdspan_t<const double, 2, stdex::layout_stride>,
    mdspan_t<const double, 2>)>;

/// @brief  Typedef
using normal_fn
    = std::function<void(std::span<double>, mdspan_t<const double, 2>,
                         mdspan_t<const double, 2>, const std::size_t)>;

/// @brief  Kernal data class
class KernelData
{
public:
  /// @brief Kernel data constructor
  ///
  /// Generates data that is common to all contact kernels
  ///
  ///@param[in] V The function space
  ///@param[in] q_rule The quadrature rules
  ///@param[in] cstrides The strides for individual coefficients used in
  /// the kernel
  KernelData(const dolfinx::fem::FunctionSpace<double>& V,
             const QuadratureRule& q_rule,
             const std::vector<std::size_t>& cstrides);

  /// Return geometrical dimension
  std::uint32_t gdim() const { return _gdim; }

  /// Return topological dimension
  std::uint32_t tdim() const { return _tdim; }

  /// Return number of dofs for geometry
  int num_coordinate_dofs() const { return _num_coordinate_dofs; }

  /// return whether cell geometry is affine
  bool affine() const { return _affine; }

  /// return number of dofs pers cell
  std::size_t ndofs_cell() const { return _ndofs_cell; }

  /// Return block size
  std::size_t bs() const { return _bs; }

  /// Return quadrature rule offsets for index f
  std::size_t qp_offsets(std::size_t f) const
  {
    assert(f < _qp_offsets.size());
    return _qp_offsets[f];
  }

  /// Return basis functions at quadrature points for facet f
  mdspan_t<const double, 2, stdex::layout_stride> phi() const
  {
    mdspan_t<const double, 4> full_basis(_basis_values.data(), _basis_shape);
    return MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        full_basis, 0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  }

  /// Return grad(_phi) at quadrature points for facet f
  mdspan_t<const double, 3, stdex::layout_stride> dphi() const
  {
    mdspan_t<const double, 4> full_basis(_basis_values.data(), _basis_shape);
    return MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        full_basis, std::pair{1, (std::size_t)_tdim + 1},
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  }

  /// Return gradient of coordinate bases at quadrature points for facet f
  mdspan_t<const double, 3, stdex::layout_stride> dphi_c() const
  {
    mdspan_t<const double, 4> full_basis(_c_basis_values.data(),
                                         _c_basis_shape);
    return MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        full_basis, std::pair{1, (std::size_t)_tdim + 1},
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  }

  /// Return coefficient offsets of coefficient i
  std::size_t offsets(std::size_t i) const { return _offsets[i]; }

  /// @brief TODO
  /// @return TODO
  const std::vector<std::size_t>& offsets_array() const { return _offsets; }

  /// Return reference facet normals
  mdspan_t<const double, 2> facet_normals() const
  {
    return mdspan_t<const double, 2>(_facet_normals.data(), _normals_shape);
  }

  /// Compute the following jacobians on a given facet.
  ///
  /// J physical cell -> reference cell (and its inverse)
  /// J_tot: physical facet -> reference facet
  ///
  /// @param[in] q index of quadrature points
  /// @param[in] facet_index The index of the facet local to the cell
  /// @param[in] detJ TODO
  /// @param[in,out] J Jacobian between reference cell and physical cell
  /// @param[in,out] K inverse of J
  /// @param[in,out] J_tot J_f*J
  /// @param[in,out] detJ_scratch Working memory to compute
  /// determinants
  /// @param[in] coords the coordinates of the facet
  /// @return absolute value of determinant of J_tot
  double update_jacobian(std::size_t q, std::size_t facet_index, double detJ,
                         mdspan_t<double, 2> J, mdspan_t<double, 2> K,
                         mdspan_t<double, 2> J_tot,
                         std::span<double> detJ_scratch,
                         mdspan_t<const double, 2> coords) const
  {
    mdspan_t<const double, 4> full_basis(_c_basis_values.data(),
                                         _c_basis_shape);
    const std::size_t q_pos = _qp_offsets[facet_index] + q;
    auto dphi_fc = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        full_basis, std::pair{1, (std::size_t)_tdim + 1}, q_pos,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
    mdspan_t<const double, 3> ref_jacs(_ref_jacobians.data(), _jac_shape);
    auto J_f = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        ref_jacs, (std::size_t)facet_index,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    return _update_jacobian(detJ, J, K, J_tot, detJ_scratch, J_f, dphi_fc,
                            coords);
  }

  /// Compute the following jacobians on a given facet at first
  /// quadrature point.
  ///
  /// J: physical cell -> reference cell (and its inverse)
  /// J_tot: physical facet -> reference facet
  ///
  /// @param[in] facet_index The index of the facet local to the cell
  /// @param[in,out] J Jacobian between reference cell and physical cell
  /// @param[in,out] K inverse of J
  /// @param[in,out] J_tot J_f*J
  /// @param[in,out] detJ_scratch Working memory, min size (2*gdim*tdim)
  /// @param[in] coords the coordinates of the facet
  /// @return absolute value of determinant of J_tot
  double compute_first_facet_jacobian(std::size_t facet_index,
                                      mdspan_t<double, 2> J,
                                      mdspan_t<double, 2> K,
                                      mdspan_t<double, 2> J_tot,
                                      std::span<double> detJ_scratch,
                                      mdspan_t<const double, 2> coords) const;

  /// update normal
  /// @param[in, out] n The facet normal
  /// @param[in] K The inverse Jacobian
  /// @param[in] local_index The facet index local to the cell
  void update_normal(std::span<double> n, mdspan_t<const double, 2> K,
                     std::size_t local_index) const;

  /// Return quadrature weights for the i-th facet
  std::span<const double> weights(std::size_t i) const;

  /// return the reference jacobians
  mdspan_t<const double, 3> ref_jacobians() const;

private:
  // geometrical dimension
  std::uint32_t _gdim;

  // topological dimension
  std::uint32_t _tdim;

  // number of dofs for geometry
  int _num_coordinate_dofs;

  // store whether cell geometry is affine
  bool _affine;

  // number of dofs per cell
  std::uint32_t _ndofs_cell;

  // block size
  std::size_t _bs;

  // quadrature point offsets
  std::vector<std::size_t> _qp_offsets;

  // Basis functions (including first order derivatives) at quadrature
  // points
  std::vector<double> _basis_values;

  // Shape of basis values
  std::array<std::size_t, 4> _basis_shape;

  // Coordinate basis functions (including first order derivatives) at
  // quadrature points
  std::vector<double> _c_basis_values;

  // Shape of coordinate basis values
  std::array<std::size_t, 4> _c_basis_shape;

  // the coefficient offsets
  std::vector<std::size_t> _offsets;

  std::vector<double> _ref_jacobians;

  std::array<std::size_t, 3> _jac_shape;

  std::vector<double> _facet_normals;

  std::array<std::size_t, 2> _normals_shape;

  jac_fn _update_jacobian;

  normal_fn _update_normal;

  std::vector<double> _q_weights;
};
} // namespace dolfinx_contact
