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
using jac_fn = std::function<double(double, mdspan2_t, mdspan2_t, mdspan2_t,
                                    std::span<double>, cmdspan2_t, s_cmdspan2_t,
                                    cmdspan2_t)>;

using normal_fn = std::function<void(std::span<double>, cmdspan2_t, cmdspan2_t,
                                     const std::size_t)>;

class KernelData
{
public:
  // Kernel data constructor
  // Generates data that is common to all contact kernels
  ///@param[in] V The function space
  ///@param[in] q_rule The quadrature rules
  ///@param[in] cstrides The strides for individual coeffcients used in the
  /// kernel
  KernelData(std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
             std::shared_ptr<const dolfinx_contact::QuadratureRule> q_rule,
             const std::vector<std::size_t>& cstrides);

  // Return geometrical dimension
  std::uint32_t gdim() const { return _gdim; }

  // Return topological dimension
  std::uint32_t tdim() const { return _tdim; }

  // Return number of dofs for geometry
  int num_coordinate_dofs() const { return _num_coordinate_dofs; }

  // return whether cell geometry is affine
  bool affine() const { return _affine; }

  // return number of dofs pers cell
  std::size_t ndofs_cell() const { return _ndofs_cell; }

  // Return block size
  std::size_t bs() const { return _bs; }

  // Return quadrature rule offsets for index f
  std::size_t qp_offsets(std::size_t f) const
  {
    assert(f < _qp_offsets.size());
    return _qp_offsets[f];
  }

  // Return basis functions at quadrature points for facet f
  s_cmdspan2_t phi() const
  {
    cmdspan4_t full_basis(_basis_values.data(), _basis_shape);
    return stdex::submdspan(full_basis, 0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
                            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  }

  // Return grad(_phi) at quadrature points for facet f
  s_cmdspan3_t dphi() const
  {
    cmdspan4_t full_basis(_basis_values.data(), _basis_shape);
    return stdex::submdspan(full_basis, std::pair{1, (std::size_t)_tdim + 1},
                            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  }

  // Return gradient of coordinate bases at quadrature points for facet f
  cmdspan3_t dphi_c() const
  {
    cmdspan4_t full_basis(_c_basis_values.data(), _c_basis_shape);
    return stdex::submdspan(full_basis, std::pair{1, (std::size_t)_tdim + 1},
                            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  }

  // Return coefficient offsets of coefficient i
  std::size_t offsets(const std::size_t i) const { return _offsets[i]; }

  const std::vector<std::size_t>& offsets_array() const { return _offsets; }

  // Return reference facet normals
  cmdspan2_t facet_normals() const
  {
    return cmdspan2_t(_facet_normals.data(), _normals_shape);
  }

  /// Compute the following jacobians on a given facet:
  /// J: physical cell -> reference cell (and its inverse)
  /// J_tot: physical facet -> reference facet
  /// @param[in] q - index of quadrature points
  /// @param[in] facet_index - The index of the facet local to the cell
  /// @param[in,out] J - Jacobian between reference cell and physical cell
  /// @param[in,out] K - inverse of J
  /// @param[in,out] J_tot - J_f*J
  /// @param[in, out] detJ_scratch - Working memory to compute determinants
  /// @param[in] coords - the coordinates of the facet
  /// @return absolute value of determinant of J_tot
  double update_jacobian(std::size_t q, const std::size_t facet_index,
                         double detJ, mdspan2_t J, mdspan2_t K, mdspan2_t J_tot,
                         std::span<double> detJ_scratch,
                         cmdspan2_t coords) const
  {
    cmdspan4_t full_basis(_c_basis_values.data(), _c_basis_shape);
    const std::size_t q_pos = _qp_offsets[facet_index] + q;
    auto dphi_fc
        = stdex::submdspan(full_basis, std::pair{1, (std::size_t)_tdim + 1},
                           q_pos, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
    cmdspan3_t ref_jacs(_ref_jacobians.data(), _jac_shape);
    auto J_f = stdex::submdspan(ref_jacs, (std::size_t)facet_index,
                                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    return _update_jacobian(detJ, J, K, J_tot, detJ_scratch, J_f, dphi_fc,
                            coords);
  }

  /// Compute the following jacobians on a given facet at first quadrature
  /// point: J: physical cell -> reference cell (and its inverse) J_tot:
  /// physical facet -> reference facet
  /// @param[in] facet_index - The index of the facet local to the cell
  /// @param[in,out] J - Jacobian between reference cell and physical cell
  /// @param[in,out] K - inverse of J
  /// @param[in,out] J_tot - J_f*J
  /// @param[in,out] detJ_scratch - Working memory, min size (2*gdim*tdim)
  /// @param[in] coords - the coordinates of the facet
  /// @return absolute value of determinant of J_tot
  double compute_first_facet_jacobian(const std::size_t facet_index,
                                      mdspan2_t J, mdspan2_t K, mdspan2_t J_tot,
                                      std::span<double> detJ_scratch,
                                      cmdspan2_t coords) const;

  /// update normal
  /// @param[in, out] n The facet normal
  /// @param[in] K The inverse Jacobian
  /// @param[in] local_index The facet index local to the cell
  void update_normal(std::span<double> n, cmdspan2_t K,
                     const std::size_t local_index) const;

  /// Return quadrature weights for the i-th facet
  std::span<const double> weights(std::size_t i) const;

  // return the reference jacobians
  cmdspan3_t ref_jacobians() const;

private:
  std::uint32_t _gdim;                  // geometrical dimension
  std::uint32_t _tdim;                  // topological dimension
  int _num_coordinate_dofs;             // number of dofs for geometry
  bool _affine;                         // store whether cell geometry is affine
  std::uint32_t _ndofs_cell;            // number of dofs per cell
  std::size_t _bs;                      // block size
  std::vector<std::size_t> _qp_offsets; // quadrature point offsets
  std::vector<double> _basis_values; // Basis functions (including first order
                                     // derivatives) at quadrature points
  std::array<std::size_t, 4> _basis_shape; // Shape of basis values
  std::vector<double>
      _c_basis_values; // Coordiante basis functions (including first order
                       // derivatives) at quadrature points
  std::array<std::size_t, 4> _c_basis_shape; // Shape of coordinate basis values
  std::vector<std::size_t> _offsets;         // the coefficient offsets
  std::vector<double> _ref_jacobians;
  std::array<std::size_t, 3> _jac_shape;
  std::vector<double> _facet_normals;
  std::array<std::size_t, 2> _normals_shape;
  jac_fn _update_jacobian;
  normal_fn _update_normal;
  std::vector<double> _q_weights;
};
} // namespace dolfinx_contact
