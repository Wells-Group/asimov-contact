// Copyright (C) 2022 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "QuadratureRule.h"
#include "utils.h"
#include <dolfinx.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <xtensor/xadapt.hpp>
using jac_fn = std::function<double(
    std::size_t, double, xt::xtensor<double, 2>&, xt::xtensor<double, 2>&,
    xt::xtensor<double, 2>&, const xt::xtensor<double, 2>&,
    const xt::xtensor<double, 3>&, const xt::xtensor<double, 2>&)>;
using normal_fn
    = std::function<void(xt::xtensor<double, 1>&, const xt::xtensor<double, 2>&,
                         const xt::xtensor<double, 2>&, const std::size_t)>;

namespace dolfinx_contact
{
class KernelData
{
public:
  // kernel data constructor
  // generates data that is common to all contact kernels
  ///@param[in] V The function space
  ///@param[in] q_rule The quadrature rules
  ///@param[in] cstrides The strides for individual coeffcients used in the
  /// kernel
  KernelData(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
             std::shared_ptr<const dolfinx_contact::QuadratureRule> q_rule,
             const std::vector<std::size_t>& cstrides);

  // return geometrical dimension
  std::uint32_t gdim() const { return _gdim; }
  // return topological dimension
  std::uint32_t tdim() const { return _tdim; }
  // return number of dofs for geometry
  int num_coordinate_dofs() const { return _num_coordinate_dofs; }
  // return whether cell geometry is affine
  bool affine() const { return _affine; }
  // return number of dofs pers cell
  std::uint32_t ndofs_cell() const { return _ndofs_cell; }
  // return block size
  std::size_t bs() const { return _bs; }
  // return quadrature rule offsets for index f
  int qp_offsets(int f) const { return _qp_offsets[f]; }
  // return basis functions at quadrature points for facet f
  const xt::xtensor<double, 2>& phi() const { return _phi; }
  // return grad(_phi) at quadrature points for facet f
  const xt::xtensor<double, 3>& dphi() const { return _dphi; }
  // return gradient of coordinate bases at quadrature points for facet f
  const xt::xtensor<double, 3>& dphi_c() const { return _dphi_c; }
  // return coefficient offsets of coefficient i
  std::size_t offsets(const std::size_t i) const { return _offsets[i]; }
  const std::vector<std::size_t>& offsets_array() const { return _offsets; }
  // return reference facet normals
  const xt::xtensor<double, 2>& facet_normals() const { return _facet_normals; }
  /// Compute the following jacobians on a given facet:
  /// J: physical cell -> reference cell (and its inverse)
  /// J_tot: physical facet -> reference facet
  /// @param[in] q - index of quadrature points
  /// @param[in] facet_index - The index of the facet local to the cell
  /// @param[in,out] J - Jacboian between reference cell and physical cell
  /// @param[in,out] K - inverse of J
  /// @param[in,out] J_tot - J_f*J
  /// @param[in] coords - the coordinates of the facet
  /// @return absolute value of determinant of J_tot
  double update_jacobian(std::size_t q, const int facet_index, double detJ,
                         xt::xtensor<double, 2>& J, xt::xtensor<double, 2>& K,
                         xt::xtensor<double, 2>& J_tot,
                         const xt::xtensor<double, 2>& coords) const
  {
    const xt::xtensor<double, 3> dphi_fc = xt::view(
        _dphi_c, xt::all(),
        xt::xrange(_qp_offsets[facet_index], _qp_offsets[facet_index + 1]),
        xt::all());
    xt::xtensor<double, 2> J_f
        = xt::view(_ref_jacobians, facet_index, xt::all(), xt::all());
    return _update_jacobian(q, detJ, J, K, J_tot, J_f, dphi_fc, coords);
  }

  /// Compute the following jacobians on a given facet at first quadrature
  /// point: J: physical cell -> reference cell (and its inverse) J_tot:
  /// physical facet -> reference facet
  /// @param[in] facet_index - The index of the facet local to the cell
  /// @param[in,out] J - Jacboian between reference cell and physical cell
  /// @param[in,out] K - inverse of J
  /// @param[in,out] J_tot - J_f*J
  /// @param[in] coords - the coordinates of the facet
  /// @return absolute value of determinant of J_tot
  double compute_facet_jacobians(const int facet_index,
                                 xt::xtensor<double, 2>& J,
                                 xt::xtensor<double, 2>& K,
                                 xt::xtensor<double, 2>& J_tot,
                                 const xt::xtensor<double, 2>& coords) const
  {
    const xt::xtensor<double, 3> dphi_fc = xt::view(
        _dphi_c, xt::all(),
        xt::xrange(_qp_offsets[facet_index], _qp_offsets[facet_index + 1]),
        xt::all());
    xt::xtensor<double, 2> J_f
        = xt::view(_ref_jacobians, facet_index, xt::all(), xt::all());
    return std::fabs(dolfinx_contact::compute_facet_jacobians(
        0, J, K, J_tot, J_f, dphi_fc, coords));
  }
  // update normal
  /// @param[in, out] n The facet normal
  /// @param[in] K The inverse Jacobian
  /// @param[in] local_index The facet index local to the cell
  void update_normal(xt::xtensor<double, 1>& n, const xt::xtensor<double, 2>& K,
                     const std::size_t local_index) const
  {
    return _update_normal(n, K, _facet_normals, local_index);
  }
  // return quadrature weights for facet f
  const std::vector<double>& q_weights() const { return _q_weights; }

  // return the reference jacobians
  const xt::xtensor<double, 3>& ref_jacobians() const { return _ref_jacobians; }

private:
  std::uint32_t _gdim;               // geometrical dimension
  std::uint32_t _tdim;               // topological dimension
  int _num_coordinate_dofs;          // number of dofs for geometry
  bool _affine;                      // store whether cell geometry is affine
  std::uint32_t _ndofs_cell;         // number of dofs per cell
  std::size_t _bs;                   // block size
  std::vector<int> _qp_offsets;      // quadrature point offsets
  xt::xtensor<double, 2> _phi;       // basis functions at quadrature points
  xt::xtensor<double, 3> _dphi;      // grad(_phi)
  xt::xtensor<double, 3> _dphi_c;    // gradient of coordinate basis
  std::vector<std::size_t> _offsets; // the coefficient offsets
  xt::xtensor<double, 3> _ref_jacobians;
  xt::xtensor<double, 2> _facet_normals;
  jac_fn _update_jacobian;
  normal_fn _update_normal;
  std::vector<double> _q_weights;
};
} // namespace dolfinx_contact
