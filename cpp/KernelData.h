// Copyright (C) 2022 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#pragma once

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
  // empty constructor
  KernelData() = default;

  // kernel datea constructor
  // generates data that is common to all contact kernels
  ///@param[in]
  KernelData(std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
             const std::vector<xt::xarray<double>>& q_points,
             const std::vector<std::vector<double>>& q_weights,
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
  // return number of quadrature points
  std::size_t num_q_points() const { return _num_q_points; }
  // return basis functions at quadrature points for facet f
  const xt::xtensor<double, 2>& phi(const int f) const { return _phi[f]; }
  // return grad(_phi) at quadrature points for facet f
  const xt::xtensor<double, 3>& dphi(const int f) const { return _dphi[f]; }
  // return gradient of coordinate bases at quadrature points for facet f
  const xt::xtensor<double, 3>& dphi_c(const int f) const { return _dphi_c[f]; }
  // return coefficient offsets of coefficient i
  std::size_t offsets(const std::size_t i) const { return _offsets[i]; }
  // return reference facet normals
  const xt::xtensor<double, 2>& facet_normals() const { return _facet_normals; }
  // update jacobian
  double update_jacobian(std::size_t q, double detJ, xt::xtensor<double, 2>& J,
                         xt::xtensor<double, 2>& K,
                         xt::xtensor<double, 2>& J_tot,
                         const xt::xtensor<double, 2>& J_f,
                         const xt::xtensor<double, 3>& dphi,
                         const xt::xtensor<double, 2>& coords) const
  {
    return _update_jacobian(q, detJ, J, K, J_tot, J_f, dphi, coords);
  }
  // update normal
  void update_normal(xt::xtensor<double, 1>& n, const xt::xtensor<double, 2>& K,
                     const std::size_t local_index) const
  {
    return _update_normal(n, K, _facet_normals, local_index);
  }
  // return quadrature weights for facet f
  const std::vector<double>& q_weights(const int f) const
  {
    return _q_weights[f];
  }

  const xt::xtensor<double, 3>& ref_jacobians() const { return _ref_jacobians; }

private:
  std::uint32_t _gdim;       // geometrical dimension
  std::uint32_t _tdim;       // topological dimension
  int _num_coordinate_dofs;  // number of dofs for geometry
  bool _affine;              // store whether cell geometry is affine
  std::uint32_t _ndofs_cell; // number of dofs per cell
  std::size_t _bs;           // block size
  std::size_t _num_q_points; // number of quadrature points
  std::vector<xt::xtensor<double, 2>>
      _phi; // basis functions at quadrature points
  std::vector<xt::xtensor<double, 3>> _dphi;   // grad(_phi)
  std::vector<xt::xtensor<double, 3>> _dphi_c; // gradient of coordinate basis
  std::vector<std::size_t> _offsets;           // the coefficient offsets
  xt::xtensor<double, 3> _ref_jacobians;
  xt::xtensor<double, 2> _facet_normals;
  jac_fn _update_jacobian;
  normal_fn _update_normal;
  std::vector<std::vector<double>> _q_weights;
};
} // namespace dolfinx_contact

// nitsche_rigid_jac but not nitsche_rigid_rhs [dphi_coeffs, num_coeffs, fdim]
// nitsche_rigid_rhs but not nitsche_unbiased_rhs [phi_coeffs, constant_normal,
// qp_offsets]
// nitsche_unbiased_rhs but not nitsche_rigid_rhs [ num_q_points]
// ///[dphi_c, phi, dphi, offsets, gdim, tdim, q_weights,
//            num_coordinate_dofs, ref_jacobians, bs, facet_normals, affine,
//            update_jacobian, update_normal, ndofs_cell, num_q_points]
