// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "Contact.h"
#include "KernelData.h"
#include "QuadratureRule.h"
#include "geometric_quantities.h"
#include "utils.h"
#include <basix/cell.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>

namespace dolfinx_contact
{
template <typename T>
kernel_fn<T> generate_contact_kernel(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V, Kernel type,
    QuadratureRule& quadrature_rule,
    std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>>
        coeffs,
    bool constant_normal)
{

  auto mesh = V->mesh();
  assert(mesh);

  // Get mesh info
  const std::uint32_t gdim = mesh->geometry().dim();
  const std::uint32_t tdim = mesh->topology().dim();

  // Create quadrature points on reference facet
  const xt::xtensor<double, 2>& q_points = quadrature_rule.points();
  const std::size_t num_quadrature_pts = quadrature_rule.weights().size();

  // Structures for coefficient data
  int num_coeffs = coeffs.size();
  std::vector<std::size_t> cstrides(num_coeffs + 3);
  for (int i = 0; i < num_coeffs; i++)
  {
    std::shared_ptr<const dolfinx::fem::FiniteElement> coeff_element
        = coeffs[i - 1]->function_space()->element();
    cstrides[i]
        = coeff_element->space_dimension() / coeff_element->block_size();
  }

  // FIXME: This will not work for prism meshes
  const std::vector<std::int32_t>& qp_offsets = quadrature_rule.offset();
  const std::size_t num_qp_per_entity = qp_offsets[1] - qp_offsets[0];
  cstrides[num_coeffs] = 1;                            // h
  cstrides[num_coeffs + 1] = gdim * num_qp_per_entity; // gap
  cstrides[num_coeffs + 2] = gdim * num_qp_per_entity; // normals

  auto kd = dolfinx_contact::KernelData(
      V, std::make_shared<dolfinx_contact::QuadratureRule>(quadrature_rule),
      cstrides);

  // Tabulate basis functions and first derivatives for all input
  // coefficients
  xt::xtensor<double, 2> phi_coeffs(
      {num_quadrature_pts, kd.offsets(num_coeffs)});
  xt::xtensor<double, 3> dphi_coeffs(
      {kd.tdim(), num_quadrature_pts, kd.offsets(num_coeffs)});

  // Create finite elements for coefficient functions and tabulate shape
  // functions
  for (int i = 0; i < num_coeffs; ++i)
  {
    std::shared_ptr<const dolfinx::fem::FiniteElement> coeff_element
        = coeffs[i]->function_space()->element();
    xt::xtensor<double, 4> coeff_basis(
        {kd.tdim() + 1, num_quadrature_pts,
         coeff_element->space_dimension() / coeff_element->block_size(), 1});
    if (coeff_element->value_size() / coeff_element->block_size() != 1)
    {
      throw std::invalid_argument(
          "Kernel does not support coefficients with value size!=1");
    }
    coeff_element->tabulate(coeff_basis, q_points, 1);
    auto phi_i = xt::view(phi_coeffs, xt::all(),
                          xt::range(kd.offsets(i), kd.offsets(i + 1)));
    phi_i = xt::view(coeff_basis, 0, xt::all(), xt::all(), 0);
    auto dphi_i = xt::view(dphi_coeffs, xt::all(), xt::all(),
                           xt::range(kd.offsets(i), kd.offsets(i + 1)));
    dphi_i = xt::view(coeff_basis, xt::range(1, kd.tdim() + 1), xt::all(),
                      xt::all(), 0);
  }

  /// @brief Kernel for contact with rigid surface (RHS).
  ////
  /// The kernel is using Nitsche's method to enforce contact between a
  /// deformable body and a rigid surface.
  ///
  /// @param[in, out] b The local vector to insert values into (List with one
  /// vector)
  /// @param[in] c The coefficients used in kernel. Assumed to be
  /// ordered as u, mu, lmbda, n_surf_x, n_surf_y, n_surf_z
  /// @param[in] w The constants used in kernel. Assumed to be ordered as
  /// gamma, theta, n_surf_x, n_surf_y, n_surf_z if a constant surface used
  /// @param[in] coordinate_dofs The flattened cell geometry, padded to 3D.
  /// @param[in] facet_index The local index (wrt. the cell) of the
  /// facet. Used to access the correct quadrature rule.
  /// @param[in] num_links Unused integer. In two sided contact this indicates
  /// how many cells are connected with the cell.
  dolfinx_contact::kernel_fn<T> nitsche_rigid_rhs
      = [kd, phi_coeffs, constant_normal](
            std::vector<std::vector<T>>& b, const T* c, const T* w,
            const double* coordinate_dofs, const int facet_index,
            [[maybe_unused]] const std::size_t num_links)
  {
    // assumption that the vector function space has block size tdim
    std::array<std::int32_t, 2> q_offset
        = {kd.qp_offsets(facet_index), kd.qp_offsets(facet_index + 1)};

    // Reshape coordinate dofs to two dimensional array
    // FIXME: These array should be views (when compute_jacobian doesn't use
    // xtensor)
    std::array<std::size_t, 2> shape = {kd.num_coordinate_dofs(), 3};
    const xt::xtensor<double, 2>& coord
        = xt::adapt(coordinate_dofs, kd.num_coordinate_dofs() * 3,
                    xt::no_ownership(), shape);
    auto c_view = xt::view(coord, xt::all(), xt::range(0, kd.gdim()));

    // Extract the first derivative of the coordinate element (cell) of
    // degrees of freedom on the facet
    const xt::xtensor<double, 3> dphi_fc
        = xt::view(kd.dphi_c(), xt::all(), xt::xrange(q_offset[0], q_offset[1]),
                   xt::all());

    // Compute Jacobian and determinant at first quadrature point
    xt::xtensor<double, 2> J = xt::zeros<double>({kd.gdim(), kd.tdim()});
    xt::xtensor<double, 2> K = xt::zeros<double>({kd.tdim(), kd.gdim()});
    xt::xtensor<double, 2> J_f
        = xt::view(kd.ref_jacobians(), facet_index, xt::all(), xt::all());
    xt::xtensor<double, 2> J_tot
        = xt::zeros<double>({J.shape(0), J_f.shape(1)});

    double detJ;
    // Normal vector on physical facet at a single quadrature point
    xt::xtensor<double, 1> n_phys = xt::zeros<double>({kd.gdim()});
    // Pre-compute jacobians and normals for affine meshes
    if (kd.affine())
    {
      detJ = std::fabs(dolfinx_contact::compute_facet_jacobians(
          0, J, K, J_tot, J_f, dphi_fc, coord));
      dolfinx_contact::physical_facet_normal(
          n_phys, K, xt::row(kd.facet_normals(), facet_index));
    }

    // Retrieve normal of rigid surface if constant
    std::array<double, 3> n_surf = {0, 0, 0};
    double n_dot = 0;
    if (constant_normal)
    {
      // If surface normal constant precompute (n_phys * n_surf)
      for (int i = 0; i < kd.gdim(); i++)
      {
        // For closest point projection the gap function is given by
        // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
        // in y = Pi(x)
        n_surf[i] = -w[i + 2];
        n_dot += n_phys(i) * n_surf[i];
      }
    }
    int c_offset = (kd.bs() - 1) * kd.offsets(1);
    // This is gamma/h
    double gamma = w[0] / c[c_offset + kd.offsets(3)];
    double gamma_inv = c[c_offset + kd.offsets(3)] / w[0];
    double theta = w[1];
    xtl::span<const double> _weights(kd.q_weights());
    auto weights = _weights.subspan(q_offset[0], q_offset[1] - q_offset[0]);

    // Temporary variable for grad(phi) on physical cell
    xt::xtensor<double, 2> dphi_phys({kd.bs(), kd.ndofs_cell()});

    // Temporary work arrays
    xt::xtensor<double, 2> tr(
        {std::uint32_t(kd.offsets(1) - kd.offsets(0)), kd.gdim()});
    xt::xtensor<double, 2> epsn(
        {std::uint32_t(kd.offsets(1) - kd.offsets(0)), kd.gdim()});

    // Loop over quadrature points
    const int num_points = q_offset[1] - q_offset[0];
    for (std::size_t q = 0; q < num_points; q++)
    {
      const std::size_t q_pos = q_offset[0] + q;

      // Update Jacobian and physical normal
      detJ = std::fabs(
          kd.update_jacobian(q, detJ, J, K, J_tot, J_f, dphi_fc, coord));
      kd.update_normal(n_phys, K, facet_index);

      double mu = 0;
      for (int j = kd.offsets(1); j < kd.offsets(2); j++)
        mu += c[j + c_offset] * phi_coeffs(q_pos, j);
      double lmbda = 0;
      for (int j = kd.offsets(2); j < kd.offsets(3); j++)
        lmbda += c[j + c_offset] * phi_coeffs(q_pos, j);

      // if normal not constant, get surface normal at current quadrature point
      int normal_offset = c_offset + kd.offsets(5);
      if (!constant_normal)
      {
        n_dot = 0;
        for (int i = 0; i < kd.gdim(); i++)
        {
          // For closest point projection the gap function is given by
          // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
          // in y = Pi(x)
          n_surf[i] = -c[normal_offset + q * kd.gdim() + i];
          n_dot += n_phys(i) * n_surf[i];
        }
      }
      int gap_offset = c_offset + kd.offsets(4);
      double gap = 0;
      for (int i = 0; i < kd.gdim(); i++)
      {
        gap += c[gap_offset + q * kd.gdim() + i] * n_surf[i];
      }

      // Extract reference to the tabulated basis function
      const xt::xtensor<double, 2>& phi = kd.phi();
      const xt::xtensor<double, 3>& dphi = kd.dphi();
      // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2
      std::fill(tr.begin(), tr.end(), 0);
      std::fill(epsn.begin(), epsn.end(), 0);
      for (int j = 0; j < kd.offsets(1) - kd.offsets(0); j++)
      {
        for (int l = 0; l < kd.bs(); l++)
        {
          for (int k = 0; k < kd.tdim(); k++)
          {
            tr(j, l) += K(k, l) * dphi(k, q_pos, j);
            for (int s = 0; s < kd.gdim(); s++)
            {
              epsn(j, l) += K(k, s) * dphi(k, q_pos, j)
                            * (n_phys(s) * n_surf[l] + n_phys(l) * n_surf[s]);
            }
          }
        }
      }
      // compute tr(eps(u)), epsn at q
      double tr_u = 0;
      double epsn_u = 0;
      double u_dot_nsurf = 0;
      for (int i = 0; i < kd.offsets(1) - kd.offsets(0); i++)
      {
        const std::int32_t block_index = (i + kd.offsets(0)) * kd.bs();
        for (int j = 0; j < kd.bs(); j++)
        {
          tr_u += c[block_index + j] * tr(i, j);
          epsn_u += c[block_index + j] * epsn(i, j);
          u_dot_nsurf += c[block_index + j] * n_surf[j] * phi(q_pos, i);
        }
      }

      // Multiply  by weight
      double sign_u = (lmbda * n_dot * tr_u + mu * epsn_u);
      double R_minus_scaled
          = dolfinx_contact::R_minus(gamma_inv * sign_u + (gap - u_dot_nsurf))
            * detJ * weights[q];
      sign_u *= detJ * weights[q];
      for (int j = 0; j < kd.ndofs_cell(); j++)
      {
        // Insert over block size in matrix
        for (int l = 0; l < kd.bs(); l++)
        {
          double sign_v = lmbda * tr(j, l) * n_dot + mu * epsn(j, l);
          double v_dot_nsurf = n_surf[l] * phi(q_pos, j);
          b[0][j * kd.bs() + l]
              += -theta * gamma_inv * sign_v * sign_u
                 + R_minus_scaled * (theta * sign_v - gamma * v_dot_nsurf);
        }
      }
    }
  };

  /// @brief Kernel for contact with rigid surface (Jacobian).
  ////
  /// The kernel is using Nitsche's method to enforce contact between a
  /// deformable body and a rigid surface.
  ///
  /// @param[in, out] A The local matrix to insert values into (List with one
  /// matrix)
  /// @param[in] c The coefficients used in kernel. Assumed to be
  /// ordered as u, mu, lmbda, n_surf_x, n_surf_y, n_surf_z
  /// @param[in] w The constants used in kernel. Assumed to be ordered as
  /// gamma, theta, n_surf_x, n_surf_y, n_surf_z if a constant surface used
  /// @param[in] coordinate_dofs The flattened cell geometry, padded to 3D.
  /// @param[in] facet_index The local index (wrt. the cell) of the
  /// facet. Used to access the correct quadrature rule.
  /// @param[in] num_links Unused integer. In two sided contact this indicates
  /// how many cells are connected with the cell.
  kernel_fn<T> nitsche_rigid_jacobian
      = [kd, phi_coeffs, dphi_coeffs, num_coeffs, constant_normal](
            std::vector<std::vector<double>>& A, const T* c, const T* w,
            const double* coordinate_dofs, const int facet_index,
            [[maybe_unused]] const std::size_t num_links)
  {
    const int fdim = kd.tdim() - 1;
    std::array<std::int32_t, 2> q_offset
        = {kd.qp_offsets(facet_index), kd.qp_offsets(facet_index + 1)};

    // Reshape coordinate dofs to two dimensional array
    // FIXME: These array should be views (when compute_jacobian doesn't use
    // xtensor)
    std::array<std::size_t, 2> shape = {kd.num_coordinate_dofs(), 3};
    xt::xtensor<double, 2> coord
        = xt::adapt(coordinate_dofs, kd.num_coordinate_dofs() * 3,
                    xt::no_ownership(), shape);

    // Extract the first derivative of the coordinate element (cell) of
    // degrees of freedom on the facet
    const xt::xtensor<double, 3> dphi_fc
        = xt::view(kd.dphi_c(), xt::all(), xt::xrange(q_offset[0], q_offset[1]),
                   xt::all());
    xt::xtensor<double, 2> J = xt::zeros<double>({kd.gdim(), kd.tdim()});
    xt::xtensor<double, 2> K = xt::zeros<double>({kd.tdim(), kd.gdim()});
    xt::xtensor<double, 1> n_phys = xt::zeros<double>({kd.gdim()});
    xt::xtensor<double, 2> J_f
        = xt::view(kd.ref_jacobians(), facet_index, xt::all(), xt::all());
    xt::xtensor<double, 2> J_tot
        = xt::zeros<double>({J.shape(0), J_f.shape(1)});
    double detJ;
    if (kd.affine())
    {
      detJ = std::fabs(dolfinx_contact::compute_facet_jacobians(
          0, J, K, J_tot, J_f, dphi_fc, coord));
      dolfinx_contact::physical_facet_normal(
          n_phys, K, xt::row(kd.facet_normals(), facet_index));
    }

    // Retrieve normal of rigid surface if constant
    std::array<double, 3> n_surf = {0, 0, 0};
    // FIXME: Code duplication from previous kernel, and should be made into a
    // lambda function
    double n_dot = 0;
    if (constant_normal)
    {
      // If surface normal constant precompute (n_phys * n_surf)
      for (int i = 0; i < kd.gdim(); i++)
      {
        // For closest point projection the gap function is given by
        // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
        // in y = Pi(x)
        n_surf[i] = -w[i + 2];
        n_dot += n_phys(i) * n_surf[i];
      }
    }
    int c_offset = (kd.bs() - 1) * kd.offsets(1);
    double gamma
        = w[0]
          / c[c_offset + kd.offsets(3)]; // This is gamma/hdouble gamma = w[0];
    double gamma_inv = c[c_offset + kd.offsets(3)] / w[0];
    double theta = w[1];

    xtl::span<const double> _weights(kd.q_weights());
    auto weights = _weights.subspan(q_offset[0], q_offset[1] - q_offset[0]);

    // Get number of dofs per cell
    // Temporary variable for grad(phi) on physical cell
    xt::xtensor<double, 2> dphi_phys({kd.bs(), kd.ndofs_cell()});
    xt::xtensor<double, 2> tr = xt::zeros<double>({kd.ndofs_cell(), kd.gdim()});
    xt::xtensor<double, 2> epsn
        = xt::zeros<double>({kd.ndofs_cell(), kd.gdim()});
    const std::uint32_t num_points = q_offset[1] - q_offset[0];
    for (std::size_t q = 0; q < num_points; q++)
    {
      const std::size_t q_pos = q_offset[0] + q;

      // Update Jacobian and physical normal
      detJ = std::fabs(
          kd.update_jacobian(q, detJ, J, K, J_tot, J_f, dphi_fc, coord));
      kd.update_normal(n_phys, K, facet_index);

      // if normal not constant, get surface normal at current quadrature point
      int normal_offset = c_offset + kd.offsets(5);
      if (!constant_normal)
      {
        n_dot = 0;
        for (int i = 0; i < kd.gdim(); i++)
        {
          // For closest point projection the gap function is given by
          // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
          // in y = Pi(x)
          n_surf[i] = -c[normal_offset + q * kd.gdim() + i];
          n_dot += n_phys(i) * n_surf[i];
        }
      }
      int gap_offset = c_offset + kd.offsets(4);
      double gap = 0;
      for (int i = 0; i < kd.gdim(); i++)
        gap += c[gap_offset + q * kd.gdim() + i] * n_surf[i];

      // Extract reference to the tabulated basis function
      const xt::xtensor<double, 2>& phi = kd.phi();
      const xt::xtensor<double, 3>& dphi = kd.dphi();
      // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2
      std::fill(tr.begin(), tr.end(), 0);
      std::fill(epsn.begin(), epsn.end(), 0);
      for (int j = 0; j < kd.ndofs_cell(); j++)
      {
        for (int l = 0; l < kd.bs(); l++)
        {
          for (int k = 0; k < kd.tdim(); k++)
          {
            tr(j, l) += K(k, l) * dphi(k, q_pos, j);
            for (int s = 0; s < kd.gdim(); s++)
            {
              epsn(j, l) += K(k, s) * dphi(k, q_pos, j)
                            * (n_phys(s) * n_surf[l] + n_phys(l) * n_surf[s]);
            }
          }
        }
      }
      double mu = 0;
      int c_offset = (kd.bs() - 1) * kd.offsets(1);
      for (int j = kd.offsets(1); j < kd.offsets(2); j++)
        mu += c[j + c_offset] * phi_coeffs(facet_index, q_pos, j);
      double lmbda = 0;
      for (int j = kd.offsets(2); j < kd.offsets(3); j++)
        lmbda += c[j + c_offset] * phi_coeffs(facet_index, q_pos, j);

      // compute tr(eps(u)), epsn at q
      double tr_u = 0;
      double epsn_u = 0;
      double u_dot_nsurf = 0;
      for (int i = 0; i < kd.offsets(1) - kd.offsets(0); i++)
      {
        const std::int32_t block_index = (i + kd.offsets(0)) * kd.bs();
        for (int j = 0; j < kd.bs(); j++)
        {
          const auto c_val = c[block_index + j];
          tr_u += c_val * tr(i, j);
          epsn_u += c_val * epsn(i, j);
          u_dot_nsurf += c_val * n_surf[j] * phi(q_pos, i);
        }
      }

      double sign_u = lmbda * tr_u * n_dot + mu * epsn_u;
      double Pn_u
          = dolfinx_contact::dR_minus(sign_u + gamma * (gap - u_dot_nsurf));
      const double w0 = weights[q] * detJ;
      for (int j = 0; j < kd.ndofs_cell(); j++)
      {
        for (int l = 0; l < kd.bs(); l++)
        {
          double sign_du = (lmbda * tr(j, l) * n_dot + mu * epsn(j, l));
          double Pn_du
              = (gamma_inv * sign_du - n_surf[l] * phi(q_pos, j)) * Pn_u * w0;
          sign_du *= w0;

          // Insert over block size in matrix
          for (int i = 0; i < kd.ndofs_cell(); i++)
          {
            for (int b = 0; b < kd.bs(); b++)
            {
              double v_dot_nsurf = n_surf[b] * phi(q_pos, i);
              double sign_v = (lmbda * tr(i, b) * n_dot + mu * epsn(i, b));
              A[0]
               [(b + i * kd.bs()) * kd.ndofs_cell() * kd.bs() + l + j * kd.bs()]
                  += -theta * gamma_inv * sign_du * sign_v
                     + Pn_du * (theta * sign_v - gamma * v_dot_nsurf);
            }
          }
        }
      }
    }
  };
  switch (type)
  {
  case dolfinx_contact::Kernel::Rhs:
    return nitsche_rigid_rhs;
  case dolfinx_contact::Kernel::Jac:
    return nitsche_rigid_jacobian;
  default:
    throw std::invalid_argument("Unrecognized kernel");
  }
}
} // namespace dolfinx_contact
