// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "Contact.h"
#include "KernelData.h"
#include "QuadratureRule.h"
#include "elasticity.h"
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
  const std::vector<double>& q_points = quadrature_rule.points();
  assert(quadrature_rule.tdim() == tdim);
  const std::size_t num_quadrature_pts = quadrature_rule.weights().size();

  // Structures for coefficient data
  int num_coeffs = coeffs.size();
  std::vector<std::size_t> cstrides(num_coeffs + 3);
  for (int i = 0; i < num_coeffs; i++)
  {
    std::shared_ptr<const dolfinx::fem::FiniteElement> coeff_element
        = coeffs[i]->function_space()->element();
    cstrides[i]
        = coeff_element->space_dimension() / coeff_element->block_size();
  }

  // FIXME: This will not work for prism meshes
  const std::vector<std::size_t>& qp_offsets = quadrature_rule.offset();
  const std::size_t num_qp_per_entity = qp_offsets[1] - qp_offsets[0];
  cstrides[num_coeffs] = 1;                            // h
  cstrides[num_coeffs + 1] = gdim * num_qp_per_entity; // gap
  cstrides[num_coeffs + 2] = gdim * num_qp_per_entity; // normals

  dolfinx_contact::KernelData kd(
      V, std::make_shared<dolfinx_contact::QuadratureRule>(quadrature_rule),
      cstrides);

  // Tabulate basis functions for all input
  // coefficients
  std::vector<double> coeff_basis_valuesb(num_quadrature_pts
                                          * kd.offsets(num_coeffs));
  const std::array<std::size_t, 2> coeff_shape
      = {num_quadrature_pts, kd.offsets(num_coeffs)};
  mdspan2_t coeff_basis_valuesm(coeff_basis_valuesb.data(), coeff_shape);

  // Create finite elements for coefficient functions and tabulate shape
  // functions
  for (int i = 0; i < num_coeffs; ++i)
  {
    std::shared_ptr<const dolfinx::fem::FiniteElement> coeff_element
        = coeffs[i]->function_space()->element();
    assert(coeff_element);
    const std::array<std::size_t, 4> coeff_shape
        = coeff_element->basix_element().tabulate_shape(1, num_quadrature_pts);
    if (coeff_shape.back() != 1)
    {
      throw std::invalid_argument(
          "Kernel does not support coefficients with value size!=1");
    }
    std::vector<double> coeff_basisb(std::reduce(
        coeff_shape.cbegin(), coeff_shape.cend(), 1, std::multiplies()));
    coeff_element->tabulate(coeff_basisb, q_points, {num_quadrature_pts, tdim},
                            1);
    cmdspan4_t coeff_basis(coeff_basisb.data(), coeff_shape);
    auto basis = stdex::submdspan(coeff_basis_valuesm, stdex::full_extent,
                                  std::pair{kd.offsets(i), kd.offsets(i + 1)});
    assert(kd.offsets(i + 1) - kd.offsets(i) == coeff_shape[2]);
    for (std::size_t k = 0; k < basis.extent(1); ++k)
      for (std::size_t l = 0; l < basis.extent(2); ++l)
        basis(k, l) = coeff_basis(0, k, l, 0);
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
      = [kd, gdim, tdim, coeff_basis_valuesb, coeff_shape, constant_normal](
            std::vector<std::vector<T>>& b, std::span<const T> c, const T* w,
            const double* coordinate_dofs, const int facet_index,
            [[maybe_unused]] const std::size_t num_links)
  {
    // Retrieve some data form kd
    const std::array<std::int32_t, 2> q_offset
        = {kd.qp_offsets(facet_index), kd.qp_offsets(facet_index + 1)};
    const std::size_t bs = kd.bs();
    const std::uint32_t ndofs_cell = kd.ndofs_cell();

    // Reshape coordinate dofs to two dimensional array
    cmdspan2_t coord(coordinate_dofs, kd.num_coordinate_dofs(), 3);
    // Compute Jacobian and determinant at first quadrature point
    std::array<double, 9> Jb;
    mdspan2_t J(Jb.data(), gdim, tdim);
    std::array<double, 9> Kb;
    mdspan2_t K(Kb.data(), tdim, gdim);
    std::array<double, 6> J_totb;
    mdspan2_t J_tot(J_totb.data(), gdim, tdim - 1);
    double detJ = 0;
    std::array<double, 18> detJ_scratch;

    // Normal vector on physical facet at a single quadrature point
    std::array<double, 3> n_phys;
    std::cout << kd.affine() << "\n";
    // Pre-compute jacobians and normals for affine meshes
    if (kd.affine())
    {
      std::cout << "Pref facet jacs\n";
      detJ = kd.compute_first_facet_jacobian(facet_index, J, K, J_tot,
                                             detJ_scratch, coord);
      std::cout << "Physical facet\n";
      dolfinx_contact::physical_facet_normal(
          std::span(n_phys.data(), gdim), K,
          stdex::submdspan(kd.facet_normals(), facet_index,
                           stdex::full_extent));
    }
    std::cout << "post physical normal\n";
    // Retrieve normal of rigid surface if constant
    std::array<double, 3> n_surf = {0, 0, 0};
    double n_dot = 0;
    if (constant_normal)
    {
      // If surface normal constant precompute (n_phys * n_surf)
      for (int i = 0; i < gdim; i++)
      {
        // For closest point projection the gap function is given by
        // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
        // in y = Pi(x)
        n_surf[i] = -w[i + 2];
        n_dot += n_phys[i] * n_surf[i];
      }
    }
    int c_offset = (bs - 1) * kd.offsets(1);
    // This is gamma/h
    double gamma = w[0] / c[c_offset + kd.offsets(3)];
    double gamma_inv = c[c_offset + kd.offsets(3)] / w[0];
    double theta = w[1];
    std::span<const double> _weights(kd.q_weights());
    auto weights = _weights.subspan(q_offset[0], q_offset[1] - q_offset[0]);

    // Temporary variable for grad(phi) on physical cell
    xt::xtensor<double, 2> dphi_phys({bs, ndofs_cell});

    // Temporary work arrays
    std::vector<double> epsnb((kd.offsets(1) - kd.offsets(0)) * gdim);
    mdspan2_t epsn(epsnb.data(), kd.offsets(1) - kd.offsets(0), gdim);
    std::vector<double> trb((kd.offsets(1) - kd.offsets(0)) * gdim);
    mdspan2_t tr(trb.data(), kd.offsets(1) - kd.offsets(0), gdim);

    // Extract basis for coefficients
    cmdspan2_t phi_coeffs(coeff_basis_valuesb.data(), coeff_shape);

    // Extract reference to the tabulated basis function
    cmdspan2_t phi = kd.phi();
    cmdspan3_t dphi = kd.dphi();
    // Loop over quadrature points
    const int num_points = q_offset[1] - q_offset[0];
    for (std::size_t q = 0; q < num_points; q++)
    {
      const std::size_t q_pos = q_offset[0] + q;

      // Update Jacobian and physical normal
      detJ = std::fabs(kd.update_jacobian(q, facet_index, detJ, J, K, J_tot,
                                          detJ_scratch, coord));
      kd.update_normal(std::span(n_phys.data(), gdim), K, facet_index);

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
        for (int i = 0; i < gdim; i++)
        {
          // For closest point projection the gap function is given by
          // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
          // in y = Pi(x)
          n_surf[i] = -c[normal_offset + q * gdim + i];
          n_dot += n_phys[i] * n_surf[i];
        }
      }
      int gap_offset = c_offset + kd.offsets(4);
      double gap = 0;
      for (int i = 0; i < gdim; i++)
      {
        gap += c[gap_offset + q * gdim + i] * n_surf[i];
      }

      compute_normal_strain_basis(epsn, tr, K, dphi, n_surf,
                                  std::span(n_phys.data(), gdim), q_pos);

      // compute tr(eps(u)), epsn at q
      double tr_u = 0;
      double epsn_u = 0;
      double u_dot_nsurf = 0;
      for (int i = 0; i < kd.offsets(1) - kd.offsets(0); i++)
      {
        const std::int32_t block_index = (i + kd.offsets(0)) * bs;
        for (int j = 0; j < bs; j++)
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
      for (int j = 0; j < ndofs_cell; j++)
      {
        // Insert over block size in matrix
        for (int l = 0; l < bs; l++)
        {
          double sign_v = lmbda * tr(j, l) * n_dot + mu * epsn(j, l);
          double v_dot_nsurf = n_surf[l] * phi(q_pos, j);
          b[0][j * bs + l]
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
      = [kd, gdim, tdim, coeff_basis_valuesb, coeff_shape, num_coeffs,
         constant_normal](std::vector<std::vector<double>>& A,
                          std::span<const T> c, const T* w,
                          const double* coordinate_dofs, const int facet_index,
                          [[maybe_unused]] const std::size_t num_links)
  {
    // Retrieve some data from kd
    std::array<std::int32_t, 2> q_offset
        = {kd.qp_offsets(facet_index), kd.qp_offsets(facet_index + 1)};
    const std::size_t bs = kd.bs();
    const std::uint32_t ndofs_cell = kd.ndofs_cell();
    const int fdim = tdim - 1;

    // Reshape coordinate dofs to two-dimensional array
    cmdspan2_t coord(coordinate_dofs, kd.num_coordinate_dofs(), 3);

    // Compute Jacobian and determinant at first quadrature point
    std::array<double, 9> Jb;
    mdspan2_t J(Jb.data(), gdim, tdim);
    std::array<double, 9> Kb;
    mdspan2_t K(Kb.data(), tdim, gdim);
    std::array<double, 6> J_totb;
    mdspan2_t J_tot(J_totb.data(), gdim, tdim - 1);
    double detJ = 0;
    std::array<double, 18> detJ_scratch;

    // Normal vector on physical facet at a single quadrature point
    std::array<double, 3> n_phys;

    // Pre-compute jacobians and normals for affine meshes
    if (kd.affine())
    {
      detJ = kd.compute_first_facet_jacobian(facet_index, J, K, J_tot,
                                             detJ_scratch, coord);

      dolfinx_contact::physical_facet_normal(
          std::span(n_phys.data(), gdim), K,
          stdex::submdspan(kd.facet_normals(), facet_index,
                           stdex::full_extent));
    }

    // Retrieve normal of rigid surface if constant
    std::array<double, 3> n_surf = {0, 0, 0};

    // FIXME: Code duplication from previous kernel, and should be made into a
    // lambda function
    double n_dot = 0;
    if (constant_normal)
    {
      // If surface normal constant precompute (n_phys * n_surf)
      for (int i = 0; i < gdim; i++)
      {
        // For closest point projection the gap function is given by
        // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
        // in y = Pi(x)
        n_surf[i] = -w[i + 2];
        n_dot += n_phys[i] * n_surf[i];
      }
    }
    int c_offset = (bs - 1) * kd.offsets(1);
    double gamma
        = w[0]
          / c[c_offset + kd.offsets(3)]; // This is gamma/hdouble gamma = w[0];
    double gamma_inv = c[c_offset + kd.offsets(3)] / w[0];
    double theta = w[1];

    std::span<const double> _weights(kd.q_weights());
    auto weights = _weights.subspan(q_offset[0], q_offset[1] - q_offset[0]);

    // Extract reference to the tabulated basis function
    cmdspan2_t phi = kd.phi();
    cmdspan3_t dphi = kd.dphi();

    // Get number of dofs per cell
    // Temporary variable for grad(phi) on physical cell
    std::vector<double> dphi_physb(bs * ndofs_cell);
    mdspan2_t dphi_phys(dphi_physb.data(), bs, ndofs_cell);
    // Temporary work arrays
    std::vector<double> epsnb(ndofs_cell * gdim);
    mdspan2_t epsn(epsnb.data(), ndofs_cell, gdim);
    std::vector<double> trb(ndofs_cell * gdim);
    mdspan2_t tr(trb.data(), ndofs_cell, gdim);

    cmdspan2_t phi_coeffs(coeff_basis_valuesb.data(), coeff_shape);
    const std::uint32_t num_points = q_offset[1] - q_offset[0];
    for (std::size_t q = 0; q < num_points; q++)
    {
      const std::size_t q_pos = q_offset[0] + q;

      // Update Jacobian and physical normal
      detJ = std::fabs(kd.update_jacobian(q, facet_index, detJ, J, K, J_tot,
                                          detJ_scratch, coord));
      kd.update_normal(n_phys, K, facet_index);

      // if normal not constant, get surface normal at current quadrature point
      int normal_offset = c_offset + kd.offsets(5);
      if (!constant_normal)
      {
        n_dot = 0;
        for (int i = 0; i < gdim; i++)
        {
          // For closest point projection the gap function is given by
          // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
          // in y = Pi(x)
          n_surf[i] = -c[normal_offset + q * gdim + i];
          n_dot += n_phys[i] * n_surf[i];
        }
      }
      int gap_offset = c_offset + kd.offsets(4);
      double gap = 0;
      for (int i = 0; i < gdim; i++)
        gap += c[gap_offset + q * gdim + i] * n_surf[i];

      compute_normal_strain_basis(epsn, tr, K, dphi, n_surf,
                                  std::span(n_phys.data(), gdim), q_pos);

      double mu = 0;
      int c_offset = (bs - 1) * kd.offsets(1);
      for (int j = kd.offsets(1); j < kd.offsets(2); j++)
        mu += c[j + c_offset] * phi_coeffs(q_pos, j);
      double lmbda = 0;
      for (int j = kd.offsets(2); j < kd.offsets(3); j++)
        lmbda += c[j + c_offset] * phi_coeffs(q_pos, j);

      // compute tr(eps(u)), epsn at q
      double tr_u = 0;
      double epsn_u = 0;
      double u_dot_nsurf = 0;
      for (int i = 0; i < kd.offsets(1) - kd.offsets(0); i++)
      {
        const std::int32_t block_index = (i + kd.offsets(0)) * bs;
        for (int j = 0; j < bs; j++)
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
      for (int j = 0; j < ndofs_cell; j++)
      {
        for (int l = 0; l < bs; l++)
        {
          double sign_du = (lmbda * tr(j, l) * n_dot + mu * epsn(j, l));
          double Pn_du
              = (gamma_inv * sign_du - n_surf[l] * phi(q_pos, j)) * Pn_u * w0;
          sign_du *= w0;

          // Insert over block size in matrix
          for (int i = 0; i < ndofs_cell; i++)
          {
            for (int b = 0; b < bs; b++)
            {
              double v_dot_nsurf = n_surf[b] * phi(q_pos, i);
              double sign_v = (lmbda * tr(i, b) * n_dot + mu * epsn(i, b));
              A[0][(b + i * bs) * ndofs_cell * bs + l + j * bs]
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
