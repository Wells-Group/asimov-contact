// Copyright (C) 2023 Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "rigid_surface_kernels.h"
#include "Contact.h"
#include "KernelData.h"
#include "elasticity.h"
#include "geometric_quantities.h"
#include "utils.h"
#include <basix/cell.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>

dolfinx_contact::kernel_fn<PetscScalar>
dolfinx_contact::generate_rigid_surface_kernel(
    std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
    dolfinx_contact::Kernel type,
    dolfinx_contact::QuadratureRule& quadrature_rule, bool constant_normal)
{

  auto mesh = V->mesh();
  assert(mesh);

  // Get mesh info
  const std::size_t gdim = mesh->geometry().dim();
  const std::size_t tdim = mesh->topology()->dim();

  // Structures for coefficient data
  // FIXME: This will not work for prism meshes
  const std::vector<std::size_t>& qp_offsets = quadrature_rule.offset();
  const std::size_t num_qp_per_entity = qp_offsets[1] - qp_offsets[0];
  // Coefficient sizes
  // Expecting coefficients in following order:
  // mu, lmbda, h, gap, u, grad(u), normals
  std::vector<std::size_t> cstrides = {1,
                                       1,
                                       1,
                                       gdim * num_qp_per_entity,
                                       gdim * num_qp_per_entity,
                                       gdim * gdim * num_qp_per_entity,
                                       gdim * num_qp_per_entity};

  dolfinx_contact::KernelData kd(
      V, std::make_shared<dolfinx_contact::QuadratureRule>(quadrature_rule),
      cstrides);

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
  /// @param[in] q_indices Unused indices. In two sided contact this yields what
  /// quadrature points to add contributions from
  dolfinx_contact::kernel_fn<PetscScalar> nitsche_rigid_rhs
      = [kd, gdim, tdim, constant_normal](
            std::vector<std::vector<PetscScalar>>& b,
            std::span<const PetscScalar> c, const PetscScalar* w,
            const double* coordinate_dofs, const std::size_t facet_index,
            [[maybe_unused]] const std::size_t num_links,
            [[maybe_unused]] std::span<const std::int32_t> q_indices)
  {
    // Retrieve some data from kd
    const std::size_t bs = kd.bs();
    const std::size_t ndofs_cell = kd.ndofs_cell();

    // Reshape coordinate dofs to two-dimensional array
    dolfinx_contact::cmdspan2_t coord(coordinate_dofs, kd.num_coordinate_dofs(),
                                      3);

    // Compute Jacobian and determinant at first quadrature point
    std::array<double, 9> Jb;
    dolfinx_contact::mdspan2_t J(Jb.data(), gdim, tdim);
    std::array<double, 9> Kb;
    dolfinx_contact::mdspan2_t K(Kb.data(), tdim, gdim);
    std::array<double, 6> J_totb;
    dolfinx_contact::mdspan2_t J_tot(J_totb.data(), gdim, tdim - 1);
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
          MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              kd.facet_normals(), facet_index,
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent));
    }
    // Retrieve normal of rigid surface if constant
    std::array<double, 3> n_surf = {0, 0, 0};
    double n_dot = 0;
    if (constant_normal)
    {
      // If surface normal constant precompute (n_phys * n_surf)
      for (std::size_t i = 0; i < gdim; i++)
      {
        // For closest point projection the gap function is given by
        // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
        // in y = Pi(x)
        n_surf[i] = -w[i + 2];
        n_dot += n_phys[i] * n_surf[i];
      }
    }
    // This is gamma/h
    double gamma = w[0] / c[2];
    double gamma_inv = c[2] / w[0];
    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];
    std::span<const double> weights = kd.weights(facet_index);

    // Temporary work arrays
    std::vector<double> epsnb(ndofs_cell * gdim);
    dolfinx_contact::mdspan2_t epsn(epsnb.data(), ndofs_cell, gdim);
    std::vector<double> trb(ndofs_cell * gdim);
    dolfinx_contact::mdspan2_t tr(trb.data(), ndofs_cell, gdim);
    std::vector<double> sig_n_u(gdim);

    // Extract reference to the tabulated basis function
    dolfinx_contact::s_cmdspan2_t phi = kd.phi();
    dolfinx_contact::s_cmdspan3_t dphi = kd.dphi();

    // Loop over quadrature points
    const std::array<std::size_t, 2> q_offset
        = {kd.qp_offsets(facet_index), kd.qp_offsets(facet_index + 1)};
    const std::size_t num_points = q_offset.back() - q_offset.front();
    for (std::size_t q = 0; q < num_points; q++)
    {
      const std::size_t q_pos = q_offset.front() + q;

      // Update Jacobian and physical normal
      detJ = std::fabs(kd.update_jacobian(q, facet_index, detJ, J, K, J_tot,
                                          detJ_scratch, coord));
      kd.update_normal(std::span(n_phys.data(), gdim), K, facet_index);

      // if normal not constant, get surface normal at current quadrature point
      std::size_t normal_offset = kd.offsets(6);
      if (!constant_normal)
      {
        n_dot = 0;
        for (std::size_t i = 0; i < gdim; i++)
        {
          // For closest point projection the gap function is given by
          // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
          // in y = Pi(x)
          n_surf[i] = -c[normal_offset + q * gdim + i];
          n_dot += n_phys[i] * n_surf[i];
        }
      }
      std::size_t gap_offset = kd.offsets(3);
      double gap = 0;
      for (std::size_t i = 0; i < gdim; i++)
        gap += c[gap_offset + q * gdim + i] * n_surf[i];

      compute_normal_strain_basis(epsn, tr, K, dphi, n_surf,
                                  std::span(n_phys.data(), gdim), q_pos);

      // compute sig(u)*n_phys
      std::fill(sig_n_u.begin(), sig_n_u.end(), 0.0);
      compute_sigma_n_u(sig_n_u,
                        c.subspan(kd.offsets(5) + q * gdim * gdim, gdim * gdim),
                        std::span(n_phys.data(), gdim), mu, lmbda);
      double sign_u = 0;
      double u_dot_nsurf = 0;
      // compute inner(sig(u)*n_phys, n_surf) and inner(u, n_surf)
      for (std::size_t j = 0; j < gdim; ++j)
      {
        sign_u += sig_n_u[j] * n_surf[j];
        u_dot_nsurf += c[kd.offsets(4) + gdim * q + j] * n_surf[j];
      }
      double R_minus_scaled
          = dolfinx_contact::R_minus(gamma_inv * sign_u + (gap - u_dot_nsurf))
            * detJ * weights[q];
      sign_u *= detJ * weights[q];
      for (std::size_t j = 0; j < ndofs_cell; j++)
      {
        // Insert over block size in matrix
        for (std::size_t l = 0; l < bs; l++)
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
  /// @param[in] q_indices Unused indices. In two sided contact this yields what
  /// quadrature points to add contributions from
  dolfinx_contact::kernel_fn<PetscScalar> nitsche_rigid_jacobian
      = [kd, gdim, tdim, constant_normal](
            std::vector<std::vector<double>>& A, std::span<const PetscScalar> c,
            const PetscScalar* w, const double* coordinate_dofs,
            const std::size_t facet_index,
            [[maybe_unused]] const std::size_t num_links,
            [[maybe_unused]] std::span<const std::int32_t> q_indices)
  {
    // Retrieve some data from kd
    const std::size_t bs = kd.bs();
    const std::uint32_t ndofs_cell = kd.ndofs_cell();

    // Reshape coordinate dofs to two-dimensional array
    dolfinx_contact::cmdspan2_t coord(coordinate_dofs, kd.num_coordinate_dofs(),
                                      3);

    // Compute Jacobian and determinant at first quadrature point
    std::array<double, 9> Jb;
    dolfinx_contact::mdspan2_t J(Jb.data(), gdim, tdim);
    std::array<double, 9> Kb;
    dolfinx_contact::mdspan2_t K(Kb.data(), tdim, gdim);
    std::array<double, 6> J_totb;
    dolfinx_contact::mdspan2_t J_tot(J_totb.data(), gdim, tdim - 1);
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
          MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              kd.facet_normals(), facet_index,
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent));
    }

    // Retrieve normal of rigid surface if constant
    std::array<double, 3> n_surf = {0, 0, 0};
    double n_dot = 0;
    if (constant_normal)
    {
      // If surface normal constant precompute (n_phys * n_surf)
      for (std::size_t i = 0; i < gdim; i++)
      {
        // For closest point projection the gap function is given by
        // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
        // in y = Pi(x)
        n_surf[i] = -w[i + 2];
        n_dot += n_phys[i] * n_surf[i];
      }
    }

    // This is gamma/h
    double gamma = w[0] / c[2];
    double gamma_inv = c[2] / w[0];
    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];
    std::span<const double> weights = kd.weights(facet_index);

    // Temporary work arrays
    std::vector<double> epsnb(ndofs_cell * gdim);
    dolfinx_contact::mdspan2_t epsn(epsnb.data(), ndofs_cell, gdim);
    std::vector<double> trb(ndofs_cell * gdim);
    dolfinx_contact::mdspan2_t tr(trb.data(), ndofs_cell, gdim);
    std::vector<double> sig_n_u(gdim);

    // Extract reference to the tabulated basis function
    dolfinx_contact::s_cmdspan2_t phi = kd.phi();
    dolfinx_contact::s_cmdspan3_t dphi = kd.dphi();

    // Loop over quadrature points
    const std::array<std::size_t, 2> q_offset
        = {kd.qp_offsets(facet_index), kd.qp_offsets(facet_index + 1)};
    const std::size_t num_points = q_offset.back() - q_offset.front();

    for (std::size_t q = 0; q < num_points; q++)
    {
      const std::size_t q_pos = q_offset.front() + q;

      // Update Jacobian and physical normal
      detJ = std::fabs(kd.update_jacobian(q, facet_index, detJ, J, K, J_tot,
                                          detJ_scratch, coord));
      kd.update_normal(std::span(n_phys.data(), gdim), K, facet_index);

      // if normal not constant, get surface normal at current quadrature point
      std::size_t normal_offset = kd.offsets(6);
      if (!constant_normal)
      {
        n_dot = 0;
        for (std::size_t i = 0; i < gdim; i++)
        {
          // For closest point projection the gap function is given by
          // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
          // in y = Pi(x)
          n_surf[i] = -c[normal_offset + q * gdim + i];
          n_dot += n_phys[i] * n_surf[i];
        }
      }
      std::size_t gap_offset = kd.offsets(3);
      double gap = 0;
      for (std::size_t i = 0; i < gdim; i++)
        gap += c[gap_offset + q * gdim + i] * n_surf[i];

      compute_normal_strain_basis(epsn, tr, K, dphi, n_surf,
                                  std::span(n_phys.data(), gdim), q_pos);

      // compute sig(u)*n_phys
      std::fill(sig_n_u.begin(), sig_n_u.end(), 0.0);
      compute_sigma_n_u(sig_n_u,
                        c.subspan(kd.offsets(5) + q * gdim * gdim, gdim * gdim),
                        std::span(n_phys.data(), gdim), mu, lmbda);

      // compute inner(sig(u)*n_phys, n_surf) and inner(u, n_surf)
      double sign_u = 0;
      double u_dot_nsurf = 0;
      for (std::size_t j = 0; j < gdim; ++j)
      {
        sign_u += sig_n_u[j] * n_surf[j];
        u_dot_nsurf += c[kd.offsets(4) + gdim * q + j] * n_surf[j];
      }
      double Pn_u
          = dolfinx_contact::dR_minus(sign_u + gamma * (gap - u_dot_nsurf));
      const double w0 = weights[q] * detJ;
      for (std::size_t j = 0; j < ndofs_cell; j++)
      {
        for (std::size_t l = 0; l < bs; l++)
        {
          double sign_du = (lmbda * tr(j, l) * n_dot + mu * epsn(j, l));
          double Pn_du
              = (gamma_inv * sign_du - n_surf[l] * phi(q_pos, j)) * Pn_u * w0;
          sign_du *= w0;

          // Insert over block size in matrix
          for (std::size_t i = 0; i < ndofs_cell; i++)
          {
            for (std::size_t b = 0; b < bs; b++)
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
