// Copyright (C) 2023 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#include "contact_kernels.h"
dolfinx_contact::kernel_fn<PetscScalar>
dolfinx_contact::generate_contact_kernel(
    dolfinx_contact::Kernel type,
    std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
    std::shared_ptr<const dolfinx_contact::QuadratureRule> quadrature_rule,
    const std::size_t max_links)
{
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh = V->mesh();
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
  const std::size_t bs = V->dofmap()->bs();
  // FIXME: This will not work for prism meshes
  const std::vector<std::size_t>& qp_offsets = quadrature_rule->offset();
  const std::size_t num_q_points = qp_offsets[1] - qp_offsets[0];
  const std::size_t ndofs_cell = V->dofmap()->element_dof_layout().num_dofs();

  // Coefficient offsets
  // Expecting coefficients in following order:
  // mu, lmbda, h, friction coefficient,
  // gap, normals, test_fn, u, grad(u), u_opposite
  std::vector<std::size_t> cstrides
      = {4,
         num_q_points * gdim,
         num_q_points * gdim,
         num_q_points * ndofs_cell * bs * max_links,
         num_q_points * gdim,
         num_q_points * gdim * gdim,
         num_q_points * bs};

  auto kd = dolfinx_contact::KernelData(V, quadrature_rule, cstrides);

  /// @brief Assemble kernel for RHS of unbiased contact problem
  ///
  /// Assemble of the residual of the unbiased contact problem into vector
  /// `b`.
  /// @param[in,out] b The vector to assemble the residual into
  /// @param[in] c The coefficients used in kernel. Assumed to be
  /// ordered as mu, lmbda, h, gap, normals, test_fn, u, u_opposite.
  /// @param[in] w The constants used in kernel. Assumed to be ordered as
  /// `gamma`, `theta`.
  /// @param[in] coordinate_dofs The physical coordinates of cell. Assumed to
  /// be padded to 3D, (shape (num_nodes, 3)).
  /// @param[in] facet_index Local facet index (relative to cell)
  /// @param[in] num_links How many cells from opposite surface are connected
  /// with the cell.
  /// @param[in] q_indices The quadrature points to loop over
  kernel_fn<PetscScalar> unbiased_rhs =
      [kd, gdim, ndofs_cell,
       bs](std::vector<std::vector<PetscScalar>>& b,
           std::span<const PetscScalar> c, const PetscScalar* w,
           const double* coordinate_dofs, const std::size_t facet_index,
           const std::size_t num_links, std::span<const std::int32_t> q_indices)

  {
    // Retrieve some data from kd
    const std::uint32_t tdim = kd.tdim();

    // NOTE: DOLFINx has 3D input coordinate dofs
    cmdspan2_t coord(coordinate_dofs, kd.num_coordinate_dofs(), 3);

    // Create data structures for jacobians
    // We allocate more memory than required, but its better for the compiler
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
      physical_facet_normal(std::span(n_phys.data(), gdim), K,
                            MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                                kd.facet_normals(), facet_index,
                                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent));
    }

    // Extract constants used inside quadrature loop
    double gamma = c[3] / w[0];     // h/gamma
    double gamma_inv = w[0] / c[3]; // gamma/h
    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];
    // Extract reference to the tabulated basis function
    s_cmdspan2_t phi = kd.phi();
    s_cmdspan3_t dphi = kd.dphi();

    // Extract reference to quadrature weights for the local facet

    auto weights = kd.weights(facet_index);

    // Temporary data structures used inside quadrature loop
    std::array<double, 3> n_surf = {0, 0, 0};
    std::vector<double> epsnb(ndofs_cell * gdim, 0);
    mdspan2_t epsn(epsnb.data(), ndofs_cell, gdim);
    std::vector<double> trb(ndofs_cell * gdim, 0);
    mdspan2_t tr(trb.data(), ndofs_cell, gdim);
    std::vector<double> sig_n_u(gdim);

    // Loop over quadrature points
    const std::size_t q_start = kd.qp_offsets(facet_index);
    const std::size_t q_end = kd.qp_offsets(facet_index + 1);
    const std::size_t num_points = q_end - q_start;
    for (auto q : q_indices)
    {
      const std::size_t q_pos = q_start + q;

      // Update Jacobian and physical normal
      detJ = kd.update_jacobian(q, facet_index, detJ, J, K, J_tot, detJ_scratch,
                                coord);
      kd.update_normal(std::span(n_phys.data(), gdim), K, facet_index);
      double n_dot = 0;
      double gap = 0;
      // For ray tracing the gap is given by n * (Pi(x) -x)
      // where n = n_x
      // For closest point n = -n_y
      for (std::size_t i = 0; i < gdim; i++)
      {
        n_surf[i] = -c[kd.offsets(2) + q * gdim + i];
        n_dot += n_phys[i] * n_surf[i];
        gap += c[kd.offsets(1) + q * gdim + i] * n_surf[i];
      }

      compute_normal_strain_basis(epsn, tr, K, dphi, n_surf,
                                  std::span(n_phys.data(), gdim), q_pos);

      // compute sig(u)*n_phys
      std::fill(sig_n_u.begin(), sig_n_u.end(), 0.0);
      compute_sigma_n_u(sig_n_u,
                        c.subspan(kd.offsets(5) + q * gdim * gdim, gdim * gdim),
                        std::span(n_phys.data(), gdim), mu, lmbda);

      // compute inner(sig(u)*n_phys, n_surf) and inner(u, n_surf)
      double sign_u = 0;
      double jump_un = 0;
      for (std::size_t j = 0; j < gdim; ++j)
      {
        sign_u += sig_n_u[j] * n_surf[j];
        jump_un += c[kd.offsets(4) + gdim * q + j] * n_surf[j];
      }
      std::size_t offset_u_opp = kd.offsets(6) + q * bs;
      for (std::size_t j = 0; j < bs; ++j)
        jump_un += -c[offset_u_opp + j] * n_surf[j];

      const double w0 = weights[q] * detJ;

      double Pn_u = R_plus((jump_un - gap) - gamma * sign_u) * w0;
      // Fill contributions of facet with itself
      for (std::size_t i = 0; i < ndofs_cell; i++)
      {
        for (std::size_t n = 0; n < bs; n++)
        {
          double v_dot_nsurf = n_surf[n] * phi(q_pos, i);
          double sign_v = (lmbda * tr(i, n) * n_dot + mu * epsn(i, n));
          // This is (1./gamma)*Pn_v to avoid the product gamma*(1./gamma)
          double Pn_v = gamma_inv * v_dot_nsurf - theta * sign_v;
          b[0][n + i * bs] += 0.5 * Pn_u * Pn_v;

          // entries corresponding to v on the other surface
          for (std::size_t k = 0; k < num_links; k++)
          {
            std::size_t index = kd.offsets(3) + k * num_points * ndofs_cell * bs
                                + i * num_points * bs + q * bs + n;
            double v_n_opp = c[index] * n_surf[n];

            b[k + 1][n + i * bs] -= 0.5 * gamma_inv * v_n_opp * Pn_u;
          }
        }
      }
    }
  };

  /// @brief Assemble kernel for Jacobian (LHS) of unbiased contact
  /// problem
  ///
  /// Assemble of the residual of the unbiased contact problem into matrix
  /// `A`.
  /// @param[in,out] A The matrix to assemble the Jacobian into
  /// @param[in] c The coefficients used in kernel. Assumed to be
  /// ordered as mu, lmbda, h, gap, normals, test_fn, u, u_opposite.
  /// @param[in] w The constants used in kernel. Assumed to be ordered as
  /// `gamma`, `theta`.
  /// @param[in] coordinate_dofs The physical coordinates of cell. Assumed
  /// to be padded to 3D, (shape (num_nodes, 3)).
  /// @param[in] facet_index Local facet index (relative to cell)
  /// @param[in] num_links How many cells from opposite surface are connected
  /// with the cell.
  /// @param[in] q_indices The quadrature points to loop over
  kernel_fn<PetscScalar> unbiased_jac
      = [kd, gdim, ndofs_cell, bs](
            std::vector<std::vector<PetscScalar>>& A, std::span<const double> c,
            const double* w, const double* coordinate_dofs,
            const std::size_t facet_index, const std::size_t num_links,
            std::span<const std::int32_t> q_indices)
  {
    // Retrieve some data from kd
    const std::uint32_t tdim = kd.tdim();

    // NOTE: DOLFINx has 3D input coordinate dofs
    cmdspan2_t coord(coordinate_dofs, kd.num_coordinate_dofs(), 3);

    // Create data structures for jacobians
    // We allocate more memory than required, but its better for the compiler
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
      physical_facet_normal(std::span(n_phys.data(), gdim), K,
                            MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                                kd.facet_normals(), facet_index,
                                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent));
    }

    // Extract scaled gamma (h/gamma) and its inverse
    double gamma = c[3] / w[0];
    double gamma_inv = w[0] / c[3];

    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];

    s_cmdspan3_t dphi = kd.dphi();
    s_cmdspan2_t phi = kd.phi();
    std::array<std::size_t, 2> q_offset
        = {kd.qp_offsets(facet_index), kd.qp_offsets(facet_index + 1)};
    const std::size_t num_points = q_offset.back() - q_offset.front();
    std::span<const double> weights = kd.weights(facet_index);
    std::array<double, 3> n_surf = {0, 0, 0};
    std::vector<double> epsnb(ndofs_cell * gdim);
    mdspan2_t epsn(epsnb.data(), ndofs_cell, gdim);
    std::vector<double> trb(ndofs_cell * gdim);
    mdspan2_t tr(trb.data(), ndofs_cell, gdim);
    std::vector<double> sig_n_u(gdim);

    // Loop over quadrature points
    for (auto q : q_indices)
    {
      const std::size_t q_pos = q_offset.front() + q;
      // Update Jacobian and physical normal
      detJ = kd.update_jacobian(q, facet_index, detJ, J, K, J_tot, detJ_scratch,
                                coord);
      kd.update_normal(std::span(n_phys.data(), gdim), K, facet_index);

      double n_dot = 0;
      double gap = 0;
      // The gap is given by n * (Pi(x) -x)
      // For raytracing n = n_x
      // For closest point n = -n_y
      for (std::size_t i = 0; i < gdim; i++)
      {
        n_surf[i] = -c[kd.offsets(2) + q * gdim + i];
        n_dot += n_phys[i] * n_surf[i];
        gap += c[kd.offsets(1) + q * gdim + i] * n_surf[i];
      }

      compute_normal_strain_basis(epsn, tr, K, dphi, n_surf,
                                  std::span(n_phys.data(), gdim), q_pos);

      // compute sig(u)*n_phys
      std::fill(sig_n_u.begin(), sig_n_u.end(), 0.0);
      compute_sigma_n_u(sig_n_u,
                        c.subspan(kd.offsets(5) + q * gdim * gdim, gdim * gdim),
                        std::span(n_phys.data(), gdim), mu, lmbda);

      // compute inner(sig(u)*n_phys, n_surf) and inner(u, n_surf)
      double sign_u = 0;
      double jump_un = 0;
      for (std::size_t j = 0; j < gdim; ++j)
      {
        sign_u += sig_n_u[j] * n_surf[j];
        jump_un += c[kd.offsets(4) + gdim * q + j] * n_surf[j];
      }
      std::size_t offset_u_opp = kd.offsets(6) + q * bs;
      for (std::size_t j = 0; j < bs; ++j)
        jump_un += -c[offset_u_opp + j] * n_surf[j];

      double Pn_u = dR_plus((jump_un - gap) - gamma * sign_u);

      // Fill contributions of facet with itself
      const double w0 = weights[q] * detJ;
      for (std::size_t j = 0; j < ndofs_cell; j++)
      {
        for (std::size_t l = 0; l < bs; l++)
        {
          double sign_du = (lmbda * tr(j, l) * n_dot + mu * epsn(j, l));
          double Pn_du
              = (phi(q_pos, j) * n_surf[l] - gamma * sign_du) * Pn_u * w0;

          sign_du *= w0;
          for (std::size_t i = 0; i < ndofs_cell; i++)
          {
            for (std::size_t b = 0; b < bs; b++)
            {
              double v_dot_nsurf = n_surf[b] * phi(q_pos, i);
              double sign_v = (lmbda * tr(i, b) * n_dot + mu * epsn(i, b));
              double Pn_v = gamma_inv * v_dot_nsurf - theta * sign_v;
              A[0][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                  += 0.5 * Pn_du * Pn_v;

              // entries corresponding to u and v on the other surface
              for (std::size_t k = 0; k < num_links; k++)
              {
                std::size_t index = kd.offsets(3)
                                    + k * num_points * ndofs_cell * bs
                                    + j * num_points * bs + q * bs + l;
                double du_n_opp = c[index] * n_surf[l];

                du_n_opp *= w0 * Pn_u;
                index = kd.offsets(3) + k * num_points * ndofs_cell * bs
                        + i * num_points * bs + q * bs + b;
                double v_n_opp = c[index] * n_surf[b];
                A[3 * k + 1][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    -= 0.5 * du_n_opp * Pn_v;
                A[3 * k + 2][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    -= 0.5 * gamma_inv * Pn_du * v_n_opp;
                A[3 * k + 3][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    += 0.5 * gamma_inv * du_n_opp * v_n_opp;
              }
            }
          }
        }
      }
    }
  };

  /// @brief Assemble kernel for RHS of the friction term for unbiased contact
  /// problem with tresca friction Assemble of the residual of the unbiased
  /// contact problem into vector `b`.
  /// @param[in,out] b The vector to assemble the residual into
  /// @param[in] c The coefficients used in kernel. Assumed to be
  /// ordered as mu, lmbda, h, gap, normals, test_fn, u, u_opposite.
  /// @param[in] w The constants used in kernel. Assumed to be ordered as
  /// `gamma`, `theta`.
  /// @param[in] coordinate_dofs The physical coordinates of cell. Assumed to
  /// be padded to 3D, (shape (num_nodes, 3)).
  /// @param[in] facet_index Local facet index (relative to cell)
  /// @param[in] num_links How many cells from opposite surface are connected
  /// with the cell.
  /// @param[in] q_indices The quadrature points to loop over
  kernel_fn<PetscScalar> tresca_rhs =
      [kd, gdim, ndofs_cell,
       bs](std::vector<std::vector<PetscScalar>>& b,
           std::span<const PetscScalar> c, const PetscScalar* w,
           const double* coordinate_dofs, const std::size_t facet_index,
           const std::size_t num_links, std::span<const std::int32_t> q_indices)

  {
    // Retrieve some data from kd
    const std::uint32_t tdim = kd.tdim();

    // NOTE: DOLFINx has 3D input coordinate dofs
    cmdspan2_t coord(coordinate_dofs, kd.num_coordinate_dofs(), 3);

    // Create data structures for jacobians
    // We allocate more memory than required, but its better for the compiler
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
      physical_facet_normal(std::span(n_phys.data(), gdim), K,
                            MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                                kd.facet_normals(), facet_index,
                                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent));
    }

    // Extract constants used inside quadrature loop
    double gamma = c[3] / w[0];     // h/gamma
    double gamma_inv = w[0] / c[3]; // gamma/h
    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];
    double fric = c[2];
    // Extract reference to the tabulated basis function
    s_cmdspan2_t phi = kd.phi();
    s_cmdspan3_t dphi = kd.dphi();

    // Extract reference to quadrature weights for the local facet

    auto weights = kd.weights(facet_index);

    // Temporary data structures used inside quadrature loop
    std::array<double, 3> n_surf = {0, 0, 0};
    std::array<double, 3> Pt_u = {0, 0, 0};
    std::vector<double> epsnb(ndofs_cell * gdim, 0);
    mdspan2_t epsn(epsnb.data(), ndofs_cell, gdim);
    std::vector<double> trb(ndofs_cell * gdim, 0);
    mdspan2_t tr(trb.data(), ndofs_cell, gdim);
    std::vector<double> sig_n_u(gdim);
    std::vector<double> sig_nb(ndofs_cell * gdim * gdim);
    mdspan3_t sig_n(sig_nb.data(), ndofs_cell, gdim, gdim);

    // Loop over quadrature points
    const std::size_t q_start = kd.qp_offsets(facet_index);
    const std::size_t q_end = kd.qp_offsets(facet_index + 1);
    const std::size_t num_points = q_end - q_start;
    for (auto q : q_indices)
    {
      const std::size_t q_pos = q_start + q;

      // Update Jacobian and physical normal
      detJ = kd.update_jacobian(q, facet_index, detJ, J, K, J_tot, detJ_scratch,
                                coord);
      kd.update_normal(std::span(n_phys.data(), gdim), K, facet_index);
      double n_dot = 0;
      double gap = 0;
      // For ray tracing the gap is given by n * (Pi(x) -x)
      // where n = n_x
      // For closest point n = -n_y
      for (std::size_t i = 0; i < gdim; i++)
      {
        n_surf[i] = -c[kd.offsets(2) + q * gdim + i];
        n_dot += n_phys[i] * n_surf[i];
        gap += c[kd.offsets(1) + q * gdim + i] * n_surf[i];
      }

      compute_normal_strain_basis(epsn, tr, K, dphi, n_surf,
                                  std::span(n_phys.data(), gdim), q_pos);

      // compute sig(u)*n_phys
      std::fill(sig_n_u.begin(), sig_n_u.end(), 0.0);
      compute_sigma_n_u(sig_n_u,
                        c.subspan(kd.offsets(5) + q * gdim * gdim, gdim * gdim),
                        std::span(n_phys.data(), gdim), mu, lmbda);

      compute_sigma_n_basis(sig_n, K, dphi, std::span(n_phys.data(), gdim), mu,
                            lmbda, q_pos);

      // compute inner(sig(u)*n_phys, n_surf) and inner(u, n_surf)
      double sign_u = 0;
      double jump_un = 0;
      for (std::size_t j = 0; j < gdim; ++j)
      {
        sign_u += sig_n_u[j] * n_surf[j];
        jump_un += c[kd.offsets(4) + gdim * q + j] * n_surf[j];
      }
      std::size_t offset_u_opp = kd.offsets(6) + q * bs;
      for (std::size_t j = 0; j < bs; ++j)
        jump_un += -c[offset_u_opp + j] * n_surf[j];

      for (std::size_t j = 0; j < bs; ++j)
      {
        Pt_u[j] = c[kd.offsets(4) + gdim * q + j] - c[offset_u_opp + j]
                  - jump_un * n_surf[j];
        Pt_u[j] -= gamma * (sig_n_u[j] - sign_u * n_surf[j]);
      }
      const double w0 = weights[q] * detJ;

      // compute ball projection
      std::array<double, 3> Pt_u_proj = ball_projection(Pt_u, gamma * fric);
      // Fill contributions of facet with itself
      for (std::size_t i = 0; i < ndofs_cell; i++)
      {
        for (std::size_t n = 0; n < bs; n++)
        {
          double v_dot_nsurf = n_surf[n] * phi(q_pos, i);
          double sign_v = (lmbda * tr(i, n) * n_dot + mu * epsn(i, n));

          // inner(Pt_u_proj, v[x])
          b[0][n + i * bs]
              += 0.5 * gamma_inv * Pt_u_proj[n] * phi(q_pos, i) * w0;
          for (std::size_t j = 0; j < bs; j++)
          {
            // -v_n[x]*n[j] - theta/gamma*sigma_t(v)[j]
            double Pt_vj
                = -v_dot_nsurf * n_surf[j]
                  - theta * gamma * (sig_n(i, n, j) - sign_v * n_surf[j]);
            // Pt_u_proj[j] * Pt_vj
            b[0][n + i * bs] += 0.5 * gamma_inv * Pt_u_proj[j] * Pt_vj * w0;
          }

          // entries corresponding to v on the other surface
          for (std::size_t k = 0; k < num_links; k++)
          {
            std::size_t index = kd.offsets(3) + k * num_points * ndofs_cell * bs
                                + i * num_points * bs + q * bs;
            double v_n_opp = c[index + n] * n_surf[n];

            // inner(Pt_u_proj, v[y])
            b[k + 1][n + i * bs]
                -= 0.5 * gamma_inv * Pt_u_proj[n] * c[index + n] * w0;
            for (std::size_t j = 0; j < bs; j++)

            { // Pt_u_proj[j] * v_n n[j]
              b[k + 1][n + i * bs]
                  += 0.5 * gamma_inv * Pt_u_proj[j] * v_n_opp * n_surf[j] * w0;
            }
          }
        }
      }
    }
  };

  /// @brief Assemble kernel for Jacobian (LHS) of the friction term for
  /// unbiased contact problem with tresca friction
  ///
  /// Assemble of the residual of the unbiased contact problem into matrix
  /// `A`.
  /// @param[in,out] A The matrix to assemble the Jacobian into
  /// @param[in] c The coefficients used in kernel. Assumed to be
  /// ordered as mu, lmbda, h, gap, normals, test_fn, u, u_opposite.
  /// @param[in] w The constants used in kernel. Assumed to be ordered as
  /// `gamma`, `theta`.
  /// @param[in] coordinate_dofs The physical coordinates of cell. Assumed
  /// to be padded to 3D, (shape (num_nodes, 3)).
  /// @param[in] facet_index Local facet index (relative to cell)
  /// @param[in] num_links How many cells from opposite surface are connected
  /// with the cell.
  /// @param[in] q_indices The quadrature points to loop over
  kernel_fn<PetscScalar> tresca_jac
      = [kd, gdim, ndofs_cell, bs](
            std::vector<std::vector<PetscScalar>>& A, std::span<const double> c,
            const double* w, const double* coordinate_dofs,
            const std::size_t facet_index, const std::size_t num_links,
            std::span<const std::int32_t> q_indices)
  {
    // Retrieve some data from kd
    const std::uint32_t tdim = kd.tdim();

    // NOTE: DOLFINx has 3D input coordinate dofs
    cmdspan2_t coord(coordinate_dofs, kd.num_coordinate_dofs(), 3);

    // Create data structures for jacobians
    // We allocate more memory than required, but its better for the compiler
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
      physical_facet_normal(std::span(n_phys.data(), gdim), K,
                            MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                                kd.facet_normals(), facet_index,
                                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent));
    }

    // Extract scaled gamma (h/gamma) and its inverse
    double gamma = c[3] / w[0];
    double gamma_inv = w[0] / c[3];

    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];
    double fric = c[2];

    s_cmdspan3_t dphi = kd.dphi();
    s_cmdspan2_t phi = kd.phi();
    std::array<std::size_t, 2> q_offset
        = {kd.qp_offsets(facet_index), kd.qp_offsets(facet_index + 1)};
    const std::size_t num_points = q_offset.back() - q_offset.front();
    std::span<const double> weights = kd.weights(facet_index);
    std::array<double, 3> n_surf = {0, 0, 0};
    std::array<double, 3> Pt_u = {0, 0, 0};
    std::vector<double> epsnb(ndofs_cell * gdim);
    mdspan2_t epsn(epsnb.data(), ndofs_cell, gdim);
    std::vector<double> trb(ndofs_cell * gdim);
    mdspan2_t tr(trb.data(), ndofs_cell, gdim);
    std::vector<double> sig_n_u(gdim);
    std::vector<double> sig_nb(ndofs_cell * gdim * gdim);
    mdspan3_t sig_n(sig_nb.data(), ndofs_cell, gdim, gdim);

    // Loop over quadrature points
    for (auto q : q_indices)
    {
      const std::size_t q_pos = q_offset.front() + q;
      // Update Jacobian and physical normal
      detJ = kd.update_jacobian(q, facet_index, detJ, J, K, J_tot, detJ_scratch,
                                coord);
      kd.update_normal(std::span(n_phys.data(), gdim), K, facet_index);

      double n_dot = 0;
      double gap = 0;
      // The gap is given by n * (Pi(x) -x)
      // For raytracing n = n_x
      // For closest point n = -n_y
      for (std::size_t i = 0; i < gdim; i++)
      {
        n_surf[i] = -c[kd.offsets(2) + q * gdim + i];
        n_dot += n_phys[i] * n_surf[i];
        gap += c[kd.offsets(1) + q * gdim + i] * n_surf[i];
      }

      compute_normal_strain_basis(epsn, tr, K, dphi, n_surf,
                                  std::span(n_phys.data(), gdim), q_pos);

      // compute sig(u)*n_phys
      std::fill(sig_n_u.begin(), sig_n_u.end(), 0.0);
      compute_sigma_n_u(sig_n_u,
                        c.subspan(kd.offsets(5) + q * gdim * gdim, gdim * gdim),
                        std::span(n_phys.data(), gdim), mu, lmbda);

      compute_sigma_n_basis(sig_n, K, dphi, std::span(n_phys.data(), gdim), mu,
                            lmbda, q_pos);

      // compute inner(sig(u)*n_phys, n_surf) and inner(u, n_surf)
      double sign_u = 0;
      double jump_un = 0;
      for (std::size_t j = 0; j < gdim; ++j)
      {
        sign_u += sig_n_u[j] * n_surf[j];
        jump_un += c[kd.offsets(4) + gdim * q + j] * n_surf[j];
      }
      std::size_t offset_u_opp = kd.offsets(6) + q * bs;
      for (std::size_t j = 0; j < bs; ++j)
        jump_un += -c[offset_u_opp + j] * n_surf[j];
      for (std::size_t j = 0; j < bs; ++j)
      {
        Pt_u[j] = c[kd.offsets(4) + gdim * q + j] - c[offset_u_opp + j]
                  - jump_un * n_surf[j];
        Pt_u[j] -= gamma * (sig_n_u[j] - sign_u * n_surf[j]);
      }

      std::array<double, 9> Pt_u_proj
          = d_ball_projection(Pt_u, gamma * fric, bs);

      // Fill contributions of facet with itself
      const double w0 = weights[q] * detJ;
      for (std::size_t j = 0; j < ndofs_cell; j++)
      {
        for (std::size_t l = 0; l < bs; l++)
        {
          double w_dot_nsurf = n_surf[l] * phi(q_pos, j);
          double sign_w = (lmbda * tr(j, l) * n_dot + mu * epsn(j, l));

          // Pt_w = J_ball * w_t[X] + gamma* J_ball * sigma_t(w)
          std::array<double, 3> Pt_w = {0, 0, 0};
          for (std::size_t m = 0; m < bs; ++m)
          { // J_ball * w[X]
            Pt_w[m] += Pt_u_proj[l * bs + m] * phi(q_pos, j);
            for (std::size_t n = 0; n < bs; n++)
            {
              // - w_n[X] J_ball * n_surf - gamma * J_ball * sgima_t(w)
              Pt_w[m] -= Pt_u_proj[n * bs + m]
                         * (w_dot_nsurf * n_surf[n]
                            + gamma * (sig_n(j, l, n) - sign_w * n_surf[n]));
            }
          }

          for (std::size_t i = 0; i < ndofs_cell; i++)
          {
            for (std::size_t b = 0; b < bs; b++)
            {
              double v_dot_nsurf = n_surf[b] * phi(q_pos, i);
              double sign_v = (lmbda * tr(i, b) * n_dot + mu * epsn(i, b));
              // inner (Pt_w, v[X])
              A[0][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                  += 0.5 * gamma_inv * Pt_w[b] * phi(q_pos, i) * w0;
              for (std::size_t n = 0; n < bs; n++)
              {
                // - inner(v[X], n_surf[X])*v_n[X] -theta/gamma*sgima_t(v)[X]
                double Pt_vn
                    = -v_dot_nsurf * n_surf[n]
                      - theta * gamma * (sig_n(i, b, n) - sign_v * n_surf[n]);
                // Pt_w[n] * Pt_vn
                A[0][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                    += 0.5 * gamma_inv * Pt_w[n] * Pt_vn * w0;
              }

              // entries corresponding to u and v on the other surface
              for (std::size_t k = 0; k < num_links; k++)
              {
                std::size_t index = kd.offsets(3)
                                    + k * num_points * ndofs_cell * bs
                                    + j * num_points * bs + q * bs + l;
                double wn_opp = c[index] * n_surf[l];
                // Pt_w_opp = - J_ball * w_t[Y]
                std::array<double, 3> Pt_w_opp = {0, 0, 0};

                for (std::size_t m = 0; m < bs; ++m)
                {
                  Pt_w_opp[m] += Pt_u_proj[l * bs + m] * c[index];
                  for (std::size_t n = 0; n < bs; ++n)
                    Pt_w_opp[m] -= Pt_u_proj[n * bs + m] * wn_opp * n_surf[n];
                }
                index = kd.offsets(3) + k * num_points * ndofs_cell * bs
                        + i * num_points * bs + q * bs;
                double v_n_opp = c[index + b] * n_surf[b];
                // inner(Pt_w_opp, v[X])
                A[3 * k + 1][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    -= 0.5 * gamma_inv * Pt_w_opp[b] * phi(q_pos, i) * w0;
                // -inner (Pt_w, v[y])
                A[3 * k + 2][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    -= 0.5 * gamma_inv * Pt_w[b] * c[index + b] * w0;
                // inner(Pt_w_opp, v[y])
                A[3 * k + 3][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    += 0.5 * gamma_inv * Pt_w_opp[b] * c[index + b] * w0;
                for (std::size_t n = 0; n < bs; ++n)

                {
                  // - inner(v[X], n_surf[X])*v_n[X] -theta/gamma*sgima_t(v)[X]
                  double Pt_vn
                      = -v_dot_nsurf * n_surf[n]
                        - theta * gamma * (sig_n(i, b, n) - sign_v * n_surf[n]);
                  // inner(Pt_w_opp, Pt_vn)
                  A[3 * k + 1][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                      -= 0.5 * gamma_inv * Pt_w_opp[n] * Pt_vn * w0;
                  // inner(Pt_w, n_surf) v_n_opp
                  A[3 * k + 2][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                      += 0.5 * gamma_inv * Pt_w[n] * n_surf[n] * v_n_opp * w0;
                  // inner(Pt_w_opp, n_surf) v_n_opp
                  A[3 * k + 3][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                      -= 0.5 * gamma_inv * Pt_w_opp[n] * n_surf[n] * v_n_opp;
                }
              }
            }
          }
        }
      }
    }
  };

  /// @brief Assemble kernel for RHS of the friction term for unbiased contact
  /// problem with coulomb friction Assemble of the residual of the unbiased
  /// contact problem into vector `b`.
  /// @param[in,out] b The vector to assemble the residual into
  /// @param[in] c The coefficients used in kernel. Assumed to be
  /// ordered as mu, lmbda, h, gap, normals, test_fn, u, u_opposite.
  /// @param[in] w The constants used in kernel. Assumed to be ordered as
  /// `gamma`, `theta`.
  /// @param[in] coordinate_dofs The physical coordinates of cell. Assumed to
  /// be padded to 3D, (shape (num_nodes, 3)).
  /// @param[in] facet_index Local facet index (relative to cell)
  /// @param[in] num_links How many cells from opposite surface are connected
  /// with the cell.
  /// @param[in] q_indices The quadrature points to loop over
  kernel_fn<PetscScalar> coulomb_rhs =
      [kd, gdim, ndofs_cell,
       bs](std::vector<std::vector<PetscScalar>>& b,
           std::span<const PetscScalar> c, const PetscScalar* w,
           const double* coordinate_dofs, const std::size_t facet_index,
           const std::size_t num_links, std::span<const std::int32_t> q_indices)

  {
    // Retrieve some data from kd
    const std::uint32_t tdim = kd.tdim();

    // NOTE: DOLFINx has 3D input coordinate dofs
    cmdspan2_t coord(coordinate_dofs, kd.num_coordinate_dofs(), 3);

    // Create data structures for jacobians
    // We allocate more memory than required, but its better for the compiler
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
      physical_facet_normal(std::span(n_phys.data(), gdim), K,
                            MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                                kd.facet_normals(), facet_index,
                                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent));
    }

    // Extract constants used inside quadrature loop
    double gamma = c[3] / w[0];     // h/gamma
    double gamma_inv = w[0] / c[3]; // gamma/h
    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];
    double fric = c[2];
    // Extract reference to the tabulated basis function
    s_cmdspan2_t phi = kd.phi();
    s_cmdspan3_t dphi = kd.dphi();

    // Extract reference to quadrature weights for the local facet

    auto weights = kd.weights(facet_index);

    // Temporary data structures used inside quadrature loop
    std::array<double, 3> n_surf = {0, 0, 0};
    std::array<double, 3> Pt_u = {0, 0, 0};
    std::vector<double> epsnb(ndofs_cell * gdim, 0);
    mdspan2_t epsn(epsnb.data(), ndofs_cell, gdim);
    std::vector<double> trb(ndofs_cell * gdim, 0);
    mdspan2_t tr(trb.data(), ndofs_cell, gdim);
    std::vector<double> sig_n_u(gdim);
    std::vector<double> sig_nb(ndofs_cell * gdim * gdim);
    mdspan3_t sig_n(sig_nb.data(), ndofs_cell, gdim, gdim);

    // Loop over quadrature points
    const std::size_t q_start = kd.qp_offsets(facet_index);
    const std::size_t q_end = kd.qp_offsets(facet_index + 1);
    const std::size_t num_points = q_end - q_start;
    for (auto q : q_indices)
    {
      const std::size_t q_pos = q_start + q;

      // Update Jacobian and physical normal
      detJ = kd.update_jacobian(q, facet_index, detJ, J, K, J_tot, detJ_scratch,
                                coord);
      kd.update_normal(std::span(n_phys.data(), gdim), K, facet_index);
      double n_dot = 0;
      double gap = 0;
      // For ray tracing the gap is given by n * (Pi(x) -x)
      // where n = n_x
      // For closest point n = -n_y
      for (std::size_t i = 0; i < gdim; i++)
      {
        n_surf[i] = -c[kd.offsets(2) + q * gdim + i];
        n_dot += n_phys[i] * n_surf[i];
        gap += c[kd.offsets(1) + q * gdim + i] * n_surf[i];
      }

      compute_normal_strain_basis(epsn, tr, K, dphi, n_surf,
                                  std::span(n_phys.data(), gdim), q_pos);

      // compute sig(u)*n_phys
      std::fill(sig_n_u.begin(), sig_n_u.end(), 0.0);
      compute_sigma_n_u(sig_n_u,
                        c.subspan(kd.offsets(5) + q * gdim * gdim, gdim * gdim),
                        std::span(n_phys.data(), gdim), mu, lmbda);

      compute_sigma_n_basis(sig_n, K, dphi, std::span(n_phys.data(), gdim), mu,
                            lmbda, q_pos);

      // compute inner(sig(u)*n_phys, n_surf) and inner(u, n_surf)
      double sign_u = 0;
      double jump_un = 0;
      for (std::size_t j = 0; j < gdim; ++j)
      {
        sign_u += sig_n_u[j] * n_surf[j];
        jump_un += c[kd.offsets(4) + gdim * q + j] * n_surf[j];
      }
      std::size_t offset_u_opp = kd.offsets(6) + q * bs;
      for (std::size_t j = 0; j < bs; ++j)
        jump_un += -c[offset_u_opp + j] * n_surf[j];

      for (std::size_t j = 0; j < bs; ++j)
      {
        Pt_u[j] = c[kd.offsets(4) + gdim * q + j] - c[offset_u_opp + j]
                  - jump_un * n_surf[j];
        Pt_u[j] -= gamma * (sig_n_u[j] - sign_u * n_surf[j]);
      }
      const double w0 = weights[q] * detJ;
      double Pn_u = R_plus((jump_un - gap) - gamma * sign_u);
      // compute ball projection
      std::array<double, 3> Pt_u_proj
          = ball_projection(Pt_u, gamma * fric * Pn_u);
      // Fill contributions of facet with itself
      for (std::size_t i = 0; i < ndofs_cell; i++)
      {
        for (std::size_t n = 0; n < bs; n++)
        {
          double v_dot_nsurf = n_surf[n] * phi(q_pos, i);
          double sign_v = (lmbda * tr(i, n) * n_dot + mu * epsn(i, n));

          // inner(Pt_u_proj, v[x])
          b[0][n + i * bs]
              += 0.5 * gamma_inv * Pt_u_proj[n] * phi(q_pos, i) * w0;
          for (std::size_t j = 0; j < bs; j++)
          {
            // -v_n[x]*n[j] - theta/gamma*sigma_t(v)[j]
            double Pt_vj
                = -v_dot_nsurf * n_surf[j]
                  - theta * gamma * (sig_n(i, n, j) - sign_v * n_surf[j]);
            // Pt_u_proj[j] * Pt_vj
            b[0][n + i * bs] += 0.5 * gamma_inv * Pt_u_proj[j] * Pt_vj * w0;
          }

          // entries corresponding to v on the other surface
          for (std::size_t k = 0; k < num_links; k++)
          {
            std::size_t index = kd.offsets(3) + k * num_points * ndofs_cell * bs
                                + i * num_points * bs + q * bs;
            double v_n_opp = c[index + n] * n_surf[n];

            // inner(Pt_u_proj, v[y])
            b[k + 1][n + i * bs]
                -= 0.5 * gamma_inv * Pt_u_proj[n] * c[index + n] * w0;
            for (std::size_t j = 0; j < bs; j++)

            { // Pt_u_proj[j] * v_n n[j]
              b[k + 1][n + i * bs]
                  += 0.5 * gamma_inv * Pt_u_proj[j] * v_n_opp * n_surf[j] * w0;
            }
          }
        }
      }
    }
  };

  /// @brief Assemble kernel for Jacobian (LHS) of the friction term for
  /// unbiased contact problem with coulomb friction
  ///
  /// Assemble of the residual of the unbiased contact problem into matrix
  /// `A`.
  /// @param[in,out] A The matrix to assemble the Jacobian into
  /// @param[in] c The coefficients used in kernel. Assumed to be
  /// ordered as mu, lmbda, h, gap, normals, test_fn, u, u_opposite.
  /// @param[in] w The constants used in kernel. Assumed to be ordered as
  /// `gamma`, `theta`.
  /// @param[in] coordinate_dofs The physical coordinates of cell. Assumed
  /// to be padded to 3D, (shape (num_nodes, 3)).
  /// @param[in] facet_index Local facet index (relative to cell)
  /// @param[in] num_links How many cells from opposite surface are connected
  /// with the cell.
  /// @param[in] q_indices The quadrature points to loop over
  kernel_fn<PetscScalar> coulomb_jac
      = [kd, gdim, ndofs_cell, bs](
            std::vector<std::vector<PetscScalar>>& A, std::span<const double> c,
            const double* w, const double* coordinate_dofs,
            const std::size_t facet_index, const std::size_t num_links,
            std::span<const std::int32_t> q_indices)
  {
    // Retrieve some data from kd
    const std::uint32_t tdim = kd.tdim();

    // NOTE: DOLFINx has 3D input coordinate dofs
    cmdspan2_t coord(coordinate_dofs, kd.num_coordinate_dofs(), 3);

    // Create data structures for jacobians
    // We allocate more memory than required, but its better for the compiler
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
      physical_facet_normal(std::span(n_phys.data(), gdim), K,
                            MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                                kd.facet_normals(), facet_index,
                                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent));
    }

    // Extract scaled gamma (h/gamma) and its inverse
    double gamma = c[3] / w[0];
    double gamma_inv = w[0] / c[3];

    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];
    double fric = c[2];

    s_cmdspan3_t dphi = kd.dphi();
    s_cmdspan2_t phi = kd.phi();
    std::array<std::size_t, 2> q_offset
        = {kd.qp_offsets(facet_index), kd.qp_offsets(facet_index + 1)};
    const std::size_t num_points = q_offset.back() - q_offset.front();
    std::span<const double> weights = kd.weights(facet_index);
    std::array<double, 3> n_surf = {0, 0, 0};
    std::array<double, 3> Pt_u = {0, 0, 0};
    std::vector<double> epsnb(ndofs_cell * gdim);
    mdspan2_t epsn(epsnb.data(), ndofs_cell, gdim);
    std::vector<double> trb(ndofs_cell * gdim);
    mdspan2_t tr(trb.data(), ndofs_cell, gdim);
    std::vector<double> sig_n_u(gdim);
    std::vector<double> sig_nb(ndofs_cell * gdim * gdim);
    mdspan3_t sig_n(sig_nb.data(), ndofs_cell, gdim, gdim);

    // Loop over quadrature points
    for (auto q : q_indices)
    {
      const std::size_t q_pos = q_offset.front() + q;
      // Update Jacobian and physical normal
      detJ = kd.update_jacobian(q, facet_index, detJ, J, K, J_tot, detJ_scratch,
                                coord);
      kd.update_normal(std::span(n_phys.data(), gdim), K, facet_index);

      double n_dot = 0;
      double gap = 0;
      // The gap is given by n * (Pi(x) -x)
      // For raytracing n = n_x
      // For closest point n = -n_y
      for (std::size_t i = 0; i < gdim; i++)
      {
        n_surf[i] = -c[kd.offsets(2) + q * gdim + i];
        n_dot += n_phys[i] * n_surf[i];
        gap += c[kd.offsets(1) + q * gdim + i] * n_surf[i];
      }

      compute_normal_strain_basis(epsn, tr, K, dphi, n_surf,
                                  std::span(n_phys.data(), gdim), q_pos);

      // compute sig(u)*n_phys
      std::fill(sig_n_u.begin(), sig_n_u.end(), 0.0);
      compute_sigma_n_u(sig_n_u,
                        c.subspan(kd.offsets(5) + q * gdim * gdim, gdim * gdim),
                        std::span(n_phys.data(), gdim), mu, lmbda);

      compute_sigma_n_basis(sig_n, K, dphi, std::span(n_phys.data(), gdim), mu,
                            lmbda, q_pos);

      // compute inner(sig(u)*n_phys, n_surf) and inner(u, n_surf)
      double sign_u = 0;
      double jump_un = 0;
      for (std::size_t j = 0; j < gdim; ++j)
      {
        sign_u += sig_n_u[j] * n_surf[j];
        jump_un += c[kd.offsets(4) + gdim * q + j] * n_surf[j];
      }
      std::size_t offset_u_opp = kd.offsets(6) + q * bs;
      for (std::size_t j = 0; j < bs; ++j)
        jump_un += -c[offset_u_opp + j] * n_surf[j];
      for (std::size_t j = 0; j < bs; ++j)
      {
        Pt_u[j] = c[kd.offsets(4) + gdim * q + j] - c[offset_u_opp + j]
                  - jump_un * n_surf[j];
        Pt_u[j] -= gamma * (sig_n_u[j] - sign_u * n_surf[j]);
      }
      double Pn_u = R_plus((jump_un - gap) - gamma * sign_u);
      std::array<double, 9> Pt_u_proj
          = d_ball_projection(Pt_u, gamma * fric * Pn_u, bs);

      double d_alpha = dR_plus((jump_un - gap) - gamma * sign_u) * gamma * fric;

      std::array<double, 3> d_alpha_ball
          = d_alpha_ball_projection(Pt_u, gamma * fric * Pn_u, d_alpha);
      // Fill contributions of facet with itself
      const double w0 = weights[q] * detJ;
      for (std::size_t j = 0; j < ndofs_cell; j++)
      {
        for (std::size_t l = 0; l < bs; l++)
        {
          double w_dot_nsurf = n_surf[l] * phi(q_pos, j);
          double sign_w = (lmbda * tr(j, l) * n_dot + mu * epsn(j, l));

          // Pt_w = J_ball * w_t[X] + gamma* J_ball * sigma_t(w)
          std::array<double, 3> Pt_w = {0, 0, 0};
          for (std::size_t m = 0; m < bs; ++m)
          { // J_ball * w[X]
            Pt_w[m] += Pt_u_proj[l * bs + m] * phi(q_pos, j);
            for (std::size_t n = 0; n < bs; n++)
            {
              // - w_n[X] J_ball * n_surf - gamma * J_ball * sgima_t(w)
              Pt_w[m] -= Pt_u_proj[n * bs + m]
                         * (w_dot_nsurf * n_surf[n]
                            + gamma * (sig_n(j, l, n) - sign_w * n_surf[n]));
            }
          }
          double Pn_w = (phi(q_pos, j) * n_surf[l] - gamma * sign_w);
          for (std::size_t i = 0; i < ndofs_cell; i++)
          {
            for (std::size_t b = 0; b < bs; b++)
            {
              double v_dot_nsurf = n_surf[b] * phi(q_pos, i);
              double sign_v = (lmbda * tr(i, b) * n_dot + mu * epsn(i, b));
              // inner (Pt_w, v[X])
              A[0][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                  += 0.5 * gamma_inv * Pt_w[b] * phi(q_pos, i) * w0;

              // inner (d_alpha_ball * Pn_w, v[x])
              A[0][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                  += 0.5 * gamma_inv * d_alpha_ball[b] * Pn_w * phi(q_pos, i)
                     * w0;
              for (std::size_t n = 0; n < bs; n++)
              {
                // - inner(v[X], n_surf[X])*v_n[X] -theta/gamma*sgima_t(v)[X]
                double Pt_vn
                    = -v_dot_nsurf * n_surf[n]
                      - theta * gamma * (sig_n(i, b, n) - sign_v * n_surf[n]);
                // Pt_w[n] * Pt_vn
                A[0][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                    += 0.5 * gamma_inv * Pt_w[n] * Pt_vn * w0;
                // d_alpha_ball * Pn_w * Pt_vn
                A[0][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                    += 0.5 * gamma_inv * d_alpha_ball[n] * Pn_w * Pt_vn * w0;
              }

              // entries corresponding to u and v on the other surface
              for (std::size_t k = 0; k < num_links; k++)
              {
                std::size_t index = kd.offsets(3)
                                    + k * num_points * ndofs_cell * bs
                                    + j * num_points * bs + q * bs + l;
                double wn_opp = c[index] * n_surf[l];
                // Pt_w_opp = - J_ball * w_t[Y]
                std::array<double, 3> Pt_w_opp = {0, 0, 0};

                for (std::size_t m = 0; m < bs; ++m)
                {
                  Pt_w_opp[m] += Pt_u_proj[l * bs + m] * c[index];
                  for (std::size_t n = 0; n < bs; ++n)
                    Pt_w_opp[m] -= Pt_u_proj[n * bs + m] * wn_opp * n_surf[n];
                }
                index = kd.offsets(3) + k * num_points * ndofs_cell * bs
                        + i * num_points * bs + q * bs;
                double v_n_opp = c[index + b] * n_surf[b];
                // -inner(Pt_w_opp, v[X])
                A[3 * k + 1][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    -= 0.5 * gamma_inv * Pt_w_opp[b] * phi(q_pos, i) * w0;
                // -inner(d_alpha_ball * wn_opp, v[X])
                A[3 * k + 1][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    -= 0.5 * gamma_inv * d_alpha_ball[b] * wn_opp
                       * phi(q_pos, i) * w0;
                // -inner (Pt_w, v[y])
                A[3 * k + 2][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    -= 0.5 * gamma_inv * Pt_w[b] * c[index + b] * w0;
                // -inner (d_alpha_ball * Pn_w, v[y])
                A[3 * k + 2][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    -= 0.5 * gamma_inv * d_alpha_ball[b] * Pn_w * c[index + b]
                       * w0;
                // inner(Pt_w_opp, v[y])
                A[3 * k + 3][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    += 0.5 * gamma_inv * Pt_w_opp[b] * c[index + b] * w0;
                // inner(d_alpha_ball * wn_opp, v[y])
                A[3 * k + 3][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    += 0.5 * gamma_inv * d_alpha_ball[b] * wn_opp * c[index + b]
                       * w0;
                for (std::size_t n = 0; n < bs; ++n)

                {
                  // - inner(v[X], n_surf[X])*v_n[X] -theta/gamma*sgima_t(v)[X]
                  double Pt_vn
                      = -v_dot_nsurf * n_surf[n]
                        - theta * gamma * (sig_n(i, b, n) - sign_v * n_surf[n]);
                  // -inner(Pt_w_opp, Pt_vn)
                  A[3 * k + 1][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                      -= 0.5 * gamma_inv * Pt_w_opp[n] * Pt_vn * w0;
                  // -inner(d_alpha_ball * wn_opp, Pt_vn)
                  A[3 * k + 1][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                      -= 0.5 * gamma_inv * d_alpha_ball[n] * wn_opp * Pt_vn
                         * w0;
                  // inner(Pt_w, n_surf) v_n_opp
                  A[3 * k + 2][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                      += 0.5 * gamma_inv * Pt_w[n] * n_surf[n] * v_n_opp * w0;
                  // inner(d_alpha_ball * Pn_w, n_surf) v_n_opp
                  A[3 * k + 2][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                      += 0.5 * gamma_inv * d_alpha_ball[n] * Pn_w * n_surf[n]
                         * v_n_opp * w0;
                  // -inner(Pt_w_opp, n_surf) v_n_opp
                  A[3 * k + 3][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                      -= 0.5 * gamma_inv * Pt_w_opp[n] * n_surf[n] * v_n_opp;
                  // -inner(d_alpha_ball * wn_opp , n_surf) v_n_opp
                  A[3 * k + 3][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                      -= 0.5 * gamma_inv * d_alpha_ball[n] * wn_opp * n_surf[n]
                         * v_n_opp;
                }
              }
            }
          }
        }
      }
    }
  };
  switch (type)
  {
  case Kernel::Rhs:
    return unbiased_rhs;
  case Kernel::Jac:
    return unbiased_jac;
  case Kernel::TrescaRhs:
    return tresca_rhs;
  case Kernel::TrescaJac:
    return tresca_jac;
  case Kernel::CoulombRhs:
    return coulomb_rhs;
  case Kernel::CoulombJac:
    return coulomb_jac;
  default:
    throw std::invalid_argument("Unrecognized kernel");
  }
}