// Copyright (C) 2022 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#include "meshtie_kernels.h"
dolfinx_contact::kernel_fn<PetscScalar>
dolfinx_contact::generate_meshtie_kernel(
    dolfinx_contact::Kernel type,
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    std::shared_ptr<const dolfinx_contact::QuadratureRule> quadrature_rule,
    const std::size_t max_links)
{
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V->mesh();
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
  const std::size_t bs = V->dofmap()->bs();
  // NOTE: Assuming same number of quadrature points on each cell
  dolfinx_contact::error::check_cell_type(mesh->topology().cell_type());
  const std::vector<std::int32_t>& qp_offsets = quadrature_rule->offset();
  const std::size_t num_q_points = qp_offsets[1] - qp_offsets[0];
  const std::size_t ndofs_cell = V->dofmap()->element_dof_layout().num_dofs();

  // Coefficient offsets
  // Expecting coefficients in following order:
  // mu, lmbda, h,test_fn, grad(test_fn), u, u_opposite,
  // grad(u_opposite)
  std::vector<std::size_t> cstrides
      = {1,
         1,
         1,
         num_q_points * ndofs_cell * bs * max_links,
         num_q_points * ndofs_cell * bs * max_links,
         ndofs_cell * bs,
         num_q_points * bs,
         num_q_points * gdim * bs};

  auto kd = dolfinx_contact::KernelData(V, quadrature_rule, cstrides);
  /// @brief Assemble kernel for RHS gluing two objects with Nitsche
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
  kernel_fn<PetscScalar> meshtie_rhs
      = [kd, gdim, bs,
         ndofs_cell](std::vector<std::vector<PetscScalar>>& b,
                     std::span<const PetscScalar> c, const PetscScalar* w,
                     const double* coordinate_dofs, const int facet_index,
                     const std::size_t num_links)
  {
    // Retrieve some data from kd
    std::array<std::int32_t, 2> q_offset
        = {kd.qp_offsets(facet_index), kd.qp_offsets(facet_index + 1)};
    auto tdim = std::size_t(kd.tdim());

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
    xt::xtensor<double, 1> n_phys = xt::zeros<double>({gdim});

    // Pre-compute jacobians and normals for affine meshes
    if (kd.affine())
    {
      detJ = kd.compute_facet_jacobians(facet_index, J, K, J_tot, detJ_scratch,
                                        coord);

      dolfinx_contact::physical_facet_normal(
          std::span(n_phys.data(), gdim), K,
          stdex::submdspan(kd.facet_normals(), facet_index,
                           stdex::full_extent));
    }

    // Extract constants used inside quadrature loop
    double gamma = w[0] / c[2]; // gamma/h
    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];

    // Extract reference to the tabulated basis function
    cmdspan2_t phi = kd.phi();
    cmdspan3_t dphi = kd.dphi();

    // Extract reference to quadrature weights for the local facet
    std::span<const double> _weights(kd.q_weights());
    auto weights = _weights.subspan(q_offset[0], q_offset[1] - q_offset[0]);

    // Temporary data structures used inside quadrature loop
    std::vector<double> sig_nb(ndofs_cell * gdim * gdim);
    std::vector<double> sig_n_oppb(num_links * ndofs_cell * gdim * gdim);
    std::vector<double> sig_n_u(gdim);
    std::vector<double> jump_u(gdim);
    mdspan3_t sig_n(sig_nb.data(), ndofs_cell, gdim, gdim);
    mdspan4_t sig_n_opp(sig_n_oppb.data(), num_links, ndofs_cell, gdim, gdim);

    // Loop over quadrature points
    const int num_points = q_offset[1] - q_offset[0];
    for (int q = 0; q < num_points; q++)
    {
      const std::size_t q_pos = q_offset[0] + q;

      // Update Jacobian and physical normal
      detJ = kd.update_jacobian(q, facet_index, detJ, J, K, J_tot, detJ_scratch,
                                coord);
      kd.update_normal(n_phys, K, facet_index);
      compute_sigma_n_basis(sig_n, K, dphi, n_phys, mu, lmbda, q_pos);
      compute_sigma_n_opp(
          sig_n_opp, c.subspan(kd.offsets(4), kd.offsets(5) - kd.offsets(4)),
          n_phys, mu, lmbda, q, num_points);

      // compute u, 0.5 sig_n(u)
      std::fill(sig_n_u.begin(), sig_n_u.end(), 0.0);
      std::fill(jump_u.begin(), jump_u.end(), 0.0);
      for (std::size_t i = 0; i < ndofs_cell; i++)
      {
        std::size_t block_index = kd.offsets(5) + i * bs;
        for (std::size_t j = 0; j < bs; j++)
        {
          PetscScalar coeff = c[block_index + j];
          jump_u[j] += coeff * phi(q_pos, i);
          for (std::size_t k = 0; k < bs; ++k)
            sig_n_u[k] += coeff * sig_n(i, j, k);
        }
      }
      // avg(sig_n(u)):  sig_n(u) +=  sig_n(u_opposite)
      compute_sigma_n_u(sig_n_u,
                        c.subspan(kd.offsets(7) + q * gdim * gdim, gdim * gdim),
                        n_phys, mu, lmbda);

      // compute [[u]] = jump(u) = u - u_opp
      std::size_t offset_u_opp = kd.offsets(6) + q * bs;
      for (std::size_t j = 0; j < bs; ++j)
        jump_u[j] += -c[offset_u_opp + j];
      const double w0 = 0.5 * weights[q] * detJ;

      // Fill contributions of facet with itself

      for (std::size_t i = 0; i < ndofs_cell; i++)
      {
        for (std::size_t n = 0; n < bs; n++)
        {
          // inner(-avg(sig(u)n) + gamma[[u]], v)
          b[0][n + i * bs]
              += (-0.5 * sig_n_u[n] + gamma * jump_u[n]) * phi(q_pos, i) * w0;

          // // -theta inner(0.5 sig(v)n, [[u]])
          for (std::size_t g = 0; g < gdim; ++g)
            b[0][n + i * bs] += -0.5 * theta * sig_n(i, n, g) * jump_u[g] * w0;

          // entries corresponding to v on the other surface
          for (std::size_t k = 0; k < num_links; k++)
          {
            std::size_t index = kd.offsets(3) + k * num_points * ndofs_cell * bs
                                + i * num_points * bs + q * bs + n;

            // -inner(-avg(sig(u)n) + gamma[[u]], v_opposite)
            b[k + 1][n + i * bs]
                += (0.5 * sig_n_u[n] - gamma * jump_u[n]) * c[index] * w0;

            // -0.5 theta inner(sig(v_opposite)n, [[u]])
            for (std::size_t g = 0; g < gdim; ++g)
              b[k + 1][n + i * bs]
                  += -0.5 * theta * sig_n_opp(k, i, n, g) * jump_u[g] * w0;
          }
        }
      }
    }
  };

  /// @brief Assemble kernel for Jacobian (LHS) for gluing two problems with
  /// Nitsche
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
  kernel_fn<PetscScalar> meshtie_jac
      = [kd, gdim, bs, ndofs_cell](
            std::vector<std::vector<PetscScalar>>& A, std::span<const double> c,
            const double* w, const double* coordinate_dofs,
            const int facet_index, const std::size_t num_links)
  {
    // Retrieve some data from kd
    std::array<std::int32_t, 2> q_offset
        = {kd.qp_offsets(facet_index), kd.qp_offsets(facet_index + 1)};
    auto tdim = std::size_t(kd.tdim());
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
    xt::xtensor<double, 1> n_phys = xt::zeros<double>({gdim});

    // Pre-compute jacobians and normals for affine meshes
    if (kd.affine())
    {
      detJ = kd.compute_facet_jacobians(facet_index, J, K, J_tot, detJ_scratch,
                                        coord);

      dolfinx_contact::physical_facet_normal(
          std::span(n_phys.data(), gdim), K,
          stdex::submdspan(kd.facet_normals(), facet_index,
                           stdex::full_extent));
    }

    // Extract constants used inside quadrature loop
    double gamma = w[0] / c[2]; // gamma/h
    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];

    // Extract reference to the tabulated basis function
    cmdspan2_t phi = kd.phi();
    cmdspan3_t dphi = kd.dphi();

    // Extract reference to quadrature weights for the local facet
    std::span<const double> _weights(kd.q_weights());
    auto weights = _weights.subspan(q_offset[0], q_offset[1] - q_offset[0]);

    // Temporary data structures used inside quadrature loop
    std::vector<double> sig_nb(ndofs_cell * gdim * gdim);
    std::vector<double> sig_n_oppb(num_links * ndofs_cell * gdim * gdim);
    mdspan3_t sig_n(sig_nb.data(), ndofs_cell, gdim, gdim);
    mdspan4_t sig_n_opp(sig_n_oppb.data(), num_links, ndofs_cell, gdim, gdim);

    // Loop over quadrature points
    const int num_points = q_offset[1] - q_offset[0];
    for (int q = 0; q < num_points; q++)
    {
      const std::size_t q_pos = q_offset[0] + q;
      // Update Jacobian and physical normal
      detJ = kd.update_jacobian(q, facet_index, detJ, J, K, J_tot, detJ_scratch,
                                coord);
      kd.update_normal(n_phys, K, facet_index);
      compute_sigma_n_basis(sig_n, K, dphi, n_phys, mu, lmbda, q_pos);
      compute_sigma_n_opp(
          sig_n_opp, c.subspan(kd.offsets(4), kd.offsets(5) - kd.offsets(4)),
          n_phys, mu, lmbda, q, num_points);

      const double w0 = 0.5 * weights[q] * detJ;
      for (std::size_t j = 0; j < ndofs_cell; j++)
      {
        for (std::size_t l = 0; l < bs; l++)
        {
          for (std::size_t i = 0; i < ndofs_cell; i++)
          {

            // gamma inner(u, v)
            A[0][(l + i * bs) * ndofs_cell * bs + l + j * bs]
                += gamma * phi(q_pos, j) * phi(q_pos, i) * w0;

            // inner products of test and trial functions only non-zero if dof
            // corresponds to same block index
            for (std::size_t k = 0; k < num_links; k++)
            {
              std::size_t index_u = kd.offsets(3)
                                    + k * num_points * ndofs_cell * bs
                                    + j * num_points * bs + q * bs + l;
              std::size_t index_v = kd.offsets(3)
                                    + k * num_points * ndofs_cell * bs
                                    + i * num_points * bs + q * bs + l;

              // - gamma inner(u_opp, v)
              A[3 * k + 1][(l + i * bs) * bs * ndofs_cell + l + j * bs]
                  += -gamma * c[index_u] * phi(q_pos, i) * w0;
              // - gamma inner(u, v_opp)
              A[3 * k + 2][(l + i * bs) * bs * ndofs_cell + l + j * bs]
                  += -gamma * phi(q_pos, j) * c[index_v] * w0;
              // + gamma inner(u_opp, v_opp)
              A[3 * k + 3][(l + i * bs) * bs * ndofs_cell + l + j * bs]
                  += gamma * c[index_u] * c[index_v] * w0;
            }

            for (std::size_t b = 0; b < bs; b++)
            {
              // Fill contributions of facet with itself
              // -0.5 inner(sig(u)n, v) - 0.5 theta inner(sig(v), u)
              A[0][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                  += (-0.5 * sig_n(j, l, b) * phi(q_pos, i)
                      - 0.5 * theta * sig_n(i, b, l) * phi(q_pos, j))
                     * w0;

              // entries corresponding to u and v on the other surface
              for (std::size_t k = 0; k < num_links; k++)
              {
                std::size_t index_u = kd.offsets(3)
                                      + k * num_points * ndofs_cell * bs
                                      + j * num_points * bs + q * bs + l;
                std::size_t index_v = kd.offsets(3)
                                      + k * num_points * ndofs_cell * bs
                                      + i * num_points * bs + q * bs + b;
                // -0.5 inner(sig(u_opp), v) +0.5 theta inner(sig(v), u_opp)
                A[3 * k + 1][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    += (-0.5 * sig_n_opp(k, j, l, b) * phi(q_pos, i)
                        + 0.5 * theta * sig_n(i, b, l) * c[index_u])
                       * w0;

                // 0.5 inner(sig(u), v_opp) -0.5 theta inner(sig(v_opp), u)
                A[3 * k + 2][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    += (0.5 * sig_n(j, l, b) * c[index_v]
                        - 0.5 * theta * sig_n_opp(k, i, b, l) * phi(q_pos, j))
                       * w0;
                // 0.5 inner(sig(u_opp), v_opp) +0.5 theta
                // inner(sig(v_opp),u_opp)
                A[3 * k + 3][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    += (0.5 * sig_n_opp(k, j, l, b) * c[index_v]
                        + 0.5 * theta * sig_n_opp(k, i, b, l) * c[index_u])
                       * w0;
              }
            }
          }
        }
      }
    }
  };
  switch (type)
  {
  case dolfinx_contact::Kernel::MeshTieRhs:
    return meshtie_rhs;
  case dolfinx_contact::Kernel::MeshTieJac:
    return meshtie_jac;
  default:
    throw std::invalid_argument("Unrecognized kernel");
  }
}