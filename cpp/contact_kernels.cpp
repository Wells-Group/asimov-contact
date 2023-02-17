// Copyright (C) 2022 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#include "contact_kernels.h"
dolfinx_contact::kernel_fn<PetscScalar>
dolfinx_contact::generate_contact_kernel(
    dolfinx_contact::Kernel type,
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    std::shared_ptr<const dolfinx_contact::QuadratureRule> quadrature_rule,
    const std::size_t max_links)
{
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V->mesh();
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
  const std::size_t bs = V->dofmap()->bs();
  // FIXME: This will not work for prism meshes
  const std::vector<std::size_t>& qp_offsets = quadrature_rule->offset();
  const std::size_t num_q_points = qp_offsets[1] - qp_offsets[0];
  const std::size_t ndofs_cell = V->dofmap()->element_dof_layout().num_dofs();

  // Strides for creating coefficient offsets
  // Expecting coefficients in following order (all arrays flattened row major):
  // kd.offsets(0)  - mu             - shape (1, )
  // kd.offsets(1)  - lmbda          - shape (1, )
  // kd.offsets(2)  - h              - shape (1, )
  // kd.offsets(3)  - gap            - shape (num_q_points, gdim)
  // kd.offsets(4)  - normalsx       - shape (num_q_points, gdim)
  // kd.offsets(5)  - normalsy       - shape (num_q_points, gdim)
  // kd.offsets(6)  - test_fn        - shape (num_q_points, ndofs_cell, bs,
  //                                          max_links)
  // kd.offsets(7)  - grad(test_fn)  - shape (num_q_points,
  //                                          ndofs_cell, bs, max_links)
  // kd.offsets(8)  - u              - shape (num_q_points, gdim)
  // kd.offsets(9)  - grad(u)        - shape (num_q_points, gdim, gdim)
  // kd.offsets(10) - u_opposite     - shape (num_q_points, bs)
  // kd.offsets(11) - grad(u_opp)    - shape (num_q_points, gdim, gdim)
  std::vector<std::size_t> cstrides
      = {1,
         1,
         1,
         num_q_points * gdim,
         num_q_points * gdim,
         num_q_points * gdim,
         num_q_points * ndofs_cell * bs * max_links,
         num_q_points * ndofs_cell * bs * max_links,
         num_q_points * gdim,
         num_q_points * gdim * gdim,
         num_q_points * bs,
         num_q_points * gdim * bs};

  auto kd = dolfinx_contact::KernelData(V, quadrature_rule, cstrides);

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
  kernel_fn<PetscScalar> ray_jac
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
    std::array<double, 9> def_gradb;
    mdspan2_t def_grad(def_gradb.data(), gdim, gdim);
    std::array<double, 9> def_grad_invb;
    mdspan2_t def_grad_inv(def_grad_invb.data(), gdim, gdim);

    // Normal vector on physical facet at a single quadrature point
    std::array<double, 3> n_phys;

    // Pre-compute jacobians and normals for affine meshes
    if (kd.affine())
    {
      detJ = kd.compute_first_facet_jacobian(facet_index, J, K, J_tot,
                                             detJ_scratch, coord);
      physical_facet_normal(std::span(n_phys.data(), gdim), K,
                            stdex::submdspan(kd.facet_normals(), facet_index,
                                             stdex::full_extent));
    }

    // Extract scaled gamma (h/gamma) and its inverse
    double gamma = c[2] / w[0];
    double gamma_inv = w[0] / c[2];

    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];

    cmdspan3_t dphi = kd.dphi();
    cmdspan2_t phi = kd.phi();
    std::array<std::size_t, 2> q_offset
        = {kd.qp_offsets(facet_index), kd.qp_offsets(facet_index + 1)};
    const std::size_t num_points = q_offset.back() - q_offset.front();
    std::span<const double> weights = kd.weights(facet_index);
    std::array<double, 3> n_x = {0, 0, 0};
    std::array<double, 3> n_y = {0, 0, 0};
    std::vector<double> epsnb(ndofs_cell * gdim);
    mdspan2_t epsn(epsnb.data(), ndofs_cell, gdim);
    std::vector<double> trb(ndofs_cell * gdim);
    mdspan2_t tr(trb.data(), ndofs_cell, gdim);
    std::vector<double> sig_n_u(gdim);
    std::vector<double> dnxb(gdim * ndofs_cell * bs);
    mdspan3_t dnx(dnxb.data(), ndofs_cell, bs, gdim);
    std::vector<double> dgb((num_links + 1) * ndofs_cell * bs);
    mdspan3_t dg(dgb.data(), num_links + 1, ndofs_cell, bs);
    std::vector<double> dyb((num_links + 1) * ndofs_cell * bs * gdim);
    mdspan4_t dy(dyb.data(), num_links + 1, ndofs_cell, bs, gdim);

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
      for (std::size_t i = 0; i < gdim; i++)
      {
        n_x[i] = -c[kd.offsets(4) + q * gdim + i];
        n_y[i] = c[kd.offsets(5) + q * gdim + i];
        n_dot += n_phys[i] * n_x[i];
        gap += c[kd.offsets(3) + q * gdim + i] * n_x[i];
      }

      compute_normal_strain_basis(epsn, tr, K, dphi, n_x,
                                  std::span(n_phys.data(), gdim), q_pos);

      // compute sig(u)*n_phys, grad(u) = c.subspan(kd.offsets(9) + q * gdim *
      // gdim, gdim * gdim),
      std::fill(sig_n_u.begin(), sig_n_u.end(), 0.0);
      compute_sigma_n_u(sig_n_u,
                        c.subspan(kd.offsets(9) + q * gdim * gdim, gdim * gdim),
                        std::span(n_phys.data(), gdim), mu, lmbda);
      // compute Dnx
      std::fill(dnxb.begin(), dnxb.end(), 0.0);
      compute_dnx(c.subspan(kd.offsets(9) + q * gdim * gdim, gdim * gdim), dphi,
                  K, n_x, dnx, def_grad, def_grad_inv, q_pos);

      // compute Dg
      std::fill(dgb.begin(), dgb.end(), 0.0);
      compute_dg(dg, dnx,
                 c.subspan(kd.offsets(6), kd.offsets(7) - kd.offsets(6)), phi,
                 n_x, n_y, q_offset.front(), q, q_indices.size(), gap);

      // compute DY
      std::fill(dyb.begin(), dyb.end(), 0.0);
      compute_dy(
          dy, dnx, c.subspan(kd.offsets(11), kd.offsets(12) - kd.offsets(11)),
          c.subspan(kd.offsets(6), kd.offsets(7) - kd.offsets(6)), phi, n_x,
          n_y, def_grad, def_grad_inv, q_offset.front(), q, q_indices.size(), gap);
      // compute inner(sig(u)*n_phys, n_surf) and inner(u, n_surf)
      double sign_u = 0;
      for (std::size_t j = 0; j < gdim; ++j)
        sign_u += sig_n_u[j] * n_x[j];

      double dPn_u = dR_minus(gap + gamma * sign_u);

      // Fill contributions of facet with itself
      const double weight = weights[q] * detJ;
      for (std::size_t j = 0; j < ndofs_cell; j++)
      {
        for (std::size_t l = 0; l < bs; l++)
        {
          double sign_w = (lmbda * tr(j, l) * n_dot + mu * epsn(j, l));

          sign_w *= weight;
          for (std::size_t i = 0; i < ndofs_cell; i++)
          {
            for (std::size_t b = 0; b < bs; b++)
            {
              double v_dot_nsurf = n_x[b] * phi(q_pos, i);
              double sign_v = (lmbda * tr(i, b) * n_dot + mu * epsn(i, b));
              double Pn_v = gamma_inv * v_dot_nsurf - theta * sign_v;
              A[0][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                  += 0.5 * (gamma *dPn_u * sign_w + dg(0, j, l)*n_x[l]*weight) * Pn_v;

              // entries corresponding to u and v on the other surface
              for (std::size_t k = 0; k < num_links; k++)
              {
                // index for trial function value on opposite surface
                std::size_t index = kd.offsets(6)
                                    + k * num_points * ndofs_cell * bs
                                    + j * num_points * bs + q * bs + l;
                double w_n_opp = c[index] * n_x[l];

                w_n_opp *= weight * Pn_u;
                // index for test function value on opposite surface
                index = kd.offsets(6) + k * num_points * ndofs_cell * bs
                        + i * num_points * bs + q * bs + b;
                double v_n_opp = c[index] * n_x[b];
                A[3 * k + 1][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    -= 0.5 * gamma_inv * dg(k, j, l)*n_x[l] * Pn_v;
                A[3 * k + 2][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    -= 0.5 * dPn_u * sign_w * v_n_opp;
                A[3 * k + 3][(b + i * bs) * bs * ndofs_cell + l + j * bs]
                    += 0.5 * gamma_inv * dg(k, j, l) *n_x[l] * v_n_opp;
              }
            }
          }
        }
      }
    }
  };

  switch (type)
  {
  case Kernel::RayJac:
  {
    return ray_jac;
  }
  default:
    throw std::invalid_argument("Unrecognized kernel");
  }
}