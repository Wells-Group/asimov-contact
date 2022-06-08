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
  // FIXME: This will not work for prism meshes
  const std::vector<std::int32_t>& qp_offsets = quadrature_rule->offset();
  const std::size_t num_q_points = qp_offsets[1] - qp_offsets[0];
  const std::size_t ndofs_cell = V->dofmap()->element_dof_layout().num_dofs();

  // Coefficient offsets
  // Expecting coefficients in following order:
  // mu, lmbda, h, gap, normals, test_fn, u, u_opposite
  std::vector<std::size_t> cstrides
      = {1,
         1,
         1,
         num_q_points * gdim,
         num_q_points * gdim,
         num_q_points * ndofs_cell * bs * max_links,
         ndofs_cell * bs,
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
  kernel_fn<PetscScalar> meshtie_rhs
      = [kd](std::vector<std::vector<PetscScalar>>& b, const PetscScalar* c,
             const PetscScalar* w, const double* coordinate_dofs,
             const int facet_index, const std::size_t num_links)
  {
    // assumption that the vector function space has block size tdim
    std::array<std::int32_t, 2> q_offset
        = {kd.qp_offsets(facet_index), kd.qp_offsets(facet_index + 1)};

    // NOTE: DOLFINx has 3D input coordinate dofs
    // FIXME: These array should be views (when compute_jacobian doesn't use
    // xtensor)
    std::array<std::size_t, 2> shape
        = {(std::size_t)kd.num_coordinate_dofs(), 3};
    xt::xtensor<double, 2> coord
        = xt::adapt(coordinate_dofs, kd.num_coordinate_dofs() * 3,
                    xt::no_ownership(), shape);

    // Create data structures for jacobians
    xt::xtensor<double, 2> J = xt::zeros<double>({kd.gdim(), kd.tdim()});
    xt::xtensor<double, 2> K = xt::zeros<double>({kd.tdim(), kd.gdim()});
    xt::xtensor<double, 2> J_tot
        = xt::zeros<double>({J.shape(0), (std::size_t)kd.tdim() - 1});
    double detJ = 0;
    auto c_view = xt::view(coord, xt::all(), xt::range(0, kd.gdim()));

    // Normal vector on physical facet at a single quadrature point
    xt::xtensor<double, 1> n_phys = xt::zeros<double>({kd.gdim()});

    // Pre-compute jacobians and normals for affine meshes
    if (kd.affine())
    {
      detJ = kd.compute_facet_jacobians(facet_index, J, K, J_tot, coord);
      dolfinx_contact::physical_facet_normal(
          n_phys, K, xt::row(kd.facet_normals(), facet_index));
    }

    // Extract constants used inside quadrature loop
    double gamma = w[0] / c[2]; // gamma/h
    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];

    // Extract reference to the tabulated basis function
    const xt::xtensor<double, 2>& phi = kd.phi();
    const xt::xtensor<double, 3>& dphi = kd.dphi();

    // Extract reference to quadrature weights for the local facet
    xtl::span<const double> _weights(kd.q_weights());
    auto weights = _weights.subspan(q_offset[0], q_offset[1] - q_offset[0]);

    // Temporary data structures used inside quadrature loop
    std::array<double, 3> n_surf = {0, 0, 0};
    xt::xtensor<double, 2> tr({kd.ndofs_cell(), kd.gdim()});
    xt::xtensor<double, 2> epsn({kd.ndofs_cell(), kd.gdim()});
    // Loop over quadrature points
    const int num_points = q_offset[1] - q_offset[0];
    for (int q = 0; q < num_points; q++)
    {
      const std::size_t q_pos = q_offset[0] + q;

      // Update Jacobian and physical normal
      detJ = kd.update_jacobian(q, facet_index, detJ, J, K, J_tot, coord);
      kd.update_normal(n_phys, K, facet_index);
      double n_dot = 0;
      double gap = 0;
      // For closest point projection the gap function is given by
      // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
      // in y = Pi(x)
      for (std::size_t i = 0; i < kd.gdim(); i++)
      {

        n_surf[i] = -c[kd.offsets(4) + q * kd.gdim() + i];
        n_dot += n_phys(i) * n_surf[i];
        gap += c[kd.offsets(3) + q * kd.gdim() + i] * n_surf[i];
      }
      compute_normal_strain_basis(epsn, tr, K, dphi, n_surf, n_phys, q_pos);
      // compute tr(eps(u)), epsn at q
      double tr_u = 0;
      double epsn_u = 0;
      double jump_un = 0;
      for (std::size_t i = 0; i < kd.ndofs_cell(); i++)
      {
        std::size_t block_index = kd.offsets(6) + i * kd.bs();
        for (std::size_t j = 0; j < kd.bs(); j++)
        {
          PetscScalar coeff = c[block_index + j];
          tr_u += coeff * tr(i, j);
          epsn_u += coeff * epsn(i, j);
          jump_un += coeff * phi(q_pos, i) * n_surf[j];
        }
      }
      std::size_t offset_u_opp = kd.offsets(7) + q * kd.bs();
      for (std::size_t j = 0; j < kd.bs(); ++j)
        jump_un += -c[offset_u_opp + j] * n_surf[j];
      double sign_u = lmbda * tr_u * n_dot + mu * epsn_u;
      const double w0 = weights[q] * detJ;

      sign_u *= w0;

      // Fill contributions of facet with itself

      for (std::size_t i = 0; i < kd.ndofs_cell(); i++)
      {
        for (std::size_t n = 0; n < kd.bs(); n++)
        {
          double v_dot_nsurf = n_surf[n] * phi(q_pos, i);
          double sign_v = (lmbda * tr(i, n) * n_dot + mu * epsn(i, n));
          // This is (1./gamma)*Pn_v to avoid the product gamma*(1./gamma)
          b[0][n + i * kd.bs()] += -sign_u * v_dot_nsurf
                                   - theta * sign_v * jump_un
                                   + gamma * jump_un * v_dot_nsurf;
          // 0.5 * (-theta * gamma * sign_v * sign_u + Pn_u * Pn_v);

          // entries corresponding to v on the other surface
          for (std::size_t k = 0; k < num_links; k++)
          {
            std::size_t index = kd.offsets(5)
                                + k * num_points * kd.ndofs_cell() * kd.bs()
                                + i * num_points * kd.bs() + q * kd.bs() + n;
            double v_n_opp = c[index] * n_surf[n];

            b[k + 1][n + i * kd.bs()]
                += -sign_u * v_n_opp + gamma * v_n_opp * jump_un;
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
  kernel_fn<PetscScalar> meshtie_jac
      = [kd](std::vector<std::vector<PetscScalar>>& A, const double* c,
             const double* w, const double* coordinate_dofs,
             const int facet_index, const std::size_t num_links) {};
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