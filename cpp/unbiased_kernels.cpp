// Copyright (C) 2022 Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT
#include "unbiased_kernels.h"

contact_kernel_fn dolfinx_contact::generate_kernel(
    dolfinx_contact::Kernel type,
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
    std::shared_ptr<dolfinx::fem::FunctionSpace> V,
    std::vector<xt::xarray<double>>& qp_ref_facet,
    std::vector<std::vector<double>>& qw_ref_facet, std::size_t max_links,
    const basix::FiniteElement basix_element)
{
  // mesh data
  const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
  const int tdim = mesh->topology().dim();         // topological dimension

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  const std::size_t ndofs_cell = dofmap->cell_dofs(0).size();
  const std::size_t bs = dofmap->bs();

  // NOTE: Assuming same number of quadrature points on each cell
  const std::size_t num_q_points = qp_ref_facet[0].shape(0);

  // Create coordinate elements (for facet and cell) _marker->mesh()
  const int num_coordinate_dofs = basix_element.dim();
  // Structures needed for basis function tabulation
  // phi and grad(phi) at quadrature points
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  std::vector<xt::xtensor<double, 2>> phi;
  phi.reserve(qp_ref_facet.size());
  std::vector<xt::xtensor<double, 3>> dphi;
  phi.reserve(qp_ref_facet.size());
  std::vector<xt ::xtensor<double, 3>> dphi_c;
  dphi_c.reserve(qp_ref_facet.size());

  // Temporary structures used in loop
  xt::xtensor<double, 4> cell_tab(
      {(std::size_t)tdim + 1, num_q_points, ndofs_cell, bs});
  xt::xtensor<double, 2> phi_i({num_q_points, ndofs_cell});
  xt::xtensor<double, 3> dphi_i({(std::size_t)tdim, num_q_points, ndofs_cell});
  std::array<std::size_t, 4> tabulate_shape
      = basix_element.tabulate_shape(1, num_q_points);
  xt::xtensor<double, 4> c_tab(tabulate_shape);
  xt::xtensor<double, 3> dphi_ci(
      {(std::size_t)tdim, tabulate_shape[1], tabulate_shape[2]});

  // Tabulate basis functions and first order derivatives for each facet in
  // the reference cell. This tabulation is done both for the finite element
  // of the unknown and the coordinate element (which might differ)
  std::for_each(qp_ref_facet.cbegin(), qp_ref_facet.cend(),
                [&](const auto& q_facet)
                {
                  assert(q_facet.shape(0) == num_q_points);
                  element->tabulate(cell_tab, q_facet, 1);

                  phi_i = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
                  phi.push_back(phi_i);

                  dphi_i
                      = xt::view(cell_tab, xt::range(1, (std::size_t)tdim + 1),
                                 xt::all(), xt::all(), 0);
                  dphi.push_back(dphi_i);

                  // Tabulate coordinate element of reference cell
                  basix_element.tabulate(1, q_facet, c_tab);
                  dphi_ci = xt::view(c_tab, xt::range(1, (std::size_t)tdim + 1),
                                     xt::all(), xt::all(), 0);
                  dphi_c.push_back(dphi_ci);
                });
  // coefficient offsets
  // expecting coefficients in following order:
  // mu, lmbda, h, gap, normals, test_fn, u, u_opposite
  // mu, lmbda, h - DG0 one value per  facet
  // gap, normals, test_fn,  u_opposite
  // packed at quadrature points
  // u packed at dofs
  // mu, lmbda, h scalar
  // gap vector valued gdim
  // test_fn, u, u_opposite vector valued bs (should be bs = gdim)
  std::vector<std::size_t> cstrides
      = {1,
         1,
         1,
         num_q_points * gdim,
         num_q_points * gdim,
         num_q_points * ndofs_cell * bs * max_links,
         ndofs_cell * bs,
         num_q_points * bs};
  // As reference facet and reference cell are affine, we do not need to
  // compute this per quadrature point
  xt::xtensor<double, 3> ref_jacobians
      = basix::cell::facet_jacobians(basix_element.cell_type());

  // Get facet normals on reference cell
  xt::xtensor<double, 2> facet_normals
      = basix::cell::facet_outward_normals(basix_element.cell_type());

  // right hand side kernel
  contact_kernel_fn unbiased_rhs
      = [=](std::vector<std::vector<PetscScalar>>& b, const double* c,
            const double* w, const double* coordinate_dofs,
            const int* entity_local_index,
            [[maybe_unused]] const std::uint8_t* quadrature_permutation,
            const std::size_t num_links)
  {
    // assumption that the vector function space has block size tdim
    assert(bs == gdim);
    const auto facet_index = size_t(*entity_local_index);

    // Reshape coordinate dofs to two dimensional array
    // NOTE: DOLFINx has 3D input coordinate dofs
    // FIXME: These array should be views (when compute_jacobian doesn't use
    // xtensor)
    std::array<std::size_t, 2> shape = {(std::size_t)num_coordinate_dofs, 3};
    xt::xtensor<double, 2> coord = xt::adapt(
        coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);

    // Extract the first derivative of the coordinate element (cell) of
    // degrees of freedom on the facet
    const xt::xtensor<double, 3>& dphi_fc = dphi_c[facet_index];
    const xt::xtensor<double, 2>& dphi0_c = xt::view(
        dphi_fc, xt::all(), 0,
        xt::all()); // FIXME: Assumed constant, i.e. only works for simplices
    // NOTE: Affine cell assumption
    // Compute Jacobian and determinant at first quadrature point
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, (std::size_t)tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({(std::size_t)tdim, gdim});
    auto c_view = xt::view(coord, xt::all(), xt::range(0, gdim));
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi0_c, c_view, J);
    dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);

    // Compute normal of physical facet using a normalized covariant Piola
    // transform n_phys = J^{-T} n_ref / ||J^{-T} n_ref|| See for instance
    // DOI: 10.1137/08073901X
    xt::xarray<double> n_phys = xt::zeros<double>({gdim});
    auto facet_normal = xt::row(facet_normals, facet_index);
    for (std::size_t i = 0; i < gdim; i++)
      for (int j = 0; j < tdim; j++)
        n_phys[i] += K(j, i) * facet_normal[j];
    double n_norm = 0;
    for (std::size_t i = 0; i < gdim; i++)
      n_norm += n_phys[i] * n_phys[i];
    n_phys /= std::sqrt(n_norm);

    // h/gamma
    double gamma = c[2] / w[0];
    // gamma/h
    double gamma_inv = w[0] / c[2];
    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];

    // Compute det(J_C J_f) as it is the mapping to the reference facet
    xt::xtensor<double, 2> J_f
        = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
    xt::xtensor<double, 2> J_tot
        = xt::zeros<double>({J.shape(0), J_f.shape(1)});
    dolfinx::math::dot(J, J_f, J_tot);
    double detJ = std::fabs(
        dolfinx::fem::CoordinateElement::compute_jacobian_determinant(J_tot));

    const xt::xtensor<double, 3>& dphi_f = dphi[facet_index];
    const xt::xtensor<double, 2>& phi_f = phi[facet_index];
    const std::vector<double>& weights = qw_ref_facet[facet_index];
    xt::xarray<double> n_surf = xt::zeros<double>({gdim});
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      double n_dot = 0;
      double gap = 0;
      const std::size_t gap_offset = 3;
      const std::size_t normal_offset = gap_offset + cstrides[3];
      for (std::size_t i = 0; i < gdim; i++)
      {
        // For closest point projection the gap function is given by
        // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
        // in y = Pi(x)
        n_surf(i) = -c[normal_offset + q * gdim + i];
        n_dot += n_phys(i) * n_surf(i);
        gap += c[gap_offset + q * gdim + i] * n_surf(i);
      }

      xt::xtensor<double, 2> tr = xt::zeros<double>({ndofs_cell, gdim});
      xt::xtensor<double, 2> epsn = xt::zeros<double>({ndofs_cell, gdim});
      // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2
      for (std::size_t j = 0; j < ndofs_cell; j++)
      {
        for (std::size_t l = 0; l < bs; l++)
        {
          for (int k = 0; k < tdim; k++)
          {
            tr(j, l) += K(k, l) * dphi_f(k, q, j);
            for (std::size_t s = 0; s < gdim; s++)
            {
              epsn(j, l) += K(k, s) * dphi_f(k, q, j)
                            * (n_phys(s) * n_surf(l) + n_phys(l) * n_surf(s));
            }
          }
        }
      }
      // compute tr(eps(u)), epsn at q
      double tr_u = 0;
      double epsn_u = 0;
      std::size_t offset_u = cstrides[0] + cstrides[1] + cstrides[2]
                             + cstrides[3] + cstrides[4] + cstrides[5];

      for (std::size_t i = 0; i < ndofs_cell; i++)
      {
        std::size_t block_index = offset_u + i * bs;
        for (std::size_t j = 0; j < bs; j++)
        {
          tr_u += c[block_index + j] * tr(i, j);
          epsn_u += c[block_index + j] * epsn(i, j);
        }
      }
      double sign_u = lmbda * tr_u * n_dot + mu * epsn_u;
      const double w0 = weights[q] * detJ;
      double Pn_u = dolfinx_contact::R_minus(gap + gamma * sign_u) * w0;
      sign_u *= w0;
      // Fill contributions of facet with itself

      for (std::size_t i = 0; i < ndofs_cell; i++)
      {
        for (std::size_t n = 0; n < bs; n++)
        {
          double v_dot_nsurf = n_surf(n) * phi_f(q, i);
          double sign_v = (lmbda * tr(i, n) * n_dot + mu * epsn(i, n));
          // This is (1./gamma)*Pn_v to avoid the product gamma*(1./gamma)
          double Pn_v = -gamma_inv * v_dot_nsurf + theta * sign_v;
          b[0][n + i * bs] += 0.5 * Pn_u * Pn_v;

          // entries corresponding to v on the other surface
          for (std::size_t k = 0; k < num_links; k++)
          {
            int index = 3 + cstrides[3] + cstrides[4]
                        + k * num_q_points * ndofs_cell * bs
                        + i * num_q_points * bs + q * bs + n;
            double v_n_opp = c[index] * n_surf(n);

            b[k + 1][n + i * bs] += 0.5 * gamma_inv * v_n_opp * Pn_u;
          }
        }
      }
    }
  };

  // jacobian kernel
  contact_kernel_fn unbiased_jac
      = [=](std::vector<std::vector<PetscScalar>>& A, const double* c,
            const double* w, const double* coordinate_dofs,
            const int* entity_local_index,
            [[maybe_unused]] const std::uint8_t* quadrature_permutation,
            const std::size_t num_links)
  {
    // assumption that the vector function space has block size tdim
    assert(bs == gdim);
    const auto facet_index = std::size_t(*entity_local_index);

    // Reshape coordinate dofs to two dimensional array
    // NOTE: DOLFINx has 3D input coordinate dofs
    std::array<std::size_t, 2> shape = {(std::size_t)num_coordinate_dofs, 3};

    // FIXME: These array should be views (when compute_jacobian doesn't use
    // xtensor)
    xt::xtensor<double, 2> coord = xt::adapt(
        coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);

    // Extract the first derivative of the coordinate element (cell) of
    // degrees of freedom on the facet
    const xt::xtensor<double, 3>& dphi_fc = dphi_c[facet_index];
    const xt::xtensor<double, 2>& dphi0_c = xt::view(
        dphi_fc, xt::all(), 0,
        xt::all()); // FIXME: Assumed constant, i.e. only works for simplices
    // NOTE: Affine cell assumption
    // Compute Jacobian and determinant at first quadrature point
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, (std::size_t)tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({(std::size_t)tdim, gdim});
    auto c_view = xt::view(coord, xt::all(), xt::range(0, gdim));
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi0_c, c_view, J);
    dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);

    // Compute normal of physical facet using a normalized covariant Piola
    // transform n_phys = J^{-T} n_ref / ||J^{-T} n_ref|| See for instance
    // DOI: 10.1137/08073901X
    xt::xarray<double> n_phys = xt::zeros<double>({gdim});
    auto facet_normal = xt::row(facet_normals, facet_index);
    for (std::size_t i = 0; i < gdim; i++)
      for (int j = 0; j < tdim; j++)
        n_phys[i] += K(j, i) * facet_normal[j];
    double n_norm = 0;

    for (std::size_t i = 0; i < gdim; i++)
      n_norm += n_phys[i] * n_phys[i];
    n_phys /= std::sqrt(n_norm);

    // Extract scaled gamma (h/gamma) and its inverse
    double gamma = c[2] / w[0];
    double gamma_inv = w[0] / c[2];

    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];

    // Compute det(J_C J_f) as it is the mapping to the reference facet
    xt::xtensor<double, 2> J_f
        = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
    xt::xtensor<double, 2> J_tot
        = xt::zeros<double>({J.shape(0), J_f.shape(1)});
    dolfinx::math::dot(J, J_f, J_tot);
    double detJ = std::fabs(
        dolfinx::fem::CoordinateElement::compute_jacobian_determinant(J_tot));

    const xt::xtensor<double, 3>& dphi_f = dphi[facet_index];
    const xt::xtensor<double, 2>& phi_f = phi[facet_index];
    const std::vector<double>& weights = qw_ref_facet[facet_index];
    xt::xarray<double> n_surf = xt::zeros<double>({gdim});
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      double n_dot = 0;
      double gap = 0;
      const std::size_t gap_offset = 3;
      const std::size_t normal_offset = gap_offset + cstrides[3];
      for (std::size_t i = 0; i < gdim; i++)
      {
        // For closest point projection the gap function is given by
        // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
        // in y = Pi(x)
        n_surf(i) = -c[normal_offset + q * gdim + i];
        n_dot += n_phys(i) * n_surf(i);
        gap += c[gap_offset + q * gdim + i] * n_surf(i);
      }

      xt::xtensor<double, 2> tr = xt::zeros<double>({ndofs_cell, gdim});
      xt::xtensor<double, 2> epsn = xt::zeros<double>({ndofs_cell, gdim});
      // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2
      for (std::size_t j = 0; j < ndofs_cell; j++)
      {
        for (std::size_t l = 0; l < bs; l++)
        {
          for (int k = 0; k < tdim; k++)
          {
            tr(j, l) += K(k, l) * dphi_f(k, q, j);
            for (std::size_t s = 0; s < gdim; s++)
            {
              epsn(j, l) += K(k, s) * dphi_f(k, q, j)
                            * (n_phys(s) * n_surf(l) + n_phys(l) * n_surf(s));
            }
          }
        }
      }

      // compute tr(eps(u)), epsn at q
      double tr_u = 0;
      double epsn_u = 0;
      std::size_t offset_u = cstrides[0] + cstrides[1] + cstrides[2]
                             + cstrides[3] + cstrides[4] + cstrides[5];
      for (std::size_t i = 0; i < ndofs_cell; i++)
      {
        std::size_t block_index = offset_u + i * bs;
        for (std::size_t j = 0; j < bs; j++)
        {
          tr_u += c[block_index + j] * tr(i, j);
          epsn_u += c[block_index + j] * epsn(i, j);
        }
      }

      double sign_u = lmbda * tr_u * n_dot + mu * epsn_u;
      double Pn_u = dolfinx_contact::dR_minus(gap + gamma * sign_u);

      // Fill contributions of facet with itself
      const double w0 = weights[q] * detJ;
      for (std::size_t j = 0; j < ndofs_cell; j++)
      {
        for (std::size_t l = 0; l < bs; l++)
        {
          double sign_du = (lmbda * tr(j, l) * n_dot + mu * epsn(j, l));
          double Pn_du
              = (gamma * sign_du - phi_f(q, j) * n_surf(l)) * Pn_u * w0;

          sign_du *= w0;
          for (std::size_t i = 0; i < ndofs_cell; i++)
          {
            for (std::size_t b = 0; b < bs; b++)
            {
              double v_dot_nsurf = n_surf(b) * phi_f(q, i);
              double sign_v = (lmbda * tr(i, b) * n_dot + mu * epsn(i, b));
              double Pn_v = theta * sign_v - gamma_inv * v_dot_nsurf;
              A[0][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                  += 0.5 * Pn_du * Pn_v;

              // entries corresponding to u and v on the other surface
              for (std::size_t k = 0; k < num_links; k++)
              {
                std::size_t index = 3 + cstrides[3] + cstrides[4]
                                    + k * num_q_points * ndofs_cell * bs
                                    + j * num_q_points * bs + q * bs + l;
                double du_n_opp = c[index] * n_surf(l);

                du_n_opp *= w0 * Pn_u;
                index = 3 + cstrides[3] + cstrides[4]
                        + k * num_q_points * ndofs_cell * bs
                        + i * num_q_points * bs + q * bs + b;
                double v_n_opp = c[index] * n_surf(b);
                A[3 * k + 1][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                    += 0.5 * du_n_opp * Pn_v;
                A[3 * k + 2][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                    += 0.5 * gamma_inv * Pn_du * v_n_opp;
                A[3 * k + 3][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                    += 0.5 * gamma_inv * du_n_opp * v_n_opp;
              }
            }
          }
        }
      }
    }
  };
  switch (type)
  {
  case dolfinx_contact::Kernel::Rhs_variable_gap:
    return unbiased_rhs;
  case dolfinx_contact::Kernel::Jac_variable_gap:
    return unbiased_jac;
  default:
    throw std::runtime_error("Unrecognized kernel");
  }
}
