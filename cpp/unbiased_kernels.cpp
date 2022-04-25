// Copyright (C) 2022 Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT
#include "unbiased_kernels.h"
#include <xtensor/xio.hpp>

contact_kernel_fn dolfinx_contact::generate_kernel(
    dolfinx_contact::Kernel type,
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
    std::shared_ptr<dolfinx::fem::FunctionSpace> V,
    std::vector<xt::xarray<double>>& qp_ref_facet,
    std::vector<std::vector<double>>& qw_ref_facet, std::size_t max_links)
{
  // Extract mesh datad
  assert(mesh);
  const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
  const std::size_t tdim = mesh->topology().dim(); // topological dimension
  const dolfinx::fem::CoordinateElement& cmap = mesh->geometry().cmap();
  bool affine = cmap.is_affine();
  const int num_coordinate_dofs = cmap.dim();

  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  if (const bool needs_dof_transformations
      = element->needs_dof_transformations();
      needs_dof_transformations)
  {
    throw std::invalid_argument("Contact-kernels are not supporting finite "
                                "elements requiring dof transformations.");
  }

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  const std::size_t ndofs_cell = dofmap->element_dof_layout().num_dofs();
  const std::size_t bs = dofmap->bs();
  if (bs != gdim)
  {
    throw std::invalid_argument(
        "The geometric dimension of the mesh is not equal to the block size "
        "of the function space.");
  }

  // NOTE: Assuming same number of quadrature points on each cell
  const std::size_t num_q_points = qp_ref_facet[0].shape(0);

  // Structures needed for basis function tabulation
  // phi and grad(phi) at quadrature points
  const std::size_t num_facets = dolfinx::mesh::cell_num_entities(
      mesh->topology().cell_type(), tdim - 1);
  assert(num_facets == qp_ref_facet.size());

  std::vector<xt::xtensor<double, 2>> phi;
  phi.reserve(num_facets);
  std::vector<xt::xtensor<double, 3>> dphi;
  phi.reserve(num_facets);
  std::vector<xt ::xtensor<double, 3>> dphi_c;
  dphi_c.reserve(num_facets);

  // Temporary structures used in loop
  xt::xtensor<double, 4> cell_tab(
      {(std::size_t)tdim + 1, num_q_points, ndofs_cell, bs});
  xt::xtensor<double, 2> phi_i({num_q_points, ndofs_cell});
  xt::xtensor<double, 3> dphi_i({(std::size_t)tdim, num_q_points, ndofs_cell});
  std::array<std::size_t, 4> tabulate_shape
      = cmap.tabulate_shape(1, num_q_points);
  xt::xtensor<double, 4> c_tab(tabulate_shape);
  assert(tabulate_shape[0] - 1 == (std::size_t)tdim);
  xt::xtensor<double, 3> dphi_ci(
      {tabulate_shape[0] - 1, tabulate_shape[1], tabulate_shape[2]});

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

                  dphi_i = xt::view(cell_tab, xt::range(1, tabulate_shape[0]),
                                    xt::all(), xt::all(), 0);
                  dphi.push_back(dphi_i);

                  // Tabulate coordinate element of reference cell
                  cmap.tabulate(1, q_facet, c_tab);
                  dphi_ci = xt::view(c_tab, xt::range(1, tabulate_shape[0]),
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
  std::vector<std::size_t> cstrides = {
      1,                                                       // mu
      1,                                                       // lambda
      1,                                                       // h
      num_q_points * gdim,                                     // gap
      num_q_points * gdim,                                     // normals
      (tdim + 1) * num_q_points * ndofs_cell * bs * max_links, // test_fn
      ndofs_cell * bs,                                         // u
      num_q_points * bs * (tdim + 1),                          // u_opposite
      num_q_points * tdim * gdim, // 1st derivative of transformation
      num_q_points * (tdim + 1) * tdim * gdim
          / 2 // 2nd derivative of transformation
  };

  // create offsets
  std::vector<int32_t> offsets(11, 0);
  offsets[0] = 0;
  std::partial_sum(cstrides.cbegin(), cstrides.cend(),
                   std::next(offsets.begin()));
  // As reference facet and reference cell are affine, we do not need to
  // compute this per quadrature point
  basix::cell::type basix_cell
      = dolfinx::mesh::cell_type_to_basix_type(mesh->topology().cell_type());
  xt::xtensor<double, 3> ref_jacobians
      = basix::cell::facet_jacobians(basix_cell);

  // Get facet normals on reference cell
  xt::xtensor<double, 2> facet_normals
      = basix::cell::facet_outward_normals(basix_cell);

  // Get update Jacobian function (for each quadrature point)
  auto update_jacobian
      = dolfinx_contact::get_update_jacobian_dependencies(cmap);

  // Get update FacetNormal function (for each quadrature point)
  auto update_normal = dolfinx_contact::get_update_normal(cmap);
  // right hand side kernel

  /// @brief Assemble kernel for RHS of unbiased contact problem with variable
  /// gap
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
  /// @param[in] entity_local_index local facet index
  /// @param[in] num_links number of facets linked by closest point projection
  /// @param[in] facet_indices indices of linked facets
  contact_kernel_fn unbiased_rhs
      = [=](std::vector<std::vector<PetscScalar>>& b, const double* c,
            const double* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::size_t num_links,
            const std::int32_t* facet_indices)
  {
    // assumption that the vector function space has block size tdim
    assert(bs == gdim);
    const auto facet_index = size_t(*entity_local_index);

    // NOTE: DOLFINx has 3D input coordinate dofs
    // FIXME: These array should be views (when compute_jacobian doesn't use
    // xtensor)
    std::array<std::size_t, 2> shape = {(std::size_t)num_coordinate_dofs, 3};
    xt::xtensor<double, 2> coord = xt::adapt(
        coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);

    // Extract the first derivative of the coordinate element (cell) of
    // degrees of freedom on the facet
    const xt::xtensor<double, 3>& dphi_fc = dphi_c[facet_index];

    // Create data structures for jacobians
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, (std::size_t)tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({(std::size_t)tdim, gdim});
    // J_f facet jacobian, J_tot = J * J_f
    xt::xtensor<double, 2> J_f
        = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
    xt::xtensor<double, 2> J_tot
        = xt::zeros<double>({J.shape(0), J_f.shape(1)});
    double detJ = 0;
    auto c_view = xt::view(coord, xt::all(), xt::range(0, gdim));

    // Normal vector on physical facet at a single quadrature point
    xt::xtensor<double, 1> n_phys = xt::zeros<double>({gdim});

    // Pre-compute jacobians and normals for affine meshes
    if (affine)
    {
      detJ = dolfinx_contact::compute_facet_jacobians(0, J, K, J_tot, J_f,
                                                      dphi_fc, coord);
      dolfinx_contact::physical_facet_normal(
          n_phys, K, xt::row(facet_normals, facet_index));
    }

    // Extract constants used inside quadrature loop
    double gamma = c[2] / w[0];     // h/gamma
    double gamma_inv = w[0] / c[2]; // gamma/h
    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];

    // Extract reference to the tabulated basis function at the local
    // facet
    const xt::xtensor<double, 2>& phi_f = phi[facet_index];
    const xt::xtensor<double, 3>& dphi_f = dphi[facet_index];

    // Extract reference to quadrature weights for the local facet
    const std::vector<double>& weights = qw_ref_facet[facet_index];

    // Temporary data structures used inside quadrature loop
    std::array<double, 3> n_surf = {0, 0, 0};
    xt::xtensor<double, 2> tr({ndofs_cell, gdim});
    xt::xtensor<double, 2> epsn({ndofs_cell, gdim});
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      // Update Jacobian and physical normal
      detJ = update_jacobian(q, detJ, J, K, J_tot, J_f, dphi_fc, coord);
      update_normal(n_phys, K, facet_normals, facet_index);
      double n_dot = 0;
      double gap = 0;
      // For closest point projection the gap function is given by
      // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
      // in y = Pi(x)
      for (std::size_t i = 0; i < gdim; i++)
      {

        n_surf[i] = -c[offsets[4] + q * gdim + i];
        n_dot += n_phys(i) * n_surf[i];
        gap += c[offsets[3] + q * gdim + i] * n_surf[i];
      }

      // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2
      std::fill(tr.begin(), tr.end(), 0.0);
      std::fill(epsn.begin(), epsn.end(), 0.0);
      for (std::size_t j = 0; j < ndofs_cell; j++)
      {
        for (std::size_t l = 0; l < bs; l++)
        {
          for (std::size_t k = 0; k < tdim; k++)
          {
            tr(j, l) += K(k, l) * dphi_f(k, q, j);
            for (std::size_t s = 0; s < gdim; s++)
            {
              epsn(j, l) += K(k, s) * dphi_f(k, q, j)
                            * (n_phys(s) * n_surf[l] + n_phys(l) * n_surf[s]);
            }
          }
        }
      }
      // compute tr(eps(u)), epsn at q
      double tr_u = 0;
      double epsn_u = 0;
      for (std::size_t i = 0; i < ndofs_cell; i++)
      {
        std::size_t block_index = offsets[6] + i * bs;
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
          double v_dot_nsurf = n_surf[n] * phi_f(q, i);
          double sign_v = (lmbda * tr(i, n) * n_dot + mu * epsn(i, n));
          // This is (1./gamma)*Pn_v to avoid the product gamma*(1./gamma)
          double Pn_v = theta * sign_v - gamma_inv * v_dot_nsurf;
          b[0][n + i * bs] += 0.5 * Pn_u * Pn_v;

          // entries corresponding to v on the other surface
          for (std::size_t k = 0; k < num_links; k++)
          {
            std::size_t index = offsets[5] + k * num_q_points * ndofs_cell * bs
                                + i * num_q_points * bs + q * bs + n;

            double v_n_opp = c[index] * n_surf[n];

            b[k + 1][n + i * bs] += 0.5 * gamma_inv * v_n_opp * Pn_u;
          }
        }
      }
    }
  };

  /// @brief Assemble kernel for Jacobian (LHS) of unbiased contact
  /// problem with variable gap
  ///
  /// Assemble of the residual of the unbiased contact problem into matrix
  /// `A`.
  /// @param[in,out] A The matrix to assemble the Jacobian into
  /// @param[in] c The coefficients used in kernel. Assumed to be
  /// ordered as mu, lmbda, h, gap, normals, test_fn, u, u_opposite,`1st
  /// derivative of transformation`, `2nd derivative of transformation`.
  /// @param[in] w The constants used in kernel. Assumed to be ordered as
  /// `gamma`, `theta`.
  /// @param[in] coordinate_dofs The physical coordinates of cell. Assumed
  /// to be padded to 3D, (shape (num_nodes, 3)).
  /// @param[in] entity_local_index local facet index
  /// @param[in] num_links number of facets linked by closest point projection
  /// @param[in] facet_indices indices of linked facets
  contact_kernel_fn unbiased_jac
      = [=](std::vector<std::vector<PetscScalar>>& A, const double* c,
            const double* w, const double* coordinate_dofs,
            const int* entity_local_index, const std::size_t num_links,
            const std::int32_t* facet_indices)
  {
    // assumption that the vector function space has block size tdim
    assert(bs == gdim);
    const auto facet_index = std::size_t(*entity_local_index);

    // Reshape coordinate dofs to two dimensional array
    // NOTE: DOLFINx has 3D input coordinate dofs
    std::array<std::size_t, 2> shape = {(std::size_t)num_coordinate_dofs, 3};

    // FIXME: These array should be views (when compute_jacobian doesn't
    // use xtensor)
    xt::xtensor<double, 2> coord = xt::adapt(
        coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);

    // Extract the first derivative of the coordinate element (cell) of
    // degrees of freedom on the facet
    const xt::xtensor<double, 3>& dphi_fc = dphi_c[facet_index];

    // Create data structures for jacobians
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, (std::size_t)tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({(std::size_t)tdim, gdim});
    // J_f facet jacobian, J_tot = J * J_f
    xt::xtensor<double, 2> J_f
        = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
    xt::xtensor<double, 2> J_tot
        = xt::zeros<double>({J.shape(0), J_f.shape(1)});
    double detJ = 0;
    auto c_view = xt::view(coord, xt::all(), xt::range(0, gdim));

    // Normal vector on physical facet at a single quadrature point
    xt::xtensor<double, 1> n_phys = xt::zeros<double>({gdim});

    // Pre-compute jacobians and normals for affine meshes
    if (affine)
    {
      detJ = dolfinx_contact::compute_facet_jacobians(0, J, K, J_tot, J_f,
                                                      dphi_fc, coord);
      dolfinx_contact::physical_facet_normal(
          n_phys, K, xt::row(facet_normals, facet_index));
    }

    // Extract scaled gamma (h/gamma) and its inverse
    double gamma = c[2] / w[0];
    double gamma_inv = w[0] / c[2];

    double theta = w[1];
    double mu = c[0];
    double lmbda = c[1];

    // Extract reference to the tabulated basis function at the local
    // facet
    const xt::xtensor<double, 2>& phi_f = phi[facet_index];
    const xt::xtensor<double, 3>& dphi_f = dphi[facet_index];

    // Extract reference to quadrature weights for the local facet
    const std::vector<double>& weights = qw_ref_facet[facet_index];

    // Temporary data structures used inside quadrature loop
    std::array<double, 3> n_surf = {0, 0, 0};
    xt::xtensor<double, 2> tr({ndofs_cell, gdim});
    xt::xtensor<double, 2> epsn({ndofs_cell, gdim});
    xt::xtensor<double, 2> J_link = xt::zeros<double>(
        {gdim, (std::size_t)tdim}); // jacobian of linked cell to ref cell
    xt::xtensor<double, 2> K_link // inverse jacobian of linked cell to ref cell
        = xt::zeros<double>({(std::size_t)tdim, gdim});
    xt::xtensor<double, 2> J_tot_link = xt::zeros<double>(
        {J_link.shape(0),
         J_link.shape(1) - 1});    // jacobian of linked facet to ref facet
    xt::xarray<double> du_tan(bs); // tangential derivative of trial function
    xt::xarray<double> du_tan_opp(
        bs); // tangential derivative of trial function at closest point
    xt::xarray<double> n_dot_grad(gdim); // -n_y * grad(v)(Y), v - test funciton
    xt::xarray<double> grad_v(gdim);     // grad(v), v -test function
    xt::xtensor<double, 2> def_grad({gdim, gdim}); // deformation gradient
    xt::xtensor<double, 2> def_grad_inv(
        {gdim, gdim}); // inverse of deformation gradient
    for (std::size_t q = 0; q < weights.size(); q++)
    {
      // Update Jacobian and physical normal
      detJ = update_jacobian(q, detJ, J, K, J_tot, J_f, dphi_fc, coord);
      update_normal(n_phys, K, facet_normals, facet_index);
      double n_dot = 0;
      double gap = 0;
      // For closest point projection the gap function is given by
      // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
      // in y = Pi(x)
      for (std::size_t i = 0; i < gdim; i++)
      {
        // For closest point projection the gap function is given by
        // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
        // in y = Pi(x)
        n_surf[i] = -c[offsets[4] + q * gdim + i];
        n_dot += n_phys(i) * n_surf[i];
        gap += c[offsets[3] + q * gdim + i] * n_surf[i];
        for (std::size_t j = 0; j < (std::size_t)tdim; j++)
          J_link(i, j) = c[offsets[8] + q * tdim * gdim + i * tdim + j];
      }
      dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J_link, K_link);

      // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2
      std::fill(tr.begin(), tr.end(), 0.0);
      std::fill(epsn.begin(), epsn.end(), 0.0);
      for (std::size_t j = 0; j < ndofs_cell; j++)
      {
        for (std::size_t l = 0; l < bs; l++)
        {
          for (std::size_t k = 0; k < tdim; k++)
          {
            tr(j, l) += K(k, l) * dphi_f(k, q, j);
            for (std::size_t s = 0; s < gdim; s++)
            {
              epsn(j, l) += K(k, s) * dphi_f(k, q, j)
                            * (n_phys(s) * n_surf[l] + n_phys(l) * n_surf[s]);
            }
          }
        }
      }

      // compute tr(eps(u)), epsn at q
      double tr_u = 0;
      double epsn_u = 0;
      for (std::size_t i = 0; i < ndofs_cell; i++)
      {
        std::size_t block_index = offsets[6] + i * bs;
        for (std::size_t j = 0; j < bs; j++)
        {
          tr_u += c[block_index + j] * tr(i, j);
          epsn_u += c[block_index + j] * epsn(i, j);
        }
      }

      double sign_u = lmbda * tr_u * n_dot + mu * epsn_u;
      const double w0 = weights[q] * detJ;
      double dPn_u = dolfinx_contact::dR_minus(gap + gamma * sign_u);
      double Pn_u = dolfinx_contact::R_minus(gap + gamma * sign_u) * w0;

      // extract deformation gradient
      def_grad.fill(0);
      for (std::size_t j = 0; j < bs; ++j)
      {

        def_grad(j, j) += 1;
        for (std::size_t l = 0; l < gdim; ++l)
        {

          for (std::size_t f = 0; f < tdim; ++f)
          {
            std::size_t index_u_opp_grad
                = offsets[7] + (f + 1) * num_q_points * bs + q * bs + j;

            def_grad(j, l) += K_link(f, l) * c[index_u_opp_grad];
          }
        }
      }
      // compute inverse of deformation gradient
      dolfinx::fem::CoordinateElement::compute_jacobian_inverse(def_grad,
                                                                def_grad_inv);

      // Fill contributions of facet with itself

      for (std::size_t l = 0; l < bs; l++)
      {
        for (std::size_t j = 0; j < ndofs_cell; j++)
        {
          double sign_du = (lmbda * tr(j, l) * n_dot + mu * epsn(j, l));
          double du_dot_nsurf = phi_f(q, j) * n_surf[l];
          double Pn_du = (gamma * sign_du - du_dot_nsurf) * dPn_u * w0;
          du_tan.fill(0);
          du_tan(l) = phi_f(q, j);
          for (std::size_t r = 0; r < bs; ++r)
            du_tan(r) -= n_surf[r] * du_dot_nsurf;

          sign_du *= w0;
          for (std::size_t i = 0; i < ndofs_cell; i++)
          {
            for (std::size_t b = 0; b < bs; b++)
            {
              double v_dot_nsurf = n_surf[b] * phi_f(q, i);
              double sign_v = (lmbda * tr(i, b) * n_dot + mu * epsn(i, b));
              double Pn_v = theta * sign_v - gamma_inv * v_dot_nsurf;

              A[0][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                  += 0.5 * Pn_du * Pn_v;

              // entries corresponding to u and v on the other surface
              for (std::size_t k = 0; k < num_links; k++)
              {
                std::size_t index_u = offsets[5]
                                      + k * num_q_points * ndofs_cell * bs
                                      + j * num_q_points * bs + q * bs + l;
                std::size_t index_v = offsets[5]
                                      + k * num_q_points * ndofs_cell * bs
                                      + i * num_q_points * bs + q * bs + b;
                double du_n_opp = c[index_u] * n_surf[l];

                du_tan_opp.fill(0);
                du_tan_opp(l) = c[index_u];
                for (std::size_t f = 0; f < bs; ++f)
                  du_tan(f) -= n_surf[f] * du_n_opp;

                double v_n_opp = c[index_v] * n_surf[b];

                // extract grad(v) - v test functions
                grad_v.fill(0);
                std::size_t offset_grad
                    = offsets[5] + ndofs_cell * num_q_points * max_links * bs;
                for (std::size_t r = 0; r < gdim; ++r)
                  for (std::size_t s = 0; s < (std::size_t)tdim; ++s)
                  {
                    std::size_t index_v_grad
                        = offset_grad
                          + s * ndofs_cell * num_q_points * max_links * bs
                          + k * ndofs_cell * num_q_points * bs
                          + i * num_q_points * bs + q * bs + b;
                    grad_v(r) += K_link(s, r) * c[index_v_grad];
                  }

                // compute (-n_y)^T grad(v)
                n_dot_grad.fill(0.0);
                for (std::size_t r = 0; r < bs; ++r)
                  n_dot_grad(r) = n_surf[b] * grad_v(r);
                du_n_opp *= w0 * dPn_u;

                // compute -n_y * D_u v(Y)
                double grad_v_u_tan = 0;
                double grad_v_u_tan_opp = 0;
                for (std::size_t r = 0; r < gdim; ++r)
                  for (std::size_t s = 0; s < gdim; ++s)

                  {
                    grad_v_u_tan
                        += n_dot_grad(s) * def_grad_inv(s, r) * du_tan(r);
                    grad_v_u_tan_opp
                        += n_dot_grad(s) * def_grad_inv(s, r) * du_tan_opp(r);
                  }

                A[3 * k + 1][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                    += 0.5 * du_n_opp * Pn_v;
                A[3 * k + 2][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                    += 0.5 * gamma_inv
                       * (Pn_du * v_n_opp - Pn_u * grad_v_u_tan);
                A[3 * k + 3][(b + i * bs) * ndofs_cell * bs + l + j * bs]
                    += 0.5 * gamma_inv
                       * (du_n_opp * v_n_opp + Pn_u * grad_v_u_tan_opp);
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
