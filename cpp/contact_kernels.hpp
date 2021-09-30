// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx_cuas/QuadratureRule.hpp>
#include <dolfinx_cuas/math.hpp>
#include <dolfinx_cuas/utils.hpp>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>

using kernel_fn = std::function<void(double *, const double *, const double *, const double *,
                                     const int *, const std::uint8_t *)>;

namespace dolfinx_contact
{
  enum Kernel
  {
    NitscheRigidSurfaceRhs,
    NitscheRigidSurfaceJac
  };

  kernel_fn generate_contact_kernel(
      std::shared_ptr<const dolfinx::fem::FunctionSpace> V, Kernel type,
      dolfinx_cuas::QuadratureRule &quadrature_rule,
      std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>> coeffs,
      bool constant_normal)
  {

    auto mesh = V->mesh();

    // Get mesh info
    const int gdim = mesh->geometry().dim(); // geometrical dimension
    const int tdim = mesh->topology().dim(); // topological dimension
    const int fdim = tdim - 1;               // topological dimension of facet

    // Create coordinate elements (for facet and cell)
    const basix::FiniteElement surface_element = dolfinx_cuas::mesh_to_basix_element(mesh, fdim);
    const basix::FiniteElement basix_element = dolfinx_cuas::mesh_to_basix_element(mesh, tdim);
    const int num_coordinate_dofs = basix_element.dim();

    // Create quadrature points on reference facet
    xt::xarray<double> &q_weights = quadrature_rule.weights_ref();
    xt::xarray<double> &qp_ref_facet = quadrature_rule.points_ref();

    // Tabulate coordinate element of reference facet (used to compute Jacobian on
    // facet) and push forward quadrature points
    auto f_tab = surface_element.tabulate(0, qp_ref_facet);
    xt::xtensor<double, 2> phi_f = xt::view(f_tab, 0, xt::all(), xt::all(), 0);

    // Structures required for pushing forward quadrature points
    auto facets = basix::cell::topology(basix_element.cell_type())[tdim - 1];          // Topology of basix facets
    const xt::xtensor<double, 2> x = basix::cell::geometry(basix_element.cell_type()); // Geometry of basix cell
    const std::uint32_t num_facets = facets.size();
    const std::uint32_t num_quadrature_pts = qp_ref_facet.shape(0);

    // Structures needed for basis function tabulation
    // phi and grad(phi) at quadrature points
    std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
    int bs = element->block_size();
    std::uint32_t ndofs_cell = element->space_dimension() / bs;
    xt::xtensor<double, 3> phi({num_facets, num_quadrature_pts, ndofs_cell});
    xt::xtensor<double, 4> dphi({num_facets, tdim, num_quadrature_pts, ndofs_cell});
    xt::xtensor<double, 4> cell_tab({tdim + 1, num_quadrature_pts, ndofs_cell, bs});

    // Structure needed for Jacobian of cell basis function
    xt::xtensor<double, 4> dphi_c({num_facets, tdim, num_quadrature_pts, basix_element.dim()});

    // Structures for coefficient data
    int num_coeffs = coeffs.size();
    std::vector<int> offsets(num_coeffs + 3);
    offsets[0] = 0;
    for (int i = 1; i < num_coeffs + 1; i++)
    {
      std::shared_ptr<const dolfinx::fem::FiniteElement> coeff_element = coeffs[i - 1]->function_space()->element();
      offsets[i] = offsets[i - 1] + coeff_element->space_dimension() / coeff_element->block_size();
    }
    offsets[num_coeffs + 1] = offsets[num_coeffs] + num_facets;
    offsets[num_coeffs + 2] = offsets[num_coeffs + 1] + gdim * num_quadrature_pts * num_facets;
    // Pack coefficients for functions and gradients of functions (untested)
    xt::xtensor<double, 3> phi_coeffs({num_facets, q_weights.size(), offsets[num_coeffs]});
    xt::xtensor<double, 4> dphi_coeffs({num_facets, tdim, q_weights.size(), offsets[num_coeffs]});
    for (int i = 0; i < num_facets; ++i)
    {
      // Push quadrature points forward
      auto facet = facets[i];
      auto coords = xt::view(x, xt::keep(facet), xt::all());
      auto q_facet = xt::linalg::dot(phi_f, coords);

      // Tabulate at quadrature points on facet
      auto phi_i = xt::view(phi, i, xt::all(), xt::all());
      element->tabulate(cell_tab, q_facet, 1);
      phi_i = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
      auto dphi_i = xt::view(dphi, i, xt::all(), xt::all(), xt::all());
      dphi_i = xt::view(cell_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);

      // Tabulate coordinate element of reference cell
      auto c_tab = basix_element.tabulate(1, q_facet);
      auto dphi_ci = xt::view(dphi_c, i, xt::all(), xt::all(), xt::all());
      dphi_ci = xt::view(c_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);
      // Create Finite elements for coefficient functions and tabulate shape functions
      for (int j = 0; j < num_coeffs; j++)
      {
        std::shared_ptr<const dolfinx::fem::FiniteElement> coeff_element = coeffs[j]->function_space()->element();
        xt::xtensor<double, 4> coeff_basis(
            {tdim + 1, q_weights.size(),
             coeff_element->space_dimension() / coeff_element->block_size(), 1});
        coeff_element->tabulate(coeff_basis, q_facet, 1);
        auto phi_ij = xt::view(phi_coeffs, i, xt::all(), xt::range(offsets[j], offsets[j + 1]));
        phi_ij = xt::view(coeff_basis, 0, xt::all(), xt::all(), 0);
        auto dphi_ij = xt::view(dphi_coeffs, i, xt::all(), xt::all(), xt::range(offsets[j], offsets[j + 1]));
        dphi_ij = xt::view(coeff_basis, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);
      }
    }

    // As reference facet and reference cell are affine, we do not need to compute this per
    // quadrature point
    auto ref_jacobians = basix::cell::facet_jacobians(basix_element.cell_type());

    // Get facet normals on reference cell
    auto facet_normals = basix::cell::facet_outward_normals(basix_element.cell_type());

    // Define kernels
    // RHS for contact with rigid surface
    // =====================================================================================
    kernel_fn nitsche_rigid_rhs = [dphi_c, phi, dphi, phi_coeffs, dphi_coeffs, offsets, num_coeffs, gdim, tdim, fdim,
                                   q_weights, num_coordinate_dofs, ref_jacobians, bs, facet_normals, constant_normal](
                                      double *b, const double *c, const double *w, const double *coordinate_dofs,
                                      const int *entity_local_index, const std::uint8_t *quadrature_permutation)
    {
      // assumption that the vector function space has block size tdim
      assert(bs == gdim);
      // assumption that u lives in the same space as v
      assert(phi.shape(2) == offsets[1] - offsets[0]);

      std::size_t facet_index = size_t(*entity_local_index);

      // Reshape coordinate dofs to two dimensional array
      // NOTE: DOLFINx has 3D input coordinate dofs
      std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};

      // FIXME: These array should be views (when compute_jacobian doesn't use xtensor)
      xt::xtensor<double, 2> coord = xt::adapt(coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);

      // Extract the first derivative of the coordinate element (cell) of degrees of freedom on
      // the facet
      xt::xtensor<double, 2> dphi0_c = xt::view(dphi_c, facet_index, xt::all(), 0,
                                                xt::all()); // FIXME: Assumed constant, i.e. only works for simplices

      // NOTE: Affine cell assumption
      // Compute Jacobian and determinant at first quadrature point
      xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
      xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});
      dolfinx_cuas::math::compute_jacobian(dphi0_c, coord, J);
      dolfinx_cuas::math::compute_inv(J, K);

      // Compute normal of physical facet using a normalized covariant Piola transform
      // n_phys = J^{-T} n_ref / ||J^{-T} n_ref||
      // See for instance DOI: 10.1137/08073901X
      xt::xarray<double> n_phys = xt::zeros<double>({gdim});
      auto facet_normal = xt::row(facet_normals, facet_index);
      for (std::size_t i = 0; i < gdim; i++)
        for (std::size_t j = 0; j < tdim; j++)
          n_phys[i] += K(j, i) * facet_normal[j];
      n_phys /= xt::linalg::norm(n_phys);

      // Retrieve normal of rigid surface if constant
      xt::xarray<double> n_surf = xt::zeros<double>({gdim});
      if (constant_normal)
      {
        for (int i = 0; i < gdim; i++)
          n_surf(i) = w[i + 2];
      }
      int c_offset = (bs - 1) * offsets[1];
      double gamma = w[0] / c[c_offset + offsets[3] + facet_index]; // This is gamma/h
      double theta = w[1];

      // If surface normal constant precompute (n_phys * n_surf)
      double n_dot = 0;
      if (constant_normal)
      {
        for (int i = 0; i < gdim; i++)
          n_dot += n_phys(i) * n_surf(i);
      }

      // Compute det(J_C J_f) as it is the mapping to the reference facet
      xt::xtensor<double, 2> J_f = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
      xt::xtensor<double, 2> J_tot = xt::linalg::dot(J, J_f);
      double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_tot));

      // Get number of dofs per cell
      // FIXME: Should be templated
      std::int32_t ndofs_cell = phi.shape(2);
      // Temporary variable for grad(phi) on physical cell
      xt::xtensor<double, 2> dphi_phys({bs, ndofs_cell});

      // Loop over quadrature points
      int num_points = phi.shape(1);
      for (std::size_t q = 0; q < num_points; q++)
      {

        double mu = 0;
        for (int j = offsets[1]; j < offsets[2]; j++)
          mu += c[j + c_offset] * phi_coeffs(facet_index, q, j);
        double lmbda = 0;
        for (int j = offsets[2]; j < offsets[3]; j++)
          lmbda += c[j + c_offset] * phi_coeffs(facet_index, q, j);
        double gap = 0;
        int gap_offset = c_offset + offsets[4] + facet_index * num_points * gdim;
        // if normal not constant, get surface normal at current quadrature point
        if (!constant_normal)
        {
          n_dot = 0;
          for (int i = 0; i < gdim; i++)
          {
            gap += c[gap_offset + q * gdim + i] * c[gap_offset + q * gdim + i];
            n_surf(i) = c[gap_offset + q * gdim + i];
            n_dot += n_phys(i) * n_surf(i);
          }
          gap = std::sqrt(gap);
          if (gap > 1e-13)
          {
            n_surf /= gap;
            n_dot /= gap;
          }
          else
          {
            n_surf = n_phys;
            n_dot = 1;
          }
        }
        else
        {
          for (int i = 0; i < gdim; i++)
          {
            gap += c[gap_offset + q * gdim + i] * n_surf(i);
          }
        }

        xt::xtensor<double, 2> tr = xt::zeros<double>({offsets[1] - offsets[0], gdim});
        xt::xtensor<double, 2> epsn = xt::zeros<double>({offsets[1] - offsets[0], gdim});
        xt::xtensor<double, 2> v_dot_nsurf = xt::zeros<double>({offsets[1] - offsets[0], gdim});
        // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2, ufl.dot(v, n_surf)
        for (int j = 0; j < offsets[1] - offsets[0]; j++)
        {
          for (int l = 0; l < bs; l++)
          {
            for (int k = 0; k < tdim; k++)
            {
              tr(j, l) += K(k, l) * dphi(facet_index, k, q, j);
              for (int s = 0; s < gdim; s++)
              {
                epsn(j, l) += K(k, s) * dphi(facet_index, k, q, j) * (n_phys(s) * n_surf(l) + n_phys(l) * n_surf(s));
              }
            }
          }
        }
        // compute tr(eps(u)), epsn at q
        double tr_u = 0;
        double epsn_u = 0;
        double u_dot_nsurf = 0;
        for (int i = 0; i < offsets[1] - offsets[0]; i++)
        {
          const std::int32_t block_index = (i + offsets[0]) * bs;
          for (int j = 0; j < bs; j++)
          {
            tr_u += c[block_index + j] * tr(i, j);
            epsn_u += c[block_index + +j] * epsn(i, j);
            u_dot_nsurf += c[block_index + j] * n_surf(j) * phi(facet_index, q, i);
          }
        }

        // Multiply  by weight
        double sign_u = (lmbda * n_dot * tr_u + mu * epsn_u);
        double R_minus = 1. / gamma * sign_u + (gap + u_dot_nsurf);
        if (R_minus > 0)
          R_minus = 0;
        else
          R_minus = R_minus * detJ * q_weights[q];
        sign_u *= -theta / gamma * detJ * q_weights[q];
        for (int j = 0; j < ndofs_cell; j++)
        {
          // Insert over block size in matrix
          for (int l = 0; l < bs; l++)
          {
            double sign_v = lmbda * tr(j, l) * n_dot + mu * epsn(j, l);
            double v_dot_nsurf = n_surf(l) * phi(facet_index, q, j);
            b[j * bs + l] += sign_v * sign_u + R_minus * (theta * sign_v + gamma * v_dot_nsurf);
          }
        }
      }
    };

    // Jacobian for contact with rigid surface
    // =====================================================================================
    kernel_fn nitsche_rigid_jacobian = [dphi_c, phi, dphi, phi_coeffs, dphi_coeffs, offsets, num_coeffs, gdim, tdim, fdim,
                                        q_weights, num_coordinate_dofs, ref_jacobians, bs, facet_normals, constant_normal](
                                           double *A, const double *c, const double *w, const double *coordinate_dofs,
                                           const int *entity_local_index, const std::uint8_t *quadrature_permutation)
    {
      // assumption that the vector function space has block size tdim
      assert(bs == gdim);
      // assumption that u lives in the same space as v
      assert(phi.shape(2) == offsets[1] - offsets[0]);

      std::size_t facet_index = size_t(*entity_local_index);

      // Reshape coordinate dofs to two dimensional array
      // NOTE: DOLFINx has 3D input coordinate dofs
      std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};

      // FIXME: These array should be views (when compute_jacobian doesn't use xtensor)
      xt::xtensor<double, 2> coord = xt::adapt(coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);

      // Extract the first derivative of the coordinate element (cell) of degrees of freedom on
      // the facet
      xt::xtensor<double, 2> dphi0_c = xt::view(dphi_c, facet_index, xt::all(), 0,
                                                xt::all()); // FIXME: Assumed constant, i.e. only works for simplices

      // NOTE: Affine cell assumption
      // Compute Jacobian and determinant at first quadrature point
      xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
      xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});
      dolfinx_cuas::math::compute_jacobian(dphi0_c, coord, J);
      dolfinx_cuas::math::compute_inv(J, K);

      // Compute normal of physical facet using a normalized covariant Piola transform
      // n_phys = J^{-T} n_ref / ||J^{-T} n_ref||
      // See for instance DOI: 10.1137/08073901X
      xt::xarray<double> n_phys = xt::zeros<double>({gdim});
      auto facet_normal = xt::row(facet_normals, facet_index);
      for (std::size_t i = 0; i < gdim; i++)
        for (std::size_t j = 0; j < tdim; j++)
          n_phys[i] += K(j, i) * facet_normal[j];
      n_phys /= xt::linalg::norm(n_phys);

      // Retrieve normal of rigid surface if constant
      xt::xarray<double> n_surf = xt::zeros<double>({gdim});
      // FIXME: Code duplication from previous kernel, and should be made into a lambda function
      if (constant_normal)
      {
        for (int i = 0; i < gdim; i++)
          n_surf(i) = w[i + 2];
      }
      int c_offset = (bs - 1) * offsets[1];
      double gamma = w[0] / c[c_offset + offsets[3] + facet_index]; // This is gamma/hdouble gamma = w[0];
      double theta = w[1];

      // If surface normal constant precompute (n_phys * n_surf)
      double n_dot = 0;
      if (constant_normal)
      {
        for (int i = 0; i < gdim; i++)
          n_dot += n_phys(i) * n_surf(i);
      }

      // Compute det(J_C J_f) as it is the mapping to the reference facet
      xt::xtensor<double, 2> J_f = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
      xt::xtensor<double, 2> J_tot = xt::linalg::dot(J, J_f);
      double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_tot));

      // Get number of dofs per cell
      // FIXME: Should be templated
      std::int32_t ndofs_cell = phi.shape(2);
      // Temporary variable for grad(phi) on physical cell
      xt::xtensor<double, 2> dphi_phys({bs, ndofs_cell});

      int num_points = phi.shape(1);
      for (std::size_t q = 0; q < num_points; q++)
      {

        xt::xtensor<double, 2> tr = xt::zeros<double>({ndofs_cell, gdim});
        xt::xtensor<double, 2> epsn = xt::zeros<double>({ndofs_cell, gdim});
        // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2
        for (int j = 0; j < ndofs_cell; j++)
        {
          for (int l = 0; l < bs; l++)
          {
            for (int k = 0; k < tdim; k++)
            {
              tr(j, l) += K(k, l) * dphi(facet_index, k, q, j);
              for (int s = 0; s < gdim; s++)
              {
                epsn(j, l) += K(k, s) * dphi(facet_index, k, q, j) * (n_phys(s) * n_surf(l) + n_phys(l) * n_surf(s));
              }
            }
          }
        }
        double mu = 0;
        int c_offset = (bs - 1) * offsets[1];
        for (int j = offsets[1]; j < offsets[2]; j++)
          mu += c[j + c_offset] * phi_coeffs(facet_index, q, j);
        double lmbda = 0;
        for (int j = offsets[2]; j < offsets[3]; j++)
          lmbda += c[j + c_offset] * phi_coeffs(facet_index, q, j);
        double gap = 0;
        int gap_offset = c_offset + offsets[4] + facet_index * num_points * gdim;
        // if normal not constant, get surface normal at current quadrature point
        if (!constant_normal)
        {
          n_dot = 0;
          for (int i = 0; i < gdim; i++)
          {
            gap += c[gap_offset + q * gdim + i] * c[gap_offset + q * gdim + i];
            n_surf(i) = c[gap_offset + q * gdim + i];
            n_dot += n_phys(i) * n_surf(i);
          }
          gap = std::sqrt(gap);
          if (gap > 1e-13)
          {
            n_surf /= gap;
            n_dot /= gap;
          }
          else
          {
            n_surf = n_phys;
            n_dot = 1;
          }
        }
        else
        {
          for (int i = 0; i < gdim; i++)
          {
            gap += c[gap_offset + q * gdim + i] * n_surf(i);
          }
        }
        // compute tr(eps(u)), epsn at q
        double tr_u = 0;
        double epsn_u = 0;
        double u_dot_nsurf = 0;
        for (int i = 0; i < offsets[1] - offsets[0]; i++)
        {
          const std::int32_t block_index = (i + offsets[0]) * bs;
          for (int j = 0; j < bs; j++)
          {
            tr_u += c[block_index + j] * tr(i, j);
            epsn_u += c[block_index + +j] * epsn(i, j);
            u_dot_nsurf += c[block_index + j] * n_surf(j) * phi(facet_index, q, i);
          }
        }
        double temp = (lmbda * n_dot * tr_u + mu * epsn_u) + gamma * (gap + u_dot_nsurf);
        const double w0 = q_weights[q] * detJ;
        for (int j = 0; j < ndofs_cell; j++)
        {
          for (int l = 0; l < bs; l++)
          {
            double sign_u = (lmbda * tr(j, l) * n_dot + mu * epsn(j, l));
            double term2 = 0;
            if (temp < 0)
              term2 = 1. / gamma * (sign_u + gamma * n_surf(l) * phi(facet_index, q, j)) * w0;
            sign_u *= w0;
            for (int i = 0; i < ndofs_cell; i++)
            { // Insert over block size in matrix
              for (int b = 0; b < bs; b++)
              {
                double v_dot_nsurf = n_surf(b) * phi(facet_index, q, i);
                double sign_v = (lmbda * tr(i, b) * n_dot + mu * epsn(i, b));
                A[(b + i * bs) * ndofs_cell * bs + l + j * bs] += -theta / gamma * sign_u * sign_v + term2 * (theta * sign_v + gamma * v_dot_nsurf);
              }
            }
          }
        }
      }
    };
    switch (type)
    {
    case dolfinx_contact::Kernel::NitscheRigidSurfaceRhs:
      return nitsche_rigid_rhs;
    case dolfinx_contact::Kernel::NitscheRigidSurfaceJac:
      return nitsche_rigid_jacobian;
    default:
      throw std::runtime_error("Unrecognized kernel");
    }
  }
} // namespace dolfinx_contact
