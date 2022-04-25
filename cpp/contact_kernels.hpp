// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "Contact.h"
#include "geometric_quantities.h"
#include <basix/cell.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx_cuas/QuadratureRule.hpp>
#include <dolfinx_cuas/kernels.hpp>
#include <dolfinx_cuas/utils.hpp>

namespace dolfinx_contact
{
template <typename T>
dolfinx_cuas::kernel_fn<T> generate_contact_kernel(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V, Kernel type,
    dolfinx_cuas::QuadratureRule& quadrature_rule,
    std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>>
        coeffs,
    bool constant_normal)
{

  auto mesh = V->mesh();
  assert(mesh);

  // Get mesh info
  const dolfinx::mesh::Geometry& geometry = mesh->geometry();
  auto cmap = geometry.cmap();

  const std::uint32_t gdim = geometry.dim();
  const dolfinx::mesh::Topology& topology = mesh->topology();
  const std::uint32_t tdim = topology.dim();
  const int fdim = tdim - 1;
  const int num_coordinate_dofs = cmap.dim();

  basix::cell::type basix_cell
      = dolfinx::mesh::cell_type_to_basix_type(topology.cell_type());

  // Create quadrature points on reference facet
  const std::vector<std::vector<double>>& q_weights
      = quadrature_rule.weights_ref();
  const std::vector<xt::xarray<double>>& q_points
      = quadrature_rule.points_ref();

  // Structures needed for basis function tabulation
  // phi and grad(phi) at quadrature points
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  const std::uint32_t bs = element->block_size();
  std::uint32_t ndofs_cell = element->space_dimension() / bs;
  auto facets
      = basix::cell::topology(basix_cell)[tdim - 1]; // Topology of basix facets
  const std::uint32_t num_facets = facets.size();
  std::vector<xt::xtensor<double, 2>> phi;
  phi.reserve(num_facets);
  std::vector<xt::xtensor<double, 3>> dphi;
  phi.reserve(num_facets);
  std::vector<xt ::xtensor<double, 3>> dphi_c;
  dphi_c.reserve(num_facets);

  // Structures for coefficient data
  int num_coeffs = coeffs.size();
  std::vector<int> offsets(num_coeffs + 4);
  offsets[0] = 0;
  for (int i = 1; i < num_coeffs + 1; i++)
  {
    std::shared_ptr<const dolfinx::fem::FiniteElement> coeff_element
        = coeffs[i - 1]->function_space()->element();
    offsets[i]
        = offsets[i - 1]
          + coeff_element->space_dimension() / coeff_element->block_size();
  }
  // FIXME: This will not work for prism meshes
  const std::uint32_t num_quadrature_pts = q_points[0].shape(0);
  offsets[num_coeffs + 1] = offsets[num_coeffs] + 1; // h
  offsets[num_coeffs + 2]
      = offsets[num_coeffs + 1] + gdim * num_quadrature_pts; // gap
  offsets[num_coeffs + 3]
      = offsets[num_coeffs + 2] + gdim * num_quadrature_pts; // normals

  // Pack coefficients for functions and gradients of functions (untested)
  // FIXME: This assumption would fail for prisms
  xt::xtensor<double, 3> phi_coeffs(
      {num_facets, q_weights[0].size(), offsets[num_coeffs]});
  xt::xtensor<double, 4> dphi_coeffs(
      {num_facets, tdim, q_weights[0].size(), offsets[num_coeffs]});

  for (int i = 0; i < num_facets; ++i)
  {
    // Push quadrature points forward
    auto facet = facets[i];
    const xt::xarray<double>& q_facet = q_points[i];

    // Tabulate at quadrature points on facet
    const int num_quadrature_points = q_facet.shape(0);
    std::array<std::size_t, 4> tabulate_shape
        = cmap.tabulate_shape(1, num_quadrature_points);
    xt::xtensor<double, 4> cell_tab(tabulate_shape);
    element->tabulate(cell_tab, q_facet, 1);
    xt::xtensor<double, 2> phi_i
        = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
    phi.push_back(phi_i);
    xt::xtensor<double, 3> dphi_i
        = xt::view(cell_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);
    dphi.push_back(dphi_i);

    // Tabulate coordinate element of reference cell
    auto c_tab = cmap.tabulate(1, q_facet);
    xt::xtensor<double, 3> dphi_ci
        = xt::view(c_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);
    dphi_c.push_back(dphi_ci);

    // Create finite elements for coefficient functions and tabulate shape
    // functions
    for (int j = 0; j < num_coeffs; j++)
    {
      std::shared_ptr<const dolfinx::fem::FiniteElement> coeff_element
          = coeffs[j]->function_space()->element();
      xt::xtensor<double, 4> coeff_basis(
          {tdim + 1, q_facet.shape(0),
           coeff_element->space_dimension() / coeff_element->block_size(), 1});
      coeff_element->tabulate(coeff_basis, q_facet, 1);
      auto phi_ij = xt::view(phi_coeffs, i, xt::all(),
                             xt::range(offsets[j], offsets[j + 1]));
      phi_ij = xt::view(coeff_basis, 0, xt::all(), xt::all(), 0);
      auto dphi_ij = xt::view(dphi_coeffs, i, xt::all(), xt::all(),
                              xt::range(offsets[j], offsets[j + 1]));
      dphi_ij = xt::view(coeff_basis, xt::range(1, tdim + 1), xt::all(),
                         xt::all(), 0);
    }
  }

  // As reference facet and reference cell are affine, we do not need to compute
  // this per quadrature point
  auto ref_jacobians = basix::cell::facet_jacobians(basix_cell);

  // Get facet normals on reference cell
  auto facet_normals = basix::cell::facet_outward_normals(basix_cell);

  auto update_jacobian
      = dolfinx_contact::get_update_jacobian_dependencies(cmap);
  auto update_normal = dolfinx_contact::get_update_normal(cmap);
  const bool affine = cmap.is_affine();
  // Define kernels
  // RHS for contact with rigid surface
  // =====================================================================================
  dolfinx_cuas::kernel_fn<T> nitsche_rigid_rhs
      = [=](double* b, const double* c, const double* w,
            const double* coordinate_dofs, const int* entity_local_index,
            const std::uint8_t* quadrature_permutation)
  {
    // assumption that the vector function space has block size tdim
    assert(bs == gdim);
    // FIXME: This assumption does not work on prism meshes
    // assumption that u lives in the same space as v
    assert(phi[0].shape(1) == offsets[1] - offsets[0]);
    std::size_t facet_index = size_t(*entity_local_index);
    assert(phi[facet_index].shape(1) == ndofs_cell);

    // Reshape coordinate dofs to two dimensional array
    // NOTE: DOLFINx has 3D input coordinate dofs
    std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};

    // FIXME: These array should be views (when compute_jacobian doesn't use
    // xtensor)
    const xt::xtensor<double, 2>& coord = xt::adapt(
        coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);
    auto c_view = xt::view(coord, xt::all(), xt::range(0, gdim));

    // Extract the first derivative of the coordinate element (cell) of degrees
    // of freedom on the facet
    const xt::xtensor<double, 3>& dphi_fc = dphi_c[facet_index];

    // Compute Jacobian and determinant at first quadrature point
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});
    xt::xtensor<double, 2> J_f
        = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
    xt::xtensor<double, 2> J_tot
        = xt::zeros<double>({J.shape(0), J_f.shape(1)});

    double detJ;
    // Normal vector on physical facet at a single quadrature point
    xt::xtensor<double, 1> n_phys = xt::zeros<double>({gdim});
    // Pre-compute jacobians and normals for affine meshes
    if (affine)
    {
      detJ = std::fabs(dolfinx_contact::compute_facet_jacobians(
          0, J, K, J_tot, J_f, dphi_fc, coord));
      dolfinx_contact::physical_facet_normal(
          n_phys, K, xt::row(facet_normals, facet_index));
    }

    // Retrieve normal of rigid surface if constant
    xt::xarray<double> n_surf = xt::zeros<double>({gdim});
    double n_dot = 0;
    if (constant_normal)
    {
      // If surface normal constant precompute (n_phys * n_surf)
      for (int i = 0; i < gdim; i++)
      {
        // For closest point projection the gap function is given by
        // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
        // in y = Pi(x)
        n_surf(i) = -w[i + 2];
        n_dot += n_phys(i) * n_surf(i);
      }
    }
    int c_offset = (bs - 1) * offsets[1];
    // This is gamma/h
    double gamma = w[0] / c[c_offset + offsets[3]];
    double gamma_inv = c[c_offset + offsets[3]] / w[0];
    double theta = w[1];

    const xt::xtensor<double, 3>& dphi_f = dphi[facet_index];
    const xt::xtensor<double, 2>& phi_f = phi[facet_index];
    const std::vector<double>& weights = q_weights[facet_index];

    // Temporary variable for grad(phi) on physical cell
    xt::xtensor<double, 2> dphi_phys({bs, ndofs_cell});

    // Temporary work arrays
    xt::xtensor<double, 2> tr({std::uint32_t(offsets[1] - offsets[0]), gdim});
    xt::xtensor<double, 2> epsn({std::uint32_t(offsets[1] - offsets[0]), gdim});

    // Loop over quadrature points
    const int num_points = phi[*entity_local_index].shape(0);
    for (std::size_t q = 0; q < num_points; q++)
    {

      // Update Jacobian and physical normal
      detJ = std::fabs(
          update_jacobian(q, detJ, J, K, J_tot, J_f, dphi_fc, coord));
      update_normal(n_phys, K, facet_normals, facet_index);

      double mu = 0;
      for (int j = offsets[1]; j < offsets[2]; j++)
        mu += c[j + c_offset] * phi_coeffs(facet_index, q, j);
      double lmbda = 0;
      for (int j = offsets[2]; j < offsets[3]; j++)
        lmbda += c[j + c_offset] * phi_coeffs(facet_index, q, j);

      // if normal not constant, get surface normal at current quadrature point
      int normal_offset = c_offset + offsets[5];
      if (!constant_normal)
      {
        n_dot = 0;
        for (int i = 0; i < gdim; i++)
        {
          // For closest point projection the gap function is given by
          // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
          // in y = Pi(x)
          n_surf(i) = -c[normal_offset + q * gdim + i];
          n_dot += n_phys(i) * n_surf(i);
        }
      }
      int gap_offset = c_offset + offsets[4];
      double gap = 0;
      for (int i = 0; i < gdim; i++)
      {
        gap += c[gap_offset + q * gdim + i] * n_surf(i);
      }

      // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2
      std::fill(tr.begin(), tr.end(), 0);
      std::fill(epsn.begin(), epsn.end(), 0);
      for (int j = 0; j < offsets[1] - offsets[0]; j++)
      {
        for (int l = 0; l < bs; l++)
        {
          for (int k = 0; k < tdim; k++)
          {
            tr(j, l) += K(k, l) * dphi_f(k, q, j);
            for (int s = 0; s < gdim; s++)
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
      double u_dot_nsurf = 0;
      for (int i = 0; i < offsets[1] - offsets[0]; i++)
      {
        const std::int32_t block_index = (i + offsets[0]) * bs;
        for (int j = 0; j < bs; j++)
        {
          tr_u += c[block_index + j] * tr(i, j);
          epsn_u += c[block_index + j] * epsn(i, j);
          u_dot_nsurf += c[block_index + j] * n_surf(j) * phi_f(q, i);
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
          double v_dot_nsurf = n_surf(l) * phi_f(q, j);
          b[j * bs + l]
              += -theta * gamma_inv * sign_v * sign_u
                 + R_minus_scaled * (theta * sign_v - gamma * v_dot_nsurf);
        }
      }
    }
  };

  // Jacobian for contact with rigid surface
  // =====================================================================================
  dolfinx_cuas::kernel_fn<T> nitsche_rigid_jacobian =
      [dphi_c, phi, dphi, phi_coeffs, dphi_coeffs, offsets, num_coeffs, gdim,
       tdim, fdim, q_weights, num_coordinate_dofs, ref_jacobians, bs,
       facet_normals, constant_normal, affine, update_jacobian, update_normal,
       ndofs_cell](double* A, const double* c, const double* w,
                   const double* coordinate_dofs, const int* entity_local_index,
                   const std::uint8_t* quadrature_permutation)
  {
    // assumption that the vector function space has block size tdim
    assert(bs == gdim);
    // FIXME: This assumption does not work on prism meshes
    // assumption that u lives in the same space as v
    assert(phi[0].shape(1) == offsets[1] - offsets[0]);
    std::size_t facet_index = size_t(*entity_local_index);
    assert(phi[facet_index].shape(1) == ndofs_cell);

    // Reshape coordinate dofs to two dimensional array
    // NOTE: DOLFINx has 3D input coordinate dofs
    std::array<std::size_t, 2> shape = {num_coordinate_dofs, 3};

    // FIXME: These array should be views (when compute_jacobian doesn't use
    // xtensor)
    xt::xtensor<double, 2> coord = xt::adapt(
        coordinate_dofs, num_coordinate_dofs * 3, xt::no_ownership(), shape);

    // Extract the first derivative of the coordinate element (cell) of degrees
    // of freedom on the facet
    const xt::xtensor<double, 3>& dphi_fc = dphi_c[facet_index];
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});
    xt::xtensor<double, 1> n_phys = xt::zeros<double>({gdim});
    xt::xtensor<double, 2> J_f
        = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
    xt::xtensor<double, 2> J_tot
        = xt::zeros<double>({J.shape(0), J_f.shape(1)});
    double detJ;
    if (affine)
    {
      detJ = std::fabs(dolfinx_contact::compute_facet_jacobians(
          0, J, K, J_tot, J_f, dphi_fc, coord));
      dolfinx_contact::physical_facet_normal(
          n_phys, K, xt::row(facet_normals, facet_index));
    }

    // Retrieve normal of rigid surface if constant
    xt::xtensor<double, 1> n_surf = xt::zeros<double>({gdim});
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
        n_surf(i) = -w[i + 2];
        n_dot += n_phys(i) * n_surf(i);
      }
    }
    int c_offset = (bs - 1) * offsets[1];
    double gamma
        = w[0]
          / c[c_offset + offsets[3]]; // This is gamma/hdouble gamma = w[0];
    double gamma_inv = c[c_offset + offsets[3]] / w[0];
    double theta = w[1];

    const xt::xtensor<double, 3>& dphi_f = dphi[facet_index];
    const xt::xtensor<double, 2>& phi_f = phi[facet_index];
    const std::vector<double>& weights = q_weights[facet_index];

    // Get number of dofs per cell
    // Temporary variable for grad(phi) on physical cell
    xt::xtensor<double, 2> dphi_phys({bs, ndofs_cell});
    xt::xtensor<double, 2> tr = xt::zeros<double>({ndofs_cell, gdim});
    xt::xtensor<double, 2> epsn = xt::zeros<double>({ndofs_cell, gdim});
    const std::uint32_t num_points = phi[facet_index].shape(0);
    for (std::size_t q = 0; q < num_points; q++)
    {

      // Update Jacobian and physical normal
      detJ = std::fabs(
          update_jacobian(q, detJ, J, K, J_tot, J_f, dphi_fc, coord));
      update_normal(n_phys, K, facet_normals, facet_index);

      // if normal not constant, get surface normal at current quadrature point
      int normal_offset = c_offset + offsets[5];
      if (!constant_normal)
      {
        n_dot = 0;
        for (int i = 0; i < gdim; i++)
        {
          // For closest point projection the gap function is given by
          // (-n_y)* (Pi(x) - x), where n_y is the outward unit normal
          // in y = Pi(x)
          n_surf(i) = -c[normal_offset + q * gdim + i];
          n_dot += n_phys(i) * n_surf(i);
        }
      }
      int gap_offset = c_offset + offsets[4];
      double gap = 0;
      for (int i = 0; i < gdim; i++)
        gap += c[gap_offset + q * gdim + i] * n_surf(i);

      // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2
      std::fill(tr.begin(), tr.end(), 0);
      std::fill(epsn.begin(), epsn.end(), 0);
      for (int j = 0; j < ndofs_cell; j++)
      {
        for (int l = 0; l < bs; l++)
        {
          for (int k = 0; k < tdim; k++)
          {
            tr(j, l) += K(k, l) * dphi_f(k, q, j);
            for (int s = 0; s < gdim; s++)
            {
              epsn(j, l) += K(k, s) * dphi_f(k, q, j)
                            * (n_phys(s) * n_surf(l) + n_phys(l) * n_surf(s));
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
          epsn_u += c[block_index + j] * epsn(i, j);
          u_dot_nsurf += c[block_index + j] * n_surf(j) * phi_f(q, i);
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
              = (gamma_inv * sign_du - n_surf(l) * phi_f(q, j)) * Pn_u * w0;
          sign_du *= w0;
          for (int i = 0; i < ndofs_cell; i++)
          { // Insert over block size in matrix
            for (int b = 0; b < bs; b++)
            {
              double v_dot_nsurf = n_surf(b) * phi_f(q, i);
              double sign_v = (lmbda * tr(i, b) * n_dot + mu * epsn(i, b));
              A[(b + i * bs) * ndofs_cell * bs + l + j * bs]
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
