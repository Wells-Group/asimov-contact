// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx_contact/Contact.hpp>
#include <dolfinx_cuas/QuadratureRule.hpp>
#include <dolfinx_cuas/math.hpp>
#include <dolfinx_cuas/utils.hpp>

using kernel_fn
    = std::function<void(double*, const double*, const double*, const double*,
                         const int*, const std::uint8_t*)>;

namespace dolfinx_contact
{

kernel_fn generate_contact_kernel(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V, Kernel type,
    dolfinx_cuas::QuadratureRule& quadrature_rule,
    std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>>
        coeffs,
    bool constant_normal)
{

  auto mesh = V->mesh();

  // Get mesh info
  const std::uint32_t gdim = mesh->geometry().dim(); // geometrical dimension
  const std::uint32_t tdim = mesh->topology().dim(); // topological dimension
  const int fdim = tdim - 1; // topological dimension of facet

  // Create coordinate elements (for facet and cell) _marker->mesh()
  const basix::FiniteElement basix_element
      = dolfinx_cuas::mesh_to_basix_element(mesh, tdim);
  const int num_coordinate_dofs = basix_element.dim();

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
  auto facets = basix::cell::topology(
      basix_element.cell_type())[tdim - 1]; // Topology of basix facets
  const std::uint32_t num_facets = facets.size();
  std::vector<xt::xtensor<double, 2>> phi;
  phi.reserve(num_facets);
  std::vector<xt::xtensor<double, 3>> dphi;
  phi.reserve(num_facets);
  std::vector<xt ::xtensor<double, 3>> dphi_c;
  dphi_c.reserve(num_facets);

  // Structures for coefficient data
  int num_coeffs = coeffs.size();
  std::vector<int> offsets(num_coeffs + 3);
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
  offsets[num_coeffs + 1] = offsets[num_coeffs] + 1;
  offsets[num_coeffs + 2] = offsets[num_coeffs + 1] + gdim * num_quadrature_pts;
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

    xt::xtensor<double, 4> cell_tab(
        {tdim + 1, num_quadrature_points, ndofs_cell, bs});
    element->tabulate(cell_tab, q_facet, 1);
    xt::xtensor<double, 2> phi_i
        = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
    phi.push_back(phi_i);
    xt::xtensor<double, 3> dphi_i
        = xt::view(cell_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);
    dphi.push_back(dphi_i);

    // Tabulate coordinate element of reference cell
    auto c_tab = basix_element.tabulate(1, q_facet);
    xt::xtensor<double, 3> dphi_ci
        = xt::view(c_tab, xt::range(1, tdim + 1), xt::all(), xt::all(), 0);
    dphi_c.push_back(dphi_ci);

    // Create Finite elements for coefficient functions and tabulate shape
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
  auto ref_jacobians = basix::cell::facet_jacobians(basix_element.cell_type());

  // Get facet normals on reference cell
  auto facet_normals
      = basix::cell::facet_outward_normals(basix_element.cell_type());
  // Define kernels
  // RHS for contact with rigid surface
  // =====================================================================================
  kernel_fn nitsche_rigid_rhs
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
    const xt::xtensor<double, 2>& dphi0_c = xt::view(
        dphi_fc, xt::all(), 0,
        xt::all()); // FIXME: Assumed constant, i.e. only works for simplices

    // NOTE: Affine cell assumption
    // Compute Jacobian and determinant at first quadrature point
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});

    dolfinx_cuas::math::compute_jacobian(dphi0_c, c_view, J);
    dolfinx_cuas::math::compute_inv(J, K);

    // Compute normal of physical facet using a normalized covariant Piola
    // transform n_phys = J^{-T} n_ref / ||J^{-T} n_ref|| See for instance
    // DOI: 10.1137/08073901X
    xt::xarray<double> n_phys = xt::zeros<double>({gdim});
    auto facet_normal = xt::row(facet_normals, facet_index);
    for (std::size_t i = 0; i < gdim; i++)
      for (std::size_t j = 0; j < tdim; j++)
        n_phys[i] += K(j, i) * facet_normal[j];
    double n_norm = 0;
    for (std::size_t i = 0; i < gdim; i++)
      n_norm += n_phys[i] * n_phys[i];
    n_phys /= std::sqrt(n_norm);

    // Retrieve normal of rigid surface if constant
    xt::xarray<double> n_surf = xt::zeros<double>({gdim});
    if (constant_normal)
    {
      for (int i = 0; i < gdim; i++)
        n_surf(i) = w[i + 2];
    }
    int c_offset = (bs - 1) * offsets[1];
    double gamma = w[0] / c[c_offset + offsets[3]]; // This is gamma/h
    double gamma_inv = c[c_offset + offsets[3]] / w[0];
    double theta = w[1];
    // If surface normal constant precompute (n_phys * n_surf)
    double n_dot = 0;
    if (constant_normal)
    {
      for (int i = 0; i < gdim; i++)
        n_dot += n_phys(i) * n_surf(i);
    }

    // Compute det(J_C J_f) as it is the mapping to the reference facet
    const xt::xtensor<double, 2>& J_f
        = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
    xt::xtensor<double, 2> J_tot
        = xt::zeros<double>({J.shape(0), J_f.shape(1)});
    dolfinx::math::dot(J, J_f, J_tot);
    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_tot));

    const xt::xtensor<double, 3>& dphi_f = dphi[facet_index];
    const xt::xtensor<double, 2>& phi_f = phi[facet_index];
    const std::vector<double>& weights = q_weights[facet_index];

    // Get number of dofs per cell
    // FIXME: Should be templated
    const std::uint32_t ndofs_cell = phi[*entity_local_index].shape(1);
    // Temporary variable for grad(phi) on physical cell
    xt::xtensor<double, 2> dphi_phys({bs, ndofs_cell});

    // Loop over quadrature points
    const int num_points = phi[*entity_local_index].shape(0);
    for (std::size_t q = 0; q < num_points; q++)
    {

      double mu = 0;
      for (int j = offsets[1]; j < offsets[2]; j++)
        mu += c[j + c_offset] * phi_coeffs(facet_index, q, j);
      double lmbda = 0;
      for (int j = offsets[2]; j < offsets[3]; j++)
        lmbda += c[j + c_offset] * phi_coeffs(facet_index, q, j);
      double gap = 0;
      int gap_offset = c_offset + offsets[4];
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
        // NOTE: when n_dot is negative, there is some penetration at the
        // contact surface in the mesh. This can be the case if there is already
        // some contact in the mesh, i.e., for zero displacement, and only up to
        // the precision of the meshing of the contact surfaces. This in
        // particular occurs if the mesh was generated by moving some initial
        // mesh according to the solution of a contact problem in which case the
        // penetration can be up to the precision of the solver.
        if (gap > 1e-13)
        {
          n_surf /= dolfinx_contact::sgn(n_dot) * gap;
          n_dot /= dolfinx_contact::sgn(n_dot) * gap;
        }
        else
        {
          n_surf = n_phys;
          n_dot = 1;
          gap = 0.0;
        }
      }
      else
      {
        for (int i = 0; i < gdim; i++)
        {
          gap += c[gap_offset + q * gdim + i] * n_surf(i);
        }
      }

      xt::xtensor<double, 2> tr
          = xt::zeros<double>({std::uint32_t(offsets[1] - offsets[0]), gdim});
      xt::xtensor<double, 2> epsn
          = xt::zeros<double>({std::uint32_t(offsets[1] - offsets[0]), gdim});
      xt::xtensor<double, 2> v_dot_nsurf
          = xt::zeros<double>({std::uint32_t(offsets[1] - offsets[0]), gdim});
      // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2, ufl.dot(v, n_surf)
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
      double R_minus = gamma_inv * sign_u + (gap - u_dot_nsurf);
      if (R_minus > 0)
        R_minus = 0;
      else
        R_minus = R_minus * detJ * weights[q];
      sign_u *= detJ * weights[q];
      for (int j = 0; j < ndofs_cell; j++)
      {
        // Insert over block size in matrix
        for (int l = 0; l < bs; l++)
        {
          double sign_v = lmbda * tr(j, l) * n_dot + mu * epsn(j, l);
          double v_dot_nsurf = n_surf(l) * phi_f(q, j);
          b[j * bs + l] += -theta * gamma_inv * sign_v * sign_u
                           + R_minus * (theta * sign_v - gamma * v_dot_nsurf);
        }
      }
    }
  };

  // Jacobian for contact with rigid surface
  // =====================================================================================
  kernel_fn nitsche_rigid_jacobian
      = [dphi_c, phi, dphi, phi_coeffs, dphi_coeffs, offsets, num_coeffs, gdim,
         tdim, fdim, q_weights, num_coordinate_dofs, ref_jacobians, bs,
         facet_normals, constant_normal](
            double* A, const double* c, const double* w,
            const double* coordinate_dofs, const int* entity_local_index,
            const std::uint8_t* quadrature_permutation)
  {
    // assumption that the vector function space has block size tdim
    assert(bs == gdim);
    // FIXME: This assumption does not work on prism meshes
    // assumption that u lives in the same space as v
    assert(phi[0].shape(1) == offsets[1] - offsets[0]);
    std::size_t facet_index = size_t(*entity_local_index);

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
    const xt::xtensor<double, 2>& dphi0_c = xt::view(
        dphi_fc, xt::all(), 0,
        xt::all()); // FIXME: Assumed constant, i.e. only works for simplices

    // NOTE: Affine cell assumption
    // Compute Jacobian and determinant at first quadrature point
    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});
    auto c_view = xt::view(coord, xt::all(), xt::range(0, gdim));
    dolfinx_cuas::math::compute_jacobian(dphi0_c, c_view, J);
    dolfinx_cuas::math::compute_inv(J, K);

    // Compute normal of physical facet using a normalized covariant Piola
    // transform n_phys = J^{-T} n_ref / ||J^{-T} n_ref|| See for instance
    // DOI: 10.1137/08073901X
    xt::xarray<double> n_phys = xt::zeros<double>({gdim});
    auto facet_normal = xt::row(facet_normals, facet_index);
    for (std::size_t i = 0; i < gdim; i++)
      for (std::size_t j = 0; j < tdim; j++)
        n_phys[i] += K(j, i) * facet_normal[j];
    double n_norm = 0;
    for (std::size_t i = 0; i < gdim; i++)
      n_norm += n_phys[i] * n_phys[i];
    n_phys /= std::sqrt(n_norm);

    // Retrieve normal of rigid surface if constant
    xt::xarray<double> n_surf = xt::zeros<double>({gdim});
    // FIXME: Code duplication from previous kernel, and should be made into a
    // lambda function
    if (constant_normal)
    {
      for (int i = 0; i < gdim; i++)
        n_surf(i) = w[i + 2];
    }
    int c_offset = (bs - 1) * offsets[1];
    double gamma
        = w[0]
          / c[c_offset + offsets[3]]; // This is gamma/hdouble gamma = w[0];
    double gamma_inv = c[c_offset + offsets[3]] / w[0];
    double theta = w[1];
    // If surface normal constant precompute (n_phys * n_surf)
    double n_dot = 0;
    if (constant_normal)
    {
      for (int i = 0; i < gdim; i++)
        n_dot += n_phys(i) * n_surf(i);
    }

    // Compute det(J_C J_f) as it is the mapping to the reference facet
    xt::xtensor<double, 2> J_f
        = xt::view(ref_jacobians, facet_index, xt::all(), xt::all());
    xt::xtensor<double, 2> J_tot
        = xt::zeros<double>({J.shape(0), J_f.shape(1)});
    dolfinx::math::dot(J, J_f, J_tot);
    double detJ = std::fabs(dolfinx_cuas::math::compute_determinant(J_tot));

    const xt::xtensor<double, 3>& dphi_f = dphi[facet_index];
    const xt::xtensor<double, 2>& phi_f = phi[facet_index];
    const std::vector<double>& weights = q_weights[facet_index];

    // Get number of dofs per cell
    // FIXME: Should be templated
    const std::uint32_t ndofs_cell = phi[*entity_local_index].shape(1);
    // Temporary variable for grad(phi) on physical cell
    xt::xtensor<double, 2> dphi_phys({bs, ndofs_cell});

    const std::uint32_t num_points = phi[*entity_local_index].shape(0);
    for (std::size_t q = 0; q < num_points; q++)
    {
      double gap = 0;
      int gap_offset = c_offset + offsets[4];
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
        // NOTE: when n_dot is negative, there is some penetration at the
        // contact surface in the mesh. This can be the case if there is already
        // some contact in the mesh, i.e., for zero displacement, and only up to
        // the precision of the meshing of the contact surfaces. This in
        // particular occurs if the mesh was generated by moving some initial
        // mesh according to the solution of a contact problem in which case the
        // penetration can be up to the precision of the solver.
        if (gap > 1e-13)
        {
          n_surf /= dolfinx_contact::sgn(n_dot) * gap;
          n_dot /= dolfinx_contact::sgn(n_dot) * gap;
        }
        else
        {
          n_surf = n_phys;
          n_dot = 1;
          gap = 0.0;
        }
      }
      else
      {
        for (int i = 0; i < gdim; i++)
        {
          gap += c[gap_offset + q * gdim + i] * n_surf(i);
        }
      }
      xt::xtensor<double, 2> tr = xt::zeros<double>({ndofs_cell, gdim});
      xt::xtensor<double, 2> epsn = xt::zeros<double>({ndofs_cell, gdim});
      // precompute tr(eps(phi_j e_l)), eps(phi^j e_l)n*n2
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
      double temp
          = (lmbda * n_dot * tr_u + mu * epsn_u) + gamma * (gap - u_dot_nsurf);
      const double w0 = weights[q] * detJ;
      for (int j = 0; j < ndofs_cell; j++)
      {
        for (int l = 0; l < bs; l++)
        {
          double sign_u = (lmbda * tr(j, l) * n_dot + mu * epsn(j, l));
          double term2 = 0;
          if (temp < 0)
            term2 = (gamma_inv * sign_u - n_surf(l) * phi_f(q, j)) * w0;
          sign_u *= w0;
          for (int i = 0; i < ndofs_cell; i++)
          { // Insert over block size in matrix
            for (int b = 0; b < bs; b++)
            {
              double v_dot_nsurf = n_surf(b) * phi_f(q, i);
              double sign_v = (lmbda * tr(i, b) * n_dot + mu * epsn(i, b));
              A[(b + i * bs) * ndofs_cell * bs + l + j * bs]
                  += -theta * gamma_inv * sign_u * sign_v
                     + term2 * (theta * sign_v - gamma * v_dot_nsurf);
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
    throw std::runtime_error("Unrecognized kernel");
  }
}
} // namespace dolfinx_contact
