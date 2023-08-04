// Copyright (C) 2021-2022 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#include "MeshTie.h"

void dolfinx_contact::MeshTie::generate_meshtie_data(
    std::shared_ptr<dolfinx::fem::Function<double>> u,
    std::shared_ptr<dolfinx::fem::Function<double>> lambda,
    std::shared_ptr<dolfinx::fem::Function<double>> mu, double gamma,
    double theta)
{
  // save nitsche parameters as constants
  _consts = {gamma, theta};
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh
      = u->function_space()->mesh();  // mesh
  int tdim = mesh->topology()->dim(); // topological dimension
  auto it = dolfinx::fem::IntegralType::exterior_facet;
  std::size_t coeff_size = coefficients_size(true); // data size

  // loop over connected pairs
  for (int i = 0; i < _num_pairs; ++i)
  {
    // retrieve indices of connected surfaces
    const std::array<int, 2>& pair = Contact::contact_pair(i);
    // retrieve integration facets
    std::span<const std::int32_t> entities = Contact::active_entities(pair[0]);
    // number of facets own by process
    std::size_t num_facets = Contact::local_facets(pair[0]);
    // Retrieve cells connected to integration facets
    std::vector<std::int32_t> cells(num_facets);
    for (std::size_t e = 0; e < num_facets; ++e)
      cells[e] = entities[2 * e];

    // compute cell sizes
    std::vector<double> h_p = dolfinx::mesh::h(*mesh, cells, tdim);
    std::size_t c_h = 1;
    auto [lm_p, c_lm]
        = pack_coefficient_quadrature(lambda, 0, entities, it); // lambda
    auto [mu_p, c_mu] = pack_coefficient_quadrature(mu, 0, entities, it); // mu
    auto [gap, cgap] = Contact::pack_gap(i);                // gap function
    auto [testfn, ctest] = Contact::pack_test_functions(i); // test functions
    auto [u_p, c_u] = pack_coefficient_quadrature(u, _q_deg, entities, it); // u
    auto [gradu, c_gu]
        = pack_gradient_quadrature(u, _q_deg, entities, it); // grad(u)
    auto [u_cd, c_uc] = Contact::pack_u_contact(i, u); // u on connected surface
    auto [u_gc, c_ugc]
        = Contact::pack_grad_u_contact(i, u); // grad(u) on connected surface
    auto [gradtst, cgt]
        = Contact::pack_grad_test_functions(i); // test fns on connected surface

    // copy data into one common data vector in the order expected by the
    // integration kernel
    std::vector<double> coeffs(coeff_size * num_facets);
    for (std::size_t e = 0; e < num_facets; ++e)
    {
      std::copy_n(std::next(mu_p.begin(), e * c_mu), c_mu,
                  std::next(coeffs.begin(), e * coeff_size));
      std::size_t offset = c_mu;
      std::copy_n(std::next(lm_p.begin(), e * c_lm), c_lm,
                  std::next(coeffs.begin(), e * coeff_size + offset));
      offset += c_lm;
      std::copy_n(std::next(h_p.begin(), e * c_h), c_h,
                  std::next(coeffs.begin(), e * coeff_size + offset));
      offset += c_h;
      std::copy_n(std::next(testfn.begin(), e * ctest), ctest,
                  std::next(coeffs.begin(), e * coeff_size + offset));
      offset += ctest;
      std::copy_n(std::next(gradtst.begin(), e * cgt), cgt,
                  std::next(coeffs.begin(), e * coeff_size + offset));
      offset += cgt;
      std::copy_n(std::next(u_p.begin(), e * c_u), c_u,
                  std::next(coeffs.begin(), e * coeff_size + offset));
      offset += c_u;
      std::copy_n(std::next(gradu.begin(), e * c_gu), c_gu,
                  std::next(coeffs.begin(), e * coeff_size + offset));
      offset += c_gu;
      std::copy_n(std::next(u_cd.begin(), e * c_uc), c_uc,
                  std::next(coeffs.begin(), e * coeff_size + offset));
      offset += c_uc;
      std::copy_n(std::next(u_gc.begin(), e * c_ugc), c_ugc,
                  std::next(coeffs.begin(), e * coeff_size + offset));
    }

    _coeffs[i] = coeffs;
  }
}

void dolfinx_contact::MeshTie::generate_meshtie_data_matrix_only(
    std::shared_ptr<dolfinx::fem::Function<double>> lambda,
    std::shared_ptr<dolfinx::fem::Function<double>> mu, double gamma,
    double theta)
{
  // save nitsche parameters as constants
  _consts = {gamma, theta};
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh
      = lambda->function_space()->mesh();  // mesh
  int tdim = mesh->topology()->dim(); // topological dimension
  auto it = dolfinx::fem::IntegralType::exterior_facet;
  std::size_t coeff_size = coefficients_size(true); // data size

  // loop over connected pairs
  for (int i = 0; i < _num_pairs; ++i)
  {
    // retrieve indices of connected surfaces
    const std::array<int, 2>& pair = Contact::contact_pair(i);
    // retrieve integration facets
    std::span<const std::int32_t> entities = Contact::active_entities(pair[0]);
    // number of facets own by process
    std::size_t num_facets = Contact::local_facets(pair[0]);
    // Retrieve cells connected to integration facets
    std::vector<std::int32_t> cells(num_facets);
    for (std::size_t e = 0; e < num_facets; ++e)
      cells[e] = entities[2 * e];

    // compute cell sizes
    std::vector<double> h_p = dolfinx::mesh::h(*mesh, cells, tdim);
    std::size_t c_h = 1;
    auto [lm_p, c_lm]
        = pack_coefficient_quadrature(lambda, 0, entities, it); // lambda
    auto [mu_p, c_mu] = pack_coefficient_quadrature(mu, 0, entities, it); // mu
    auto [gap, cgap] = Contact::pack_gap(i);                // gap function
    auto [testfn, ctest] = Contact::pack_test_functions(i); // test functions
    auto [gradtst, cgt]
        = Contact::pack_grad_test_functions(i); // test fns on connected surface

    // copy data into one common data vector in the order expected by the
    // integration kernel
    std::vector<double> coeffs(coeff_size * num_facets);
    for (std::size_t e = 0; e < num_facets; ++e)
    {
      std::copy_n(std::next(mu_p.begin(), e * c_mu), c_mu,
                  std::next(coeffs.begin(), e * coeff_size));
      std::size_t offset = c_mu;
      std::copy_n(std::next(lm_p.begin(), e * c_lm), c_lm,
                  std::next(coeffs.begin(), e * coeff_size + offset));
      offset += c_lm;
      std::copy_n(std::next(h_p.begin(), e * c_h), c_h,
                  std::next(coeffs.begin(), e * coeff_size + offset));
      offset += c_h;
      std::copy_n(std::next(testfn.begin(), e * ctest), ctest,
                  std::next(coeffs.begin(), e * coeff_size + offset));
      offset += ctest;
      std::copy_n(std::next(gradtst.begin(), e * cgt), cgt,
                  std::next(coeffs.begin(), e * coeff_size + offset));
    }

    _coeffs[i] = coeffs;
  }
}
void dolfinx_contact::MeshTie::assemble_vector(std::span<PetscScalar> b)
{
  std::size_t cstride = coefficients_size(true);
  for (int i = 0; i < _num_pairs; ++i)
    assemble_vector(b, i, _kernel_rhs, _coeffs[i], (int)cstride, _consts);
}

void dolfinx_contact::MeshTie::assemble_matrix(const mat_set_fn& mat_set)
{
  std::size_t cstride = coefficients_size(true);
  for (int i = 0; i < _num_pairs; ++i)
    assemble_matrix(mat_set, i, _kernel_jac, _coeffs[i], (int)cstride, _consts);
}

std::pair<std::vector<double>, std::size_t>
dolfinx_contact::MeshTie::coeffs(int pair)
{
  std::size_t cstride = coefficients_size(true);
  std::vector<double>& coeffs = _coeffs[pair];
  return {coeffs, cstride};
}
