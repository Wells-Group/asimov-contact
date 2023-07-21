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
  _consts = {gamma, theta};
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh
      = u->function_space()->mesh(); // mesh
  std::size_t tdim = mesh->topology()->dim();
  auto it = dolfinx::fem::IntegralType::exterior_facet;
  std::size_t coeff_size = coefficients_size(true);

  for (std::size_t i = 0; i < _num_pairs; ++i)
  {
    const std::array<int, 2>& pair = contact_pair(i);
    std::span<const std::int32_t> entities = active_entities(pair[0]);
    std::vector<std::int32_t> cells(entities.size() / 2);
    for (std::size_t e = 0; e < entities.size() / 2; ++e)
      cells[e] = entities[2 * e];

    std::vector<double> h_p = dolfinx::mesh::h(*mesh, cells, tdim);
    std::size_t c_h = 1;
    auto [lm_p, c_lm] = pack_coefficient_quadrature(lambda, 0, entities, it);
    auto [mu_p, c_mu] = pack_coefficient_quadrature(mu, 0, entities, it);
    // auto [h_p, c_h] = pack_coefficient_quadrature(h, 0, entities, it);
    auto [gap, cgap] = pack_gap(i);
    auto [testfn, ctest] = pack_test_functions(i);
    std::vector<double> dummy(gap.size(), 0.0);
    auto [gradtst, cgt]
        = pack_grad_test_functions(i, gap, std::span<double>(dummy));
    auto [u_p, c_u] = pack_coefficient_quadrature(u, _q_deg, entities, it);
    auto [gradu, c_gu] = pack_gradient_quadrature(u, _q_deg, entities, it);
    auto [u_cd, c_uc] = pack_u_contact(i, u);
    auto [u_gc, c_ugc] = pack_grad_u_contact(i, u, gap, dummy);

    std::vector<double> coeffs(coeff_size * entities.size() / 2);
    for (std::size_t e = 0; e < entities.size() / 2; ++e)
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
void dolfinx_contact::MeshTie::assemble_vector(std::span<PetscScalar> b)
{
  std::size_t cstride = coefficients_size(true);
  for (std::size_t i = 0; i < _num_pairs; ++i)
    assemble_vector(b, i, _kernel_rhs, _coeffs[i], cstride, _consts);
}

void dolfinx_contact::MeshTie::assemble_matrix(
    const mat_set_fn& mat_set,
    const std::vector<
        std::shared_ptr<const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs)
{
  std::size_t cstride = coefficients_size(true);
  for (std::size_t i = 0; i < _num_pairs; ++i)
    assemble_matrix(mat_set, bcs, i, _kernel_jac, _coeffs[i], cstride, _consts);
}

std::pair<std::vector<double>, std::size_t>
dolfinx_contact::MeshTie::coeffs(int pair)
{
  std::size_t cstride = coefficients_size(true);
  std::vector<double>& coeffs = _coeffs[pair];
  return {coeffs, cstride};
}
