// Copyright (C) 2021-2022 Sarah Roggendorf
// 
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT


#include "MeshTie.h"


 

dolfinx_contact::MeshTie::MeshTie(
    const std::vector<std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>>&
        markers,
    std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>> surfaces,
    const std::vector<std::array<int, 2>>& contact_pairs,
    std::shared_ptr<dolfinx::fem::FunctionSpace<double>> V, const int q_deg) 
{
_contact = dolfinx_contact::Contact(markers, surfaces, contact_pairs, V, q_deg, ContactMode::ClosestPoint);
for(std::size_t i = 0; i < contact_pairs.size(); ++i)
    _contact.create_distance_map(i);
_kernel_rhs = _contact.generate_kernel(Kernel::MeshTieRhs);
_kernel_jac = _contact.generate_kernel(Kernel::MeshTieJac);
_num_pairs = contact_pairs.size();
_coeffs.resize(_num_pairs);
_q_deg = q_deg;
}

void dolfinx_contact::MeshTie::pack_coefficients(std::shared_ptr<dolfinx::fem::Function<double>> u,
                         std::shared_ptr<dolfinx::fem::Function<double>> lambda,
                         std::shared_ptr<dolfinx::fem::Function<double>> mu,
                         std::shared_ptr<dolfinx::fem::Function<double>> h,
                         double gamma, double theta)
{
    auto it = dolfinx::fem::IntegralType::exterior_facet;
    std::size_t coeff_size = _contact.coefficients_size(true);
    for(std::size_t i = 0; i < _num_pairs; ++i){
        std::span<const std::int32_t>active_entities = _contact.active_entities(i);
        std::vector<double> coeffs(coeff_size * active_entities.size()/2);
        auto [lm_p, c_lm] = pack_coefficient_quadrature(lambda, 0, active_entities, it);
        auto [mu_p, c_mu] = pack_coefficient_quadrature(mu, 0, active_entities, it);
        auto [h_p, c_h] = pack_coefficient_quadrature(h, 0, active_entities, it);
        auto [gap, cgap]  = _contact.pack_gap(i);
        auto [testfn, ctest] = _contact.pack_test_functions(i);
        std::vector<double>dummy(gap.size(), 0.0);
        auto [gradtst, cgt]  = _contact.pack_grad_test_functions(i, gap, std::span<double>(dummy));
        auto [u_p, c_u] = pack_coefficient_quadrature(u, _q_deg, active_entities, it);
        auto [gradu, c_gu] = pack_gradient_quadrature(u, _q_deg, active_entities, it);
        auto [u_cd, c_uc] = _contact.pack_u_contact(i, u);
        auto [u_gc, c_ugc] = _contact.pack_grad_u_contact(i, u, gap, dummy);
        for (std::size_t e = 0; e<active_entities.size()/2; ++e)
        {
            std::copy_n(std::next(mu_p.begin(), e * c_mu), c_mu, std::next(coeffs.begin(), e * coeff_size));
            std::size_t offset = c_mu;
            std::copy_n(std::next(lm_p.begin(), e * c_lm), c_lm, std::next(coeffs.begin(), e * coeff_size + offset));
            offset += c_lm;
            std::copy_n(std::next(h_p.begin(), e * c_h), c_h, std::next(coeffs.begin(), e * coeff_size + offset));
            offset += c_h;
            std::copy_n(std::next(testfn.begin(), e * ctest), ctest, std::next(coeffs.begin(), e * coeff_size + offset));
            offset += ctest;
            std::copy_n(std::next(gradtst.begin(), e * cgt), cgt, std::next(coeffs.begin(), e * coeff_size + offset));
            offset += cgt;
            std::copy_n(std::next(u_p.begin(), e * c_u), c_u, std::next(coeffs.begin(), e * coeff_size + offset));
            offset += c_u;
            std::copy_n(std::next(gradu.begin(), e * c_gu), c_gu, std::next(coeffs.begin(), e * coeff_size + offset));
            offset += c_gu;
            std::copy_n(std::next(u_cd.begin(), e * c_uc), c_uc, std::next(coeffs.begin(), e * coeff_size + offset));
            offset += c_uc;
            std::copy_n(std::next(u_gc.begin(), e * c_ugc), c_ugc, std::next(coeffs.begin(), e * coeff_size + offset));
        }

        _coeffs[i] = coeffs;
    }
}