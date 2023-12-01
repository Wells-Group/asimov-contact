// Copyright (C) 2023 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#include "MeshTie.h"
std::size_t dolfinx_contact::MeshTie::offset_elasticity(
    std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V)
{
  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  const std::size_t ndofs_cell = dofmap->cell_dofs(0).size();
  const std::size_t bs = dofmap->bs();
  std::size_t num_pts = Contact::num_q_points();
  std::size_t max_links = Contact::max_links();
  return 3 + 2 * (num_pts * max_links * bs * ndofs_cell);
}
std::size_t dolfinx_contact::MeshTie::offset_poisson(
    std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V)
{
  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  const std::size_t ndofs_cell = dofmap->cell_dofs(0).size();
  const std::size_t gdim = V->mesh()->geometry().dim();
  std::size_t num_pts = Contact::num_q_points();
  std::size_t max_links = Contact::max_links();
  return 2 + (1 + gdim) * (num_pts * max_links * ndofs_cell);
}

void dolfinx_contact::MeshTie::generate_kernel_data(
    dolfinx_contact::Problem problem_type,
    std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
    const std::map<std::string,
                   std::shared_ptr<dolfinx::fem::Function<double>>>&
        coefficients,
    double gamma, double theta)
{
  std::vector<std::shared_ptr<dolfinx::fem::Function<double>>> coeff_list;
  switch (problem_type)
  {
  
  using enum dolfinx_contact::Problem;
  case Elasticity:

    if (auto it = coefficients.find("mu"); it != coefficients.end())
      coeff_list.push_back(it->second);
    else
    {
      throw std::invalid_argument("Lame parameter mu not provided.");
    }
    if (auto it = coefficients.find("lambda"); it != coefficients.end())
      coeff_list.push_back(it->second);
    else
    {
      throw std::invalid_argument("Lame parameter lambda not provided.");
    }
    if (auto it = coefficients.find("u"); it != coefficients.end())
    {
      coeff_list.push_back(it->second);
      dolfinx_contact::MeshTie::generate_meshtie_data(
          coeff_list[2], coeff_list[1], coeff_list[0], gamma, theta);
    }
    else
    {
      dolfinx_contact::MeshTie::generate_meshtie_data_matrix_only(
          V, coeff_list[1], coeff_list[0], gamma, theta);
    }
    break;
  case Poisson:
    if (auto it = coefficients.find("kdt"); it != coefficients.end())
      coeff_list.push_back(it->second);
    else
    {
      throw std::invalid_argument("kdt not provided.");
    }
    if (auto it = coefficients.find("T"); it != coefficients.end())
    {
      coeff_list.push_back(it->second);
      dolfinx_contact::MeshTie::generate_poisson_data(
          coeff_list[1], coeff_list[0], gamma, theta);
    }
    else
    {
      dolfinx_contact::MeshTie::generate_poisson_data_matrix_only(
          V, coeff_list[0], gamma, theta);
    }
    break;
  case ThermoElasticity:
    throw std::invalid_argument("Problem type not implemented");
    break;
  default:
    throw std::invalid_argument("Problem type not implemented");
  }
}
void dolfinx_contact::MeshTie::generate_meshtie_data(
    std::shared_ptr<dolfinx::fem::Function<double>> u,
    std::shared_ptr<dolfinx::fem::Function<double>> lambda,
    std::shared_ptr<dolfinx::fem::Function<double>> mu, double gamma,
    double theta)
{
  // generate the data used for the matrix
  generate_meshtie_data_matrix_only(u->function_space(), lambda, mu, gamma,
                                    theta);
  std::size_t coeff_size
      = coefficients_size(true, u->function_space()); // data size
  update_meshtie_data(u, _coeffs, offset_elasticity(u->function_space()),
                      coeff_size);
}

void dolfinx_contact::MeshTie::generate_meshtie_data_matrix_only(
    std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
    std::shared_ptr<dolfinx::fem::Function<double>> lambda,
    std::shared_ptr<dolfinx::fem::Function<double>> mu, double gamma,
    double theta)
{
  // Generate integration kernels
  _kernel_rhs = Contact::generate_kernel(Kernel::MeshTieRhs, V);
  _kernel_jac = Contact::generate_kernel(Kernel::MeshTieJac, V);

  // save nitsche parameters as constants
  _consts = {gamma, theta};
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh
      = lambda->function_space()->mesh(); // mesh
  int tdim = mesh->topology()->dim();     // topological dimension
  auto it = dolfinx::fem::IntegralType::exterior_facet;
  std::size_t coeff_size = coefficients_size(true, V); // data size
  _cstride = coeff_size;

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
    auto [gap, cgap] = Contact::pack_gap(i);                   // gap function
    auto [testfn, ctest] = Contact::pack_test_functions(i, V); // test functions
    auto [gradtst, cgt] = Contact::pack_grad_test_functions(
        i, V); // test fns on connected surface

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
void dolfinx_contact::MeshTie::update_meshtie_data(
    std::shared_ptr<dolfinx::fem::Function<double>> u,
    dolfinx_contact::Problem problem_type)
{
  switch (problem_type)
  {
  using enum dolfinx_contact::Problem;
  case Elasticity:
    update_meshtie_data(u, _coeffs, offset_elasticity(u->function_space()),
                        _cstride);
    break;
  case Poisson:
    update_meshtie_data(u, _coeffs_poisson, offset_poisson(u->function_space()),
                        _cstride_poisson);
    break;
  case ThermoElasticity:
    update_meshtie_data(u, _coeffs, offset_elasticity(u->function_space()),
                        _cstride);
    break;
  default:
    throw std::invalid_argument("Problem type not implemented");
  }
}

void dolfinx_contact::MeshTie::update_meshtie_data(
    std::shared_ptr<dolfinx::fem::Function<double>> u,
    std::vector<std::vector<double>>& coeffs, std::size_t offset0,
    std::size_t coeff_size)
{
  std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V
      = u->function_space();                                           // mesh
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh = V->mesh(); // mesh
  auto it = dolfinx::fem::IntegralType::exterior_facet;

  // loop over connected pairs
  for (int i = 0; i < _num_pairs; ++i)
  {
    // retrieve indices of connected surfaces
    const std::array<int, 2>& pair = Contact::contact_pair(i);
    // retrieve integration facets
    std::span<const std::int32_t> entities = Contact::active_entities(pair[0]);
    // number of facets own by process
    std::size_t num_facets = Contact::local_facets(pair[0]);
    auto [u_p, c_u] = pack_coefficient_quadrature(u, _q_deg, entities, it); // u
    auto [gradu, c_gu]
        = pack_gradient_quadrature(u, _q_deg, entities, it); // grad(u)
    auto [u_cd, c_uc] = Contact::pack_u_contact(i, u); // u on connected surface
    auto [u_gc, c_ugc]
        = Contact::pack_grad_u_contact(i, u); // grad(u) on connected surface

    // copy data into _coeffs in the order expected by the
    // integration kernel
    for (std::size_t e = 0; e < num_facets; ++e)
    {
      std::size_t offset = offset0;
      std::copy_n(std::next(u_p.begin(), e * c_u), c_u,
                  std::next(coeffs[i].begin(), e * coeff_size + offset));
      offset += c_u;
      std::copy_n(std::next(gradu.begin(), e * c_gu), c_gu,
                  std::next(coeffs[i].begin(), e * coeff_size + offset));
      offset += c_gu;
      std::copy_n(std::next(u_cd.begin(), e * c_uc), c_uc,
                  std::next(coeffs[i].begin(), e * coeff_size + offset));
      offset += c_uc;
      std::copy_n(std::next(u_gc.begin(), e * c_ugc), c_ugc,
                  std::next(coeffs[i].begin(), e * coeff_size + offset));
    }
  }
}

void dolfinx_contact::MeshTie::generate_poisson_data_matrix_only(
    std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
    std::shared_ptr<dolfinx::fem::Function<double>> kdt, double gamma,
    double theta)
{
  // mesh data
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh = V->mesh();
  const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
  int tdim = mesh->topology()->dim();              // topological dimension

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  const std::size_t ndofs_cell = dofmap->cell_dofs(0).size();
  const std::size_t num_q_points
      = dolfinx_contact::Contact::quadrature_rule()->offset()[1]
        - dolfinx_contact::Contact::quadrature_rule()->offset()[0];

  const std::size_t max_links = dolfinx_contact::Contact::max_links();
  // Coefficient offsets
  //  Expecting coefficients in following order:
  //  h, test_fn, grad(test_fn), T, grad(T), T_opposite,
  // grad(T_opposite)
  std::vector<std::size_t> cstrides
      = {2,
         num_q_points * ndofs_cell * max_links,
         num_q_points * ndofs_cell * gdim * max_links,
         num_q_points,
         num_q_points * gdim,
         num_q_points,
         num_q_points * gdim};

  _cstride_poisson = std::accumulate(cstrides.cbegin(), cstrides.cend(), 0);
  _kernel_rhs_poisson = dolfinx_contact::generate_poisson_kernel(
      Kernel::MeshTieRhs, V, Contact::quadrature_rule(), cstrides);
  _kernel_jac_poisson = dolfinx_contact::generate_poisson_kernel(
      Kernel::MeshTieJac, V, Contact::quadrature_rule(), cstrides);

  // save nitsche parameters as constants
  _consts_poisson = {gamma, theta};

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
    auto it = dolfinx::fem::IntegralType::exterior_facet;
    auto [kdt_p, c_kdt]
        = pack_coefficient_quadrature(kdt, 0, entities, it); // lambda
    auto [testfn, ctest] = Contact::pack_test_functions(i, V); // test functions
    auto [gradtst, cgt] = Contact::pack_grad_test_functions(
        i, V); // test fns on connected surface

    // copy data into one common data vector in the order expected by the
    // integration kernel
    std::vector<double> coeffs(_cstride_poisson * num_facets);
    for (std::size_t e = 0; e < num_facets; ++e)
    {
      std::copy_n(std::next(h_p.begin(), e * c_h), c_h,
                  std::next(coeffs.begin(), e * _cstride_poisson));
      std::size_t offset = c_h;
      std::copy_n(std::next(kdt_p.begin(), e * c_kdt), c_kdt,
                  std::next(coeffs.begin(), e * _cstride_poisson + offset));  
      offset += c_kdt;
      std::copy_n(std::next(testfn.begin(), e * ctest), ctest,
                  std::next(coeffs.begin(), e * _cstride_poisson + offset));
      offset += ctest;
      std::copy_n(std::next(gradtst.begin(), e * cgt), cgt,
                  std::next(coeffs.begin(), e * _cstride_poisson + offset));
    }

    _coeffs_poisson[i] = coeffs;
  }
}

void dolfinx_contact::MeshTie::generate_poisson_data(
    std::shared_ptr<dolfinx::fem::Function<double>> T,
    std::shared_ptr<dolfinx::fem::Function<double>> kdt, double gamma,
    double theta)
{
  // generate the data used for the matrix
  generate_poisson_data_matrix_only(T->function_space(), kdt, gamma, theta);
  update_meshtie_data(T, _coeffs_poisson, offset_poisson(T->function_space()),
                      _cstride_poisson);
}

void dolfinx_contact::MeshTie::assemble_vector(
    std::span<PetscScalar> b,
    std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
    dolfinx_contact::Problem problem_type)
{
  switch (problem_type)
  {
  using enum dolfinx_contact::Problem;
  case Elasticity:
    for (int i = 0; i < _num_pairs; ++i)
      assemble_vector(b, i, _kernel_rhs, _coeffs[i], (int)_cstride, _consts, V);
    break;
  case Poisson:
    for (int i = 0; i < _num_pairs; ++i)
      assemble_vector(b, i, _kernel_rhs_poisson, _coeffs_poisson[i],
                      (int)_cstride_poisson, _consts_poisson, V);
    break;
  case ThermoElasticity:
    for (int i = 0; i < _num_pairs; ++i)
      assemble_vector(b, i, _kernel_rhs, _coeffs[i], (int)_cstride, _consts, V);
    break;
  default:
    throw std::invalid_argument("Problem type not implemented");
  }
}

void dolfinx_contact::MeshTie::assemble_matrix(
    const mat_set_fn& mat_set,
    std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
    dolfinx_contact::Problem problem_type)
{
  switch (problem_type)
  {
  using enum dolfinx_contact::Problem;
  case Elasticity:
    for (int i = 0; i < _num_pairs; ++i)
      assemble_matrix(mat_set, i, _kernel_jac, _coeffs[i], (int)_cstride,
                      _consts, V);
    break;
  case Poisson:
    for (int i = 0; i < _num_pairs; ++i)
      assemble_matrix(mat_set, i, _kernel_jac_poisson, _coeffs_poisson[i],
                      (int)_cstride_poisson, _consts_poisson, V);
    break;
  case ThermoElasticity:
    for (int i = 0; i < _num_pairs; ++i)
      assemble_matrix(mat_set, i, _kernel_jac, _coeffs[i], (int)_cstride,
                      _consts, V);
    break;
  default:
    throw std::invalid_argument("Problem type not implemented");
  }
}

std::pair<std::vector<double>, std::size_t>
dolfinx_contact::MeshTie::coeffs(int pair)
{
  std::vector<double>& coeffs = _coeffs[pair];
  return {coeffs, _cstride};
}
