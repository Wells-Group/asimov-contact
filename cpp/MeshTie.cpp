// Copyright (C) 2023 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#include "MeshTie.h"

//-----------------------------------------------------------------------------
dolfinx_contact::MeshTie::MeshTie(
    const std::vector<
        std::shared_ptr<const dolfinx::mesh::MeshTags<std::int32_t>>>& markers,
    const dolfinx::graph::AdjacencyList<std::int32_t>& surfaces,
    const std::vector<std::array<int, 2>>& connected_pairs,
    std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh, int q_deg)
    : Contact::Contact(markers, surfaces, connected_pairs, mesh,
                       std::vector<ContactMode>(connected_pairs.size(),
                                                ContactMode::ClosestPoint),
                       q_deg)
{
  // Find closest points
  for (std::size_t i = 0; i < connected_pairs.size(); ++i)
  {
    Contact::create_distance_map(i);
    std::array<int, 2> pair = Contact::contact_pair(i);
    std::size_t num_facets = Contact::local_facets(pair[0]);
    if (num_facets > 0)
    {
      auto [ny, cstride1] = Contact::pack_ny(i);
      auto [gap, cstride] = Contact::pack_gap(i);

      std::span<const std::int32_t> entities
          = Contact::active_entities(pair[0]);

      // Retrieve cells connected to integration facets
      std::vector<std::int32_t> cells(num_facets);
      for (std::size_t e = 0; e < num_facets; ++e)
        cells[e] = entities[2 * e];
      std::vector<double> h_p
          = dolfinx::mesh::h(*mesh, cells, mesh->topology()->dim());
      Contact::crop_invalid_points(i, gap, ny,
                                   *std::max_element(h_p.begin(), h_p.end()));
    }
  }
  // initialise internal variables
  _num_pairs = connected_pairs.size();
  _coeffs.resize(_num_pairs);
  _coeffs_poisson.resize(_num_pairs);
  _q_deg = q_deg;
};
//-----------------------------------------------------------------------------
void dolfinx_contact::MeshTie::generate_kernel_data(
    dolfinx_contact::Problem problem_type,
    const dolfinx::fem::FunctionSpace<double>& V,
    const std::map<std::string,
                   std::shared_ptr<const dolfinx::fem::Function<double>>>&
        coefficients,
    double gamma, double theta)
{
  std::vector<std::shared_ptr<const dolfinx::fem::Function<double>>> coeff_list;
  switch (problem_type)
  {
  case Problem::Elasticity:
    if (auto it = coefficients.find("mu"); it != coefficients.end())
      coeff_list.push_back(it->second);
    else
      throw std::invalid_argument("Lame parameter mu not provided.");

    if (auto it = coefficients.find("lambda"); it != coefficients.end())
      coeff_list.push_back(it->second);
    else
      throw std::invalid_argument("Lame parameter lambda not provided.");

    generate_meshtie_data_matrix_only(problem_type, V, coeff_list, gamma,
                                      theta);
    if (auto it = coefficients.find("u"); it != coefficients.end())
      update_kernel_data(coefficients, problem_type);
    break;
  case Problem::Poisson:
    if (auto it = coefficients.find("kdt"); it != coefficients.end())
      coeff_list.push_back(it->second);
    else
      throw std::invalid_argument("kdt not provided.");

    generate_poisson_data_matrix_only(V, *coeff_list[0], gamma, theta);
    if (auto it = coefficients.find("T"); it != coefficients.end())
      update_kernel_data(coefficients, problem_type);
    break;
  case Problem::ThermoElasticity:
    if (auto it = coefficients.find("mu"); it != coefficients.end())
      coeff_list.push_back(it->second);
    else
      throw std::invalid_argument("Lame parameter mu not provided.");

    if (auto it = coefficients.find("lambda"); it != coefficients.end())
      coeff_list.push_back(it->second);
    else
      throw std::invalid_argument("Lame parameter lambda not provided.");

    if (auto it = coefficients.find("alpha"); it != coefficients.end())
      coeff_list.push_back(it->second);
    else
      throw std::invalid_argument("Thermal expansion coefficient.");

    generate_meshtie_data_matrix_only(problem_type, V, coeff_list, gamma,
                                      theta);
    if (auto it = coefficients.find("u"); it != coefficients.end())
      update_kernel_data(coefficients, problem_type);
    break;
  default:
    throw std::invalid_argument("Problem type not implemented");
  }
}
//-----------------------------------------------------------------------------
void dolfinx_contact::MeshTie::generate_meshtie_data_matrix_only(
    Problem problem_type, const dolfinx::fem::FunctionSpace<double>& V,
    std::vector<std::shared_ptr<const dolfinx::fem::Function<double>>> coeffs,
    double gamma, double theta)
{
  // mesh data
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh = V.mesh();
  const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
  int tdim = mesh->topology()->dim();              // topological dimension

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V.dofmap();
  const std::size_t ndofs_cell = dofmap->cell_dofs(0).size();
  const std::size_t bs = dofmap->bs();
  const std::size_t num_q_points = Contact::quadrature_rule().offset()[1]
                                   - Contact::quadrature_rule().offset()[0];

  const std::size_t max_links = Contact::max_links();
  // Coefficient offsets
  // Coefficient offsets
  // Expecting coefficients in following order:
  // mu, lmbda, h,test_fn, grad(test_fn), u, grad(u), u_opposite,
  // grad(u_opposite)

  std::vector<std::size_t> cstrides
      = {3,
         num_q_points * ndofs_cell * bs * max_links,
         num_q_points * ndofs_cell * bs * max_links,
         num_q_points * gdim,
         num_q_points * gdim * gdim,
         num_q_points * bs,
         num_q_points * gdim * bs};

  if (problem_type == Problem::ThermoElasticity)
  {
    cstrides[0] = 4;
    cstrides.push_back(num_q_points);
    cstrides.push_back(num_q_points);
    _kernel_thermo_el = generate_meshtie_kernel(
        Kernel::ThermoElasticRhs, V, Contact::quadrature_rule(), cstrides);
  }

  // Generate integration kernels
  _kernel_rhs = generate_meshtie_kernel(Kernel::MeshTieRhs, V,
                                        Contact::quadrature_rule(), cstrides);
  _kernel_jac = generate_meshtie_kernel(Kernel::MeshTieJac, V,
                                        Contact::quadrature_rule(), cstrides);

  // save nitsche parameters as constants
  _consts = {gamma, theta};
  auto it = dolfinx::fem::IntegralType::exterior_facet;
  _cstride = std::accumulate(cstrides.cbegin(), cstrides.cend(), 0);

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
        = pack_coefficient_quadrature(*coeffs[1], 0, entities, it); // lambda
    auto [mu_p, c_mu]
        = pack_coefficient_quadrature(*coeffs[0], 0, entities, it); // mu
    auto [gap, cgap] = Contact::pack_gap(i);                   // gap function
    auto [testfn, ctest] = Contact::pack_test_functions(i, V); // test functions
    auto [gradtst, cgt] = Contact::pack_grad_test_functions(
        i, V); // test fns on connected surface

    // copy data into one common data vector in the order expected by the
    // integration kernel
    _coeffs[i].resize(_cstride * num_facets);
    for (std::size_t e = 0; e < num_facets; ++e)
    {
      std::copy_n(std::next(mu_p.begin(), e * c_mu), c_mu,
                  std::next(_coeffs[i].begin(), e * _cstride));
      std::size_t offset = c_mu;
      std::copy_n(std::next(lm_p.begin(), e * c_lm), c_lm,
                  std::next(_coeffs[i].begin(), e * _cstride + offset));
      offset += c_lm;
      std::copy_n(std::next(h_p.begin(), e * c_h), c_h,
                  std::next(_coeffs[i].begin(), e * _cstride + offset));

      offset = cstrides[0];
      std::copy_n(std::next(testfn.begin(), e * ctest), ctest,
                  std::next(_coeffs[i].begin(), e * _cstride + offset));
      offset += ctest;
      std::copy_n(std::next(gradtst.begin(), e * cgt), cgt,
                  std::next(_coeffs[i].begin(), e * _cstride + offset));
    }

    if (problem_type == Problem::ThermoElasticity)
    {
      auto [alpha, c_alpha]
          = pack_coefficient_quadrature(*coeffs[2], 0, entities, it); // alpha
      for (std::size_t e = 0; e < num_facets; ++e)
      {
        std::copy_n(std::next(alpha.begin(), e * c_alpha), c_alpha,
                    std::next(_coeffs[i].begin(), e * _cstride + 3));
      }
    }
  }
}
//-----------------------------------------------------------------------------
void dolfinx_contact::MeshTie::update_kernel_data(
    const std::map<std::string,
                   std::shared_ptr<const dolfinx::fem::Function<double>>>&
        coefficients,
    Problem problem_type)
{
  // declare variables
  std::size_t gdim = 0; // geometrical dimension
  std::size_t ndofs_cell = 0;
  std::size_t bs = 1;
  std::size_t num_pts = Contact::num_q_points();
  std::size_t max_links = Contact::max_links();
  std::size_t offset0 = 0;
  std::size_t offset1 = 0;

  std::vector<std::shared_ptr<const dolfinx::fem::Function<double>>> coeff_list;
  switch (problem_type)
  {
  case Problem::Elasticity:
    if (auto it = coefficients.find("u"); it != coefficients.end())
      coeff_list.push_back(it->second);
    else
      throw std::invalid_argument("Displacement function u not provided.");

    gdim = coeff_list[0]->function_space()->mesh()->geometry().dim();
    ndofs_cell = coeff_list[0]->function_space()->dofmap()->cell_dofs(0).size();
    bs = coeff_list[0]->function_space()->dofmap()->bs();
    offset0 = 3 + 2 * (num_pts * max_links * bs * ndofs_cell);
    offset1 = offset0 + (1 + gdim) * num_pts * bs;
    update_function_data(*coeff_list[0], _coeffs, offset0, offset1, _cstride);
    offset0 += num_pts * bs;
    offset1 += num_pts * bs;
    update_gradient_data(*coeff_list[0], _coeffs, offset0, offset1, _cstride);
    break;
  case Problem::Poisson:
    if (auto it = coefficients.find("T"); it != coefficients.end())
      coeff_list.push_back(it->second);
    else
      throw std::invalid_argument("Function T not provided.");

    gdim = coeff_list[0]->function_space()->mesh()->geometry().dim();
    ndofs_cell = coeff_list[0]->function_space()->dofmap()->cell_dofs(0).size();
    offset0 = 2 + (1 + gdim) * (num_pts * max_links * ndofs_cell);
    offset1 = offset0 + (1 + gdim) * num_pts;
    update_function_data(*coeff_list[0], _coeffs_poisson, offset0, offset1,
                         _cstride_poisson);
    offset0 += num_pts;
    offset1 += num_pts;
    update_gradient_data(*coeff_list[0], _coeffs_poisson, offset0, offset1,
                         _cstride_poisson);
    break;
  case Problem::ThermoElasticity:
    if (auto it = coefficients.find("u"); it != coefficients.end())
      coeff_list.push_back(it->second);
    else
      throw std::invalid_argument("Displacement function u not provided.");

    if (auto it = coefficients.find("T"); it != coefficients.end())
      coeff_list.push_back(it->second);
    else
      throw std::invalid_argument("Temparature function T not provided.");

    gdim = coeff_list[0]->function_space()->mesh()->geometry().dim();
    ndofs_cell = coeff_list[0]->function_space()->dofmap()->cell_dofs(0).size();
    bs = coeff_list[0]->function_space()->dofmap()->bs();
    offset0 = 4 + 2 * (num_pts * max_links * bs * ndofs_cell);
    offset1 = offset0 + (1 + gdim) * num_pts * bs;
    update_function_data(*coeff_list[0], _coeffs, offset0, offset1, _cstride);
    offset0 += num_pts * bs;
    offset1 += num_pts * bs;
    update_gradient_data(*coeff_list[0], _coeffs, offset0, offset1, _cstride);
    offset0 = offset1 + num_pts * bs * gdim;
    offset1 = offset0 + num_pts;
    update_function_data(*coeff_list[1], _coeffs, offset0, offset1, _cstride);
    break;
  default:
    throw std::invalid_argument("Problem type not implemented");
  }
}
//-----------------------------------------------------------------------------
void dolfinx_contact::MeshTie::update_function_data(
    const dolfinx::fem::Function<double>& u,
    std::vector<std::vector<double>>& coeffs, std::size_t offset0,
    std::size_t offset1, std::size_t coeff_size)
{
  std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V
      = u.function_space();                                            // mesh
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
    auto [u_cd, c_uc] = Contact::pack_u_contact(i, u); // u on connected surface

    // copy data into _coeffs in the order expected by the
    // integration kernel
    for (std::size_t e = 0; e < num_facets; ++e)
    {
      std::copy_n(std::next(u_p.begin(), e * c_u), c_u,
                  std::next(coeffs[i].begin(), e * coeff_size + offset0));
      std::copy_n(std::next(u_cd.begin(), e * c_uc), c_uc,
                  std::next(coeffs[i].begin(), e * coeff_size + offset1));
    }
  }
}
//-----------------------------------------------------------------------------
void dolfinx_contact::MeshTie::update_gradient_data(
    const dolfinx::fem::Function<double>& u,
    std::vector<std::vector<double>>& coeffs, std::size_t offset0,
    std::size_t offset1, std::size_t coeff_size)
{
  std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V
      = u.function_space();                                            // mesh
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
    auto [gradu, c_gu]
        = pack_gradient_quadrature(u, _q_deg, entities, it); // grad(u)
    auto [u_gc, c_ugc]
        = Contact::pack_grad_u_contact(i, u); // grad(u) on connected surface

    // copy data into _coeffs in the order expected by the
    // integration kernel
    for (std::size_t e = 0; e < num_facets; ++e)
    {
      std::copy_n(std::next(gradu.begin(), e * c_gu), c_gu,
                  std::next(coeffs[i].begin(), e * coeff_size + offset0));

      std::copy_n(std::next(u_gc.begin(), e * c_ugc), c_ugc,
                  std::next(coeffs[i].begin(), e * coeff_size + offset1));
    }
  }
}
//-----------------------------------------------------------------------------
void dolfinx_contact::MeshTie::generate_poisson_data_matrix_only(
    const dolfinx::fem::FunctionSpace<double>& V,
    const dolfinx::fem::Function<double>& kdt, double gamma, double theta)
{
  // mesh data
  std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh = V.mesh();
  const std::size_t gdim = mesh->geometry().dim(); // geometrical dimension
  int tdim = mesh->topology()->dim();              // topological dimension

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V.dofmap();
  const std::size_t ndofs_cell = dofmap->cell_dofs(0).size();
  const std::size_t num_q_points = Contact::quadrature_rule().offset()[1]
                                   - Contact::quadrature_rule().offset()[0];

  const std::size_t max_links = Contact::max_links();
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
  _kernel_rhs_poisson = generate_poisson_kernel(
      Kernel::MeshTieRhs, V, Contact::quadrature_rule(), cstrides);
  _kernel_jac_poisson = generate_poisson_kernel(
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
        = pack_coefficient_quadrature(kdt, 0, entities, it);   // lambda
    auto [testfn, ctest] = Contact::pack_test_functions(i, V); // test functions
    auto [gradtst, cgt] = Contact::pack_grad_test_functions(
        i, V); // test fns on connected surface

    // copy data into one common data vector in the order expected by the
    // integration kernel
    _coeffs_poisson[i].resize(_cstride_poisson * num_facets);
    for (std::size_t e = 0; e < num_facets; ++e)
    {
      std::copy_n(std::next(h_p.begin(), e * c_h), c_h,
                  std::next(_coeffs_poisson[i].begin(), e * _cstride_poisson));
      std::size_t offset = c_h;
      std::copy_n(
          std::next(kdt_p.begin(), e * c_kdt), c_kdt,
          std::next(_coeffs_poisson[i].begin(), e * _cstride_poisson + offset));
      offset += c_kdt;
      std::copy_n(
          std::next(testfn.begin(), e * ctest), ctest,
          std::next(_coeffs_poisson[i].begin(), e * _cstride_poisson + offset));
      offset += ctest;
      std::copy_n(
          std::next(gradtst.begin(), e * cgt), cgt,
          std::next(_coeffs_poisson[i].begin(), e * _cstride_poisson + offset));
    }
  }
}
//-----------------------------------------------------------------------------
void dolfinx_contact::MeshTie::assemble_vector(
    std::span<PetscScalar> b, const dolfinx::fem::FunctionSpace<double>& V,
    Problem problem_type)
{
  switch (problem_type)
  {
  case Problem::Elasticity:
    for (int i = 0; i < _num_pairs; ++i)
    {
      Contact::assemble_vector(b, i, _kernel_rhs, _coeffs[i], _cstride, _consts,
                               V);
    }
    break;
  case Problem::Poisson:
    for (int i = 0; i < _num_pairs; ++i)
    {
      Contact::assemble_vector(b, i, _kernel_rhs_poisson, _coeffs_poisson[i],
                               _cstride_poisson, _consts_poisson, V);
    }
    break;
  case Problem::ThermoElasticity:
    for (int i = 0; i < _num_pairs; ++i)
    {
      Contact::assemble_vector(b, i, _kernel_rhs, _coeffs[i], _cstride, _consts,
                               V);
      Contact::assemble_vector(b, i, _kernel_thermo_el, _coeffs[i], _cstride,
                               _consts, V);
    }
    break;
  default:
    throw std::invalid_argument("Problem type not implemented");
  }
}
//-----------------------------------------------------------------------------
void dolfinx_contact::MeshTie::assemble_matrix(
    const mat_set_fn& mat_set, const dolfinx::fem::FunctionSpace<double>& V,
    Problem problem_type)
{
  switch (problem_type)
  {
  case Problem::Elasticity:
    for (int i = 0; i < _num_pairs; ++i)
    {
      Contact::assemble_matrix(mat_set, i, _kernel_jac, _coeffs[i], _cstride,
                               _consts, V);
    }
    break;
  case Problem::Poisson:
    for (int i = 0; i < _num_pairs; ++i)
    {
      Contact::assemble_matrix(mat_set, i, _kernel_jac_poisson,
                               _coeffs_poisson[i], _cstride_poisson,
                               _consts_poisson, V);
    }
    break;
  case Problem::ThermoElasticity:
    for (int i = 0; i < _num_pairs; ++i)
    {
      Contact::assemble_matrix(mat_set, i, _kernel_jac, _coeffs[i], _cstride,
                               _consts, V);
    }
    break;
  default:
    throw std::invalid_argument("Problem type not implemented");
  }
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::size_t>
dolfinx_contact::MeshTie::coeffs(int pair)
{
  std::vector<double>& coeffs = _coeffs[pair];
  return {coeffs, _cstride};
}
//-----------------------------------------------------------------------------
