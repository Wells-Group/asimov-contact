// Copyright (C) 2021-2022 Jørgen S. Dokken and Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT
#include "utils.h"
#include "geometric_quantities.h"

using namespace dolfinx_contact;

//-----------------------------------------------------------------------------
void dolfinx_contact::pull_back(xt::xtensor<double, 3>& J,
                                xt::xtensor<double, 3>& K,
                                xt::xtensor<double, 1>& detJ,
                                const xt::xtensor<double, 2>& x,
                                xt::xtensor<double, 2>& X,
                                const xt::xtensor<double, 2>& coordinate_dofs,
                                const dolfinx::fem::CoordinateElement& cmap)
{
  // number of points
  const std::size_t num_points = x.shape(0);
  assert(J.shape(0) >= num_points);
  assert(K.shape(0) >= num_points);
  assert(detJ.shape(0) >= num_points);

  // Get mesh data from input
  const size_t tdim = K.shape(1);

  // -- Lambda function for affine pull-backs
  xt::xtensor<double, 4> data(cmap.tabulate_shape(1, 1));
  const xt::xtensor<double, 2> X0(xt::zeros<double>({std::size_t(1), tdim}));
  cmap.tabulate(1, X0, data);
  const xt::xtensor<double, 2> dphi_i
      = xt::view(data, xt::range(1, tdim + 1), 0, xt::all(), 0);
  auto pull_back_affine = [dphi_i](auto&& X, const auto& cell_geometry,
                                   auto&& J, auto&& K, const auto& x) mutable
  {
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi_i, cell_geometry, J);
    dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
    dolfinx::fem::CoordinateElement::pull_back_affine(
        X, K, dolfinx::fem::CoordinateElement::x0(cell_geometry), x);
  };

  xt::xtensor<double, 2> dphi;

  if (cmap.is_affine())
  {
    xt::xtensor<double, 4> phi(cmap.tabulate_shape(1, 1));
    J.fill(0);
    pull_back_affine(X, coordinate_dofs, xt::view(J, 0, xt::all(), xt::all()),
                     xt::view(K, 0, xt::all(), xt::all()), x);
    detJ[0] = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(
        xt::view(J, 0, xt::all(), xt::all()));
    for (std::size_t p = 1; p < num_points; ++p)
    {
      xt::view(J, p, xt::all(), xt::all())
          = xt::view(J, 0, xt::all(), xt::all());
      xt::view(K, p, xt::all(), xt::all())
          = xt::view(K, 0, xt::all(), xt::all());
      detJ[p] = detJ[0];
    }
  }
  else
  {
    cmap.pull_back_nonaffine(X, x, coordinate_dofs);
    xt::xtensor<double, 4> phi(cmap.tabulate_shape(1, X.shape(0)));
    cmap.tabulate(1, X, phi);
    J.fill(0);
    for (std::size_t p = 0; p < X.shape(0); ++p)
    {
      auto _J = xt::view(J, p, xt::all(), xt::all());
      dphi = xt::view(phi, xt::range(1, tdim + 1), p, xt::all(), 0);
      dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs,
                                                        _J);
      dolfinx::fem::CoordinateElement::compute_jacobian_inverse(
          _J, xt::view(K, p, xt::all(), xt::all()));
      detJ[p]
          = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(_J);
    }
  }
}

//-----------------------------------------------------------------------------
xt::xtensor<double, 3> dolfinx_contact::get_basis_functions(
    xt::xtensor<double, 3>& J, xt::xtensor<double, 3>& K,
    xt::xtensor<double, 1>& detJ, const xt::xtensor<double, 2>& x,
    const xt::xtensor<double, 2>& coordinate_dofs, const std::int32_t index,
    const std::int32_t perm,
    std::shared_ptr<const dolfinx::fem::FiniteElement> element,
    const dolfinx::fem::CoordinateElement& cmap)
{
  // number of points
  const std::size_t num_points = x.shape(0);
  assert(J.shape(0) >= num_points);
  assert(K.shape(0) >= num_points);
  assert(detJ.shape(0) >= num_points);

  // Get mesh data from input
  const size_t tdim = K.shape(1);

  // Get element data
  const size_t block_size = element->block_size();
  const size_t reference_value_size
      = element->reference_value_size() / block_size;
  const size_t value_size = element->value_size() / block_size;
  const size_t space_dimension = element->space_dimension() / block_size;

  xt::xtensor<double, 2> X({x.shape(0), tdim});

  // Skip negative cell indices
  xt::xtensor<double, 3> basis_array = xt::zeros<double>(
      {num_points, space_dimension * block_size, value_size * block_size});
  if (index < 0)
    return basis_array;

  pull_back(J, K, detJ, x, X, coordinate_dofs, cmap);
  // Prepare basis function data structures
  xt::xtensor<double, 4> tabulated_data(
      {1, num_points, space_dimension, reference_value_size});
  xt::xtensor<double, 3> basis_values(
      {num_points, space_dimension, value_size});

  // Get push forward function
  xt::xtensor<double, 2> point_basis_values({space_dimension, value_size});
  using u_t = xt::xview<decltype(basis_values)&, std::size_t,
                        xt::xall<std::size_t>, xt::xall<std::size_t>>;
  using U_t = xt::xview<decltype(point_basis_values)&, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  using J_t = xt::xview<decltype(J)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  using K_t = xt::xview<decltype(K)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  // FIXME: These should be moved out of this function
  auto push_forward_fn = element->map_fn<u_t, U_t, J_t, K_t>();
  const std::function<void(const xtl::span<PetscScalar>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      transformation = element->get_dof_transformation_function<PetscScalar>();

  // Compute basis on reference element
  element->tabulate(tabulated_data, X, 0);
  for (std::size_t q = 0; q < num_points; ++q)
  {
    // Permute the reference values to account for the cell's orientation
    point_basis_values = xt::view(tabulated_data, 0, q, xt::all(), xt::all());
    element->apply_dof_transformation(
        xtl::span<double>(point_basis_values.data(), point_basis_values.size()),
        perm, 1);
    // Push basis forward to physical element
    auto _J = xt::view(J, q, xt::all(), xt::all());
    auto _K = xt::view(K, q, xt::all(), xt::all());
    auto _u = xt::view(basis_values, q, xt::all(), xt::all());
    auto _U = xt::view(point_basis_values, xt::all(), xt::all());
    push_forward_fn(_u, _U, _J, detJ[q], _K);
  }

  // Expand basis values for each dof
  for (std::size_t p = 0; p < num_points; ++p)
  {
    for (std::size_t block = 0; block < block_size; ++block)
    {
      for (std::size_t i = 0; i < space_dimension; ++i)
      {
        for (std::size_t j = 0; j < value_size; ++j)
        {
          basis_array(p, i * block_size + block, j * block_size + block)
              = basis_values(p, i, j);
        }
      }
    }
  }
  return basis_array;
}

//-----------------------------------------------------------------------------
std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
dolfinx_contact::sort_cells(const xtl::span<const std::int32_t>& cells,
                            const xtl::span<std::int32_t>& perm)
{
  assert(perm.size() == cells.size());

  const auto num_cells = (std::int32_t)cells.size();
  std::vector<std::int32_t> unique_cells(num_cells);
  std::vector<std::int32_t> offsets(num_cells + 1, 0);
  std::iota(perm.begin(), perm.end(), 0);
  dolfinx::argsort_radix<std::int32_t>(cells, perm);

  // Sort cells in accending order
  for (std::int32_t i = 0; i < num_cells; ++i)
    unique_cells[i] = cells[perm[i]];

  // Compute the number of identical cells
  std::int32_t index = 0;
  for (std::int32_t i = 0; i < num_cells - 1; ++i)
    if (unique_cells[i] != unique_cells[i + 1])
      offsets[++index] = i + 1;

  offsets[index + 1] = num_cells;
  unique_cells.erase(std::unique(unique_cells.begin(), unique_cells.end()),
                     unique_cells.end());
  offsets.resize(unique_cells.size() + 1);

  return std::make_pair(unique_cells, offsets);
}

//-------------------------------------------------------------------------------------
void dolfinx_contact::update_geometry(
    const dolfinx::fem::Function<PetscScalar>& u,
    std::shared_ptr<dolfinx::mesh::Mesh> mesh)
{
  auto V = u.function_space();
  assert(V);
  auto dofmap = V->dofmap();
  assert(dofmap);
  // Check that mesh to be updated and underlying mesh of u are the same
  assert(mesh == V->mesh());

  // The Function and the mesh must have identical element_dof_layouts
  // (up to the block size)
  assert(dofmap->element_dof_layout()
         == mesh->geometry().cmap().create_dof_layout());

  const int tdim = mesh->topology().dim();
  auto cell_map = mesh->topology().index_map(tdim);
  assert(cell_map);
  const std::int32_t num_cells
      = cell_map->size_local() + cell_map->num_ghosts();

  // Get dof array and retrieve u at the mesh dofs
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap_x
      = mesh->geometry().dofmap();
  const int bs = dofmap->bs();
  const auto& u_data = u.x()->array();
  xtl::span<double> coords = mesh->geometry().x();
  std::vector<double> dx(coords.size());
  for (std::int32_t c = 0; c < num_cells; ++c)
  {
    auto dofs = dofmap->cell_dofs(c);
    auto dofs_x = dofmap_x.links(c);
    for (std::size_t i = 0; i < dofs.size(); ++i)
      for (int j = 0; j < bs; ++j)
      {
        dx[3 * dofs_x[i] + j] = u_data[bs * dofs[i] + j];
      }
  }
  // add u to mesh dofs
  std::transform(coords.begin(), coords.end(), dx.begin(), coords.begin(),
                 std::plus<double>());
}
//-------------------------------------------------------------------------------------
double dolfinx_contact::R_plus(double x) { return 0.5 * (std::abs(x) + x); }
//-------------------------------------------------------------------------------------
double dolfinx_contact::R_minus(double x) { return 0.5 * (x - std::abs(x)); }
//-------------------------------------------------------------------------------------
double dolfinx_contact::dR_minus(double x) { return double(x < 0); }
//-------------------------------------------------------------------------------------

double dolfinx_contact::dR_plus(double x) { return double(x > 0); }
//-------------------------------------------------------------------------------------
std::array<std::size_t, 3>
dolfinx_contact::evaulate_basis_shape(const dolfinx::fem::FunctionSpace& V,
                                      const std::size_t num_points)
{
  // Get element
  assert(V.element());
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V.element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t value_size = element->value_size() / bs_element;
  const std::size_t space_dimension = element->space_dimension() / bs_element;
  return {num_points, space_dimension, value_size};
}
//-----------------------------------------------------------------------------
void dolfinx_contact::evaluate_basis_functions(
    const dolfinx::fem::FunctionSpace& V, const xt::xtensor<double, 2>& x,
    const xtl::span<const std::int32_t>& cells,
    xt::xtensor<double, 3>& basis_values)
{
  if (x.shape(0) != cells.size())
  {
    throw std::invalid_argument(
        "Number of points and number of cells must be equal.");
  }
  if (x.shape(0) != basis_values.shape(0))
  {
    throw std::invalid_argument("Length of array for basis values must be the "
                                "same as the number of points.");
  }
  if (x.shape(0) == 0)
    return;

  // Get mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V.mesh();
  assert(mesh);
  const dolfinx::mesh::Geometry& geometry = mesh->geometry();
  const dolfinx::mesh::Topology& topology = mesh->topology();

  // Get topology data
  const std::size_t tdim = topology.dim();
  auto map = topology.index_map((int)tdim);

  // Get geometry data
  const std::size_t gdim = geometry.dim();
  xtl::span<const double> x_g = geometry.x();
  const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
      = geometry.dofmap();
  const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
  const std::size_t num_dofs_g = cmap.dim();

  // Get element
  assert(V.element());
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V.element();
  assert(element);
  const int bs_element = element->block_size();
  const std::size_t reference_value_size
      = element->reference_value_size() / bs_element;
  const std::size_t space_dimension = element->space_dimension() / bs_element;

  // If the space has sub elements, concatenate the evaluations on the sub
  // elements
  if (const int num_sub_elements = element->num_sub_elements();
      num_sub_elements > 1 && num_sub_elements != bs_element)
  {
    throw std::invalid_argument("Canot evaluate basis functions for mixed "
                                "function spaces. Extract subspaces.");
  }

  // Get dofmap
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V.dofmap();
  assert(dofmap);

  xtl::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(topology.get_cell_permutation_info());
  }

  xt::xtensor<double, 2> coordinate_dofs
      = xt::zeros<double>({num_dofs_g, gdim});
  xt::xtensor<double, 2> xp = xt::zeros<double>({std::size_t(1), gdim});

  // -- Lambda function for affine pull-backs
  xt::xtensor<double, 4> data(cmap.tabulate_shape(1, 1));
  const xt::xtensor<double, 2> X0(xt::zeros<double>({std::size_t(1), tdim}));
  cmap.tabulate(1, X0, data);
  const xt::xtensor<double, 2> dphi_i
      = xt::view(data, xt::range(1, tdim + 1), 0, xt::all(), 0);
  auto pull_back_affine = [&dphi_i](auto&& X, const auto& cell_geometry,
                                    auto&& J, auto&& K, const auto& x) mutable
  {
    dolfinx::fem::CoordinateElement::compute_jacobian(dphi_i, cell_geometry, J);
    dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
    dolfinx::fem::CoordinateElement::pull_back_affine(
        X, K, dolfinx::fem::CoordinateElement::x0(cell_geometry), x);
  };

  xt::xtensor<double, 2> dphi;
  xt::xtensor<double, 2> X({x.shape(0), tdim});
  xt::xtensor<double, 3> J = xt::zeros<double>({x.shape(0), gdim, tdim});
  xt::xtensor<double, 3> K = xt::zeros<double>({x.shape(0), tdim, gdim});
  xt::xtensor<double, 1> detJ = xt::zeros<double>({x.shape(0)});
  xt::xtensor<double, 4> phi(cmap.tabulate_shape(1, 1));

  xt::xtensor<double, 2> _Xp({1, tdim});
  for (std::size_t p = 0; p < cells.size(); ++p)
  {
    const int cell_index = cells[p];

    // Skip negative cell indices
    if (cell_index < 0)
      continue;

    // Get cell geometry (coordinate dofs)
    auto x_dofs = x_dofmap.links(cell_index);
    for (std::size_t i = 0; i < num_dofs_g; ++i)
    {
      const int pos = 3 * x_dofs[i];
      for (std::size_t j = 0; j < gdim; ++j)
        coordinate_dofs(i, j) = x_g[pos + j];
    }

    for (std::size_t j = 0; j < gdim; ++j)
      xp(0, j) = x(p, j);

    auto _J = xt::view(J, p, xt::all(), xt::all());
    auto _K = xt::view(K, p, xt::all(), xt::all());

    // Compute reference coordinates X, and J, detJ and K
    if (cmap.is_affine())
    {
      pull_back_affine(_Xp, coordinate_dofs, _J, _K, xp);
      detJ[p]
          = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(_J);
    }
    else
    {
      cmap.pull_back_nonaffine(_Xp, xp, coordinate_dofs);
      cmap.tabulate(1, _Xp, phi);
      dphi = xt::view(phi, xt::range(1, tdim + 1), 0, xt::all(), 0);
      dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs,
                                                        _J);
      dolfinx::fem::CoordinateElement::compute_jacobian_inverse(_J, _K);
      detJ[p]
          = dolfinx::fem::CoordinateElement::compute_jacobian_determinant(_J);
    }
    xt::row(X, p) = xt::row(_Xp, 0);
  }

  // Prepare basis function data structures
  xt::xtensor<double, 4> basis_reference_values(
      {1, x.shape(0), space_dimension, reference_value_size});

  // Compute basis on reference element
  element->tabulate(basis_reference_values, X, 0);

  using u_t = xt::xview<decltype(basis_values)&, std::size_t,
                        xt::xall<std::size_t>, xt::xall<std::size_t>>;
  using U_t
      = xt::xview<decltype(basis_reference_values)&, std::size_t, std::size_t,
                  xt::xall<std::size_t>, xt::xall<std::size_t>>;
  using J_t = xt::xview<decltype(J)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  using K_t = xt::xview<decltype(K)&, std::size_t, xt::xall<std::size_t>,
                        xt::xall<std::size_t>>;
  auto push_forward_fn = element->map_fn<u_t, U_t, J_t, K_t>();
  const std::function<void(const xtl::span<double>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      apply_dof_transformation
      = element->get_dof_transformation_function<double>();
  const std::size_t num_basis_values = space_dimension * reference_value_size;
  for (std::size_t p = 0; p < cells.size(); ++p)
  {
    const int cell_index = cells[p];

    // Skip negative cell indices
    if (cell_index < 0)
      continue;

    // Permute the reference values to account for the cell's orientation
    apply_dof_transformation(
        xtl::span(basis_reference_values.data() + p * num_basis_values,
                  num_basis_values),
        cell_info, cell_index, (int)reference_value_size);

    // Push basis forward to physical element
    auto _K = xt::view(K, p, xt::all(), xt::all());
    auto _J = xt::view(J, p, xt::all(), xt::all());
    auto _u = xt::view(basis_values, p, xt::all(), xt::all());
    auto _U = xt::view(basis_reference_values, (std::size_t)0, p, xt::all(),
                       xt::all());
    push_forward_fn(_u, _U, _J, detJ[p], _K);
  }
};

double dolfinx_contact::compute_facet_jacobians(
    std::size_t q, xt::xtensor<double, 2>& J, xt::xtensor<double, 2>& K,
    xt::xtensor<double, 2>& J_tot, const xt::xtensor<double, 2>& J_f,
    const xt::xtensor<double, 3>& dphi, const xt::xtensor<double, 2>& coords)
{
  std::size_t gdim = J.shape(0);
  const xt::xtensor<double, 2>& dphi0_c
      = xt::view(dphi, xt::all(), q, xt::all());
  auto c_view = xt::view(coords, xt::all(), xt::range(0, gdim));
  std::fill(J.begin(), J.end(), 0.0);
  dolfinx::fem::CoordinateElement::compute_jacobian(dphi0_c, c_view, J);
  dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);
  std::fill(J_tot.begin(), J_tot.end(), 0.0);
  dolfinx::math::dot(J, J_f, J_tot);
  return std::fabs(
      dolfinx::fem::CoordinateElement::compute_jacobian_determinant(J_tot));
}
//-------------------------------------------------------------------------------------
std::function<double(
    std::size_t, double, xt::xtensor<double, 2>&, xt::xtensor<double, 2>&,
    xt::xtensor<double, 2>&, const xt::xtensor<double, 2>&,
    const xt::xtensor<double, 3>&, const xt::xtensor<double, 2>&)>
dolfinx_contact::get_update_jacobian_dependencies(
    const dolfinx::fem::CoordinateElement& cmap)
{
  if (cmap.is_affine())
  {
    // Return function that returns the input determinant
    return []([[maybe_unused]] std::size_t q, double detJ,
              [[maybe_unused]] xt::xtensor<double, 2>& J,
              [[maybe_unused]] xt::xtensor<double, 2>& K,
              [[maybe_unused]] xt::xtensor<double, 2>& J_tot,
              [[maybe_unused]] const xt::xtensor<double, 2>& J_f,
              [[maybe_unused]] const xt::xtensor<double, 3>& dphi,
              [[maybe_unused]] const xt::xtensor<double, 2>& coords)
    { return detJ; };
  }
  else
  {
    // Return function that returns the input determinant
    return [](std::size_t q, [[maybe_unused]] double detJ,
              xt::xtensor<double, 2>& J, xt::xtensor<double, 2>& K,
              xt::xtensor<double, 2>& J_tot, const xt::xtensor<double, 2>& J_f,
              const xt::xtensor<double, 3>& dphi,
              const xt::xtensor<double, 2>& coords)
    {
      double new_detJ = dolfinx_contact::compute_facet_jacobians(
          q, J, K, J_tot, J_f, dphi, coords);
      return new_detJ;
    };
  }
}
//-------------------------------------------------------------------------------------
std::function<void(xt::xtensor<double, 1>&, const xt::xtensor<double, 2>&,
                   const xt::xtensor<double, 2>&, const std::size_t)>
dolfinx_contact::get_update_normal(const dolfinx::fem::CoordinateElement& cmap)
{
  if (cmap.is_affine())
  {
    // Return function that returns the input determinant
    return []([[maybe_unused]] xt::xtensor<double, 1>& n,
              [[maybe_unused]] const xt::xtensor<double, 2>& K,
              [[maybe_unused]] const xt::xtensor<double, 2>& n_ref,
              [[maybe_unused]] const std::size_t local_index)
    {
      // Do nothing
    };
  }
  else
  {
    // Return function that updates the physical normal based on K
    return [](xt::xtensor<double, 1>& n, const xt::xtensor<double, 2>& K,
              const xt::xtensor<double, 2>& n_ref,
              const std::size_t local_index) {
      dolfinx_contact::physical_facet_normal(n, K, xt::row(n_ref, local_index));
    };
  }
}
//-------------------------------------------------------------------------------------

std::variant<std::vector<std::int32_t>,
             std::vector<std::pair<std::int32_t, int>>,
             std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>>
dolfinx_contact::compute_active_entities(
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
    tcb::span<const std::int32_t> entities, dolfinx::fem::IntegralType integral)
{

  // Determine variant type by integral
  std::variant<std::vector<std::int32_t>,
               std::vector<std::pair<std::int32_t, int>>,
               std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>>
      active_entities;
  switch (integral)
  {
  case dolfinx::fem::IntegralType::cell:
    active_entities = std::vector<std::int32_t>(entities.size());
    break;
  case dolfinx::fem::IntegralType::exterior_facet:
    active_entities
        = std::vector<std::pair<std::int32_t, int>>(entities.size());
    break;
  case dolfinx::fem::IntegralType::interior_facet:
    active_entities
        = std::vector<std::tuple<std::int32_t, int, std::int32_t, int>>(
            entities.size());
    break;
  default:
    throw std::runtime_error("Unknown integral type");
  }

  std::visit(
      [&](auto&& output)
      {
        const dolfinx::mesh::Topology& topology = mesh->topology();
        using U = std::decay_t<decltype(output)>;
        if constexpr (std::is_same_v<U, std::vector<std::int32_t>>)
        {
          // Do nothing if cell integral
          std::transform(entities.begin(), entities.end(), output.begin(),
                         [](std::int32_t cell) { return cell; });
        }
        else if constexpr (std::is_same_v<
                               U, std::vector<std::pair<std::int32_t, int>>>)
        {
          int tdim = topology.dim();
          auto f_to_c = topology.connectivity(tdim - 1, tdim);
          assert(f_to_c);
          auto c_to_f = topology.connectivity(tdim, tdim - 1);
          assert(c_to_f);
          for (std::size_t f = 0; f < entities.size(); f++)
          {
            assert(f_to_c->num_links(entities[f]) == 1);
            const std::int32_t cell = f_to_c->links(entities[f])[0];
            auto cell_facets = c_to_f->links(cell);

            auto facet_it = std::find(cell_facets.begin(), cell_facets.end(),
                                      entities[f]);
            assert(facet_it != cell_facets.end());
            const auto local_f = std::distance(cell_facets.begin(), facet_it);
            output[f] = {cell, (int)local_f};
          }
        }
        else if constexpr (std::is_same_v<
                               U, std::vector<std::tuple<std::int32_t, int,
                                                         std::int32_t, int>>>)
        {
          int tdim = topology.dim();
          auto f_to_c = topology.connectivity(tdim - 1, tdim);
          assert(f_to_c);
          auto c_to_f = topology.connectivity(tdim, tdim - 1);
          assert(c_to_f);
          std::array<std::pair<std::int32_t, int>, 2> interior_facets;
          for (std::size_t f = 0; f < entities.size(); f++)
          {
            assert(f_to_c->num_links(entities[f]) == 2);
            auto cells = f_to_c->links(entities[f]);
            for (std::size_t i = 0; i < 2; i++)
            {
              auto cell_facets = c_to_f->links(cells[i]);
              auto facet_it = std::find(cell_facets.begin(), cell_facets.end(),
                                        entities[f]);
              assert(facet_it != cell_facets.end());
              const auto local_f
                  = (int)std::distance(cell_facets.begin(), facet_it);
              interior_facets[i] = {cells[i], local_f};
            }
            output[f] = {interior_facets[0].first, interior_facets[0].second,
                         interior_facets[1].first, interior_facets[1].second};
          }
        }
      },
      active_entities);
  return active_entities;
}
//-------------------------------------------------------------------------------------
dolfinx::graph::AdjacencyList<std::int32_t>
dolfinx_contact::entities_to_geometry_dofs(
    const dolfinx::mesh::Mesh& mesh, int dim,
    const xtl::span<const std::int32_t>& entity_list)
{

  // Get mesh geometry and topology data
  const dolfinx::mesh::Geometry& geometry = mesh.geometry();
  const dolfinx::fem::ElementDofLayout layout
      = geometry.cmap().create_dof_layout();
  // FIXME: What does this return for prisms?
  const std::size_t num_entity_dofs = layout.num_entity_closure_dofs(dim);
  const graph::AdjacencyList<std::int32_t>& xdofs = geometry.dofmap();

  const mesh::Topology& topology = mesh.topology();
  const int tdim = topology.dim();
  mesh.topology_mutable().create_entities(dim);
  mesh.topology_mutable().create_connectivity(dim, tdim);
  mesh.topology_mutable().create_connectivity(tdim, dim);

  // Create arrays for the adjacency-list
  std::vector<std::int32_t> geometry_indices(num_entity_dofs
                                             * entity_list.size());
  std::vector<std::int32_t> offsets(entity_list.size() + 1, 0);
  for (std::size_t i = 0; i < entity_list.size(); ++i)
    offsets[i + 1] = std::int32_t((i + 1) * num_entity_dofs);

  // Fetch connectivities required to get entity dofs
  const std::vector<std::vector<std::vector<int>>>& closure_dofs
      = layout.entity_closure_dofs_all();
  const auto e_to_c = topology.connectivity(dim, tdim);
  assert(e_to_c);
  const auto c_to_e = topology.connectivity(tdim, dim);
  assert(c_to_e);
  for (std::size_t i = 0; i < entity_list.size(); ++i)
  {
    const std::int32_t idx = entity_list[i];
    const std::int32_t cell = e_to_c->links(idx)[0];
    auto cell_entities = c_to_e->links(cell);
    auto it = std::find(cell_entities.begin(), cell_entities.end(), idx);
    assert(it != cell_entities.end());
    const auto local_entity = std::distance(cell_entities.begin(), it);
    const std::vector<std::int32_t>& entity_dofs
        = closure_dofs[dim][local_entity];

    const auto xc = xdofs.links(cell);
    for (std::size_t j = 0; j < num_entity_dofs; ++j)
      geometry_indices[i * num_entity_dofs + j] = xc[entity_dofs[j]];
  }

  return dolfinx::graph::AdjacencyList<std::int32_t>(geometry_indices, offsets);
}