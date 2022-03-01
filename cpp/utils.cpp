// Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT
#include "utils.h"

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
void dolfinx_contact::pull_back_2(xt::xtensor<double, 3>& J,
                                  xt::xtensor<double, 3>& K,
                                  xt::xtensor<double, 3>& H,
                                  const xt::xtensor<double, 2>& x,
                                  xt::xtensor<double, 2>& X,
                                  const xt::xtensor<double, 2>& coordinate_dofs,
                                  const dolfinx::fem::CoordinateElement& cmap)
{
  // number of points
  const std::size_t num_points = x.shape(0);
  assert(J.shape(0) >= num_points);
  assert(K.shape(0) >= num_points);

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
  xt::xtensor<double, 2> ddphi;

  if (cmap.is_affine())
  {
    xt::xtensor<double, 4> phi(cmap.tabulate_shape(1, 1));
    J.fill(0);
    pull_back_affine(X, coordinate_dofs, xt::view(J, 0, xt::all(), xt::all()),
                     xt::view(K, 0, xt::all(), xt::all()), x);
    for (std::size_t p = 1; p < num_points; ++p)
    {
      xt::view(J, p, xt::all(), xt::all())
          = xt::view(J, 0, xt::all(), xt::all());
      xt::view(K, p, xt::all(), xt::all())
          = xt::view(K, 0, xt::all(), xt::all());
    }
    H.fill(0);
  }
  else
  {
    cmap.pull_back_nonaffine(X, x, coordinate_dofs);
    // For the non-affine case we need the second derivative
    // in the affine case H is left untouched as it is just 0
    xt::xtensor<double, 4> phi(cmap.tabulate_shape(2, X.shape(0)));
    cmap.tabulate(2, X, phi);
    J.fill(0);
    H.fill(0);
    for (std::size_t p = 0; p < X.shape(0); ++p)
    {
      auto _J = xt::view(J, p, xt::all(), xt::all());
      auto _H = xt::view(H, p, xt::all(), xt::all());
      dphi = xt::view(phi, xt::range(1, tdim + 1), p, xt::all(), 0);
      dolfinx::fem::CoordinateElement::compute_jacobian(dphi, coordinate_dofs,
                                                        _J);

      dolfinx::fem::CoordinateElement::compute_jacobian_inverse(
          _J, xt::view(K, p, xt::all(), xt::all()));
      ddphi = xt::view(phi, xt::range(tdim + 1, phi.shape(0)), p, xt::all(), 0);

      /// FIXME: Is this correct? compute jacobian wraps math::dot.
      // Would it make things more clear to explicitly call math::dot
      dolfinx::fem::CoordinateElement::compute_jacobian(ddphi, coordinate_dofs,
                                                        _H);
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
double dolfinx_contact::dR_plus(double x) { return double(x > 0); }
//-------------------------------------------------------------------------------------
double dolfinx_contact::R_minus(double x) { return 0.5 * (x - std::abs(x)); }
//-------------------------------------------------------------------------------------
double dolfinx_contact::dR_minus(double x) { return double(x < 0); }
