// Copyright (C) 2022 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#include "KernelData.h"

dolfinx_contact::KernelData::KernelData(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    const std::vector<xt::xarray<double>>& q_points,
    const std::vector<std::vector<double>>& q_weights,
    const std::vector<std::size_t>& cstrides)
    : _q_weights(q_weights)
{
  // Extract mesh data
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V->mesh();
  assert(mesh);
  const dolfinx::mesh::Geometry& geometry = mesh->geometry();
  const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
  _affine = cmap.is_affine();
  _num_coordinate_dofs = cmap.dim();

  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  if (const bool needs_dof_transformations
      = element->needs_dof_transformations();
      needs_dof_transformations)
  {
    throw std::invalid_argument("Contact-kernels are not supporting finite "
                                "elements requiring dof transformations.");
  }

  _gdim = geometry.dim();
  const dolfinx::mesh::Topology& topology = mesh->topology();
  _tdim = topology.dim();
  const dolfinx::mesh::CellType ct = topology.cell_type();

  if ((ct == dolfinx::mesh::CellType::prism)
      or (ct == dolfinx::mesh::CellType::pyramid))
  {
    throw std::invalid_argument("Unsupported cell type");
  }

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  _ndofs_cell = dofmap->element_dof_layout().num_dofs();
  _bs = dofmap->bs();
  if (_bs != _gdim)
  {
    throw std::invalid_argument(
        "The geometric dimension of the mesh is not equal to the block size "
        "of the function space.");
  }

  // NOTE: Assuming same number of quadrature points on each cell
  _num_q_points = q_points[0].shape(0);

  // Structures needed for basis function tabulation
  // phi and grad(phi) at quadrature points
  const std::size_t num_facets = dolfinx::mesh::cell_num_entities(
      mesh->topology().cell_type(), _tdim - 1);
  assert(num_facets == q_points.size());

  _phi.reserve(num_facets);
  _dphi.reserve(num_facets);
  _dphi_c.reserve(num_facets);

  // Temporary structures used in loop
  xt::xtensor<double, 4> cell_tab(
      {(std::size_t)_tdim + 1, _num_q_points, _ndofs_cell, _bs});
  xt::xtensor<double, 2> phi_i({_num_q_points, _ndofs_cell});
  xt::xtensor<double, 3> dphi_i(
      {(std::size_t)_tdim, _num_q_points, _ndofs_cell});
  std::array<std::size_t, 4> tabulate_shape
      = cmap.tabulate_shape(1, _num_q_points);
  xt::xtensor<double, 4> c_tab(tabulate_shape);
  assert(tabulate_shape[0] - 1 == (std::size_t)_tdim);
  xt::xtensor<double, 3> dphi_ci(
      {tabulate_shape[0] - 1, tabulate_shape[1], tabulate_shape[2]});

  // Tabulate basis functions and first order derivatives for each facet in
  // the reference cell. This tabulation is done both for the finite element
  // of the unknown and the coordinate element (which might differ)
  std::for_each(q_points.cbegin(), q_points.cend(),
                [&](const auto& q_facet)
                {
                  assert(q_facet.shape(0) == _num_q_points);
                  element->tabulate(cell_tab, q_facet, 1);

                  phi_i = xt::view(cell_tab, 0, xt::all(), xt::all(), 0);
                  _phi.push_back(phi_i);

                  dphi_i = xt::view(cell_tab, xt::range(1, tabulate_shape[0]),
                                    xt::all(), xt::all(), 0);
                  _dphi.push_back(dphi_i);

                  // Tabulate coordinate element of reference cell
                  cmap.tabulate(1, q_facet, c_tab);
                  dphi_ci = xt::view(c_tab, xt::range(1, tabulate_shape[0]),
                                     xt::all(), xt::all(), 0);
                  _dphi_c.push_back(dphi_ci);
                });

  // Create offsets from cstrides
  _offsets.reserve(cstrides.size() + 1);
  _offsets.push_back(0);
  std::partial_sum(cstrides.cbegin(), cstrides.cend(),
                   std::next(_offsets.begin()));

  // As reference facet and reference cell are affine, we do not need to
  // compute this per quadrature point
  basix::cell::type basix_cell
      = dolfinx::mesh::cell_type_to_basix_type(mesh->topology().cell_type());
  _ref_jacobians = basix::cell::facet_jacobians(basix_cell);

  // Get facet normals on reference cell
  _facet_normals = basix::cell::facet_outward_normals(basix_cell);

  // Get update Jacobian function (for each quadrature point)
  _update_jacobian = dolfinx_contact::get_update_jacobian_dependencies(cmap);

  // Get update FacetNormal function (for each quadrature point)
  _update_normal = dolfinx_contact::get_update_normal(cmap);
}