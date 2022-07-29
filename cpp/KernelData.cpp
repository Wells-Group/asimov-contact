// Copyright (C) 2022 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#include "KernelData.h"

dolfinx_contact::KernelData::KernelData(
    std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
    std::shared_ptr<const dolfinx_contact::QuadratureRule> q_rule,
    const std::vector<std::size_t>& cstrides)
    : _qp_offsets(q_rule->offset()), _q_weights(q_rule->weights())
{
  // Extract mesh
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = V->mesh();
  assert(mesh);
  // Get mesh info
  const dolfinx::mesh::Geometry& geometry = mesh->geometry();
  const dolfinx::fem::CoordinateElement& cmap = geometry.cmap();
  _affine = cmap.is_affine();
  _num_coordinate_dofs = cmap.dim();

  _gdim = geometry.dim();
  const dolfinx::mesh::Topology& topology = mesh->topology();
  _tdim = topology.dim();

  dolfinx_contact::error::check_cell_type(topology.cell_type());
  // Create quadrature points on reference facet
  const std::vector<double>& q_points = q_rule->points();
  const std::size_t num_quadrature_pts = _q_weights.size();

  // Extract function space data (assuming same test and trial space)
  std::shared_ptr<const dolfinx::fem::FiniteElement> element = V->element();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap = V->dofmap();
  _ndofs_cell = dofmap->element_dof_layout().num_dofs();
  _bs = dofmap->bs();

  if (const bool needs_dof_transformations
      = element->needs_dof_transformations();
      needs_dof_transformations)
  {
    throw std::invalid_argument("Contact-kernels are not supporting finite "
                                "elements requiring dof transformations.");
  }

  if (element->value_size() / _bs != 1)
  {
    throw std::invalid_argument(
        "Contact kernel not supported for spaces with value size!=1");
  }

  if (_bs != _gdim)
  {
    throw std::invalid_argument(
        "The geometric dimension of the mesh is not equal to the block size "
        "of the function space.");
  }

  /// Pack test and trial functions
  const basix::FiniteElement& basix_element = element->basix_element();
  _basis_shape = basix_element.tabulate_shape(1, num_quadrature_pts);
  _basis_values = std::vector<double>(std::reduce(
      _basis_shape.begin(), _basis_shape.end(), 1, std::multiplies()));
  element->tabulate(_basis_values, q_points, {num_quadrature_pts, _tdim}, 1);

  // Tabulate Coordinate element (first derivative to compute Jacobian)
  _c_basis_shape = cmap.tabulate_shape(1, _q_weights.size());
  _c_basis_values = std::vector<double>(std::reduce(
      _c_basis_shape.begin(), _c_basis_shape.end(), 1, std::multiplies()));
  cmap.tabulate(1, q_points, {num_quadrature_pts, _tdim}, _c_basis_values);

  // Create offsets from cstrides
  _offsets.resize(cstrides.size() + 1);
  _offsets[0] = 0;
  std::partial_sum(cstrides.cbegin(), cstrides.cend(),
                   std::next(_offsets.begin()));

  // As reference facet and reference cell are affine, we do not need to
  // compute this per quadrature point
  basix::cell::type basix_cell
      = dolfinx::mesh::cell_type_to_basix_type(mesh->topology().cell_type());
  std::tie(_ref_jacobians, _jac_shape)
      = basix::cell::facet_jacobians(basix_cell);

  // Get facet normals on reference cell
  std::tie(_facet_normals, _normals_shape)
      = basix::cell::facet_outward_normals(basix_cell);

  // Get update Jacobian function (for each quadrature point)
  _update_jacobian = dolfinx_contact::get_update_jacobian_dependencies(cmap);

  // Get update FacetNormal function (for each quadrature point)
  _update_normal = dolfinx_contact::get_update_normal(cmap);
}