
// Copyright (C) 2021-2022 Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "QuadratureRule.h"
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/common/math.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>

using namespace dolfinx_contact;

dolfinx_contact::QuadratureRule::QuadratureRule(dolfinx::mesh::CellType ct,
                                                int degree, int dim,
                                                basix::quadrature::type type)
    : _cell_type(ct), _degree(degree), _type(type), _dim(dim)
{

  basix::cell::type b_ct = dolfinx::mesh::cell_type_to_basix_type(ct);
  _num_sub_entities = basix::cell::num_sub_entities(b_ct, dim);
  _tdim = basix::cell::topological_dimension(b_ct);
  assert(dim <= 3);
  // If cell dimension no pushing forward
  if (_tdim == std::size_t(dim))
  {
    std::array<std::vector<double>, 2> quadrature
        = basix::quadrature::make_quadrature(type, b_ct, degree);
    std::vector<double>& q_weights = quadrature.back();
    std::size_t num_points = q_weights.size();
    std::size_t pt_shape = quadrature.front().size() / num_points;
    cmdspan2_t qp(quadrature.front().data(), num_points, pt_shape);

    _points = std::vector<double>(num_points * _num_sub_entities * _tdim);
    _entity_offset = std::vector<std::size_t>(_num_sub_entities + 1, 0);
    _weights = std::vector<double>(num_points * _num_sub_entities);
    for (std::int32_t i = 0; i < _num_sub_entities; i++)
    {
      _entity_offset[i + 1] = (i + 1) * q_weights.size();
      for (std::size_t j = 0; j < num_points; ++j)
      {
        _weights[i * num_points * _num_sub_entities + j] = q_weights[j];
        for (std::size_t k = 0; k < _tdim; ++k)
          _points[i * num_points * _num_sub_entities + j * _tdim + k]
              = qp(j, k);
      }
    }
  }
  else
  {
    // Create reference topology and geometry
    auto entity_topology = basix::cell::topology(b_ct)[dim];

    // Create map for each facet type to the local index
    std::vector<std::size_t> num_points_per_entity(_num_sub_entities);
    for (std::int32_t i = 0; i < _num_sub_entities; i++)
    {
      // Create reference element to map facet quadrature to
      basix::cell::type et = basix::cell::sub_entity_type(b_ct, dim, i);
      basix::FiniteElement entity_element
          = basix::create_element(basix::element::family::P, et, 1,
                                  basix::element::lagrange_variant::gll_warped,
                                  basix::element::dpc_variant::unset, false);
      // Create quadrature and tabulate on entity
      std::array<std::vector<double>, 2> quadrature
          = basix::quadrature::make_quadrature(type, et, degree);
      const std::vector<double>& q_weights = quadrature.back();
      const std::vector<double>& q_points = quadrature.front();
      const std::size_t num_points = q_weights.size();
      const std::size_t tdim = q_points.size() / q_weights.size();
      num_points_per_entity[i] = num_points;

      const std::array<std::size_t, 4> e_tab_shape
          = entity_element.tabulate_shape(0, num_points);
      std::vector<double> reference_entity_b(std::reduce(
          e_tab_shape.cbegin(), e_tab_shape.cend(), 1, std::multiplies{}));

      entity_element.tabulate(0, q_points, {num_points, tdim},
                              reference_entity_b);

      cmdspan4_t basis_full(reference_entity_b.data(), e_tab_shape);
      auto phi = stdex::submdspan(basis_full, 0, stdex::full_extent,
                                  stdex::full_extent, 0);

      auto [sub_geomb, sub_geom_shape]
          = basix::cell::sub_entity_geometry(b_ct, dim, i);
      cmdspan2_t coords(sub_geomb.data(), sub_geom_shape);

      // Push forward quadrature point from reference entity to reference
      // entity on cell
      const std::size_t offset = _points.size();
      _points.resize(_points.size() + num_points * coords.extent(1));
      mdspan2_t entity_qp(_points.data() + offset, num_points,
                          coords.extent(1));
      assert(coords.extent(1) == _tdim);
      dolfinx::math::dot(phi, coords, entity_qp);
      const std::size_t weights_offset = _weights.size();
      _weights.resize(_weights.size() + q_weights.size());
      std::copy(q_weights.cbegin(), q_weights.cend(),
                std::next(_weights.begin(), weights_offset));
    }
    _entity_offset = std::vector<std::size_t>(_num_sub_entities + 1, 0);
    std::partial_sum(num_points_per_entity.begin(), num_points_per_entity.end(),
                     std::next(_entity_offset.begin()));
  }
}

//-----------------------------------------------------------------------------------------------
dolfinx::mesh::CellType QuadratureRule::cell_type(int i) const
{
  basix::cell::type b_ct = dolfinx::mesh::cell_type_to_basix_type(_cell_type);
  assert(i < _num_sub_entities);

  basix::cell::type et = basix::cell::sub_entity_type(b_ct, _dim, i);
  return dolfinx::mesh::cell_type_from_basix_type(et);
}
//-----------------------------------------------------------------------------------------------
int QuadratureRule::degree() const { return _degree; }
//-----------------------------------------------------------------------------------------------
basix::quadrature::type QuadratureRule::type() const { return _type; }
//-----------------------------------------------------------------------------------------------
std::size_t QuadratureRule::num_points(int i) const
{
  assert(i < _num_sub_entities);
  return _entity_offset[i + 1] - _entity_offset[i];
}
//-----------------------------------------------------------------------------------------------
cmdspan2_t QuadratureRule::points(int i) const
{
  assert(i < _num_sub_entities);
  cmdspan2_t all_points(_points.data(), _weights.size(), _tdim);
  return stdex::submdspan(all_points,
                          std::pair(_entity_offset[i], _entity_offset[i + 1]),
                          stdex::full_extent);
}
//-----------------------------------------------------------------------------------------------
std::span<const double> QuadratureRule::weights(int i) const
{
  assert(i < _num_sub_entities);
  return std::span(_weights.data() + _entity_offset[i],
                   _entity_offset[i + 1] - _entity_offset[i]);
}
