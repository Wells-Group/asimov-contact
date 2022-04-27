
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
  const int tdim = basix::cell::topological_dimension(b_ct);

  // If cell dimension no pushing forward
  if (tdim == dim)
  {
    auto [q_points, q_weights]
        = basix::quadrature::make_quadrature(type, b_ct, degree);

    std::size_t num_points = q_points.shape(0);
    _points = xt::empty<double>(
        {std::size_t(num_points * _num_sub_entities), (std::size_t)tdim});
    _entity_offset = std::vector<std::int32_t>(_num_sub_entities + 1, 0);
    std::vector<double> weights(num_points * _num_sub_entities);
    for (std::int32_t i = 0; i < _num_sub_entities; i++)
    {
      _entity_offset[i + 1] = (i + 1) * q_weights.size();
      for (std::size_t j = 0; j < num_points; ++j)
      {
        weights[i * num_points * _num_sub_entities + j] = q_weights[j];
        for (std::size_t k = 0; k < tdim; ++k)
          _points[i * num_points * _num_sub_entities + j * tdim + k]
              = q_points[j * tdim + k];
      }
    }
    _weights = weights;
  }
  else
  {
    // Create reference topology and geometry
    auto entity_topology = basix::cell::topology(b_ct)[dim];
    const xt::xtensor<double, 2> ref_geom = basix::cell::geometry(b_ct);

    // Create map for each facet type to the local index
    std::vector<xt::xarray<double>> quadrature_points;
    std::vector<std::vector<double>> quadrature_weights;
    quadrature_points.reserve(_num_sub_entities);
    quadrature_weights.reserve(_num_sub_entities);
    std::vector<std::int32_t> num_points_per_entity(_num_sub_entities);
    for (std::int32_t i = 0; i < _num_sub_entities; i++)
    {
      basix::cell::type et = basix::cell::sub_entity_type(b_ct, dim, i);
      basix::FiniteElement entity_element
          = basix::create_element(basix::element::family::P, et, 1,
                                  basix::element::lagrange_variant::gll_warped);
      // Create quadrature and tabulate on entity
      auto [q_points, q_weights]
          = basix::quadrature::make_quadrature(et, degree);
      num_points_per_entity[i] = (std::int32_t)q_weights.size();

      auto c_tab = entity_element.tabulate(0, q_points);
      xt::xtensor<double, 2> phi_s
          = xt::view(c_tab, 0, xt::all(), xt::all(), 0);
      auto cell_entity = entity_topology[i];
      auto coords = xt::view(ref_geom, xt::keep(cell_entity), xt::all());

      // Push forward quadrature point from reference entity to reference
      // entity on cell
      const size_t num_quadrature_pts = q_weights.size();
      xt::xtensor<double, 2> entity_qp = xt::zeros<double>(
          {num_quadrature_pts, static_cast<std::size_t>(ref_geom.shape(1))});
      dolfinx::math::dot(phi_s, coords, entity_qp);

      quadrature_points.push_back(entity_qp);
      quadrature_weights.push_back(q_weights);
    }
    _entity_offset = std::vector<std::int32_t>(_num_sub_entities + 1, 0);
    std::partial_sum(num_points_per_entity.begin(), num_points_per_entity.end(),
                     std::next(_entity_offset.begin()));
    _points = xt::empty<double>(
        {(std::size_t)_entity_offset.back(), (std::size_t)tdim});
    std::vector<double> weights(_entity_offset.back());
    for (std::int32_t i = 0; i < _num_sub_entities; i++)
    {
      const std::int32_t num_points = _entity_offset[i + 1] - _entity_offset[i];
      for (std::size_t j = 0; j < (std::size_t)num_points; ++j)
      {
        weights[i * num_points + j] = quadrature_weights[i][j];
        for (std::size_t k = 0; k < tdim; ++k)
          _points[i * num_points + j * tdim + k]
              = quadrature_points[i][j * tdim + k];
      }
    }
    _weights = weights;
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
std::int32_t QuadratureRule::num_points(int i) const
{
  assert(i < _num_sub_entities);
  return _entity_offset[i + 1] - _entity_offset[i];
}
//-----------------------------------------------------------------------------------------------
xt::xtensor<double, 2> QuadratureRule::points(int i) const
{
  assert(i < _num_sub_entities);
  return xt::view(_points, xt::xrange(_entity_offset[i], _entity_offset[i + 1]),
                  xt::all());
}