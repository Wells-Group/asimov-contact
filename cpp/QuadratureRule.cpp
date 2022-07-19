
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
    std::array<std::vector<double>, 2> quadrature
        = basix::quadrature::make_quadrature(type, b_ct, degree);

    std::size_t num_points = quadrature[1].size();
    std::size_t pt_shape = quadrature[0].size() / quadrature[1].size();
    xt::xtensor<double, 2> q_points({num_points, pt_shape});
    std::copy(quadrature[0].cbegin(), quadrature[0].cend(), q_points.begin());
    std::vector<double>& q_weights = quadrature[1];

    _points = xt::empty<double>(
        {std::size_t(num_points * _num_sub_entities), (std::size_t)tdim});
    _entity_offset = std::vector<std::int32_t>(_num_sub_entities + 1, 0);
    std::vector<double> weights(num_points * _num_sub_entities);
    for (std::int32_t i = 0; i < _num_sub_entities; i++)
    {
      _entity_offset[i + 1] = (i + 1) * (std::int32_t)q_weights.size();
      for (std::size_t j = 0; j < num_points; ++j)
      {
        weights[i * num_points * _num_sub_entities + j] = q_weights[j];
        for (int k = 0; k < tdim; ++k)
          _points[i * num_points * _num_sub_entities + j * tdim + k]
              = q_points(j, k);
      }
    }
    _weights = weights;
  }
  else
  {
    // Create reference topology and geometry
    auto entity_topology = basix::cell::topology(b_ct)[dim];

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
      std::array<std::vector<double>, 2> quadrature
          = basix::quadrature::make_quadrature(et, degree);
      const std::vector<double>& q_weights = quadrature[1];
      std::size_t num_points = quadrature[1].size();
      std::size_t pt_shape = quadrature[0].size() / quadrature[1].size();
      std::array<std::size_t, 2> pts_shape = {num_points, pt_shape};
      xt::xtensor<double, 2> q_points(pts_shape);
      std::copy(quadrature[0].cbegin(), quadrature[0].cend(), q_points.begin());
      num_points_per_entity[i] = (std::int32_t)num_points;

      auto [c_tab_data, c_tab_shape] = entity_element.tabulate(
          0, basix::impl::cmdspan2_t(q_points.data(), pts_shape));
      xt::xtensor<double, 4> c_tab(c_tab_shape);
      std::copy(c_tab_data.cbegin(), c_tab_data.cend(), c_tab.begin());
      xt::xtensor<double, 2> phi_s
          = xt::view(c_tab, 0, xt::all(), xt::all(), 0);

      auto [sub_geom_data, sub_geom_shape]
          = basix::cell::sub_entity_geometry(b_ct, dim, i);
      xt::xtensor<double, 2> coords(sub_geom_shape);
      std::copy(sub_geom_data.cbegin(), sub_geom_data.cend(), coords.begin());

      // Push forward quadrature point from reference entity to reference
      // entity on cell
      const size_t num_quadrature_pts = q_weights.size();
      xt::xtensor<double, 2> entity_qp
          = xt::zeros<double>({num_quadrature_pts, coords.shape(1)});
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
        for (int k = 0; k < tdim; ++k)
          _points(i * num_points + j, k) = quadrature_points[i](j, k);
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