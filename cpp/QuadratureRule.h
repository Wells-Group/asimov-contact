
// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/utils.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>

namespace dolfinx_contact
{

class QuadratureRule
{
  // Contains quadrature points and weights on a cell on a set of entities

public:
  /// Constructor
  /// @param[in] ct The cell type
  /// @param[in] degree Degree of quadrature rule
  /// @param[in] Dimension of entity
  /// @param[in] type Type of quadrature rule
  QuadratureRule(dolfinx::mesh::CellType ct, int degree, int dim,
                 basix::quadrature::type type
                 = basix::quadrature::type::Default);

  /// Return a list of quadrature points for each entity in the cell
  const xt::xtensor<double, 2>& points() const { return _points; }

  /// Return a list of quadrature weights for each entity in the cell (using
  /// local entity index as in DOLFINx/Basix)
  const std::vector<double>& weights() const { return _weights; }

  /// Return dimension of entity in the quadrature rule
  int dim() const { return _dim; }

  /// Return the cell type for the ith quadrature rule
  /// @param[in] Local entity number
  dolfinx::mesh::CellType cell_type(int i) const;

  /// @brief Return degree of quadrature rule
  ///
  /// For reconstruction of a quadrature rule on another entity
  int degree() const;

  /// @brief Return type of the quadrature rule
  ///
  /// For reconstruction of a quadrature rule on another entity
  basix::quadrature::type type() const;

  /// Return the number of quadrature points per entity
  std::int32_t num_points(int i) const;

  /// Return the quadrature points for the ith entity
  /// @param[in] i The local entity index
  xt::xtensor<double, 2> points(int i) const;

private:
  dolfinx::mesh::CellType _cell_type;
  int _degree;
  basix::quadrature::type _type;
  int _dim;
  xt::xtensor<double, 2>
      _points; // Quadrature points for each entity on the cell
  std::vector<double>
      _weights; // Quadrature weights for each entity on the cell
  std::vector<std::int32_t> _entity_offset; // The offset for each entity

  int _num_sub_entities; // Number of sub entities
};

} // namespace dolfinx_contact