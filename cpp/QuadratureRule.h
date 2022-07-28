
// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/mesh/cell_types.h>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace dolfinx_contact
{
namespace stdex = std::experimental;
using mdspan2_t = stdex::mdspan<double, stdex::dextents<std::size_t, 2>>;
using cmdspan2_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 2>>;
using mdspan3_t = stdex::mdspan<double, stdex::dextents<std::size_t, 3>>;
using cmdspan3_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 3>>;
using mdspan4_t = stdex::mdspan<double, stdex::dextents<std::size_t, 4>>;
using cmdspan4_t = stdex::mdspan<const double, stdex::dextents<std::size_t, 4>>;

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
  const std::vector<double>& points() const { return _points; }

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
  std::size_t num_points(int i) const;

  /// Return the topological dimension of the quadrature rule
  std::size_t tdim() const { return _tdim; };

  /// Return the quadrature points for the ith entity
  /// @param[in] i The local entity index
  cmdspan2_t points(int i) const;

  /// Return the quadrature weights for the ith entity
  /// @param[in] i The local entity index
  std::span<const double> weights(int i) const;

  /// Return offset for quadrature rule of the ith entity
  const std::vector<std::size_t>& offset() const { return _entity_offset; }

private:
  dolfinx::mesh::CellType _cell_type;
  std::size_t _tdim;
  int _degree;
  basix::quadrature::type _type;
  int _dim;
  std::vector<double>
      _points; // Quadrature points for each entity on the cell. Shape (entity,
               // num_points, tdim). Flattened row-major.
  std::vector<double>
      _weights; // Quadrature weights for each entity on the cell
  std::vector<std::size_t> _entity_offset; // The offset for each entity

  int _num_sub_entities; // Number of sub entities
};

} // namespace dolfinx_contact