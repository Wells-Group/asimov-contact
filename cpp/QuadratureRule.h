
// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier: MIT

#pragma once

#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/mesh/cell_types.h>

namespace dolfinx_contact
{
template <typename T, std::size_t d,
          typename S = std::experimental::layout_right>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, d>, S>;

/// @brief Quadrature schemes for each entity of given dimension dim of
/// a reference cell.
class QuadratureRule
{
public:
  /// Constructor
  /// @param[in] cell Reference cell type.
  /// @param[in] degree Degree of quadrature rule to compute for each
  /// entity.
  /// @param[in] dim Topological dimension of the reference cell
  /// entities.
  /// @param[in] type Quadrature rule type to compute for reference cell
  /// entity.
  QuadratureRule(dolfinx::mesh::CellType cell, int degree, int dim,
                 basix::quadrature::type type
                 = basix::quadrature::type::Default);

  /// @brief Quadrature points for each entity of the cell.
  ///
  /// The shape is (entity index, num_quad_points, dim). Storage is
  /// row-major.
  const std::vector<double>& points() const;

  /// Return a list of quadrature weights for each entity in the cell
  /// (using local entity index as in DOLFINx/Basix)
  const std::vector<double>& weights() const;

  /// Return dimension of entity in the quadrature rule
  int dim() const;

  /// @brief Cell type for the ith quadrature rule
  /// @param[in] i Local entity number
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

  /// Topological dimension of the quadrature rule
  /// @todo Docstring doesn't make sense
  std::size_t tdim() const;

  /// Return the quadrature points for the ith entity
  /// @param[in] i The local entity index
  mdspan_t<const double, 2> points(int i) const;

  /// Return the quadrature weights for the ith entity
  /// @param[in] i The local entity index
  std::span<const double> weights(int i) const;

  /// Return offset for quadrature rule of the ith entity
  const std::vector<std::size_t>& offset() const;

private:
  // What is this?
  dolfinx::mesh::CellType _cell_type;

  // What is this?
  int _degree;

  // What is this?
  basix::quadrature::type _type;

  // What is this?
  int _dim;

  // Quadrature points for each entity on the cell. Shape (entity,
  // num_points, tdim). Flattened row-major.
  std::vector<double> _points;

  // Quadrature weights for each entity on the cell
  std::vector<double> _weights;

  // The offset for each entity
  std::vector<std::size_t> _entity_offset;

  // Number of sub entities
  int _num_sub_entities;
};

} // namespace dolfinx_contact