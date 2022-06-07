// Copyright (C) 2022 Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once
#include <dolfinx/mesh/cell_types.h>
#include <exception>
namespace dolfinx_contact
{
namespace error
{

constexpr std::array<dolfinx::mesh::CellType, 3> unsupported_cells
    = {dolfinx::mesh::CellType::prism, dolfinx::mesh::CellType::pyramid,
       dolfinx::mesh::CellType::interval};

class UnsupportedCellException : public std::exception
{
public:
  const char* what() const throw() { return message.c_str(); }

private:
  std::string message = "Cell-type not supported";
};

/// Check if cell type is supported
/// @param[in] cell_type The cell type
void check_cell_type(dolfinx::mesh::CellType cell_type);
} // namespace error
} // namespace dolfinx_contact
