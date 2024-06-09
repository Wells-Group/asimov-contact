// Copyright (C) 2022 Jørgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "error_handling.h"
#include <algorithm>

void dolfinx_contact::error::check_cell_type(dolfinx::mesh::CellType cell_type)
{

  if (std::find(dolfinx_contact::error::unsupported_cells.cbegin(),
                dolfinx_contact::error::unsupported_cells.cend(), cell_type)
      != unsupported_cells.cend())
  {
    throw dolfinx_contact::error::UnsupportedCellException();
  }
}
