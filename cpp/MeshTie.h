// Copyright (C) 2023 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#include "Contact.h"
#include "utils.h"
#include "coefficients.h"
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/Form.h>


namespace dolfinx_contact
{

class MeshTie
{
public:
  /// Constructor
  /// @param[in] markers List of meshtags defining the contact surfaces
  /// @param[in] surfaces Adjacency list. Links of i contains meshtag values
  /// associated with ith meshtag in markers
  /// @param[in] contact_pairs list of pairs (i, j) marking the ith and jth
  /// surface in surfaces->array() as a contact pair
  /// @param[in] V The functions space
  /// @param[in] q_deg The quadrature degree.
  MeshTie(const std::vector<
              std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>>& markers,
          std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
              surfaces,
          const std::vector<std::array<int, 2>>& contact_pairs,
          std::shared_ptr<dolfinx::fem::FunctionSpace<double>> V,
          const int q_deg = 3);

  void pack_coefficients(std::shared_ptr<dolfinx::fem::Function<double>> u,
                         std::shared_ptr<dolfinx::fem::Function<double>> lambda,
                         std::shared_ptr<dolfinx::fem::Function<double>> mu,
                         std::shared_ptr<dolfinx::fem::Function<double>> h,
                         double gamma, double theta);

private:
  Contact _contact;
  kernel_fn<PetscScalar> _kernel_rhs;
  kernel_fn<PetscScalar> _kernel_jac;
  std::size_t _num_pairs;
  std::vector<std::vector<double>> _coeffs;
  std::int32_t _q_deg;
};
} // namespace dolfinx_contact