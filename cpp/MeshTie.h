// Copyright (C) 2023 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#include "Contact.h"
#include "coefficients.h"
#include "utils.h"
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/utils.h>

namespace dolfinx_contact
{

class MeshTie : public Contact
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
          const int q_deg = 3)
      : Contact::Contact(markers, surfaces, contact_pairs, V, q_deg,
                ContactMode::ClosestPoint)
  {
    for (std::size_t i = 0; i < contact_pairs.size(); ++i)
      Contact::create_distance_map(i);
    _kernel_rhs = Contact::generate_kernel(Kernel::MeshTieRhs);
    _kernel_jac = Contact::generate_kernel(Kernel::MeshTieJac);
    _num_pairs = contact_pairs.size();
    _coeffs.resize(_num_pairs);
    _q_deg = q_deg;
  };

  void generate_meshtie_data(std::shared_ptr<dolfinx::fem::Function<double>> u,
                         std::shared_ptr<dolfinx::fem::Function<double>> lambda,
                         std::shared_ptr<dolfinx::fem::Function<double>> mu,
                         double gamma, double theta);

  using Contact::assemble_vector;
  void assemble_vector(std::span<PetscScalar> b);
  using Contact::assemble_matrix;
  void assemble_matrix(
      const mat_set_fn& mat_set);

  std::pair<std::vector<double>, std::size_t> coeffs(int pair);

private:
  kernel_fn<PetscScalar> _kernel_rhs;
  kernel_fn<PetscScalar> _kernel_jac;
  std::size_t _num_pairs;
  std::vector<std::vector<double>> _coeffs;
  std::vector<double> _consts;
  std::int32_t _q_deg;
};
} // namespace dolfinx_contact