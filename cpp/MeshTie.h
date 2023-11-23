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
  /// @param[in] markers List of meshtags defining the connected surfaces
  /// @param[in] surfaces Adjacency list. Links of i contains meshtag values
  /// associated with ith meshtag in markers
  /// @param[in] connected_pairs list of pairs (i, j) marking the ith and jth
  /// surface in surfaces->array() as a pair of connected surfaces
  /// @param[in] V The functions space
  /// @param[in] q_deg The quadrature degree.
  MeshTie(
      const std::vector<std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>>&
          markers,
      std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
          surfaces,
      const std::vector<std::array<int, 2>>& connected_pairs,
      std::shared_ptr<dolfinx::mesh::Mesh<double>> mesh, const int q_deg = 3)
      : Contact::Contact(markers, surfaces, connected_pairs, mesh, q_deg,
                         ContactMode::ClosestPoint)
  {
    // Finde closest pointes
    for (int i = 0; i < (int)connected_pairs.size(); ++i)
    {
      Contact::create_distance_map(i);
      auto [ny, cstride1] = Contact::pack_ny(i);
      auto [gap, cstride] = Contact::pack_gap(i);
      Contact::update_distance_map(i, gap, ny);
    }

    // initialise internal variables
    _num_pairs = (int)connected_pairs.size();
    _coeffs.resize(_num_pairs);
    _coeffs_heat_transfer.resize(_num_pairs);
    _q_deg = q_deg;
  };

  std::size_t offset_elasticity(
      std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V);
  std::size_t
  offset_poisson(std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V);
void generate_kernel_data(Problem problem_type, const std::map<std::string, std::shared_ptr<const dolfinx::fem::Function<double>>>&
        coefficients, double gamma, double theta);
  /// Generate data for matrix/vector assembly
  /// @param[in] u - the displacement function
  /// @param[in] lambda - lame parameter lambda as DG0 function
  /// @param[in] mu - lame parameter mu as DG0 function
  /// @param[in] gamma - Nitsche penalty parameter
  /// @param[in] theta - Nitsche parameter
  void
  generate_meshtie_data(std::shared_ptr<dolfinx::fem::Function<double>> u,
                        std::shared_ptr<dolfinx::fem::Function<double>> lambda,
                        std::shared_ptr<dolfinx::fem::Function<double>> mu,
                        double gamma, double theta);

  /// Generate data for matrix assembly
  /// @param[in] lambda - lame parameter lambda as DG0 function
  /// @param[in] mu - lame parameter mu as DG0 function
  /// @param[in] gamma - Nitsche penalty parameter
  /// @param[in] theta - Nitsche parameter
  void generate_meshtie_data_matrix_only(
      std::shared_ptr<const dolfinx::fem::FunctionSpace<double>>,
      std::shared_ptr<dolfinx::fem::Function<double>> lambda,
      std::shared_ptr<dolfinx::fem::Function<double>> mu, double gamma,
      double theta);

  /// Update data for vector assembly based on current displacements
  /// @param[in] u - the displacement function
  void update_meshtie_data(std::shared_ptr<dolfinx::fem::Function<double>> u,
                           std::vector<std::vector<double>>& coeffs,
                           std::size_t offset0, std::size_t coeff_size);

  /// Generate data for matrix assembly
  /// @param[in] gamma - Nitsche penalty parameter
  /// @param[in] theta - Nitsche parameter
  void generate_heattransfer_data_matrix_only(
      std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V, double kdt,
      double gamma, double theta);
  void
  generate_heat_transfer_data(std::shared_ptr<dolfinx::fem::Function<double>> T,
                              double kdt, double gamma, double theta);
  using Contact::assemble_vector;
  /// Assemble right hand side
  /// @param[in] b - the vector to assemble into
  void
  assemble_vector(std::span<PetscScalar> b,
                  std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
                  Problem problemtyp);

  using Contact::assemble_matrix;
  /// Assemble matrix
  /// @param[in] mat_set function for setting matrix entries
  /// @param[in] V function space for Trial/Test functions
  /// @param[in] Problem problem type 
  void
  assemble_matrix(const mat_set_fn& mat_set,
                  std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
                  Problem problemtype);

  /// Return data generated with generate_meshtie_data
  /// @param[in] pair - the index of the pair of connected surfaces
  std::pair<std::vector<double>, std::size_t> coeffs(int pair);

private:
  // kernel function for rhs
  kernel_fn<PetscScalar> _kernel_rhs;
  // kernel function for matrix
  kernel_fn<PetscScalar> _kernel_jac;
  // kernel function for rhs
  kernel_fn<PetscScalar> _kernel_rhs_heat_transfer;
  // kernel function for matrix
  kernel_fn<PetscScalar> _kernel_jac_heat_transfer;
  // number of pairs of connected surfaces
  int _num_pairs;
  // storage for generated data
  std::vector<std::vector<double>> _coeffs;
  std::vector<std::vector<double>> _coeffs_heat_transfer;
  // constant input parameters for kernels
  std::vector<double> _consts;
  std::vector<double> _consts_heat_transfer;
  // quadrature degree
  std::int32_t _q_deg;
  std::size_t _cstride = 0;
  std::size_t _cstride_heat_transfer = 0;
};
} // namespace dolfinx_contact