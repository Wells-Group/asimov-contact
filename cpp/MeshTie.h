// Copyright (C) 2023 Sarah Roggendorf
//
// This file is part of DOLFINx_Contact
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "Contact.h"
#include "coefficients.h"
#include "utils.h"
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/mesh/utils.h>
#include <map>

namespace dolfinx_contact
{
/// MeshTye class
class MeshTie : public Contact
{
public:
  /// @brief Constructor
  ///
  /// @param[in] markers List of meshtags defining the connected
  /// surfaces
  /// @param[in] surfaces Adjacency list. Links of i contains meshtag
  /// values associated with ith meshtag in markers
  /// @param[in] connected_pairs list of pairs (i, j) marking the ith
  /// and jth surface in surfaces->array() as a pair of connected
  /// surfaces
  /// @param[in] mesh
  /// @param[in] q_deg The quadrature degree.
  MeshTie(const std::vector<std::shared_ptr<
              const dolfinx::mesh::MeshTags<std::int32_t>>>& markers,
          const dolfinx::graph::AdjacencyList<std::int32_t>& surfaces,
          const std::vector<std::array<int, 2>>& connected_pairs,
          std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh,
          int q_deg = 3);

  /// @brief TODO
  /// @param V
  /// @return TODP
  std::size_t offset_elasticity(const dolfinx::fem::FunctionSpace<double>& V);

  /// @brief TODO
  /// @param V
  /// @return TODO
  std::size_t offset_poisson(const dolfinx::fem::FunctionSpace<double>& V);

  /// @brief Generate the input data for the custom integration kernel.
  /// @param[in] problem_type specifies the type of the equation, e.g,
  /// elasticity
  /// @param[in] V TODO
  /// @param[in] coefficients maps coefficients to their names used for
  /// the kernel
  /// @param[in] gamma Nitsche parameter
  /// @param[in] theta determines version of Nitsche's method
  void generate_kernel_data(
      Problem problem_type, const dolfinx::fem::FunctionSpace<double>& V,
      const std::map<std::string,
                     std::shared_ptr<const dolfinx::fem::Function<double>>>&
          coefficients,
      double gamma, double theta);

  /// Generate data for matrix assembly
  /// @param[in] problem_type lame parameter lambda as DG0 function
  /// @param[in] V lame parameter mu as DG0 function
  /// @param[in] coeffs
  /// @param[in] gamma Nitsche penalty parameter
  /// @param[in] theta Nitsche parameter
  void generate_meshtie_data_matrix_only(
      Problem problem_type, const dolfinx::fem::FunctionSpace<double>& V,
      std::vector<std::shared_ptr<const dolfinx::fem::Function<double>>> coeffs,
      double gamma, double theta);

  /// Update data for vector assembly based on state
  /// @param[in] coefficients maps coefficients to their names used for
  /// the kernel, u is used for displacements, T for temperature/scalar
  /// valued function
  /// @param[in] problem_type the type of equation, e.g. elasticity
  void update_kernel_data(
      const std::map<std::string,
                     std::shared_ptr<const dolfinx::fem::Function<double>>>&
          coefficients,
      Problem problem_type);

  /// Update funciton value data for vector assembly based on state
  /// @param[in] u the function
  /// @param[in] coeffs the coefficient vector to be updated
  /// @param[in] offset0 position within coeffs where data on
  /// integration surface should be added
  /// @param[in] offset1 position within coeffs where data on contacting
  /// surface should be added
  /// @param[in] coeff_size total size of the coefficient array per facet
  void update_function_data(const dolfinx::fem::Function<double>& u,
                            std::vector<std::vector<double>>& coeffs,
                            std::size_t offset0, std::size_t offset1,
                            std::size_t coeff_size);

  /// Update gradient value data for vector assembly based on state
  /// @param[in] u the function
  /// @param[in] coeffs the coefficient vector to be updated
  /// @param[in] offset0 position within coeffs where data on integration
  /// surface should be added
  /// @param[in] offset1 position within coeffs where data on contacting
  /// surface should be added
  /// @param[in] coeff_size total size of the coefficient array per facet
  void update_gradient_data(const dolfinx::fem::Function<double>& u,
                            std::vector<std::vector<double>>& coeffs,
                            std::size_t offset0, std::size_t offset1,
                            std::size_t coeff_size);

  /// Generate data for matrix assembly for Poisson
  /// @param[in] V The FunctionSpace
  /// @param[in] kdt scalar in front of laplace operator
  /// @param[in] gamma Nitsche penalty parameter
  /// @param[in] theta Nitsche parameter
  void generate_poisson_data_matrix_only(
      const dolfinx::fem::FunctionSpace<double>& V,
      const dolfinx::fem::Function<double>& kdt, double gamma, double theta);

  /// Assemble right hand side
  /// @param[in] b the vector to assemble into
  /// @param[in] V the associated FunctionSpace
  /// @param[in] problem_type - the type of equation, e.g. elasticity
  void assemble_vector(std::span<PetscScalar> b,
                       const dolfinx::fem::FunctionSpace<double>& V,
                       Problem problem_type);

  /// Assemble matrix
  /// @param[in] mat_set function for setting matrix entries
  /// @param[in] V function space for Trial/Test functions
  /// @param[in] problem_type the type of equation, e.g. elasticity
  void assemble_matrix(const mat_set_fn& mat_set,
                       const dolfinx::fem::FunctionSpace<double>& V,
                       Problem problem_type);

  /// Return data generated with generate_meshtie_data.
  /// @param[in] pair - the index of the pair of connected surfaces.
  std::pair<std::vector<double>, std::size_t> coeffs(int pair);

private:
  // kernel function for rhs
  kernel_fn<PetscScalar> _kernel_rhs;

  // kernel function adding temperature dependent thermo-elasticity
  // terms to matrix
  kernel_fn<PetscScalar> _kernel_thermo_el;

  // kernel function for matrix

  kernel_fn<PetscScalar> _kernel_jac;

  // kernel function for rhs
  kernel_fn<PetscScalar> _kernel_rhs_poisson;

  // kernel function for matrix
  kernel_fn<PetscScalar> _kernel_jac_poisson;

  // number of pairs of connected surfaces
  int _num_pairs;

  // storage for generated data
  std::vector<std::vector<double>> _coeffs;
  std::vector<std::vector<double>> _coeffs_poisson;

  // constant input parameters for kernels
  std::vector<double> _consts;
  std::vector<double> _consts_poisson;

  // quadrature degree
  std::int32_t _q_deg;
  std::size_t _cstride = 0;
  std::size_t _cstride_poisson = 0;
};
} // namespace dolfinx_contact