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
      const std::array<int, 2>& pair = Contact::contact_pair(i);
      std::size_t num_facets = Contact::local_facets(pair[0]);
      if (num_facets > 0)
      {
        auto [ny, cstride1] = Contact::pack_ny(i);
        auto [gap, cstride] = Contact::pack_gap(i);

        std::span<const std::int32_t> entities
            = Contact::active_entities(pair[0]);

        // Retrieve cells connected to integration facets
        std::vector<std::int32_t> cells(num_facets);
        for (std::size_t e = 0; e < num_facets; ++e)
          cells[e] = entities[2 * e];
        std::vector<double> h_p
            = dolfinx::mesh::h(*mesh, cells, mesh->topology()->dim());
        Contact::crop_invalid_points(i, gap, ny,
                                     *std::max_element(h_p.begin(), h_p.end()));
      }
    }

    // initialise internal variables
    _num_pairs = (int)connected_pairs.size();
    _coeffs.resize(_num_pairs);
    _coeffs_poisson.resize(_num_pairs);
    _q_deg = q_deg;
  };

  std::size_t offset_elasticity(
      std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V);
  std::size_t
  offset_poisson(std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V);

  /// Generate the input data for the custom integration kernel
  /// @param[in] problem_type specifies the type of the equation, e.g,
  /// elasticity
  /// @param[in] coefficients maps coefficients to their names used for the
  /// kernel
  /// @param[in] gamma Nitsche parameter
  /// @param[in] theta determines version of Nitsche's method
  void generate_kernel_data(
      Problem problem_type,
      std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
      const std::map<std::string,
                     std::shared_ptr<dolfinx::fem::Function<double>>>&
          coefficients,
      double gamma, double theta, double alpha=-1);

  /// Generate data for matrix assembly
  /// @param[in] lambda - lame parameter lambda as DG0 function
  /// @param[in] mu - lame parameter mu as DG0 function
  /// @param[in] gamma - Nitsche penalty parameter
  /// @param[in] theta - Nitsche parameter
  void generate_meshtie_data_matrix_only(
      dolfinx_contact::Problem problem_type,
      std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
      std::shared_ptr<dolfinx::fem::Function<double>> lambda,
      std::shared_ptr<dolfinx::fem::Function<double>> mu, double gamma,
      double theta, double alpha=-1);

  /// Update data for vector assembly based on state
  /// @param[in] coefficients maps coefficients to their names used for the
  /// kernel, u is used for displacements, T for temperature/scalar valued
  /// function
  /// @param[in] problem_type - the type of equation, e.g. elasticity
  void update_meshtie_data(
      const std::map<std::string,
                     std::shared_ptr<dolfinx::fem::Function<double>>>&
          coefficients,
      dolfinx_contact::Problem problem_type);

  /// Update funciton value data for vector assembly based on state
  /// @param[in] u - the function
  /// @param[in] coeffs - the coefficient vector to be updated
  /// @param[in] offset0 - position within coeffs where data on integration
  /// surface should be added
  /// @param[in] offset1 - position within coeffs where data on contacting
  /// surface should be added
  /// @param[in] coeff_size - total size of the coefficient array per facet
  void update_function_data(std::shared_ptr<dolfinx::fem::Function<double>> u,
                            std::vector<std::vector<double>>& coeffs,
                            std::size_t offset0, std::size_t offset1,
                            std::size_t coeff_size);

  /// Update gradient value data for vector assembly based on state
  /// @param[in] u - the function
  /// @param[in] coeffs - the coefficient vector to be updated
  /// @param[in] offset0 - position within coeffs where data on integration
  /// surface should be added
  /// @param[in] offset1 - position within coeffs where data on contacting
  /// surface should be added
  /// @param[in] coeff_size - total size of the coefficient array per facet
  void update_gradient_data(std::shared_ptr<dolfinx::fem::Function<double>> u,
                            std::vector<std::vector<double>>& coeffs,
                            std::size_t offset0, std::size_t offset1,
                            std::size_t coeff_size);

  /// Generate data for matrix assembly for Poisson
  /// @param[in] V - The FunctionSpace
  /// @param[in] kdt - scalar in front of laplace operator
  /// @param[in] gamma - Nitsche penalty parameter
  /// @param[in] theta - Nitsche parameter
  void generate_poisson_data_matrix_only(
      std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
      std::shared_ptr<dolfinx::fem::Function<double>> kdt, double gamma,
      double theta);

  using Contact::assemble_vector;
  /// Assemble right hand side
  /// @param[in] b - the vector to assemble into
  /// @param[in] V - the associated FunctionSpace
  /// @param[in] problem_type - the type of equation, e.g. elasticity
  void
  assemble_vector(std::span<PetscScalar> b,
                  std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
                  Problem problem_type);

  using Contact::assemble_matrix;
  /// Assemble matrix
  /// @param[in] mat_set function for setting matrix entries
  /// @param[in] V function space for Trial/Test functions
  /// @param[in] problem_type - the type of equation, e.g. elasticity
  void
  assemble_matrix(const mat_set_fn& mat_set,
                  std::shared_ptr<const dolfinx::fem::FunctionSpace<double>> V,
                  Problem problem_type);

  /// Return data generated with generate_meshtie_data
  /// @param[in] pair - the index of the pair of connected surfaces
  std::pair<std::vector<double>, std::size_t> coeffs(int pair);

private:
  // kernel function for rhs
  kernel_fn<PetscScalar> _kernel_rhs;
  // kernel function addding temperature dependent thermo-elasticity terms to
  // matrix
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