// Copyright (C) 2021 Jørgen S. Dokken and Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "QuadratureRule.h"
#include "RayTracing.h"
#include "error_handling.h"
#include "geometric_quantities.h"
#include <basix/cell.h>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/sort.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/gjk.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/mesh/Mesh.h>
namespace dolfinx_contact
{

enum class ContactMode
{
  ClosestPoint,
  RayTracing
};

enum class Kernel
{
  Rhs,
  Jac,
  MeshTieRhs,
  MeshTieJac
};
// NOTE: this function should change signature to T * ,..... , num_links,
// num_dofs_per_link
template <typename T>
using kernel_fn
    = std::function<void(std::vector<std::vector<T>>&, std::span<const T>,
                         const T*, const double*, const std::size_t,
                         const std::size_t, std::span<const std::int32_t>)>;

/// This function computes the pull back for a set of points x on a cell
/// described by coordinate_dofs as well as the corresponding Jacobian, their
/// inverses and their determinants
/// @param[in, out] J: Jacobians of transformation from reference element to
/// physical element. Shape = (num_points, tdim, gdim). Computed at each point
/// in x
/// @param[in, out] K: inverse of J at each point.
/// @param[in, out] detJ: determinant of J at each  point
/// @param[in ,out] X: pull pack of x (points on reference element)
/// @param[in] x: points on physical element
/// @param[in] coordinate_dofs: geometry coordinates of cell
/// @param[in] cmap: the coordinate element
//-----------------------------------------------------------------------------
void pull_back(mdspan3_t J, mdspan3_t K, std::span<double> detJ,
               std::span<double> X, cmdspan2_t x, cmdspan2_t coordinate_dofs,
               const dolfinx::fem::CoordinateElement& cmap);

/// @param[in] cells: the cells to be sorted
/// @param[in, out] perm: the permutation for the sorted cells
/// @param[out] pair(unique_cells, offsets): unique_cells is a vector of
/// sorted cells with all duplicates deleted, offsets contains the start and
/// end for each unique value in the sorted vector with all duplicates
// Example: cells = [5, 7, 6, 5]
//          unique_cells = [5, 6, 7]
//          offsets = [0, 2, 3, 4]
//          perm = [0, 3, 2, 1]
// Then given a cell and its index ("i") in unique_cells, one can recover the
// indices for its occurance in cells with perm[k], where
// offsets[i]<=k<offsets[i+1]. In the example if i = 0, then perm[k] = 0 or
// perm[k] = 3.
std::pair<std::vector<std::int32_t>, std::vector<std::int32_t>>
sort_cells(const std::span<const std::int32_t>& cells,
           const std::span<std::int32_t>& perm);

/// @param[in] u: dolfinx function on function space base on basix element
/// @param[in] mesh: mesh to be updated
/// Adds perturbation u to mesh
void update_geometry(const dolfinx::fem::Function<PetscScalar>& u,
                     std::shared_ptr<dolfinx::mesh::Mesh> mesh);

/// Compute the positive restriction of a double, i.e. f(x)= x if x>0 else 0
double R_plus(double x);

/// Compute the negative restriction of a double, i.e. f(x)= x if x<0 else 0
double R_minus(double x);

/// Compute the derivative of the positive restriction (i.e.) the step function.
/// @note Evaluates to 0 at x=0
double dR_plus(double x);

/// Compute the derivative of the negative restriction (i.e.) the step function.
/// @note Evaluates to 0 at x=0
double dR_minus(double x);

/// Get shape of in,out variable for filling basis functions in for
/// evaluate_basis_functions
std::array<std::size_t, 4>
evaluate_basis_shape(const dolfinx::fem::FunctionSpace& V,
                     const std::size_t num_points,
                     const std::size_t num_derivatives);

/// Get basis values (not unrolled for block size) for a set of points and
/// corresponding cells.
/// @param[in] V The function space
/// @param[in] x The coordinates of the points. It has shape
/// (num_points, gdim). Flattened row major
/// @param[in] cells An array of cell indices. cells[i] is the index
/// of the cell that contains the point x(i). Negative cell indices
/// can be passed, and the corresponding point will be ignored.
/// @param[in,out] basis_values The values at the points. Values are not
/// computed for points with a negative cell index. This argument must be passed
/// with the correct size (num_points, number_of_dofs, value_size). The basis
/// values are flattened row-major.
/// @param[in] number_of_derivatives FIXME: Add docs
void evaluate_basis_functions(const dolfinx::fem::FunctionSpace& V,
                              std::span<const double> x,
                              std::span<const std::int32_t> cells,
                              std::span<double> basis_values,
                              std::size_t num_derivatives);

/// Compute the following jacobians on a given facet:
/// J: physical cell -> reference cell (and its inverse)
/// J_tot: physical facet -> reference facet
/// @param[in,out] J - Jacobian between reference cell and physical cell
/// @param[in,out] K - inverse of J
/// @param[in,out] J_tot - J_f*J
/// @param[in,out] detJ_scratch Working memory for Jacobian computation. Has to
/// be at least 2*gdim*tdim.
/// @param[in] J_f - the Jacobian between reference facet and reference cell
/// @param[in] dphi - derivatives of coordinate basis tabulated for quardrature
/// points
/// @param[in] coords - the coordinates of the facet
/// @return absolute value of determinant of J_tot
double compute_facet_jacobian(mdspan2_t J, mdspan2_t K, mdspan2_t J_tot,
                              std::span<double> detJ_scratch, cmdspan2_t J_f,
                              s_cmdspan2_t dphi, cmdspan2_t coords);

/// @brief Convenience function to update Jacobians
///
/// For affine geometries, the input determinant is returned.
/// For non-affine geometries, the Jacobian, it's inverse and the total Jacobian
/// (J*J_f) is computed.
/// @param[in] cmap The coordinate element
std::function<double(double, mdspan2_t, mdspan2_t, mdspan2_t, std::span<double>,
                     cmdspan2_t, s_cmdspan2_t, cmdspan2_t)>
get_update_jacobian_dependencies(const dolfinx::fem::CoordinateElement& cmap);

/// @brief Convenience function to update facet normals
///
/// For affine geometries, a do nothing function is returned.
/// For non-affine geometries, a function updating the physical facet normal is
/// returned.
/// @param[in] cmap The coordinate element
std::function<void(std::span<double>, cmdspan2_t, cmdspan2_t,
                   const std::size_t)>
get_update_normal(const dolfinx::fem::CoordinateElement& cmap);

/// @brief Convert local entity indices to integration entities
///
/// Compute the active entities in DOLFINx format for a given integral type over
/// a set of entities If the integral type is cell, return the input, if it is
/// exterior facets, return a list of pairs (cell, local_facet_index), and if it
/// is interior facets, return a list of tuples (cell_0, local_facet_index_0,
/// cell_1, local_facet_index_1) for each entity.
/// @param[in] mesh The mesh
/// @param[in] entities List of mesh entities
/// @param[in] integral The type of integral
std::pair<std::vector<std::int32_t>, std::size_t>
compute_active_entities(std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
                        std::span<const std::int32_t> entities,
                        dolfinx::fem::IntegralType integral);

/// @brief Compute the geometry dof indices for a set of entities
///
/// For a set of entities, compute the geometry closure dofs of the entity.
///
/// @param[in] mesh The mesh
/// @param[in] dim The dimension of the entities
/// @param[in] entities List of mesh entities
/// @returns An adjacency list where the i-th link corresponds to the
/// closure dofs of the i-th input entity
dolfinx::graph::AdjacencyList<std::int32_t>
entities_to_geometry_dofs(const mesh::Mesh& mesh, int dim,
                          const std::span<const std::int32_t>& entity_list);

/// @brief find candidate facets within a given radius of quadrature facet
///
/// Given one quadrature facet and a list of candidate facets return
/// the indices of only those candidate facet within the given radius
/// sorted according to the distance measured at the midpoints
///
/// @param[in] mesh The mesh
/// @param[in] quadrature facet Single quadrature facet
/// @param[in] candidate_facets Candidate facets
/// @param[in] radius The search radius
/// @return sorted indices of candidate facets within radius of quadrature facet
std::vector<std::size_t>
find_candidate_facets(const dolfinx::mesh::Mesh& quadrature_mesh,
                      const dolfinx::mesh::Mesh& candidate_mesh,
                      const std::int32_t quadrature_facet,
                      std::span<const std::int32_t> candidate_facets,
                      const double radius);
/// @brief find candidate facets within a given radius of quadratuere facets
///
/// Given a list of quadrature facets and a list of candidate facets return
/// only those candidate facet within the given radius
///
/// @param[in] mesh The mesh
/// @param[in] quadrature_facets Quadrature facets
/// @param[in] candidate_facets Candidate facets
/// @param[in] radius The search radius
/// @return candidate facets within radius of quadrature facets
std::vector<std::int32_t> find_candidate_surface_segment(
    std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
    const std::vector<std::int32_t>& quadrature_facets,
    const std::vector<std::int32_t>& candidate_facets, const double radius);

/// @brief compute physical points on set of facets
///
/// Given a list of facets and the basis functions evaluated at set of points on
/// reference facets compute physical points
///
/// @param[in] mesh The mesh
/// @param[in] facets The list of facets as (cell, local_facet). The data is
/// flattened row-major
/// @param[in] offsets for accessing the basis_values for local_facet
/// @param[in] phi Basis functions evaluated at desired set of points on
/// reference facet
/// @param[in, out] qp_phys Vector to store the quadrature points. Shape
/// (num_facets, num_q_points_per_facet, gdim). Flattened row-major
void compute_physical_points(const dolfinx::mesh::Mesh& mesh,
                             std::span<const std::int32_t> facets,
                             std::span<const std::size_t> offsets,
                             cmdspan4_t phi, std::span<double> qp_phys);

/// Compute the closest entity at every quadrature point on a subset of facets
/// on one mesh, to a subset of facets on the other mesh.
/// @param[in] quadrature_mesh The mesh to compute quadrature points on
/// @param[in] quadrature_facets The facets to compute quadrature points on,
/// defined as (cell, local_facet_index). Flattened row-major.
/// @param[in] candidate_mesh The mesh with the facets we want to compute the
/// distance to
/// @param[in] candidate_facets The facets on candidate_mesh,defined as (cell,
/// local_facet_index). Flattened row-major.
/// @param[in] q_rule The quadrature rule for the input facets
/// @param[in] mode The contact mode, either closest point or ray-tracing
/// @param[in] radius The search radius. Only used for ray-tracing at the moment
/// @returns A tuple (closest_facets, reference_points) where `closest_facets`
/// is an adjacency list for each input facet in quadrature facets, where the
/// links indicate which facet on the other mesh is closest for each quadrature
/// point.`reference_points` is the corresponding points on the reference
/// element for each quadrature point.  Shape (num_facets, num_q_points, tdim).
/// Flattened to (num_facets*num_q_points, tdim).
std::tuple<dolfinx::graph::AdjacencyList<std::int32_t>, std::vector<double>,
           std::array<std::size_t, 2>>
compute_distance_map(const dolfinx::mesh::Mesh& quadrature_mesh,
                     std::span<const std::int32_t> quadrature_facets,
                     const dolfinx::mesh::Mesh& candidate_mesh,
                     std::span<const std::int32_t> candidate_facets,
                     const QuadratureRule& q_rule,
                     dolfinx_contact::ContactMode mode, const double radius);

/// Compute facet indices from given pairs (cell, local__facet)
/// @param[in] facet_pairs The facets given as pair (cell, local_facet).
/// Flattened row major.
/// @param[in] mesh The mesh
/// @return vector of facet indices
std::vector<int32_t>
facet_indices_from_pair(std::span<const std::int32_t> facet_pairs,
                        const dolfinx::mesh::Mesh& mesh);

/// Compute the relation between a set of points and a mesh by computing the
/// closest point on mesh at a specific set of points. There is also a subset
/// of facets on mesh we use for intersection checks.
/// @param[in] mesh The mesh to compute the closest point at
/// @param[in] facet_tuples Set of facets in the of
/// tuples (cell_index, local_facet_index) for the
/// `quadrature_mesh`. Flattened row major.
/// @param[in] points The points to compute the closest entity from.
/// Shape (num_quadrature_points, 3). Flattened row-major
/// @returns A tuple (closest_facets, reference_points), where
/// `closest_entities[i]` is the closest entity in `facet_tuples` for the ith
/// input point
template <std::size_t tdim, std::size_t gdim>
std::tuple<std::vector<std::int32_t>, std::vector<double>,
           std::array<std::size_t, 2>>
compute_projection_map(const dolfinx::mesh::Mesh& mesh,
                       std::span<const std::int32_t> facet_tuples,
                       std::span<const double> points)
{
  assert(tdim == mesh.topology().dim());
  assert(mesh.geometry().dim() == gdim);

  const std::size_t num_points = points.size() / 3;

  // Convert cell,local_facet_index to facet_index (local
  // to proc)
  std::vector<std::int32_t> facets(facet_tuples.size() / 2);
  auto c_to_f = mesh.topology().connectivity(tdim, tdim - 1);
  if (!c_to_f)
  {
    throw std::runtime_error("Missing cell->facet connectivity on candidate "
                             "mesh.");
  }

  for (std::size_t i = 0, j = 0; i < facet_tuples.size(); j++, i += 2)
  {
    auto local_facets = c_to_f->links(facet_tuples[i]);
    assert(!local_facets.empty());
    assert((std::size_t)facet_tuples[i + 1] < local_facets.size());
    facets[j] = local_facets[facet_tuples[i + 1]];
  }

  // Compute closest entity for each point
  dolfinx::geometry::BoundingBoxTree bbox(mesh, tdim - 1, facets);
  dolfinx::geometry::BoundingBoxTree midpoint_tree
      = dolfinx::geometry::create_midpoint_tree(mesh, tdim - 1, facets);
  std::vector<std::int32_t> closest_facets
      = dolfinx::geometry::compute_closest_entity(bbox, midpoint_tree, mesh,
                                                  points);
  std::vector<double> candidate_x(num_points * 3);
  std::span<const double> mesh_geometry = mesh.geometry().x();
  const dolfinx::fem::CoordinateElement& cmap = mesh.geometry().cmap();
  {
    // Find displacement vector from each point
    // to closest entity. As a point on the surface
    // might have penetrated the cell in question, we use
    // the convex hull of the surface facet for distance
    // computations

    // Get information aboute cell type and number of
    // closure dofs on the facet NOTE: Assumption that we
    // do not have variable facet types (prism/pyramid
    // cell)
    const dolfinx::fem::ElementDofLayout layout = cmap.create_dof_layout();

    error::check_cell_type(mesh.topology().cell_type());
    const std::vector<std::int32_t>& closure_dofs
        = layout.entity_closure_dofs(tdim - 1, 0);
    const std::size_t num_facet_dofs = closure_dofs.size();

    // Get the geometry dofs of closest facets
    const dolfinx::graph::AdjacencyList<std::int32_t> facets_geometry
        = dolfinx_contact::entities_to_geometry_dofs(mesh, tdim - 1,
                                                     closest_facets);
    assert(facets_geometry.num_nodes() == (int)num_points);

    // Compute physical points for each facet
    std::vector<double> coordinate_dofs(3 * num_facet_dofs);
    for (std::size_t i = 0; i < num_points; ++i)
    {
      // Get the geometry dofs for the ith facet, qth
      // quadrature point
      auto candidate_facet_dofs = facets_geometry.links(i);
      assert(num_facet_dofs == candidate_facet_dofs.size());

      // Get the (geometrical) coordinates of the facets
      for (std::size_t l = 0; l < num_facet_dofs; ++l)
      {
        std::copy_n(
            std::next(mesh_geometry.begin(), 3 * candidate_facet_dofs[l]), 3,
            std::next(coordinate_dofs.begin(), 3 * l));
      }

      // Compute distance between convex hull of facet and point
      std::array<double, 3> dist_vec = dolfinx::geometry::compute_distance_gjk(
          coordinate_dofs, std::span(points.data() + 3 * i, 3));

      // Compute point on closest facet
      for (std::size_t l = 0; l < 3; ++l)
        candidate_x[3 * i + l] = points[3 * i + l] + dist_vec[l];
    }
  }

  // Pull back to reference point for each facet on the surface
  std::vector<double> candidate_X(num_points * tdim);
  {
    // Temporary data structures used in loop over each
    // quadrature point on each facet
    std::array<double, 9> Jb;
    mdspan3_t J(Jb.data(), 1, gdim, tdim);
    std::array<double, 9> Kb;
    mdspan3_t K(Kb.data(), 1, tdim, gdim);
    std::array<double, 1> detJ;

    std::array<double, gdim> x;
    std::array<double, tdim> X;
    const std::size_t num_dofs_g = cmap.dim();
    const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh.geometry().dofmap();
    std::vector<double> coordinate_dofsb(num_dofs_g * gdim);
    cmdspan2_t coordinate_dofs(coordinate_dofsb.data(), num_dofs_g, gdim);
    auto f_to_c = mesh.topology().connectivity(tdim - 1, tdim);
    if (!f_to_c)
      throw std::runtime_error("Missing facet to cell connectivity");
    for (std::size_t i = 0; i < closest_facets.size(); ++i)
    {
      // Get cell connected to facet
      auto cells = f_to_c->links(closest_facets[i]);
      assert(cells.size() == 1);

      // Pack coordinate dofs
      auto x_dofs = x_dofmap.links(cells.front());
      assert(x_dofs.size() == num_dofs_g);
      for (std::size_t j = 0; j < num_dofs_g; ++j)
      {
        std::copy_n(std::next(mesh_geometry.begin(), 3 * x_dofs[j]), gdim,
                    std::next(coordinate_dofsb.begin(), j * gdim));
      }

      // Copy closest point in physical space
      std::fill(x.begin(), x.end(), 0);
      std::copy_n(std::next(candidate_x.begin(), 3 * i), gdim, x.begin());

      // NOTE: Would benefit from pulling back all points
      // in a single cell at the same time
      // Pull back coordinates
      std::fill(Jb.begin(), Jb.end(), 0);
      pull_back(J, K, detJ, X, cmdspan2_t(x.data(), 1, gdim), coordinate_dofs,
                cmap);
      // Copy into output
      std::copy_n(X.begin(), tdim, std::next(candidate_X.begin(), i * tdim));
    }
  }
  return {closest_facets, candidate_X, {candidate_X.size() / tdim, tdim}};
}

/// Compute the relation between two meshes (mesh_q) and
/// (mesh_c) by computing the intersection of rays from
/// mesh_q onto mesh_c at a specific set of quadrature
/// points on a subset of facets. There is also a subset of
/// facets on mesh_c we use for intersection checks.
/// NOTE: If the ray intersects with more than one facet
/// in the subset of facets on mesh_c, only one of these
/// facets is detected and it is not guaranteed to be the
/// closest.
/// @param[in] quadrature_mesh The mesh to compute rays
/// from
/// @param[in] quadrature_facets Set of facets in the of
/// tuples (cell_index, local_facet_index) for the
/// `quadrature_mesh`. Flattened row major.
/// @param[in] q_rule The quadrature rule to use on the facets
/// @param[in] candidate_mesh The mesh to compute ray
/// intersections with
/// @param[in] candidate_facets Set of facets in the of
/// tuples (cell_index, local_facet_index) for the
/// `quadrature_mesh`. Flattened row major.
/// @param[in] radius The search radius
/// @returns A tuple (facet_map, reference_points), where
/// `facet_map` is an AdjacencyList from the ith facet
/// tuple in `quadrature_facets` to the facet (index local
/// to process) in `candidate_facets`. `reference_points`
/// are the reference points for the point of intersection
/// for each of the quadrature points on each facet. Shape
/// (num_facets, num_q_points, tdim). Flattened to
/// (num_facets*num_q_points, tdim).
template <std::size_t tdim, std::size_t gdim>
std::tuple<dolfinx::graph::AdjacencyList<std::int32_t>, std::vector<double>,
           std::array<std::size_t, 2>>
compute_raytracing_map(const dolfinx::mesh::Mesh& quadrature_mesh,
                       std::span<const std::int32_t> quadrature_facets,
                       const QuadratureRule& q_rule,
                       const dolfinx::mesh::Mesh& candidate_mesh,
                       std::span<const std::int32_t> candidate_facets,
                       const double search_radius = -1.)
{
  dolfinx::common::Timer timer("~Raytracing");
  assert(candidate_mesh.geometry().dim() == gdim);
  assert(quadrature_mesh.geometry().dim() == gdim);
  assert(candidate_mesh.topology().dim() == tdim);
  assert(quadrature_mesh.topology().dim() == tdim);

  // Get quadrature points on reference facets
  const std::vector<double>& q_points = q_rule.points();
  const std::vector<std::size_t>& q_offset = q_rule.offset();
  const std::size_t num_q_points = q_offset[1] - q_offset[0];
  const std::size_t sum_q_points = q_offset.back();

  // Get facet indices for qudrature and candidate facets
  std::vector<std::int32_t> q_facets = dolfinx_contact::facet_indices_from_pair(
      quadrature_facets, quadrature_mesh);
  std::vector<std::int32_t> c_facets = dolfinx_contact::facet_indices_from_pair(
      candidate_facets, candidate_mesh);
  // Structures used for computing physical normal
  std::array<double, 9> Jb;
  mdspan2_t J(Jb.data(), gdim, tdim);
  std::array<double, 9> Kb;
  mdspan2_t K(Kb.data(), tdim, gdim);
  std::array<double, 9> Kcb;
  mdspan2_t K_c(Kcb.data(), tdim, gdim);

  // Get relevant information from quadrature mesh
  const dolfinx::mesh::Geometry& geom_q = quadrature_mesh.geometry();
  const dolfinx::fem::CoordinateElement& cmap_q
      = quadrature_mesh.geometry().cmap();
  const dolfinx::mesh::Topology& top_q = quadrature_mesh.topology();
  std::span<const double> q_x = geom_q.x();
  const dolfinx::graph::AdjacencyList<std::int32_t>& q_dofmap = geom_q.dofmap();
  const std::size_t num_nodes_q = cmap_q.dim();
  std::vector<double> coordinate_dofs_qb(num_nodes_q * gdim);
  cmdspan2_t coordinate_dofs_q(coordinate_dofs_qb.data(), num_nodes_q, gdim);
  auto [reference_normals, rn_shape] = basix::cell::facet_outward_normals(
      dolfinx::mesh::cell_type_to_basix_type(top_q.cell_type()));

  // Tabulate at all quadrature points in quadrature rule with quadrature cmap
  const std::array<std::size_t, 4> basis_shape_q
      = cmap_q.tabulate_shape(1, sum_q_points);
  std::vector<double> basis_q(std::reduce(
      basis_shape_q.cbegin(), basis_shape_q.cend(), 1, std::multiplies{}));
  cmap_q.tabulate(1, q_points, {sum_q_points, (std::size_t)tdim}, basis_q);

  // Push forward quadrature points to physical space
  std::vector<double> quadrature_points(quadrature_facets.size() / 2
                                        * num_q_points * gdim);
  cmdspan4_t basis_values_q(basis_q.data(), basis_shape_q);
  compute_physical_points(quadrature_mesh, quadrature_facets, q_offset,
                          basis_values_q, quadrature_points);

  // Structures used for raytracing
  dolfinx::mesh::CellType cell_type = candidate_mesh.topology().cell_type();
  const dolfinx::mesh::Geometry& c_geometry = candidate_mesh.geometry();
  const dolfinx::fem::CoordinateElement& cmap_c = c_geometry.cmap();
  const dolfinx::graph::AdjacencyList<std::int32_t>& c_dofmap
      = c_geometry.dofmap();
  std::span<const double> c_x = c_geometry.x();

  const std::array<std::size_t, 4> basis_shape_c = cmap_c.tabulate_shape(1, 1);
  const std::size_t num_nodes_c = cmap_c.dim();
  std::vector<double> coordinate_dofs_c(num_nodes_c * gdim);
  std::vector<double> basis_values_c(std::reduce(
      basis_shape_c.begin(), basis_shape_c.end(), 1, std::multiplies{}));
  std::array<double, 3> normal_c;

  // Variable to hold jth point for Jacbian computation
  std::array<double, 3> normal;
  std::vector<std::int32_t> colliding_facet(
      quadrature_facets.size() / 2 * num_q_points, -1);
  std::vector<double> reference_points(
      quadrature_facets.size() / 2 * num_q_points * tdim, 0);

  // Check for parameterization and jacobian parameterization
  error::check_cell_type(cell_type);
  basix::cell::type basix_cell
      = dolfinx::mesh::cell_type_to_basix_type(cell_type);
  assert(dolfinx::mesh::cell_dim(cell_type) == tdim);

  // Get facet jacobians from Basix
  auto [ref_jac, jac_shape] = basix::cell::facet_jacobians(basix_cell);
  assert(tdim == jac_shape[1]);
  assert(tdim - 1 == jac_shape[2]);
  cmdspan3_t facet_jacobians(ref_jac.data(), jac_shape);

  // Get basix geometry information
  std::pair<std::vector<double>, std::array<std::size_t, 2>> geometry
      = basix::cell::geometry(basix_cell);
  auto xb = geometry.first;
  auto x_shape = geometry.second;
  const std::vector<std::vector<int>> bfacets
      = basix::cell::topology(basix_cell)[tdim - 1];

  NewtonStorage<tdim, gdim> allocated_memory;
  auto tangents = allocated_memory.tangents();
  auto point = allocated_memory.point();
  auto dxi = allocated_memory.dxi();
  auto X_fin = allocated_memory.X_k();

  // This array stores for the current facet for which quadrature point no
  // valid contact point is determined
  std::vector<std::size_t> missing_matches(num_q_points);

  for (std::size_t i = 0; i < quadrature_facets.size(); i += 2)
  {
    std::size_t count_missing_matches = 0; // counter for missing contact points

    // Determine candidate facets within search radius
    // FIXME: This is not the most efficient way of finding close facets
    std::vector<size_t> cand_patch
        = find_candidate_facets(quadrature_mesh, candidate_mesh,
                                q_facets[i / 2], c_facets, 2 * search_radius);

    // Pack coordinate dofs
    auto x_dofs = q_dofmap.links(quadrature_facets[i]);
    assert(x_dofs.size() == num_nodes_q);
    for (std::size_t j = 0; j < num_nodes_q; ++j)
    {
      std::copy_n(std::next(q_x.begin(), 3 * x_dofs[j]), gdim,
                  std::next(coordinate_dofs_qb.begin(), j * gdim));
    }
    const std::int32_t facet_index = quadrature_facets[i + 1];
    for (std::size_t j = 0; j < num_q_points; ++j)
    {

      auto dphi_q = stdex::submdspan(
          basis_values_q, std::pair{1, (std::size_t)tdim + 1},
          std::size_t(num_q_points * facet_index + j), stdex::full_extent, 0);
      std::fill(Jb.begin(), Jb.end(), 0);
      dolfinx::fem::CoordinateElement::compute_jacobian(dphi_q,
                                                        coordinate_dofs_q, J);
      dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J, K);

      // Push forward normal using covariant Piola
      // transform
      std::fill(normal.begin(), normal.end(), 0);
      physical_facet_normal(
          std::span(normal.data(), gdim), K,
          std::span(reference_normals.data() + rn_shape[1] * facet_index,
                    rn_shape[1]));

      // Copy data regarding quadrature point into allocated memory for
      // raytracing
      std::copy_n(std::next(quadrature_points.cbegin(),
                            (i / 2 * num_q_points + j) * gdim),
                  gdim, point.begin());
      impl::compute_tangents<gdim>(std::span<double, gdim>(normal.data(), gdim),
                                   tangents);

      std::size_t cell_idx = -1;
      int status = 0;
      for (std::size_t c = 0; c < cand_patch.size(); ++c)
      {
        std::int32_t cell = candidate_facets[2 * cand_patch[c]];
        std::int32_t facet_index_c = candidate_facets[2 * cand_patch[c] + 1];
        // Get cell geometry for candidate cell, reusing
        // coordinate dofs to store new coordinate
        auto x_dofs_c = c_dofmap.links(cell);
        for (std::size_t k = 0; k < x_dofs_c.size(); ++k)
        {
          std::copy_n(std::next(c_x.begin(), 3 * x_dofs_c[k]), gdim,
                      std::next(coordinate_dofs_c.begin(), gdim * k));
        }
        // Assign Jacobian of reference mapping
        for (std::size_t l = 0; l < tdim; ++l)
          for (std::size_t m = 0; m < tdim - 1; ++m)
            dxi(l, m) = facet_jacobians(facet_index_c, l, m);

        // Get parameterization map
        std::function<void(std::span<const double, tdim - 1>,
                           std::span<double, tdim>)>
            reference_map
            = [&xb, &x_shape, &bfacets, facet_index = facet_index_c](
                  std::span<const double, tdim - 1> xi,
                  std::span<double, tdim> X)
        {
          const std::vector<int>& facet = bfacets[facet_index];
          dolfinx_contact::cmdspan2_t x(xb.data(), x_shape);
          const int f0 = facet.front();
          for (std::size_t i = 0; i < tdim; ++i)
          {
            X[i] = x(f0, i);
            for (std::size_t j = 0; j < tdim - 1; ++j)
              X[i] += (x(facet[j + 1], i) - x(f0, i)) * xi[j];
          }
        };

        status = raytracing_cell<tdim, gdim>(
            allocated_memory, basis_values_c, basis_shape_c, 25, 1e-8, cmap_c,
            cell_type, coordinate_dofs_c, reference_map);

        // compute normal of candidate facet
        std::fill(normal_c.begin(), normal_c.end(), 0);
        auto J_c = allocated_memory.J();
        dolfinx::fem::CoordinateElement::compute_jacobian_inverse(J_c, K_c);
        dolfinx_contact::physical_facet_normal(
            std::span(normal_c.data(), gdim), K_c,
            std::span(reference_normals.data() + rn_shape[1] * facet_index_c,
                      rn_shape[1]));

        // retrieve ray
        std::array<double, gdim> ray;
        for (std::size_t l = 0; l < gdim; ++l)
          ray[l] = allocated_memory.x_k()[l] - point[l];

        // Compute norm of ray and dot product of normals
        double norm = 0;
        double dot = 0;
        for (std::size_t l = 0; l < gdim; ++l)
        {
          dot += normal[l] * normal_c[l];
          norm += ray[l] * ray[l];
        }

        // check criteria for valid contact pair
        // 1. Compatible normals (normals pointing in opposite directions)
        // 2. Point within search radius
        if (dot > 0 || (search_radius > 0 && norm > search_radius))
          status = -5;
        if (status > 0)
        {
          cell_idx = cand_patch[c];
          // Break loop
          c = cand_patch.size();
        }
      }
      if (status > 0)
      {
        colliding_facet[i / 2 * num_q_points + j] = c_facets[cell_idx];
        std::copy_n(X_fin.begin(), tdim,
                    std::next(reference_points.begin(),
                              tdim * (i / 2 * num_q_points + j)));
      }
      else
      {
        // save quadrature points with no valid contact point
        missing_matches[count_missing_matches] = j;
        count_missing_matches += 1;
      }
    }
    // If contact points are found for some, but not all quadrature points
    // Use closest point projection to add contact points for remainig
    // quadrature points
    if (count_missing_matches > 0 && count_missing_matches < num_q_points)
    {
      std::vector<std::int32_t> cand_facets_patch(2 * cand_patch.size());
      std::vector<double> padded_qpsb(count_missing_matches * 3);
      dolfinx_contact::mdspan2_t padded_qps(padded_qpsb.data(),
                                            count_missing_matches, 3);
      dolfinx_contact::cmdspan3_t qps(quadrature_points.data(),
                                      quadrature_facets.size() / 2,
                                      num_q_points, gdim);

      // Retrieve remaining quadrature points
      for (std::size_t j = 0; j < padded_qps.extent(0); ++j)
        for (std::size_t k = 0; k < qps.extent(2); ++k)
          padded_qps(j, k) = qps(i / 2, missing_matches[j], k);

      // Retrieve candidate facets as (cell, local_facet) pair
      for (std::size_t c = 0; c < cand_patch.size(); ++c)
      {
        cand_facets_patch[2 * c] = candidate_facets[2 * cand_patch[c]];
        cand_facets_patch[2 * c + 1] = candidate_facets[2 * cand_patch[c] + 1];
      }
      // find closest enities
      auto [closest_entities, reference_points_2, shape_2]
          = compute_projection_map<tdim, gdim>(candidate_mesh,
                                               cand_facets_patch, padded_qpsb);

      // insert facets and reference points into the relevant arrays
      for (std::size_t j = 0; j < count_missing_matches; ++j)
      {
        colliding_facet[i / 2 * num_q_points + missing_matches[j]]
            = closest_entities[j];
        std::copy_n(
            std::next(reference_points_2.begin(), j * tdim), tdim,
            std::next(reference_points.begin(),
                      tdim * (i / 2 * num_q_points + missing_matches[j])));
      }
    }
  }
  std::vector<std::int32_t> offset(quadrature_facets.size() / 2 + 1);
  std::iota(offset.begin(), offset.end(), 0);
  std::for_each(offset.begin(), offset.end(),
                [num_q_points](auto& i) { i *= num_q_points; });
  timer.stop();
  return {dolfinx::graph::AdjacencyList<std::int32_t>(colliding_facet, offset),
          reference_points,
          std::array<std::size_t, 2>{reference_points.size() / tdim, tdim}};
}

} // namespace dolfinx_contact
