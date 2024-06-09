// Copyright (C) 2021-2022 Sarah Roggendorf and Jorgen S. Dokken
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "kernelwrapper.h"
#include <array.h>
#include <caster_petsc.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx_contact/Contact.h>
#include <dolfinx_contact/MeshTie.h>
#include <dolfinx_contact/QuadratureRule.h>
#include <dolfinx_contact/RayTracing.h>
#include <dolfinx_contact/SubMesh.h>
#include <dolfinx_contact/coefficients.h>
#include <dolfinx_contact/elasticity.h>
#include <dolfinx_contact/parallel_mesh_ghosting.h>
#include <dolfinx_contact/point_cloud.h>
#include <dolfinx_contact/rigid_surface_kernels.h>
#include <dolfinx_contact/utils.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <petsc4py/petsc4py.h>
#include <petscmat.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace nb = nanobind;
using namespace nb::literals;

using scalar_t = PetscScalar;

NB_MODULE(cpp, m)
{
  // Create module for C++ wrappers
  m.doc() = "DOLFINx Contact Python interface";
#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  // Kernel wrapper class
  nb::class_<contact_wrappers::KernelWrapper<T>>(
      m, "KernelWrapper", "Wrapper for C++ contact integration kernels");

  nb::enum_<dolfinx_contact::ContactMode>(m, "ContactMode")
      .value("ClosestPoint", dolfinx_contact::ContactMode::ClosestPoint)
      .value("Raytracing", dolfinx_contact::ContactMode::RayTracing);

  nb::enum_<dolfinx_contact::Problem>(m, "Problem")
      .value("Elasticity", dolfinx_contact::Problem::Elasticity)
      .value("Poisson", dolfinx_contact::Problem::Poisson)
      .value("ThermoElasticity", dolfinx_contact::Problem::ThermoElasticity);

  // QuadratureRule
  nb::class_<dolfinx_contact::QuadratureRule>(m, "QuadratureRule",
                                              "QuadratureRule object")
      .def(
          "__init__",
          [](dolfinx_contact::QuadratureRule* qr,
             dolfinx::mesh::CellType cell_type, int degree, int dim,
             basix::quadrature::type type) {
            new (qr)
                dolfinx_contact::QuadratureRule(cell_type, degree, dim, type);
          },
          nb::arg("cell_type"), nb::arg("degree"), nb::arg("dim"),
          nb::arg("type"))
      .def("points",
           [](dolfinx_contact::QuadratureRule& self)
           {
             std::vector<double> _points = self.points();
             return dolfinx_wrappers::as_nbarray(
                 std::move(_points),
                 {_points.size() / self.tdim(), self.tdim()});
           })
      .def("points",
           [](dolfinx_contact::QuadratureRule& self, int i)
           {
             dolfinx_contact::cmdspan2_t points_i = self.points(i);
             std::array<std::size_t, 2> shape
                 = {points_i.extent(0), points_i.extent(1)};
             std::vector<double> _points(shape[0] * shape[1]);
             for (std::size_t i = 0; i < points_i.extent(0); ++i)
               for (std::size_t j = 0; j < points_i.extent(1); ++j)
                 _points[i * shape[1] + j] = points_i(i, j);
             return dolfinx_wrappers::as_nbarray(std::move(_points),
                                                 {shape[0], shape[1]});
           })
      .def("weights", [](dolfinx_contact::QuadratureRule& self)
           { return dolfinx_wrappers::as_nbarray(self.weights()); })
      .def("weights",
           [](dolfinx_contact::QuadratureRule& self, int i)
           {
             std::span<const double> weights_i = self.weights(i);
             return dolfinx_wrappers::as_nbarray(
                 std::vector(weights_i.begin(), weights_i.end()));
           });

  // Contact
  nb::class_<dolfinx_contact::Contact>(m, "Contact", "Contact object")
      .def(
          nb::init<const std::vector<
                       std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>>&,
                   std::shared_ptr<
                       const dolfinx::graph::AdjacencyList<std::int32_t>>,
                   const std::vector<std::array<int, 2>>&,
                   std::shared_ptr<dolfinx::mesh::Mesh<double>>,
                   std::vector<dolfinx_contact::ContactMode>, const int>(),
          nb::arg("markers"), nb::arg("surfaces"), nb::arg("contact_pairs"),
          nb::arg("mesh"), nb::arg("search_method"),
          nb::arg("quadrature_degree") = 3)
      .def("create_distance_map",
           [](dolfinx_contact::Contact& self, int pair)
           {
             self.create_distance_map(pair);
             return;
           })
      .def("pack_gap_plane",
           [](dolfinx_contact::Contact& self, int origin_meshtag, double g)
           {
             auto [coeffs, cstride] = self.pack_gap_plane(origin_meshtag, g);
             std::size_t shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
             return dolfinx_wrappers::as_nbarray(
                 std::move(coeffs), {shape0, (std::size_t)cstride});
           })
      .def("pack_gap",
           [](dolfinx_contact::Contact& self, int origin_meshtag)
           {
             auto [coeffs, cstride] = self.pack_gap(origin_meshtag);
             std::size_t shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
             return dolfinx_wrappers::as_nbarray(
                 std::move(coeffs), {shape0, (std::size_t)cstride});
           })
      .def("local_facets", &dolfinx_contact::Contact::local_facets)
      .def("contact_pair", &dolfinx_contact::Contact::contact_pair)
      .def(
          "create_matrix",
          [](dolfinx_contact::Contact& self, dolfinx::fem::Form<scalar_t>& a,
             std::string type)
          {
            Mat A = self.create_petsc_matrix(a, type);
            return A;
          },
          nb::rv_policy::take_ownership, nb::arg("a"),
          nb::arg("type") = std::string(),
          "Create a PETSc Mat for two-sided contact.")
      .def(
          "qp_phys",
          [](dolfinx_contact::Contact& self, int origin_meshtag, int facet)
          {
            auto [qp, qp_shape] = self.qp_phys(origin_meshtag);
            dolfinx_contact::cmdspan3_t qp_span(qp.data(), qp_shape);
            std::vector<double> qp_vec(qp_shape[1] * qp_shape[2]);
            for (std::size_t i = 0; i < qp_shape[1]; ++i)
              for (std::size_t j = 0; j < qp_shape[2]; ++j)
                qp_vec[i * qp_shape[2] + j] = qp_span(facet, i, j);

            return dolfinx_wrappers::as_nbarray(std::move(qp_vec),
                                                {qp_shape[1], qp_shape[2]});
          },
          "Get quadrature points for the jth facet of the ith contact pair")
      .def("active_entities",
           [](dolfinx_contact::Contact& self, int s)
           {
             std::span<const std::int32_t> active_entities
                 = self.active_entities(s);
             return nb::ndarray<const std::int32_t, nb::numpy>(
                 active_entities.data(), {active_entities.size() / 2, 2},
                 nb::handle());
           })
      .def("facet_map",
           [](dolfinx_contact::Contact& self, int pair)
           {
             // This exposes facet_map() to Python but replaces the
             // facet indices on the submesh with the facet indices in
             // the parent mesh This is only exposed for testing (in
             // particular.
             // nitsche_rigid_surface.py/demo_nitsche_rigid_surface_ufl.py)
             //  auto contact_pair = self.contact_pair(pair);
             std::shared_ptr<const dolfinx::mesh::Mesh<double>> mesh
                 = self.mesh();
             int tdim = mesh->topology()->dim(); // topological dimension

             int fdim = tdim - 1; // topological dimension of facet
             auto c_to_f = mesh->topology()->connectivity(tdim, fdim);
             assert(c_to_f);
             std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
                 submesh_map = self.facet_map(pair);
             const std::vector<int>& offsets = submesh_map->offsets();
             const std::vector<std::int32_t>& old_data = submesh_map->array();
             std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
                 = self.submesh().facet_map();
             std::span<const std::int32_t> parent_cells
                 = self.submesh().parent_cells();
             std::vector<std::int32_t> data(old_data.size());
             std::transform(
                 old_data.cbegin(), old_data.cend(), data.begin(),
                 [&facet_map, &parent_cells, &c_to_f](auto submesh_facet)
                 {
                   if (submesh_facet < 0)
                     return -1;
                   else
                   {
                     auto facet_pair = facet_map->links(submesh_facet);
                     assert(facet_pair.size() == 2);
                     auto cell_parent = parent_cells[facet_pair.front()];
                     return c_to_f->links(cell_parent)[facet_pair.back()];
                   }
                 });
             return std::make_shared<
                 dolfinx::graph::AdjacencyList<std::int32_t>>(
                 std::move(data), std::move(offsets));
           })
      .def("submesh",
           [](dolfinx_contact::Contact& self)
           {
             const dolfinx_contact::SubMesh& submesh = self.submesh();
             return submesh.mesh();
           })
      .def("mesh", &dolfinx_contact::Contact::mesh)
      .def("copy_to_submesh",
           [](const dolfinx_contact::Contact& self,
              const dolfinx::fem::Function<scalar_t>& u,
              dolfinx::fem::Function<scalar_t>& u_sub)
           {
             dolfinx_contact::SubMesh submesh = self.submesh();
             submesh.copy_function(u, u_sub);
           })
      .def("coefficients_size", &dolfinx_contact::Contact::coefficients_size,
           nb::arg("meshtie"), nb::arg("V"))
      .def("set_quadrature_rule",
           &dolfinx_contact::Contact::set_quadrature_rule)
      .def("set_search_radius", &dolfinx_contact::Contact::set_search_radius)
      .def("generate_kernel",
           [](dolfinx_contact::Contact& self, dolfinx_contact::Kernel type,
              const dolfinx::fem::FunctionSpace<double>& V) {
             return contact_wrappers::KernelWrapper<T>(
                 self.generate_kernel(type, V));
           })
      .def("assemble_matrix",
           [](dolfinx_contact::Contact& self, Mat A, int origin_meshtag,
              contact_wrappers::KernelWrapper<T>& kernel,
              nb::ndarray<const scalar_t, nb::c_contig> coeffs,
              nb::ndarray<const scalar_t, nb::c_contig> constants,
              const dolfinx::fem::FunctionSpace<double>& V)
           {
             auto ker = kernel.get();
             self.assemble_matrix(
                 dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES),
                 origin_meshtag, ker,
                 std::span<const scalar_t>(coeffs.data(), coeffs.size()),
                 coeffs.shape(1),
                 std::span(constants.data(), constants.shape(0)), V);
           })
      .def("assemble_vector",
           [](dolfinx_contact::Contact& self,
              nb::ndarray<scalar_t, nb::ndim<1>, nb::c_contig> b,
              int origin_meshtag, contact_wrappers::KernelWrapper<T>& kernel,
              nb::ndarray<const scalar_t, nb::c_contig> coeffs,
              nb::ndarray<const scalar_t, nb::c_contig> constants,
              const dolfinx::fem::FunctionSpace<double>& V)
           {
             auto ker = kernel.get();
             self.assemble_vector(
                 std::span(b.data(), b.size()), origin_meshtag, ker,
                 std::span(coeffs.data(), coeffs.size()), coeffs.shape(1),
                 std::span(constants.data(), constants.size()), V);
           })
      .def("pack_test_functions",
           [](dolfinx_contact::Contact& self, int origin_meshtag,
              const dolfinx::fem::FunctionSpace<double>& V)
           {
             auto [coeffs, cstride]
                 = self.pack_test_functions(origin_meshtag, V);
             std::size_t shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
             return dolfinx_wrappers::as_nbarray(
                 std::move(coeffs), {shape0, (std::size_t)cstride});
           })
      .def("pack_grad_test_functions",
           [](dolfinx_contact::Contact& self, int origin_meshtag,
              const dolfinx::fem::FunctionSpace<double>& V)
           {
             auto [coeffs, cstride]
                 = self.pack_grad_test_functions(origin_meshtag, V);
             std::size_t shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
             return dolfinx_wrappers::as_nbarray(
                 std::move(coeffs), {shape0, (std::size_t)cstride});
           })
      .def("pack_nx",
           [](dolfinx_contact::Contact& self, int origin_meshtag)
           {
             auto [coeffs, cstride] = self.pack_nx(origin_meshtag);
             std::size_t shape0
                 = cstride == 0 ? 0 : coeffs.size() / cstride;
             return dolfinx_wrappers::as_nbarray(
                 std::move(coeffs), {shape0, (std::size_t)cstride});
           })
      .def("pack_ny",
           [](dolfinx_contact::Contact& self, int origin_meshtag)
           {
             auto [coeffs, cstride] = self.pack_ny(origin_meshtag);
             std::size_t shape0
                 = cstride == 0 ? 0 : coeffs.size() / cstride;
             return dolfinx_wrappers::as_nbarray(
                 std::move(coeffs), {shape0, (std::size_t)cstride});
           })
      .def(
          "pack_u_contact",
          [](dolfinx_contact::Contact& self, int origin_meshtag,
             const dolfinx::fem::Function<scalar_t>& u)
          {
            auto [coeffs, cstride] = self.pack_u_contact(origin_meshtag, u);
            int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
            return dolfinx_wrappers::as_nbarray(
                std::move(coeffs), {(std::size_t)shape0, (std::size_t)cstride});
          },
          nb::arg("origin_meshtag"), nb::arg("u"))
      .def("pack_grad_u_contact",
           [](dolfinx_contact::Contact& self, int origin_meshtag,
              const dolfinx::fem::Function<scalar_t>& u)
           {
             auto [coeffs, cstride]
                 = self.pack_grad_u_contact(origin_meshtag, u);
             std::size_t shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
             return dolfinx_wrappers::as_nbarray(
                 std::move(coeffs), {shape0, (std::size_t)cstride});
           })
      .def("update_submesh_geometry",
           &dolfinx_contact::Contact::update_submesh_geometry)
      .def("crop_invalid_points",
           [](dolfinx_contact::Contact& self, int pair,
              nb::ndarray<const scalar_t, nb::ndim<1>, nb::c_contig> gap,
              nb::ndarray<const scalar_t, nb::ndim<1>, nb::c_contig> n_y,
              double tol)
           {
             return self.crop_invalid_points(
                 pair, std::span(gap.data(), gap.size()),
                 std::span(n_y.data(), n_y.size()), tol);
           })
      .def("max_links",
           [](dolfinx_contact::Contact& self) { return self.max_links(); });

  m.def(
      "generate_rigid_surface_kernel",
      [](const dolfinx::fem::FunctionSpace<double>& V,
         dolfinx_contact::Kernel type, dolfinx_contact::QuadratureRule& q_rule,
         bool constant_normal)
      {
        return contact_wrappers::KernelWrapper<T>(
            dolfinx_contact::generate_rigid_surface_kernel(V, type, q_rule,
                                                           constant_normal));
      },
      nb::arg("V"), nb::arg("kernel_type"), nb::arg("quadrature_rule"),
      nb::arg("constant_normal") = true);

  nb::enum_<dolfinx_contact::Kernel>(m, "Kernel")
      .value("Rhs", dolfinx_contact::Kernel::Rhs)
      .value("Jac", dolfinx_contact::Kernel::Jac)
      .value("TrescaRhs", dolfinx_contact::Kernel::TrescaRhs)
      .value("TrescaJac", dolfinx_contact::Kernel::TrescaJac)
      .value("CoulombRhs", dolfinx_contact::Kernel::CoulombRhs)
      .value("CoulombJac", dolfinx_contact::Kernel::CoulombJac)
      .value("MeshTieRhs", dolfinx_contact::Kernel::MeshTieRhs)
      .value("MeshTieJac", dolfinx_contact::Kernel::MeshTieJac)
      .value("ThermoElasticRhs", dolfinx_contact::Kernel::ThermoElasticRhs);

  // Contact
  nb::class_<dolfinx_contact::MeshTie>(m, "MeshTie", "meshtie object")
      .def(nb::init<std::vector<
                        std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>>,
                    std::shared_ptr<
                        const dolfinx::graph::AdjacencyList<std::int32_t>>,
                    std::vector<std::array<int, 2>>,
                    std::shared_ptr<dolfinx::mesh::Mesh<double>>, const int>(),
           nb::arg("markers"), nb::arg("surfaces"), nb::arg("contact_pairs"),
           nb::arg("V"), nb::arg("quadrature_degree") = 3)
      .def(
          "coeffs",
          [](dolfinx_contact::MeshTie& self, int pair)
          {
            auto [coeffs, cstride] = self.coeffs(pair);
            return dolfinx_wrappers::as_nbarray(
                std::move(coeffs), {coeffs.size() / cstride, cstride});
          },
          "Get packed coefficients")
      .def("generate_kernel_data",
           &dolfinx_contact::MeshTie::generate_kernel_data,
           nb::arg("problem_type"), nb::arg("V"), nb::arg("coefficients"),
           nb::arg("gamma"), nb::arg("theta"))
      .def("update_kernel_data", &dolfinx_contact::MeshTie::update_kernel_data)
      .def("generate_meshtie_data_matrix_only",
           &dolfinx_contact::MeshTie::generate_meshtie_data_matrix_only)
      .def("generate_poisson_data_matrix_only",
           &dolfinx_contact::MeshTie::generate_poisson_data_matrix_only)
      .def("assemble_matrix",
           [](dolfinx_contact::MeshTie& self, Mat A,
              const dolfinx::fem::FunctionSpace<double>& V,
              dolfinx_contact::Problem problemtype)
           {
             self.assemble_matrix(
                 dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES), V,
                 problemtype);
           })
      .def("assemble_vector",
           [](dolfinx_contact::MeshTie& self,
              nb::ndarray<scalar_t, nb::ndim<1>, nb::c_contig> b,
              const dolfinx::fem::FunctionSpace<double>& V,
              dolfinx_contact::Problem problemtype) {
             self.assemble_vector(std::span(b.data(), b.size()), V,
                                  problemtype);
           })
      .def(
          "create_matrix",
          [](dolfinx_contact::MeshTie& self, dolfinx::fem::Form<scalar_t>& a,
             std::string type) { return self.create_petsc_matrix(a, type); },
          nb::rv_policy::take_ownership, nb::arg("a"),
          nb::arg("type") = std::string(),
          "Create a PETSc Mat for tying disconnected meshes.");
  m.def(
      "pack_coefficient_quadrature",
      [](const dolfinx::fem::Function<scalar_t>& coeff, int q,
         nb::ndarray<const std::int32_t, nb::c_contig> entities)
      {
        std::span<const std::int32_t> e_span(entities.data(), entities.size());
        if (entities.ndim() == 1)
        {
          auto [coeffs, cstride] = dolfinx_contact::pack_coefficient_quadrature(
              coeff, q, e_span, dolfinx::fem::IntegralType::cell);
          std::size_t shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
          return dolfinx_wrappers::as_nbarray(std::move(coeffs),
                                              {shape0, (std::size_t)cstride});
        }
        else if (entities.ndim() == 2)
        {
          auto [coeffs, cstride] = dolfinx_contact::pack_coefficient_quadrature(
              coeff, q, e_span, dolfinx::fem::IntegralType::exterior_facet);
          int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
          return dolfinx_wrappers::as_nbarray(
              std::move(coeffs), {(std::size_t)shape0, (std::size_t)cstride});
        }
        else
          throw std::invalid_argument("Unsupported entities");
      });
  m.def("pack_gradient_quadrature",
        [](const dolfinx::fem::Function<scalar_t>& coeff, int q,
           nb::ndarray<const std::int32_t, nb::c_contig> entities)
        {
          std::span<const std::int32_t> e_span(entities.data(),
                                               entities.size());
          if (entities.ndim() == 1)
          {
            auto [coeffs, cstride] = dolfinx_contact::pack_gradient_quadrature(
                coeff, q, e_span, dolfinx::fem::IntegralType::cell);
            std::size_t shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
            return dolfinx_wrappers::as_nbarray(std::move(coeffs),
                                                {shape0, (std::size_t)cstride});
          }
          else if (entities.ndim() == 2)
          {
            auto [coeffs, cstride] = dolfinx_contact::pack_gradient_quadrature(
                coeff, q, e_span, dolfinx::fem::IntegralType::exterior_facet);
            std::size_t shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
            return dolfinx_wrappers::as_nbarray(std::move(coeffs),
                                                {shape0, (std::size_t)cstride});
          }
          else
            throw std::invalid_argument("Unsupported entities");
        });

  m.def("pack_circumradius",
        [](const dolfinx::mesh::Mesh<double>& mesh,
           nb::ndarray<const std::int32_t, nb::ndim<2>, nb::c_contig>
               active_facets)
        {
          std::span<const std::int32_t> e_span(active_facets.data(),
                                               active_facets.size());
          return dolfinx_wrappers::as_nbarray(
              dolfinx_contact::pack_circumradius(
                  mesh, std::span(active_facets.data(), active_facets.size())),
              {std::size_t(active_facets.size() / 2), (std::size_t)1});
        });
  m.def("update_geometry", [](const dolfinx::fem::Function<scalar_t>& u,
                              dolfinx::mesh::Mesh<double>& mesh)
        { dolfinx_contact::update_geometry(u, mesh); });

  m.def("compute_active_entities",
        [](const dolfinx::mesh::Mesh<double>& mesh,
           nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entities,
           dolfinx::fem::IntegralType integral)
        {
          std::span<const std::int32_t> entity_span(entities.data(),
                                                    entities.size());
          auto [active_entities, num_local]
              = dolfinx_contact::compute_active_entities(mesh, entity_span,
                                                         integral);
          switch (integral)
          {
          case dolfinx::fem::IntegralType::cell:
          {
            return nb::make_tuple(
                dolfinx_wrappers::as_nbarray(std::move(active_entities)),
                num_local);
          }
          case dolfinx::fem::IntegralType::exterior_facet:
          {
            std::array<std::size_t, 2> shape
                = {std::size_t(active_entities.size() / 2), 2};
            return nb::make_tuple(
                dolfinx_wrappers::as_nbarray(std::move(active_entities),
                                             {shape[0], shape[1]}),
                num_local);
          }
          default:
            throw std::invalid_argument(
                "Integral type not supported. Note that this function "
                "has not been implemented for interior facets.");
          }
        });

  m.def(
      "find_candidate_surface_segment",
      [](const dolfinx::mesh::Mesh<double>& mesh,
         const std::vector<std::int32_t>& quadrature_facets,
         const std::vector<std::int32_t>& candidate_facets, double radius)
      {
        return dolfinx_contact::find_candidate_surface_segment(
            mesh, quadrature_facets, candidate_facets, radius);
      },
      nb::arg("mesh"), nb::arg("quadrature_facets"),
      nb::arg("candidate_facets"), nb::arg("radius") = -1.0);

  m.def(
      "point_cloud_pairs",
      [](nb::ndarray<const double, nb::ndim<2>, nb::c_contig> points, double r)
      {
        return dolfinx_contact::point_cloud_pairs(
            std::span(points.data(), points.size()), r);
      });

  m.def("compute_ghost_cell_destinations",
        [](const dolfinx::mesh::Mesh<double>& mesh,
           nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig>
               marker_subset,
           double r)
        {
          return dolfinx_contact::compute_ghost_cell_destinations(
              mesh, std::span(marker_subset.data(), marker_subset.size()), r);
        });

  m.def("lex_match", &dolfinx_contact::lex_match);

  m.def("create_contact_mesh_cpp", &dolfinx_contact::create_contact_mesh);

  m.def(
      "raytracing",
      [](const dolfinx::mesh::Mesh<double>& mesh,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> point,
         nb::ndarray<const double, nb::ndim<1>, nb::c_contig> normal,
         nb::ndarray<const std::int32_t, nb::ndim<2>, nb::c_contig> cells,
         int max_iter, double tol)
      {
        std::span<const std::int32_t> facet_span(cells.data(), cells.size());
        std::size_t gdim = mesh.geometry().dim();
        if (std::size_t(point.size()) != gdim)
        {
          throw std::invalid_argument(
              "Input point has to have same dimension as gdim");
        }

        std::span<const double> _point(point.data(), point.size());
        if (std::size_t(normal.size()) != gdim)
        {
          throw std::invalid_argument(
              "Input normal has to have dimension gdim");
        }
        std::span<const double> _normal(normal.data(), normal.size());
        std::tuple<int, std::int32_t, std::vector<double>, std::vector<double>>
            output = dolfinx_contact::raytracing(mesh, _point, _normal,
                                                 facet_span, max_iter, tol);
        int status = std::get<0>(output);
        auto x = std::get<2>(output);
        auto X = std::get<3>(output);
        std::int32_t idx = std::get<1>(output);
        return nb::make_tuple(status, idx,
                              dolfinx_wrappers::as_nbarray(std::move(x)),
                              dolfinx_wrappers::as_nbarray(std::move(X)));
      },
      nb::arg("mesh"), nb::arg("point"), nb::arg("tangents"), nb::arg("cells"),
      nb::arg("max_iter") = 25, nb::arg("tol") = 1e-8);

  m.def("compute_contact_forces",
        [](nb::ndarray<const scalar_t, nb::ndim<1>, nb::c_contig> grad_u,
           nb::ndarray<const scalar_t, nb::ndim<1>, nb::c_contig> n_x,
           int num_q_points, int num_facets, int gdim, double mu, double lambda)
        {
          return dolfinx_contact::compute_contact_forces(
              std::span(grad_u.data(), grad_u.size()),
              std::span(n_x.data(), n_x.size()), num_q_points, num_facets, gdim,
              mu, lambda);
        });

  m.def(
      "entities_to_geometry_dofs",
      [](const dolfinx::mesh::Mesh<double>& mesh, int dim,
         nb::ndarray<const std::int32_t, nb::ndim<1>, nb::c_contig> entity_list)
      {
        return dolfinx_contact::entities_to_geometry_dofs(
            mesh, dim, std::span(entity_list.data(), entity_list.size()));
      });
}
