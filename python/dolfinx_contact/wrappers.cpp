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
#include <dolfinx_contact/QuadratureRule.h>
#include <dolfinx_contact/RayTracing.h>
#include <dolfinx_contact/coefficients.h>
#include <dolfinx_contact/contact_kernels.hpp>
#include <dolfinx_contact/utils.h>
#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(cpp, m)
{
  // Load basix and dolfinx to use Pybindings
  py::module basix = py::module::import("basix");

  // Create module for C++ wrappers
  m.doc() = "DOLFINX Contact Python interface";
#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  // Kernel wrapper class
  py::class_<contact_wrappers::KernelWrapper,
             std::shared_ptr<contact_wrappers::KernelWrapper>>(
      m, "KernelWrapper", "Wrapper for C++ contact integration kernels");

  // QuadratureRule
  py::class_<dolfinx_contact::QuadratureRule,
             std::shared_ptr<dolfinx_contact::QuadratureRule>>(
      m, "QuadratureRule", "QuadratureRule object")
      .def(py::init<dolfinx::mesh::CellType, int, int,
                    basix::quadrature::type>(),
           py::arg("cell_type"), py::arg("degree"), py::arg("dim"),
           py::arg("type") = basix::quadrature::type::Default)
      .def("points",
           [](dolfinx_contact::QuadratureRule& self) {
             return dolfinx_wrappers::xt_as_pyarray(std::move(self.points()));
           })
      .def("weights", [](dolfinx_contact::QuadratureRule& self)
           { return dolfinx_wrappers::as_pyarray(std::move(self.weights())); });

  // Contact
  py::class_<dolfinx_contact::Contact,
             std::shared_ptr<dolfinx_contact::Contact>>(m, "Contact",
                                                        "Contact object")
      .def(py::init<std::vector<
                        std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>>,
                    std::shared_ptr<
                        const dolfinx::graph::AdjacencyList<std::int32_t>>,
                    std::vector<std::array<int, 2>>,
                    std::shared_ptr<dolfinx::fem::FunctionSpace>, const int>(),
           py::arg("markers"), py::arg("sufaces"), py::arg("contact_pairs"),
           py::arg("V"), py::arg("quadrature_degree") = 3)
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
             int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
             return dolfinx_wrappers::as_pyarray(std::move(coeffs),
                                                 std::array{shape0, cstride});
           })
      .def("pack_gap",
           [](dolfinx_contact::Contact& self, int origin_meshtag)
           {
             auto [coeffs, cstride] = self.pack_gap(origin_meshtag);
             int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
             return dolfinx_wrappers::as_pyarray(std::move(coeffs),
                                                 std::array{shape0, cstride});
           })
      .def(
          "create_matrix",
          [](dolfinx_contact::Contact& self, dolfinx::fem::Form<PetscScalar>& a,
             std::string type) { return self.create_petsc_matrix(a, type); },
          py::return_value_policy::take_ownership, py::arg("a"),
          py::arg("type") = std::string(),
          "Create a PETSc Mat for two-sided contact.")
      .def("qp_phys",
           [](dolfinx_contact::Contact& self, int origin_meshtag, int facet)
           {
             auto qp = self.qp_phys(origin_meshtag)[facet];
             return dolfinx_wrappers::xt_as_pyarray(std::move(qp));
           })
      .def("active_entities",
           [](dolfinx_contact::Contact& self, int s)
           {
             const std::vector<std::int32_t>& active_entities
                 = self.active_entities(s);
             std::array<py::ssize_t, 2> shape
                 = {py::ssize_t(active_entities.size() / 2), 2};
             return py::array_t<std::int32_t>(shape, active_entities.data(),
                                              py::cast(self));
           })
      .def("facet_map",
           [](dolfinx_contact::Contact& self, int pair)
           {
             // This exposes facet_map() to python but replaces the
             // facet indices on the submesh with the facet indices in
             // the parent mesh This is only exposed for testing (in
             // particular
             // nitsche_rigid_surface.py/demo_nitsche_rigid_surface_ufl.py)
             auto contact_pair = self.contact_pair(pair);
             std::shared_ptr<const dolfinx::mesh::Mesh> mesh = self.mesh();
             const int tdim = mesh->topology().dim(); // topological dimension
             const int fdim = tdim - 1; // topological dimension of facet
             auto c_to_f = mesh->topology().connectivity(tdim, fdim);
             assert(c_to_f);
             std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>>
                 submesh_map = self.facet_map(pair);
             const std::vector<int>& offsets = submesh_map->offsets();
             const std::vector<std::int32_t>& old_data = submesh_map->array();
             std::shared_ptr<const dolfinx::graph::AdjacencyList<int>> facet_map
                 = self.submesh(contact_pair[1]).facet_map();
             const std::vector<std::int32_t>& parent_cells
                 = self.submesh(contact_pair[1]).parent_cells();
             std::vector<std::int32_t> data(old_data.size());
             for (std::size_t i = 0; i < old_data.size(); ++i)
             {
               auto facet_sub = old_data[i];
               auto facet_pair = facet_map->links(facet_sub);
               auto cell_parent = parent_cells[facet_pair[0]];
               data[i] = c_to_f->links(cell_parent)[facet_pair[1]];
             }
             return std::make_shared<
                 dolfinx::graph::AdjacencyList<std::int32_t>>(
                 std::move(data), std::move(offsets));
           })
      .def("coefficients_size", &dolfinx_contact::Contact::coefficients_size,
           py::arg("meshtie") = false)
      .def("set_quadrature_rule",
           &dolfinx_contact::Contact::set_quadrature_rule)
      .def("generate_kernel",
           [](dolfinx_contact::Contact& self, dolfinx_contact::Kernel type) {
             return contact_wrappers::KernelWrapper(self.generate_kernel(type));
           })

      .def("assemble_matrix",
           [](dolfinx_contact::Contact& self, Mat A,
              const std::vector<std::shared_ptr<
                  const dolfinx::fem::DirichletBC<PetscScalar>>>& bcs,
              int origin_meshtag, contact_wrappers::KernelWrapper& kernel,
              const py::array_t<PetscScalar, py::array::c_style>& coeffs,
              const py::array_t<PetscScalar, py::array::c_style>& constants)
           {
             auto ker = kernel.get();
             self.assemble_matrix(
                 dolfinx::la::petsc::Matrix::set_block_fn(A, ADD_VALUES), bcs,
                 origin_meshtag, ker,
                 std::span<const PetscScalar>(coeffs.data(), coeffs.size()),
                 coeffs.shape(1),
                 std::span(constants.data(), constants.shape(0)));
           })
      .def("assemble_vector",
           [](dolfinx_contact::Contact& self,
              py::array_t<PetscScalar, py::array::c_style>& b,
              int origin_meshtag, contact_wrappers::KernelWrapper& kernel,
              const py::array_t<PetscScalar, py::array::c_style>& coeffs,
              const py::array_t<PetscScalar, py::array::c_style>& constants)
           {
             auto ker = kernel.get();
             self.assemble_vector(
                 std::span(b.mutable_data(), b.shape(0)), origin_meshtag, ker,
                 std::span<const PetscScalar>(coeffs.data(), coeffs.size()),
                 coeffs.shape(1),
                 std::span(constants.data(), constants.shape(0)));
           })
      .def("pack_test_functions",
           [](dolfinx_contact::Contact& self, int origin_meshtag,
              const py::array_t<PetscScalar, py::array::c_style>& gap)
           {
             auto [coeffs, cstride] = self.pack_test_functions(
                 origin_meshtag,
                 std::span<const PetscScalar>(gap.data(), gap.size()));
             int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
             return dolfinx_wrappers::as_pyarray(std::move(coeffs),
                                                 std::array{shape0, cstride});
           })
      .def(
          "pack_grad_test_functions",
          [](dolfinx_contact::Contact& self, int origin_meshtag,
             const py::array_t<PetscScalar, py::array::c_style>& gap,
             const py::array_t<PetscScalar, py::array::c_style>& u_packed)
          {
            auto [coeffs, cstride] = self.pack_grad_test_functions(
                origin_meshtag,
                std::span<const PetscScalar>(gap.data(), gap.size()),
                std::span<const PetscScalar>(u_packed.data(), u_packed.size()));
            int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
            return dolfinx_wrappers::as_pyarray(std::move(coeffs),
                                                std::array{shape0, cstride});
          })
      .def("pack_ny",
           [](dolfinx_contact::Contact& self, int origin_meshtag,
              const py::array_t<PetscScalar, py::array::c_style>& gap)
           {
             auto [coeffs, cstride] = self.pack_ny(
                 origin_meshtag,
                 std::span<const PetscScalar>(gap.data(), gap.size()));
             int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
             return dolfinx_wrappers::as_pyarray(std::move(coeffs),
                                                 std::array{shape0, cstride});
           })
      .def(
          "pack_u_contact",
          [](dolfinx_contact::Contact& self, int origin_meshtag,
             std::shared_ptr<dolfinx::fem::Function<PetscScalar>> u,
             const py::array_t<PetscScalar, py::array::c_style>& gap)
          {
            auto [coeffs, cstride] = self.pack_u_contact(
                origin_meshtag, u,
                std::span<const PetscScalar>(gap.data(), gap.size()));
            int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
            return dolfinx_wrappers::as_pyarray(std::move(coeffs),
                                                std::array{shape0, cstride});
          },
          py::arg("origin_meshtag"), py::arg("u"), py::arg("gap"))
      .def(
          "pack_grad_u_contact",
          [](dolfinx_contact::Contact& self, int origin_meshtag,
             std::shared_ptr<dolfinx::fem::Function<PetscScalar>> u,
             const py::array_t<PetscScalar, py::array::c_style>& gap,
             const py::array_t<PetscScalar, py::array::c_style>& u_packed)
          {
            auto [coeffs, cstride] = self.pack_grad_u_contact(
                origin_meshtag, u,
                std::span<const PetscScalar>(gap.data(), gap.size()),
                std::span<const PetscScalar>(u_packed.data(), u_packed.size()));
            int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
            return dolfinx_wrappers::as_pyarray(std::move(coeffs),
                                                std::array{shape0, cstride});
          })
      .def("update_submesh_geometry",
           &dolfinx_contact::Contact::update_submesh_geometry);
  m.def(
      "generate_contact_kernel",
      [](std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
         dolfinx_contact::Kernel type, dolfinx_contact::QuadratureRule& q_rule,
         std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>>
             coeffs,
         bool constant_normal)
      {
        return contact_wrappers::KernelWrapper(
            dolfinx_contact::generate_contact_kernel<PetscScalar>(
                V, type, q_rule, coeffs, constant_normal));
      },
      py::arg("V"), py::arg("kernel_type"), py::arg("quadrature_rule"),
      py::arg("coeffs"), py::arg("constant_normal") = true);
  py::enum_<dolfinx_contact::Kernel>(m, "Kernel")
      .value("Rhs", dolfinx_contact::Kernel::Rhs)
      .value("Jac", dolfinx_contact::Kernel::Jac)
      .value("MeshTieRhs", dolfinx_contact::Kernel::MeshTieRhs)
      .value("MeshTieJac", dolfinx_contact::Kernel::MeshTieJac);
  m.def(
      "pack_coefficient_quadrature",
      [](std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff,
         int q, const py::array_t<std::int32_t, py::array::c_style>& entities)
      {
        auto e_span
            = std::span<const std::int32_t>(entities.data(), entities.size());
        if (entities.ndim() == 1)
        {

          auto [coeffs, cstride] = dolfinx_contact::pack_coefficient_quadrature(
              coeff, q, e_span, dolfinx::fem::IntegralType::cell);
          int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
          return dolfinx_wrappers::as_pyarray(std::move(coeffs),
                                              std::array{shape0, cstride});
        }
        else if (entities.ndim() == 2)
        {

          auto [coeffs, cstride] = dolfinx_contact::pack_coefficient_quadrature(
              coeff, q, e_span, dolfinx::fem::IntegralType::exterior_facet);
          int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
          return dolfinx_wrappers::as_pyarray(std::move(coeffs),
                                              std::array{shape0, cstride});
        }
        else
        {
          throw std::invalid_argument("Unsupported entities");
        }
      });

  m.def(
      "pack_circumradius",
      [](const dolfinx::mesh::Mesh& mesh,
         const py::array_t<std::int32_t, py::array::c_style>& active_facets)
      {
        auto e_span = std::span<const std::int32_t>(active_facets.data(),
                                                    active_facets.size());
        std::vector<double> coeffs
            = dolfinx_contact::pack_circumradius(mesh, e_span);
        return dolfinx_wrappers::as_pyarray(
            std::move(coeffs),
            std::array{std::size_t(active_facets.size() / 2), (std::size_t)1});
      });
  m.def("update_geometry", [](const dolfinx::fem::Function<PetscScalar>& u,
                              std::shared_ptr<dolfinx::mesh::Mesh> mesh)
        { dolfinx_contact::update_geometry(u, mesh); });

  m.def("compute_active_entities",
        [](std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
           py::array_t<std::int32_t, py::array::c_style>& entities,
           dolfinx::fem::IntegralType integral)
        {
          auto entity_span
              = std::span<const std::int32_t>(entities.data(), entities.size());
          std::vector<std::int32_t> active_entities
              = dolfinx_contact::compute_active_entities(mesh, entity_span,
                                                         integral);
          switch (integral)
          {
          case dolfinx::fem::IntegralType::cell:
          {
            py::array_t<std::int32_t> domains(active_entities.size(),
                                              active_entities.data());
            return domains;
          }
          case dolfinx::fem::IntegralType::exterior_facet:
          {
            std::array<py::ssize_t, 2> shape
                = {py::ssize_t(active_entities.size() / 2), 2};
            return dolfinx_wrappers::as_pyarray(std::move(active_entities),
                                                shape);
          }
          case dolfinx::fem::IntegralType::interior_facet:
          {
            std::array<py::ssize_t, 3> shape
                = {py::ssize_t(active_entities.size() / 4), 2, 2};
            return dolfinx_wrappers::as_pyarray(std::move(active_entities),
                                                shape);
          }
          default:
            throw std::invalid_argument("Unsupported integral type");
          }
        });

  m.def(
      "find_candidate_surface_segment",
      [](std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
         const std::vector<std::int32_t>& puppet_facets,
         const std::vector<std::int32_t>& candidate_facets, const double radius)
      {
        return dolfinx_contact::find_candidate_surface_segment(
            mesh, puppet_facets, candidate_facets, radius);
      },
      py::arg("mesh"), py::arg("puppet_facets"), py::arg("candidate_facets"),
      py::arg("radius") = -1.0);
  m.def(
      "raytracing",
      [](const dolfinx::mesh::Mesh& mesh,
         py::array_t<double, py::array::c_style>& point,
         py::array_t<double, py::array::c_style>& normal,
         py::array_t<std::int32_t, py::array::c_style>& cells, int max_iter,
         double tol)
      {
        auto facet_span
            = std::span<const std::int32_t>(cells.data(), cells.size());
        const std::size_t gdim = mesh.geometry().dim();
        std::array<std::size_t, 1> s_p = {(std::size_t)point.shape(0)};
        if (std::size_t(point.shape(0)) != gdim)
        {
          throw std::invalid_argument(
              "Input point has to have same dimension as gdim");
        }
        auto _point
            = xt::adapt(point.data(), point.size(), xt::no_ownership(), s_p);

        if (std::size_t(normal.shape(0)) != gdim)
        {
          throw std::invalid_argument(
              "Input normal has to have dimension gdim");
        }
        auto _normal
            = xt::adapt(normal.data(), normal.size(), xt::no_ownership(), s_p);
        std::tuple<int, std::int32_t, xt::xtensor<double, 1>,
                   xt::xtensor<double, 1>>
            output = dolfinx_contact::raytracing(mesh, _point, _normal,
                                                 facet_span, max_iter, tol);
        int status = std::get<0>(output);
        auto x = std::get<2>(output);
        auto X = std::get<3>(output);
        std::int32_t idx = std::get<1>(output);

        return py::make_tuple(status, idx,
                              dolfinx_wrappers::xt_as_pyarray(std::move(x)),
                              dolfinx_wrappers::xt_as_pyarray(std::move(X)));
      },
      py::arg("mesh"), py::arg("point"), py::arg("tangents"), py::arg("cells"),
      py::arg("max_iter") = 25, py::arg("tol") = 1e-8);
}
