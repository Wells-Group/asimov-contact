// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_CONTACT
//
// SPDX-License-Identifier:    MIT

#include "kernelwrapper.h"
#include <array.h>
#include <caster_petsc.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx_contact/Contact.hpp>
#include <dolfinx_contact/coefficients.h>
#include <dolfinx_contact/contact_kernels.hpp>
#include <dolfinx_contact/utils.h>
#include <dolfinx_cuas/kernelwrapper.h>
#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtensor/xio.hpp>
#include <xtl/xspan.hpp>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(cpp, m)
{
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
      m, "KernelWrapper", "Wrapper for C++ integration kernels");
  py::class_<dolfinx_contact::Contact,
             std::shared_ptr<dolfinx_contact::Contact>>(m, "Contact",
                                                        "Contact object")
      .def(py::init<std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>,
                    std::array<int, 2>,
                    std::shared_ptr<dolfinx::fem::FunctionSpace>>(),
           py::arg("marker"), py::arg("sufaces"), py::arg("V"))
      .def("create_distance_map",
           [](dolfinx_contact::Contact& self, int puppet_mt, int candidate_mt)
           {
             self.create_distance_map(puppet_mt, candidate_mt);
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
             std::string type) { return self.create_matrix(a, type); },
          py::return_value_policy::take_ownership, py::arg("a"),
          py::arg("type") = std::string(),
          "Create a PETSc Mat for two-sided contact.")
      .def("qp_phys",
           [](dolfinx_contact::Contact& self, int origin_meshtag, int facet)
           {
             auto qp = self.qp_phys(origin_meshtag)[facet];
             return dolfinx_wrappers::xt_as_pyarray(std::move(qp));
           })
      .def("facet_map", &dolfinx_contact::Contact::facet_map)
      .def("set_quadrature_degree",
           &dolfinx_contact::Contact::set_quadrature_degree)
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
                 xtl::span<const PetscScalar>(coeffs.data(), coeffs.size()),
                 coeffs.shape(1),
                 xtl::span(constants.data(), constants.shape(0)));
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
                 xtl::span(b.mutable_data(), b.shape(0)), origin_meshtag, ker,
                 xtl::span<const PetscScalar>(coeffs.data(), coeffs.size()),
                 coeffs.shape(1),
                 xtl::span(constants.data(), constants.shape(0)));
           })
      .def("pack_test_functions",
           [](dolfinx_contact::Contact& self, int origin_meshtag,
              const py::array_t<PetscScalar, py::array::c_style>& gap)
           {
             auto [coeffs, cstride] = self.pack_test_functions(
                 origin_meshtag,
                 xtl::span<const PetscScalar>(gap.data(), gap.size()));
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
                 xtl::span<const PetscScalar>(gap.data(), gap.size()));
             int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
             return dolfinx_wrappers::as_pyarray(std::move(coeffs),
                                                 std::array{shape0, cstride});
           })
      .def("pack_u_contact",
           [](dolfinx_contact::Contact& self, int origin_meshtag,
              std::shared_ptr<dolfinx::fem::Function<PetscScalar>> u,
              const py::array_t<PetscScalar, py::array::c_style>& gap)
           {
             auto [coeffs, cstride] = self.pack_u_contact(
                 origin_meshtag, u,
                 xtl::span<const PetscScalar>(gap.data(), gap.size()));
             int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
             return dolfinx_wrappers::as_pyarray(std::move(coeffs),
                                                 std::array{shape0, cstride});
           });

  m.def(
      "generate_contact_kernel",
      [](std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
         dolfinx_contact::Kernel type, dolfinx_cuas::QuadratureRule& q_rule,
         std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>>
             coeffs,
         bool constant_normal)
      {
        return cuas_wrappers::KernelWrapper<PetscScalar>(
            dolfinx_contact::generate_contact_kernel<PetscScalar>(
                V, type, q_rule, coeffs, constant_normal));
      },
      py::arg("V"), py::arg("kernel_type"), py::arg("quadrature_rule"),
      py::arg("coeffs"), py::arg("constant_normal") = true);
  py::enum_<dolfinx_contact::Kernel>(m, "Kernel")
      .value("Rhs", dolfinx_contact::Kernel::Rhs)
      .value("Jac", dolfinx_contact::Kernel::Jac);
  m.def("pack_coefficient_quadrature",
        [](std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff,
           int q, dolfinx::fem::IntegralType integral,
           const py::array_t<std::int32_t, py::array::c_style>& active_entities)
        {
          auto [coeffs, cstride] = dolfinx_contact::pack_coefficient_quadrature(
              coeff, q, integral,
              xtl::span<const std::int32_t>(active_entities.data(),
                                            active_entities.size()));
          int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
          return dolfinx_wrappers::as_pyarray(std::move(coeffs),
                                              std::array{shape0, cstride});
        });

  m.def("pack_circumradius",
        [](const dolfinx::mesh::Mesh& mesh,
           const py::array_t<std::int32_t, py::array::c_style>& active_facets)
        {
          assert(active_facets.ndim() == 2);
          const std::size_t shape_0 = active_facets.shape(0);
          auto ents = active_facets.unchecked();
          // FIXME: How to avoid copy here
          std::vector<std::pair<std::int32_t, int>> facets;
          facets.reserve(shape_0);
          for (std::size_t i = 0; i < shape_0; i++)
            facets.emplace_back(ents(i, 0), ents(i, 1));
          auto facet_span = xtl::span<const std::pair<std::int32_t, int>>(
              facets.data(), facets.size());
          auto [coeffs, cstride]
              = dolfinx_contact::pack_circumradius(mesh, facet_span);
          int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
          return dolfinx_wrappers::as_pyarray(std::move(coeffs),
                                              std::array{shape0, cstride});
        });
  m.def("update_geometry", [](const dolfinx::fem::Function<PetscScalar>& u,
                              std::shared_ptr<dolfinx::mesh::Mesh> mesh)
        { dolfinx_contact::update_geometry(u, mesh); });
}
