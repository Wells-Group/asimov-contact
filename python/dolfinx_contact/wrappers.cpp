// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_petsc.h"
#include "kernelwrapper.h"
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx_contact/Contact.hpp>
#include <dolfinx_contact/contact_kernels.hpp>
#include <dolfinx_contact/utils.hpp>
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
      .def(py::init<std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>, int,
                    int, std::shared_ptr<dolfinx::fem::FunctionSpace>>(),
           py::arg("marker"), py::arg("suface_0"), py::arg("surface_1"),
           py::arg("V"))
      .def("create_distance_map",
           [](dolfinx_contact::Contact& self, int origin_meshtag)
           {
             self.create_distance_map(origin_meshtag);
             return;
           })
      .def("pack_gap_plane",
           [](dolfinx_contact::Contact& self, int origin_meshtag, double g)
           {
             auto [coeffs, cstride] = self.pack_gap_plane(origin_meshtag, g);
             int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
             return dolfinx_contact_wrappers::as_pyarray(
                 std::move(coeffs), std::array{shape0, cstride});
           })
      .def("pack_gap",
           [](dolfinx_contact::Contact& self, int origin_meshtag)
           {
             auto [coeffs, cstride] = self.pack_gap(origin_meshtag);
             int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
             return dolfinx_contact_wrappers::as_pyarray(
                 std::move(coeffs), std::array{shape0, cstride});
           })
      .def(
          "create_matrix",
          [](dolfinx_contact::Contact& self, dolfinx::fem::Form<PetscScalar>& a,
             std::string type) { return self.create_matrix(a, type); },
          py::return_value_policy::take_ownership, py::arg("a"),
          py::arg("type") = std::string(),
          "Create a PETSc Mat for two-sided contact.")
      .def("facet_0", &dolfinx_contact::Contact::facet_0)
      .def("facet_1", &dolfinx_contact::Contact::facet_1)
      .def("qp_phys",
           [](dolfinx_contact::Contact& self, int origin_meshtag, int facet)
           {
             if (origin_meshtag == 0)
             {

               auto qp = self.qp_phys_0()[facet];
               return dolfinx_contact_wrappers::xt_as_pyarray(std::move(qp));
             }
             else
             {
               auto qp = self.qp_phys_1()[facet];
               return dolfinx_contact_wrappers::xt_as_pyarray(std::move(qp));
             }
           })
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
                 dolfinx::la::PETScMatrix::set_block_fn(A, ADD_VALUES), bcs,
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
             return dolfinx_contact_wrappers::as_pyarray(
                 std::move(coeffs), std::array{shape0, cstride});
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
             return dolfinx_contact_wrappers::as_pyarray(
                 std::move(coeffs), std::array{shape0, cstride});
           })
      .def("pack_coefficient_dofs",
           [](dolfinx_contact::Contact& self, int origin_meshtag,
              std::shared_ptr<dolfinx::fem::Function<PetscScalar>> coeff)
           {
             auto [coeffs, cstride]
                 = self.pack_coefficient_dofs(origin_meshtag, coeff);
             int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
             return dolfinx_contact_wrappers::as_pyarray(
                 std::move(coeffs), std::array{shape0, cstride});
           })
      .def("pack_coeffs_const",
           [](dolfinx_contact::Contact& self, int origin_meshtag,
              std::shared_ptr<dolfinx::fem::Function<PetscScalar>> mu,
              std::shared_ptr<dolfinx::fem::Function<PetscScalar>> lmbda)
           {
             auto [coeffs, cstride]
                 = self.pack_coeffs_const(origin_meshtag, mu, lmbda);
             int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
             return dolfinx_contact_wrappers::as_pyarray(
                 std::move(coeffs), std::array{shape0, cstride});
           });

  m.def(
      "generate_contact_kernel",
      [](std::shared_ptr<const dolfinx::fem::FunctionSpace> V,
         dolfinx_contact::Kernel type, dolfinx_cuas::QuadratureRule& q_rule,
         std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>>
             coeffs,
         bool constant_normal)
      {
        return cuas_wrappers::KernelWrapper(
            dolfinx_contact::generate_contact_kernel(V, type, q_rule, coeffs,
                                                     constant_normal));
      },
      py::arg("V"), py::arg("kernel_type"), py::arg("quadrature_rule"),
      py::arg("coeffs"), py::arg("constant_normal") = true);
  py::enum_<dolfinx_contact::Kernel>(m, "Kernel")
      .value("Rhs", dolfinx_contact::Kernel::Rhs)
      .value("Jac", dolfinx_contact::Kernel::Jac);
  m.def("pack_coefficient_quadrature",
        [](std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff,
           int q)
        {
          auto [coeffs, cstride]
              = dolfinx_contact::pack_coefficient_quadrature(coeff, q);
          int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
          return dolfinx_contact_wrappers::as_pyarray(
              std::move(coeffs), std::array{shape0, cstride});
        });
  m.def("pack_coefficient_facet",
        [](std::shared_ptr<const dolfinx::fem::Function<PetscScalar>> coeff,
           int q,
           const py::array_t<std::int32_t, py::array::c_style>& active_facets)
        {
          auto [coeffs, cstride] = dolfinx_contact::pack_coefficient_facet(
              coeff, q,
              xtl::span<const std::int32_t>(active_facets.data(),
                                            active_facets.size()));
          int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
          return dolfinx_contact_wrappers::as_pyarray(
              std::move(coeffs), std::array{shape0, cstride});
        });

  m.def("pack_circumradius_facet",
        [](std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
           const py::array_t<std::int32_t, py::array::c_style>& active_facets)
        {
          auto [coeffs, cstride] = dolfinx_contact::pack_circumradius_facet(
              mesh, xtl::span<const std::int32_t>(active_facets.data(),
                                                  active_facets.size()));
          int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
          return dolfinx_contact_wrappers::as_pyarray(
              std::move(coeffs), std::array{shape0, cstride});
        });
  m.def("facet_to_cell_data",
        [](std::shared_ptr<const dolfinx::mesh::Mesh> mesh,
           const py::array_t<std::int32_t, py::array::c_style>& active_facets,
           const py::array_t<PetscScalar, py::array::c_style>& data,
           int num_cols)
        {
          auto [coeffs, cstride] = dolfinx_contact::facet_to_cell_data(
              mesh,
              xtl::span<const std::int32_t>(active_facets.data(),
                                            active_facets.size()),
              xtl::span<const PetscScalar>(data.data(), data.size()), num_cols);
          int shape0 = cstride == 0 ? 0 : coeffs.size() / cstride;
          return dolfinx_contact_wrappers::as_pyarray(
              std::move(coeffs), std::array{shape0, cstride});
        });
}
