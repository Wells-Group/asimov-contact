// Copyright (C) 2021 Sarah Roggendorf
//
// This file is part of DOLFINx_CUAS
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "array.h"
#include "caster_petsc.h"
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx_contact/Contact.hpp>
#include <dolfinx_contact/NewContact.hpp>
#include <dolfinx_contact/contact_kernels.hpp>
#include <dolfinx_contact/utils.hpp>
#include <dolfinx_cuas/kernelwrapper.h>
#include <iostream>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <xtl/xspan.hpp>
#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(cpp, m)
{
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
      .def("map_0_to_1", &dolfinx_contact::Contact::map_0_to_1)
      .def("map_1_to_0", &dolfinx_contact::Contact::map_1_to_0)
      .def("facet_0", &dolfinx_contact::Contact::facet_0)
      .def("facet_1", &dolfinx_contact::Contact::facet_1)
      .def("set_quadrature_degree",
           &dolfinx_contact::Contact::set_quadrature_degree);
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
      .value("NitscheRigidSurfaceRhs",
             dolfinx_contact::Kernel::NitscheRigidSurfaceRhs)
      .value("NitscheRigidSurfaceJac",
             dolfinx_contact::Kernel::NitscheRigidSurfaceJac);
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
  py::class_<dolfinx_contact::ContactInterface,
             std::shared_ptr<dolfinx_contact::ContactInterface>>(
      m, "ContactInterface", "Contact object")
      .def(py::init<std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>>, int,
                    int>(),
           py::arg("marker"), py::arg("surface_0"), py::arg("surface_1"))
      .def_property_readonly("surface_0",
                             &dolfinx_contact::ContactInterface::surface_0)

      .def_property_readonly("surface_1",
                             &dolfinx_contact::ContactInterface::surface_1)
      .def("create_cell_map",
           &dolfinx_contact::ContactInterface::create_cell_map);
}