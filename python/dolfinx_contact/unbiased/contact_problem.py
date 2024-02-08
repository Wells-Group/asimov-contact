# Copyright (C) 2024 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from enum import Enum
import numpy as np
from typing import Union
from dolfinx import common, cpp, fem

import dolfinx_contact
import dolfinx_contact.cpp

kt = dolfinx_contact.cpp.Kernel


class FrictionLaw(Enum):
    Frictionless = 1
    Coulomb = 2
    Tresca = 3


class ContactProblem(dolfinx_contact.cpp.Contact):
    __slots__ = ["_matrix_kernels", "_vector_kernels", "coeffs", "_consts", "q_deg",
                 "_num_pairs", "_cstrides", "entities", "_normals", "search_method",
                 "_grad_u", "_num_q_points"]

    def __init__(self, markers, surfaces, contact_pairs, mesh, quadrature_degree, search_method, search_radius=-1.0):

        # create contact class
        markers_cpp = [marker._cpp_object for marker in markers]
        with common.Timer("~Contact: Init"):
            super().__init__(markers_cpp, surfaces, contact_pairs,
                             mesh._cpp_object, quadrature_degree=quadrature_degree,
                             search_method=search_method)

        self.set_search_radius(search_radius)
        # Perform contact detection
        for j in range(len(contact_pairs)):
            self.create_distance_map(j)

        self.q_deg = quadrature_degree
        self._num_pairs = len(contact_pairs)

        # Retrieve active entities
        self.entities = []
        with common.Timer("~Contact: Compute active entities"):
            for pair in contact_pairs:
                self.entities.append(self.active_entities(pair[0]))

        self.search_method = search_method
        self.coeffs = None

    @common.timed("~Contact: Update coefficients")
    def update_contact_data(self, du):
        max_links = self.max_links()
        ndofs_cell = len(du.function_space.dofmap.cell_dofs(0))
        gdim = du.function_space.mesh.geometry.dim

        with common.Timer("~~Contact: Pack u"):
            for i in range(self._num_pairs):
                offset0 = 4 + self._num_q_points[i] * gdim * (2 + ndofs_cell * max_links)
                offset1 = offset0 + self._num_q_points[i] * gdim
                self.coeffs[i][:, offset0:offset1] = dolfinx_contact.cpp.pack_coefficient_quadrature(
                    du._cpp_object, self.q_deg, self.entities[i])[:, :]
                offset0 = offset1
                offset1 = offset0 + self._num_q_points[i] * gdim * gdim
                self.coeffs[i][:, offset0:offset1] = dolfinx_contact.cpp.pack_gradient_quadrature(
                    du._cpp_object, self.q_deg, self.entities[i])[:, :] + self._grad_u[i][:, :]
                offset0 = offset1
                offset1 = offset0 + self._num_q_points[i] * gdim
                self.coeffs[i][:, offset0:offset1] = self.pack_u_contact(i, du._cpp_object)[:, :]

    def generate_contact_data(self, friction_law: FrictionLaw, function_space: fem.FunctionSpaceBase,
                              coefficients: dict[str, Union[fem.Function, fem.Constant]],
                              gamma, theta):

        # generate kernels
        self._matrix_kernels = []
        with common.Timer("~Contact: Generate Jacobian kernel"):
            self._matrix_kernels.append(self.generate_kernel(
                kt.Jac, function_space._cpp_object))
            if friction_law == FrictionLaw.Coulomb:
                self._matrix_kernels.append(self.generate_kernel(
                    kt.CoulombJac, function_space._cpp_object))
            elif friction_law == FrictionLaw.Tresca:
                self._matrix_kernels.append(self.generate_kernel(
                    kt.TrescaJac, function_space._cpp_object))

        self._vector_kernels = []
        with common.Timer("~Contact: Generate residual kernel"):
            self._vector_kernels.append(self.generate_kernel(
                kt.Rhs, function_space._cpp_object))
            if friction_law == FrictionLaw.Coulomb:
                self._vector_kernels.append(self.generate_kernel(
                    kt.CoulombRhs, function_space._cpp_object))
            elif friction_law == FrictionLaw.Tresca:
                self._vector_kernels.append(self.generate_kernel(
                    kt.TrescaRhs, function_space._cpp_object))
        # pack constants
        self._consts = np.array([gamma, theta], dtype=np.float64)

        # Pack material parameters
        keys = []
        if coefficients.get("mu") is None:
            raise RuntimeError("Lame parameter mu missing.")
        else:
            keys.append("mu")
        if coefficients.get("lambda") is None:
            raise RuntimeError("Lame parameter lambda missing.")
        else:
            keys.append("lambda")
        if coefficients.get("fric") is not None:
            keys.append("fric")

        # coefficient arrays
        numcoeffs = self.coefficients_size(
            False, function_space._cpp_object)

        new_model = False
        if self.coeffs is None:
            new_model = True
            self.coeffs = [np.zeros((len(self.entities[i]), numcoeffs))
                           for i in range(self._num_pairs)]

        with common.Timer("~Contact: Pack coeffs (mu, lmbda, fric)"):
            for i in range(self._num_pairs):
                for j, key in enumerate(keys):
                    self.coeffs[i][:, j] = dolfinx_contact.cpp.pack_coefficient_quadrature(
                        coefficients[key]._cpp_object, 0, self.entities[i])[:, 0]

        mesh = function_space.mesh
        V2 = fem.FunctionSpace(mesh, ("DG", 0))
        tdim = mesh.topology.dim
        gdim = mesh.geometry.dim

        h = fem.Function(V2)
        ncells = mesh.topology.index_map(tdim).size_local
        h_vals = cpp.mesh.h(mesh._cpp_object, mesh.topology.dim,
                            np.arange(0, ncells, dtype=np.int32))
        h.x.array[:ncells] = h_vals[:]
        # Pack celldiameter on each surface
        with common.Timer("~Contact: Compute and pack celldiameter"):
            for i in range(self._num_pairs):
                self.coeffs[i][:, 3] = dolfinx_contact.cpp.pack_coefficient_quadrature(
                    h._cpp_object, 0, self.entities[i])[:, 0]

        for j in range(self._num_pairs):
            self.create_distance_map(j)

        # Pack gap, normals and test functions on each surface
        self._num_q_points = []
        max_links = self.max_links()
        ndofs_cell = len(function_space.dofmap.cell_dofs(0))
        with common.Timer("~Contact: Pack gap, normals, testfunction"):
            for i in range(self._num_pairs):
                if self.search_method[i] == dolfinx_contact.cpp.ContactMode.Raytracing:
                    normals = -self.pack_nx(i)
                else:
                    normals = self.pack_ny(i)
                self._num_q_points.append(normals.shape[1] // gdim)
                offset0 = 4
                offset1 = offset0 + self._num_q_points[i] * gdim
                self.coeffs[i][:, offset0:offset1] = self.pack_gap(i)
                offset0 = offset1
                offset1 = offset0 + self._num_q_points[i] * gdim
                self.coeffs[i][:, offset0:offset1] = normals[:, :]
                offset0 = offset1
                offset1 = offset0 + self._num_q_points[i] * gdim * max_links * ndofs_cell
                self.coeffs[i][:, offset0:offset1] = self.pack_test_functions(
                    i, function_space._cpp_object)
                if new_model:
                    self.coeffs[i][:, -normals.shape[1]:] = normals[:, :]

        # pack grad u
        # This is to track grad_u if several load steps are used
        # grad(u_total) = grad(u) + grad(du),
        # where u is the displacement to date and du the displacement update
        # if u is not provided, this is set to zero
        self._grad_u = [np.zeros((len(self.entities[i]), self._num_q_points[i] * gdim * gdim))
                        for i in range(self._num_pairs)]

        if coefficients.get("u") is not None:
            with common.Timer("~~Contact: Pack grad(u)"):
                for i in range(self._num_pairs):
                    self._grad_u[i][:, :] = dolfinx_contact.cpp.pack_gradient_quadrature(
                        coefficients["u"]._cpp_object, self.q_deg, self.entities[i])[:, :]

        if coefficients.get("du") is not None:
            self.update_contact_data(coefficients["du"])

    def update_contact_detection(self, u):
        self.update_submesh_geometry(u._cpp_object)
        for j in range(self._num_pairs):
            self.create_distance_map(j)

        max_links = self.max_links()
        ndofs_cell = len(u.function_space.dofmap.cell_dofs(0))
        gdim = super().mesh().geometry.dim
        num_pairs = self._num_pairs
        for i in range(num_pairs):
            offsetn = 4 + self._num_q_points[i] * gdim * (2 + ndofs_cell * max_links)\
                + self._num_q_points[i] * gdim * (2 + gdim)
            offset0 = 4 + self._num_q_points[i] * gdim
            offset1 = offset0 + self._num_q_points[i] * gdim
            self.coeffs[i][:, offsetn:] = self.coeffs[i][:, offset0:offset1]

        # Pack gap, normals and test functions on each surface
        with common.Timer("~Contact: Pack gap, normals, testfunction"):
            for i in range(num_pairs):
                offset0 = 4
                offset1 = offset0 + self._num_q_points[i] * gdim
                self.coeffs[i][:, offset0:offset1] = self.pack_gap(i)
                offset0 = offset1
                offset1 = offset0 + self._num_q_points[i] * gdim
                if self.search_method[i] == dolfinx_contact.cpp.ContactMode.Raytracing:
                    self.coeffs[i][:, offset0:offset1] = -self.pack_nx(i)[:, :]
                else:
                    self.coeffs[i][:, offset0:offset1] = self.pack_ny(i)[:, :]
                offset0 = offset1
                offset1 = offset0 + self._num_q_points[i] * gdim * max_links * ndofs_cell
                self.coeffs[i][:, offset0:offset1] = self.pack_test_functions(
                    i, u.function_space._cpp_object)

        # pack grad u
        self._grad_u = []
        with common.Timer("~~Contact: Pack grad(u)"):
            for i in range(num_pairs):
                self._grad_u.append(dolfinx_contact.cpp.pack_gradient_quadrature(
                    u._cpp_object, self.q_deg, self.entities[i]))

    def update_nitsche_parameters(self, gamma, theta):
        self._consts[0] = gamma
        self._consts[1] = theta

    def h_surfaces(self):
        h = []
        for i in range(len(self.coeffs)):
            h.append(np.sum(self.coeffs[i][:, 3]) / self.coeffs[i].shape[0])
        return h

    def create_matrix(self, J):
        return super().create_matrix(J._cpp_object)

    # vector assembly

    def assemble_vector(self, b, function_space):
        for i in range(self._num_pairs):
            for kernel in self._vector_kernels:
                super().assemble_vector(
                    b, i, kernel, self.coeffs[i], self._consts, function_space._cpp_object)

    # matrix assembly

    @common.timed("~Contact: Assemble matrix")
    def assemble_matrix(self, A, function_space):
        for i in range(self._num_pairs):
            for kernel in self._matrix_kernels:
                super().assemble_matrix(
                    A, i, kernel, self.coeffs[i], self._consts, function_space._cpp_object)

    def crop_invalid_points(self, tol):
        gdim = super().mesh().geometry.dim
        for i in range(self._num_pairs):
            cstride = self._num_q_points[i] * gdim
            super().crop_invalid_points(i, self.coeffs[i][:, 4:4 + cstride],
                                        self.coeffs[i][:, 4 + cstride:4 + 2 * cstride], tol)
