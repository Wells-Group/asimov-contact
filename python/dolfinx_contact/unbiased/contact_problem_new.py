from dolfinx import common, cpp, fem
import numpy as np
from petsc4py import PETSc as _PETSc

import dolfinx_contact
import dolfinx_contact.cpp

kt = dolfinx_contact.cpp.Kernel


class ContactProblem:
    __slots__ = ["F", "J", "bcs", "u", "du", "contact", "markers", "entities",
                 "quadrature_degree", "const_coeffs", "coeffs", "consts", "search_method", "coulomb", "normals",
                 "kernel_jac", "kernel_friction_jac", "kernel_rhs", "kernel_friction_rhs",
                 "contact_pairs", "function_space", "grad_u", "u_old", "u_old_opp"]

    def __init__(self, markers, surfaces, contact_pairs, V, search_method, quadrature_degree,
                 search_radius=-1.0, coulomb=False):

        # create contact class
        markers_cpp = [marker._cpp_object for marker in markers]
        mesh = V.mesh
        with common.Timer("~Contact: Init"):
            self.contact = dolfinx_contact.cpp.Contact(markers_cpp, surfaces, contact_pairs,
                                                       mesh._cpp_object, quadrature_degree=quadrature_degree,
                                                       search_method=search_method)

        self.contact.set_search_radius(search_radius)
        # Perform contact detection
        for j in range(len(contact_pairs)):
            self.contact.create_distance_map(j)

        # generate kernels
        with common.Timer("~Contact: Generate Jacobian kernel"):
            self.kernel_jac = self.contact.generate_kernel(
                kt.Jac, V._cpp_object)
            if coulomb:
                self.kernel_friction_jac = self.contact.generate_kernel(
                    kt.CoulombJac, V._cpp_object)
            else:
                self.kernel_friction_jac = self.contact.generate_kernel(
                    kt.TrescaJac, V._cpp_object)
        with common.Timer("~Contact: Generate residual kernel"):
            self.kernel_rhs = self.contact.generate_kernel(
                kt.Rhs, V._cpp_object)
            if coulomb:
                self.kernel_friction_rhs = self.contact.generate_kernel(
                    kt.CoulombRhs, V._cpp_object)
            else:
                self.kernel_friction_rhs = self.contact.generate_kernel(
                    kt.TrescaRhs, V._cpp_object)

        self.markers = markers
        self.search_method = search_method
        self.contact_pairs = contact_pairs
        self.function_space = V
        self.quadrature_degree = quadrature_degree

    @common.timed("~Contact: Update coefficients")
    def update_kernel_data(self, du):
        num_pairs = len(self.contact_pairs)
        u_candidate = []
        with common.Timer("~~Contact: Pack u contact"):
            for i in range(num_pairs):
                u_candidate.append(
                    self.contact.pack_u_contact(i, du._cpp_object))
        u_puppet = []
        grad_u_puppet = []
        with common.Timer("~~Contact: Pack u"):
            for i in range(num_pairs):
                u_puppet.append(dolfinx_contact.cpp.pack_coefficient_quadrature(
                    du._cpp_object, self.quadrature_degree, self.entities[i]))
                grad_u_puppet.append(dolfinx_contact.cpp.pack_gradient_quadrature(
                    du._cpp_object, self.quadrature_degree, self.entities[i]))
        for i in range(num_pairs):
            c_0 = np.hstack([self.const_coeffs[i], u_puppet[i], grad_u_puppet[i]
                            + self.grad_u[i], u_candidate[i], self.normals[i], self.u_old_opp[i]])
            self.coeffs[i][:, :] = c_0[:, :]

    def generate_kernel_data(self, u, du, mu, lmbda, fric_coeff, gamma, theta):

        # pack constants
        self.consts = np.array([gamma, theta], dtype=np.float64)

        # Retrieve active entities
        self.entities = []
        with common.Timer("~Contact: Compute active entities"):
            for pair in self.contact_pairs:
                self.entities.append(self.contact.active_entities(pair[0]))

        # Pack material parameters
        material = []
        with common.Timer("~Contact: Pack coeffs (mu, lmbda"):
            for i in range(len(self.contact_pairs)):
                material.append(np.hstack([dolfinx_contact.cpp.pack_coefficient_quadrature(
                    mu._cpp_object, 0, self.entities[i]),
                    dolfinx_contact.cpp.pack_coefficient_quadrature(
                    lmbda._cpp_object, 0, self.entities[i]),
                    dolfinx_contact.cpp.pack_coefficient_quadrature(
                    fric_coeff._cpp_object, 0, self.entities[i])]))

        mesh = self.function_space.mesh
        V2 = fem.FunctionSpace(mesh, ("DG", 0))
        tdim = mesh.topology.dim

        h = fem.Function(V2)
        ncells = mesh.topology.index_map(tdim).size_local
        h_vals = cpp.mesh.h(mesh._cpp_object, mesh.topology.dim,
                            np.arange(0, ncells, dtype=np.int32))
        h.x.array[:ncells] = h_vals[:]
        # Pack celldiameter on each surface
        h_packed = []
        with common.Timer("~Contact: Compute and pack celldiameter"):
            for i in range(len(self.contact_pairs)):
                h_packed.append(dolfinx_contact.cpp.pack_coefficient_quadrature(
                    h._cpp_object, 0, self.entities[i]))

        for j in range(len(self.contact_pairs)):
            self.contact.create_distance_map(j)

        # Pack gap, normals and test functions on each surface
        gaps = []
        self.normals = []
        test_fns = []
        num_pairs = len(self.contact_pairs)
        with common.Timer("~Contact: Pack gap, normals, testfunction"):
            for i in range(num_pairs):
                gaps.append(self. contact.pack_gap(i))
                if self.search_method[i] == dolfinx_contact.cpp.ContactMode.Raytracing:
                    self.normals.append(-self.contact.pack_nx(i))
                else:
                    self.normals.append(self.contact.pack_ny(i))
                test_fns.append(self.contact.pack_test_functions(
                    i, self.function_space._cpp_object))

        # pack grad u
        self.grad_u = []
        self.u_old = []
        self.u_old_opp = []
        with common.Timer("~~Contact: Pack grad(u)"):
            for i in range(num_pairs):
                self.grad_u.append(dolfinx_contact.cpp.pack_gradient_quadrature(
                    u._cpp_object, self.quadrature_degree, self.entities[i]))
                self.u_old.append(dolfinx_contact.cpp.pack_coefficient_quadrature(
                    u._cpp_object, self.quadrature_degree, self.entities[i]))
                self.u_old_opp.append(
                    self.contact.pack_u_contact(i, u._cpp_object))

        # Concatenate material parameters, he4
        self.const_coeffs = []
        for i in range(len(self.contact_pairs)):
            self.const_coeffs.append(
                np.hstack([material[i], h_packed[i], gaps[i], self.normals[i], test_fns[i]]))

        # coefficient arrays
        num_coeffs = self.contact.coefficients_size(
            False, self.function_space._cpp_object)

        self.coeffs = [np.zeros((len(self.entities[i]), num_coeffs))
                       for i in range(num_pairs)]

        self.update_kernel_data(du)

    def update_contact_detection(self, u):
        self.contact.update_submesh_geometry(u._cpp_object)
        for j in range(len(self.contact_pairs)):
            self.contact.create_distance_map(j)

        num_pairs = len(self.contact_pairs)
        for i in range(num_pairs):
            self.normals[i] = self.const_coeffs[i][:,
                                                   7:7 + self.normals[i].shape[1]]

        # Pack gap, normals and test functions on each surface
        gaps = []
        normals = []
        test_fns = []
        with common.Timer("~Contact: Pack gap, normals, testfunction"):
            for i in range(num_pairs):
                gaps.append(self. contact.pack_gap(i))
                if self.search_method[i] == dolfinx_contact.cpp.ContactMode.Raytracing:
                    normals.append(-self.contact.pack_nx(i))
                else:
                    normals.append(self.contact.pack_ny(i))
                test_fns.append(self.contact.pack_test_functions(
                    i, self.function_space._cpp_object))

        # Concatenate all coeffs
        for i in range(num_pairs):
            c_0 = np.hstack([gaps[i], normals[i], test_fns[i]])
            self.const_coeffs[i][:, 4:] = c_0[:, :]

        # pack grad u
        self.grad_u = []
        self.u_old = []
        self.u_old_opp = []
        with common.Timer("~~Contact: Pack grad(u)"):
            for i in range(num_pairs):
                self.grad_u.append(dolfinx_contact.cpp.pack_gradient_quadrature(
                    u._cpp_object, self.quadrature_degree, self.entities[i]))
                self.u_old.append(dolfinx_contact.cpp.pack_coefficient_quadrature(
                    u._cpp_object, self.quadrature_degree, self.entities[i]))
                self.u_old_opp.append(
                    self.contact.pack_u_contact(i, u._cpp_object))

    def set_normals(self):
        normals = []
        for i in range(len(self.normals)):
            if self.search_method[i] == dolfinx_contact.cpp.ContactMode.Raytracing:
                normals.append(-self.contact.pack_nx(i))
            else:
                normals.append(self.contact.pack_ny(i))
        self.normals = normals

    def update_friction_coefficient(self, s):
        mesh = self.u.function_space.mesh
        V2 = fem.FunctionSpace(mesh, ("DG", 0))
        # interpolate friction coefficient
        fric_coeff = fem.Function(V2)
        fric_coeff.interpolate(lambda x: np.full((1, x.shape[1]), s))
        for i in range(len(self.coeffs)):
            fric = dolfinx_contact.cpp.pack_coefficient_quadrature(
                fric_coeff._cpp_object, 0, self.entities[i])
            self.coeffs[i][:, 2] = fric[:, 0]

    def update_nitsche_parameters(self, gamma, theta):
        self.consts[0] = gamma
        self.consts[1] = theta

    def h_surfaces(self):
        h = []
        for i in range(len(self.coeffs)):
            h.append(np.sum(self.coeffs[i][:, 3]) / self.coeffs[i].shape[0])
        return h

    def create_matrix(self):
        return self.contact.create_matrix(self.J._cpp_object)

    def set_forms(self, F, J, bcs):
        self.F = F
        self.J = J
        self.bcs = bcs

    # function for computing residual
    @common.timed("~Contact: Assemble residual")
    def compute_residual(self, x, b):
        b.zeroEntries()
        b.ghostUpdate(addv=_PETSc.InsertMode.INSERT,
                      mode=_PETSc.ScatterMode.FORWARD)
        with common.Timer("~~Contact: Contact contributions (in assemble vector)"):
            for i in range(len(self.contact_pairs)):
                self.contact.assemble_vector(
                    b, i, self.kernel_rhs, self.coeffs[i], self.consts, self.function_space._cpp_object)
                self.contact.assemble_vector(
                    b, i, self.kernel_friction_rhs, self.coeffs[i], self.consts, self.function_space._cpp_object)
        with common.Timer("~~Contact: Standard contributions (in assemble vector)"):
            fem.petsc.assemble_vector(b, self.F)

        # Apply boundary condition
        if len(self.bcs) > 0:
            fem.petsc.apply_lifting(
                b, [self.J], bcs=[self.bcs], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=_PETSc.InsertMode.ADD,
                      mode=_PETSc.ScatterMode.REVERSE)
        if len(self.bcs) > 0:
            fem.petsc.set_bc(b, self.bcs, x, -1.0)

    # function for computing jacobian
    @common.timed("~Contact: Assemble matrix")
    def compute_jacobian_matrix(self, x, A):
        A.zeroEntries()
        with common.Timer("~~Contact: Contact contributions (in assemble matrix)"):
            for i in range(len(self.contact_pairs)):
                self.contact.assemble_matrix(
                    A, i, self.kernel_jac, self.coeffs[i], self.consts, self.function_space._cpp_object)
                self.contact.assemble_matrix(
                    A, i, self.kernel_friction_jac, self.coeffs[i], self.consts, self.function_space._cpp_object)
        with common.Timer("~~Contact: Standard contributions (in assemble matrix)"):
            fem.petsc.assemble_matrix(A, self.J, bcs=self.bcs)
        A.assemble()
