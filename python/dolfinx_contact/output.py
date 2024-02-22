# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT
import dolfinx
from matplotlib import pyplot as plt
from dolfinx.io import VTXWriter
from dolfinx.fem import form, Function, FunctionSpace
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.mesh import create_submesh
import numpy as np
import dolfinx_contact
import petsc4py.PETSc as PETSc
import ufl


class ContactWriter:
    __slots__ = ["vtx", "contact", "u", "contact_pairs", "material",
                 "projection_coordinates", "mesh", "facet_list", "facet_mesh",
                 "msh_to_fm", "pn", "pt", "a_form", "L", "L2", "p_f", "pt_f", "p_hertz",
                 "t_hertz"]

    def __init__(self, mesh, contact, u, contact_pairs,
                 material, order, simplex,
                 projection_coordinates, fname):
        self.contact = contact
        self.u = u
        self.contact_pairs = contact_pairs
        self.material = material
        self.projection_coordinates = projection_coordinates
        self.mesh = mesh

        tdim = mesh.topology.dim
        c_to_f = mesh.topology.connectivity(tdim, tdim - 1)
        facet_list = []
        for j in range(len(contact_pairs)):
            facet_list.append(np.zeros(len(contact.entities[j]), dtype=np.int32))
            for i, e in enumerate(contact.entities[j]):
                facet = c_to_f.links(e[0])[e[1]]
                facet_list[j][i] = facet
        self.facet_list = facet_list
        facets = np.unique(np.sort(np.hstack([facet_list[j] for j in range(len(contact_pairs))])))
        facet_mesh, fm_to_msh = create_submesh(mesh, tdim - 1, facets)[:2]

        self.facet_mesh = facet_mesh
        # Create msh to submsh entity map
        num_facets = mesh.topology.index_map(tdim - 1).size_local + \
            mesh.topology.index_map(tdim - 1).num_ghosts
        msh_to_fm = np.full(num_facets, -1)
        msh_to_fm[fm_to_msh] = np.arange(len(fm_to_msh))
        self.msh_to_fm = msh_to_fm

        # Use quadrature element
        if tdim == 2:
            quadr_element = ufl.FiniteElement("Quadrature", ufl.Cell(
                "interval", geometric_dimension=facet_mesh.geometry.dim),
                degree=contact.q_deg, quad_scheme="default")
        else:
            if simplex:
                quadr_element = ufl.FiniteElement("Quadrature", ufl.Cell(
                    "triangle", geometric_dimension=facet_mesh.geometry.dim),
                    contact.q_deg, quad_scheme="default")
            else:
                quadr_element = ufl.FiniteElement("Quadrature", ufl.Cell(
                    "quadrilateral", geometric_dimension=facet_mesh.geometry.dim),
                    contact.q_deg, quad_scheme="default")

        quadr_space = FunctionSpace(facet_mesh, quadr_element)
        interpolation_space = FunctionSpace(facet_mesh, ("DG", max(order - 1, 1)))

        self.pn = Function(quadr_space)
        self.pt = Function(quadr_space)
        u_f = ufl.TrialFunction(interpolation_space)
        v_f = ufl.TestFunction(interpolation_space)

        # Define forms for the projection
        dx_f = ufl.Measure("dx", domain=facet_mesh)
        self.a_form = form(ufl.inner(u_f, v_f) * dx_f)
        self.L = form(ufl.inner(self.pn, v_f) * dx_f)
        self.L2 = form(ufl.inner(self.pt, v_f) * dx_f)
        self.p_f = Function(interpolation_space)
        self.pt_f = Function(interpolation_space)
        self.p_hertz = Function(interpolation_space)
        self.t_hertz = Function(interpolation_space)
        self.p_f.name = "Computed: Normal"
        self.pt_f.name = "Computed: Tangential"
        self.p_hertz.name = "Hertz: Normal"
        self.t_hertz.name = "Hertz: Tangential"

        self.vtx = VTXWriter(self.facet_mesh.comm, f"{fname}_surface_forces.bp", [
                             self.p_f, self.pt_f, self.p_hertz, self.t_hertz], "bp4")

    def project(self):
        tdim = self.mesh.topology.dim
        for i in range(len(self.contact_pairs)):
            fgeom = dolfinx_contact.cpp.entities_to_geometry_dofs(self.facet_mesh._cpp_object, tdim - 1,
                                                                  np.array(self.msh_to_fm[self.facet_list[i]],
                                                                           dtype=np.int32))
            fgeom = fgeom.array
            vali = self.projection_coordinates[i][1]
            xi = self.projection_coordinates[i][0]
            self.facet_mesh.geometry.x[fgeom, xi] = vali

    def restore(self, geom):
        self.facet_mesh.geometry.x[:, :] = geom[:, :]

    def write(self, t, pressure_function, tangent_force):
        normals = []
        for i in range(len(self.contact_pairs)):
            if self.contact.search_method[i] == dolfinx_contact.cpp.ContactMode.Raytracing:
                n_contact = np.array(-self.contact.pack_nx(i))
            else:
                n_contact = np.array(self.contact.pack_ny(i))
            normals.append(n_contact)

        gdim = self.mesh.geometry.dim
        V = self.u.function_space
        dummy = Function(V)
        self.contact.update_submesh_geometry(dummy._cpp_object)
        forces = []
        sig_n = []
        for i in range(len(self.contact_pairs)):
            n_contact = normals[i]
            n_x = self.contact.pack_nx(i)
            grad_u = dolfinx_contact.cpp.pack_gradient_quadrature(
                self.u._cpp_object, self.contact.q_deg, self.contact.entities[i])
            num_facets = self.contact.entities[i].shape[0]
            num_q_points = n_x.shape[1] // gdim
            # this assumes mu, lmbda are constant for each body
            sign = np.array(dolfinx_contact.cpp.compute_contact_forces(
                grad_u, n_x, num_q_points, num_facets, gdim, self.material[i][0, 0], self.material[i][0, 1]))
            sign = sign.reshape(num_facets, num_q_points, gdim)
            n_contact = n_contact.reshape(num_facets, num_q_points, gdim)
            pn = np.sum(sign * n_contact, axis=2)
            pt = sign - np.multiply(n_contact, pn[:, :, np.newaxis])
            pt = np.sqrt(np.sum(pt * pt, axis=2)).reshape(-1)
            pn = pn.reshape(-1)
            sig_n.append(sign.reshape(-1, gdim))
            forces.append([pn, pt])

        num_q_points = np.int32(len(forces[0][0]) / len(self.contact.entities[0]))
        for j in range(len(self.contact_pairs)):
            dofs = np.array(np.hstack([range(self.msh_to_fm[self.facet_list[j]][i] * num_q_points,
                            num_q_points * (self.msh_to_fm[self.facet_list[j]][i] + 1))
                for i in range(len(self.contact.entities[j]))]))
            self.pn.x.array[dofs] = forces[j][0][:]
            self.pt.x.array[dofs] = forces[j][1][:]

        # Assemble matrix and vector
        A = assemble_matrix(self.a_form)
        A.assemble()
        b = assemble_vector(self.L)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b2 = assemble_vector(self.L2)
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Setup solver
        ksp = PETSc.KSP().create(self.facet_mesh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")

        # Compute projection
        ksp.solve(b, self.p_f.vector)
        self.p_f.x.scatter_forward()

        ksp.solve(b2, self.pt_f.vector)
        self.pt_f.x.scatter_forward()

        # interpolate exact pressure

        self.p_hertz.interpolate(pressure_function)
        self.t_hertz.interpolate(tangent_force)

        geom = self.facet_mesh.geometry.x.copy()
        self.project()
        self.vtx.write(t)
        self.restore(geom)


# Visualise the gap. For debugging. Works in 2D only
def plot_gap(mesh, contact, gaps, entities, num_pairs):
    gdim = mesh.geometry.dim
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh_geometry = mesh.geometry.x

    for i in range(num_pairs):
        facet_map = contact.facet_map(i)
        c_to_f = mesh.topology.connectivity(tdim, tdim - 1)
        num_facets = entities[i].shape[0]
        facet_origin = np.zeros(num_facets, dtype=np.int32)
        for j in range(num_facets):
            cell = entities[i][j, 0]
            f_index = entities[i][j, 1]
            facet_origin[j] = c_to_f.links(cell)[f_index]
        facets_opp = facet_map.array
        facets_opp = facets_opp[facets_opp >= 0]

        # Draw facets on opposite surface
        plt.figure(dpi=600)
        for facet in facets_opp:
            facet_geometry = dolfinx.cpp.mesh.entities_to_geometry(mesh._cpp_object, fdim, [facet], False)
            coords = mesh_geometry[facet_geometry][0]
            plt.plot(coords[:, 0], coords[:, 1], color="black")
        min_x = 1
        max_x = 0
        for j in range(num_facets):
            facet = facet_origin[j]
            facet_geometry = dolfinx.cpp.mesh.entities_to_geometry(mesh._cpp_object, fdim, [facet], False)
            coords = mesh_geometry[facet_geometry][0]
            plt.plot(coords[:, 0], coords[:, 1], color="black")
            qp = contact.qp_phys(i, j)
            num_qp = qp.shape[0]
            for q in range(num_qp):
                g = gaps[i][j, q * gdim:(q + 1) * gdim]
                x = [qp[q, 0], qp[q, 0] + g[0]]
                y = [qp[q, 1], qp[q, 1] + g[1]]
                max_x = max(x[0], x[1], max_x)
                min_x = min(x[0], x[1], min_x)
                plt.plot(x, y)
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.xlim(min_x, max_x)
        rank = mesh.comm.rank
        plt.savefig(f"gap_{i}_{rank}.png")
