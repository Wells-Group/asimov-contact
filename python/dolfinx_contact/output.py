# Copyright (C) 2023 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT
import dolfinx
from matplotlib import pyplot as plt
from dolfinx.io import XDMFFile
from dolfinx.fem import form, Function, FunctionSpace, assemble_scalar
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
from dolfinx.mesh import create_submesh
import numpy as np
import dolfinx_contact
import petsc4py.PETSc as PETSc
import ufl
from mpi4py import MPI


# write pressure on surface to file for visualisation
def write_pressure_xdmf(mesh, contact, u, du, contact_pairs, quadrature_degree,
                        search_method, entities, material, order, simplex,
                        pressure_function, projection_coordinates, fname):

    # Recover original geometry for pressure computation
    du.x.array[:] = 0
    contact.update_submesh_geometry(du._cpp_object)

    # Compute contact pressure on surfaces
    gdim = mesh.geometry.dim
    tdim = mesh.topology.dim
    forces = []
    sig_n = []
    for i in range(len(contact_pairs)):
        n_x = contact.pack_nx(i)
        grad_u = dolfinx_contact.cpp.pack_gradient_quadrature(
            u._cpp_object, quadrature_degree, entities[i])
        if search_method[i] == dolfinx_contact.cpp.ContactMode.Raytracing:
            n_contact = np.array(-contact.pack_nx(i))
        else:
            n_contact = np.array(contact.pack_ny(i))

        num_facets = entities[i].shape[0]
        num_q_points = n_x.shape[1] // gdim
        # this assumes mu, lmbda are constant for each body
        sign = np.array(dolfinx_contact.cpp.compute_contact_forces(
            grad_u, n_x, num_q_points, num_facets, gdim, material[i][0, 0], material[i][0, 1]))
        sign = sign.reshape(num_facets, num_q_points, gdim)
        n_contact = n_contact.reshape(num_facets, num_q_points, gdim)
        pn = np.sum(sign*n_contact, axis=2)
        pt = sign - np.multiply(n_contact, pn[:, :, np.newaxis])
        pt = np.sqrt(np.sum(pt*pt, axis=2)).reshape(-1)
        pn = pn.reshape(-1)
        sig_n.append(sign.reshape(-1, gdim))
        forces.append([pn, pt])

    c_to_f = mesh.topology.connectivity(tdim, tdim - 1)
    facet_list = []
    for j in range(len(contact_pairs)):
        facet_list.append(np.zeros(len(entities[j]), dtype=np.int32))
        for i, e in enumerate(entities[j]):
            facet = c_to_f.links(e[0])[e[1]]
            facet_list[j][i] = facet

    facets = np.unique(np.sort(np.hstack([facet_list[j] for j in range(len(contact_pairs))])))
    facet_mesh, fm_to_msh = create_submesh(mesh, tdim - 1, facets)[:2]

    # Create msh to submsh entity map
    num_facets = mesh.topology.index_map(tdim - 1).size_local + \
        mesh.topology.index_map(tdim - 1).num_ghosts
    msh_to_fm = np.full(num_facets, -1)
    msh_to_fm[fm_to_msh] = np.arange(len(fm_to_msh))

    # Use quadrature element
    if tdim == 2:
        Q_element = ufl.FiniteElement("Quadrature", ufl.Cell(
            "interval", geometric_dimension=facet_mesh.geometry.dim), degree=quadrature_degree, quad_scheme="default")
    else:
        if simplex:
            Q_element = ufl.FiniteElement("Quadrature", ufl.Cell(
                "triangle", geometric_dimension=facet_mesh.geometry.dim), quadrature_degree, quad_scheme="default")
        else:
            Q_element = ufl.FiniteElement("Quadrature", ufl.Cell(
                "quadrilateral", geometric_dimension=facet_mesh.geometry.dim), quadrature_degree, quad_scheme="default")

    Q = FunctionSpace(facet_mesh, Q_element)
    P = FunctionSpace(facet_mesh, ("DG", order - 1))
    P_exact = FunctionSpace(facet_mesh, ("DG", order + 1))
    num_q_points = np.int32(len(forces[0][0]) / len(entities[0]))
    pn = Function(Q)
    pt = Function(Q)
    sig_x = Function(Q)
    sig_y = Function(Q)
    for j in range(len(contact_pairs)):
        dofs = np.array(np.hstack([range(msh_to_fm[facet_list[j]][i] * num_q_points,
                        num_q_points * (msh_to_fm[facet_list[j]][i] + 1)) for i in range(len(entities[j]))]))
        pn.x.array[dofs] = forces[j][0][:]
        pt.x.array[dofs] = forces[j][1][:]
        if j == 0:
            sig_x.x.array[dofs] = sig_n[j][:, 0]
            sig_y.x.array[dofs] = sig_n[j][:, 0]
    u_f = ufl.TrialFunction(P)
    v_f = ufl.TestFunction(P)



    # Define forms for the projection
    dx_f = ufl.Measure("dx", domain=facet_mesh)
    a_form = form(ufl.inner(u_f, v_f) * dx_f)
    L = form(ufl.inner(pn, v_f) * dx_f)
    L2 = form(ufl.inner(pt, v_f) * dx_f)

    force_x = form(sig_x * dx_f)
    force_y = form(sig_y * dx_f)
    R_x = facet_mesh.comm.allreduce(assemble_scalar(force_x), op=MPI.SUM)
    R_y = facet_mesh.comm.allreduce(assemble_scalar(force_y), op=MPI.SUM)
    print("Rx/Ry", R_x/R_y)

    # Assemble matrix and vector
    A = assemble_matrix(a_form)
    A.assemble()
    b = assemble_vector(L)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b2 = assemble_vector(L2)
    b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Setup solver
    ksp = PETSc.KSP().create(facet_mesh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")

    # Compute projection
    p_f = Function(P)
    ksp.solve(b, p_f.vector)
    p_f.x.scatter_forward()
    pt_f = Function(P)
    ksp.solve(b2, pt_f.vector)
    pt_f.x.scatter_forward()

    # interpolate exact pressure
    p_hertz = Function(P_exact)
    p_hertz.interpolate(pressure_function)
    geom = facet_mesh.geometry.x.copy()

    def _project():
        for i in range(len(contact_pairs)):
            fgeom = dolfinx_contact.cpp.entities_to_geometry_dofs(facet_mesh._cpp_object, tdim - 1, 
                         np.array(msh_to_fm[facet_list[i]], dtype=np.int32))
            fgeom = fgeom.array
            vali = projection_coordinates[i][1]
            xi = projection_coordinates[i][0]
            facet_mesh.geometry.x[fgeom, xi] = vali

    def _restore():
        facet_mesh.geometry.x[:, :] = geom[:, :]

    with XDMFFile(facet_mesh.comm, f"{fname}_surface_pressure.xdmf", "w") as xdmf:
        _project()
        xdmf.write_mesh(facet_mesh)
        p_f.name = "pressure"
        _restore()
        xdmf.write_function(p_f)

    with XDMFFile(facet_mesh.comm, f"{fname}_tangent_force.xdmf", "w") as xdmf:
        _project()
        xdmf.write_mesh(facet_mesh)
        pt_f.name = "tangential"
        _restore()
        xdmf.write_function(pt_f)

    with XDMFFile(facet_mesh.comm, f"{fname}_hertz_pressure.xdmf", "w") as xdmf:
        _project()
        xdmf.write_mesh(facet_mesh)
        p_hertz.name = "analytical"
        _restore()
        xdmf.write_function(p_hertz)


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
