# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from typing import Tuple, Union

from dolfinx import common, fem, io, log
import numpy as np
import ufl
from dolfinx.cpp.graph import AdjacencyList_int32
import dolfinx.cpp as _cpp
from petsc4py import PETSc as _PETSc

import dolfinx_contact
import dolfinx_contact.cpp
from dolfinx_contact.helpers import (rigid_motions_nullspace_subdomains, sigma_func)

kt = dolfinx_contact.cpp.Kernel

__all__ = ["nitsche_pseudo_time"]


def setup_newton_solver(t, steps, contact, contact_pairs, markers, V, material, h_packed, bcs, F_custom,
                        J_custom, entities, quadrature_degree, u, du, kernel_rhs, kernel_jac, consts, newton_options, petsc_options):

    A = contact.create_matrix(J_custom)
    b = fem.petsc.create_vector(F_custom)
    mesh = V.mesh
    # Pack gap, normals and test functions on each surface
    gaps = []
    normals = []
    test_fns = []
    with common.Timer("~Contact: Pack gap, normals, testfunction"):
        for i in range(len(contact_pairs)):
            gaps.append(contact.pack_gap(i))
            normals.append(contact.pack_ny(i))
            test_fns.append(contact.pack_test_functions(i))

    # Concatenate all coeffs
    coeffs_const = []
    for i in range(len(contact_pairs)):
        coeffs_const.append(np.hstack([material[i], h_packed[i], gaps[i], normals[i], test_fns[i]]))
    tbcs = []
    for bc in bcs:
        d = bc[0] / steps
        tag = bc[1]
        if mesh.geometry.dim == 3:
            g = fem.Constant(mesh, _PETSc.ScalarType((d[0], d[1], d[2])))
        else:
            g = fem.Constant(mesh, _PETSc.ScalarType((d[0], d[1])))
        bdy_dofs = fem.locate_dofs_topological(V, mesh.topology.dim - 1, markers[1].find(tag))
        tbcs.append(fem.dirichletbc(g, bdy_dofs, V))

    grad_u = []
    with common.Timer("~~Contact: Pack u"):
        for i in range(len(contact_pairs)):
            grad_u.append(dolfinx_contact.cpp.pack_gradient_quadrature(
                u._cpp_object, quadrature_degree, entities[i]))

    @ common.timed("~Contact: Update coefficients")
    def compute_coefficients(x, coeffs):
        size_local = V.dofmap.index_map.size_local
        bs = V.dofmap.index_map_bs
        du.x.array[:size_local * bs] = x.array_r[:size_local * bs]
        du.x.scatter_forward()
        u_candidate = []
        with common.Timer("~~Contact: Pack u contact"):
            for i in range(len(contact_pairs)):
                u_candidate.append(contact.pack_u_contact(i, du._cpp_object))
        u_puppet = []
        grad_u_puppet = []
        with common.Timer("~~Contact: Pack u"):
            for i in range(len(contact_pairs)):
                u_puppet.append(dolfinx_contact.cpp.pack_coefficient_quadrature(
                    du._cpp_object, quadrature_degree, entities[i]))
                grad_u_puppet.append(dolfinx_contact.cpp.pack_gradient_quadrature(
                    du._cpp_object, quadrature_degree, entities[i]))
        for i in range(len(contact_pairs)):
            c_0 = np.hstack([coeffs_const[i], u_puppet[i], grad_u_puppet[i] + grad_u[i], u_candidate[i]])
            coeffs[i][:, :] = c_0[:, :]

    @ common.timed("~Contact: Assemble residual")
    def compute_residual(x, b, coeffs):
        b.zeroEntries()
        b.ghostUpdate(addv=_PETSc.InsertMode.INSERT, mode=_PETSc.ScatterMode.FORWARD)
        with common.Timer("~~Contact: Contact contributions (in assemble vector)"):
            for i in range(len(contact_pairs)):
                contact.assemble_vector(b, i, kernel_rhs, coeffs[i], consts)
        with common.Timer("~~Contact: Standard contributions (in assemble vector)"):
            fem.petsc.assemble_vector(b, F_custom)

        # Apply boundary condition
        if len(bcs) > 0:
            fem.petsc.apply_lifting(b, [J_custom], bcs=[tbcs], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=_PETSc.InsertMode.ADD, mode=_PETSc.ScatterMode.REVERSE)
        if len(bcs) > 0:
            fem.petsc.set_bc(b, tbcs, x, -1.0)

    @ common.timed("~Contact: Assemble matrix")
    def compute_jacobian_matrix(x, A, coeffs):
        A.zeroEntries()
        with common.Timer("~~Contact: Contact contributions (in assemble matrix)"):
            for i in range(len(contact_pairs)):
                contact.assemble_matrix(A, [], i, kernel_jac, coeffs[i], consts)
        with common.Timer("~~Contact: Standard contributions (in assemble matrix)"):
            fem.petsc.assemble_matrix(A, J_custom, bcs=tbcs)
        A.assemble()

    # coefficient arrays
    num_coeffs = contact.coefficients_size(False)
    coeffs = np.array([np.zeros((len(entities[i]), num_coeffs)) for i in range(len(contact_pairs))])
    newton_solver = dolfinx_contact.NewtonSolver(mesh.comm, A, b, coeffs)

    # Set matrix-vector computations
    newton_solver.set_residual(compute_residual)
    newton_solver.set_jacobian(compute_jacobian_matrix)
    newton_solver.set_coefficients(compute_coefficients)

    # Set rigid motion nullspace
    null_space = rigid_motions_nullspace_subdomains(V, markers[0], np.unique(markers[0].values))
    newton_solver.A.setNearNullSpace(null_space)

    # Set Newton solver options
    newton_solver.set_newton_options(newton_options)

    # Set Krylov solver options
    newton_solver.set_krylov_options(petsc_options)

    return newton_solver


def nitsche_pseudo_time(steps: int, lhs: ufl.Form, rhs: ufl.Form, u: fem.Function,
                        rhs_fns: list[Union[fem.Function, fem.Constant]], markers: list[_cpp.mesh.MeshTags_int32],
                        contact_data: Tuple[AdjacencyList_int32, list[Tuple[int, int]]],
                        bcs: list[fem.DirichletBCMetaClass],
                        problem_parameters: dict[str, np.float64],
                        quadrature_degree: int = 5, form_compiler_options: dict = None, jit_options: dict = None,
                        petsc_options: dict = None, newton_options: dict = None,
                        outfile: str = None) -> Tuple[fem.Function, int, int, float]:
    """
    Use custom kernel to compute the contact problem with two elastic bodies coming into contact.

    Parameters
    ==========
    F The residual without contact contributions
    J The Jacobian without contact contributions
    u The function to be solved for. Also serves as initial value.
    markers
        A list of meshtags. The first element must mark all separate objects in order to create the correct nullspace.
        The second element must contain the mesh_tags for all puppet surfaces,
        Dirichlet-surfaces and Neumann-surfaces
        All further elements may contain candidate_surfaces
    contact_data = (surfaces, contact_pairs), where
        surfaces: Adjacency list. Links of i are meshtag values for contact
                  surfaces in ith mesh_tag in mesh_tags
        contact_pairs: list of pairs (i, j) marking the ith surface as a puppet
                  surface and the jth surface as the corresponding candidate
                  surface
    problem_parameters
        Dictionary with lame parameters and Nitsche parameters.
        Valid (key, value) tuples are: ('gamma': float), ('theta', float), ('mu', float),
        (lambda, float),
        where theta can be -1, 0 or 1 for skew-symmetric, penalty like or symmetric
        enforcement of Nitsche conditions
    quadrature_degree
        The quadrature degree to use for the custom contact kernels
    form_compiler_options
        Parameters used in FFCX compilation of this form. Run `ffcx --help` at
        the commandline to see all available options. Takes priority over all
        other parameter values, except for `scalar_type` which is determined by
        DOLFINX.
    jit_options
        Parameters used in CFFI JIT compilation of C code generated by FFCX.
        See https://github.com/FEniCS/dolfinx/blob/main/python/dolfinx/jit.py
        for all available parameters. Takes priority over all other parameter values.
    petsc_options
        Parameters that is passed to the linear algebra backend
        PETSc. For available choices for the 'petsc_options' kwarg,
        see the `PETSc-documentation
        <https://petsc4py.readthedocs.io/en/stable/manual/ksp/>`
    newton_options
        Dictionary with Newton-solver options. Valid (key, item) tuples are:
        ("atol", float), ("rtol", float), ("convergence_criterion", "str"),
        ("max_it", int), ("error_on_nonconvergence", bool), ("relaxation_parameter", float)
    outfile
        File to append solver summary

    """
    form_compiler_options = {} if form_compiler_options is None else form_compiler_options
    jit_options = {} if jit_options is None else jit_options
    petsc_options = {} if petsc_options is None else petsc_options
    newton_options = {} if newton_options is None else newton_options

    if problem_parameters.get("mu") is None:
        raise RuntimeError("Need to supply lame paramters")
    else:
        mu = mu = problem_parameters.get("mu")

    if problem_parameters.get("lambda") is None:
        raise RuntimeError("Need to supply lame paramters")
    else:
        lmbda = problem_parameters.get("lambda")
    if problem_parameters.get("theta") is None:
        raise RuntimeError("Need to supply theta for Nitsche's method")
    else:
        theta = problem_parameters["theta"]
    if problem_parameters.get("gamma") is None:
        raise RuntimeError("Need to supply gamma for Nitsche's method")
    else:
        gamma = problem_parameters.get("gamma")
    sigma = sigma_func(mu, lmbda)

    # Contact data
    contact_pairs = contact_data[1]
    contact_surfaces = contact_data[0]

    # Mesh, function space and FEM functions
    V = u.function_space
    mesh = V.mesh
    v = lhs.arguments()[0]  # Test function
    w = ufl.TrialFunction(V)     # Trial function
    du = fem.Function(V)

    h = ufl.CellDiameter(mesh)
    n = ufl.FacetNormal(mesh)

    # Integration measure and ufl part of linear/bilinear form
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=markers[1])

    # ufl part of contact
    for contact_pair in contact_pairs:
        surface_value = int(contact_surfaces.links(0)[contact_pair[0]])
        lhs += - 0.5 * theta * h / gamma * ufl.inner(sigma(u) * n, sigma(v) * n) * \
            ds(surface_value)
    F = ufl.replace(lhs, {u: u + du})
    J = ufl.derivative(F, du, w)

    # create contact class
    with common.Timer("~Contact: Init"):
        contact = dolfinx_contact.cpp.Contact(markers[1:], contact_surfaces, contact_pairs,
                                              V._cpp_object, quadrature_degree=quadrature_degree)

    # pack constants
    consts = np.array([gamma, theta])

    # Pack material parameters mu and lambda on each contact surface
    with common.Timer("~Contact: Interpolate coeffs (mu, lmbda)"):
        V2 = fem.FunctionSpace(mesh, ("DG", 0))
        lmbda2 = fem.Function(V2)
        lmbda2.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))
        mu2 = fem.Function(V2)
        mu2.interpolate(lambda x: np.full((1, x.shape[1]), mu))

    entities = []
    with common.Timer("~Contact: Compute active entities"):
        for pair in contact_pairs:
            entities.append(contact.active_entities(pair[0]))

    material = []
    with common.Timer("~Contact: Pack coeffs (mu, lmbda"):
        for i in range(len(contact_pairs)):
            material.append(np.hstack([dolfinx_contact.cpp.pack_coefficient_quadrature(
                mu2._cpp_object, 0, entities[i]),
                dolfinx_contact.cpp.pack_coefficient_quadrature(
                lmbda2._cpp_object, 0, entities[i])]))

    # Pack celldiameter on each surface
    h_packed = []
    with common.Timer("~Contact: Compute and pack celldiameter"):
        surface_cells = np.unique(np.hstack([entities[i][:, 0] for i in range(len(contact_pairs))]))
        h_int = fem.Function(V2)
        expr = fem.Expression(h, V2.element.interpolation_points())
        h_int.interpolate(expr, surface_cells)
        for i in range(len(contact_pairs)):
            h_packed.append(dolfinx_contact.cpp.pack_coefficient_quadrature(
                h_int._cpp_object, 0, entities[i]))

    vtx = io.VTXWriter(mesh.comm, "pseudo_time_quads.bp", [u])
    submesh0 = contact.submesh(0)
    submesh1 = contact.submesh(1)
    vtx_mesh0 = io.VTXWriter(mesh.comm, "submesh_0.bp", submesh0)
    vtx_mesh1 = io.VTXWriter(mesh.comm, "submesh_1.bp", submesh1)
    vtx.write(0)
    with common.Timer("~Contact: Distance maps"):
        for i in range(len(contact_pairs)):
            contact.create_distance_map(i)
    for tt in range(steps):
        t = (tt + 1) / steps
        F_bc = ufl.replace(rhs, {fn: t * fn for fn in rhs_fns})
        F_custom = fem.form(F - F_bc, form_compiler_options=form_compiler_options, jit_options=jit_options)
        J_custom = fem.form(J, form_compiler_options=form_compiler_options, jit_options=jit_options)

        with common.Timer("~Contact: Generate Jacobian kernel"):
            kernel_jac = contact.generate_kernel(kt.Jac)
        with common.Timer("~Contact: Generate residual kernel"):
            kernel_rhs = contact.generate_kernel(kt.Rhs)
        newton_solver = setup_newton_solver(t, steps, contact, contact_pairs, markers, V, material, h_packed, bcs, F_custom,
                                            J_custom, entities, quadrature_degree, u, du, kernel_rhs, kernel_jac, consts, newton_options, petsc_options)
        dofs_global = V.dofmap.index_map_bs * V.dofmap.index_map.size_global
        log.set_log_level(log.LogLevel.OFF)
        # Solve non-linear problem
        timing_str = f"~Contact: {id(dofs_global)} Solve Nitsche"
        with common.Timer(timing_str):
            n, converged = newton_solver.solve(du)

        if not converged:
            print("Newton solver did not converge")
        du.x.scatter_forward()
        u.x.array[:] += du.x.array[:]
        u.x.scatter_forward()
        with common.Timer("~Contact: Distance maps"):
            contact.update_submesh_geometry(u._cpp_object)
            for i in range(len(contact_pairs)):
                contact.create_distance_map(i)
        du.x.array[:].fill(0)
        du.x.scatter_forward()
        print(f"writing out solution for time t = {t}")
        vtx.write(t)
        vtx_mesh0.write((tt) / steps)
        vtx_mesh1.write((tt) / steps)
    contact.update_submesh_geometry(u._cpp_object)
    vtx_mesh0.write(1)
    vtx_mesh1.write(1)
    vtx.close()
    vtx_mesh0.close()
    vtx_mesh1.close()
