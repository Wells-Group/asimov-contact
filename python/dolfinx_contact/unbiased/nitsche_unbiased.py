# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from typing import Optional, Tuple, Union

from dolfinx import common, fem, mesh, io, log
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx.cpp.graph import AdjacencyList_int32
import dolfinx.cpp as _cpp
from petsc4py import PETSc as _PETSc

import dolfinx_contact
import dolfinx_contact.cpp
from dolfinx_contact.helpers import (rigid_motions_nullspace_subdomains, sigma_func)

kt = dolfinx_contact.cpp.Kernel

__all__ = ["nitsche_unbiased"]


def setup_newton_solver(F_custom: fem.forms.FormMetaClass, J_custom: fem.forms.FormMetaClass,
                        bcs: Tuple[npt.NDArray[np.int32], list[Union[fem.Function, fem.Constant]]],
                        u: fem.Function, du: fem.Function,
                        contact: dolfinx_contact.cpp.Contact, markers: list[_cpp.mesh.MeshTags_int32],
                        entities: list[npt.NDArray[np.int32]], quadrature_degree: int,
                        const_coeffs: list[npt.NDArray[np.float64]], consts: npt.NDArray[np.float64]):
    """
    Set up newton solver for contact problem.
    Generate kernels and define functions for updating coefficients, stiffness matrix and residual vector.

    Parameters
    ==========
    incr               The load increment in [0,1]
    F_custom           The compiled form for the residual vector
    J_custom           The compiled form for the jacobian
    u                  The displacement from previous step
    du                 The change in displacement to be computed
    contact            The contact class
    markers            The meshtags marking surfaces and domains
    entities           The contact surface entities for integration
    quadrature_degree  The quadrature degree
    const_coeffs       The coefficients for material parameters and h
    consts             The constants in the forms
    """

    num_pairs = len(const_coeffs)
    V = u.function_space
    mesh = V.mesh

    # generate kernels
    with common.Timer("~Contact: Generate Jacobian kernel"):
        kernel_jac = contact.generate_kernel(kt.RayJac)
    with common.Timer("~Contact: Generate residual kernel"):
        kernel_rhs = contact.generate_kernel(kt.Rhs)

    # create vector and matrix
    A = contact.create_matrix(J_custom)
    b = fem.petsc.create_vector(F_custom)

    # Pack gap, normals and test functions on each surface
    gaps = []
    normals = []
    test_fns = []
    with common.Timer("~Contact: Pack gap, normals, testfunction"):
        for i in range(num_pairs):
            gaps.append(contact.pack_gap(i))
            normals.append(contact.pack_ny(i))
            test_fns.append(contact.pack_test_functions(i))

    # Concatenate all coeffs
    ccfs = []
    for i in range(num_pairs):
        ccfs.append(np.hstack([const_coeffs[i], gaps[i], normals[i], test_fns[i]]))

    # retrieve boundary conditions for time step
    tbcs = []
    for k, g in enumerate(bcs[1]):
        tag = bcs[0][k][0]
        sub = bcs[0][k][1]
        if sub == -1:
            fn_space = V
        else:
            fn_space = V.sub(sub)
        bdy_dofs = fem.locate_dofs_topological(fn_space, mesh.topology.dim - 1, markers[1].find(tag))
        tbcs.append(fem.dirichletbc(g, bdy_dofs, fn_space))

    # pack grad u
    grad_u = []
    with common.Timer("~~Contact: Pack u"):
        for i in range(num_pairs):
            grad_u.append(dolfinx_contact.cpp.pack_gradient_quadrature(
                u._cpp_object, quadrature_degree, entities[i]))

    # function for updating coefficients coefficients
    @common.timed("~Contact: Update coefficients")
    def compute_coefficients(x, coeffs):
        size_local = V.dofmap.index_map.size_local
        bs = V.dofmap.index_map_bs
        du.x.array[:size_local * bs] = x.array_r[:size_local * bs]
        du.x.scatter_forward()
        u_candidate = []
        with common.Timer("~~Contact: Pack u contact"):
            for i in range(num_pairs):
                u_candidate.append(contact.pack_u_contact(i, du._cpp_object))
        u_puppet = []
        grad_u_puppet = []
        with common.Timer("~~Contact: Pack u"):
            for i in range(num_pairs):
                u_puppet.append(dolfinx_contact.cpp.pack_coefficient_quadrature(
                    du._cpp_object, quadrature_degree, entities[i]))
                grad_u_puppet.append(dolfinx_contact.cpp.pack_gradient_quadrature(
                    du._cpp_object, quadrature_degree, entities[i]))
        for i in range(num_pairs):
            c_0 = np.hstack([ccfs[i], u_puppet[i], grad_u_puppet[i] + grad_u[i], u_candidate[i]])
            coeffs[i][:, :] = c_0[:, :]

    # function for computing residual
    @common.timed("~Contact: Assemble residual")
    def compute_residual(x, b, coeffs):
        b.zeroEntries()
        b.ghostUpdate(addv=_PETSc.InsertMode.INSERT, mode=_PETSc.ScatterMode.FORWARD)
        with common.Timer("~~Contact: Contact contributions (in assemble vector)"):
            for i in range(num_pairs):
                contact.assemble_vector(b, i, kernel_rhs, coeffs[i], consts)
        with common.Timer("~~Contact: Standard contributions (in assemble vector)"):
            fem.petsc.assemble_vector(b, F_custom)

        # Apply boundary condition
        if len(bcs) > 0:
            fem.petsc.apply_lifting(b, [J_custom], bcs=[tbcs], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=_PETSc.InsertMode.ADD, mode=_PETSc.ScatterMode.REVERSE)
        if len(bcs) > 0:
            fem.petsc.set_bc(b, tbcs, x, -1.0)

    # function for computing jacobian
    @common.timed("~Contact: Assemble matrix")
    def compute_jacobian_matrix(x, A, coeffs):
        A.zeroEntries()
        with common.Timer("~~Contact: Contact contributions (in assemble matrix)"):
            for i in range(num_pairs):
                contact.assemble_matrix(A, [], i, kernel_jac, coeffs[i], consts)
        with common.Timer("~~Contact: Standard contributions (in assemble matrix)"):
            fem.petsc.assemble_matrix(A, J_custom, bcs=tbcs)
        A.assemble()

    # coefficient arrays
    num_coeffs = contact.coefficients_size(False)
    coeffs = np.array([np.zeros((len(entities[i]), num_coeffs)) for i in range(num_pairs)])
    newton_solver = dolfinx_contact.NewtonSolver(mesh.comm, A, b, coeffs)

    # Set matrix-vector computations
    newton_solver.set_residual(compute_residual)
    newton_solver.set_jacobian(compute_jacobian_matrix)
    newton_solver.set_coefficients(compute_coefficients)

    # Set rigid motion nullspace
    null_space = rigid_motions_nullspace_subdomains(V, markers[0], np.unique(markers[0].values))
    newton_solver.A.setNearNullSpace(null_space)

    return newton_solver


def get_problem_parameters(problem_parameters: dict[str, np.float64]):
    """
    Retrieve problem parameters and throw error if parameter missing
    """
    if problem_parameters.get("mu") is None:
        raise RuntimeError("Need to supply lame paramters")
    else:
        mu = problem_parameters.get("mu")

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

    return mu, lmbda, theta, gamma, sigma


def copy_fns(fns: list[Union[fem.Function, fem.Constant]],
             mesh: mesh.Mesh) -> list[Union[fem.Function, fem.Constant]]:
    """
    Create copy of list of finite element functions/constanst
    """
    old_fns = []
    for fn in fns:
        if fn is fem.Function:
            new_fn = fem.Function(fn.function_space)
            new_fn.x.array[:] = fn.x.array[:]
            new_fn.scatter_forward()
        else:
            shape = fn.value.shape
            temp = np.zeros(shape, dtype=_PETSc.ScalarType)
            new_fn = fem.Constant(mesh, temp)
            new_fn.value = fn.value
        old_fns.append(new_fn)
    return old_fns


def update_fns(t: float, fns: list[Union[fem.Function, fem.Constant]],
               old_fns: list[Union[fem.Function, fem.Constant]]) -> None:
    """
    Replace function values of function in fns with
    t* function value of function in old_fns
    """
    for k, fn in enumerate(fns):
        if fn is fem.Function:
            fn.x.array[:] = t * old_fns[k].x.array[:]
            fn.scatter_forward()
        else:
            fn.value = t * old_fns[k].value


def nitsche_unbiased(steps: int, ufl_form: ufl.Form, u: fem.Function,
                     rhs_fns: list[Union[fem.Function, fem.Constant]], markers: list[_cpp.mesh.MeshTags_int32],
                     contact_data: Tuple[AdjacencyList_int32, list[Tuple[int, int]]],
                     bcs: Tuple[npt.NDArray[np.int32], list[Union[fem.Function, fem.Constant]]],
                     problem_parameters: dict[str, np.float64],
                     search_method: dolfinx_contact.cpp.ContactMode,
                     quadrature_degree: int = 5,
                     form_compiler_options: Optional[dict] = None,
                     jit_options: Optional[dict] = None,
                     petsc_options: Optional[dict] = None,
                     newton_options: Optional[dict] = None,
                     outfile: Optional[str] = None,
                     fname: str = "pseudo_time",
                     search_radius: np.float64 = np.float64(-1.)) -> Tuple[fem.Function, list[int],
                                                                           list[int], list[float]]:
    """
    Use custom kernel to compute the contact problem with two elastic bodies coming into contact.

    Parameters
    ==========
    steps:    The number of pseudo time steps
    ufl_form: The variational form without contact contribution
    u:        The function to be solved for. Also serves as initial value.
    rhs_fns:  The functions defining forces/boundary conditions in the variational form
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
    search_method
        Way of detecting contact. Either closest point projection or raytracing
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
    mu, lmbda, theta, gamma, sigma = get_problem_parameters(problem_parameters)

    # Contact data
    contact_pairs = contact_data[1]
    contact_surfaces = contact_data[0]

    # Mesh, function space and FEM functions
    V = u.function_space
    mesh = V.mesh
    v = ufl_form.arguments()[0]  # Test function
    w = ufl.TrialFunction(V)     # Trial function
    du = fem.Function(V)
    du.x.array[:] = u.x.array[:]
    u.x.array[:].fill(0)
    h = ufl.CellDiameter(mesh)
    n = ufl.FacetNormal(mesh)

    # Integration measure and ufl part of linear/bilinear form
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=markers[1])

    # ufl part of contact
    for contact_pair in contact_pairs:
        surface_value = int(contact_surfaces.links(0)[contact_pair[0]])
        ufl_form += - 0.5 * theta * h / gamma * ufl.inner(sigma(u) * n, sigma(v) * n) * \
            ds(surface_value)
    F = ufl.replace(ufl_form, {u: u + du})
    J = ufl.derivative(F, du, w)

    # compiled forms for rhs and tangen system
    F_custom = fem.form(F, form_compiler_options=form_compiler_options, jit_options=jit_options)
    J_custom = fem.form(J, form_compiler_options=form_compiler_options, jit_options=jit_options)

    # store original rhs information and bcs
    old_rhs_fns = copy_fns(rhs_fns, mesh)
    old_bc_fns = copy_fns(bcs[1], mesh)

    # create contact class
    with common.Timer("~Contact: Init"):
        contact = dolfinx_contact.cpp.Contact(markers[1:], contact_surfaces, contact_pairs,
                                              V._cpp_object, quadrature_degree=quadrature_degree,
                                              search_method=search_method)
    contact.set_search_radius(search_radius)

    # pack constants
    consts = np.array([gamma, theta], dtype=np.float64)

    # Pack material parameters mu and lambda on each contact surface
    with common.Timer("~Contact: Interpolate coeffs (mu, lmbda)"):
        V2 = fem.FunctionSpace(mesh, ("DG", 0))
        lmbda2 = fem.Function(V2)
        lmbda2.interpolate(lambda x: np.full((1, x.shape[1]), lmbda))
        mu2 = fem.Function(V2)
        mu2.interpolate(lambda x: np.full((1, x.shape[1]), mu))

    # Retrieve active entities
    entities = []
    with common.Timer("~Contact: Compute active entities"):
        for pair in contact_pairs:
            entities.append(contact.active_entities(pair[0]))

    # Pack material parameters
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

    # Concatenate material parameters, h
    const_coeffs = []
    for i in range(len(contact_pairs)):
        const_coeffs.append(np.hstack([material[i], h_packed[i]]))

    # initialise vtx write
    vtx = io.VTXWriter(mesh.comm, f"{fname}.bp", [u])

    # write initial value
    vtx.write(0)
    timings = []
    newton_its = []
    krylov_its = []
    for tt in range(steps):

        # create distance map
        with common.Timer("~Contact: Distance maps"):
            for i in range(len(contact_pairs)):
                contact.create_distance_map(i)

        # current time
        t = (tt + 1) / steps

        # update forces and bcs
        update_fns(t, rhs_fns, old_rhs_fns)
        update_fns(1. / steps, bcs[1], old_bc_fns)

        # setup newton solver
        newton_solver = setup_newton_solver(F_custom, J_custom, bcs, u, du, contact, markers,
                                            entities, quadrature_degree, const_coeffs, consts)

        # Set Newton solver options
        newton_solver.set_newton_options(newton_options)

        # Set Krylov solver options
        newton_solver.set_krylov_options(petsc_options)
        dofs_global = V.dofmap.index_map_bs * V.dofmap.index_map.size_global
        log.set_log_level(log.LogLevel.OFF)
        # Solve non-linear problem
        timing_str = f"~Contact: {id(dofs_global)} Solve Nitsche"
        with common.Timer(timing_str):
            n, converged = newton_solver.solve(du)
        if outfile is not None:
            viewer = _PETSc.Viewer().createASCII(outfile, "a")
            newton_solver.krylov_solver.view(viewer)

        # collect solver stats
        timings.append(common.timing(timing_str)[1])
        newton_its.append(n)
        krylov_its.append(newton_solver.krylov_iterations)

        if not converged:
            print("Newton solver did not converge")

        # update u and mesh
        du.x.scatter_forward()
        u.x.array[:] += du.x.array[:]
        contact.update_submesh_geometry(u._cpp_object)

        # reset du
        du.x.array[:].fill(0)

        # write solution
        vtx.write(t)

    contact.update_submesh_geometry(u._cpp_object)
    vtx.close()
    return u, newton_its, krylov_its, timings
