# Copyright (C) 2022 Sarah Roggendorf
#
# SPDX-License-Identifier:    MIT

from typing import Optional, Tuple, Union, Any
from dolfinx import common, fem, io, log, cpp
from dolfinx import mesh as _mesh
from dolfinx.fem.petsc import create_vector
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx.cpp.graph import AdjacencyList_int32
from petsc4py import PETSc as _PETSc
from mpi4py import MPI

import dolfinx_contact
import dolfinx_contact.cpp
from dolfinx_contact.helpers import (rigid_motions_nullspace_subdomains, sigma_func)
from dolfinx_contact.output import write_pressure_xdmf

kt = dolfinx_contact.cpp.Kernel

__all__ = ["setup_newton_solver", "nitsche_unbiased"]


def setup_newton_solver(F_custom: fem.forms.Form, J_custom: fem.forms.Form,
                        bcs: Tuple[npt.NDArray[np.int32], list[Union[fem.Function, fem.function.Constant]]],
                        u: fem.Function, du: fem.Function,
                        contact: dolfinx_contact.cpp.Contact, markers: list[_mesh.MeshTags],
                        entities: list[npt.NDArray[np.int32]], quadrature_degree: int,
                        const_coeffs: list[npt.NDArray[np.float64]], consts: npt.NDArray[np.float64],
                        search_method: list[dolfinx_contact.cpp.ContactMode],
                        coulomb: bool, normals_old=None):
    """
    Set up newton solver for contact problem.
    Generate kernels and define functions for updating coefficients, stiffness matrix and residual vector.

    Parameters
    ==========
    F_custom           The compiled form for the residual vector
    J_custom           The compiled form for the jacobian
    bcs                The boundary conditions
    u                  The displacement from previous step
    du                 The change in displacement to be computed
    contact            The contact class
    markers            The meshtags marking surfaces and domains
    entities           The contact surface entities for integration
    quadrature_degree  The quadrature degree
    const_coeffs       The coefficients for material parameters and h
    consts             The constants in the forms
    search_method      The contact detection algorithms used for each pair
    """

    num_pairs = len(const_coeffs)
    V = u.function_space
    mesh = V.mesh

    # generate kernels
    with common.Timer("~Contact: Generate Jacobian kernel"):
        kernel_jac = contact.generate_kernel(kt.Jac)
        if coulomb:
            kernel_friction_jac = contact.generate_kernel(kt.CoulombJac)
        else:
            kernel_friction_jac = contact.generate_kernel(kt.TrescaJac)
    with common.Timer("~Contact: Generate residual kernel"):
        kernel_rhs = contact.generate_kernel(kt.Rhs)
        if coulomb:
            kernel_friction_rhs = contact.generate_kernel(kt.CoulombRhs)
        else:
            kernel_friction_rhs = contact.generate_kernel(kt.TrescaRhs)

    # create vector and matrix
    A = contact.create_matrix(J_custom._cpp_object)
    b = create_vector(F_custom)

    # Pack gap, normals and test functions on each surface
    gaps = []
    normals = []
    test_fns = []
    with common.Timer("~Contact: Pack gap, normals, testfunction"):
        for i in range(num_pairs):
            gaps.append(contact.pack_gap(i))
            if search_method[i] == dolfinx_contact.cpp.ContactMode.Raytracing:
                normals.append(-contact.pack_nx(i))
            else:
                normals.append(contact.pack_ny(i))
            contact.update_distance_map(i, gaps[i], normals[i])
            test_fns.append(contact.pack_test_functions(i))

    # Concatenate all coeffs
    ccfs = []
    for i in range(num_pairs):
        ccfs.append(np.hstack([const_coeffs[i], gaps[i], normals[i], test_fns[i]]))

    # retrieve boundary conditions for time step
    tbcs = []
    for k, g in enumerate(bcs[1]):
        bdy_dofs = bcs[0][k][0]
        sub = bcs[0][k][1]
        if sub == -1:
            fn_space = V
        else:
            fn_space = V.sub(sub)
        # bdy_dofs = fem.locate_dofs_topological(fn_space, mesh.topology.dim - 1, markers[1].find(tag))
        tbcs.append(fem.dirichletbc(g, bdy_dofs, fn_space))

    # pack grad u
    grad_u = []
    u_old = []
    u_old_opp = []
    with common.Timer("~~Contact: Pack grad(u)"):
        for i in range(num_pairs):
            grad_u.append(dolfinx_contact.cpp.pack_gradient_quadrature(
                u._cpp_object, quadrature_degree, entities[i]))
            u_old.append(dolfinx_contact.cpp.pack_coefficient_quadrature(
                u._cpp_object, quadrature_degree, entities[i]))
            u_old_opp.append(contact.pack_u_contact(i, u._cpp_object))

    # FIXME: temporary work around
    if normals_old is None:
        normals_old = u_old

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
            c_0 = np.hstack([ccfs[i], u_puppet[i], grad_u_puppet[i]
                            + grad_u[i], u_candidate[i], normals_old[i], u_old_opp[i]])
            coeffs[i][:, :] = c_0[:, :]

    # function for computing residual
    @common.timed("~Contact: Assemble residual")
    def compute_residual(x, b, coeffs):
        b.zeroEntries()
        b.ghostUpdate(addv=_PETSc.InsertMode.INSERT, mode=_PETSc.ScatterMode.FORWARD)
        with common.Timer("~~Contact: Contact contributions (in assemble vector)"):
            for i in range(num_pairs):
                contact.assemble_vector(b, i, kernel_rhs, coeffs[i], consts)
                contact.assemble_vector(b, i, kernel_friction_rhs, coeffs[i], consts)
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
                contact.assemble_matrix(A, i, kernel_jac, coeffs[i], consts)
                contact.assemble_matrix(A, i, kernel_friction_jac, coeffs[i], consts)
        with common.Timer("~~Contact: Standard contributions (in assemble matrix)"):
            fem.petsc.assemble_matrix(A, J_custom, bcs=tbcs)
        A.assemble()

    # coefficient arrays
    num_coeffs = contact.coefficients_size(False)

    coeffs = [np.zeros((len(entities[i]), num_coeffs)) for i in range(num_pairs)]
    newton_solver = dolfinx_contact.NewtonSolver(mesh.comm, A, b, coeffs)

    # Set matrix-vector computations
    newton_solver.set_residual(compute_residual)
    newton_solver.set_jacobian(compute_jacobian_matrix)
    newton_solver.set_coefficients(compute_coefficients)

    # Set rigid motion nullspace
    null_space = rigid_motions_nullspace_subdomains(V, markers[0], np.unique(
        markers[0].values), num_domains=len(np.unique(markers[0].values)))
    newton_solver.A.setNearNullSpace(null_space)

    return newton_solver


def get_problem_parameters(problem_parameters: dict[str, np.float64]):
    """
    Retrieve problem parameters and throw error if parameter missing
    """
    if problem_parameters.get("theta") is None:
        raise RuntimeError("Need to supply theta for Nitsche's method")
    else:
        theta = problem_parameters["theta"]
    if problem_parameters.get("gamma") is None:
        raise RuntimeError("Need to supply gamma for Nitsche's method")
    else:
        gamma = problem_parameters.get("gamma")
    s = problem_parameters.get("friction", np.float64(0.0))

    return theta, gamma, s


def copy_fns(fns: list[Union[fem.Function, fem.Constant]],
             mesh: _mesh.Mesh) -> list[Union[fem.Function, fem.Constant]]:
    """
    Create copy of list of finite element functions/constanst
    """
    old_fns = []
    for fn in fns:
        if type(fn) is fem.Function:
            new_fn = fem.Function(fn.function_space)
            new_fn.x.array[:] = fn.x.array[:]
            new_fn.x.scatter_forward()
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
        if type(fn) is fem.Function:
            fn.x.array[:] = t * old_fns[k].x.array[:]
            fn.x.scatter_forward()
        else:
            fn.value = t * old_fns[k].value


def nitsche_unbiased(steps: int, ufl_form: ufl.Form, u: fem.Function, mu: fem.Function, lmbda: fem.Function,
                     rhs_fns: list[Any], markers: list[_mesh.MeshTags],
                     contact_data: Tuple[AdjacencyList_int32, list[Tuple[int, int]]],
                     bcs: Tuple[list[Tuple[npt.NDArray[np.int32], int]], list[Union[fem.Function, fem.Constant]]],
                     problem_parameters: dict[str, np.float64],
                     search_method: list[dolfinx_contact.cpp.ContactMode],
                     quadrature_degree: int = 5,
                     form_compiler_options: Optional[dict] = None,
                     jit_options: Optional[dict] = None,
                     petsc_options: Optional[dict] = None,
                     newton_options: Optional[dict] = None,
                     outfile: Optional[str] = None,
                     fname: str = "pseudo_time",
                     search_radius: np.float64 = np.float64(-1.),
                     order=1, simplex=True, pressure_function=None,
                     projection_coordinates=[(0, 0), (0, 0)],
                     coulomb: bool = False) -> Tuple[fem.Function, list[int],
                                                     list[int], list[float]]:
    """
    Use custom kernel to compute the contact problem with two elastic bodies coming into contact.

    Parameters
    ==========
    steps:    The number of pseudo time steps
    ufl_form: The variational form without contact contribution
    u:        The function to be solved for. Also serves as initial value.
    mu:       The first Lame parameter as a DG0 function
    lmbda:    The second Lame parameter as a DG0 function
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
        Dictionary with Nitsche parameters.
        Valid (key, value) tuples are: ('gamma': float), ('theta', float)
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
    fname
        filename for vtx output
    search_radius
        search radius for raytracing
    order
        order of function space and geometry
    simplex
        bool indicating whether it is a simplicial mesh
    pressure_function
        function for analytical surface pressure
    projection_coordinates
        work around for pressure visualisation on curved meshes

    """
    form_compiler_options = {} if form_compiler_options is None else form_compiler_options
    jit_options = {} if jit_options is None else jit_options
    petsc_options = {} if petsc_options is None else petsc_options
    newton_options = {} if newton_options is None else newton_options
    theta, gamma, s = get_problem_parameters(problem_parameters)
    sigma = sigma_func(mu, lmbda)

    # Contact data
    contact_pairs = contact_data[1]
    contact_surfaces = contact_data[0]

    # Mesh, function space and FEM functions
    V = u.function_space
    mesh = V.mesh
    V2 = fem.FunctionSpace(mesh, ("DG", 0))
    v = ufl_form.arguments()[0]  # Test function
    w = ufl.TrialFunction(V)     # Trial function
    du = fem.Function(V)
    du.x.array[:] = u.x.array[:]
    u.x.array[:].fill(0)
    n = ufl.FacetNormal(mesh)
    tdim = mesh.topology.dim
    ncells = mesh.topology.index_map(tdim).size_local
    h = fem.Function(V2)
    h_vals = cpp.mesh.h(mesh._cpp_object, mesh.topology.dim, np.arange(0, ncells, dtype=np.int32))
    h.x.array[:ncells] = h_vals[:]

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
    markers_cpp = [marker._cpp_object for marker in markers[1:]]
    with common.Timer("~Contact: Init"):
        contact = dolfinx_contact.cpp.Contact(markers_cpp, contact_surfaces, contact_pairs,
                                              V._cpp_object, quadrature_degree=quadrature_degree,
                                              search_method=search_method)

    xdmf = cpp.io.XDMFFile(mesh.comm, "debug.xdmf", "w")
    xdmf.write_mesh(contact.submesh(), xpath="/Xdmf/Domain")
    del (xdmf)

    contact.set_search_radius(search_radius)

    # pack constants
    consts = np.array([gamma, theta, 1.0], dtype=np.float64)

    # interpolate friction coefficient
    fric_coeff = fem.Function(V2)
    fric_coeff.interpolate(lambda x: np.full((1, x.shape[1]), s))

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
                mu._cpp_object, 0, entities[i]),
                dolfinx_contact.cpp.pack_coefficient_quadrature(
                lmbda._cpp_object, 0, entities[i]),
                dolfinx_contact.cpp.pack_coefficient_quadrature(
                fric_coeff._cpp_object, 0, entities[i])]))

    # Pack celldiameter on each surface
    h_packed = []
    with common.Timer("~Contact: Compute and pack celldiameter"):
        surface_cells = np.unique(np.hstack([entities[i][:, 0] for i in range(len(contact_pairs))]))
        h_int = fem.Function(V2)
        expr = fem.Expression(h, V2.element.interpolation_points())
        h_int.interpolate(expr, surface_cells)
        for i in range(len(contact_pairs)):
            h_packed.append(dolfinx_contact.cpp.pack_coefficient_quadrature(
                h._cpp_object, 0, entities[i]))

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
        log.log(log.LogLevel.WARNING, "Time step " + str(tt + 1) + " of " + str(steps)
                + " +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
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
        bcs_n = (np.array([[bc[0], bc[1]] for bc in bcs[0]], dtype=np.int32), bcs[1])
        newton_solver = setup_newton_solver(F_custom, J_custom, bcs_n, u, du, contact, markers,
                                            entities, quadrature_degree, const_coeffs, consts,
                                            search_method, coulomb)

        # Set Newton solver options
        newton_solver.set_newton_options(newton_options)

        # Set Krylov solver options
        newton_solver.set_krylov_options(petsc_options)
        log.set_log_level(log.LogLevel.WARNING)
        # Solve non-linear problem
        timing_str = f"~Contact: {tt+1} Solve Nitsche"
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

        # take a fraction of du as initial guess
        # this is to ensure non-singular matrices in the case of no Dirichlet boundary
        du.x.array[:] = (1. / steps) * du.x.array[:]

        # write solution
        vtx.write(t)

    vtx.close()
    if pressure_function is not None:
        sig_n = write_pressure_xdmf(mesh, contact, u, du, contact_pairs, quadrature_degree,
                                    search_method, entities, material, order, simplex, pressure_function,
                                    projection_coordinates, fname)
    else:
        sig_n = None
    gdim = mesh.geometry.dim

    c_to_f = mesh.topology.connectivity(tdim, tdim - 1)
    facet_list = []
    for j in range(len(contact_pairs)):
        facet_list.append(np.zeros(len(entities[j]), dtype=np.int32))
        for i, e in enumerate(entities[j]):
            facet = c_to_f.links(e[0])[e[1]]
            facet_list[j][i] = facet

    facets = np.unique(np.sort(np.hstack([facet_list[j] for j in range(len(contact_pairs))])))
    facet_mesh, fm_to_msh = _mesh.create_submesh(mesh, tdim - 1, facets)[:2]

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

    Q = fem.FunctionSpace(facet_mesh, Q_element)
    sig_n = []
    for i in range(len(contact_pairs)):
        n_x = contact.pack_nx(i)
        grad_u = dolfinx_contact.cpp.pack_gradient_quadrature(
            u._cpp_object, quadrature_degree, entities[i])
        num_facets = entities[i].shape[0]
        num_q_points = n_x.shape[1] // gdim
        # this assumes mu, lmbda are constant for each body
        sign = np.array(dolfinx_contact.cpp.compute_contact_forces(
            grad_u, n_x, num_q_points, num_facets, gdim, material[i][0, 0], material[i][0, 1]))
        sign = sign.reshape(num_facets * num_q_points, gdim)
        sig_n.append(sign)

    sig_x = fem.Function(Q)
    sig_y = fem.Function(Q)
    for j in range(len(contact_pairs)):
        dofs = np.array(np.hstack([range(msh_to_fm[facet_list[j]][i] * num_q_points,
                        num_q_points * (msh_to_fm[facet_list[j]][i] + 1)) for i in range(len(entities[j]))]))
        if j == 0:
            sig_x.x.array[dofs] = sig_n[j][:, 0]
            sig_y.x.array[dofs] = sig_n[j][:, 1]

    # Define forms for the projection
    dx_f = ufl.Measure("dx", domain=facet_mesh)
    force_x = fem.form(sig_x * dx_f)
    force_y = fem.form(sig_y * dx_f)
    R_x = facet_mesh.comm.allreduce(fem.assemble_scalar(force_x), op=MPI.SUM)
    R_y = facet_mesh.comm.allreduce(fem.assemble_scalar(force_y), op=MPI.SUM)
    print("Rx/Ry", R_x, R_y, s, R_x / R_y, (s + np.tan(np.pi / 6)) / (1 + s * np.tan(np.pi / 6)))

    contact.update_submesh_geometry(u._cpp_object)
    return u, newton_its, krylov_its, timings
