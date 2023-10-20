from typing import Optional, Tuple
from dolfinx import common, cpp, fem
from dolfinx import mesh as _mesh
from dolfinx.fem.petsc import create_vector
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx.cpp.graph import AdjacencyList_int32
from petsc4py import PETSc as _PETSc

import dolfinx_contact
import dolfinx_contact.cpp

from .nitsche_unbiased import setup_newton_solver, get_problem_parameters
kt = dolfinx_contact.cpp.Kernel
from dolfinx_contact.helpers import rigid_motions_nullspace_subdomains


def setup_snes_solver(F_custom: fem.forms.Form, J_custom: fem.forms.Form,
                        bcs: list[fem.DirichletBC],
                        u: fem.Function, du: fem.Function,
                        contact: dolfinx_contact.cpp.Contact, markers: list[_mesh.MeshTags],
                        entities: list[npt.NDArray[np.int32]], quadrature_degree: int,
                        const_coeffs: list[npt.NDArray[np.float64]], consts: npt.NDArray[np.float64],
                        search_method: list[dolfinx_contact.cpp.ContactMode],
                        petsc_options, snes_options,
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
            # contact.update_distance_map(i, gaps[i], normals[i])
            test_fns.append(contact.pack_test_functions(i))

    # Concatenate all coeffs
    ccfs = []
    for i in range(num_pairs):
        ccfs.append(np.hstack([const_coeffs[i], gaps[i], normals[i], test_fns[i]]))

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
    def compute_coefficients(x):
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
        coeffs = []
        for i in range(num_pairs):
            coeffs.append(np.hstack([ccfs[i], u_puppet[i], grad_u_puppet[i]
                            + grad_u[i], u_candidate[i], normals_old[i], u_old_opp[i]]))
        return coeffs
            

    # function for computing residual
    @common.timed("~Contact: Assemble residual")
    def compute_residual(snes, x, b):
        coeffs = compute_coefficients(x)
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
            fem.petsc.apply_lifting(b, [J_custom], bcs=[bcs], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=_PETSc.InsertMode.ADD, mode=_PETSc.ScatterMode.REVERSE)
        if len(bcs) > 0:
            fem.petsc.set_bc(b, bcs, x, -1.0)

    # function for computing jacobian
    @common.timed("~Contact: Assemble matrix")
    def compute_jacobian_matrix(snes, x, A, P):
        coeffs = compute_coefficients(x)
        A.zeroEntries()
        with common.Timer("~~Contact: Contact contributions (in assemble matrix)"):
            for i in range(num_pairs):
                contact.assemble_matrix(A, i, kernel_jac, coeffs[i], consts)
                contact.assemble_matrix(A, i, kernel_friction_jac, coeffs[i], consts)
        with common.Timer("~~Contact: Standard contributions (in assemble matrix)"):
            fem.petsc.assemble_matrix(A, J_custom, bcs=bcs)
        A.assemble()

    # Create semismooth Newton solver (SNES)
    snes = _PETSc.SNES().create()
    # Set SNES options
    opts = _PETSc.Options()
    snes.setOptionsPrefix(f"snes_solve_{id(snes)}")
    option_prefix = snes.getOptionsPrefix()
    opts.prefixPush(option_prefix)
    for k, v in snes_options.items():
        opts[k] = v
    opts.prefixPop()
    snes.setFromOptions()

    # Set solve functions and variable bounds
    snes.setFunction(compute_residual, b)
    snes.setJacobian(compute_jacobian_matrix, A)
    null_space = rigid_motions_nullspace_subdomains(V, markers[0], np.unique(
        markers[0].values), num_domains=len(np.unique(markers[0].values)))
    A.setNearNullSpace(null_space)

    # Set ksp options
    ksp = snes.ksp
    ksp.setOptionsPrefix(f"snes_ksp_{id(ksp)}")
    opts = _PETSc.Options()
    option_prefix = ksp.getOptionsPrefix()
    opts.prefixPush(option_prefix)
    for k, v in petsc_options.items():
        opts[k] = v
    opts.prefixPop()
    ksp.setFromOptions()

    return snes


class ContactProblem:
    __slots__ = ["F", "J", "bcs", "u", "du", "contact", "markers", "entities",
                 "q_deg", "coeffs", "consts", "search_method", "newton_options",
                 "petsc_options", "coulomb", "normals"]

    def __init__(self, F: ufl.Form, J: ufl.Form,
                 bcs: list[fem.DirichletBC],
                 u: fem.Function, du: fem.Function, contact: dolfinx_contact.cpp.Contact,
                 markers: list[_mesh.MeshTags], entities: list[npt.NDArray[np.int32]],
                 quadrature_degree: int, const_coeffs: list[npt.NDArray[np.float64]],
                 consts: npt.NDArray[np.float64], search_method: list[dolfinx_contact.cpp.ContactMode],
                 petsc_options: Optional[dict] = None, newton_options: Optional[dict] = None, coulomb: bool = False,
                 normals: Optional[list[npt.NDArray[np.float64]]] = None):

        self.F = F
        self.J = J
        self.bcs = bcs
        self.u = u
        self.du = du
        self.contact = contact
        self.markers = markers
        self.entities = entities
        self.q_deg = quadrature_degree
        self.coeffs = const_coeffs
        self.consts = consts
        self.search_method = search_method
        self.newton_options = newton_options
        self.petsc_options = petsc_options
        self.coulomb = coulomb
        self.normals = normals

    def solve(self):
        newton_solver = setup_newton_solver(self.F, self.J, self.bcs, self.u, self.du, self.contact, self.markers,
                                            self.entities, self.q_deg, self.coeffs, self.consts,
                                            self.search_method, self.coulomb, self.normals)
        # Set Newton solver options
        newton_solver.set_newton_options(self.newton_options)
        print(self.newton_options)

        # Set Krylov solver options
        newton_solver.set_krylov_options(self.petsc_options)
        n, converged = newton_solver.solve(self.du, write_solution=True, offset_fun=self.u)
        if not converged:
            print("Newton solver did not converge")
        return n
    
        # newton_solver = setup_snes_solver(self.F, self.J, self.bcs, self.u, self.du, self.contact, self.markers,
        #                                     self.entities, self.q_deg, self.coeffs, self.consts,
        #                                      self.search_method, self.petsc_options, self.newton_options, 
        #                                      self.coulomb, self.normals)
        # newton_solver.solve(None, self.du.vector)
        # if (newton_solver.getConvergedReason() <= 1) or (newton_solver.getConvergedReason() >= 4):
        #     print(f"Snes solver did not converge. Converged Reason {newton_solver.getConvergedReason()}")
        
        # return newton_solver.getIterationNumber()
    # def generate_integration_kernels(self):
    #         # generate kernels
    #     with common.Timer("~Contact: Generate Jacobian kernel"):
    #         self.kernel_jac = self.generate_kernel(kt.Jac)
    #         if self.coulomb:
    #             self.kernel_friction_jac = self.generate_kernel(kt.CoulombJac)
    #         else:
    #             self.kernel_friction_jac = self.generate_kernel(kt.TrescaJac)
    #     with common.Timer("~Contact: Generate residual kernel"):
    #         self.kernel_rhs = self.generate_kernel(kt.Rhs)
    #         if self.coulomb:
    #             self.kernel_friction_rhs = self.generate_kernel(kt.CoulombRhs)
    #         else:
    #             self.kernel_friction_rhs = self.generate_kernel(kt.TrescaRhs)
                
    # def compute_coefficients(self, x, coeffs):
    #     num_pairs = len(self.const_coeffs)
    #     V = self.u.function_space
    
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


def create_contact_solver(ufl_form: ufl.Form, u: fem.Function,
                          mu: fem.Function, lmbda: fem.Function,
                          markers: list[_mesh.MeshTags],
                          contact_data: Tuple[AdjacencyList_int32, list[Tuple[int, int]]],
                          bcs: list[fem.DirichletBC],
                          problem_parameters: dict[str, np.float64],
                          search_method: list[dolfinx_contact.cpp.ContactMode],
                          quadrature_degree: int = 5,
                          form_compiler_options: Optional[dict] = None,
                          jit_options: Optional[dict] = None,
                          petsc_options: Optional[dict] = None,
                          newton_options: Optional[dict] = None,
                          search_radius: np.float64 = np.float64(-1.),
                          coulomb: bool = False, dt=1.0) -> ContactProblem:
    """
    Use custom kernel to compute the contact problem with two elastic bodies coming into contact.

    Parameters
    ==========
    ufl_form: The variational form without contact contribution
    u:        The function to be solved for. Also serves as initial value.
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

    """
    form_compiler_options = {} if form_compiler_options is None else form_compiler_options
    jit_options = {} if jit_options is None else jit_options
    petsc_options = {} if petsc_options is None else petsc_options
    newton_options = {} if newton_options is None else newton_options
    theta, gamma, s = get_problem_parameters(problem_parameters)
    # sigma = sigma_func(mu, lmbda)

    # Contact data
    contact_pairs = contact_data[1]
    contact_surfaces = contact_data[0]

    # Mesh, function space and FEM functions
    V = u.function_space
    mesh = V.mesh
    V2 = fem.FunctionSpace(mesh, ("DG", 0))
    # v = ufl_form.arguments()[0]  # Test function
    w = ufl.TrialFunction(V)     # Trial function
    du = fem.Function(V)
    du.x.array[:] = u.x.array[:]
    u.x.array[:].fill(0)
    h = fem.Function(V2)
    tdim = mesh.topology.dim
    ncells = mesh.topology.index_map(tdim).size_local
    h_vals = cpp.mesh.h(mesh._cpp_object, mesh.topology.dim, np.arange(0, ncells, dtype=np.int32))
    h.x.array[:ncells] = h_vals[:]
    # n = ufl.FacetNormal(mesh)

    # Integration measure and ufl part of linear/bilinear form
    # ds = ufl.Measure("ds", domain=mesh, subdomain_data=markers[1])

    # ufl part of contact
    # for contact_pair in contact_pairs:
    #     surface_value = int(contact_surfaces.links(0)[contact_pair[0]])
    #     ufl_form += - 0.5 * theta * h / gamma * ufl.inner(sigma(u) * n, sigma(v) * n) * \
    #         ds(surface_value)
    F = ufl.replace(ufl_form, {u: u + du})
    J = ufl.derivative(F, du, w)

    # compiled forms for rhs and tangen system
    F_custom = fem.form(F, form_compiler_options=form_compiler_options, jit_options=jit_options)
    J_custom = fem.form(J, form_compiler_options=form_compiler_options, jit_options=jit_options)

    # create contact class
    markers_cpp = [marker._cpp_object for marker in markers[1:]]
    with common.Timer("~Contact: Init"):
        contact = dolfinx_contact.cpp.Contact(markers_cpp, contact_surfaces, contact_pairs,
                                              V._cpp_object, quadrature_degree=quadrature_degree,
                                              search_method=search_method)

    contact.set_search_radius(search_radius)

    # pack constants
    consts = np.array([gamma, theta, dt], dtype=np.float64)

    # Retrieve active entities
    entities = []
    with common.Timer("~Contact: Compute active entities"):
        for pair in contact_pairs:
            entities.append(contact.active_entities(pair[0]))

    # interpolate friction coefficient
    fric_coeff = fem.Function(V2)
    fric_coeff.interpolate(lambda x: np.full((1, x.shape[1]), s))

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
                h_int._cpp_object, 0, entities[i]))

    for j in range(len(contact_pairs)):
        contact.create_distance_map(j)
    normals = []
    for i in range(len(contact_pairs)):
        if search_method[i] == dolfinx_contact.cpp.ContactMode.Raytracing:
            normals.append(-contact.pack_nx(i))
        else:
            normals.append(contact.pack_ny(i))

    # Concatenate material parameters, he4
    const_coeffs = []
    for i in range(len(contact_pairs)):
        const_coeffs.append(np.hstack([material[i], h_packed[i]]))

    problem = ContactProblem(F_custom, J_custom, bcs, u, du, contact, markers, entities,
                             quadrature_degree, const_coeffs, consts, search_method,
                             petsc_options, newton_options, coulomb, normals)

    return problem
